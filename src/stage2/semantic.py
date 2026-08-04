"""OpenEarthMap semantic-model backend used by the unified Stage 2 runner."""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .classes import CLASS_NAMES


class SemanticBackend(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def health_check(self) -> dict[str, Any]: ...

    @abstractmethod
    def infer(self, image: np.ndarray) -> dict[str, np.ndarray]: ...


class OpenEarthMapBackend(SemanticBackend):
    """Official OpenEarthMap FasterSeg eight-class semantic model."""

    def __init__(
        self,
        repository: str | Path,
        architecture: str | Path,
        checkpoint: str | Path,
        device: str = "auto",
        input_size: int = 1024,
        use_tta: bool = True,
    ) -> None:
        self.repository = Path(repository).expanduser().resolve()
        self.architecture = Path(architecture).expanduser().resolve()
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.requested_device = device
        self.device = device
        self.input_size = int(input_size)
        self.use_tta = bool(use_tta)
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        if not self.repository.is_dir():
            raise FileNotFoundError(f"OpenEarthMap repository not found: {self.repository}")
        if not self.architecture.is_file():
            raise FileNotFoundError(f"FasterSeg architecture file not found: {self.architecture}")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"FasterSeg checkpoint not found: {self.checkpoint}")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("OpenEarthMap FasterSeg requires PyTorch; install requirements.txt") from exc

        api_path = str(self.repository / "fasterseg_api")
        if api_path not in sys.path:
            sys.path.insert(0, api_path)
        try:
            from model_seg import Network_Multi_Path_Infer as Network
        except ImportError as exc:
            raise RuntimeError(f"Could not import FasterSeg from {api_path}: {exc}") from exc

        # The official 2022 architecture contains NumPy scalar metadata. Allow
        # only those legacy scalar types while retaining weights_only loading.
        with torch.serialization.safe_globals(
            [
                (np.core.multiarray.scalar, "numpy.core.multiarray.scalar"),
                np.dtype,
                type(np.dtype(np.float64)),
            ]
        ):
            state = torch.load(self.architecture, map_location="cpu", weights_only=True)
        model = Network(
            [state["alpha_1_0"].detach(), state["alpha_1_1"].detach(), state["alpha_1_2"].detach()],
            [None, state["beta_1_1"].detach(), state["beta_1_2"].detach()],
            [state["ratio_1_0"].detach(), state["ratio_1_1"].detach(), state["ratio_1_2"].detach()],
            num_classes=len(CLASS_NAMES),
            layers=16,
            Fch=12,
            width_mult_list=[4.0 / 12, 6.0 / 12, 8.0 / 12, 10.0 / 12, 1.0],
            stem_head_width=(8.0 / 12, 8.0 / 12),
            ignore_skip=False,
        )
        model.build_structure([2, 1])
        weights = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        current = model.state_dict()
        compatible = {key: value for key, value in weights.items() if key in current}
        if not compatible:
            raise ValueError(f"No compatible FasterSeg tensors found in {self.checkpoint}")
        current.update(compatible)
        model.load_state_dict(current)

        self.device = "cuda" if self.requested_device == "auto" and torch.cuda.is_available() else (
            "cpu" if self.requested_device == "auto" else self.requested_device
        )
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected for OpenEarthMap but torch.cuda.is_available() is false")
        self.model = model.to(self.device).eval()

    def health_check(self) -> dict[str, Any]:
        try:
            import torch

            version = torch.__version__
            cuda = torch.cuda.is_available()
            gpu = torch.cuda.get_device_name(0) if cuda else None
        except ImportError:
            version, cuda, gpu = None, False, None
        parameters = sum(parameter.numel() for parameter in self.model.parameters()) if self.model is not None else None
        return {
            "backend_name": "openearthmap-fasterseg",
            "class_names": list(CLASS_NAMES),
            "repository": str(self.repository),
            "architecture_path": str(self.architecture),
            "architecture_available": self.architecture.is_file(),
            "checkpoint_path": str(self.checkpoint),
            "checkpoint_available": self.checkpoint.is_file(),
            "selected_device": self.device,
            "pytorch_version": version,
            "cuda_available": cuda,
            "gpu_name": gpu,
            "model_loaded_successfully": self.model is not None,
            "parameter_count": parameters,
            "use_tta": self.use_tta,
        }

    def infer(self, image: np.ndarray) -> dict[str, np.ndarray]:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("OpenEarthMap input must be HWC uint8 RGB")
        self.load()
        import torch

        original_shape = image.shape[:2]
        resized = cv2.resize(image, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized.copy()).permute(2, 0, 1).float().div_(255.0)
        mean = torch.tensor([0.4325, 0.4483, 0.3879])[:, None, None]
        std = torch.tensor([0.0195, 0.0169, 0.0179])[:, None, None]
        tensor = ((tensor - mean) / std).unsqueeze(0).to(self.device)
        try:
            with torch.inference_mode():
                logits = self.model(tensor)
                if self.use_tta:
                    logits = logits + self.model(tensor.flip(-1)).flip(-1)
                probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError("CUDA out of memory during OpenEarthMap inference; disable TTA or use CPU") from exc
        finally:
            del tensor
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

        if probabilities.shape[1:] != original_shape:
            probabilities = np.stack(
                [
                    cv2.resize(layer, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
                    for layer in probabilities
                ]
            )
            probabilities /= np.maximum(probabilities.sum(axis=0, keepdims=True), 1e-8)
        return {
            "label_map": probabilities.argmax(axis=0).astype(np.uint8),
            "confidence": probabilities.max(axis=0).astype(np.float32),
            "probabilities": probabilities.astype(np.float32),
        }


class FakeSemanticBackend(SemanticBackend):
    """Deterministic injected backend for tests; it is never selected implicitly."""

    def __init__(self) -> None:
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def health_check(self) -> dict[str, Any]:
        return {
            "backend_name": "fake-openearthmap",
            "class_names": list(CLASS_NAMES),
            "selected_device": "cpu",
            "model_loaded_successfully": self.loaded,
            "parameter_count": 0,
        }

    def infer(self, image: np.ndarray) -> dict[str, np.ndarray]:
        self.load()
        labels = (image[..., 0].astype(np.uint16) % len(CLASS_NAMES)).astype(np.uint8)
        probabilities = np.full(
            (len(CLASS_NAMES), *labels.shape),
            0.01 / (len(CLASS_NAMES) - 1),
            np.float32,
        )
        for class_id in range(len(CLASS_NAMES)):
            probabilities[class_id][labels == class_id] = 0.99
        return {
            "label_map": labels,
            "confidence": probabilities.max(axis=0),
            "probabilities": probabilities,
        }


def create_semantic_backend(cfg: dict[str, Any], project_root: Path) -> SemanticBackend:
    if cfg["backend"] == "fake":
        return FakeSemanticBackend()
    resolve = lambda value: Path(value) if Path(value).is_absolute() else project_root / value
    inference = cfg["inference"]
    return OpenEarthMapBackend(
        resolve(cfg["repository"]),
        resolve(cfg["architecture"]),
        resolve(cfg["checkpoint"]),
        cfg.get("device", "auto"),
        inference.get("input_size", 1024),
        inference.get("use_tta", True),
    )


def timed_semantic_inference(
    backend: SemanticBackend,
    image: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    started = time.perf_counter()
    result = backend.infer(image)
    return result, {
        "inference_seconds": time.perf_counter() - started,
        "health": backend.health_check(),
    }
