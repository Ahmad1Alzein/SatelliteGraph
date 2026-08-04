from __future__ import annotations

import time
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class SegmentationBackend(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def health_check(self) -> dict[str, Any]: ...

    @abstractmethod
    def infer_raw(self, image: np.ndarray, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]: ...


class Sam2Backend(SegmentationBackend):
    """Official Meta SAM 2/2.1 automatic-mask backend."""

    def __init__(
        self,
        checkpoint: str | None,
        model_type: str | None,
        model_config: str | None,
        device: str,
        mixed_precision: bool,
        generator_config: dict[str, Any],
    ):
        self.checkpoint = Path(checkpoint).expanduser().resolve() if checkpoint else None
        self.model_type = model_type
        self.model_config = model_config
        self.requested_device = device
        self.mixed_precision = mixed_precision
        self.generator_config = dict(generator_config)
        self.effective_generator_config = dict(generator_config)
        self.model = self.generator = None
        self.device = device
        self.compiled_postprocessing_available: bool | None = None
        self.postprocessing_backend = "not_initialized"
        self.cpu_postprocess_region_area = 0
        self._remove_small_regions = None

    def load(self) -> None:
        if self.model is not None:
            return
        if self.checkpoint is None:
            raise FileNotFoundError("SAM 2 checkpoint is not configured; set segmentation.checkpoint")
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"SAM 2 checkpoint not found: {self.checkpoint}")
        if not self.model_config:
            raise ValueError("SAM 2 requires segmentation.model_config, for example configs/sam2.1/sam2.1_hiera_l.yaml")
        try:
            import torch
            try:
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
                from sam2.build_sam import build_sam2
            except ModuleNotFoundError as exc:
                # A process started before an editable install may not see its
                # new .pth entry. Make the checked-in official source available
                # without downloading or mutating the environment at runtime.
                vendor_root = Path(__file__).resolve().parents[2] / "third_party" / "sam2"
                if exc.name == "sam2" and (vendor_root / "sam2" / "__init__.py").is_file():
                    sys.path.insert(0, str(vendor_root))
                    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
                    from sam2.build_sam import build_sam2
                else:
                    raise
        except ImportError as exc:
            missing = getattr(exc, "name", None) or "unknown dependency"
            raise RuntimeError(
                f"SAM 2 import failed for '{missing}' using Python '{sys.executable}': {exc}. "
                "Install the project dependencies with "
                f"'{sys.executable}' -m pip install -r requirements.txt and retry."
            ) from exc
        if self.requested_device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.requested_device
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected but torch.cuda.is_available() is false")
        requested_region_area = int(self.generator_config.get("min_mask_region_area", 0))
        try:
            from sam2 import _C as _sam2_extension  # type: ignore  # noqa: F401

            self.compiled_postprocessing_available = True
        except ImportError:
            self.compiled_postprocessing_available = False
        self.effective_generator_config = dict(self.generator_config)
        if requested_region_area <= 0:
            self.postprocessing_backend = "disabled"
        elif self.device.startswith("cuda") and self.compiled_postprocessing_available:
            self.postprocessing_backend = "sam2_compiled_extension"
        else:
            # SAM 2's connected-component extension is CUDA-only and is often
            # unavailable on native Windows/CPU. Disable that predictor hook to
            # avoid its warning, then apply the equivalent holes/islands cleanup
            # to the returned masks with the official OpenCV helper.
            from sam2.utils.amg import remove_small_regions

            self.effective_generator_config["min_mask_region_area"] = 0
            self.cpu_postprocess_region_area = requested_region_area
            self._remove_small_regions = remove_small_regions
            self.postprocessing_backend = "opencv_cpu_fallback"
        self.model = build_sam2(
            self.model_config,
            str(self.checkpoint),
            device=self.device,
            mode="eval",
            apply_postprocessing=True,
        )
        self.generator = SAM2AutomaticMaskGenerator(self.model, **self.effective_generator_config)

    def health_check(self) -> dict[str, Any]:
        try:
            import torch

            version, cuda = torch.__version__, torch.cuda.is_available()
            gpu = torch.cuda.get_device_name(0) if cuda else None
        except ImportError:
            version, cuda, gpu = None, False, None
        memory = None
        if self.model is not None:
            memory = sum(p.numel() * p.element_size() for p in self.model.parameters())
        return {
            "backend_name": "official-meta-sam2",
            "model_type": self.model_type,
            "model_config": self.model_config,
            "checkpoint_path": str(self.checkpoint) if self.checkpoint else None,
            "checkpoint_available": bool(self.checkpoint and self.checkpoint.is_file()),
            "selected_device": self.device,
            "pytorch_version": version,
            "cuda_available": cuda,
            "gpu_name": gpu,
            "model_loaded_successfully": self.model is not None,
            "approximate_model_bytes": memory,
            "mixed_precision_active": bool(self.mixed_precision and self.device.startswith("cuda")),
            "compiled_postprocessing_available": self.compiled_postprocessing_available,
            "postprocessing_backend": self.postprocessing_backend,
            "minimum_mask_region_area": int(self.generator_config.get("min_mask_region_area", 0)),
        }

    def infer_raw(self, image: np.ndarray, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("SAM 2 input must be HWC uint8 RGB")
        self.load()
        import torch

        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        try:
            with torch.inference_mode():
                enabled = bool(self.mixed_precision and self.device.startswith("cuda"))
                with torch.autocast(
                    device_type="cuda" if self.device.startswith("cuda") else "cpu",
                    dtype=torch.bfloat16,
                    enabled=enabled,
                ):
                    masks = self.generator.generate(image)
            if self.cpu_postprocess_region_area > 0:
                masks = self._apply_cpu_postprocessing(masks, image.shape[:2])
            return masks
        except torch.cuda.OutOfMemoryError as exc:
            raise RuntimeError(
                "CUDA out of memory during SAM 2 inference. Reduce points_per_batch, "
                "points_per_side, or crop_n_layers; the configured model is unchanged."
            ) from exc
        finally:
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

    def _apply_cpu_postprocessing(
        self,
        records: list[dict[str, Any]],
        shape: tuple[int, int],
    ) -> list[dict[str, Any]]:
        if self._remove_small_regions is None:
            return records
        from .masks import decode_raw_segmentation

        for record in records:
            segmentation = record["segmentation"]
            mask = (
                segmentation.astype(bool)
                if isinstance(segmentation, np.ndarray)
                else decode_raw_segmentation(segmentation, shape)
            )
            mask, holes_changed = self._remove_small_regions(
                mask,
                self.cpu_postprocess_region_area,
                mode="holes",
            )
            mask, islands_changed = self._remove_small_regions(
                mask,
                self.cpu_postprocess_region_area,
                mode="islands",
            )
            record["segmentation"] = mask.astype(bool)
            rows, columns = np.nonzero(mask)
            if rows.size:
                x0, x1 = int(columns.min()), int(columns.max()) + 1
                y0, y1 = int(rows.min()), int(rows.max()) + 1
                record["bbox"] = [x0, y0, x1 - x0, y1 - y0]
            else:
                record["bbox"] = [0, 0, 0, 0]
            record["area"] = int(mask.sum())
            record["postprocessing"] = {
                "backend": "opencv_cpu_fallback",
                "minimum_region_area": self.cpu_postprocess_region_area,
                "holes_changed": bool(holes_changed),
                "islands_changed": bool(islands_changed),
            }
        return records


class FakeSamBackend(SegmentationBackend):
    """Deterministic test backend; never used implicitly."""

    def __init__(self):
        self.loaded = False

    def load(self) -> None:
        self.loaded = True

    def health_check(self) -> dict[str, Any]:
        return {"backend_name": "fake-sam", "model_type": "synthetic", "checkpoint_path": None, "checkpoint_available": True, "selected_device": "cpu", "pytorch_version": None, "cuda_available": False, "gpu_name": None, "model_loaded_successfully": self.loaded, "approximate_model_bytes": 0}

    def infer_raw(self, image: np.ndarray, metadata: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.load()
        h, w = image.shape[:2]
        mask = np.zeros((h, w), bool)
        mask[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = True
        return [{"segmentation": mask, "area": int(mask.sum()), "bbox": [w // 4, h // 4, w // 2, h // 2], "predicted_iou": 0.95, "stability_score": 0.97, "point_coords": [[w / 2, h / 2]], "crop_box": [0, 0, w, h]}]


def create_backend(cfg: dict[str, Any], project_root: Path | None = None) -> SegmentationBackend:
    if cfg["backend"] == "fake":
        return FakeSamBackend()
    if cfg["backend"] == "sam2":
        checkpoint = cfg.get("checkpoint")
        if checkpoint and project_root is not None and not Path(checkpoint).is_absolute():
            checkpoint = str(project_root / checkpoint)
        return Sam2Backend(
            checkpoint,
            cfg.get("model_type"),
            cfg.get("model_config"),
            cfg.get("device", "auto"),
            cfg.get("mixed_precision", False),
            cfg["automatic_mask_generator"],
        )
    raise ValueError(f"Unsupported Stage 2 instance backend: {cfg['backend']!r}; use 'sam2' or 'fake'")


def timed_inference(backend: SegmentationBackend, image: np.ndarray, metadata: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    start = time.perf_counter()
    masks = backend.infer_raw(image, metadata)
    elapsed = time.perf_counter() - start
    peak = None
    try:
        import torch
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated()
    except ImportError:
        pass
    return masks, {"mask_count": len(masks), "inference_seconds": elapsed, "peak_gpu_memory_bytes": peak, "health": backend.health_check()}
