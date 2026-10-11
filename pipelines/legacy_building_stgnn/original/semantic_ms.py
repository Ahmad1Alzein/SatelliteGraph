"""Scale-corrected SegFormer (OpenEarthMap) inference.

OpenEarthMap is 0.25-0.5 m GSD; the tiles are 1.5 m. The HF processor resizes every
input to 512x512, so the model saw buildings 3-6x smaller than in training and
fused neighbouring roofs. Here the image is upsampled by `scale`, inferred in
overlapping 512 windows (no forced resize), with flip TTA, and probabilities are
returned at native resolution (and optionally at the upsampled resolution).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSemanticSegmentation

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "semantic_hybrid" / "models" / "satlens-segformer"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_model = None


def load_model(device="cuda"):
    global _model
    if _model is None:
        _model = AutoModelForSemanticSegmentation.from_pretrained(str(MODEL_DIR), local_files_only=True).to(device).eval().float()
    return _model


def _window_starts(length, window, stride):
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if starts[-1] != length - window:
        starts.append(length - window)
    return starts


@torch.inference_mode()
def infer(image: np.ndarray, scale: float = 2.0, window: int = 512, overlap: int = 128,
          tta: bool = True, device: str = "cuda", return_upsampled: bool = False):
    model = load_model(device)
    h, w = image.shape[:2]
    x = torch.from_numpy(image).permute(2, 0, 1)[None].float() / 255.0
    H, W = int(round(h * scale)), int(round(w * scale))
    if scale != 1:
        x = F.interpolate(x, size=(H, W), mode="bicubic", align_corners=False).clamp(0, 1)
    x = ((x - MEAN) / STD).to(device)
    n_cls = model.config.num_labels
    acc = torch.zeros((n_cls, H, W), device=device)
    wsum = torch.zeros((1, H, W), device=device)
    # Cosine-ish weight to suppress window-border artefacts.
    ramp = torch.hann_window(window, periodic=False, device=device).clamp_min(0.05)
    weight = (ramp[:, None] * ramp[None, :])[None]
    for y0 in _window_starts(H, window, window - overlap):
        for x0 in _window_starts(W, window, window - overlap):
            crop = x[:, :, y0:y0 + window, x0:x0 + window]
            ch, cw = crop.shape[-2:]
            flips = [None, (-1,), (-2,), (-1, -2)] if tta else [None]
            prob = 0
            for f in flips:
                inp = crop if f is None else torch.flip(crop, f)
                logits = model(pixel_values=inp).logits
                logits = F.interpolate(logits, size=(ch, cw), mode="bilinear", align_corners=False)
                if f is not None:
                    logits = torch.flip(logits, f)
                prob = prob + logits.softmax(1)
            prob = prob[0] / len(flips)
            wt = weight[:, :ch, :cw]
            acc[:, y0:y0 + ch, x0:x0 + cw] += prob * wt
            wsum[:, y0:y0 + ch, x0:x0 + cw] += wt
    up = acc / wsum
    native = F.interpolate(up[None], size=(h, w), mode="area")[0] if scale != 1 else up
    native = native.cpu().numpy().astype(np.float32)
    if return_upsampled:
        return native, up.cpu().numpy().astype(np.float32)
    return native
