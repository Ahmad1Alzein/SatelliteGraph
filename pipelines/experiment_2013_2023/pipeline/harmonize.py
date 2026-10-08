"""Harmonize two acquisitions of the same area before segmenting them for change detection.

Validation showed that sharpness differences between images are the largest source of false
changes for every segmentation method (a sigma-0.7 px blur alone causes ~45 false changes per 100
buildings), followed by radiometric differences. Segmenting both years at the SAME radiometry and
the SAME effective sharpness removes most of that.

1. Radiometry: per-channel histogram matching of `moving` to `reference`.
2. Sharpness: the sharper image is Gaussian-blurred until its mean gradient magnitude equals the
   blurrier image's (bisection on sigma). The blurrier image is never sharpened.

Both inputs must be co-registered RGB uint8 arrays covering the same ground.
"""
from __future__ import annotations

import cv2
import numpy as np


def match_histograms(moving: np.ndarray, reference: np.ndarray) -> np.ndarray:
    out = np.empty_like(moving)
    for c in range(moving.shape[2]):
        src = moving[:, :, c].ravel()
        ref = reference[:, :, c].ravel()
        src_values, src_idx, src_counts = np.unique(src, return_inverse=True, return_counts=True)
        ref_values, ref_counts = np.unique(ref, return_counts=True)
        src_cdf = np.cumsum(src_counts) / src.size
        ref_cdf = np.cumsum(ref_counts) / ref.size
        mapped = np.interp(src_cdf, ref_cdf, ref_values)
        out[:, :, c] = np.clip(np.round(mapped[src_idx]), 0, 255).astype(np.uint8).reshape(moving.shape[:2])
    return out


def sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.sqrt(gx * gx + gy * gy).mean())


def blur_to_sharpness(image: np.ndarray, target: float, max_sigma: float = 3.0) -> tuple[np.ndarray, float]:
    """Smallest Gaussian blur that brings `image` down to `target` sharpness."""
    if sharpness(image) <= target:
        return image, 0.0
    lo, hi = 0.0, max_sigma
    for _ in range(14):
        mid = (lo + hi) / 2
        if sharpness(cv2.GaussianBlur(image, (0, 0), mid)) > target:
            lo = mid
        else:
            hi = mid
    return cv2.GaussianBlur(image, (0, 0), hi), hi


def harmonize_pair(reference: np.ndarray, moving: np.ndarray, tolerance: float = 0.02):
    """Return (reference', moving', info). `moving` takes the reference radiometry; whichever image is
    sharper (by more than `tolerance`, relative) is blurred to the other's sharpness."""
    moving = match_histograms(moving, reference)
    s_ref, s_mov = sharpness(reference), sharpness(moving)
    sigma_ref = sigma_mov = 0.0
    if s_ref > s_mov * (1 + tolerance):
        reference, sigma_ref = blur_to_sharpness(reference, s_mov)
    elif s_mov > s_ref * (1 + tolerance):
        moving, sigma_mov = blur_to_sharpness(moving, s_ref)
    info = {"sharpness_reference": round(s_ref, 3), "sharpness_moving": round(s_mov, 3),
            "blur_sigma_reference": round(sigma_ref, 3), "blur_sigma_moving": round(sigma_mov, 3)}
    return reference, moving, info
