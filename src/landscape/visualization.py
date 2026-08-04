from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.stage2.visualization import overlay

from .config import CLASS_NAMES


CLASS_COLORS = np.asarray(
    [[128, 0, 0], [0, 255, 0], [192, 192, 192], [255, 255, 255], [49, 139, 87], [0, 0, 255], [127, 255, 0], [255, 0, 0]],
    dtype=np.uint8,
)


def landcover_map(label_map: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = CLASS_COLORS[label_map].copy()
    result[~valid] = [0, 0, 0]
    return result


def landcover_overlay(image: np.ndarray, label_map: np.ndarray, valid: np.ndarray, alpha: float) -> np.ndarray:
    colors = landcover_map(label_map, valid)
    result = image.astype(np.float32)
    result[valid] = (1 - alpha) * result[valid] + alpha * colors[valid]
    return np.clip(result, 0, 255).astype(np.uint8)


def binary_overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    result = image.astype(np.float32)
    result[mask] = (1 - alpha) * result[mask] + alpha * np.asarray(color)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.clip(result, 0, 255).astype(np.uint8)
    cv2.drawContours(result, contours, -1, color, 2)
    return result


def completed_overlay(image: np.ndarray, masks: list[np.ndarray], records: list[dict[str, Any]], alpha: float, seed: int) -> np.ndarray:
    return overlay(image, masks, records, alpha=alpha, seed=seed, show_ids=False, show_scores=False)


def semantic_label_map(masks: list[np.ndarray], records: list[dict[str, Any]], shape: tuple[int, int]) -> np.ndarray:
    result = np.zeros((*shape, 3), np.uint8)
    for mask, record in zip(masks, records):
        name = record.get("semantic_class", "unknown")
        color = CLASS_COLORS[CLASS_NAMES.index(name)] if name in CLASS_NAMES else np.asarray([255, 165, 0])
        result[mask] = color
    return result
