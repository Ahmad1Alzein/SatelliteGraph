from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


def overlay(image: np.ndarray, masks: list[np.ndarray], records: list[dict[str, Any]], alpha: float = .45, seed: int = 42, show_ids: bool = False, show_scores: bool = False) -> np.ndarray:
    result = image.astype(np.float32).copy()
    rng = np.random.default_rng(seed)
    colors = rng.integers(32, 256, size=(len(masks), 3), dtype=np.uint8)
    for mask, color in zip(masks, colors):
        result[mask] = (1 - alpha) * result[mask] + alpha * color
    output = np.clip(result, 0, 255).astype(np.uint8)
    if show_ids or show_scores:
        canvas = Image.fromarray(output); draw = ImageDraw.Draw(canvas)
        for record in records:
            x, y = record["bbox_xywh"][:2]
            text = record["segment_id"] if show_ids else ""
            if show_scores and record.get("predicted_iou") is not None: text += f" {record['predicted_iou']:.2f}"
            draw.text((x, y), text.strip(), fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
        output = np.asarray(canvas)
    return output


def boundary_overlay(image: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    result = image.copy()
    for mask in masks:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (255, 0, 0), 1)
    return result


def label_map(masks: list[np.ndarray], shape: tuple[int, int], seed: int = 42) -> np.ndarray:
    labels = np.zeros(shape, np.int32)
    for index, mask in enumerate(masks, 1): labels[mask] = index
    rng = np.random.default_rng(seed)
    colors = np.vstack([np.zeros((1, 3), np.uint8), rng.integers(32, 256, size=(len(masks), 3), dtype=np.uint8)])
    return colors[labels]


def coverage_preview(count_map: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.zeros((*valid.shape, 3), np.uint8)
    out[~valid] = (80, 80, 80); out[valid & (count_map == 0)] = (220, 40, 40)
    out[valid & (count_map == 1)] = (40, 180, 60); out[valid & (count_map > 1)] = (255, 180, 0)
    return out


def overlap_preview(count_map: np.ndarray) -> np.ndarray:
    scale = np.clip(count_map.astype(float) / max(1, int(count_map.max())), 0, 1)
    return cv2.applyColorMap(np.rint(scale * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]


def comparison(images: list[np.ndarray]) -> np.ndarray:
    if not images: raise ValueError("At least one comparison image is required")
    h = max(im.shape[0] for im in images)
    resized = [cv2.copyMakeBorder(im, 0, h-im.shape[0], 0, 0, cv2.BORDER_CONSTANT) for im in images]
    return np.concatenate(resized, axis=1)


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    Image.fromarray(image).save(temporary, format="PNG")
    temporary.replace(path)


def inspection_rows(records: list[dict[str, Any]], duplicate_groups: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    membership = {}
    for index, group in enumerate(duplicate_groups or []):
        for item in group["members"]: membership[item] = index
    return [{"segment_id": r["segment_id"], "area": r.get("area_pixels"), "bbox": r.get("bbox_xywh"), "centroid": r.get("centroid_rc"), "predicted_iou": r.get("predicted_iou"), "stability_score": r.get("stability_score"), "valid_pixel_fraction": r.get("valid_pixel_fraction"), "boundary_contact": r.get("touches_tile_boundary"), "duplicate_group": membership.get(r["segment_id"]), "accepted": r.get("accepted", True), "rejection_reason": r.get("rejection_reasons", [])} for r in records]
