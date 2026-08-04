from __future__ import annotations

from copy import deepcopy
from typing import Any

import cv2
import numpy as np

from .config import CLASS_NAMES


def complete_non_tree_landscape(
    sam_masks: list[np.ndarray],
    sam_records: list[dict[str, Any]],
    label_map: np.ndarray,
    confidence: np.ndarray,
    valid: np.ndarray,
    config: dict[str, Any],
) -> tuple[list[np.ndarray], list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    """Clip trees from SAM masks and fill missed non-tree regions semantically."""
    if label_map.shape != valid.shape or confidence.shape != valid.shape:
        raise ValueError("Land-cover, confidence, and valid-pixel shapes must agree")
    if len(sam_masks) != len(sam_records):
        raise ValueError("SAM mask and record counts differ")
    tree_id = CLASS_NAMES.index("tree")
    tree_mask = valid & (label_map == tree_id) & (confidence >= float(config["tree_confidence_threshold"]))
    dilation = int(config.get("tree_dilation_pixels", 0))
    if dilation > 0:
        kernel = np.ones((2 * dilation + 1, 2 * dilation + 1), np.uint8)
        tree_mask = cv2.dilate(tree_mask.astype(np.uint8), kernel, iterations=1).astype(bool) & valid

    output_masks: list[np.ndarray] = []
    output_records: list[dict[str, Any]] = []
    removed_tree_masks: list[dict[str, Any]] = []
    clipped_count = 0
    minimum_sam_area = int(config["minimum_sam_component_area"])
    overlap_threshold = float(config["sam_tree_overlap_threshold"])

    ordered = sorted(
        zip(sam_masks, sam_records),
        key=lambda item: item[1].get("predicted_iou") if item[1].get("predicted_iou") is not None else -1,
        reverse=True,
    )
    for mask, source_record in ordered:
        mask = np.asarray(mask, bool)
        if mask.shape != valid.shape:
            raise ValueError(f"SAM mask {source_record.get('segment_id')} shape {mask.shape} != {valid.shape}")
        original_area = int(mask.sum())
        tree_overlap = float(np.logical_and(mask, tree_mask).sum() / max(1, original_area))
        clipped = mask & valid & ~tree_mask
        if tree_overlap >= overlap_threshold and not clipped.any():
            removed_tree_masks.append({"segment_id": source_record.get("segment_id"), "tree_overlap_fraction": tree_overlap})
            continue
        components = _components(clipped, minimum_sam_area)
        if tree_overlap > 0 or len(components) > 1:
            clipped_count += 1
        for component_index, component in enumerate(components):
            record = deepcopy(source_record)
            source_id = str(source_record.get("segment_id", "sam"))
            record["segment_id"] = f"sam_{source_id}_{component_index:02d}"
            record["source"] = "stage2_sam"
            record["semantic_class"] = _majority_class(component, label_map, confidence, config)
            record["tree_overlap_fraction_before_clipping"] = tree_overlap
            record.setdefault("provenance", {})["landscape_operation"] = "tree_clipped_and_split"
            record["provenance"]["stage2_segment_id"] = source_id
            _refresh(record, component)
            output_masks.append(component)
            output_records.append(record)

    occupied = np.logical_or.reduce(output_masks) if output_masks else np.zeros(valid.shape, bool)
    added_by_class: dict[str, int] = {}
    semantic_threshold = float(config["semantic_confidence_threshold"])
    component_minimums = config["minimum_component_area"]
    for class_name in config["priority"]:
        class_id = CLASS_NAMES.index(class_name)
        candidate = valid & ~tree_mask & ~occupied & (label_map == class_id) & (confidence >= semantic_threshold)
        candidate = _clean_class_mask(candidate, class_name, config)
        candidate &= valid & ~tree_mask & ~occupied
        components = _components(candidate, int(component_minimums.get(class_name, 1)))
        added_by_class[class_name] = len(components)
        for index, component in enumerate(components):
            record = {
                "segment_id": f"oem_{class_id:02d}_{index:05d}",
                "source_tile_id": sam_records[0].get("source_tile_id", "unknown") if sam_records else "unknown",
                "predicted_iou": None,
                "stability_score": None,
                "valid_pixel_fraction": 1.0,
                "source_raw_mask_id": None,
                "source": "openearthmap_fasterseg",
                "semantic_class": class_name,
                "mean_semantic_confidence": float(confidence[component].mean()),
                "provenance": {"landcover_class_id": class_id, "operation": "fill_uncovered"},
            }
            _refresh(record, component)
            output_masks.append(component)
            output_records.append(record)
            occupied |= component

    unknown_added = 0
    if config.get("fill_low_confidence_as_unknown", True):
        remainder = valid & ~tree_mask & ~occupied
        for index, component in enumerate(_components(remainder, int(component_minimums.get("unknown", 1)))):
            record = {
                "segment_id": f"unknown_{index:05d}",
                "source_tile_id": sam_records[0].get("source_tile_id", "unknown") if sam_records else "unknown",
                "predicted_iou": None,
                "stability_score": None,
                "valid_pixel_fraction": 1.0,
                "source_raw_mask_id": None,
                "source": "landscape_completion",
                "semantic_class": "unknown",
                "mean_semantic_confidence": float(confidence[component].mean()),
                "provenance": {"operation": "fill_low_confidence_non_tree"},
            }
            _refresh(record, component)
            output_masks.append(component)
            output_records.append(record)
            occupied |= component
            unknown_added += 1

    target = valid & ~tree_mask
    report = {
        "input_sam_mask_count": len(sam_masks),
        "output_segment_count": len(output_masks),
        "tree_pixel_count": int(tree_mask.sum()),
        "tree_percentage_of_valid": float(100 * tree_mask.sum() / max(1, valid.sum())),
        "sam_masks_clipped_or_split": clipped_count,
        "sam_masks_removed_as_tree": removed_tree_masks,
        "semantic_components_added_by_class": added_by_class,
        "unknown_components_added": unknown_added,
        "non_tree_valid_pixel_count": int(target.sum()),
        "covered_non_tree_pixel_count": int((target & occupied).sum()),
        "non_tree_coverage_percentage": float(100 * (target & occupied).sum() / max(1, target.sum())),
        "remaining_uncovered_non_tree_pixels": int((target & ~occupied).sum()),
    }
    return output_masks, output_records, tree_mask, report


def _components(mask: np.ndarray, minimum_area: int) -> list[np.ndarray]:
    number, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [labels == index for index in range(1, number) if int(stats[index, cv2.CC_STAT_AREA]) >= minimum_area]


def _clean_class_mask(mask: np.ndarray, class_name: str, config: dict[str, Any]) -> np.ndarray:
    pixels = int(config.get("morphology_pixels", {}).get(class_name, 0))
    if pixels <= 0:
        return mask
    kernel = np.ones((2 * pixels + 1, 2 * pixels + 1), np.uint8)
    operation = cv2.MORPH_CLOSE if class_name == "road" else cv2.MORPH_OPEN
    return cv2.morphologyEx(mask.astype(np.uint8), operation, kernel).astype(bool)


def _majority_class(mask: np.ndarray, labels: np.ndarray, confidence: np.ndarray, config: dict[str, Any]) -> str:
    confident = mask & (confidence >= float(config["semantic_confidence_threshold"]))
    if not confident.any():
        return "unknown"
    counts = np.bincount(labels[confident], minlength=len(CLASS_NAMES))
    counts[CLASS_NAMES.index("tree")] = 0
    return CLASS_NAMES[int(counts.argmax())] if counts.max() else "unknown"


def _refresh(record: dict[str, Any], mask: np.ndarray) -> None:
    rows, cols = np.nonzero(mask)
    x0, x1, y0, y1 = int(cols.min()), int(cols.max()) + 1, int(rows.min()), int(rows.max()) + 1
    record.update(
        area_pixels=int(mask.sum()),
        bbox_xywh=[x0, y0, x1 - x0, y1 - y0],
        centroid_rc=[float(rows.mean()), float(cols.mean())],
        touches_tile_boundary=bool(x0 == 0 or y0 == 0 or x1 == mask.shape[1] or y1 == mask.shape[0]),
    )
