"""Deterministic fusion of SAM 2 instances and OpenEarthMap semantics.

The fusion contract is deliberately lossless about where each part of a final
segment came from: SAM 2 may provide geometry, while OpenEarthMap provides the
semantic label.  Pixels not claimed by SAM 2 are completed from connected
OpenEarthMap components, and every remaining valid pixel becomes ``unknown``.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np

from .classes import (
    CLASS_NAMES,
    NODATA_CLASS_ID,
    SOURCE_CODES,
    UNKNOWN_CLASS_ID,
    class_registry,
)
from .coordinates import attach_segment_coordinates


FusionOutput = tuple[
    list[np.ndarray],
    list[dict[str, Any]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]


def fuse_segmentations(
    sam_masks: Sequence[np.ndarray],
    sam_records: Sequence[Mapping[str, Any]],
    label_map: np.ndarray,
    confidence: np.ndarray,
    valid: np.ndarray,
    tile_metadata: Mapping[str, Any],
    config: Mapping[str, Any] | None,
) -> FusionOutput:
    """Return one exclusive, gap-free segmentation from the two model outputs.

    The returned tuple is ``(masks, records, instance_map, semantic_map,
    confidence_map, source_map, report)``.  Instance id zero is reserved for
    nodata.  Semantic/source nodata is 255; valid semantic ids are 0..8.

    ``confidence_map`` is confidence in the *final* label.  For direct OEM
    segments this is the OEM maximum-class confidence.  For SAM-majority
    segments OEM confidence is retained where the OEM pixel label agrees with
    the selected majority label and is zero elsewhere because class-specific
    probabilities were not supplied.  Fusion ``unknown`` has confidence zero.
    """
    cfg = dict(config or {})
    if cfg.get("fill_low_confidence_as_unknown", True) is not True:
        raise ValueError(
            "fill_low_confidence_as_unknown must be true because the full Stage 2 contract requires gap-free valid-pixel coverage"
        )
    labels, safe_confidence, usable_confidence, valid_mask = _validate_inputs(
        sam_masks, sam_records, label_map, confidence, valid, tile_metadata
    )
    shape = valid_mask.shape
    threshold = _unit_interval(
        cfg.get(
            "semantic_confidence_threshold",
            cfg.get("oem_confidence_threshold", cfg.get("confidence_threshold", 0.0)),
        ),
        "semantic_confidence_threshold",
    )
    minimum_sam_area = _positive_int(cfg.get("minimum_sam_component_area", 1), "minimum_sam_component_area")
    minimum_oem_areas = _minimum_oem_areas(cfg)

    semantic_map = np.full(shape, NODATA_CLASS_ID, dtype=np.uint8)
    source_map = np.full(shape, SOURCE_CODES["nodata"], dtype=np.uint8)
    instance_map = np.zeros(shape, dtype=np.uint32)
    fused_confidence = np.zeros(shape, dtype=np.float32)
    occupied = np.zeros(shape, dtype=bool)
    masks: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    tile_id = str(tile_metadata.get("tile_id") or _first_tile_id(sam_records) or "unknown")

    def append_segment(
        component: np.ndarray,
        record: dict[str, Any],
        class_id: int,
        source_key: str,
        confidence_values: np.ndarray | float,
    ) -> None:
        if not component.any():
            return
        if np.logical_and(component, occupied).any():
            raise RuntimeError("Fusion attempted to assign an already occupied pixel")
        if np.logical_and(component, ~valid_mask).any():
            raise RuntimeError("Fusion attempted to assign a nodata pixel")
        instance_id = len(masks) + 1
        record["instance_id"] = instance_id
        record.setdefault("source_tile_id", tile_id)
        record["semantic_class_id"] = int(class_id)
        record["semantic_class"] = _class_name(class_id)
        record["class_id"] = int(class_id)
        record["class_name"] = _class_name(class_id)
        record["source"] = source_key
        record["source_code"] = int(SOURCE_CODES[source_key])
        record["valid_pixel_fraction"] = 1.0
        _refresh_geometry(record, component)
        if np.isscalar(confidence_values):
            selected_confidence = np.full(int(component.sum()), float(confidence_values), dtype=np.float32)
        else:
            values = np.asarray(confidence_values, dtype=np.float32)
            if values.shape == shape:
                selected_confidence = values[component]
            elif values.ndim == 1 and values.size == int(component.sum()):
                selected_confidence = values
            else:
                raise RuntimeError("Internal fusion confidence shape mismatch")
        mean_final_confidence = float(selected_confidence.mean())
        record["mean_semantic_confidence"] = mean_final_confidence
        if isinstance(record.get("label"), dict):
            label_record = record["label"]
            label_record["mean_final_label_confidence"] = mean_final_confidence
            label_record.setdefault(
                "evidence",
                {
                    key: copy.deepcopy(value)
                    for key, value in label_record.items()
                    if key
                    not in {
                        "id",
                        "name",
                        "source_model",
                        "method",
                        "explanation",
                        "evidence",
                    }
                },
            )
        record = attach_segment_coordinates(record, tile_metadata)

        masks.append(component.copy())
        records.append(record)
        occupied[component] = True
        instance_map[component] = instance_id
        semantic_map[component] = class_id
        source_map[component] = SOURCE_CODES[source_key]
        fused_confidence[component] = selected_confidence

    prepared_sam: list[dict[str, Any]] = []
    for input_index, (mask_value, record_value) in enumerate(zip(sam_masks, sam_records)):
        mask = np.asarray(mask_value, dtype=bool)
        source_id = str(record_value.get("segment_id", f"sam_{input_index:05d}"))
        valid_area = int(np.logical_and(mask, valid_mask).sum())
        reported_area = _finite_float(record_value.get("area_pixels"))
        sort_area = reported_area if reported_area is not None and reported_area >= 0 else float(valid_area)
        prepared_sam.append(
            {
                "input_index": input_index,
                "mask": mask,
                "record": record_value,
                "segment_id": source_id,
                "predicted_iou": _finite_float(record_value.get("predicted_iou")),
                "stability_score": _finite_float(record_value.get("stability_score")),
                "valid_area": valid_area,
                "sort_area": sort_area,
            }
        )
    prepared_sam.sort(key=_sam_sort_key)

    sam_audit: list[dict[str, Any]] = []
    for sort_rank, item in enumerate(prepared_sam):
        source_mask = item["mask"] & valid_mask
        available = source_mask & ~occupied
        all_components = _components(available)
        accepted_components = [component for component in all_components if int(component.sum()) >= minimum_sam_area]
        audit = {
            "source_segment_id": item["segment_id"],
            "sort_rank": sort_rank,
            "predicted_iou": item["predicted_iou"],
            "stability_score": item["stability_score"],
            "sort_area": item["sort_area"],
            "valid_input_area": item["valid_area"],
            "pixels_removed_by_higher_priority_sam": int((source_mask & occupied).sum()),
            "unoccupied_area": int(available.sum()),
            "connected_component_count": len(all_components),
            "accepted_component_count": len(accepted_components),
            "discarded_small_component_pixels": int(
                sum(int(component.sum()) for component in all_components if int(component.sum()) < minimum_sam_area)
            ),
        }
        sam_audit.append(audit)
        for component_index, component in enumerate(accepted_components):
            class_id, label_audit = _majority_label(
                component, labels, safe_confidence, usable_confidence, threshold
            )
            record = copy.deepcopy(dict(item["record"]))
            source_id = item["segment_id"]
            input_provenance = record.get("provenance")
            provenance = copy.deepcopy(input_provenance) if isinstance(input_provenance, Mapping) else {}
            provenance["fusion"] = {
                "geometry_model": "sam2",
                "semantic_model": "openearthmap_fasterseg",
                "operation": "priority_overlap_subtraction_then_connected_components",
                "source_segment_id": source_id,
                "sam_sort_rank": sort_rank,
                "component_index": component_index,
                "minimum_component_area": minimum_sam_area,
            }
            record.update(
                {
                    "segment_id": f"sam_{sort_rank:05d}_{component_index:05d}",
                    "source_tile_id": record.get("source_tile_id", tile_id),
                    "geometry_source": {
                        "model": "sam2",
                        "method": "priority_overlap_subtraction_then_connected_components",
                        "source_segment_id": source_id,
                        "sort_rank": sort_rank,
                        "component_index": component_index,
                        "explanation": (
                            f"SAM 2 segment '{source_id}' supplied this geometry after higher-priority "
                            "SAM pixels were subtracted and the remainder was split into connected components."
                        ),
                    },
                    "label_source": label_audit["source_model"],
                    "label": label_audit,
                    "provenance": provenance,
                }
            )
            record["explanation"] = _sam_explanation(source_id, label_audit)
            if class_id != UNKNOWN_CLASS_ID:
                final_label_confidence = np.where(
                    labels[component] == class_id,
                    safe_confidence[component],
                    0.0,
                ).astype(np.float32)
            else:
                final_label_confidence = 0.0
            append_segment(
                component,
                record,
                class_id,
                "sam2_geometry_openearthmap_label",
                final_label_confidence,
            )

    oem_audit: dict[str, dict[str, int]] = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        candidate = (
            valid_mask
            & ~occupied
            & (labels == class_id)
            & usable_confidence
            & (safe_confidence >= threshold)
        )
        all_components = _components(candidate)
        minimum_area = minimum_oem_areas[class_name]
        accepted_count = 0
        discarded_pixels = 0
        for component_index, component in enumerate(all_components):
            area = int(component.sum())
            if area < minimum_area:
                discarded_pixels += area
                continue
            accepted_count += 1
            component_confidence = safe_confidence[component]
            label_audit = {
                "id": class_id,
                "name": class_name,
                "source_model": "openearthmap_fasterseg",
                "method": "confident_class_connected_component",
                "confidence_threshold": threshold,
                "supporting_pixel_count": area,
                "component_pixel_count": area,
                "mean_source_confidence": float(component_confidence.mean()),
                "minimum_source_confidence": float(component_confidence.min()),
                "maximum_source_confidence": float(component_confidence.max()),
                "explanation": (
                    f"OpenEarthMap FasterSeg predicted '{class_name}' for every pixel in this "
                    f"connected component at confidence >= {threshold:.3f}."
                ),
            }
            record = {
                "segment_id": f"oem_{class_id:02d}_{component_index:05d}",
                "source_tile_id": tile_id,
                "source_raw_mask_id": None,
                "predicted_iou": None,
                "stability_score": None,
                "geometry_source": {
                    "model": "openearthmap_fasterseg",
                    "method": "uncovered_confident_class_connected_component",
                    "class_id": class_id,
                    "explanation": (
                        f"OpenEarthMap FasterSeg supplied the geometry as an uncovered connected "
                        f"component of confidently predicted '{class_name}' pixels."
                    ),
                },
                "label_source": "openearthmap_fasterseg",
                "label": label_audit,
                "provenance": {
                    "fusion": {
                        "geometry_model": "openearthmap_fasterseg",
                        "semantic_model": "openearthmap_fasterseg",
                        "operation": "uncovered_confident_class_connected_component",
                        "class_id": class_id,
                        "confidence_threshold": threshold,
                        "minimum_component_area": minimum_area,
                    }
                },
                "explanation": (
                    f"Both geometry and the '{class_name}' label come from an uncovered, "
                    "confident OpenEarthMap FasterSeg connected component."
                ),
            }
            append_segment(
                component,
                record,
                class_id,
                "openearthmap_geometry_and_label",
                safe_confidence[component],
            )
        oem_audit[class_name] = {
            "candidate_pixels": int(candidate.sum()),
            "connected_component_count": len(all_components),
            "accepted_component_count": accepted_count,
            "discarded_small_component_pixels": discarded_pixels,
            "minimum_component_area": minimum_area,
        }

    remainder = valid_mask & ~occupied
    unknown_components = _components(remainder)
    unknown_reason_totals = _unknown_reason_counts(
        remainder, labels, safe_confidence, usable_confidence, threshold
    )
    for component_index, component in enumerate(unknown_components):
        reason_counts = _unknown_reason_counts(
            component, labels, safe_confidence, usable_confidence, threshold
        )
        mean_oem = float(safe_confidence[component].mean())
        label_audit = {
            "id": UNKNOWN_CLASS_ID,
            "name": "unknown",
            "source_model": "stage2_fusion_rule",
            "method": "residual_valid_pixel_fallback",
            "confidence_threshold": threshold,
            "component_pixel_count": int(component.sum()),
            "mean_oem_max_confidence": mean_oem,
            "reason_pixel_counts": reason_counts,
            "explanation": (
                "No accepted SAM/OpenEarthMap assignment covered these valid pixels. "
                "They are explicitly labeled unknown so the final segmentation has no gaps."
            ),
        }
        record = {
            "segment_id": f"unknown_{component_index:05d}",
            "source_tile_id": tile_id,
            "source_raw_mask_id": None,
            "predicted_iou": None,
            "stability_score": None,
            "geometry_source": {
                "model": "stage2_fusion_rule",
                "method": "residual_valid_pixel_connected_component",
                "explanation": (
                    "The gap-free fusion rule supplied this geometry from connected valid pixels "
                    "left after all accepted SAM and OpenEarthMap components."
                ),
            },
            "label_source": "stage2_fusion_rule",
            "label": label_audit,
            "provenance": {
                "fusion": {
                    "operation": "gap_free_unknown_fallback",
                    "semantic_confidence_threshold": threshold,
                    "reason_pixel_counts": reason_counts,
                }
            },
            "explanation": label_audit["explanation"],
        }
        append_segment(component, record, UNKNOWN_CLASS_ID, "fusion_unknown", 0.0)

    report = _build_report(
        valid_mask=valid_mask,
        masks=masks,
        records=records,
        semantic_map=semantic_map,
        source_map=source_map,
        instance_map=instance_map,
        confidence_map=fused_confidence,
        threshold=threshold,
        minimum_sam_area=minimum_sam_area,
        minimum_oem_areas=minimum_oem_areas,
        sam_audit=sam_audit,
        oem_audit=oem_audit,
        unknown_reason_totals=unknown_reason_totals,
        invalid_oem_label_pixels=int((valid_mask & (labels < 0)).sum()),
        invalid_oem_confidence_pixels=int((valid_mask & ~usable_confidence).sum()),
    )
    return masks, records, instance_map, semantic_map, fused_confidence, source_map, report


def _validate_inputs(
    sam_masks: Sequence[np.ndarray],
    sam_records: Sequence[Mapping[str, Any]],
    label_map: np.ndarray,
    confidence: np.ndarray,
    valid: np.ndarray,
    tile_metadata: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(sam_masks) != len(sam_records):
        raise ValueError("SAM mask and record counts differ")
    if not isinstance(tile_metadata, Mapping):
        raise TypeError("tile_metadata must be a mapping")
    valid_mask = np.asarray(valid, dtype=bool)
    if valid_mask.ndim != 2:
        raise ValueError("valid must be a two-dimensional mask")
    raw_labels = np.asarray(label_map)
    raw_confidence = np.asarray(confidence)
    if raw_labels.shape != valid_mask.shape or raw_confidence.shape != valid_mask.shape:
        raise ValueError("OpenEarthMap label_map, confidence, and valid shapes must agree")
    if not np.issubdtype(raw_labels.dtype, np.number):
        raise TypeError("OpenEarthMap label_map must be numeric")
    if not np.issubdtype(raw_confidence.dtype, np.number):
        raise TypeError("OpenEarthMap confidence must be numeric")
    for index, (mask, record) in enumerate(zip(sam_masks, sam_records)):
        if np.asarray(mask).shape != valid_mask.shape:
            raise ValueError(f"SAM mask {index} shape {np.asarray(mask).shape} != {valid_mask.shape}")
        if not isinstance(record, Mapping):
            raise TypeError(f"SAM record {index} must be a mapping")

    numeric_labels = raw_labels.astype(np.float64, copy=False)
    integral = np.isfinite(numeric_labels) & (numeric_labels == np.floor(numeric_labels))
    known = integral & (numeric_labels >= 0) & (numeric_labels < len(CLASS_NAMES))
    labels = np.full(valid_mask.shape, -1, dtype=np.int16)
    labels[known] = numeric_labels[known].astype(np.int16)

    numeric_confidence = raw_confidence.astype(np.float32, copy=False)
    usable = np.isfinite(numeric_confidence) & (numeric_confidence >= 0.0) & (numeric_confidence <= 1.0)
    safe = np.where(usable, numeric_confidence, 0.0).astype(np.float32)
    return labels, safe, usable, valid_mask


def _majority_label(
    component: np.ndarray,
    labels: np.ndarray,
    confidence: np.ndarray,
    usable_confidence: np.ndarray,
    threshold: float,
) -> tuple[int, dict[str, Any]]:
    confident = component & (labels >= 0) & usable_confidence & (confidence >= threshold)
    counts = np.bincount(labels[confident], minlength=len(CLASS_NAMES)).astype(np.int64)
    confident_count = int(counts.sum())
    vote_counts = {name: int(counts[class_id]) for class_id, name in enumerate(CLASS_NAMES)}
    if confident_count == 0:
        audit = {
            "id": UNKNOWN_CLASS_ID,
            "name": "unknown",
            "source_model": "stage2_fusion_rule",
            "method": "openearthmap_confident_pixel_majority_no_votes",
            "confidence_threshold": threshold,
            "component_pixel_count": int(component.sum()),
            "confident_voter_count": 0,
            "supporting_pixel_count": 0,
            "vote_counts": vote_counts,
            "tie_break_rule": "lowest id in the fixed OpenEarthMap class order",
            "explanation": (
                "SAM 2 supplied the geometry, but no OpenEarthMap pixel in the component "
                f"had a valid class and confidence >= {threshold:.3f}; fusion assigned unknown."
            ),
        }
        return UNKNOWN_CLASS_ID, audit

    class_id = int(np.argmax(counts))
    support = component & confident & (labels == class_id)
    tied_ids = np.flatnonzero(counts == counts[class_id]).tolist()
    audit = {
        "id": class_id,
        "name": CLASS_NAMES[class_id],
        "source_model": "openearthmap_fasterseg",
        "method": "confident_pixel_unweighted_majority_vote",
        "confidence_threshold": threshold,
        "component_pixel_count": int(component.sum()),
        "confident_voter_count": confident_count,
        "supporting_pixel_count": int(counts[class_id]),
        "support_fraction_of_confident_votes": float(counts[class_id] / confident_count),
        "mean_supporting_confidence": float(confidence[support].mean()),
        "vote_counts": vote_counts,
        "tied_class_ids": [int(value) for value in tied_ids],
        "tie_break_rule": "lowest id in the fixed OpenEarthMap class order",
        "explanation": (
            f"OpenEarthMap label '{CLASS_NAMES[class_id]}' won an unweighted majority of "
            f"confident pixels ({int(counts[class_id])}/{confident_count}); ties resolve to "
            "the lowest id in the fixed OpenEarthMap class order."
        ),
    }
    return class_id, audit


def _components(mask: np.ndarray) -> list[np.ndarray]:
    if not mask.any():
        return []
    number, component_map, stats, _ = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(mask, dtype=np.uint8), connectivity=8
    )
    values: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for component_id in range(1, number):
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        values.append(((y, x, -area, component_id), component_map == component_id))
    values.sort(key=lambda value: value[0])
    return [component for _, component in values]


def _refresh_geometry(record: dict[str, Any], mask: np.ndarray) -> None:
    rows, cols = np.nonzero(mask)
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    centroid_row, centroid_col = float(rows.mean()), float(cols.mean())
    area = int(mask.sum())
    width, height = x1 - x0, y1 - y0
    touches_boundary = bool(
        x0 == 0 or y0 == 0 or x1 == mask.shape[1] or y1 == mask.shape[0]
    )
    record.update(
        {
            "area_pixels": area,
            "bbox_xywh": [x0, y0, width, height],
            "bbox_xyxy": [x0, y0, x1, y1],
            "centroid_rc": [centroid_row, centroid_col],
            "centroid_xy": [centroid_col + 0.5, centroid_row + 0.5],
            "touches_tile_boundary": touches_boundary,
            "accepted": True,
            "rejection_reasons": [],
        }
    )
    # Filtering metrics copied from a source SAM record describe the pre-fusion
    # mask. Refresh every geometry-dependent value after overlap subtraction
    # and connected-component splitting so nested metadata cannot go stale.
    metrics = copy.deepcopy(record.get("metrics")) if isinstance(record.get("metrics"), Mapping) else {}
    metrics.update(
        {
            "area_pixels": area,
            "area_ratio": float(area / mask.size),
            "valid_pixel_fraction": 1.0,
            "nodata_fraction": 0.0,
            "fill_ratio": float(area / max(1, width * height)),
            "touches_tile_boundary": touches_boundary,
        }
    )
    record["metrics"] = metrics


def _build_report(
    *,
    valid_mask: np.ndarray,
    masks: list[np.ndarray],
    records: list[dict[str, Any]],
    semantic_map: np.ndarray,
    source_map: np.ndarray,
    instance_map: np.ndarray,
    confidence_map: np.ndarray,
    threshold: float,
    minimum_sam_area: int,
    minimum_oem_areas: dict[str, int],
    sam_audit: list[dict[str, Any]],
    oem_audit: dict[str, dict[str, int]],
    unknown_reason_totals: dict[str, int],
    invalid_oem_label_pixels: int,
    invalid_oem_confidence_pixels: int,
) -> dict[str, Any]:
    count_map = np.zeros(valid_mask.shape, dtype=np.uint32)
    for mask in masks:
        count_map += mask.astype(np.uint32)
    gap_count = int((valid_mask & (count_map == 0)).sum())
    overlap_count = int((valid_mask & (count_map > 1)).sum())
    outside_count = int((~valid_mask & (count_map > 0)).sum())
    map_mismatch = int((valid_mask & (instance_map == 0)).sum())
    exact = gap_count == overlap_count == outside_count == map_mismatch == 0
    if not exact:
        raise RuntimeError(
            "Fusion invariant failed: "
            f"gaps={gap_count}, overlaps={overlap_count}, outside={outside_count}, "
            f"instance_gaps={map_mismatch}"
        )
    class_pixel_counts = {
        name: int((valid_mask & (semantic_map == class_id)).sum())
        for class_id, name in enumerate((*CLASS_NAMES, "unknown"))
    }
    source_pixel_counts = {
        key: int((valid_mask & (source_map == code)).sum())
        for key, code in SOURCE_CODES.items()
        if key != "nodata"
    }
    class_segment_counts = {name: 0 for name in (*CLASS_NAMES, "unknown")}
    source_segment_counts = {key: 0 for key in SOURCE_CODES if key != "nodata"}
    for record in records:
        class_segment_counts[record["semantic_class"]] += 1
        source_segment_counts[record["source"]] += 1
    return {
        "schema_version": "stage2-fusion-v1",
        "shape": [int(value) for value in valid_mask.shape],
        "class_registry": class_registry(),
        "source_registry": [
            {
                "code": int(code),
                "name": key,
                "description": _source_description(key),
            }
            for key, code in SOURCE_CODES.items()
        ],
        "map_contract": {
            "instance_map": "uint32; 0 is nodata, positive values index records/masks from 1",
            "semantic_map": f"uint8; classes 0..{UNKNOWN_CLASS_ID}, nodata={NODATA_CLASS_ID}",
            "confidence_map": (
                "float32 final-label confidence; zero means unavailable/fusion unknown; nodata is zero"
            ),
            "source_map": f"uint8; see source_registry; nodata={SOURCE_CODES['nodata']}",
        },
        "fusion_rules": {
            "sam_order": (
                "predicted_iou desc, stability_score desc, area_pixels desc "
                "(valid mask area fallback), segment_id asc"
            ),
            "connectivity": 8,
            "semantic_confidence_threshold": threshold,
            "minimum_sam_component_area": minimum_sam_area,
            "minimum_oem_component_area": minimum_oem_areas,
            "sam_label_method": "unweighted majority of confident OpenEarthMap pixels",
            "majority_tie_break": "lowest id in the fixed OpenEarthMap class order",
            "completion": "confident OEM class components including tree, then unknown residual",
        },
        "input_sam_mask_count": len(sam_audit),
        "output_segment_count": len(records),
        "valid_pixel_count": int(valid_mask.sum()),
        "nodata_pixel_count": int((~valid_mask).sum()),
        "assigned_valid_pixel_count": int((valid_mask & (count_map == 1)).sum()),
        "gap_pixel_count": gap_count,
        "overlap_pixel_count": overlap_count,
        "assigned_outside_valid_pixel_count": outside_count,
        "every_valid_pixel_exactly_once": exact,
        "class_pixel_counts": class_pixel_counts,
        "class_segment_counts": class_segment_counts,
        "source_pixel_counts": source_pixel_counts,
        "source_segment_counts": source_segment_counts,
        "sam_processing": sam_audit,
        "openearthmap_completion": oem_audit,
        "unknown_reason_pixel_counts": unknown_reason_totals,
        "invalid_oem_label_pixel_count": invalid_oem_label_pixels,
        "invalid_oem_confidence_pixel_count": invalid_oem_confidence_pixels,
        "confidence_summary_valid": {
            "minimum": float(confidence_map[valid_mask].min()) if valid_mask.any() else None,
            "mean": float(confidence_map[valid_mask].mean()) if valid_mask.any() else None,
            "maximum": float(confidence_map[valid_mask].max()) if valid_mask.any() else None,
        },
    }


def _unknown_reason_counts(
    mask: np.ndarray,
    labels: np.ndarray,
    confidence: np.ndarray,
    usable_confidence: np.ndarray,
    threshold: float,
) -> dict[str, int]:
    invalid_label = mask & (labels < 0)
    invalid_confidence = mask & ~usable_confidence
    low_confidence = mask & (labels >= 0) & usable_confidence & (confidence < threshold)
    confident_but_unassigned = (
        mask & (labels >= 0) & usable_confidence & (confidence >= threshold)
    )
    return {
        "invalid_oem_label": int(invalid_label.sum()),
        "invalid_oem_confidence": int(invalid_confidence.sum()),
        "below_semantic_confidence_threshold": int(low_confidence.sum()),
        "confident_component_below_minimum_area": int(confident_but_unassigned.sum()),
    }


def _minimum_oem_areas(config: Mapping[str, Any]) -> dict[str, int]:
    configured = config.get("minimum_component_area", {})
    alternate = config.get("minimum_oem_component_area", 1)
    result: dict[str, int] = {}
    for name in CLASS_NAMES:
        if isinstance(configured, Mapping) and name in configured:
            value = configured[name]
        elif isinstance(alternate, Mapping):
            value = alternate.get(name, 1)
        else:
            value = alternate
        result[name] = _positive_int(value, f"minimum_component_area.{name}")
    return result


def _sam_sort_key(item: Mapping[str, Any]) -> tuple[float, float, float, str, int]:
    predicted_iou = item["predicted_iou"] if item["predicted_iou"] is not None else -math.inf
    stability = item["stability_score"] if item["stability_score"] is not None else -math.inf
    return (
        -float(predicted_iou),
        -float(stability),
        -float(item["sort_area"]),
        str(item["segment_id"]),
        int(item["input_index"]),
    )


def _sam_explanation(source_id: str, label: Mapping[str, Any]) -> str:
    return (
        f"Geometry comes from SAM 2 segment '{source_id}' after deterministic priority "
        "overlap subtraction and connected-component splitting. "
        + str(label["explanation"])
    )


def _source_description(key: str) -> str:
    return {
        "sam2_geometry_openearthmap_label": (
            "Geometry from SAM 2; semantic label from an OpenEarthMap confident-pixel majority "
            "or the explicit unknown fallback when no confident vote exists."
        ),
        "openearthmap_geometry_and_label": (
            "Geometry and semantic label from an uncovered confident OpenEarthMap connected component."
        ),
        "fusion_unknown": "Valid residual geometry and unknown label created by the gap-free fusion rule.",
        "nodata": "Outside the input valid-pixel mask.",
    }[key]


def _first_tile_id(records: Sequence[Mapping[str, Any]]) -> Any:
    for record in records:
        if record.get("source_tile_id") is not None:
            return record["source_tile_id"]
    return None


def _class_name(class_id: int) -> str:
    return "unknown" if class_id == UNKNOWN_CLASS_ID else CLASS_NAMES[class_id]


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _unit_interval(value: Any, name: str) -> float:
    number = _finite_float(value)
    if number is None or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a finite number between 0 and 1")
    return number


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise ValueError(f"{name} must be a non-negative integer")
    return int(numeric)


__all__ = ["FusionOutput", "fuse_segmentations"]
