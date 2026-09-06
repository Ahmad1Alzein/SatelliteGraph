"""Strict raster, metadata, and lineage invariants for Stage 2 refinement."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np

from src.stage2.classes import NODATA_CLASS_ID, SOURCE_CODES

from .schema import json_safe


class RefinementInvariantError(ValueError):
    """Raised when a refined product cannot be safely consumed downstream."""


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """JSON-safe summary of all checked refinement invariants."""

    valid: bool
    shape: tuple[int, int] | None
    segment_count: int
    valid_pixel_count: int
    assigned_valid_pixel_count: int
    gap_pixel_count: int
    outside_assignment_pixel_count: int
    disconnected_instance_ids: tuple[int, ...]
    semantic_mismatch_instance_ids: tuple[int, ...]
    mask_mismatch_instance_ids: tuple[int, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return json_safe(
            {
                "valid": self.valid,
                "shape": self.shape,
                "segment_count": self.segment_count,
                "valid_pixel_count": self.valid_pixel_count,
                "assigned_valid_pixel_count": self.assigned_valid_pixel_count,
                "gap_pixel_count": self.gap_pixel_count,
                "outside_assignment_pixel_count": self.outside_assignment_pixel_count,
                "disconnected_instance_ids": self.disconnected_instance_ids,
                "semantic_mismatch_instance_ids": self.semantic_mismatch_instance_ids,
                "mask_mismatch_instance_ids": self.mask_mismatch_instance_ids,
                "errors": self.errors,
            }
        )


def validate_refined_segmentation(
    instance_map: np.ndarray,
    semantic_map: np.ndarray,
    valid_mask: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    masks: Sequence[np.ndarray] | None = None,
    *,
    confidence_map: np.ndarray | None = None,
    source_map: np.ndarray | None = None,
    reference_semantic_map: np.ndarray | None = None,
    reference_confidence_map: np.ndarray | None = None,
    reference_source_map: np.ndarray | None = None,
    reference_valid_mask: np.ndarray | None = None,
    require_four_connected: bool = True,
    require_refinement_metadata: bool = True,
    raise_on_error: bool = True,
) -> ValidationReport:
    """Validate exact coverage, connectivity, maps, records, and lineage.

    Reference maps make the refinement contract explicit: connectivity
    splitting and same-class merge/absorb may change instance IDs, but they
    must not alter semantics, confidence, source provenance, or validity.
    """

    errors: list[str] = []
    disconnected: list[int] = []
    semantic_mismatch: list[int] = []
    mask_mismatch: list[int] = []
    instances = np.asarray(instance_map)
    semantics = np.asarray(semantic_map)
    valid = np.asarray(valid_mask, dtype=bool)

    shape: tuple[int, int] | None = None
    if instances.ndim != 2:
        errors.append("instance_map must be two-dimensional")
    else:
        shape = (int(instances.shape[0]), int(instances.shape[1]))
    if semantics.ndim != 2:
        errors.append("semantic_map must be two-dimensional")
    if valid.ndim != 2:
        errors.append("valid_mask must be two-dimensional")
    if shape is not None and (semantics.shape != instances.shape or valid.shape != instances.shape):
        errors.append("instance_map, semantic_map, and valid_mask shapes must agree")

    valid_count = int(valid.sum()) if valid.ndim == 2 else 0
    assigned_count = gap_count = outside_count = 0
    if not errors:
        if not np.issubdtype(instances.dtype, np.integer):
            errors.append("instance_map must have an integer dtype")
        if not np.issubdtype(semantics.dtype, np.integer):
            errors.append("semantic_map must have an integer dtype")
        if np.any(instances < 0):
            errors.append("instance_map must not contain negative IDs")
        gap_count = int((valid & (instances <= 0)).sum())
        assigned_count = int((valid & (instances > 0)).sum())
        outside_count = int((~valid & (instances != 0)).sum())
        if gap_count:
            errors.append(f"{gap_count} valid pixels have no positive instance ID")
        if outside_count:
            errors.append(f"{outside_count} invalid pixels have a nonzero instance ID")
        invalid_semantic = int((~valid & (semantics != NODATA_CLASS_ID)).sum())
        valid_nodata = int((valid & (semantics == NODATA_CLASS_ID)).sum())
        if invalid_semantic:
            errors.append(f"{invalid_semantic} invalid pixels are not semantic nodata")
        if valid_nodata:
            errors.append(f"{valid_nodata} valid pixels are incorrectly semantic nodata")

    _validate_auxiliary_map(
        "confidence_map",
        confidence_map,
        instances.shape if instances.ndim == 2 else None,
        errors,
        valid=valid if valid.ndim == 2 else None,
    )
    _validate_auxiliary_map(
        "source_map",
        source_map,
        instances.shape if instances.ndim == 2 else None,
        errors,
        valid=valid if valid.ndim == 2 else None,
    )

    if confidence_map is not None and not errors:
        confidence = np.asarray(confidence_map)
        if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
            errors.append("confidence_map must contain finite values between zero and one")
        if np.any(confidence[~valid] != 0):
            errors.append("confidence_map must be zero outside valid pixels")
    if source_map is not None and not errors:
        sources = np.asarray(source_map)
        nodata_source = int(SOURCE_CODES["nodata"])
        if np.any(sources[~valid] != nodata_source):
            errors.append("source_map must contain the nodata source code outside valid pixels")
        if np.any(sources[valid] == nodata_source):
            errors.append("source_map must not contain the nodata source code on valid pixels")

    actual_ids: list[int] = []
    if instances.ndim == 2 and np.issubdtype(instances.dtype, np.integer):
        actual_ids = [int(value) for value in np.unique(instances[instances > 0]).tolist()]
        expected_ids = list(range(1, len(records) + 1))
        if actual_ids != expected_ids:
            errors.append(
                f"instance IDs must be contiguous and match record count: expected {expected_ids}, got {actual_ids}"
            )
    if masks is not None and len(masks) != len(records):
        errors.append("mask and record counts differ")

    entity_ids: set[str] = set()
    segment_ids: set[str] = set()
    for expected_id, record in enumerate(records, 1):
        if not isinstance(record, Mapping):
            errors.append(f"record {expected_id} is not a mapping")
            continue
        if int(record.get("instance_id", 0)) != expected_id:
            errors.append(f"record {expected_id} has a mismatching instance_id")
        if int(record.get("mask_index", expected_id - 1)) != expected_id - 1:
            errors.append(f"record {expected_id} has a mismatching mask_index")
        entity_id = str(record.get("entity_id", ""))
        segment_id = str(record.get("segment_id", ""))
        if not entity_id or entity_id in entity_ids:
            errors.append(f"record {expected_id} has an empty or duplicate entity_id")
        if not segment_id or segment_id in segment_ids:
            errors.append(f"record {expected_id} has an empty or duplicate segment_id")
        entity_ids.add(entity_id)
        segment_ids.add(segment_id)

        if require_refinement_metadata:
            primary_stage2 = record.get("stage2_metadata")
            if not isinstance(primary_stage2, Mapping):
                errors.append(f"record {expected_id} does not preserve stage2_metadata")
            lineage = record.get("lineage")
            if not isinstance(lineage, Mapping):
                errors.append(f"record {expected_id} has no lineage mapping")
            source_ids = record.get("source_stage2_instance_ids")
            if (
                not isinstance(source_ids, Sequence)
                or isinstance(source_ids, (str, bytes, bytearray))
                or not source_ids
                or any(int(value) <= 0 for value in source_ids)
            ):
                errors.append(f"record {expected_id} has invalid Stage 2 source IDs")
            else:
                clean_source_ids = sorted({int(value) for value in source_ids})
                originals = record.get("stage2_metadata_records")
                if (
                    not isinstance(originals, Sequence)
                    or isinstance(originals, (str, bytes, bytearray))
                    or not originals
                    or any(not isinstance(value, Mapping) for value in originals)
                ):
                    errors.append(f"record {expected_id} has no Stage 2 metadata record list")
                else:
                    original_ids = sorted(
                        {
                            int(value.get("instance_id", 0))
                            for value in originals
                            if value.get("instance_id") is not None
                        }
                    )
                    if original_ids != clean_source_ids:
                        errors.append(
                            f"record {expected_id} Stage 2 metadata IDs disagree with its lineage"
                        )
                if isinstance(primary_stage2, Mapping) and int(
                    primary_stage2.get("instance_id", 0)
                ) not in clean_source_ids:
                    errors.append(
                        f"record {expected_id} primary Stage 2 metadata is not a declared ancestor"
                    )
                if isinstance(lineage, Mapping):
                    lineage_ids = sorted(
                        int(value)
                        for value in lineage.get("source_stage2_instance_ids", [])
                    )
                    if lineage_ids != clean_source_ids:
                        errors.append(f"record {expected_id} lineage mapping is stale")
                    if list(lineage.get("refinement_operation_ids", [])) != list(
                        record.get("refinement_history", [])
                    ):
                        errors.append(
                            f"record {expected_id} operation history and lineage mapping disagree"
                        )

        if expected_id not in actual_ids:
            continue
        expected_mask = instances == expected_id
        area = int(expected_mask.sum())
        if int(record.get("area_pixels", -1)) != area:
            errors.append(f"record {expected_id} pixel area disagrees with instance_map")
        if list(record.get("bbox_xyxy", [])) != _bbox_xyxy(expected_mask):
            errors.append(f"record {expected_id} bbox_xyxy disagrees with instance_map")
        actual_component_count = _component_count_4(expected_mask)
        if actual_component_count != 1:
            if require_four_connected:
                disconnected.append(expected_id)
            else:
                metrics = record.get("metrics")
                declared_count = record.get("connected_component_count")
                declared_connected = record.get("is_four_connected")
                if isinstance(metrics, Mapping):
                    if declared_count is None:
                        declared_count = metrics.get("connected_component_count")
                    if declared_connected is None:
                        declared_connected = metrics.get("is_four_connected")
                if (
                    declared_count is None
                    or int(declared_count) != actual_component_count
                    or declared_connected is not False
                ):
                    disconnected.append(expected_id)
        if masks is not None and expected_id - 1 < len(masks):
            supplied = np.asarray(masks[expected_id - 1], dtype=bool)
            if supplied.shape != instances.shape or not np.array_equal(supplied, expected_mask):
                mask_mismatch.append(expected_id)
        elif masks is not None:
            mask_mismatch.append(expected_id)

        values = np.unique(semantics[expected_mask]) if semantics.shape == instances.shape else np.array([])
        label = record.get("label")
        label_id = label.get("id") if isinstance(label, Mapping) else None
        if len(values) != 1 or label_id is None or int(label_id) != int(values[0]):
            semantic_mismatch.append(expected_id)
        else:
            for field in ("semantic_class_id", "class_id"):
                if record.get(field) is not None and int(record[field]) != int(values[0]):
                    semantic_mismatch.append(expected_id)
                    break

    if disconnected:
        if require_four_connected:
            errors.append(f"instances are not 4-connected: {disconnected}")
        else:
            errors.append(
                "instances are disconnected without exact multipart metadata: "
                f"{disconnected}"
            )
    if semantic_mismatch:
        semantic_mismatch = sorted(set(semantic_mismatch))
        errors.append(f"instance semantics and records disagree: {semantic_mismatch}")
    if mask_mismatch:
        errors.append(f"stored masks and instance_map disagree: {mask_mismatch}")

    _require_unchanged("semantic_map", semantics, reference_semantic_map, errors)
    _require_unchanged("confidence_map", confidence_map, reference_confidence_map, errors)
    _require_unchanged("source_map", source_map, reference_source_map, errors)
    _require_unchanged("valid_mask", valid, reference_valid_mask, errors)

    report = ValidationReport(
        valid=not errors,
        shape=shape,
        segment_count=len(records),
        valid_pixel_count=valid_count,
        assigned_valid_pixel_count=assigned_count,
        gap_pixel_count=gap_count,
        outside_assignment_pixel_count=outside_count,
        disconnected_instance_ids=tuple(disconnected),
        semantic_mismatch_instance_ids=tuple(semantic_mismatch),
        mask_mismatch_instance_ids=tuple(mask_mismatch),
        errors=tuple(errors),
    )
    if errors and raise_on_error:
        raise RefinementInvariantError("Refinement invariants failed: " + "; ".join(errors))
    return report


def validate_mark_nodata_mask(pixel_mask: np.ndarray, valid_mask: np.ndarray) -> None:
    """Enforce that a ``mark_nodata`` operation touches only invalid pixels."""

    selected = np.asarray(pixel_mask, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)
    if selected.ndim != 2 or valid.ndim != 2 or selected.shape != valid.shape:
        raise RefinementInvariantError("mark_nodata mask and valid_mask must have the same 2D shape")
    invalid_request_count = int((selected & valid).sum())
    if invalid_request_count:
        raise RefinementInvariantError(
            f"mark_nodata cannot remove {invalid_request_count} valid image pixels"
        )


def is_four_connected(mask: np.ndarray) -> bool:
    """Return whether a non-empty mask consists of one 4-connected component."""

    value = np.asarray(mask, dtype=bool)
    return bool(value.ndim == 2 and value.any() and _component_count_4(value) == 1)


def _component_count_4(mask: np.ndarray) -> int:
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return 0
    cropped = mask[
        int(rows.min()) : int(rows.max()) + 1,
        int(columns.min()) : int(columns.max()) + 1,
    ]
    number, _ = cv2.connectedComponents(
        np.ascontiguousarray(cropped, dtype=np.uint8),
        connectivity=4,
        ltype=cv2.CV_32S,
    )
    return int(number - 1)


def _bbox_xyxy(mask: np.ndarray) -> list[int]:
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return []
    return [
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    ]


def _validate_auxiliary_map(
    name: str,
    value: np.ndarray | None,
    shape: tuple[int, ...] | None,
    errors: list[str],
    *,
    valid: np.ndarray | None,
) -> None:
    if value is None:
        return
    array = np.asarray(value)
    if array.ndim != 2:
        errors.append(f"{name} must be two-dimensional")
    elif shape is not None and array.shape != shape:
        errors.append(f"{name} shape does not match instance_map")


def _require_unchanged(
    name: str,
    value: np.ndarray | None,
    reference: np.ndarray | None,
    errors: list[str],
) -> None:
    if reference is None:
        return
    if value is None:
        errors.append(f"{name} is missing but an unchanged reference was supplied")
        return
    if not np.array_equal(np.asarray(value), np.asarray(reference)):
        errors.append(f"{name} changed during a topology-only/same-class refinement operation")


__all__ = [
    "RefinementInvariantError",
    "ValidationReport",
    "is_four_connected",
    "validate_mark_nodata_mask",
    "validate_refined_segmentation",
]
