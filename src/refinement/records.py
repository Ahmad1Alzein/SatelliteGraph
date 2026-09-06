"""Stable lineage and metadata helpers for refined segments."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from src.stage2.coordinates import attach_segment_coordinates

from .schema import json_safe, stable_digest


ENTITY_POLICY_BY_CLASS = {
    "buildings": "one_physical_building",
    "tree": "split_only_across_non_tree_gap",
    "road": "junction_to_junction",
    "agriculture land": "preserve_individual_field_boundaries",
    "water": "connected_homogeneous_region",
    "rangeland": "connected_homogeneous_region",
    "bareland": "connected_homogeneous_region",
    "developed space": "connected_homogeneous_region",
    "unknown": "retain_until_supported",
}

ENTITY_TYPE_BY_CLASS = {
    "buildings": "physical_building",
    "tree": "continuous_canopy",
    "road": "network_reach",
    "agriculture land": "field",
    "water": "continuous_homogeneous_region",
    "rangeland": "continuous_homogeneous_region",
    "bareland": "continuous_homogeneous_region",
    "developed space": "continuous_homogeneous_region",
    "unknown": "unresolved_region",
}


def make_entity_group_id(tile_id: str, stage2_instance_id: int) -> str:
    """Return the stable identity of one unsplit Stage 2 parent."""

    return f"{tile_id}::stage2-entity::s{int(stage2_instance_id):06d}"


def make_component_entity_id(tile_id: str, stage2_instance_id: int, component_index: int) -> str:
    """Return a stable entity ID for a row-major 4-connected component."""

    return (
        f"{tile_id}::refined-entity::s{int(stage2_instance_id):06d}"
        f"::c{int(component_index):04d}"
    )


def make_component_segment_id(stage2_instance_id: int, component_index: int) -> str:
    """Return the compact segment ID used by downstream graph node IDs."""

    return f"refined_s{int(stage2_instance_id):06d}_c{int(component_index):04d}"


def make_component_group_segment_id(stage2_instance_id: int) -> str:
    """Return the graph-facing ID for one consolidated Stage 2 object."""

    return f"refined_group_s{int(stage2_instance_id):06d}"


def make_merge_entity_id(tile_id: str, operation_identity: Any) -> str:
    """Return a stable identity for the union created by a merge."""

    return f"{tile_id}::refined-entity::merge::{stable_digest(operation_identity)[:20]}"


def make_merge_segment_id(operation_identity: Any) -> str:
    """Return the compact segment ID for a merged entity."""

    return f"refined_merge_{stable_digest(operation_identity)[:16]}"


def make_split_group_id(tile_id: str, proposal_id: str) -> str:
    """Return the common stable group identity for reviewed split children."""

    identity = {"tile_id": str(tile_id), "proposal_id": str(proposal_id)}
    return f"{tile_id}::refined-group::split::{stable_digest(identity)[:20]}"


def make_split_entity_id(tile_id: str, proposal_id: str, child_index: int) -> str:
    """Return a stable semantic-split child identity in partition-label order."""

    index = int(child_index)
    if index < 0:
        raise ValueError("semantic split child_index must be non-negative")
    group = make_split_group_id(tile_id, proposal_id)
    return f"{group}::child-{index + 1:04d}"


def make_split_segment_id(proposal_id: str, child_index: int) -> str:
    """Return the compact graph-facing ID for a reviewed split child."""

    index = int(child_index)
    if index < 0:
        raise ValueError("semantic split child_index must be non-negative")
    return f"refined_split_{stable_digest(str(proposal_id))[:16]}_c{index + 1:04d}"


def build_component_record(
    stage2_record: Mapping[str, Any],
    mask: np.ndarray,
    *,
    tile_id: str,
    instance_id: int,
    stage2_instance_id: int,
    component_index: int,
    component_count: int,
    semantic_class_id: int,
    confidence_map: np.ndarray | None = None,
    source_map: np.ndarray | None = None,
    tile_metadata: Mapping[str, Any] | None = None,
    operation_id: str | None = None,
) -> dict[str, Any]:
    """Deep-copy a Stage 2 record and refresh it for one 4-component."""

    # ``json_safe`` already builds a detached recursive copy.
    original = json_safe(dict(stage2_record))
    source_segment_id = str(original.get("segment_id", f"stage2_{stage2_instance_id:06d}"))
    entity_id = make_component_entity_id(tile_id, stage2_instance_id, component_index)
    group_id = make_entity_group_id(tile_id, stage2_instance_id)
    history = [str(operation_id)] if operation_id else []

    record = copy.deepcopy(original)
    record.update(
        {
            "instance_id": int(instance_id),
            "mask_index": int(instance_id) - 1,
            "segment_id": make_component_segment_id(stage2_instance_id, component_index),
            "entity_id": entity_id,
            "entity_group_id": group_id,
            "stage2_instance_id": int(stage2_instance_id),
            "stage2_segment_id": source_segment_id,
            "parent_instance_id": int(stage2_instance_id),
            "parent_segment_id": source_segment_id,
            "component_index": int(component_index),
            "component_count": int(component_count),
            "component_connectivity": 4,
            "source_stage2_instance_ids": [int(stage2_instance_id)],
            "source_stage2_segment_ids": [source_segment_id],
            "refinement_history": history,
            "stage2_metadata": original,
            "stage2_metadata_records": [copy.deepcopy(original)],
        }
    )
    _attach_lineage(record)
    _attach_entity_policy(record)
    refresh_segment_record(
        record,
        mask,
        semantic_class_id=semantic_class_id,
        confidence_map=confidence_map,
        source_map=source_map,
        tile_metadata=tile_metadata,
    )
    return record


def build_component_group_record(
    input_records: Sequence[Mapping[str, Any]],
    mask: np.ndarray,
    *,
    tile_id: str,
    instance_id: int,
    semantic_class_id: int,
    confidence_map: np.ndarray | None = None,
    source_map: np.ndarray | None = None,
    tile_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Consolidate topology-only micro-components into one object identity.

    This changes identity only: the input masks are unioned without changing a
    semantic, confidence, source, or valid-data pixel.  The resulting mask may
    be a ``MultiPolygon`` downstream, which is preferable to promoting every
    one-pixel disconnected speckle into its own object/graph node.
    """

    if len(input_records) < 2:
        raise ValueError("Component identity consolidation requires two records")
    ordered = sorted(
        (copy.deepcopy(dict(value)) for value in input_records),
        key=lambda value: (
            int(value.get("component_index", 0)),
            int(value.get("instance_id", 0)),
            str(value.get("entity_id", "")),
        ),
    )
    group_ids = {str(value.get("entity_group_id", "")) for value in ordered}
    stage2_ids = {
        int(value.get("stage2_instance_id"))
        for value in ordered
        if value.get("stage2_instance_id") is not None
    }
    class_ids = {
        int(value.get("label", {}).get("id"))
        for value in ordered
        if isinstance(value.get("label"), Mapping)
        and value.get("label", {}).get("id") is not None
    }
    if len(group_ids) != 1 or not next(iter(group_ids)):
        raise ValueError("Components must share one non-empty entity_group_id")
    if len(stage2_ids) != 1:
        raise ValueError("Components must descend from one Stage 2 instance")
    if class_ids != {int(semantic_class_id)}:
        raise ValueError("Components must share the consolidated semantic class")

    stage2_instance_id = next(iter(stage2_ids))
    group_id = next(iter(group_ids))
    original_component_count = max(
        int(value.get("component_count", len(ordered))) for value in ordered
    )
    expected_group = make_entity_group_id(tile_id, stage2_instance_id)
    if group_id != expected_group:
        raise ValueError("Only topology-split Stage 2 component groups may consolidate")
    source_stage2_segments = sorted(
        {
            str(value)
            for item in ordered
            for value in item.get("source_stage2_segment_ids", [])
        }
    )
    histories = sorted(
        {
            str(value)
            for item in ordered
            for value in item.get("refinement_history", [])
        }
    )
    originals = _unique_stage2_records(ordered)
    record = ordered[0]
    record.update(
        {
            "instance_id": int(instance_id),
            "mask_index": int(instance_id) - 1,
            "segment_id": make_component_group_segment_id(stage2_instance_id),
            "entity_id": group_id,
            "entity_group_id": group_id,
            "stage2_instance_id": stage2_instance_id,
            "stage2_segment_id": (
                source_stage2_segments[0]
                if len(source_stage2_segments) == 1
                else None
            ),
            "parent_instance_id": stage2_instance_id,
            "parent_segment_id": (
                source_stage2_segments[0]
                if len(source_stage2_segments) == 1
                else None
            ),
            "component_index": 0,
            "component_count": original_component_count,
            "component_connectivity": 4,
            "connected_component_count": len(ordered),
            "is_four_connected": False,
            "source_stage2_instance_ids": [stage2_instance_id],
            "source_stage2_segment_ids": source_stage2_segments,
            "refinement_history": histories,
            "stage2_metadata": copy.deepcopy(originals[0]) if originals else {},
            "stage2_metadata_records": originals,
            "geometry_source": {
                "model": "stage2_refinement",
                "method": "component_identity_consolidation",
                "input_entity_ids": [str(item.get("entity_id")) for item in ordered],
                "explanation": (
                    "Topology-only micro-components were retained as exact pixels "
                    "but consolidated into their original Stage 2 object identity."
                ),
            },
            "explanation": (
                f"Consolidated {len(ordered)} disconnected topology components "
                "without changing their semantic pixels."
            ),
        }
    )
    _attach_lineage(record)
    _attach_entity_policy(record)
    refresh_segment_record(
        record,
        mask,
        semantic_class_id=semantic_class_id,
        confidence_map=confidence_map,
        source_map=source_map,
        tile_metadata=tile_metadata,
    )
    # ``refresh_segment_record`` describes raster adjacency connectivity, not
    # whether the full object mask has one component; retain the explicit fact.
    record["metrics"]["connected_component_count"] = len(ordered)
    record["metrics"]["is_four_connected"] = False
    return record


def build_combined_record(
    input_records: Sequence[Mapping[str, Any]],
    mask: np.ndarray,
    *,
    tile_id: str,
    instance_id: int,
    entity_id: str,
    segment_id: str,
    semantic_class_id: int,
    operation_id: str,
    operation_type: str,
    base_entity_id: str | None = None,
    semantic_label: Mapping[str, Any] | None = None,
    semantic_label_source: str | None = None,
    confidence_map: np.ndarray | None = None,
    source_map: np.ndarray | None = None,
    tile_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build truthful metadata for a merge or absorption result."""

    if not input_records:
        raise ValueError("A combined refinement record requires at least one input record")
    ordered = sorted(
        (copy.deepcopy(dict(value)) for value in input_records),
        key=lambda value: (int(value.get("instance_id", 0)), str(value.get("entity_id", ""))),
    )
    record = ordered[0]
    if base_entity_id is not None:
        matching = [
            item for item in ordered if str(item.get("entity_id")) == str(base_entity_id)
        ]
        if len(matching) != 1:
            raise ValueError("Combined-record base_entity_id must identify exactly one input")
        record = matching[0]
    source_stage2_ids = sorted(
        {
            int(value)
            for item in ordered
            for value in item.get("source_stage2_instance_ids", [item.get("stage2_instance_id")])
            if value is not None
        }
    )
    source_stage2_segments = sorted(
        {
            str(value)
            for item in ordered
            for value in item.get("source_stage2_segment_ids", [item.get("stage2_segment_id")])
            if value is not None
        }
    )
    originals = _unique_stage2_records(ordered)
    histories = sorted(
        {
            str(value)
            for item in ordered
            for value in item.get("refinement_history", [])
        }
    )
    if operation_id not in histories:
        histories.append(str(operation_id))
    input_class_ids = {
        int(item["label"]["id"])
        for item in ordered
        if isinstance(item.get("label"), Mapping) and item["label"].get("id") is not None
    }
    same_semantic_class = len(input_class_ids) <= 1

    record.update(
        {
            "instance_id": int(instance_id),
            "mask_index": int(instance_id) - 1,
            "segment_id": str(segment_id),
            "entity_id": str(entity_id),
            "entity_group_id": str(entity_id),
            "component_index": 0,
            "component_count": 1,
            "component_connectivity": 4,
            "source_stage2_instance_ids": source_stage2_ids,
            "source_stage2_segment_ids": source_stage2_segments,
            "stage2_instance_id": source_stage2_ids[0] if len(source_stage2_ids) == 1 else None,
            "stage2_segment_id": source_stage2_segments[0] if len(source_stage2_segments) == 1 else None,
            "parent_instance_id": None,
            "parent_segment_id": None,
            "refinement_history": histories,
            "stage2_metadata": copy.deepcopy(originals[0]) if originals else {},
            "stage2_metadata_records": originals,
            "geometry_source": {
                "model": "stage2_refinement",
                "method": str(operation_type),
                "operation_id": str(operation_id),
                "input_entity_ids": [str(item.get("entity_id")) for item in ordered],
                "explanation": (
                    f"This geometry is the exact raster union produced by the approved "
                    f"Stage 2 refinement operation '{operation_type}'."
                ),
            },
            "explanation": (
                f"Approved Stage 2 refinement operation '{operation_type}' combined "
                f"{len(ordered)} connected entities"
                + (
                    "."
                    if same_semantic_class
                    else " and assigned the reviewed target entity's semantic class."
                )
            ),
        }
    )
    if semantic_label is not None:
        target_label = copy.deepcopy(dict(semantic_label))
        target_label["id"] = int(semantic_class_id)
        record["label"] = target_label
        if target_label.get("name") is not None:
            record["semantic_class"] = str(target_label["name"])
            record["class_name"] = str(target_label["name"])
    if semantic_label_source is not None:
        record["label_source"] = str(semantic_label_source)
    _attach_lineage(record)
    _attach_entity_policy(record)
    _attach_refinement_provenance(record, operation_id, operation_type, ordered)
    refresh_segment_record(
        record,
        mask,
        semantic_class_id=semantic_class_id,
        confidence_map=confidence_map,
        source_map=source_map,
        tile_metadata=tile_metadata,
    )
    return record


def build_split_record(
    parent_record: Mapping[str, Any],
    mask: np.ndarray,
    *,
    tile_id: str,
    instance_id: int,
    entity_id: str,
    segment_id: str,
    semantic_class_id: int,
    operation_id: str,
    proposal_id: str,
    child_index: int,
    child_count: int,
    confidence_map: np.ndarray | None = None,
    source_map: np.ndarray | None = None,
    tile_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one child record for an explicitly reviewed semantic partition."""

    index = int(child_index)
    count = int(child_count)
    if count < 2 or index < 0 or index >= count:
        raise ValueError("semantic split child index/count are inconsistent")
    parent = copy.deepcopy(dict(parent_record))
    parent_entity_id = str(parent.get("entity_id", ""))
    parent_segment_id = str(parent.get("segment_id", ""))
    if not parent_entity_id or not parent_segment_id:
        raise ValueError("semantic split parent must have entity_id and segment_id")
    histories = [str(value) for value in parent.get("refinement_history", [])]
    if str(operation_id) not in histories:
        histories.append(str(operation_id))

    record = parent
    record.update(
        {
            "instance_id": int(instance_id),
            "mask_index": int(instance_id) - 1,
            "entity_id": str(entity_id),
            "segment_id": str(segment_id),
            "entity_group_id": make_split_group_id(tile_id, proposal_id),
            "component_index": 0,
            "component_count": 1,
            "component_connectivity": 4,
            "refinement_parent_entity_id": parent_entity_id,
            "refinement_parent_segment_id": parent_segment_id,
            "semantic_split_proposal_id": str(proposal_id),
            "semantic_split_child_index": index,
            "semantic_split_child_label": index + 1,
            "semantic_split_child_count": count,
            "refinement_history": histories,
            "geometry_source": {
                "model": "stage2_refinement",
                "method": "semantic_split",
                "operation_id": str(operation_id),
                "proposal_id": str(proposal_id),
                "input_entity_ids": [parent_entity_id],
                "explanation": (
                    "This geometry is one exact child of an explicitly reviewed "
                    "semantic partition of its parent entity."
                ),
            },
            "explanation": (
                f"Approved semantic split child {index + 1} of {count} from "
                f"parent entity '{parent_entity_id}'."
            ),
        }
    )
    _attach_lineage(record)
    _attach_entity_policy(record)
    _attach_refinement_provenance(record, operation_id, "semantic_split", [parent_record])
    refresh_segment_record(
        record,
        mask,
        semantic_class_id=semantic_class_id,
        confidence_map=confidence_map,
        source_map=source_map,
        tile_metadata=tile_metadata,
    )
    return record


def refresh_segment_record(
    record: dict[str, Any],
    mask: np.ndarray,
    *,
    semantic_class_id: int,
    confidence_map: np.ndarray | None = None,
    source_map: np.ndarray | None = None,
    tile_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Refresh every mask-dependent field of a record in place."""

    component = np.asarray(mask, dtype=bool)
    if component.ndim != 2 or not component.any():
        raise ValueError("A refined segment mask must be a non-empty two-dimensional array")
    rows, columns = np.nonzero(component)
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    width, height = x1 - x0, y1 - y0
    area = int(component.sum())
    centroid_row, centroid_column = float(rows.mean()), float(columns.mean())
    touches_boundary = bool(
        x0 == 0 or y0 == 0 or x1 == component.shape[1] or y1 == component.shape[0]
    )
    record.update(
        {
            "semantic_class_id": int(semantic_class_id),
            "class_id": int(semantic_class_id),
            "area_pixels": area,
            "bbox_xywh": [x0, y0, width, height],
            "bbox_xyxy": [x0, y0, x1, y1],
            "centroid_rc": [centroid_row, centroid_column],
            "centroid_xy": [centroid_column + 0.5, centroid_row + 0.5],
            "touches_tile_boundary": touches_boundary,
            "valid_pixel_fraction": 1.0,
            "accepted": True,
            "rejection_reasons": [],
        }
    )
    label = copy.deepcopy(record.get("label")) if isinstance(record.get("label"), Mapping) else {}
    label["id"] = int(semantic_class_id)
    if record.get("semantic_class") is not None:
        label.setdefault("name", str(record["semantic_class"]))
    elif record.get("class_name") is not None:
        label.setdefault("name", str(record["class_name"]))
    if not label.get("name"):
        label["name"] = f"class_{semantic_class_id}"
    record["label"] = label
    record["semantic_class"] = str(label["name"])
    record["class_name"] = str(label["name"])

    metrics = copy.deepcopy(record.get("metrics")) if isinstance(record.get("metrics"), Mapping) else {}
    metrics.update(
        {
            "area_pixels": area,
            "area_ratio": float(area / component.size),
            "valid_pixel_fraction": 1.0,
            "nodata_fraction": 0.0,
            "fill_ratio": float(area / max(1, width * height)),
            "touches_tile_boundary": touches_boundary,
            "connectivity": 4,
        }
    )
    record["metrics"] = metrics

    if confidence_map is not None:
        confidence = np.asarray(confidence_map)
        if confidence.shape != component.shape:
            raise ValueError("confidence_map shape does not match the refined segment mask")
        selected = confidence[component].astype(np.float64, copy=False)
        if not np.isfinite(selected).all():
            raise ValueError("confidence_map contains non-finite values inside a refined segment")
        record["mean_semantic_confidence"] = float(selected.mean())
        label["mean_final_label_confidence"] = float(selected.mean())
        label["refinement_confidence"] = {
            "minimum": float(selected.min()),
            "mean": float(selected.mean()),
            "maximum": float(selected.max()),
            "pixel_count": area,
        }

    if source_map is not None:
        sources = np.asarray(source_map)
        if sources.shape != component.shape:
            raise ValueError("source_map shape does not match the refined segment mask")
        codes, counts = np.unique(sources[component], return_counts=True)
        record["pixel_source_codes"] = [
            {"code": int(code), "pixel_count": int(count)}
            for code, count in zip(codes.tolist(), counts.tolist())
        ]
        if len(codes) == 1:
            record["source_code"] = int(codes[0])
        else:
            record["source"] = "mixed_stage2_sources"
            record["source_code"] = None

    # Copied Stage 2 coordinate records describe the parent mask. Never leave
    # those stale when the caller cannot supply complete tile metadata.
    record.pop("coordinates", None)
    if tile_metadata is not None:
        try:
            # The record is owned by this refinement result, so the in-place
            # path avoids deep-copying its complete immutable Stage 2 lineage
            # once more for every connected component.
            refreshed = attach_segment_coordinates(record, tile_metadata, in_place=True)
        except (TypeError, ValueError) as exc:
            record["coordinate_refresh"] = {
                "available": False,
                "reason": str(exc),
            }
        else:
            record.update(refreshed)
            record["coordinate_refresh"] = {"available": True, "reason": None}


def source_stage2_metadata(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return deep, JSON-safe Stage 2 originals carried by a refined record."""

    records = record.get("stage2_metadata_records")
    if isinstance(records, Sequence) and not isinstance(records, (str, bytes, bytearray)):
        values = [json_safe(copy.deepcopy(dict(item))) for item in records if isinstance(item, Mapping)]
        if values:
            return values
    original = record.get("stage2_metadata")
    if isinstance(original, Mapping):
        return [json_safe(copy.deepcopy(dict(original)))]
    return []


def _unique_stage2_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_identity: dict[tuple[Any, Any], dict[str, Any]] = {}
    for record in records:
        for original in source_stage2_metadata(record):
            key = (original.get("instance_id"), original.get("segment_id"))
            by_identity.setdefault(key, original)
    def order(value: tuple[Any, Any]) -> tuple[int, str, str]:
        try:
            numeric = int(value[0])
        except (TypeError, ValueError):
            return (1, str(value[0]), str(value[1]))
        return (0, f"{numeric:020d}", str(value[1]))

    return [by_identity[key] for key in sorted(by_identity, key=order)]


def _attach_lineage(record: dict[str, Any]) -> None:
    record["lineage"] = {
        "source_stage": "stage2.12_final",
        "source_stage2_instance_ids": list(record.get("source_stage2_instance_ids", [])),
        "source_stage2_segment_ids": list(record.get("source_stage2_segment_ids", [])),
        "entity_group_id": record.get("entity_group_id"),
        "refinement_operation_ids": list(record.get("refinement_history", [])),
    }


def _attach_entity_policy(record: dict[str, Any]) -> None:
    label = record.get("label") if isinstance(record.get("label"), Mapping) else {}
    class_name = str(label.get("name", record.get("semantic_class", "unknown")))
    policy = ENTITY_POLICY_BY_CLASS.get(class_name, "retain_until_supported")
    entity_type = ENTITY_TYPE_BY_CLASS.get(class_name, "unresolved_region")
    record["entity_policy"] = policy
    record["entity_type"] = entity_type
    if class_name == "road":
        # Dedicated road topology comes later; this prevents falsely claiming
        # that connectivity normalization alone found real-world junctions.
        record["segment_subtype"] = "road_reach_candidate"


def _attach_refinement_provenance(
    record: dict[str, Any],
    operation_id: str,
    operation_type: str,
    inputs: Sequence[Mapping[str, Any]],
) -> None:
    provenance = copy.deepcopy(record.get("provenance")) if isinstance(record.get("provenance"), Mapping) else {}
    history = provenance.get("refinement")
    if not isinstance(history, list):
        history = []
    history.append(
        {
            "operation_id": str(operation_id),
            "operation_type": str(operation_type),
            "input_entity_ids": [str(value.get("entity_id")) for value in inputs],
        }
    )
    provenance["refinement"] = history
    record["provenance"] = provenance


__all__ = [
    "ENTITY_POLICY_BY_CLASS",
    "ENTITY_TYPE_BY_CLASS",
    "build_combined_record",
    "build_component_record",
    "build_component_group_record",
    "build_split_record",
    "make_component_entity_id",
    "make_component_group_segment_id",
    "make_component_segment_id",
    "make_entity_group_id",
    "make_merge_entity_id",
    "make_merge_segment_id",
    "make_split_entity_id",
    "make_split_group_id",
    "make_split_segment_id",
    "refresh_segment_record",
    "source_stage2_metadata",
]
