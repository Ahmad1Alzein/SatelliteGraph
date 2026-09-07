"""Pure topology normalization and reviewed-operation replay.

This module owns raster truth for Stage 2 refinement. Downstream polygons and
graph edges must be rebuilt from the returned ``instance_map``; they are never
patched directly.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np

from src.stage2.classes import NODATA_CLASS_ID, SOURCE_CODES

from .automatic import evaluate_automatic_proposal
from .evidence import pair_evidence
from .policy import classify_hard_vetoes, orient_absorb_evidence
from .records import (
    build_combined_record,
    build_component_record,
    build_component_group_record,
    build_split_record,
    make_component_entity_id,
    make_component_segment_id,
    make_entity_group_id,
    make_merge_entity_id,
    make_merge_segment_id,
    make_split_entity_id,
    make_split_segment_id,
    refresh_segment_record,
)
from .schema import (
    OPERATION_SCHEMA_VERSION,
    PROPOSAL_SCHEMA_VERSION,
    SCHEMA_VERSION,
    canonical_json,
    make_operation,
    stable_digest,
    validate_decision,
    validate_operation,
    validate_proposal,
)
from .validation import is_four_connected, validate_refined_segmentation
from .splits import validate_semantic_split_partition


@dataclass(slots=True)
class RefinementResult:
    """One in-memory, gap-free refined segmentation product."""

    masks: list[np.ndarray]
    records: list[dict[str, Any]]
    instance_map: np.ndarray
    semantic_map: np.ndarray
    confidence_map: np.ndarray | None
    source_map: np.ndarray | None
    valid_mask: np.ndarray
    operations: list[dict[str, Any]]
    lineage: dict[str, Any]
    report: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "segment_count": len(self.records),
            "operation_count": len(self.operations),
            **copy.deepcopy(self.report),
        }


# The more descriptive alias is retained for callers that want to distinguish
# the first deterministic pass from later reviewed refinement results.
NormalizationResult = RefinementResult


def connected_components_4(mask: np.ndarray) -> list[np.ndarray]:
    """Return deterministic 4-connected components of a 2D boolean mask.

    Components are sorted by top row, left column, then decreasing area. The
    first two keys uniquely determine order for disjoint components in normal
    raster data; the remaining keys provide an explicit stable tie-break.
    """

    value = np.asarray(mask, dtype=bool)
    if value.ndim != 2:
        raise ValueError("connected_components_4 requires a two-dimensional mask")
    if not value.any():
        return []
    rows, columns = np.nonzero(value)
    y0, y1 = int(rows.min()), int(rows.max()) + 1
    x0, x1 = int(columns.min()), int(columns.max()) + 1
    cropped = value[y0:y1, x0:x1]
    number, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.ascontiguousarray(cropped, dtype=np.uint8),
        connectivity=4,
        ltype=cv2.CV_32S,
    )
    components: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
    for component_id in range(1, int(number)):
        left = x0 + int(stats[component_id, cv2.CC_STAT_LEFT])
        top = y0 + int(stats[component_id, cv2.CC_STAT_TOP])
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        component = np.zeros(value.shape, dtype=bool)
        component[y0:y1, x0:x1] = labels == component_id
        components.append(
            ((top, left, -area, component_id), component)
        )
    components.sort(key=lambda item: item[0])
    return [component for _, component in components]


def normalize_4_connected_components(
    instance_map: np.ndarray,
    semantic_map: np.ndarray,
    confidence_map: np.ndarray | None,
    source_map: np.ndarray | None,
    valid_mask: np.ndarray,
    records: Sequence[Mapping[str, Any]],
    *,
    tile_id: str,
    tile_metadata: Mapping[str, Any] | None = None,
) -> RefinementResult:
    """Split every positive Stage 2 ID into stable 4-connected entities.

    Raster IDs are rebuilt contiguously. The semantic, confidence, source, and
    validity maps are copied byte-for-byte; only the instance IDs and their
    mask-dependent records change.
    """

    tile = str(tile_id)
    if not tile:
        raise ValueError("tile_id must be non-empty")
    instances, semantics, confidence, sources, valid = _validate_stage2_input(
        instance_map,
        semantic_map,
        confidence_map,
        source_map,
        valid_mask,
        records,
    )
    original_records = _records_by_instance(records)
    refined_map = np.zeros(instances.shape, dtype=np.int32)
    refined_masks: list[np.ndarray] = []
    refined_records: list[dict[str, Any]] = []
    operations: list[dict[str, Any]] = []
    component_counts: dict[int, int] = {}
    next_instance_id = 1

    for stage2_instance_id in sorted(original_records):
        parent_mask = instances == stage2_instance_id
        components = connected_components_4(parent_mask)
        if not components:
            raise ValueError(f"Stage 2 instance {stage2_instance_id} has no pixels")
        component_count = len(components)
        component_counts[stage2_instance_id] = component_count
        output_instance_ids = list(
            range(next_instance_id, next_instance_id + component_count)
        )
        output_entity_ids = [
            make_component_entity_id(tile, stage2_instance_id, index)
            for index in range(component_count)
        ]
        operation: dict[str, Any] | None = None
        if component_count > 1:
            operation = make_operation(
                tile,
                "split_components",
                [make_entity_group_id(tile, stage2_instance_id)],
                output_entity_ids,
                source_instance_ids=[stage2_instance_id],
                output_instance_ids=output_instance_ids,
                parameters={
                    "connectivity": 4,
                    "component_order": "top_row,left_column,negative_area,component_label",
                    "automatic": True,
                },
                evidence={
                    "input_component_count_4": component_count,
                    "parent_area_pixels": int(parent_mask.sum()),
                    "output_area_pixels": [int(component.sum()) for component in components],
                },
                decision_source="deterministic_topology_rule",
                invariants={
                    "children_disjoint": True,
                    "children_union_equals_parent": bool(
                        np.array_equal(np.logical_or.reduce(components), parent_mask)
                    ),
                    "semantic_map_unchanged": True,
                    "valid_mask_unchanged": True,
                },
            )
            operations.append(operation)

        class_values = np.unique(semantics[parent_mask])
        if len(class_values) != 1:
            raise ValueError(
                f"Stage 2 instance {stage2_instance_id} contains multiple semantic classes"
            )
        class_id = int(class_values[0])
        original_record = original_records[stage2_instance_id]
        for component_index, (new_instance_id, component) in enumerate(
            zip(output_instance_ids, components)
        ):
            refined_map[component] = new_instance_id
            refined_masks.append(component.copy())
            refined_records.append(
                build_component_record(
                    original_record,
                    component,
                    tile_id=tile,
                    instance_id=new_instance_id,
                    stage2_instance_id=stage2_instance_id,
                    component_index=component_index,
                    component_count=component_count,
                    semantic_class_id=class_id,
                    confidence_map=confidence,
                    source_map=sources,
                    tile_metadata=tile_metadata,
                    operation_id=operation["operation_id"] if operation else None,
                )
            )
        next_instance_id += component_count

    validation = validate_refined_segmentation(
        refined_map,
        semantics,
        valid,
        refined_records,
        refined_masks,
        confidence_map=confidence,
        source_map=sources,
        reference_semantic_map=semantics,
        reference_confidence_map=confidence,
        reference_source_map=sources,
        reference_valid_mask=valid,
    )
    split_parent_count = sum(count > 1 for count in component_counts.values())
    report = {
        "status": "complete",
        "phase": "connectivity_normalization",
        "tile_id": tile,
        "connectivity": 4,
        "input_segment_count": len(original_records),
        "output_segment_count": len(refined_records),
        "split_parent_count": int(split_parent_count),
        "additional_component_count": int(len(refined_records) - len(original_records)),
        "valid_pixel_count": int(valid.sum()),
        "every_valid_pixel_exactly_once": True,
        "semantic_map_unchanged": True,
        "confidence_map_unchanged": confidence is not None,
        "source_map_unchanged": sources is not None,
        "valid_mask_unchanged": True,
        "validation": validation.to_dict(),
    }
    lineage = _lineage_document(tile, refined_records)
    return RefinementResult(
        masks=refined_masks,
        records=refined_records,
        instance_map=refined_map,
        semantic_map=semantics,
        confidence_map=confidence,
        source_map=sources,
        valid_mask=valid,
        operations=operations,
        lineage=lineage,
        report=report,
    )


def consolidate_micro_components(
    result: RefinementResult,
    *,
    minimum_object_component_area_pixels: int,
    source_stage2_segment_count: int,
    tile_metadata: Mapping[str, Any] | None = None,
) -> RefinementResult:
    """Prevent topology-only micro-components from becoming object nodes.

    Refinement still operates on connected component entities.  Immediately
    before publication, secondary components below the configured object-area
    floor are consolidated with the largest surviving sibling from the same
    immutable Stage 2 object.  No raster pixel or semantic value changes.
    Larger disconnected components may remain separate because they can be
    genuine objects, but topology-only promotion is capped so it cannot make
    the published count exceed Stage 2 plus explicitly approved semantic-split
    growth.
    """

    _validate_result(result)
    if (
        isinstance(minimum_object_component_area_pixels, bool)
        or not isinstance(minimum_object_component_area_pixels, int)
        or minimum_object_component_area_pixels < 1
    ):
        raise ValueError("minimum_object_component_area_pixels must be positive")
    if (
        isinstance(source_stage2_segment_count, bool)
        or not isinstance(source_stage2_segment_count, int)
        or source_stage2_segment_count < 1
    ):
        raise ValueError("source_stage2_segment_count must be positive")

    tile_id = _tile_id(result)
    grouped: dict[str, list[int]] = {}
    standalone: list[int] = []
    for index, record in enumerate(result.records):
        stage2_id = record.get("stage2_instance_id")
        group_id = str(record.get("entity_group_id", ""))
        expected_group = (
            make_entity_group_id(tile_id, int(stage2_id))
            if stage2_id is not None
            else None
        )
        if expected_group is not None and group_id == expected_group:
            grouped.setdefault(group_id, []).append(index)
        else:
            standalone.append(index)

    entries: list[tuple[int, str, list[int]]] = []
    consolidated_fragment_count = 0
    promoted_large_component_count = 0
    promoted_entries: list[tuple[str, int]] = []
    for group_id, indices in sorted(grouped.items()):
        ordered = sorted(
            indices,
            key=lambda index: (
                -int(result.records[index].get("area_pixels", 0)),
                int(result.records[index].get("instance_id", index + 1)),
            ),
        )
        anchor = ordered[0]
        micro = [
            index
            for index in ordered[1:]
            if int(result.records[index].get("area_pixels", 0))
            < minimum_object_component_area_pixels
        ]
        micro_indices = set(micro)
        large = [index for index in ordered[1:] if index not in micro_indices]
        if micro:
            bundle = [anchor, *micro]
            entries.append(
                (
                    min(int(result.records[index]["instance_id"]) for index in bundle),
                    group_id,
                    bundle,
                )
            )
            consolidated_fragment_count += len(micro)
        else:
            entries.append(
                (
                    int(result.records[anchor]["instance_id"]),
                    group_id,
                    [anchor],
                )
            )
        for index in large:
            promoted_large_component_count += 1
            promoted_entries.append((group_id, index))
            entries.append(
                (
                    int(result.records[index]["instance_id"]),
                    str(result.records[index]["entity_id"]),
                    [index],
                )
            )
    for index in standalone:
        entries.append(
            (
                int(result.records[index]["instance_id"]),
                str(result.records[index]["entity_id"]),
                [index],
            )
        )

    semantic_split_growth = sum(
        max(0, len(operation.get("output_entity_ids", [])) - 1)
        for operation in result.operations
        if operation.get("operation_type") == "semantic_split"
        and operation.get("status") == "applied"
    )
    maximum_topology_safe_count = source_stage2_segment_count + semantic_split_growth
    budget_consolidated_count = 0
    if len(entries) > maximum_topology_safe_count:
        promotion_order = sorted(
            promoted_entries,
            key=lambda value: (
                int(result.records[value[1]].get("area_pixels", 0)),
                int(result.records[value[1]].get("instance_id", value[1] + 1)),
                value[0],
            ),
        )
        for group_id, promoted_index in promotion_order:
            if len(entries) <= maximum_topology_safe_count:
                break
            anchor_position = next(
                position
                for position, entry in enumerate(entries)
                if entry[1] == group_id and promoted_index not in entry[2]
            )
            promoted_position = next(
                position
                for position, entry in enumerate(entries)
                if entry[2] == [promoted_index]
            )
            anchor_entry = entries[anchor_position]
            entries[anchor_position] = (
                min(anchor_entry[0], int(result.records[promoted_index]["instance_id"])),
                group_id,
                [*anchor_entry[2], promoted_index],
            )
            entries.pop(promoted_position)
            promoted_large_component_count -= 1
            consolidated_fragment_count += 1
            budget_consolidated_count += 1
    entries.sort(key=lambda value: (value[0], value[1]))

    rebuilt_map = np.zeros(result.instance_map.shape, dtype=np.int32)
    rebuilt_masks: list[np.ndarray] = []
    rebuilt_records: list[dict[str, Any]] = []
    multipart_object_count = 0
    for instance_id, (_, _, indices) in enumerate(entries, 1):
        masks = [np.asarray(result.masks[index], dtype=bool) for index in indices]
        mask = np.logical_or.reduce(masks)
        if np.any(rebuilt_map[mask] != 0):
            raise ValueError("Component identity consolidation produced overlapping objects")
        rebuilt_map[mask] = instance_id
        if len(indices) > 1:
            class_values = np.unique(result.semantic_map[mask])
            if len(class_values) != 1:
                raise ValueError("Consolidated components do not share one semantic class")
            record = build_component_group_record(
                [result.records[index] for index in indices],
                mask,
                tile_id=tile_id,
                instance_id=instance_id,
                semantic_class_id=int(class_values[0]),
                confidence_map=result.confidence_map,
                source_map=result.source_map,
                tile_metadata=tile_metadata,
            )
            multipart_object_count += 1
        else:
            record = copy.deepcopy(result.records[indices[0]])
            record["instance_id"] = instance_id
            record["mask_index"] = instance_id - 1
            refresh_segment_record(
                record,
                mask,
                semantic_class_id=_record_class_id(record),
                confidence_map=result.confidence_map,
                source_map=result.source_map,
                tile_metadata=tile_metadata,
            )
            record["connected_component_count"] = 1
            record["is_four_connected"] = True
            record["metrics"]["connected_component_count"] = 1
            record["metrics"]["is_four_connected"] = True
        rebuilt_masks.append(mask.copy())
        rebuilt_records.append(record)

    require_connected = multipart_object_count == 0
    validation = validate_refined_segmentation(
        rebuilt_map,
        result.semantic_map,
        result.valid_mask,
        rebuilt_records,
        rebuilt_masks,
        confidence_map=result.confidence_map,
        source_map=result.source_map,
        reference_semantic_map=result.semantic_map,
        reference_confidence_map=result.confidence_map,
        reference_source_map=result.source_map,
        reference_valid_mask=result.valid_mask,
        require_four_connected=require_connected,
    )
    report = copy.deepcopy(result.report)
    report.update(
        {
            "phase": "object_identity_consolidation",
            "preconsolidation_segment_count": len(result.records),
            "output_segment_count": len(rebuilt_records),
            "validation": validation.to_dict(),
            "object_identity": {
                "strategy": "consolidate_topology_micro_components_v1",
                "source_stage2_segment_count": source_stage2_segment_count,
                "internal_connected_component_count": len(result.records),
                "output_object_count": len(rebuilt_records),
                "minimum_promoted_component_area_pixels": (
                    minimum_object_component_area_pixels
                ),
                "consolidated_micro_component_count": (
                    consolidated_fragment_count
                ),
                "promoted_large_secondary_component_count": (
                    promoted_large_component_count
                ),
                "budget_consolidated_component_count": budget_consolidated_count,
                "semantic_split_growth_allowance": semantic_split_growth,
                "maximum_topology_safe_output_count": maximum_topology_safe_count,
                "topology_alone_cannot_increase_object_count": True,
                "multipart_object_count": multipart_object_count,
                "all_pixels_preserved": True,
                "semantic_map_unchanged": True,
                "require_every_object_four_connected": require_connected,
            },
        }
    )
    return RefinementResult(
        masks=rebuilt_masks,
        records=rebuilt_records,
        instance_map=rebuilt_map,
        semantic_map=result.semantic_map.copy(),
        confidence_map=(
            result.confidence_map.copy()
            if result.confidence_map is not None
            else None
        ),
        source_map=result.source_map.copy() if result.source_map is not None else None,
        valid_mask=result.valid_mask.copy(),
        operations=copy.deepcopy(result.operations),
        lineage=_lineage_document(tile_id, rebuilt_records),
        report=report,
    )


def apply_reviewed_operations(
    normalized_result: RefinementResult,
    operations_or_proposals: Sequence[Mapping[str, Any]],
    decisions: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    *,
    tile_metadata: Mapping[str, Any] | None = None,
    semantic_probabilities: np.ndarray | None = None,
    rgb_image: np.ndarray | None = None,
    split_partitions: Mapping[str, np.ndarray] | None = None,
    policy_config: Mapping[str, Any] | None = None,
) -> RefinementResult:
    """Apply approved, non-conflicting merge, absorb, and semantic-split decisions.

    Rejected, uncertain, and pending proposals are retained only in the report.
    Every accepted operation is evaluated against the same normalized input;
    an entity may participate in at most one operation per replay batch.

    Cross-class absorption is intentionally exceptional: it requires an
    explicit ``approved`` decision (never ``auto_apply``), class-first semantic
    probability rasters, and a known target class represented by a probability
    channel. Only source pixels are relabeled; source provenance is preserved.

    A connected semantic split is exceptional in a different way: the exact
    child-label raster must be supplied in ``split_partitions`` and checksum-
    bound to the proposal. It is applied only from a separate explicit
    ``approved`` decision and can never be auto-applied.
    """

    _validate_result(normalized_result)
    probabilities = _validate_semantic_probabilities(
        semantic_probabilities,
        normalized_result.instance_map.shape,
    )
    partition_index = _split_partition_index(split_partitions)
    working_semantics = normalized_result.semantic_map.copy()
    working_confidence = (
        normalized_result.confidence_map.copy()
        if normalized_result.confidence_map is not None
        else None
    )
    semantic_map_changed = False
    confidence_map_changed = False
    cross_class_absorbed_pixel_count = 0
    semantic_split_parent_count = 0
    semantic_split_child_count = 0
    candidates = [copy.deepcopy(dict(item)) for item in operations_or_proposals]
    decision_index = _decision_index(decisions)
    candidate_id_values = [_candidate_id(item) for item in candidates]
    if len(set(candidate_id_values)) != len(candidate_id_values):
        raise ValueError("Refinement replay contains duplicate proposal/operation IDs")
    candidate_ids = set(candidate_id_values)
    unknown_decisions = sorted(set(decision_index) - candidate_ids)
    if unknown_decisions:
        raise ValueError(f"Decisions reference unknown proposals/operations: {unknown_decisions}")
    known_partition_keys = candidate_ids | {
        str(item["proposal_id"])
        for item in candidates
        if item.get("proposal_id") is not None
    }
    unknown_partitions = sorted(set(partition_index) - known_partition_keys)
    if unknown_partitions:
        raise ValueError(
            f"Split partitions reference unknown proposals/operations: {unknown_partitions}"
        )

    approved: list[tuple[dict[str, Any], str]] = []
    decision_counts = {
        "approved": 0,
        "auto_apply": 0,
        "rejected": 0,
        "uncertain": 0,
        "pending": 0,
    }
    for candidate in candidates:
        _validate_candidate(candidate)
        decision, source = _resolve_decision(candidate, decision_index)
        if str(candidate["operation_type"]) == "semantic_split":
            if decision == "auto_apply":
                raise ValueError("semantic_split can never be auto-applied")
            if decision == "approved" and str(source).startswith("automatic_"):
                raise ValueError(
                    "semantic_split requires human review and cannot use an "
                    "automatic decision source"
                )
            if (
                decision == "approved"
                and candidate.get("schema_version") == PROPOSAL_SCHEMA_VERSION
                and _candidate_id(candidate) not in decision_index
            ):
                raise ValueError(
                    "semantic_split requires a separate explicit approved review decision"
                )
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        if decision in {"approved", "auto_apply"}:
            approved.append((candidate, source))
    approved.sort(key=lambda item: _candidate_id(item[0]))

    states: dict[str, _EntityState] = {}
    for record, mask in zip(normalized_result.records, normalized_result.masks):
        entity_id = str(record["entity_id"])
        states[entity_id] = _EntityState(
            entity_id=entity_id,
            segment_id=str(record["segment_id"]),
            mask=np.asarray(mask, dtype=bool).copy(),
            record=copy.deepcopy(record),
            order_key=int(record["instance_id"]),
        )
    initial_entity_ids = set(states)
    initial_instance_by_entity = {
        entity_id: int(state.order_key) for entity_id, state in states.items()
    }
    initial_mask_by_entity = {
        entity_id: np.asarray(state.mask, dtype=bool).copy()
        for entity_id, state in states.items()
    }
    automatic_pair_rows: list[dict[str, Any]] | None = None
    if any(str(source).startswith("automatic_") for _, source in approved):
        if rgb_image is None:
            raise ValueError(
                "Automatic generic refinement requires the source RGB image for "
                "replay-time evidence validation"
            )
        if policy_config is None:
            raise ValueError(
                "Automatic generic refinement requires a validated policy configuration"
            )
        automatic_pair_rows = pair_evidence(
            normalized_result.instance_map,
            np.asarray(rgb_image),
        )

    participant_roles: dict[str, list[tuple[str, str]]] = {}
    for candidate, _ in approved:
        candidate_id = _candidate_id(candidate)
        operation_type = str(candidate["operation_type"])
        source_ids = [str(value) for value in candidate["source_entity_ids"]]
        target_id = (
            str(candidate["target_entity_id"])
            if candidate.get("target_entity_id") is not None
            else None
        )
        roles = [(entity_id, "exclusive_source") for entity_id in source_ids]
        if target_id is not None:
            roles.append(
                (
                    target_id,
                    "reusable_absorb_target"
                    if operation_type == "absorb"
                    else "exclusive_target",
                )
            )
        for entity_id, role in roles:
            prior = participant_roles.get(entity_id, [])
            reusable = role == "reusable_absorb_target" and all(
                prior_role == "reusable_absorb_target"
                for prior_role, _ in prior
            )
            if prior and not reusable:
                conflicts = [prior_id for _, prior_id in prior]
                raise ValueError(
                    f"Conflicting refinement proposals touch entity {entity_id!r}: "
                    f"{conflicts!r} and {candidate_id!r}"
                )
            participant_roles.setdefault(entity_id, []).append((role, candidate_id))

    pending_operations: list[_PendingOperation] = []
    for candidate, decision_source in approved:
        operation_type = str(candidate["operation_type"])
        if operation_type not in {"merge", "absorb", "semantic_split"}:
            raise ValueError(
                f"Reviewed replay does not support {operation_type!r}. "
                "split_components is applied by connectivity normalization and mark_nodata may only target invalid pixels."
            )
        sources_ids = [str(value) for value in candidate["source_entity_ids"]]
        target_id = (
            str(candidate["target_entity_id"])
            if candidate.get("target_entity_id") is not None
            else None
        )
        participating = [*sources_ids, *([target_id] if target_id is not None else [])]
        for entity_id in participating:
            if entity_id not in initial_entity_ids:
                raise ValueError(f"Refinement candidate references missing entity {entity_id!r}")
            if entity_id not in states:
                raise ValueError(
                    f"Refinement candidate { _candidate_id(candidate)!r} depends on an entity "
                    "already replaced in this replay batch"
                )

        if operation_type == "semantic_split":
            source_id = sources_ids[0]
            source_state = states[source_id]
            candidate_id = _candidate_id(candidate)
            partition = partition_index.get(candidate_id)
            proposal_id_value = candidate.get("proposal_id")
            if partition is None and proposal_id_value is not None:
                partition = partition_index.get(str(proposal_id_value))
            if partition is None:
                raise ValueError(
                    f"Approved semantic_split {candidate_id!r} has no exact partition map"
                )
            parameters = copy.deepcopy(dict(candidate.get("parameters", {})))
            _, child_masks = validate_semantic_split_partition(
                partition,
                source_state.mask,
                expected_sha256=str(parameters["partition_sha256"]),
                expected_child_count=int(parameters["child_count"]),
                minimum_child_area_pixels=int(parameters["minimum_child_area_pixels"]),
            )
            semantic_split_parent_count += 1
            semantic_split_child_count += len(child_masks)
            proposal_id = (
                str(proposal_id_value)
                if proposal_id_value is not None
                else candidate_id
            )
            if candidate.get("schema_version") == OPERATION_SCHEMA_VERSION:
                output_entity_ids = [str(value) for value in candidate["output_entity_ids"]]
                if len(output_entity_ids) != len(child_masks):
                    raise ValueError(
                        "semantic_split operation outputs disagree with its partition child count"
                    )
            else:
                output_entity_ids = [
                    make_split_entity_id(_tile_id(normalized_result), proposal_id, index)
                    for index in range(len(child_masks))
                ]
            if len(set(output_entity_ids)) != len(output_entity_ids):
                raise ValueError("semantic_split output entity IDs must be unique")
            for output_entity_id in output_entity_ids:
                if output_entity_id in states and output_entity_id != source_id:
                    raise ValueError(
                        f"semantic_split output entity {output_entity_id!r} collides with an existing entity"
                    )
            evidence = copy.deepcopy(dict(candidate.get("evidence", {})))
            evidence["verified_partition"] = {
                "parent_area_pixels": int(source_state.mask.sum()),
                "child_area_pixels": [int(mask.sum()) for mask in child_masks],
                "child_count": len(child_masks),
                "partition_sha256": str(parameters["partition_sha256"]),
            }
            provisional = make_operation(
                _tile_id(normalized_result),
                "semantic_split",
                [source_id],
                output_entity_ids,
                source_instance_ids=[source_state.order_key],
                output_instance_ids=[],
                proposal_id=(
                    str(candidate["proposal_id"])
                    if candidate.get("proposal_id") is not None
                    else None
                ),
                parameters=parameters,
                evidence=evidence,
                decision_source=decision_source,
                invariants={
                    "children_disjoint": True,
                    "children_union_equals_parent": True,
                    "children_are_4_connected": True,
                    "valid_pixel_coverage_unchanged": True,
                    "semantic_map_unchanged": True,
                    "confidence_map_unchanged": True,
                    "source_map_unchanged": True,
                    "valid_mask_unchanged": True,
                },
            )
            class_id = _record_class_id(source_state.record)
            del states[source_id]
            for child_index, (output_entity_id, child_mask) in enumerate(
                zip(output_entity_ids, child_masks)
            ):
                output_segment_id = make_split_segment_id(proposal_id, child_index)
                child_record = build_split_record(
                    source_state.record,
                    child_mask,
                    tile_id=_tile_id(normalized_result),
                    instance_id=1,
                    entity_id=output_entity_id,
                    segment_id=output_segment_id,
                    semantic_class_id=class_id,
                    operation_id=str(provisional["operation_id"]),
                    proposal_id=proposal_id,
                    child_index=child_index,
                    child_count=len(child_masks),
                    confidence_map=working_confidence,
                    source_map=normalized_result.source_map,
                    tile_metadata=tile_metadata,
                )
                states[output_entity_id] = _EntityState(
                    entity_id=output_entity_id,
                    segment_id=output_segment_id,
                    mask=child_mask,
                    record=child_record,
                    order_key=source_state.order_key,
                )
            pending_operations.append(
                _PendingOperation(
                    candidate=candidate,
                    provisional=provisional,
                    source_instance_ids=[source_state.order_key],
                    output_entity_ids=output_entity_ids,
                )
            )
            continue

        candidate_evidence = candidate.get("evidence", {})
        if not isinstance(candidate_evidence, Mapping):
            raise ValueError("Refinement candidate evidence must be a mapping")
        declared_vetoes = candidate_evidence.get(
            "hard_vetoes", candidate_evidence.get("vetoes", [])
        )
        if declared_vetoes:
            raise ValueError(
                f"Approved {operation_type} is blocked by hard policy vetoes: "
                f"{list(declared_vetoes)}"
            )
        if policy_config is not None:
            if operation_type == "merge":
                policy_pairs = [
                    (states[sources_ids[0]].record, states[entity_id].record)
                    for entity_id in sources_ids[1:]
                ]
            else:
                if target_id is None:
                    raise ValueError("absorb requires target_entity_id")
                policy_pairs = [
                    (states[sources_ids[0]].record, states[target_id].record)
                ]
            hard_vetoes = sorted(
                {
                    veto
                    for first_record, second_record in policy_pairs
                    for veto in classify_hard_vetoes(
                        operation_type,
                        first_record,
                        second_record,
                        policy_config,
                    )
                }
            )
            if hard_vetoes:
                raise ValueError(
                    f"Approved {operation_type} violates class policy: {hard_vetoes}"
                )

        if str(decision_source).startswith("automatic_"):
            if automatic_pair_rows is None or policy_config is None:
                raise ValueError("Automatic replay evidence was not initialized")
            candidate = _revalidate_automatic_generic_operation(
                candidate,
                states,
                automatic_pair_rows,
                policy_config,
                initial_instance_by_entity,
                initial_mask_by_entity,
                decision_source=decision_source,
            )
            candidate_evidence = candidate["evidence"]

        input_states = [states[value] for value in participating]
        class_ids = {_record_class_id(state.record) for state in input_states}
        cross_class_absorb = len(class_ids) != 1
        target_label: Mapping[str, Any] | None = None
        target_label_source: str | None = None
        cross_class_probability_stats: dict[str, Any] | None = None
        source_class_id: int | None = None
        if cross_class_absorb:
            if operation_type != "absorb":
                raise ValueError(
                    f"merge is restricted to same-class entities; got classes {sorted(class_ids)}"
                )
            resolved_decision, _ = _resolve_decision(candidate, decision_index)
            if resolved_decision == "auto_apply":
                _validate_automatic_unknown_absorb(
                    candidate,
                    states[sources_ids[0]],
                    states[target_id] if target_id is not None else None,
                    probabilities,
                    policy_config,
                    states,
                )
            elif resolved_decision != "approved":
                raise ValueError(
                    "Cross-class absorb must be explicitly approved or pass the "
                    "configured automatic unknown-cleanup rule"
                )
            if probabilities is None:
                raise ValueError("Cross-class absorb requires semantic_probabilities")
            if working_confidence is None:
                raise ValueError("Cross-class absorb requires an existing confidence_map")
            if target_id is None:
                raise ValueError("Cross-class absorb requires target_entity_id")
            target_state = states[target_id]
            class_id = _record_class_id(target_state.record)
            source_class_id = _record_class_id(states[sources_ids[0]].record)
            if class_id < 0 or class_id >= probabilities.shape[0]:
                raise ValueError(
                    f"Cross-class absorb target class {class_id} has no semantic probability channel"
                )
            label_value = target_state.record.get("label")
            if not isinstance(label_value, Mapping):
                raise ValueError("Cross-class absorb target has no semantic label metadata")
            target_label = label_value
            if target_state.record.get("label_source") is not None:
                target_label_source = str(target_state.record["label_source"])
            source_mask = states[sources_ids[0]].mask
            target_confidence = probabilities[class_id][source_mask]
            if not np.isfinite(target_confidence).all():
                raise ValueError("Target-class probabilities contain non-finite values on absorbed pixels")
            working_semantics[source_mask] = class_id
            working_confidence[source_mask] = target_confidence.astype(
                working_confidence.dtype,
                copy=False,
            )
            semantic_map_changed = True
            confidence_map_changed = True
            cross_class_absorbed_pixel_count += int(source_mask.sum())
            cross_class_probability_stats = {
                "source_class_id": source_class_id,
                "target_class_id": class_id,
                "pixel_count": int(source_mask.sum()),
                "minimum": float(target_confidence.min()),
                "mean": float(target_confidence.mean()),
                "maximum": float(target_confidence.max()),
            }
        else:
            class_id = next(iter(class_ids))
            if operation_type == "absorb" and target_id is not None:
                label_value = states[target_id].record.get("label")
                target_label = label_value if isinstance(label_value, Mapping) else None
                if states[target_id].record.get("label_source") is not None:
                    target_label_source = str(states[target_id].record["label_source"])
        union_mask = np.logical_or.reduce([state.mask for state in input_states])
        if not is_four_connected(union_mask):
            raise ValueError(
                f"{operation_type} inputs do not form one 4-connected raster region: {participating}"
            )
        input_area = sum(int(state.mask.sum()) for state in input_states)
        if int(union_mask.sum()) != input_area:
            raise ValueError(f"{operation_type} inputs overlap, so their exact union is ambiguous")

        parameters = copy.deepcopy(dict(candidate.get("parameters", {})))
        parameters["same_class_required"] = not cross_class_absorb
        parameters["cross_class_absorb"] = bool(cross_class_absorb)
        if cross_class_absorb:
            parameters["source_class_id"] = source_class_id
            parameters["target_class_id"] = class_id
        evidence = copy.deepcopy(dict(candidate.get("evidence", {})))
        if cross_class_probability_stats is not None:
            evidence["target_class_probability_on_absorbed_source"] = (
                cross_class_probability_stats
            )
        if operation_type == "merge":
            identity = {
                "operation_type": operation_type,
                "source_entity_ids": sorted(sources_ids),
                "parameters": parameters,
            }
            output_entity_id = (
                str(candidate["output_entity_ids"][0])
                if candidate.get("schema_version") == OPERATION_SCHEMA_VERSION
                else make_merge_entity_id(_tile_id(normalized_result), identity)
            )
            if output_entity_id in states and output_entity_id not in participating:
                raise ValueError(
                    f"merge output entity {output_entity_id!r} collides with an existing entity"
                )
            output_segment_id = make_merge_segment_id(identity)
        else:
            if target_id is None:  # guarded by proposal validation; keeps typing honest
                raise ValueError("absorb requires target_entity_id")
            output_entity_id = target_id
            output_segment_id = states[target_id].segment_id

        provisional = make_operation(
            _tile_id(normalized_result),
            operation_type,
            sources_ids,
            [output_entity_id],
            target_entity_id=target_id,
            source_instance_ids=[states[value].order_key for value in sources_ids],
            output_instance_ids=[],
            proposal_id=(
                str(candidate["proposal_id"])
                if candidate.get("proposal_id") is not None
                else None
            ),
            parameters=parameters,
            evidence=evidence,
            decision_source=decision_source,
            invariants={
                "same_semantic_class": not cross_class_absorb,
                "output_is_4_connected": True,
                "output_union_equals_inputs": True,
                "valid_pixel_coverage_unchanged": True,
                "semantic_map_unchanged": not cross_class_absorb,
                "confidence_map_unchanged": not cross_class_absorb,
                "source_map_unchanged": True,
                "valid_mask_unchanged": True,
            },
        )
        combined_record = build_combined_record(
            [state.record for state in input_states],
            union_mask,
            tile_id=_tile_id(normalized_result),
            instance_id=1,
            entity_id=output_entity_id,
            segment_id=output_segment_id,
            semantic_class_id=class_id,
            operation_id=str(provisional["operation_id"]),
            operation_type=operation_type,
            base_entity_id=target_id if operation_type == "absorb" else None,
            semantic_label=target_label,
            semantic_label_source=target_label_source,
            confidence_map=working_confidence,
            source_map=normalized_result.source_map,
            tile_metadata=tile_metadata,
        )
        for entity_id in participating:
            del states[entity_id]
        states[output_entity_id] = _EntityState(
            entity_id=output_entity_id,
            segment_id=output_segment_id,
            mask=union_mask,
            record=combined_record,
            order_key=min(state.order_key for state in input_states),
        )
        pending_operations.append(
            _PendingOperation(
                candidate=candidate,
                provisional=provisional,
                source_instance_ids=list(provisional["source_instance_ids"]),
                output_entity_ids=[output_entity_id],
            )
        )

    ordered_states = sorted(states.values(), key=lambda state: (state.order_key, state.entity_id))
    rebuilt_map = np.zeros(normalized_result.instance_map.shape, dtype=np.int32)
    rebuilt_masks: list[np.ndarray] = []
    rebuilt_records: list[dict[str, Any]] = []
    output_instance_by_entity: dict[str, int] = {}
    for instance_id, state in enumerate(ordered_states, 1):
        if np.any(rebuilt_map[state.mask] != 0):
            raise ValueError(f"Refinement result contains overlapping entity {state.entity_id!r}")
        rebuilt_map[state.mask] = instance_id
        record = copy.deepcopy(state.record)
        record["instance_id"] = instance_id
        record["mask_index"] = instance_id - 1
        preserved_coordinates = (
            copy.deepcopy(record.get("coordinates"))
            if tile_metadata is None and isinstance(record.get("coordinates"), Mapping)
            else None
        )
        preserved_coordinate_status = (
            copy.deepcopy(record.get("coordinate_refresh"))
            if preserved_coordinates is not None
            else None
        )
        refresh_segment_record(
            record,
            state.mask,
            semantic_class_id=_record_class_id(record),
            confidence_map=working_confidence,
            source_map=normalized_result.source_map,
            tile_metadata=tile_metadata,
        )
        if preserved_coordinates is not None:
            record["coordinates"] = preserved_coordinates
            if preserved_coordinate_status is not None:
                record["coordinate_refresh"] = preserved_coordinate_status
        rebuilt_masks.append(state.mask.copy())
        rebuilt_records.append(record)
        output_instance_by_entity[state.entity_id] = instance_id

    applied_operations: list[dict[str, Any]] = []
    for pending in pending_operations:
        provisional = pending.provisional
        operation = make_operation(
            str(provisional["tile_id"]),
            str(provisional["operation_type"]),
            list(provisional["source_entity_ids"]),
            list(provisional["output_entity_ids"]),
            target_entity_id=provisional.get("target_entity_id"),
            source_instance_ids=pending.source_instance_ids,
            output_instance_ids=[
                output_instance_by_entity[entity_id]
                for entity_id in pending.output_entity_ids
            ],
            proposal_id=provisional.get("proposal_id"),
            parameters=provisional["parameters"],
            evidence=provisional["evidence"],
            decision_source=str(provisional["decision_source"]),
            invariants=provisional["invariants"],
        )
        if operation["operation_id"] != provisional["operation_id"]:
            raise RuntimeError("Operation identity changed while assigning contiguous output IDs")
        applied_operations.append(operation)

    validation = validate_refined_segmentation(
        rebuilt_map,
        working_semantics,
        normalized_result.valid_mask,
        rebuilt_records,
        rebuilt_masks,
        confidence_map=working_confidence,
        source_map=normalized_result.source_map,
        reference_semantic_map=(
            None if semantic_map_changed else normalized_result.semantic_map
        ),
        reference_confidence_map=(
            None if confidence_map_changed else normalized_result.confidence_map
        ),
        reference_source_map=normalized_result.source_map,
        reference_valid_mask=normalized_result.valid_mask,
    )
    report = {
        "status": "complete",
        "phase": "reviewed_operation_replay",
        "tile_id": _tile_id(normalized_result),
        "input_segment_count": len(normalized_result.records),
        "output_segment_count": len(rebuilt_records),
        "candidate_count": len(candidates),
        "applied_operation_count": len(applied_operations),
        "decision_counts": decision_counts,
        "every_valid_pixel_exactly_once": True,
        "semantic_map_unchanged": not semantic_map_changed,
        "confidence_map_unchanged": not confidence_map_changed,
        "source_map_unchanged": normalized_result.source_map is not None,
        "valid_mask_unchanged": True,
        "maps_changed": {
            "instance_map": not np.array_equal(rebuilt_map, normalized_result.instance_map),
            "semantic_map": semantic_map_changed,
            "confidence_map": confidence_map_changed,
            "source_map": False,
            "valid_mask": False,
        },
        "cross_class_absorbed_pixel_count": cross_class_absorbed_pixel_count,
        "semantic_split_parent_count": semantic_split_parent_count,
        "semantic_split_child_count": semantic_split_child_count,
        "validation": validation.to_dict(),
    }
    return RefinementResult(
        masks=rebuilt_masks,
        records=rebuilt_records,
        instance_map=rebuilt_map,
        semantic_map=working_semantics,
        confidence_map=working_confidence,
        source_map=(
            normalized_result.source_map.copy()
            if normalized_result.source_map is not None
            else None
        ),
        valid_mask=normalized_result.valid_mask.copy(),
        operations=[*copy.deepcopy(normalized_result.operations), *applied_operations],
        lineage=_lineage_document(_tile_id(normalized_result), rebuilt_records),
        report=report,
    )


@dataclass(slots=True)
class _EntityState:
    entity_id: str
    segment_id: str
    mask: np.ndarray
    record: dict[str, Any]
    order_key: int


@dataclass(slots=True)
class _PendingOperation:
    candidate: dict[str, Any]
    provisional: dict[str, Any]
    source_instance_ids: list[int]
    output_entity_ids: list[str]


def _validate_stage2_input(
    instance_map: np.ndarray,
    semantic_map: np.ndarray,
    confidence_map: np.ndarray | None,
    source_map: np.ndarray | None,
    valid_mask: np.ndarray,
    records: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
    instances = np.asarray(instance_map)
    semantics = np.asarray(semantic_map)
    valid = np.asarray(valid_mask, dtype=bool)
    if instances.ndim != 2 or semantics.ndim != 2 or valid.ndim != 2:
        raise ValueError("Stage 2 refinement input maps must be two-dimensional")
    if instances.shape != semantics.shape or instances.shape != valid.shape:
        raise ValueError("Stage 2 refinement input map shapes differ")
    if not np.issubdtype(instances.dtype, np.integer):
        raise TypeError("Stage 2 instance_map must have an integer dtype")
    if not np.issubdtype(semantics.dtype, np.integer):
        raise TypeError("Stage 2 semantic_map must have an integer dtype")
    if np.any(instances < 0):
        raise ValueError("Stage 2 instance_map contains negative IDs")
    if np.any(instances[valid] <= 0):
        raise ValueError("Stage 2 refinement requires every valid pixel to have an instance")
    if np.any(instances[~valid] != 0):
        raise ValueError("Stage 2 invalid pixels must have instance ID zero")
    if np.any(semantics[~valid] != NODATA_CLASS_ID):
        raise ValueError("Stage 2 invalid pixels must have the semantic nodata class")
    if np.any(semantics[valid] == NODATA_CLASS_ID):
        raise ValueError("Stage 2 valid pixels must not have the semantic nodata class")

    expected_ids = np.arange(1, len(records) + 1, dtype=np.int64)
    actual_ids = np.unique(instances[valid]).astype(np.int64, copy=False)
    if not np.array_equal(actual_ids, expected_ids):
        raise ValueError(
            f"Stage 2 instance IDs must be contiguous: expected {expected_ids.tolist()}, got {actual_ids.tolist()}"
        )
    by_instance = _records_by_instance(records)
    if sorted(by_instance) != expected_ids.tolist():
        raise ValueError("Stage 2 record IDs do not match the raster IDs")
    for instance_id, record in by_instance.items():
        mask = instances == instance_id
        values = np.unique(semantics[mask])
        label = record.get("label")
        if len(values) != 1 or not isinstance(label, Mapping) or label.get("id") is None:
            raise ValueError(f"Stage 2 instance {instance_id} lacks one semantic label")
        if int(label["id"]) != int(values[0]):
            raise ValueError(f"Stage 2 instance {instance_id} record label disagrees with semantic_map")

    confidence = None if confidence_map is None else np.asarray(confidence_map).copy()
    if confidence is not None:
        if confidence.shape != instances.shape:
            raise ValueError("Stage 2 confidence_map shape differs")
        if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
            raise ValueError("Stage 2 confidence_map must be finite and in [0, 1]")
        if np.any(confidence[~valid] != 0):
            raise ValueError("Stage 2 confidence_map must be zero outside valid pixels")
    sources = None if source_map is None else np.asarray(source_map).copy()
    if sources is not None:
        if sources.shape != instances.shape:
            raise ValueError("Stage 2 source_map shape differs")
        nodata_source = int(SOURCE_CODES["nodata"])
        if np.any(sources[~valid] != nodata_source):
            raise ValueError("Stage 2 source_map must be nodata outside valid pixels")
        if np.any(sources[valid] == nodata_source):
            raise ValueError("Stage 2 source_map must not be nodata on valid pixels")
    return (
        instances.copy(),
        semantics.copy(),
        confidence,
        sources,
        valid.copy(),
    )


def _records_by_instance(
    records: Sequence[Mapping[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for index, record in enumerate(records, 1):
        if not isinstance(record, Mapping):
            raise TypeError(f"Stage 2 record {index} is not a mapping")
        instance_id = int(record.get("instance_id", index))
        if instance_id in result:
            raise ValueError(f"Duplicate Stage 2 record instance ID {instance_id}")
        result[instance_id] = record
    return result


def _validate_result(result: RefinementResult) -> None:
    validate_refined_segmentation(
        result.instance_map,
        result.semantic_map,
        result.valid_mask,
        result.records,
        result.masks,
        confidence_map=result.confidence_map,
        source_map=result.source_map,
    )


def _lineage_document(tile_id: str, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    entities = []
    for record in records:
        entities.append(
            {
                "entity_id": str(record["entity_id"]),
                "segment_id": str(record["segment_id"]),
                "instance_id": int(record["instance_id"]),
                "source_stage2_instance_ids": [
                    int(value) for value in record["source_stage2_instance_ids"]
                ],
                "source_stage2_segment_ids": [
                    str(value) for value in record["source_stage2_segment_ids"]
                ],
                "refinement_operation_ids": [
                    str(value) for value in record.get("refinement_history", [])
                ],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "tile_id": str(tile_id),
        "entity_count": len(entities),
        "entities": entities,
    }


def _tile_id(result: RefinementResult) -> str:
    value = result.lineage.get("tile_id") or result.report.get("tile_id")
    if not value:
        raise ValueError("RefinementResult has no tile_id")
    return str(value)


def _candidate_id(candidate: Mapping[str, Any]) -> str:
    value = candidate.get("proposal_id") or candidate.get("operation_id")
    if not value:
        raise ValueError("A refinement candidate has no proposal_id or operation_id")
    return str(value)


def _validate_candidate(candidate: Mapping[str, Any]) -> None:
    schema = candidate.get("schema_version")
    if schema == PROPOSAL_SCHEMA_VERSION:
        validate_proposal(candidate)
    elif schema == OPERATION_SCHEMA_VERSION:
        validate_operation(candidate)
    else:
        raise ValueError(f"Unsupported refinement candidate schema: {schema!r}")


def _split_partition_index(
    split_partitions: Mapping[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    if split_partitions is None:
        return {}
    if not isinstance(split_partitions, Mapping):
        raise TypeError("split_partitions must be a proposal-ID keyed mapping")
    result: dict[str, np.ndarray] = {}
    for raw_key, raw_partition in split_partitions.items():
        key = str(raw_key)
        if not key:
            raise ValueError("split_partitions cannot contain an empty proposal ID")
        if key in result:
            raise ValueError(f"duplicate semantic split partition key {key!r}")
        if raw_partition is None:
            raise ValueError(f"semantic split partition {key!r} is missing")
        result[key] = np.asarray(raw_partition)
    return result


def _decision_index(
    decisions: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    if decisions is None:
        return {}
    if isinstance(decisions, Mapping):
        if "decisions" in decisions:
            nested = decisions["decisions"]
            if not isinstance(nested, Sequence) or isinstance(nested, (str, bytes, bytearray)):
                raise ValueError("decisions.decisions must be a sequence")
            return _decision_index(nested)
        if decisions.get("proposal_id") or decisions.get("operation_id"):
            if decisions.get("schema_version") is not None:
                validate_decision(decisions)
            key = str(decisions.get("proposal_id") or decisions.get("operation_id"))
            return {key: copy.deepcopy(dict(decisions))}
        return {str(key): copy.deepcopy(value) for key, value in decisions.items()}
    if not isinstance(decisions, Sequence) or isinstance(decisions, (str, bytes, bytearray)):
        raise ValueError("decisions must be a mapping or a sequence of decision mappings")
    result: dict[str, Any] = {}
    for decision in decisions:
        if not isinstance(decision, Mapping):
            raise ValueError("Every refinement decision must be a mapping")
        if decision.get("schema_version") is not None:
            validate_decision(decision)
        key = decision.get("proposal_id") or decision.get("operation_id")
        if not key:
            raise ValueError("Every refinement decision must identify its proposal or operation")
        key = str(key)
        if key in result:
            raise ValueError(f"Duplicate refinement decision for {key!r}")
        result[key] = copy.deepcopy(dict(decision))
    return result


def _resolve_decision(
    candidate: Mapping[str, Any],
    decision_index: Mapping[str, Any],
) -> tuple[str, str]:
    candidate_id = _candidate_id(candidate)
    raw = decision_index.get(candidate_id)
    if raw is None:
        if candidate.get("schema_version") == OPERATION_SCHEMA_VERSION:
            if candidate.get("status") == "applied":
                return "approved", str(candidate.get("decision_source", "operation_replay"))
            return "rejected", str(candidate.get("decision_source", "operation_replay"))
        decision = str(candidate.get("decision", "pending"))
        source = "proposal_embedded"
    elif isinstance(raw, str):
        decision = raw
        source = "review_decision"
    elif isinstance(raw, Mapping):
        decision = str(raw.get("decision", raw.get("status", "")))
        source = str(raw.get("decision_source", "human_review"))
    else:
        raise ValueError(f"Invalid decision value for {candidate_id!r}")
    if decision not in {"pending", "approved", "auto_apply", "rejected", "uncertain"}:
        raise ValueError(f"Unsupported decision {decision!r} for {candidate_id!r}")
    return decision, source


def _record_class_id(record: Mapping[str, Any]) -> int:
    label = record.get("label")
    if not isinstance(label, Mapping) or label.get("id") is None:
        raise ValueError(f"Entity {record.get('entity_id')!r} has no semantic label")
    return int(label["id"])


def _record_class_name(record: Mapping[str, Any]) -> str:
    label = record.get("label")
    if not isinstance(label, Mapping) or not str(label.get("name", "")):
        raise ValueError(f"Entity {record.get('entity_id')!r} has no semantic class name")
    return str(label["name"])


def _revalidate_automatic_generic_operation(
    candidate: Mapping[str, Any],
    states: Mapping[str, _EntityState],
    pair_rows: Sequence[Mapping[str, Any]],
    policy_config: Mapping[str, Any],
    initial_instance_by_entity: Mapping[str, int],
    initial_mask_by_entity: Mapping[str, np.ndarray],
    *,
    decision_source: str,
) -> dict[str, Any]:
    """Recompute every generic automatic gate from the immutable tile inputs."""

    automatic_config = policy_config.get("automatic")
    if not isinstance(automatic_config, Mapping):
        raise ValueError("Automatic refinement has no validated configuration")
    policy_version = str(automatic_config.get("policy_version", ""))
    if decision_source != policy_version:
        raise ValueError(
            "Automatic decision source does not match the configured policy version"
        )

    replayed = copy.deepcopy(dict(candidate))
    operation_type = str(replayed.get("operation_type", ""))
    source_ids = [str(value) for value in replayed.get("source_entity_ids", [])]
    target_value = replayed.get("target_entity_id")
    target_id = None if target_value is None else str(target_value)
    if operation_type == "merge":
        if len(source_ids) != 2 or target_id is not None:
            raise ValueError("Automatic merge replay requires exactly two sources")
        first_id, second_id = source_ids
        first_instance = int(initial_instance_by_entity[first_id])
        second_instance = int(initial_instance_by_entity[second_id])
        row = _automatic_pair_row(pair_rows, first_instance, second_instance)
        minimum_fraction = float(
            row["minimum_perimeter_shared_boundary_fraction"]
        )
        weights = policy_config.get("evidence")
        if not isinstance(weights, Mapping):
            raise ValueError("Automatic merge replay requires evidence weights")
        score = (
            float(weights["semantic_weight"])
            + float(weights["boundary_weight"])
            * float(row["weak_color_boundary_score"])
            + float(weights["appearance_weight"])
            * float(row["region_mean_color_similarity"])
            + float(weights["shape_weight"])
            * min(1.0, minimum_fraction / 0.25)
        )
        first_class = _record_class_name(states[first_id].record)
        second_class = _record_class_name(states[second_id].record)
        replayed["evidence"] = {
            **copy.deepcopy(dict(row)),
            "source_touches_tile_boundary": _mask_touches_tile_boundary(
                initial_mask_by_entity[first_id]
            ),
            "target_touches_tile_boundary": _mask_touches_tile_boundary(
                initial_mask_by_entity[second_id]
            ),
            "semantic_class_match": first_class == second_class,
            "hard_vetoes": [],
            "policy_gate_passed": True,
        }
        replayed["score"] = min(1.0, max(0.0, float(score)))
    elif operation_type == "absorb":
        if len(source_ids) != 1 or target_id is None:
            raise ValueError("Automatic absorb replay requires one source and one target")
        source_id = source_ids[0]
        source_instance = int(initial_instance_by_entity[source_id])
        target_instance = int(initial_instance_by_entity[target_id])
        contacts: list[dict[str, Any]] = []
        for pair in pair_rows:
            first = int(pair["source_instance_id"])
            second = int(pair["target_instance_id"])
            if source_instance not in {first, second}:
                continue
            neighbor = second if first == source_instance else first
            contacts.append(
                orient_absorb_evidence(pair, source_instance, neighbor)
            )
        if not contacts:
            raise ValueError("Automatic absorb source has no shared-edge neighbor")
        contacts.sort(
            key=lambda value: (
                -float(value["candidate_shared_boundary_fraction"]),
                -float(value["weak_color_boundary_score"]),
                int(value["neighbor_instance_id"]),
            )
        )
        best = contacts[0]
        if int(best["neighbor_instance_id"]) != target_instance:
            raise ValueError(
                "Automatic absorb target is no longer the deterministic best neighbor"
            )
        runner_fraction = (
            float(contacts[1]["candidate_shared_boundary_fraction"])
            if len(contacts) > 1
            else 0.0
        )
        best_fraction = float(best["candidate_shared_boundary_fraction"])
        best["runner_up_boundary_fraction"] = runner_fraction
        best["best_to_runner_up_margin"] = best_fraction - runner_fraction
        best["best_to_runner_up_boundary_fraction_margin"] = (
            best_fraction - runner_fraction
        )
        replayed["evidence"] = {
            **copy.deepcopy(best),
            "source_touches_tile_boundary": _mask_touches_tile_boundary(
                initial_mask_by_entity[source_id]
            ),
            "target_touches_tile_boundary": _mask_touches_tile_boundary(
                initial_mask_by_entity[target_id]
            ),
            "hard_vetoes": [],
            "policy_gate_passed": True,
        }
        replayed["score"] = min(
            1.0,
            0.50 * best_fraction
            + 0.25 * float(best["weak_color_boundary_score"])
            + 0.25 * float(best["region_mean_color_similarity"]),
        )
    else:
        raise ValueError(
            f"Automatic selective replay does not support {operation_type!r}"
        )

    evaluation = evaluate_automatic_proposal(replayed, automatic_config)
    if not evaluation.eligible:
        reasons = list(evaluation.reasons)
        raise ValueError(
            "Automatic refinement proposal failed replay-time evidence gates: "
            f"{reasons}"
        )
    replayed["evidence"]["automatic_policy_replay"] = {
        "policy_version": policy_version,
        "evidence_recomputed": True,
        "decision": "approved",
    }
    return replayed


def _automatic_pair_row(
    pair_rows: Sequence[Mapping[str, Any]],
    first_instance: int,
    second_instance: int,
) -> dict[str, Any]:
    expected = tuple(sorted((int(first_instance), int(second_instance))))
    for row in pair_rows:
        actual = (
            int(row["source_instance_id"]),
            int(row["target_instance_id"]),
        )
        if actual == expected:
            return copy.deepcopy(dict(row))
    raise ValueError(
        "Automatic refinement inputs are not directly shared-edge adjacent"
    )


def _mask_touches_tile_boundary(mask: np.ndarray) -> bool:
    values = np.asarray(mask, dtype=bool)
    return bool(
        values[0, :].any()
        or values[-1, :].any()
        or values[:, 0].any()
        or values[:, -1].any()
    )


def _validate_automatic_unknown_absorb(
    candidate: Mapping[str, Any],
    source_state: _EntityState,
    target_state: _EntityState | None,
    probabilities: np.ndarray | None,
    policy_config: Mapping[str, Any] | None,
    states: Mapping[str, _EntityState],
) -> None:
    """Recheck every safety-critical automatic cleanup condition from rasters.

    Proposal evidence is useful for audit and visualization, but it is not the
    source of truth for an automatic semantic edit.  This gate recomputes the
    probability and contact conditions from the immutable model raster and the
    exact masks immediately before the absorb is applied.
    """

    if target_state is None:
        raise ValueError("Automatic unknown cleanup requires a target entity")
    if probabilities is None or policy_config is None:
        raise ValueError(
            "Automatic unknown cleanup requires semantic probabilities and policy configuration"
        )
    operations = policy_config.get("operations")
    cleanup = operations.get("unknown_cleanup") if isinstance(operations, Mapping) else None
    if not isinstance(cleanup, Mapping):
        raise ValueError("Automatic unknown cleanup has no validated configuration")
    if cleanup.get("enabled") is not True or cleanup.get("auto_apply") is not True:
        raise ValueError("Automatic unknown cleanup is disabled")

    parameters = candidate.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("Automatic unknown cleanup proposal parameters are missing")
    rule_version = parameters.get("cleanup_rule_version")
    if parameters.get("automatic_unknown_cleanup") is not True:
        raise ValueError("Cross-class auto_apply is restricted to unknown cleanup proposals")
    if rule_version == "contextual-micro-v1":
        _validate_automatic_contextual_absorb(
            candidate,
            source_state,
            target_state,
            probabilities,
            policy_config,
            states,
        )
        return
    if rule_version != "micro-unknown-v1":
        raise ValueError("Cross-class auto_apply uses an unsupported cleanup rule")

    expected_thresholds = {
        key: copy.deepcopy(cleanup[key])
        for key in (
            "maximum_segment_area_pixels",
            "allowed_target_classes",
            "minimum_mean_target_probability",
            "minimum_probability_margin",
            "minimum_dominant_pixel_fraction",
            "minimum_shared_boundary_fraction",
            "minimum_weak_color_boundary_score",
            "minimum_region_mean_color_similarity",
            "require_not_tile_boundary",
        )
    }
    if canonical_json(parameters.get("thresholds", {})) != canonical_json(expected_thresholds):
        raise ValueError("Automatic unknown cleanup threshold snapshot is stale or altered")

    if _record_class_name(source_state.record) != "unknown":
        raise ValueError("Automatic unknown cleanup source must be class 'unknown'")
    target_class_name = _record_class_name(target_state.record)
    allowed_classes = [str(value) for value in cleanup["allowed_target_classes"]]
    if target_class_name not in allowed_classes:
        raise ValueError(
            f"Automatic unknown cleanup cannot absorb into protected class {target_class_name!r}"
        )

    source_mask = np.asarray(source_state.mask, dtype=bool)
    target_mask = np.asarray(target_state.mask, dtype=bool)
    area = int(source_mask.sum())
    if area <= 0 or area > int(cleanup["maximum_segment_area_pixels"]):
        raise ValueError("Automatic unknown cleanup source area is outside the configured limit")
    if cleanup.get("require_not_tile_boundary") is True and (
        source_mask[0, :].any()
        or source_mask[-1, :].any()
        or source_mask[:, 0].any()
        or source_mask[:, -1].any()
    ):
        raise ValueError("Automatic unknown cleanup cannot modify a tile-boundary segment")

    target_class_id = _record_class_id(target_state.record)
    selected = np.asarray(probabilities[:, source_mask], dtype=np.float64)
    if selected.size == 0 or target_class_id < 0 or target_class_id >= selected.shape[0]:
        raise ValueError("Automatic unknown cleanup target has no probability channel")
    means = selected.mean(axis=1)
    target_probability = float(means[target_class_id])
    competitors = np.delete(means, target_class_id)
    probability_margin = target_probability - (
        float(competitors.max()) if competitors.size else 0.0
    )
    dominant_class_id = int(np.argmax(means))
    dominant_pixel_fraction = float(
        np.mean(np.argmax(selected, axis=0) == target_class_id)
    )
    if dominant_class_id != target_class_id:
        raise ValueError("Automatic cleanup target is not the source's dominant model class")
    if target_probability < float(cleanup["minimum_mean_target_probability"]):
        raise ValueError("Automatic cleanup target probability is below threshold")
    if probability_margin < float(cleanup["minimum_probability_margin"]):
        raise ValueError("Automatic cleanup probability margin is below threshold")
    if dominant_pixel_fraction < float(cleanup["minimum_dominant_pixel_fraction"]):
        raise ValueError("Automatic cleanup dominant-pixel fraction is below threshold")

    shared_sides = _shared_mask_side_count(source_mask, target_mask)
    source_perimeter = _mask_perimeter_sides(source_mask)
    boundary_fraction = shared_sides / max(1, source_perimeter)
    if boundary_fraction < float(cleanup["minimum_shared_boundary_fraction"]):
        raise ValueError("Automatic cleanup shared-boundary fraction is below threshold")

    evidence = candidate.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("Automatic unknown cleanup proposal evidence is missing")
    if float(evidence.get("weak_color_boundary_score", -1.0)) < float(
        cleanup["minimum_weak_color_boundary_score"]
    ):
        raise ValueError("Automatic cleanup color-boundary support is below threshold")
    if float(evidence.get("region_mean_color_similarity", -1.0)) < float(
        cleanup["minimum_region_mean_color_similarity"]
    ):
        raise ValueError("Automatic cleanup region similarity is below threshold")


def _shared_mask_side_count(first: np.ndarray, second: np.ndarray) -> int:
    return int(
        np.count_nonzero(first[:, :-1] & second[:, 1:])
        + np.count_nonzero(first[:, 1:] & second[:, :-1])
        + np.count_nonzero(first[:-1, :] & second[1:, :])
        + np.count_nonzero(first[1:, :] & second[:-1, :])
    )


def _validate_automatic_contextual_absorb(
    candidate: Mapping[str, Any],
    source_state: _EntityState,
    target_state: _EntityState,
    probabilities: np.ndarray,
    policy_config: Mapping[str, Any],
    states: Mapping[str, _EntityState],
) -> None:
    operations = policy_config.get("operations")
    cleanup = (
        operations.get("contextual_cleanup")
        if isinstance(operations, Mapping)
        else None
    )
    if not isinstance(cleanup, Mapping):
        raise ValueError("Contextual cleanup has no validated configuration")
    if cleanup.get("enabled") is not True or cleanup.get("auto_apply") is not True:
        raise ValueError("Contextual micro cleanup is disabled")

    parameters = candidate.get("parameters")
    evidence = candidate.get("evidence")
    if not isinstance(parameters, Mapping) or not isinstance(evidence, Mapping):
        raise ValueError("Contextual cleanup proposal metadata is incomplete")
    if parameters.get("contextual_evidence_digest") != stable_digest(evidence):
        raise ValueError("Contextual cleanup evidence digest is stale or altered")
    expected_thresholds = {
        key: copy.deepcopy(value)
        for key, value in cleanup.items()
        if key != "auto_apply"
    }
    if canonical_json(parameters.get("thresholds", {})) != canonical_json(expected_thresholds):
        raise ValueError("Contextual cleanup threshold snapshot is stale or altered")

    if _record_class_name(source_state.record) != "unknown":
        raise ValueError("Contextual cleanup source must be class 'unknown'")
    target_class_name = _record_class_name(target_state.record)
    if target_class_name == "unknown":
        raise ValueError("Contextual cleanup cannot target another unknown entity")
    source_mask = np.asarray(source_state.mask, dtype=bool)
    target_mask = np.asarray(target_state.mask, dtype=bool)
    area = int(source_mask.sum())
    if area <= 0 or area > int(cleanup["maximum_segment_area_pixels"]):
        raise ValueError("Contextual cleanup source area is outside its bounded scope")
    if (
        source_mask[0, :].any()
        or source_mask[-1, :].any()
        or source_mask[:, 0].any()
        or source_mask[:, -1].any()
    ):
        raise ValueError("Contextual cleanup cannot modify a tile-boundary segment")
    shared_sides = _shared_mask_side_count(source_mask, target_mask)
    if shared_sides <= 0:
        raise ValueError("Contextual cleanup target is not directly raster-adjacent")
    source_perimeter = _mask_perimeter_sides(source_mask)
    boundary_fraction = shared_sides / max(1, source_perimeter)

    selected = np.asarray(probabilities[:, source_mask], dtype=np.float64)
    means = selected.mean(axis=1)
    order = sorted(range(len(means)), key=lambda value: (-means[value], value))
    top_class_id = int(order[0])
    runner_up_id = int(order[1]) if len(order) > 1 else None
    top_probability = float(means[top_class_id])
    probability_margin = top_probability - (
        float(means[runner_up_id]) if runner_up_id is not None else 0.0
    )
    pixel_maxima = selected.max(axis=0)
    ties = np.isclose(selected, pixel_maxima[None, :], rtol=0.0, atol=1e-12)
    top_purity = float(np.mean(ties[top_class_id] & (ties.sum(axis=0) == 1)))
    target_class_id = _record_class_id(target_state.record)
    target_probability = float(means[target_class_id])

    neighbors = [
        state
        for entity_id, state in states.items()
        if entity_id != source_state.entity_id
        and _shared_mask_side_count(source_mask, np.asarray(state.mask, dtype=bool)) > 0
    ]
    known_neighbors = [
        state for state in neighbors if _record_class_name(state.record) != "unknown"
    ]
    same_target_class_neighbors = [
        state
        for state in known_neighbors
        if _record_class_name(state.record) == target_class_name
    ]

    top_class_name = {
        0: "bareland",
        1: "rangeland",
        2: "developed space",
        3: "road",
        4: "tree",
        5: "water",
        6: "agriculture land",
        7: "buildings",
    }.get(top_class_id)
    matching_top_neighbors = [
        state
        for state in known_neighbors
        if _record_class_id(state.record) == top_class_id
    ]
    if not matching_top_neighbors and top_class_name == "tree" and (
        area >= int(cleanup["standalone_tree_minimum_area_pixels"])
        and top_probability >= float(cleanup["standalone_tree_minimum_probability"])
        and probability_margin >= float(cleanup["standalone_tree_minimum_margin"])
        and top_purity >= float(cleanup["standalone_tree_minimum_purity"])
    ):
        raise ValueError("Strong standalone tree evidence must be preserved")
    if not matching_top_neighbors and top_class_name == "buildings" and (
        area >= int(cleanup["standalone_building_minimum_area_pixels"])
        and top_probability >= float(cleanup["standalone_building_minimum_probability"])
        and probability_margin >= float(cleanup["standalone_building_minimum_margin"])
        and top_purity >= float(cleanup["standalone_building_minimum_purity"])
    ):
        raise ValueError("Strong standalone building evidence must be preserved")

    if target_class_name == "agriculture land":
        if len(same_target_class_neighbors) != 1:
            raise ValueError("Contextual agriculture cleanup requires one unique field target")
        if boundary_fraction < float(cleanup["agriculture_minimum_boundary_fraction"]):
            raise ValueError("Contextual agriculture boundary support is insufficient")
    elif target_class_name == "buildings":
        if len(same_target_class_neighbors) != 1:
            raise ValueError("Contextual building cleanup requires one unique building target")
        if target_probability < float(cleanup["building_minimum_probability"]):
            raise ValueError("Contextual building probability support is insufficient")
    elif target_class_name == "road":
        if (
            target_probability < float(cleanup["road_minimum_probability"])
            and target_class_id not in order[:2]
        ):
            raise ValueError("Contextual road probability/top-two support is insufficient")

    target_rows = evidence.get("target_candidates")
    if not isinstance(target_rows, Sequence) or isinstance(
        target_rows, (str, bytes, bytearray)
    ):
        raise ValueError("Contextual cleanup has no target-candidate audit rows")
    eligible_rows = [
        row
        for row in target_rows
        if isinstance(row, Mapping)
        and row.get("eligible") is True
        and row.get("energy") is not None
    ]
    if not eligible_rows:
        raise ValueError("Contextual cleanup has no eligible target")
    expected_row = min(
        eligible_rows,
        key=lambda row: (float(row["energy"]), int(row["target_instance_id"])),
    )
    if str(expected_row.get("target_entity_id")) != target_state.entity_id:
        raise ValueError("Contextual cleanup target is not the deterministic energy minimum")
    semantic_consensus = bool(
        target_class_id == top_class_id
        and top_probability
        >= float(cleanup["semantic_consensus_minimum_probability"])
        and probability_margin
        >= float(cleanup["semantic_consensus_minimum_margin"])
        and top_purity >= float(cleanup["semantic_consensus_minimum_purity"])
    )
    if not semantic_consensus:
        from .contextual_cleanup import contextual_energy_limit

        selected_energy = float(expected_row["energy"])
        energy_limit = contextual_energy_limit(
            {
                **expected_row,
                "target_class_id": target_class_id,
                "mean_target_probability_on_source": target_probability,
                "source_boundary_fraction": boundary_fraction,
            },
            order[:2],
            cleanup,
        )
        if selected_energy > energy_limit:
            raise ValueError("Contextual cleanup energy is above the automatic maximum")
        ranked_rows = sorted(
            eligible_rows,
            key=lambda row: (
                float(row["energy"]),
                int(row["target_instance_id"]),
            ),
        )
        if len(ranked_rows) > 1:
            energy_margin = float(ranked_rows[1]["energy"]) - selected_energy
            if energy_margin < float(cleanup["minimum_energy_margin"]):
                raise ValueError(
                    "Contextual cleanup best-to-runner-up energy margin is below minimum"
                )


def _mask_perimeter_sides(mask: np.ndarray) -> int:
    area = int(np.count_nonzero(mask))
    internal = int(
        np.count_nonzero(mask[:, :-1] & mask[:, 1:])
        + np.count_nonzero(mask[:-1, :] & mask[1:, :])
    )
    return 4 * area - 2 * internal


def _validate_semantic_probabilities(
    value: np.ndarray | None,
    raster_shape: tuple[int, int],
) -> np.ndarray | None:
    if value is None:
        return None
    probabilities = np.asarray(value)
    if probabilities.ndim != 3 or probabilities.shape[1:] != raster_shape:
        raise ValueError(
            "semantic_probabilities must use class-first shape (class_count, height, width)"
        )
    if probabilities.shape[0] <= 0:
        raise ValueError("semantic_probabilities must contain at least one class channel")
    if not np.issubdtype(probabilities.dtype, np.number):
        raise TypeError("semantic_probabilities must have a numeric dtype")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("semantic_probabilities must contain finite values between zero and one")
    return probabilities


__all__ = [
    "NormalizationResult",
    "RefinementResult",
    "apply_reviewed_operations",
    "consolidate_micro_components",
    "connected_components_4",
    "normalize_4_connected_components",
]
