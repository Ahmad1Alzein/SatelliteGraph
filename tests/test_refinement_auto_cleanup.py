from __future__ import annotations

import copy

import numpy as np
import pytest

from src.refinement.config import select_refinement_config
from src.refinement.pipeline import _attach_ontology, _build_proposals
from src.refinement.topology import (
    apply_reviewed_operations,
    normalize_4_connected_components,
)


def _record(instance_id: int, class_id: int, class_name: str, mask: np.ndarray) -> dict:
    rows, columns = np.nonzero(mask)
    return {
        "instance_id": instance_id,
        "segment_id": f"segment-{instance_id}",
        "area_pixels": int(mask.sum()),
        "bbox_xywh": [
            int(columns.min()),
            int(rows.min()),
            int(columns.max() - columns.min() + 1),
            int(rows.max() - rows.min() + 1),
        ],
        "bbox_xyxy": [
            int(columns.min()),
            int(rows.min()),
            int(columns.max() + 1),
            int(rows.max() + 1),
        ],
        "centroid_rc": [float(rows.mean()), float(columns.mean())],
        "centroid_xy": [float(columns.mean() + 0.5), float(rows.mean() + 0.5)],
        "touches_tile_boundary": bool(
            rows.min() == 0
            or columns.min() == 0
            or rows.max() == mask.shape[0] - 1
            or columns.max() == mask.shape[1] - 1
        ),
        "mean_semantic_confidence": 0.9,
        "label": {
            "id": class_id,
            "name": class_name,
            "source_model": "fixture",
            "method": "fixture",
        },
        "geometry_source": {"model": "fixture", "method": "fixture"},
        "metrics": {"area_pixels": int(mask.sum())},
        "provenance": {"fixture": True},
    }


def test_two_micro_unknowns_auto_absorb_into_one_target_with_exact_lineage() -> None:
    instance_map = np.ones((5, 7), dtype=np.int32)
    instance_map[2, 2] = 2
    instance_map[2, 4] = 3
    masks = [instance_map == value for value in (1, 2, 3)]
    records = [
        _record(1, 1, "rangeland", masks[0]),
        _record(2, 8, "unknown", masks[1]),
        _record(3, 8, "unknown", masks[2]),
    ]
    semantic_map = np.full(instance_map.shape, 1, dtype=np.uint8)
    semantic_map[instance_map > 1] = 8
    confidence_map = np.full(instance_map.shape, 0.9, dtype=np.float32)
    source_map = np.full(instance_map.shape, 2, dtype=np.uint8)
    valid_mask = np.ones(instance_map.shape, dtype=bool)

    normalized = normalize_4_connected_components(
        instance_map,
        semantic_map,
        confidence_map,
        source_map,
        valid_mask,
        records,
        tile_id="fixture__tile_00000",
    )
    config = select_refinement_config()
    _attach_ontology(normalized.records, config)

    probabilities = np.zeros((8, *instance_map.shape), dtype=np.float32)
    probabilities[1, instance_map > 1] = 0.85
    probabilities[2, instance_map > 1] = 0.05
    rgb = np.full((*instance_map.shape, 3), 128, dtype=np.uint8)
    proposals, report = _build_proposals(normalized, rgb, probabilities, config)
    auto_absorbs = [
        proposal
        for proposal in proposals
        if proposal["operation_type"] == "absorb"
        and proposal["decision"] == "auto_apply"
    ]

    assert len(auto_absorbs) == 2
    assert report["unknown_cleanup_auto_apply_count"] == 2
    assert len({proposal["target_entity_id"] for proposal in auto_absorbs}) == 1

    refined = apply_reviewed_operations(
        normalized,
        [proposal for proposal in proposals if proposal["operation_type"] != "split_components"],
        semantic_probabilities=probabilities,
        policy_config=config,
    )

    assert len(refined.records) == 1
    assert np.all(refined.instance_map == 1)
    assert np.all(refined.semantic_map == 1)
    assert len([op for op in refined.operations if op["operation_type"] == "absorb"]) == 2
    output = refined.records[0]
    assert output["label"]["name"] == "rangeland"
    assert output["source_stage2_instance_ids"] == [1, 2, 3]
    assert len(output["stage2_metadata_records"]) == 3


def test_automatic_cleanup_rechecks_probability_raster_at_replay() -> None:
    instance_map = np.ones((3, 3), dtype=np.int32)
    instance_map[1, 1] = 2
    masks = [instance_map == value for value in (1, 2)]
    records = [
        _record(1, 1, "rangeland", masks[0]),
        _record(2, 8, "unknown", masks[1]),
    ]
    semantic_map = np.where(instance_map == 2, 8, 1).astype(np.uint8)
    normalized = normalize_4_connected_components(
        instance_map,
        semantic_map,
        np.full(instance_map.shape, 0.9, dtype=np.float32),
        np.full(instance_map.shape, 2, dtype=np.uint8),
        np.ones(instance_map.shape, dtype=bool),
        records,
        tile_id="fixture__tile_00001",
    )
    config = select_refinement_config()
    _attach_ontology(normalized.records, config)
    strong = np.zeros((8, *instance_map.shape), dtype=np.float32)
    strong[1, 1, 1] = 0.9
    proposals, _ = _build_proposals(
        normalized,
        np.full((*instance_map.shape, 3), 100, dtype=np.uint8),
        strong,
        config,
    )
    absorb = [proposal for proposal in proposals if proposal["operation_type"] == "absorb"]
    assert len(absorb) == 1 and absorb[0]["decision"] == "auto_apply"

    weak = strong.copy()
    weak[:, 1, 1] = 0.0
    weak[1, 1, 1] = 0.4
    weak[2, 1, 1] = 0.35
    try:
        apply_reviewed_operations(
            normalized,
            absorb,
            semantic_probabilities=weak,
            policy_config=config,
        )
    except ValueError as exc:
        assert "probability" in str(exc).lower() or "dominant" in str(exc).lower()
    else:  # pragma: no cover - a safety regression must fail loudly
        raise AssertionError("Replay accepted a stale automatic cleanup decision")


def test_contextual_micro_cleanup_is_bounded_auditable_and_digest_bound() -> None:
    instance_map = np.ones((3, 3), dtype=np.int32)
    instance_map[1, 1] = 2
    masks = [instance_map == value for value in (1, 2)]
    records = [
        _record(1, 1, "rangeland", masks[0]),
        _record(2, 8, "unknown", masks[1]),
    ]
    semantic_map = np.where(instance_map == 2, 8, 1).astype(np.uint8)
    normalized = normalize_4_connected_components(
        instance_map,
        semantic_map,
        np.full(instance_map.shape, 0.5, dtype=np.float32),
        np.full(instance_map.shape, 2, dtype=np.uint8),
        np.ones(instance_map.shape, dtype=bool),
        records,
        tile_id="fixture__tile_00002",
    )
    config = select_refinement_config()
    _attach_ontology(normalized.records, config)
    probabilities = np.zeros((8, *instance_map.shape), dtype=np.float32)
    probabilities[1, 1, 1] = 0.40
    probabilities[4, 1, 1] = 0.35
    proposals, _ = _build_proposals(
        normalized,
        np.full((*instance_map.shape, 3), 100, dtype=np.uint8),
        probabilities,
        config,
    )
    absorb = [proposal for proposal in proposals if proposal["operation_type"] == "absorb"]
    assert len(absorb) == 1
    assert absorb[0]["decision"] == "auto_apply"
    assert absorb[0]["parameters"]["cleanup_rule_version"] == "contextual-micro-v1"
    assert absorb[0]["evidence"]["decision_tier"] == "contextual"

    refined = apply_reviewed_operations(
        normalized,
        absorb,
        semantic_probabilities=probabilities,
        policy_config=config,
    )
    assert len(refined.records) == 1
    assert np.all(refined.semantic_map == 1)

    tampered = copy.deepcopy(absorb[0])
    tampered["evidence"]["selected_energy"] = 0.0
    with pytest.raises(ValueError, match="digest"):
        apply_reviewed_operations(
            normalized,
            tampered if isinstance(tampered, list) else [tampered],
            semantic_probabilities=probabilities,
            policy_config=config,
        )
