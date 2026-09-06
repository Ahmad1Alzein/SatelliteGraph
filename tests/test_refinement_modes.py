from __future__ import annotations

import copy

import numpy as np
import pytest

from src.refinement.automatic import select_automatic_decisions
from src.refinement.config import (
    normalize_refinement_mode,
    select_refinement_config,
)
from src.refinement.evidence import pair_evidence
from src.refinement.schema import make_proposal
from src.refinement.topology import (
    apply_reviewed_operations,
    normalize_4_connected_components,
)


def _automatic_merge_case() -> tuple[object, dict, dict, np.ndarray]:
    instance_map = np.full((8, 8), 3, dtype=np.int32)
    instance_map[2:5, 2] = 1
    instance_map[2:5, 3] = 2
    semantic_map = np.full(instance_map.shape, 5, dtype=np.uint8)
    confidence = np.full(instance_map.shape, 0.95, dtype=np.float32)
    source_map = np.full(instance_map.shape, 2, dtype=np.uint8)
    valid = np.ones(instance_map.shape, dtype=bool)
    records = [
        {
            "instance_id": instance_id,
            "segment_id": f"water-{instance_id}",
            "source_tile_id": "automatic-tile",
            "label": {"id": 5, "name": "water"},
        }
        for instance_id in (1, 2, 3)
    ]
    normalized = normalize_4_connected_components(
        instance_map,
        semantic_map,
        confidence,
        source_map,
        valid,
        records,
        tile_id="automatic-tile",
    )
    cfg = select_refinement_config()
    cfg["automatic"]["merge"].update(
        {
            "minimum_score": 0.90,
            "minimum_shared_boundary_pixels": 1,
            "minimum_shared_boundary_fraction": 0.25,
        }
    )
    rgb = np.zeros((*instance_map.shape, 3), dtype=np.uint8)
    rows = pair_evidence(normalized.instance_map, rgb)
    first, second = normalized.records[0], normalized.records[1]
    row = next(
        value
        for value in rows
        if {
            int(value["source_instance_id"]),
            int(value["target_instance_id"]),
        }
        == {int(first["instance_id"]), int(second["instance_id"])}
    )
    minimum_fraction = float(row["minimum_perimeter_shared_boundary_fraction"])
    weights = cfg["evidence"]
    score = (
        float(weights["semantic_weight"])
        + float(weights["boundary_weight"]) * float(row["weak_color_boundary_score"])
        + float(weights["appearance_weight"])
        * float(row["region_mean_color_similarity"])
        + float(weights["shape_weight"])
        * min(1.0, minimum_fraction / 0.25)
    )
    proposal = make_proposal(
        "automatic-tile",
        "merge",
        [str(first["entity_id"]), str(second["entity_id"])],
        parameters={
            "class_name": "water",
            "entity_policy": "connected_homogeneous_region",
            "review_only": True,
            "score_kind": "uncalibrated_review_priority",
        },
        evidence={
            **copy.deepcopy(row),
            "source_touches_tile_boundary": False,
            "target_touches_tile_boundary": False,
            "semantic_class_match": True,
            "hard_vetoes": [],
            "policy_gate_passed": True,
        },
        score=score,
        decision="pending",
    )
    return normalized, cfg, proposal, rgb


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("human_review", "human_review"),
        ("review", "human_review"),
        ("proposal_only", "human_review"),
        ("automatic", "automatic"),
        ("auto", "automatic"),
    ],
)
def test_refinement_mode_aliases_are_safe(value: str, expected: str) -> None:
    assert normalize_refinement_mode(value) == expected


def test_unknown_refinement_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="human_review.*automatic"):
        normalize_refinement_mode("approve_everything")


def test_automatic_merge_is_selected_and_revalidated_from_rgb() -> None:
    normalized, cfg, proposal, rgb = _automatic_merge_case()
    selection = select_automatic_decisions([proposal], cfg["automatic"])

    assert selection.report["approved_count"] == 1
    assert selection.decisions[0]["decision"] == "approved"

    refined = apply_reviewed_operations(
        normalized,
        [proposal],
        selection.decisions,
        rgb_image=rgb,
        policy_config=cfg,
    )

    assert len(refined.records) == len(normalized.records) - 1
    generic = [
        operation
        for operation in refined.operations
        if operation["operation_type"] == "merge"
    ]
    assert len(generic) == 1
    assert generic[0]["decision_source"] == "automatic_selective_v1"
    assert generic[0]["evidence"]["automatic_policy_replay"] == {
        "policy_version": "automatic_selective_v1",
        "evidence_recomputed": True,
        "decision": "approved",
    }

    changed_rgb = rgb.copy()
    second_instance = int(normalized.records[1]["instance_id"])
    changed_rgb[normalized.instance_map == second_instance] = 255
    with pytest.raises(ValueError, match="replay-time evidence gates"):
        apply_reviewed_operations(
            normalized,
            [proposal],
            selection.decisions,
            rgb_image=changed_rgb,
            policy_config=cfg,
        )


def test_human_decision_does_not_masquerade_as_automatic() -> None:
    normalized, cfg, proposal, rgb = _automatic_merge_case()
    weak_rgb = rgb.copy()
    weak_rgb[normalized.instance_map == int(normalized.records[1]["instance_id"])] = 255

    # A real reviewer may override appearance heuristics; the class-policy and
    # topology invariants are still replay-checked by the common engine.
    refined = apply_reviewed_operations(
        normalized,
        [proposal],
        {proposal["proposal_id"]: "approved"},
        rgb_image=weak_rgb,
        policy_config=cfg,
    )
    assert any(operation["operation_type"] == "merge" for operation in refined.operations)

