from __future__ import annotations

import copy
import json

import pytest

from src.refinement.automatic import (
    AUTOMATIC_DECISION_SOURCE,
    default_automatic_config,
    evaluate_automatic_proposal,
    select_automatic_config,
    select_automatic_decisions,
    validate_automatic_config,
)
from src.refinement.schema import make_proposal, validate_decision


def merge_proposal(
    source_a: str,
    source_b: str,
    *,
    tile_id: str = "tile-1",
    class_name: str = "water",
    score: float = 1.0,
    shared_fraction: float = 0.60,
    weak_boundary: float = 0.99,
    similarity: float = 0.99,
    source_boundary: bool = False,
    target_boundary: bool = False,
    decision: str = "pending",
) -> dict:
    return make_proposal(
        tile_id,
        "merge",
        [source_a, source_b],
        parameters={
            "class_name": class_name,
            "review_only": True,
            "score_kind": "uncalibrated_review_priority",
        },
        evidence={
            "policy_gate_passed": True,
            "hard_vetoes": [],
            "semantic_class_match": True,
            "adjacency_definition": "shared_pixel_edge_4_connected",
            "shared_boundary_side_count": 20,
            "minimum_perimeter_shared_boundary_fraction": shared_fraction,
            "weak_color_boundary_score": weak_boundary,
            "region_mean_color_similarity": similarity,
            "source_touches_tile_boundary": source_boundary,
            "target_touches_tile_boundary": target_boundary,
        },
        score=score,
        decision=decision,
    )


def absorb_proposal(
    source: str,
    target: str,
    *,
    tile_id: str = "tile-1",
    class_name: str = "water",
    target_class_name: str | None = None,
    score: float = 1.0,
    source_area: int = 5,
    shared_fraction: float = 0.90,
    margin: float = 0.50,
    weak_boundary: float = 0.99,
    similarity: float = 0.99,
    source_boundary: bool = False,
    automatic_unknown_cleanup: bool = False,
    decision: str = "pending",
) -> dict:
    target_class = target_class_name or class_name
    same_class = class_name == target_class
    return make_proposal(
        tile_id,
        "absorb",
        [source],
        target_entity_id=target,
        parameters={
            "class_name": class_name,
            "target_class_name": target_class,
            "automatic_unknown_cleanup": automatic_unknown_cleanup,
            "review_only": not automatic_unknown_cleanup,
            "same_class_required": same_class,
            "cross_class_relabel": not same_class,
            "score_kind": "uncalibrated_review_priority",
        },
        evidence={
            "policy_gate_passed": True,
            "hard_vetoes": [],
            "adjacency_definition": "shared_pixel_edge_4_connected",
            "candidate_record_area_pixels": source_area,
            "shared_boundary_side_count": 12,
            "candidate_shared_boundary_fraction": shared_fraction,
            "best_to_runner_up_boundary_fraction_margin": margin,
            "weak_color_boundary_score": weak_boundary,
            "region_mean_color_similarity": similarity,
            "source_touches_tile_boundary": source_boundary,
        },
        score=score,
        decision=decision,
    )


def semantic_split_proposal() -> dict:
    return make_proposal(
        "tile-1",
        "semantic_split",
        ["parent"],
        parameters={
            "partition_sha256": "a" * 64,
            "partition_shape": [4, 4],
            "partition_dtype": "int32",
            "child_count": 2,
            "child_label_order": "ascending_contiguous_from_1",
            "minimum_child_area_pixels": 2,
            "review_only": True,
            "automatic": False,
        },
        evidence={"score_only_is_not_enough": True},
        score=1.0,
        decision="pending",
    )


def test_default_config_is_independent_and_same_class_is_an_invariant() -> None:
    first = default_automatic_config()
    second = default_automatic_config()
    first["merge"]["minimum_score"] = 0.5
    assert second["merge"]["minimum_score"] == 0.99

    invalid = default_automatic_config()
    invalid["absorb"]["require_same_class"] = False
    with pytest.raises(ValueError, match="must remain true"):
        validate_automatic_config(invalid)

    selected = select_automatic_config(
        {"protected_classes": ["tree", "road"], "merge": {"minimum_score": 1.0}}
    )
    assert selected["protected_classes"] == ["road", "tree"]
    assert selected["merge"]["minimum_score"] == 1.0
    assert selected["absorb"]["minimum_score"] == 0.99


def test_score_alone_never_approves_and_thresholds_are_auditable() -> None:
    proposal = merge_proposal("a", "b", score=1.0, weak_boundary=0.40)

    evaluation = evaluate_automatic_proposal(proposal)
    selected = select_automatic_decisions([proposal])

    assert evaluation.eligible is False
    assert "weak_color_boundary_score_below_automatic_minimum" in evaluation.reasons
    assert selected.decisions[0]["decision"] == "uncertain"
    assert selected.decisions[0]["decision_source"] == AUTOMATIC_DECISION_SOURCE
    validate_decision(selected.decisions[0])

    reason = json.loads(selected.decisions[0]["reason"])
    assert reason["threshold_snapshot"] == evaluation.threshold_snapshot
    assert reason["threshold_snapshot"]["merge"][
        "minimum_weak_color_boundary_score"
    ] == 0.95
    assert selected.report["threshold_snapshots"]["merge"] == evaluation.threshold_snapshot


@pytest.mark.parametrize(
    "proposal",
    [
        merge_proposal("road-a", "road-b", class_name="road"),
        merge_proposal("edge-a", "edge-b", source_boundary=True),
        absorb_proposal(
            "water-source",
            "range-target",
            class_name="water",
            target_class_name="rangeland",
        ),
        absorb_proposal(
            "building-fragment", "building", class_name="buildings"
        ),
    ],
    ids=("protected-merge", "tile-boundary", "cross-class", "protected-absorb"),
)
def test_protected_cross_class_and_tile_boundary_proposals_abstain(
    proposal: dict,
) -> None:
    result = select_automatic_decisions([proposal])

    assert result.decisions[0]["decision"] == "uncertain"
    assert result.report["approved_count"] == 0
    assert result.report["abstained_count"] == 1


def test_missing_tile_boundary_evidence_abstains_instead_of_guessing() -> None:
    proposal = merge_proposal("a", "b")
    del proposal["evidence"]["target_touches_tile_boundary"]

    result = evaluate_automatic_proposal(proposal)

    assert result.eligible is False
    assert "tile_boundary_evidence_unavailable" in result.reasons


def test_semantic_split_is_never_automatic() -> None:
    proposal = semantic_split_proposal()

    evaluation = evaluate_automatic_proposal(proposal)
    selected = select_automatic_decisions([proposal])

    assert evaluation.eligible is False
    assert evaluation.reasons == ("semantic_split_requires_human_review",)
    assert selected.decisions[0]["decision"] == "uncertain"
    assert "semantic_split_requires_human_review" in selected.decisions[0]["reason"]


def test_conflicts_are_deterministic_and_absorb_targets_are_reusable() -> None:
    # Both high-contact micro sources may reuse one target.  The weaker proposal
    # for source-a loses because a source is exclusive in topology replay.
    first = absorb_proposal("source-a", "common-target", shared_fraction=0.95)
    second = absorb_proposal("source-b", "common-target", shared_fraction=0.93)
    conflicting = absorb_proposal("source-a", "other-target", shared_fraction=0.90)

    forward = select_automatic_decisions([conflicting, second, first])
    reverse = select_automatic_decisions([first, second, conflicting])

    assert forward.decisions == reverse.decisions
    assert forward.report == reverse.report
    by_id = {row["proposal_id"]: row for row in forward.decisions}
    assert by_id[first["proposal_id"]]["decision"] == "approved"
    assert by_id[second["proposal_id"]]["decision"] == "approved"
    assert by_id[conflicting["proposal_id"]]["decision"] == "uncertain"
    assert first["proposal_id"] in by_id[conflicting["proposal_id"]]["reason"]
    assert forward.report["approved_count"] == 2


def test_merge_conflicts_with_absorb_target_but_two_absorbs_do_not() -> None:
    first = absorb_proposal("small-a", "target", shared_fraction=0.95)
    second = absorb_proposal("small-b", "target", shared_fraction=0.94)
    merge = merge_proposal("target", "neighbor")

    result = select_automatic_decisions([merge, second, first])
    decisions = {row["proposal_id"]: row["decision"] for row in result.decisions}

    assert decisions[first["proposal_id"]] == "approved"
    assert decisions[second["proposal_id"]] == "approved"
    assert decisions[merge["proposal_id"]] == "uncertain"


def test_per_tile_cap_and_embedded_operations_are_not_overridden() -> None:
    config = {"maximum_selected_operations_per_tile": 1}
    high = absorb_proposal("a", "target-a", shared_fraction=0.95)
    lower = absorb_proposal("b", "target-b", shared_fraction=0.90)
    other_tile = absorb_proposal(
        "c", "target-c", tile_id="tile-2", shared_fraction=0.90
    )
    embedded = absorb_proposal(
        "unknown",
        "known",
        class_name="unknown",
        target_class_name="water",
        automatic_unknown_cleanup=True,
        decision="auto_apply",
    )
    connectivity = make_proposal(
        "tile-1",
        "split_components",
        ["parent"],
        parameters={},
        evidence={},
        score=1.0,
        decision="auto_apply",
    )

    result = select_automatic_decisions(
        [lower, connectivity, other_tile, embedded, high], config
    )
    decisions = {row["proposal_id"]: row["decision"] for row in result.decisions}

    assert decisions[high["proposal_id"]] == "approved"
    assert decisions[lower["proposal_id"]] == "uncertain"
    assert decisions[other_tile["proposal_id"]] == "approved"
    assert embedded["proposal_id"] not in decisions
    assert connectivity["proposal_id"] not in decisions
    assert result.report["passthrough_count"] == 2
    assert result.report["selected_count_by_tile"] == {"tile-1": 1, "tile-2": 1}


def test_embedded_auto_apply_participants_are_reserved_before_selection() -> None:
    embedded = absorb_proposal(
        "unknown-source",
        "known-target",
        class_name="unknown",
        target_class_name="water",
        automatic_unknown_cleanup=True,
        decision="auto_apply",
    )
    overlapping = merge_proposal("known-target", "water-neighbor")

    result = select_automatic_decisions([overlapping, embedded])

    assert len(result.decisions) == 1
    assert result.decisions[0]["proposal_id"] == overlapping["proposal_id"]
    assert result.decisions[0]["decision"] == "uncertain"
    assert "conflicts_with_selected_proposal" in result.decisions[0]["reason"]


def test_evaluation_and_selection_do_not_mutate_inputs() -> None:
    proposal = absorb_proposal("source", "target")
    config = default_automatic_config()
    before_proposal = copy.deepcopy(proposal)
    before_config = copy.deepcopy(config)

    assert evaluate_automatic_proposal(proposal, config).eligible is True
    assert select_automatic_decisions([proposal], config).report["approved_count"] == 1

    assert proposal == before_proposal
    assert config == before_config
