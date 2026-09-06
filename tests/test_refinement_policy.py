from __future__ import annotations

import copy

import numpy as np
import pytest

from src.refinement.policy import (
    classify_hard_vetoes,
    evaluate_absorb_candidate,
    evaluate_merge_candidate,
    orient_absorb_evidence,
)
from src.refinement.schema import make_proposal
from src.refinement.topology import (
    apply_reviewed_operations,
    normalize_4_connected_components,
)


CLASS_IDS = {
    "bareland": 0,
    "rangeland": 1,
    "developed space": 2,
    "road": 3,
    "tree": 4,
    "water": 5,
    "agriculture land": 6,
    "buildings": 7,
    "unknown": 8,
}


@pytest.fixture
def policy_config() -> dict:
    ontology = {
        name: {"allow_merge": name != "unknown"}
        for name in CLASS_IDS
    }
    return {
        "ontology": ontology,
        "operations": {
            "merge": {
                "enabled": True,
                "score_threshold": 0.98,
                "minimum_shared_boundary_pixels": 8,
                "minimum_shared_boundary_fraction": 0.15,
            },
            "absorb": {
                "enabled": True,
                "score_threshold": 0.98,
                "maximum_segment_area_pixels": 40,
                "score_margin_threshold": 0.25,
                "minimum_shared_boundary_fraction": 0.80,
            },
        },
    }


def record(instance_id: int, class_name: str, area: int = 20) -> dict:
    return {
        "instance_id": instance_id,
        "area_pixels": area,
        "label": {"id": CLASS_IDS[class_name], "name": class_name},
    }


def pair_evidence(
    source: int = 1,
    target: int = 2,
    *,
    source_area: int = 20,
    target_area: int = 200,
    source_fraction: float = 0.95,
    target_fraction: float = 0.20,
    shared: int = 20,
) -> dict:
    return {
        "source_instance_id": source,
        "target_instance_id": target,
        "adjacency_definition": "shared_pixel_edge_4_connected",
        "shared_boundary_side_count": shared,
        "source_area_pixels": source_area,
        "target_area_pixels": target_area,
        "source_perimeter_pixel_sides": 21,
        "target_perimeter_pixel_sides": 100,
        "source_shared_boundary_fraction": source_fraction,
        "target_shared_boundary_fraction": target_fraction,
        "minimum_perimeter_shared_boundary_fraction": min(
            source_fraction, target_fraction
        ),
        "best_to_runner_up_margin": 0.50,
        "weak_color_boundary_score": 0.99,
        "region_mean_color_similarity": 0.99,
    }


def test_orient_absorb_evidence_uses_the_actual_candidate_side() -> None:
    original = pair_evidence(
        source_area=11_828,
        target_area=8,
        source_fraction=0.02,
        target_fraction=1.0,
    )

    oriented = orient_absorb_evidence(original, 2, 1)

    assert oriented["absorb_orientation"] == "canonical_reverse"
    assert oriented["candidate_instance_id"] == 2
    assert oriented["absorb_target_instance_id"] == 1
    assert oriented["candidate_area_pixels"] == 8
    assert oriented["absorb_target_area_pixels"] == 11_828
    assert oriented["candidate_shared_boundary_fraction"] == 1.0
    assert oriented["absorb_target_shared_boundary_fraction"] == 0.02
    assert original.get("candidate_area_pixels") is None


@pytest.mark.parametrize(
    ("class_name", "expected_veto"),
    [
        ("road", "road_merge_requires_junction_topology"),
        ("agriculture land", "agriculture_merge_may_erase_field_boundary"),
        ("unknown", "unknown_merge_must_abstain"),
    ],
)
def test_unsafe_merge_classes_are_hard_vetoed(
    policy_config: dict, class_name: str, expected_veto: str
) -> None:
    first, second = record(1, class_name), record(2, class_name)

    result = evaluate_merge_candidate(
        policy_config,
        first,
        second,
        pair_evidence(),
        score=1.0,
    )

    assert result.eligible is False
    assert result.reasons == ("hard_policy_veto",)
    assert expected_veto in result.hard_vetoes


def test_merge_requires_score_and_meaningful_contact(policy_config: dict) -> None:
    first, second = record(1, "water"), record(2, "water")
    good = pair_evidence(source_fraction=0.20, target_fraction=0.20)

    accepted = evaluate_merge_candidate(
        policy_config, first, second, good, score=0.99
    )
    below_score = evaluate_merge_candidate(
        policy_config, first, second, good, score=0.97
    )
    weak_contact = evaluate_merge_candidate(
        policy_config,
        first,
        second,
        pair_evidence(
            shared=7,
            source_fraction=0.10,
            target_fraction=0.10,
        ),
        score=0.99,
    )

    assert accepted.eligible is True
    assert accepted.reasons == ()
    assert below_score.eligible is False
    assert "score_below_configured_threshold" in below_score.reasons
    assert weak_contact.eligible is False
    assert "shared_boundary_below_configured_minimum" in weak_contact.reasons
    assert (
        "shared_boundary_fraction_below_configured_minimum"
        in weak_contact.reasons
    )


@pytest.mark.parametrize(
    "source_class", ["road", "agriculture land", "tree", "buildings"]
)
def test_protected_objects_cannot_be_cross_class_absorbed_by_size(
    policy_config: dict, source_class: str
) -> None:
    source = record(1, source_class, area=1)
    target = record(2, "rangeland", area=1_000)
    probabilities, mask = supported_probabilities(target["label"]["id"])

    result = evaluate_absorb_candidate(
        policy_config,
        source,
        target,
        pair_evidence(source_area=1, source_fraction=1.0),
        score=1.0,
        semantic_probabilities=probabilities,
        candidate_mask=mask,
    )

    assert result.eligible is False
    assert result.reasons == ("hard_policy_veto",)
    assert any(value.startswith("cross_class_absorb_forbidden_for_") for value in result.hard_vetoes)


def test_unknown_cross_class_absorb_requires_strong_probability_and_enclosure(
    policy_config: dict,
) -> None:
    source = record(1, "unknown", area=4)
    target = record(2, "rangeland", area=1_000)
    target_id = target["label"]["id"]
    probabilities, mask = supported_probabilities(
        target_id, target_probability=0.92, runner_up_probability=0.12
    )
    evidence = pair_evidence(source_area=4, source_fraction=0.95)

    accepted = evaluate_absorb_candidate(
        policy_config,
        source,
        target,
        evidence,
        score=0.99,
        semantic_probabilities=probabilities,
        candidate_mask=mask,
    )
    low_probability, _ = supported_probabilities(
        target_id, target_probability=0.89, runner_up_probability=0.10
    )
    rejected_probability = evaluate_absorb_candidate(
        policy_config,
        source,
        target,
        evidence,
        score=0.99,
        semantic_probabilities=low_probability,
        candidate_mask=mask,
    )
    low_margin, _ = supported_probabilities(
        target_id, target_probability=0.92, runner_up_probability=0.75
    )
    rejected_margin = evaluate_absorb_candidate(
        policy_config,
        source,
        target,
        evidence,
        score=0.99,
        semantic_probabilities=low_margin,
        candidate_mask=mask,
    )
    rejected_boundary = evaluate_absorb_candidate(
        policy_config,
        source,
        target,
        pair_evidence(source_area=4, source_fraction=0.89),
        score=0.99,
        semantic_probabilities=probabilities,
        candidate_mask=mask,
    )

    assert accepted.eligible is True
    assert accepted.evidence["mean_target_class_probability_on_source"] == pytest.approx(0.92)
    assert accepted.evidence["target_class_probability_margin_on_source"] == pytest.approx(0.80)
    assert (
        "target_class_probability_below_safety_minimum"
        in rejected_probability.reasons
    )
    assert (
        "target_class_probability_margin_below_safety_minimum"
        in rejected_margin.reasons
    )
    assert (
        "cross_class_boundary_fraction_below_safety_minimum"
        in rejected_boundary.reasons
    )


def test_hard_veto_classification_is_pure(policy_config: dict) -> None:
    config_before = copy.deepcopy(policy_config)
    source = record(1, "road")
    target = record(2, "rangeland")

    vetoes = classify_hard_vetoes(
        "absorb", source, target, policy_config
    )

    assert "cross_class_absorb_forbidden_for_road" in vetoes
    assert policy_config == config_before


def test_replay_rechecks_hard_class_policy(policy_config: dict) -> None:
    instance_map = np.asarray([[1, 2]], dtype=np.int32)
    normalized = normalize_4_connected_components(
        instance_map,
        np.asarray([[3, 3]], dtype=np.uint8),
        np.asarray([[0.9, 0.9]], dtype=np.float32),
        np.asarray([[2, 2]], dtype=np.uint8),
        np.asarray([[True, True]], dtype=bool),
        [
            {
                "instance_id": instance_id,
                "segment_id": f"road_{instance_id}",
                "source_tile_id": "policy_tile",
                "label": {"id": 3, "name": "road"},
            }
            for instance_id in (1, 2)
        ],
        tile_id="policy_tile",
    )
    proposal = make_proposal(
        "policy_tile",
        "merge",
        [record["entity_id"] for record in normalized.records],
        parameters={"review_only": True},
        evidence={},
        decision="pending",
    )

    with pytest.raises(ValueError, match="violates class policy"):
        apply_reviewed_operations(
            normalized,
            [proposal],
            decisions={proposal["proposal_id"]: "approved"},
            policy_config=policy_config,
        )


def supported_probabilities(
    target_class_id: int,
    *,
    target_probability: float = 0.95,
    runner_up_probability: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.zeros((9, 2, 2), dtype=np.float32)
    mask = np.zeros((2, 2), dtype=bool)
    mask[0, 0] = True
    probabilities[target_class_id, mask] = target_probability
    runner_up_id = 0 if target_class_id != 0 else 1
    probabilities[runner_up_id, mask] = runner_up_probability
    return probabilities, mask
