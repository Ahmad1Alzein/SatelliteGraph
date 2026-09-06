from __future__ import annotations

import json

import numpy as np
import pytest

from src.refinement.evidence import ADJACENCY_DEFINITION
from src.refinement.unknown_cleanup import evaluate_unknown_cleanup_candidates


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
def cleanup_config() -> dict:
    return {
        "enabled": True,
        "auto_apply": True,
        "maximum_segment_area_pixels": 4,
        "allowed_target_classes": [
            "bareland",
            "rangeland",
            "developed space",
            "tree",
            "water",
        ],
        "minimum_mean_target_probability": 0.60,
        "minimum_probability_margin": 0.20,
        "minimum_dominant_pixel_fraction": 0.90,
        "minimum_shared_boundary_fraction": 0.50,
        "minimum_weak_color_boundary_score": 0.70,
        "minimum_region_mean_color_similarity": 0.55,
        "require_not_tile_boundary": True,
    }


def _record(instance_id: int, class_name: str) -> dict:
    return {
        "instance_id": instance_id,
        "entity_id": f"entity-{instance_id}",
        "label": {"id": CLASS_IDS[class_name], "name": class_name},
    }


def _pair(
    first: int,
    second: int,
    *,
    shared: int = 4,
    weak_boundary: float = 0.90,
    mean_similarity: float = 0.90,
) -> dict:
    return {
        "source_instance_id": min(first, second),
        "target_instance_id": max(first, second),
        "shared_boundary_side_count": shared,
        "weak_color_boundary_score": weak_boundary,
        "region_mean_color_similarity": mean_similarity,
        "adjacency_definition": ADJACENCY_DEFINITION,
    }


def _probabilities(
    shape: tuple[int, int],
    source_mask: np.ndarray,
    *,
    target_class: str = "rangeland",
    target_probability: float = 0.80,
    runner_up_probability: float = 0.10,
) -> np.ndarray:
    values = np.zeros((8, *shape), dtype=np.float32)
    target_id = CLASS_IDS[target_class]
    runner_up_id = 0 if target_id != 0 else 1
    values[target_id, source_mask] = target_probability
    values[runner_up_id, source_mask] = runner_up_probability
    return values


def _enclosed_case(
    *,
    target_class: str = "rangeland",
) -> tuple[np.ndarray, list[dict], list[dict], np.ndarray]:
    instance_map = np.asarray(
        [
            [2, 2, 2],
            [2, 1, 2],
            [2, 2, 2],
        ],
        dtype=np.int32,
    )
    records = [_record(1, "unknown"), _record(2, target_class)]
    pairs = [_pair(1, 2)]
    probabilities = _probabilities(
        instance_map.shape,
        instance_map == 1,
        target_class=target_class,
    )
    return instance_map, records, pairs, probabilities


def test_valid_micro_unknown_candidate_is_json_safe_and_records_actual_gates(
    cleanup_config: dict,
) -> None:
    instance_map, records, pairs, probabilities = _enclosed_case()

    candidates = evaluate_unknown_cleanup_candidates(
        records, instance_map, pairs, probabilities, cleanup_config
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["eligible"] is True
    assert candidate["eligible_for_auto_apply"] is True
    assert candidate["reasons"] == []
    assert candidate["source_instance_id"] == 1
    assert candidate["target_instance_id"] == 2
    assert candidate["target_class_name"] == "rangeland"
    assert candidate["evidence"][
        "selected_target_shared_boundary_fraction_of_source_perimeter"
    ] == pytest.approx(1.0)
    assert candidate["evidence"]["mean_target_class_probability_on_source"] == pytest.approx(
        0.80
    )
    assert candidate["evidence"]["target_class_probability_margin_on_source"] == pytest.approx(
        0.70
    )
    assert candidate["evidence"]["dominant_target_pixel_fraction_on_source"] == 1.0
    assert candidate["evidence"]["selected_target_weak_color_boundary_score"] == 0.90
    assert candidate["evidence"]["selected_target_region_mean_color_similarity"] == 0.90
    assert candidate["thresholds"] == {
        key: cleanup_config[key]
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
    json.dumps(candidates, allow_nan=False)


def test_ambiguous_oem_mean_class_abstains(cleanup_config: dict) -> None:
    instance_map, records, pairs, probabilities = _enclosed_case()
    source = instance_map == 1
    probabilities[:, source] = 0.0
    probabilities[CLASS_IDS["rangeland"], source] = 0.70
    probabilities[CLASS_IDS["developed space"], source] = 0.70

    candidate = evaluate_unknown_cleanup_candidates(
        records, instance_map, pairs, probabilities, cleanup_config
    )[0]

    assert candidate["eligible"] is False
    assert "dominant_mean_class_ambiguous" in candidate["reasons"]
    assert "probability_margin_below_minimum" in candidate["reasons"]
    assert "dominant_pixel_fraction_below_minimum" in candidate["reasons"]
    assert candidate["evidence"]["dominant_mean_class_tie_count"] == 2


@pytest.mark.parametrize("protected_class", ["road", "agriculture land", "buildings"])
def test_protected_target_class_never_becomes_eligible(
    cleanup_config: dict,
    protected_class: str,
) -> None:
    instance_map, records, pairs, probabilities = _enclosed_case(
        target_class=protected_class
    )

    candidate = evaluate_unknown_cleanup_candidates(
        records, instance_map, pairs, probabilities, cleanup_config
    )[0]

    assert candidate["target_class_name"] == protected_class
    assert candidate["eligible"] is False
    assert "target_class_not_allowed" in candidate["reasons"]


def test_tile_border_unknown_abstains(cleanup_config: dict) -> None:
    instance_map = np.asarray(
        [
            [2, 1, 2],
            [2, 2, 2],
            [2, 2, 2],
        ],
        dtype=np.int32,
    )
    records = [_record(1, "unknown"), _record(2, "rangeland")]
    probabilities = _probabilities(instance_map.shape, instance_map == 1)

    candidate = evaluate_unknown_cleanup_candidates(
        records,
        instance_map,
        [_pair(1, 2, shared=3)],
        probabilities,
        cleanup_config,
    )[0]

    assert candidate["eligible"] is False
    assert candidate["evidence"]["source_touches_tile_boundary"] is True
    assert "source_touches_tile_boundary" in candidate["reasons"]


def test_source_above_micro_area_limit_abstains(cleanup_config: dict) -> None:
    instance_map = np.full((5, 5), 2, dtype=np.int32)
    instance_map[1:3, 1:4] = 1
    records = [_record(1, "unknown"), _record(2, "rangeland")]
    probabilities = _probabilities(instance_map.shape, instance_map == 1)

    candidate = evaluate_unknown_cleanup_candidates(
        records,
        instance_map,
        [_pair(1, 2, shared=10)],
        probabilities,
        cleanup_config,
    )[0]

    assert candidate["evidence"]["source_area_pixels"] == 6
    assert candidate["eligible"] is False
    assert "source_area_above_maximum" in candidate["reasons"]


@pytest.mark.parametrize(
    ("changed", "reason"),
    [
        ({"target_probability": 0.59}, "mean_target_probability_below_minimum"),
        (
            {"target_probability": 0.60, "runner_up_probability": 0.45},
            "probability_margin_below_minimum",
        ),
        ({"shared": 1}, "shared_boundary_fraction_below_minimum"),
        ({"weak_boundary": 0.69}, "weak_color_boundary_score_below_minimum"),
        (
            {"mean_similarity": 0.54},
            "region_mean_color_similarity_below_minimum",
        ),
    ],
)
def test_low_evidence_abstains_for_each_scalar_gate(
    cleanup_config: dict,
    changed: dict,
    reason: str,
) -> None:
    instance_map, records, _, _ = _enclosed_case()
    probability_options = {
        key: changed[key]
        for key in ("target_probability", "runner_up_probability")
        if key in changed
    }
    probabilities = _probabilities(
        instance_map.shape,
        instance_map == 1,
        **probability_options,
    )
    pair_options = {
        key: changed[key]
        for key in ("shared", "weak_boundary", "mean_similarity")
        if key in changed
    }

    candidate = evaluate_unknown_cleanup_candidates(
        records,
        instance_map,
        [_pair(1, 2, **pair_options)],
        probabilities,
        cleanup_config,
    )[0]

    assert candidate["eligible"] is False
    assert reason in candidate["reasons"]


def test_low_dominant_pixel_fraction_abstains(cleanup_config: dict) -> None:
    instance_map = np.asarray(
        [
            [2, 2, 2, 2],
            [2, 1, 1, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
        ],
        dtype=np.int32,
    )
    records = [_record(1, "unknown"), _record(2, "rangeland")]
    probabilities = np.zeros((8, *instance_map.shape), dtype=np.float32)
    source_columns = np.argwhere(instance_map == 1)
    first, second = tuple(source_columns[0]), tuple(source_columns[1])
    probabilities[CLASS_IDS["rangeland"], first[0], first[1]] = 0.90
    probabilities[CLASS_IDS["rangeland"], second[0], second[1]] = 0.60
    probabilities[CLASS_IDS["developed space"], second[0], second[1]] = 0.61

    candidate = evaluate_unknown_cleanup_candidates(
        records,
        instance_map,
        [_pair(1, 2, shared=6)],
        probabilities,
        cleanup_config,
    )[0]

    assert candidate["evidence"]["dominant_mean_class_id"] == CLASS_IDS["rangeland"]
    assert candidate["evidence"]["dominant_target_pixel_fraction_on_source"] == 0.5
    assert candidate["eligible"] is False
    assert "dominant_pixel_fraction_below_minimum" in candidate["reasons"]


def test_target_selection_is_deterministic_by_shared_sides_boundary_then_id(
    cleanup_config: dict,
) -> None:
    instance_map = np.asarray(
        [
            [2, 2, 2],
            [2, 1, 3],
            [3, 3, 3],
        ],
        dtype=np.int32,
    )
    records = [
        _record(1, "unknown"),
        _record(2, "rangeland"),
        _record(3, "rangeland"),
    ]
    probabilities = _probabilities(instance_map.shape, instance_map == 1)
    weaker_low_id = _pair(1, 2, shared=2, weak_boundary=0.80)
    stronger_high_id = _pair(1, 3, shared=2, weak_boundary=0.90)

    by_boundary = evaluate_unknown_cleanup_candidates(
        records,
        instance_map,
        [stronger_high_id, weaker_low_id],
        probabilities,
        cleanup_config,
    )[0]
    tied_by_id = evaluate_unknown_cleanup_candidates(
        records,
        instance_map,
        [
            _pair(1, 3, shared=2, weak_boundary=0.90),
            _pair(1, 2, shared=2, weak_boundary=0.90),
        ],
        probabilities,
        cleanup_config,
    )[0]

    assert by_boundary["target_instance_id"] == 3
    assert tied_by_id["target_instance_id"] == 2
    assert by_boundary["evidence"]["matching_dominant_class_target_count"] == 2


def test_all_unknown_sources_are_reported_even_when_probability_data_is_missing(
    cleanup_config: dict,
) -> None:
    instance_map, records, pairs, _ = _enclosed_case()

    candidate = evaluate_unknown_cleanup_candidates(
        records, instance_map, pairs, None, cleanup_config
    )[0]

    assert candidate["source_instance_id"] == 1
    assert candidate["target_instance_id"] is None
    assert candidate["eligible"] is False
    assert candidate["reasons"] == ["semantic_probabilities_unavailable"]
