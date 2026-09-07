from __future__ import annotations

import json

import numpy as np
import pytest

from src.refinement.contextual_cleanup import (
    evaluate_contextual_micro_cleanup,
)
from src.refinement.evidence import pair_evidence


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


def test_supported_energy_extension_requires_each_independent_gate():
    instance_map = np.asarray([[2, 2, 2], [2, 1, 3], [3, 3, 3]], dtype=np.int32)
    records = _records(instance_map, {1: "unknown", 2: "rangeland", 3: "tree"})
    probabilities = _probabilities(instance_map, 1, {"rangeland": .35, "tree": .20, "buildings": .45})
    pairs = _pairs(instance_map)
    old = evaluate_contextual_micro_cleanup(records, instance_map, pairs, probabilities, {})[0]
    assert old["action"] == "abstain"
    cfg = {"supported_maximum_energy": .50}
    new = evaluate_contextual_micro_cleanup(records, instance_map, pairs, probabilities, cfg)[0]
    assert new["action"] == "absorb"
    assert new["target_instance_id"] == 2
    assert new["evidence"]["selected_energy_limit"] == .50
    for key, value in (
        ("supported_minimum_probability", .36),
        ("supported_minimum_boundary_fraction", .51),
    ):
        rejected = evaluate_contextual_micro_cleanup(records, instance_map, pairs, probabilities, {**cfg, key: value})[0]
        assert rejected["action"] == "abstain"
    for pair in pairs:
        if pair["source_instance_id"] == 1 and pair["target_instance_id"] == 2:
            pair["weak_color_boundary_score"] = .49
    assert evaluate_contextual_micro_cleanup(records, instance_map, pairs, probabilities, cfg)[0]["action"] == "abstain"


@pytest.fixture
def cleanup_config() -> dict:
    return {
        "enabled": True,
        "maximum_segment_area_pixels": 40,
        "semantic_consensus_minimum_probability": 0.55,
        "semantic_consensus_minimum_margin": 0.15,
        "semantic_consensus_minimum_purity": 0.80,
    }


def _record(instance_id: int, class_name: str) -> dict:
    return {
        "instance_id": instance_id,
        "entity_id": f"entity-{instance_id}",
        "label": {"id": CLASS_IDS[class_name], "name": class_name},
    }


def _records(instance_map: np.ndarray, classes: dict[int, str]) -> list[dict]:
    assert set(np.unique(instance_map)) == set(classes)
    return [_record(instance_id, classes[instance_id]) for instance_id in sorted(classes)]


def _pairs(instance_map: np.ndarray) -> list[dict]:
    rgb = np.zeros((*instance_map.shape, 3), dtype=np.uint8)
    return pair_evidence(instance_map, rgb)


def _probabilities(
    instance_map: np.ndarray,
    source_id: int,
    values: dict[str, float],
) -> np.ndarray:
    probabilities = np.zeros((8, *instance_map.shape), dtype=np.float32)
    source = instance_map == source_id
    for class_name, probability in values.items():
        probabilities[CLASS_IDS[class_name], source] = probability
    return probabilities


def _enclosed_case(
    target_class: str = "rangeland",
) -> tuple[np.ndarray, list[dict], list[dict]]:
    instance_map = np.asarray(
        [
            [2, 2, 2],
            [2, 1, 2],
            [2, 2, 2],
        ],
        dtype=np.int32,
    )
    records = _records(instance_map, {1: "unknown", 2: target_class})
    return instance_map, records, _pairs(instance_map)


def test_semantic_consensus_records_exact_energy_and_is_json_safe(
    cleanup_config: dict,
) -> None:
    instance_map, records, pairs = _enclosed_case()
    probabilities = _probabilities(
        instance_map,
        1,
        {"rangeland": 0.80, "tree": 0.10},
    )

    assignments = evaluate_contextual_micro_cleanup(
        records, instance_map, pairs, probabilities, cleanup_config
    )

    assert len(assignments) == 1
    assignment = assignments[0]
    assert assignment["decision_tier"] == "semantic_consensus"
    assert assignment["action"] == "absorb"
    assert assignment["target"] == {
        "instance_id": 2,
        "entity_id": "entity-2",
        "class_id": CLASS_IDS["rangeland"],
        "class_name": "rangeland",
    }
    assert assignment["evidence"]["selected_energy"] == pytest.approx(0.09)
    assert assignment["evidence"]["energy_weights"] == {
        "semantic_probability": 0.45,
        "boundary_contact": 0.25,
        "weak_rgb_boundary": 0.20,
        "region_rgb_similarity": 0.10,
    }
    json.dumps(assignments, allow_nan=False)


def test_contextual_energy_selects_a_clear_minimum_and_abstains_on_a_tie(
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
    records = _records(
        instance_map,
        {1: "unknown", 2: "rangeland", 3: "tree"},
    )
    pairs = _pairs(instance_map)
    probabilities = _probabilities(
        instance_map,
        1,
        {"rangeland": 0.40, "tree": 0.50},
    )

    selected = evaluate_contextual_micro_cleanup(
        records, instance_map, pairs, probabilities, cleanup_config
    )[0]
    probabilities[:, instance_map == 1] = 0.0
    probabilities[CLASS_IDS["rangeland"], instance_map == 1] = 0.40
    probabilities[CLASS_IDS["tree"], instance_map == 1] = 0.40
    tied = evaluate_contextual_micro_cleanup(
        records, instance_map, pairs, probabilities, cleanup_config
    )[0]

    assert selected["decision_tier"] == "contextual"
    assert selected["target_instance_id"] == 3
    assert tied["decision_tier"] == "abstain"
    assert tied["target_instance_id"] is None
    assert "best_to_runner_up_energy_margin_below_minimum" in tied["reasons"]
    target_rows = selected["evidence"]["target_candidates"]
    assert [row["target_instance_id"] for row in target_rows] == [2, 3]
    assert target_rows[0]["source_boundary_fraction"] == 0.5
    assert target_rows[1]["source_boundary_fraction"] == 0.5


def test_tile_border_source_always_abstains(cleanup_config: dict) -> None:
    instance_map = np.asarray(
        [
            [2, 1, 2],
            [2, 2, 2],
            [2, 2, 2],
        ],
        dtype=np.int32,
    )
    records = _records(instance_map, {1: "unknown", 2: "rangeland"})
    probabilities = _probabilities(instance_map, 1, {"rangeland": 0.90})

    assignment = evaluate_contextual_micro_cleanup(
        records, instance_map, _pairs(instance_map), probabilities, cleanup_config
    )[0]

    assert assignment["decision_tier"] == "abstain"
    assert assignment["action"] == "abstain"
    assert assignment["target"] is None
    assert "source_touches_tile_boundary" in assignment["reasons"]


@pytest.mark.parametrize(
    ("object_class", "probability", "runner_up"),
    [
        ("tree", 0.60, 0.20),
        ("buildings", 0.70, 0.10),
    ],
)
def test_strong_standalone_object_is_preserved_as_an_abstention(
    cleanup_config: dict,
    object_class: str,
    probability: float,
    runner_up: float,
) -> None:
    instance_map = np.full((8, 9), 2, dtype=np.int32)
    instance_map[2:6, 2:7] = 1  # 20-pixel interior object candidate.
    records = _records(instance_map, {1: "unknown", 2: "rangeland"})
    probabilities = _probabilities(
        instance_map,
        1,
        {object_class: probability, "rangeland": runner_up},
    )

    assignment = evaluate_contextual_micro_cleanup(
        records, instance_map, _pairs(instance_map), probabilities, cleanup_config
    )[0]

    assert assignment["decision_tier"] == "preserved_object"
    assert assignment["action"] == "preserve"
    assert assignment["target"] is None
    assert assignment["evidence"]["preserved_object_class"] == object_class
    assert any("strong_standalone" in reason for reason in assignment["reasons"])


def test_agriculture_requires_one_field_and_three_quarters_contact(
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
    records = _records(
        instance_map,
        {1: "unknown", 2: "agriculture land", 3: "rangeland"},
    )
    probabilities = _probabilities(
        instance_map,
        1,
        {"agriculture land": 0.70, "rangeland": 0.20},
    )

    assignment = evaluate_contextual_micro_cleanup(
        records, instance_map, _pairs(instance_map), probabilities, cleanup_config
    )[0]
    agriculture = next(
        row
        for row in assignment["evidence"]["target_candidates"]
        if row["target_class_name"] == "agriculture land"
    )

    assert agriculture["source_boundary_fraction"] == 0.5
    assert agriculture["eligible"] is False
    assert (
        "agriculture_boundary_fraction_below_minimum"
        in agriculture["policy_reasons"]
    )
    assert assignment["target_instance_id"] is None
    assert "selected_energy_above_automatic_maximum" in assignment["reasons"]


def test_building_requires_probability_and_a_unique_building_neighbor(
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
    records = _records(
        instance_map,
        {1: "unknown", 2: "buildings", 3: "rangeland"},
    )
    probabilities = _probabilities(
        instance_map,
        1,
        {"buildings": 0.34, "rangeland": 0.33},
    )

    assignment = evaluate_contextual_micro_cleanup(
        records, instance_map, _pairs(instance_map), probabilities, cleanup_config
    )[0]
    building = next(
        row
        for row in assignment["evidence"]["target_candidates"]
        if row["target_class_name"] == "buildings"
    )

    assert building["eligible"] is False
    assert "building_probability_below_minimum" in building["policy_reasons"]
    assert assignment["target_instance_id"] is None
    assert "selected_energy_above_automatic_maximum" in assignment["reasons"]


@pytest.mark.parametrize(
    ("values", "road_allowed"),
    [
        ({"road": 0.29, "tree": 0.40, "rangeland": 0.20}, True),
        (
            {
                "road": 0.29,
                "tree": 0.40,
                "rangeland": 0.35,
                "developed space": 0.30,
            },
            False,
        ),
    ],
)
def test_road_requires_probability_or_top_two_support(
    cleanup_config: dict,
    values: dict[str, float],
    road_allowed: bool,
) -> None:
    instance_map, records, pairs = _enclosed_case("road")
    probabilities = _probabilities(instance_map, 1, values)

    assignment = evaluate_contextual_micro_cleanup(
        records, instance_map, pairs, probabilities, cleanup_config
    )[0]
    road = assignment["evidence"]["target_candidates"][0]

    assert road["eligible"] is road_allowed
    if road_allowed:
        assert assignment["target_instance_id"] == 2
    else:
        assert assignment["decision_tier"] == "abstain"
        assert "road_probability_or_top_two_support_required" in assignment["reasons"]


def test_scope_is_ordered_unknown_components_no_larger_than_configured_limit() -> None:
    instance_map = np.asarray(
        [
            [4, 4, 4, 4, 4],
            [4, 3, 4, 1, 4],
            [4, 4, 4, 4, 4],
            [4, 2, 2, 4, 4],
            [4, 4, 4, 4, 4],
        ],
        dtype=np.int32,
    )
    records = _records(
        instance_map,
        {1: "unknown", 2: "unknown", 3: "unknown", 4: "rangeland"},
    )
    probabilities = np.zeros((8, *instance_map.shape), dtype=np.float32)
    probabilities[CLASS_IDS["rangeland"], instance_map != 4] = 0.80

    assignments = evaluate_contextual_micro_cleanup(
        records,
        instance_map,
        _pairs(instance_map),
        probabilities,
        {"maximum_segment_area_pixels": 1},
    )

    assert [item["source_instance_id"] for item in assignments] == [1, 3]


def test_missing_probabilities_and_disconnected_source_abstain_or_fail(
    cleanup_config: dict,
) -> None:
    instance_map, records, pairs = _enclosed_case()
    assignment = evaluate_contextual_micro_cleanup(
        records, instance_map, pairs, None, cleanup_config
    )[0]
    assert assignment["decision_tier"] == "abstain"
    assert assignment["reasons"] == ["semantic_probabilities_unavailable"]

    disconnected = np.asarray(
        [
            [2, 2, 2, 2],
            [2, 1, 2, 2],
            [2, 2, 1, 2],
            [2, 2, 2, 2],
        ],
        dtype=np.int32,
    )
    disconnected_records = _records(disconnected, {1: "unknown", 2: "rangeland"})
    with pytest.raises(ValueError, match="four-connected"):
        evaluate_contextual_micro_cleanup(
            disconnected_records,
            disconnected,
            _pairs(disconnected),
            _probabilities(disconnected, 1, {"rangeland": 0.80}),
            cleanup_config,
        )


def test_hard_area_ceiling_cannot_be_configured_above_40() -> None:
    instance_map, records, pairs = _enclosed_case()
    with pytest.raises(ValueError, match="cannot exceed 40"):
        evaluate_contextual_micro_cleanup(
            records,
            instance_map,
            pairs,
            _probabilities(instance_map, 1, {"rangeland": 0.80}),
            {"maximum_segment_area_pixels": 41},
        )
