"""Deterministic, class-aware decisions for contextual micro-unknown cleanup.

The evaluator in this module is pure: it does not mutate segmentation rasters
or construct refinement operations.  It accounts for every normalized,
four-connected ``unknown`` instance at or below the configured micro-area
limit and either selects one directly adjacent known target, preserves strong
standalone object evidence, or records an explicit abstention.

Selection minimizes a fixed, interpretable energy::

    0.45 * (1 - target probability)
  + 0.25 * (1 - source-boundary contact fraction)
  + 0.20 * (1 - weak RGB boundary score)
  + 0.10 * (1 - region RGB similarity)

The maximum supported micro area is intentionally capped at 40 pixels.  This
is a safety invariant, not a ranking parameter: larger uncertain regions must
remain available for semantic review.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .evidence import ADJACENCY_DEFINITION
from .schema import json_safe


UNKNOWN_CLASS_NAME = "unknown"
MAXIMUM_SAFE_MICRO_AREA_PIXELS = 40

_CLASS_NAMES = {
    0: "bareland",
    1: "rangeland",
    2: "developed space",
    3: "road",
    4: "tree",
    5: "water",
    6: "agriculture land",
    7: "buildings",
}
_ROAD = "road"
_TREE = "tree"
_AGRICULTURE = "agriculture land"
_BUILDINGS = "buildings"

_ENERGY_WEIGHTS = {
    "semantic_probability": 0.45,
    "boundary_contact": 0.25,
    "weak_rgb_boundary": 0.20,
    "region_rgb_similarity": 0.10,
}

_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "maximum_segment_area_pixels": MAXIMUM_SAFE_MICRO_AREA_PIXELS,
    "semantic_consensus_minimum_probability": 0.55,
    "semantic_consensus_minimum_margin": 0.15,
    "semantic_consensus_minimum_purity": 0.80,
    "maximum_selected_energy": 0.40,
    "supported_maximum_energy": 0.40,
    "supported_minimum_probability": 0.20,
    "supported_minimum_boundary_fraction": 0.25,
    "supported_minimum_boundary_similarity": 0.50,
    "supported_minimum_region_similarity": 0.50,
    "minimum_energy_margin": 0.02,
    "standalone_tree_minimum_area_pixels": 20,
    "standalone_tree_minimum_probability": 0.55,
    "standalone_tree_minimum_margin": 0.15,
    "standalone_tree_minimum_purity": 0.80,
    "standalone_building_minimum_area_pixels": 20,
    "standalone_building_minimum_probability": 0.65,
    "standalone_building_minimum_margin": 0.20,
    "standalone_building_minimum_purity": 0.90,
    "agriculture_minimum_boundary_fraction": 0.75,
    "building_minimum_probability": 0.35,
    "road_minimum_probability": 0.30,
}


def evaluate_contextual_micro_cleanup(
    records: Sequence[Mapping[str, Any]],
    instance_map: np.ndarray,
    pair_rows: Sequence[Mapping[str, Any]],
    probabilities: np.ndarray | None,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return ordered contextual assignments for normalized micro-unknowns.

    Only four-connected ``unknown`` instances whose raster area is no greater
    than ``maximum_segment_area_pixels`` are returned.  The configured maximum
    may be lowered but may never exceed the hard 40-pixel safety ceiling.

    Every returned item has one of four decision tiers:

    ``semantic_consensus``
        The selected adjacent target matches strong source semantic evidence.
    ``contextual``
        A target is selected by the fixed contextual energy.
    ``preserved_object``
        Strong standalone tree/building evidence is retained as its own object.
    ``abstain``
        No automatic edit is allowed, with explicit reasons.
    """

    cfg = _select_config(config)
    instances = _validate_instance_map(instance_map)
    by_instance, class_names = _record_index(records, instances)
    semantic = _validate_probabilities(probabilities, instances.shape)
    actual_contacts = _raster_contacts(instances)
    pair_index = _pair_index(pair_rows, by_instance, actual_contacts)

    assignments: list[dict[str, Any]] = []
    for source_id in sorted(by_instance):
        source_record = by_instance[source_id]
        source_class_id, source_class_name = _record_class(source_record)
        if source_class_name != UNKNOWN_CLASS_NAME:
            continue

        source_mask = instances == source_id
        area = int(source_mask.sum())
        if area > cfg["maximum_segment_area_pixels"]:
            continue
        if not _is_four_connected(source_mask):
            raise ValueError(
                f"unknown instance {source_id} must be four-connected before "
                "contextual cleanup"
            )

        perimeter = _mask_perimeter(source_mask)
        touches_border = _touches_tile_boundary(source_mask)
        probability_evidence = _probability_evidence(
            semantic, source_mask, class_names
        )
        adjacent_ids = sorted(
            neighbor
            for pair in actual_contacts
            if source_id in pair
            for neighbor in pair
            if neighbor != source_id
        )
        adjacent_ids = sorted(set(adjacent_ids))
        known_neighbor_ids = [
            neighbor
            for neighbor in adjacent_ids
            if _record_class(by_instance[neighbor])[1] != UNKNOWN_CLASS_NAME
        ]
        unknown_neighbor_count = len(adjacent_ids) - len(known_neighbor_ids)

        target_rows: list[dict[str, Any]] = []
        missing_pair_neighbors: list[int] = []
        for target_id in known_neighbor_ids:
            key = (min(source_id, target_id), max(source_id, target_id))
            pair = pair_index.get(key)
            if pair is None:
                missing_pair_neighbors.append(target_id)
                continue
            target_rows.append(
                _evaluate_target(
                    source_id=source_id,
                    target_id=target_id,
                    target_record=by_instance[target_id],
                    pair=pair,
                    source_perimeter=perimeter,
                    probability_evidence=probability_evidence,
                    known_neighbor_ids=known_neighbor_ids,
                    by_instance=by_instance,
                    cfg=cfg,
                )
            )
        target_rows.sort(key=lambda value: int(value["target_instance_id"]))

        top_class_name = probability_evidence["top_class_name"]
        matching_known_targets = [
            row
            for row in target_rows
            if row["target_class_name"] == top_class_name
        ]
        preserved_class = _preserved_object_class(
            area=area,
            probability_evidence=probability_evidence,
            matching_known_targets=matching_known_targets,
            cfg=cfg,
        )

        selected: dict[str, Any] | None = None
        best_candidate_energy: float | None = None
        runner_up_energy: float | None = None
        energy_margin: float | None = None
        decision_tier = "abstain"
        action = "abstain"
        reasons: list[str] = []
        if not cfg["enabled"]:
            reasons.append("contextual_micro_cleanup_disabled")
        if touches_border:
            reasons.append("source_touches_tile_boundary")
        if semantic is None:
            reasons.append("semantic_probabilities_unavailable")
        if missing_pair_neighbors:
            reasons.append("adjacent_pair_evidence_missing")

        if not reasons and preserved_class is not None:
            decision_tier = "preserved_object"
            action = "preserve"
            reasons.append(
                f"strong_standalone_{_reason_slug(preserved_class)}_evidence"
            )
        elif not reasons:
            eligible_targets = [row for row in target_rows if row["eligible"]]
            if eligible_targets:
                ranked_targets = sorted(
                    eligible_targets,
                    key=lambda value: (
                        float(value["energy"]),
                        int(value["target_instance_id"]),
                    ),
                )
                best = ranked_targets[0]
                best_candidate_energy = float(best["energy"])
                if len(ranked_targets) > 1:
                    runner_up_energy = float(ranked_targets[1]["energy"])
                    energy_margin = runner_up_energy - best_candidate_energy
                if _has_semantic_consensus(best, probability_evidence, cfg):
                    selected = best
                    decision_tier = "semantic_consensus"
                    reasons.append("selected_target_matches_semantic_consensus")
                    action = "absorb"
                elif best_candidate_energy > contextual_energy_limit(
                    best, probability_evidence["top_two_class_ids"], cfg
                ):
                    reasons.append("selected_energy_above_automatic_maximum")
                elif (
                    energy_margin is not None
                    and energy_margin < cfg["minimum_energy_margin"]
                ):
                    reasons.append("best_to_runner_up_energy_margin_below_minimum")
                else:
                    selected = best
                    decision_tier = "contextual"
                    reasons.append("selected_target_minimizes_contextual_energy")
                    action = "absorb"
            else:
                reasons.extend(_abstention_reasons(target_rows, known_neighbor_ids))

        target = None
        if selected is not None:
            target_id = int(selected["target_instance_id"])
            target_record = by_instance[target_id]
            target = {
                "instance_id": target_id,
                "entity_id": _optional_text(target_record.get("entity_id")),
                "class_id": int(selected["target_class_id"]),
                "class_name": str(selected["target_class_name"]),
            }

        evidence = {
            "source_area_pixels": area,
            "source_perimeter_pixel_sides": perimeter,
            "source_touches_tile_boundary": touches_border,
            "source_is_four_connected": True,
            "adjacent_instance_count": len(adjacent_ids),
            "adjacent_known_target_count": len(known_neighbor_ids),
            "adjacent_unknown_target_count": unknown_neighbor_count,
            "missing_pair_evidence_target_ids": missing_pair_neighbors,
            **probability_evidence,
            "preserved_object_class": preserved_class,
            "target_candidates": target_rows,
            "selected_energy": (
                float(selected["energy"]) if selected is not None else None
            ),
            "best_candidate_energy": best_candidate_energy,
            "selected_energy_limit": (
                contextual_energy_limit(selected, probability_evidence["top_two_class_ids"], cfg)
                if selected is not None else None
            ),
            "runner_up_energy": runner_up_energy,
            "best_to_runner_up_energy_margin": energy_margin,
            "energy_weights": dict(_ENERGY_WEIGHTS),
            "thresholds": _reported_thresholds(cfg),
        }
        assignments.append(
            json_safe(
                {
                    "source_instance_id": source_id,
                    "source_entity_id": _optional_text(
                        source_record.get("entity_id")
                    ),
                    "source_class_id": source_class_id,
                    "source_class_name": source_class_name,
                    "decision_tier": decision_tier,
                    "action": action,
                    "target": target,
                    "target_instance_id": (
                        target["instance_id"] if target is not None else None
                    ),
                    "reasons": sorted(set(reasons)),
                    "evidence": evidence,
                }
            )
        )
    return assignments


def contextual_energy_limit(
    target: Mapping[str, Any], top_two_class_ids: Sequence[int], cfg: Mapping[str, Any]
) -> float:
    """Allow a larger energy only when independent source evidence supports it.

    Used by proposal generation and replay. Area, borders, protected targets,
    and runner-up ambiguity are checked separately by both callers.
    """
    base = float(cfg["maximum_selected_energy"])
    supported = (
        target["target_class_id"] in top_two_class_ids
        and float(target["mean_target_probability_on_source"])
        >= float(cfg["supported_minimum_probability"])
        and float(target["source_boundary_fraction"])
        >= float(cfg["supported_minimum_boundary_fraction"])
        and float(target["weak_color_boundary_score"])
        >= float(cfg["supported_minimum_boundary_similarity"])
        and float(target["region_mean_color_similarity"])
        >= float(cfg["supported_minimum_region_similarity"])
    )
    return max(base, float(cfg["supported_maximum_energy"])) if supported else base


def _evaluate_target(
    *,
    source_id: int,
    target_id: int,
    target_record: Mapping[str, Any],
    pair: Mapping[str, Any],
    source_perimeter: int,
    probability_evidence: Mapping[str, Any],
    known_neighbor_ids: Sequence[int],
    by_instance: Mapping[int, Mapping[str, Any]],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    target_class_id, target_class_name = _record_class(target_record)
    reasons: list[str] = []
    if target_class_name == UNKNOWN_CLASS_NAME:
        reasons.append("target_is_unknown")

    shared_sides = int(pair["shared_boundary_side_count"])
    boundary_fraction = float(shared_sides / source_perimeter)
    weak_boundary = _optional_fraction(
        pair.get("weak_color_boundary_score"),
        "pair_evidence.weak_color_boundary_score",
    )
    region_similarity = _optional_fraction(
        pair.get("region_mean_color_similarity"),
        "pair_evidence.region_mean_color_similarity",
    )
    adjacency_definition = pair.get("adjacency_definition")
    if adjacency_definition != ADJACENCY_DEFINITION:
        reasons.append("adjacency_definition_not_supported")
    if weak_boundary is None:
        reasons.append("weak_color_boundary_score_unavailable")
    if region_similarity is None:
        reasons.append("region_mean_color_similarity_unavailable")

    class_means = probability_evidence.get("class_mean_probabilities")
    target_probability: float | None = None
    if isinstance(class_means, Sequence) and target_class_id < len(class_means):
        target_probability = float(class_means[target_class_id])
    else:
        reasons.append("target_class_probability_unavailable")

    same_class_neighbors = [
        neighbor
        for neighbor in known_neighbor_ids
        if _record_class(by_instance[neighbor])[1] == target_class_name
    ]
    if target_class_name == _AGRICULTURE:
        if len(same_class_neighbors) != 1:
            reasons.append("agriculture_requires_unique_field_target")
        if boundary_fraction < cfg["agriculture_minimum_boundary_fraction"]:
            reasons.append("agriculture_boundary_fraction_below_minimum")
    elif target_class_name == _BUILDINGS:
        if len(same_class_neighbors) != 1:
            reasons.append("building_requires_unique_target")
        if (
            target_probability is None
            or target_probability < cfg["building_minimum_probability"]
        ):
            reasons.append("building_probability_below_minimum")
    elif target_class_name == _ROAD:
        top_two_ids = probability_evidence.get("top_two_class_ids") or []
        if (
            target_probability is None
            or (
                target_probability < cfg["road_minimum_probability"]
                and target_class_id not in top_two_ids
            )
        ):
            reasons.append("road_probability_or_top_two_support_required")

    energy: float | None = None
    if (
        target_probability is not None
        and weak_boundary is not None
        and region_similarity is not None
    ):
        energy = float(
            _ENERGY_WEIGHTS["semantic_probability"] * (1.0 - target_probability)
            + _ENERGY_WEIGHTS["boundary_contact"] * (1.0 - boundary_fraction)
            + _ENERGY_WEIGHTS["weak_rgb_boundary"] * (1.0 - weak_boundary)
            + _ENERGY_WEIGHTS["region_rgb_similarity"]
            * (1.0 - region_similarity)
        )

    return {
        "target_instance_id": target_id,
        "target_entity_id": _optional_text(target_record.get("entity_id")),
        "target_class_id": target_class_id,
        "target_class_name": target_class_name,
        "eligible": not reasons and energy is not None,
        "policy_reasons": sorted(set(reasons)),
        "shared_boundary_side_count": shared_sides,
        "source_boundary_fraction": boundary_fraction,
        "mean_target_probability_on_source": target_probability,
        "weak_color_boundary_score": weak_boundary,
        "region_mean_color_similarity": region_similarity,
        "energy": energy,
    }


def _preserved_object_class(
    *,
    area: int,
    probability_evidence: Mapping[str, Any],
    matching_known_targets: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
) -> str | None:
    if matching_known_targets:
        return None
    top_class = probability_evidence.get("top_class_name")
    probability = probability_evidence.get("top_class_probability")
    margin = probability_evidence.get("probability_margin")
    purity = probability_evidence.get("top_class_pixel_purity")
    if probability is None or margin is None or purity is None:
        return None
    if top_class == _TREE and (
        area >= cfg["standalone_tree_minimum_area_pixels"]
        and probability >= cfg["standalone_tree_minimum_probability"]
        and margin >= cfg["standalone_tree_minimum_margin"]
        and purity >= cfg["standalone_tree_minimum_purity"]
    ):
        return _TREE
    if top_class == _BUILDINGS and (
        area >= cfg["standalone_building_minimum_area_pixels"]
        and probability >= cfg["standalone_building_minimum_probability"]
        and margin >= cfg["standalone_building_minimum_margin"]
        and purity >= cfg["standalone_building_minimum_purity"]
    ):
        return _BUILDINGS
    return None


def _has_semantic_consensus(
    target: Mapping[str, Any],
    probability_evidence: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> bool:
    return bool(
        target["target_class_id"] == probability_evidence.get("top_class_id")
        and probability_evidence.get("top_class_probability") is not None
        and probability_evidence["top_class_probability"]
        >= cfg["semantic_consensus_minimum_probability"]
        and probability_evidence["probability_margin"]
        >= cfg["semantic_consensus_minimum_margin"]
        and probability_evidence["top_class_pixel_purity"]
        >= cfg["semantic_consensus_minimum_purity"]
    )


def _abstention_reasons(
    target_rows: Sequence[Mapping[str, Any]],
    known_neighbor_ids: Sequence[int],
) -> list[str]:
    if not known_neighbor_ids:
        return ["no_directly_adjacent_known_target"]
    if not target_rows:
        return ["adjacent_target_evidence_unavailable"]
    policy_reasons = sorted(
        {
            str(reason)
            for row in target_rows
            for reason in row.get("policy_reasons", [])
        }
    )
    return policy_reasons or ["no_eligible_contextual_target"]


def _probability_evidence(
    probabilities: np.ndarray | None,
    mask: np.ndarray,
    class_names: Mapping[int, str],
) -> dict[str, Any]:
    if probabilities is None:
        return {
            "class_mean_probabilities": None,
            "top_class_id": None,
            "top_class_name": None,
            "top_class_probability": None,
            "runner_up_class_id": None,
            "runner_up_probability": None,
            "probability_margin": None,
            "top_two_class_ids": [],
            "top_class_pixel_purity": None,
        }
    selected = probabilities[:, mask]
    means = selected.mean(axis=1, dtype=np.float64)
    order = sorted(range(len(means)), key=lambda class_id: (-means[class_id], class_id))
    top_id = int(order[0])
    runner_id = int(order[1]) if len(order) > 1 else None
    pixel_maxima = selected.max(axis=0)
    ties = np.isclose(selected, pixel_maxima[None, :], rtol=0.0, atol=1e-12)
    unique_top_wins = ties[top_id] & (ties.sum(axis=0) == 1)
    return {
        "class_mean_probabilities": means.tolist(),
        "top_class_id": top_id,
        "top_class_name": class_names.get(top_id, _CLASS_NAMES.get(top_id)),
        "top_class_probability": float(means[top_id]),
        "runner_up_class_id": runner_id,
        "runner_up_probability": (
            float(means[runner_id]) if runner_id is not None else 0.0
        ),
        "probability_margin": float(
            means[top_id] - (means[runner_id] if runner_id is not None else 0.0)
        ),
        "top_two_class_ids": [int(value) for value in order[:2]],
        "top_class_pixel_purity": float(unique_top_wins.mean()),
    }


def _select_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("contextual cleanup config must be a mapping")
    selected: Mapping[str, Any] = config
    operations = config.get("operations")
    if isinstance(operations, Mapping):
        candidate = operations.get("contextual_cleanup")
        if not isinstance(candidate, Mapping):
            candidate = operations.get("unknown_cleanup")
        if isinstance(candidate, Mapping):
            selected = candidate
    elif isinstance(config.get("contextual_cleanup"), Mapping):
        selected = config["contextual_cleanup"]

    output = dict(_DEFAULTS)
    for key in output:
        if key in selected:
            output[key] = selected[key]
    # Existing unknown-cleanup configuration uses these equivalent names.
    aliases = {
        "minimum_mean_target_probability": "semantic_consensus_minimum_probability",
        "minimum_probability_margin": "semantic_consensus_minimum_margin",
        "minimum_dominant_pixel_fraction": "semantic_consensus_minimum_purity",
    }
    for old, new in aliases.items():
        if old in selected and new not in selected:
            output[new] = selected[old]

    if not isinstance(output["enabled"], bool):
        raise ValueError("contextual cleanup enabled must be a boolean")
    output["maximum_segment_area_pixels"] = _positive_integer(
        output["maximum_segment_area_pixels"],
        "contextual_cleanup.maximum_segment_area_pixels",
    )
    if output["maximum_segment_area_pixels"] > MAXIMUM_SAFE_MICRO_AREA_PIXELS:
        raise ValueError(
            "contextual cleanup maximum_segment_area_pixels cannot exceed 40"
        )
    for key in (
        "standalone_tree_minimum_area_pixels",
        "standalone_building_minimum_area_pixels",
    ):
        output[key] = _positive_integer(output[key], f"contextual_cleanup.{key}")
    for key in (
        "semantic_consensus_minimum_probability",
        "semantic_consensus_minimum_margin",
        "semantic_consensus_minimum_purity",
        "maximum_selected_energy",
        "supported_maximum_energy",
        "supported_minimum_probability",
        "supported_minimum_boundary_fraction",
        "supported_minimum_boundary_similarity",
        "supported_minimum_region_similarity",
        "minimum_energy_margin",
        "standalone_tree_minimum_probability",
        "standalone_tree_minimum_margin",
        "standalone_tree_minimum_purity",
        "standalone_building_minimum_probability",
        "standalone_building_minimum_margin",
        "standalone_building_minimum_purity",
        "agriculture_minimum_boundary_fraction",
        "building_minimum_probability",
        "road_minimum_probability",
    ):
        output[key] = _fraction(output[key], f"contextual_cleanup.{key}")
    return output


def _reported_thresholds(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: config[key] for key in _DEFAULTS}


def _pair_index(
    pair_rows: Sequence[Mapping[str, Any]],
    records: Mapping[int, Mapping[str, Any]],
    actual_contacts: Mapping[tuple[int, int], int],
) -> dict[tuple[int, int], dict[str, Any]]:
    if isinstance(pair_rows, (str, bytes, bytearray)) or not isinstance(
        pair_rows, Sequence
    ):
        raise TypeError("pair_rows must be a sequence of pair-evidence mappings")
    output: dict[tuple[int, int], dict[str, Any]] = {}
    for index, raw in enumerate(pair_rows):
        if not isinstance(raw, Mapping):
            raise TypeError(f"pair_rows[{index}] must be a mapping")
        first = _positive_integer(
            raw.get("source_instance_id"),
            f"pair_rows[{index}].source_instance_id",
        )
        second = _positive_integer(
            raw.get("target_instance_id"),
            f"pair_rows[{index}].target_instance_id",
        )
        if first == second:
            raise ValueError("pair evidence cannot reference one instance twice")
        if first not in records or second not in records:
            raise ValueError("pair evidence references an unknown instance")
        key = (min(first, second), max(first, second))
        if key in output:
            raise ValueError(f"duplicate pair evidence for instances {key}")
        if key not in actual_contacts:
            raise ValueError(f"pair evidence {key} is not a direct raster adjacency")
        shared = _positive_integer(
            raw.get("shared_boundary_side_count"),
            f"pair_rows[{index}].shared_boundary_side_count",
        )
        if shared != actual_contacts[key]:
            raise ValueError(
                f"pair evidence {key} shared-side count does not match the raster"
            )
        output[key] = dict(raw)
    return output


def _raster_contacts(instance_map: np.ndarray) -> dict[tuple[int, int], int]:
    output: dict[tuple[int, int], int] = {}
    for first, second in (
        (instance_map[:, :-1], instance_map[:, 1:]),
        (instance_map[:-1, :], instance_map[1:, :]),
    ):
        changed = (first > 0) & (second > 0) & (first != second)
        for left, right in zip(first[changed], second[changed]):
            key = (min(int(left), int(right)), max(int(left), int(right)))
            output[key] = output.get(key, 0) + 1
    return output


def _record_index(
    records: Sequence[Mapping[str, Any]],
    instance_map: np.ndarray,
) -> tuple[dict[int, Mapping[str, Any]], dict[int, str]]:
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(
        records, Sequence
    ):
        raise TypeError("records must be a sequence of mappings")
    by_instance: dict[int, Mapping[str, Any]] = {}
    class_names = dict(_CLASS_NAMES)
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"records[{index}] must be a mapping")
        instance_id = _positive_integer(
            record.get("instance_id"), f"records[{index}].instance_id"
        )
        if instance_id in by_instance:
            raise ValueError(f"duplicate record instance ID {instance_id}")
        class_id, class_name = _record_class(record)
        existing = class_names.get(class_id)
        if existing is not None and existing != class_name:
            raise ValueError(
                f"semantic class ID {class_id} disagrees with {existing!r}"
            )
        class_names[class_id] = class_name
        by_instance[instance_id] = record
    raster_ids = {int(value) for value in np.unique(instance_map) if value > 0}
    if raster_ids != set(by_instance):
        raise ValueError("records and instance_map positive IDs do not agree")
    return by_instance, class_names


def _record_class(record: Mapping[str, Any]) -> tuple[int, str]:
    label = record.get("label")
    if not isinstance(label, Mapping):
        raise ValueError("record requires a label mapping")
    class_id = _nonnegative_integer(label.get("id"), "record.label.id")
    class_name = label.get("name")
    if not isinstance(class_name, str) or not class_name.strip():
        raise ValueError("record requires a non-empty label.name")
    return class_id, class_name


def _validate_instance_map(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or not array.size:
        raise ValueError("instance_map must be a non-empty two-dimensional array")
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.integer):
        raise TypeError("instance_map must use an integer dtype")
    if np.any(array < 0):
        raise ValueError("instance_map cannot contain negative instance IDs")
    return array


def _validate_probabilities(
    value: np.ndarray | None,
    shape: tuple[int, int],
) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[1:] != shape or array.shape[0] < 1:
        raise ValueError(
            "probabilities must have class-first shape (classes, height, width)"
        )
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.number):
        raise TypeError("probabilities must use a numeric dtype")
    result = array.astype(np.float64, copy=False)
    if not np.isfinite(result).all() or np.any((result < 0.0) | (result > 1.0)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    return result


def _mask_perimeter(mask: np.ndarray) -> int:
    padded = np.pad(mask, 1, constant_values=False)
    center = padded[1:-1, 1:-1]
    perimeter = int(
        np.count_nonzero(center & ~padded[:-2, 1:-1])
        + np.count_nonzero(center & ~padded[2:, 1:-1])
        + np.count_nonzero(center & ~padded[1:-1, :-2])
        + np.count_nonzero(center & ~padded[1:-1, 2:])
    )
    if perimeter <= 0:
        raise ValueError("instance mask has no perimeter")
    return perimeter


def _is_four_connected(mask: np.ndarray) -> bool:
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        return False
    start = tuple(int(value) for value in coordinates[0])
    pending = [start]
    visited = {start}
    height, width = mask.shape
    while pending:
        row, column = pending.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = (row + dr, column + dc)
            if (
                0 <= neighbor[0] < height
                and 0 <= neighbor[1] < width
                and mask[neighbor]
                and neighbor not in visited
            ):
                visited.add(neighbor)
                pending.append(neighbor)
    return len(visited) == len(coordinates)


def _touches_tile_boundary(mask: np.ndarray) -> bool:
    return bool(
        mask[0, :].any()
        or mask[-1, :].any()
        or mask[:, 0].any()
        or mask[:, -1].any()
    )


def _optional_fraction(value: Any, name: str) -> float | None:
    return None if value is None else _fraction(value, name)


def _fraction(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite fraction in [0, 1]")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite fraction in [0, 1]") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite fraction in [0, 1]")
    return result


def _positive_integer(value: Any, name: str) -> int:
    result = _nonnegative_integer(value, name)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


def _reason_slug(value: str) -> str:
    return value.replace(" ", "_").replace("-", "_")


__all__ = [
    "MAXIMUM_SAFE_MICRO_AREA_PIXELS",
    "UNKNOWN_CLASS_NAME",
    "evaluate_contextual_micro_cleanup",
]
