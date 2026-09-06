"""Conservative evidence gates for automatic micro-unknown cleanup.

This module is deliberately pure: it neither changes the segmentation nor
creates refinement-schema proposals.  It evaluates every normalized
``unknown`` instance and returns deterministic, JSON-safe candidate records
that a caller may translate into operations.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .evidence import ADJACENCY_DEFINITION
from .schema import json_safe


UNKNOWN_CLASS_NAME = "unknown"

_CONFIG_KEYS = frozenset(
    {
        "enabled",
        "auto_apply",
        "maximum_segment_area_pixels",
        "allowed_target_classes",
        "minimum_mean_target_probability",
        "minimum_probability_margin",
        "minimum_dominant_pixel_fraction",
        "minimum_shared_boundary_fraction",
        "minimum_weak_color_boundary_score",
        "minimum_region_mean_color_similarity",
        "require_not_tile_boundary",
    }
)


def evaluate_unknown_cleanup_candidates(
    records: Sequence[Mapping[str, Any]],
    instance_map: np.ndarray,
    pair_rows: Sequence[Mapping[str, Any]],
    semantic_probabilities: np.ndarray | None,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate one deterministic absorb target for every unknown instance.

    A source is eligible only when it is a small interior ``unknown`` region,
    its dominant mean OEM class is unambiguous, and a directly adjacent known
    target of that exact class passes all configured semantic, contact, and
    appearance gates.  When several matching targets exist, selection uses:

    1. greatest number of shared pixel sides;
    2. greatest weak-color-boundary score;
    3. lowest target instance ID.

    The result is JSON-safe decision evidence, not a refinement proposal.  All
    unknown sources are emitted, including rejected ones, so callers can
    account for every abstention.
    """

    cfg = _validate_config(config)
    instances = _validate_instance_map(instance_map)
    by_instance, _ = _record_index(records, instances)
    probabilities = _validate_probabilities(semantic_probabilities, instances.shape)
    contacts = _contact_index(pair_rows, by_instance)
    thresholds = _thresholds(cfg)

    candidates: list[dict[str, Any]] = []
    for source_id in sorted(by_instance):
        source_record = by_instance[source_id]
        source_class_id, source_class_name = _record_class(source_record)
        if source_class_name != UNKNOWN_CLASS_NAME:
            continue

        mask = instances == source_id
        area = int(mask.sum())
        perimeter = _mask_perimeter(mask)
        touches_tile_boundary = _touches_tile_boundary(mask)
        reasons: list[str] = []
        if not cfg["enabled"]:
            reasons.append("unknown_cleanup_disabled")
        if area > cfg["maximum_segment_area_pixels"]:
            reasons.append("source_area_above_maximum")
        if cfg["require_not_tile_boundary"] and touches_tile_boundary:
            reasons.append("source_touches_tile_boundary")

        probability_evidence = _probability_evidence(probabilities, mask)
        dominant_class_id = probability_evidence["dominant_mean_class_id"]
        if probabilities is None:
            reasons.append("semantic_probabilities_unavailable")
        elif probability_evidence["dominant_mean_class_is_ambiguous"]:
            reasons.append("dominant_mean_class_ambiguous")

        adjacent = contacts.get(source_id, ())
        known_adjacent = [
            item
            for item in adjacent
            if _record_class(by_instance[item["neighbor_instance_id"]])[1]
            != UNKNOWN_CLASS_NAME
        ]
        matching = [
            item
            for item in known_adjacent
            if dominant_class_id is not None
            and _record_class(by_instance[item["neighbor_instance_id"]])[0]
            == dominant_class_id
        ]
        matching.sort(key=_target_order_key)
        selected = matching[0] if matching else None

        if probabilities is not None and not matching:
            reasons.append("no_adjacent_target_matching_dominant_class")

        target_record: Mapping[str, Any] | None = None
        target_id: int | None = None
        target_class_id: int | None = None
        target_class_name: str | None = None
        shared_sides: int | None = None
        contact_fraction: float | None = None
        weak_boundary: float | None = None
        mean_color_similarity: float | None = None
        adjacency_definition: str | None = None
        if selected is not None:
            target_id = int(selected["neighbor_instance_id"])
            target_record = by_instance[target_id]
            target_class_id, target_class_name = _record_class(target_record)
            shared_sides = int(selected["shared_boundary_side_count"])
            contact_fraction = float(shared_sides / perimeter)
            weak_boundary = _optional_fraction(
                selected.get("weak_color_boundary_score"),
                "pair_evidence.weak_color_boundary_score",
            )
            mean_color_similarity = _optional_fraction(
                selected.get("region_mean_color_similarity"),
                "pair_evidence.region_mean_color_similarity",
            )
            adjacency_definition = selected.get("adjacency_definition")

            if target_class_name not in cfg["allowed_target_classes"]:
                reasons.append("target_class_not_allowed")
            if adjacency_definition != ADJACENCY_DEFINITION:
                reasons.append("adjacency_definition_not_supported")
            if contact_fraction < cfg["minimum_shared_boundary_fraction"]:
                reasons.append("shared_boundary_fraction_below_minimum")
            if weak_boundary is None:
                reasons.append("weak_color_boundary_score_unavailable")
            elif weak_boundary < cfg["minimum_weak_color_boundary_score"]:
                reasons.append("weak_color_boundary_score_below_minimum")
            if mean_color_similarity is None:
                reasons.append("region_mean_color_similarity_unavailable")
            elif (
                mean_color_similarity
                < cfg["minimum_region_mean_color_similarity"]
            ):
                reasons.append("region_mean_color_similarity_below_minimum")

        mean_target_probability = probability_evidence[
            "mean_target_class_probability_on_source"
        ]
        probability_margin = probability_evidence[
            "target_class_probability_margin_on_source"
        ]
        dominant_pixel_fraction = probability_evidence[
            "dominant_target_pixel_fraction_on_source"
        ]
        if mean_target_probability is not None and (
            mean_target_probability < cfg["minimum_mean_target_probability"]
        ):
            reasons.append("mean_target_probability_below_minimum")
        if probability_margin is not None and (
            probability_margin < cfg["minimum_probability_margin"]
        ):
            reasons.append("probability_margin_below_minimum")
        if dominant_pixel_fraction is not None and (
            dominant_pixel_fraction < cfg["minimum_dominant_pixel_fraction"]
        ):
            reasons.append("dominant_pixel_fraction_below_minimum")

        ordered_reasons = sorted(set(reasons))
        eligible = not ordered_reasons
        evidence = {
            "source_area_pixels": area,
            "source_perimeter_pixel_sides": perimeter,
            "source_touches_tile_boundary": touches_tile_boundary,
            "adjacent_known_target_count": len(known_adjacent),
            "matching_dominant_class_target_count": len(matching),
            **probability_evidence,
            "selected_target_shared_boundary_side_count": shared_sides,
            # This is the fraction of the unknown source perimeter that
            # directly touches the selected target, not the smaller of two
            # perimeter fractions used by generic merge ranking.
            "selected_target_shared_boundary_fraction_of_source_perimeter": (
                contact_fraction
            ),
            "selected_target_weak_color_boundary_score": weak_boundary,
            "selected_target_region_mean_color_similarity": mean_color_similarity,
            "adjacency_definition": adjacency_definition,
        }
        candidates.append(
            json_safe(
                {
                    "source_instance_id": source_id,
                    "source_entity_id": _optional_text(source_record.get("entity_id")),
                    "source_class_id": source_class_id,
                    "source_class_name": source_class_name,
                    "target_instance_id": target_id,
                    "target_entity_id": (
                        _optional_text(target_record.get("entity_id"))
                        if target_record is not None
                        else None
                    ),
                    "target_class_id": target_class_id,
                    "target_class_name": target_class_name,
                    "eligible": eligible,
                    "eligible_for_auto_apply": bool(eligible and cfg["auto_apply"]),
                    "reasons": ordered_reasons,
                    "evidence": evidence,
                    "thresholds": thresholds,
                }
            )
        )
    return candidates


def _probability_evidence(
    probabilities: np.ndarray | None,
    mask: np.ndarray,
) -> dict[str, Any]:
    if probabilities is None:
        return {
            "dominant_mean_class_id": None,
            "dominant_mean_class_is_ambiguous": None,
            "dominant_mean_class_tie_count": None,
            "mean_target_class_probability_on_source": None,
            "target_class_probability_margin_on_source": None,
            "dominant_target_pixel_fraction_on_source": None,
            "ambiguous_dominant_pixel_fraction_on_source": None,
        }

    selected = probabilities[:, mask]
    class_means = selected.mean(axis=1, dtype=np.float64)
    top_mean = float(class_means.max())
    mean_ties = np.flatnonzero(np.isclose(class_means, top_mean, rtol=0.0, atol=1e-12))
    dominant_class_id = int(mean_ties[0])
    competitors = np.delete(class_means, dominant_class_id)
    runner_up = float(competitors.max()) if competitors.size else 0.0

    pixel_maxima = selected.max(axis=0)
    pixel_ties = np.isclose(selected, pixel_maxima[None, :], rtol=0.0, atol=1e-12)
    pixel_tie_counts = pixel_ties.sum(axis=0)
    unique_target_wins = pixel_ties[dominant_class_id] & (pixel_tie_counts == 1)
    ambiguous_pixels = pixel_tie_counts > 1
    return {
        "dominant_mean_class_id": dominant_class_id,
        "dominant_mean_class_is_ambiguous": len(mean_ties) != 1,
        "dominant_mean_class_tie_count": int(len(mean_ties)),
        "mean_target_class_probability_on_source": top_mean,
        "target_class_probability_margin_on_source": float(
            top_mean - runner_up
        ),
        "dominant_target_pixel_fraction_on_source": float(unique_target_wins.mean()),
        "ambiguous_dominant_pixel_fraction_on_source": float(ambiguous_pixels.mean()),
    }


def _contact_index(
    pair_rows: Sequence[Mapping[str, Any]],
    records: Mapping[int, Mapping[str, Any]],
) -> dict[int, list[dict[str, Any]]]:
    if isinstance(pair_rows, (str, bytes, bytearray)) or not isinstance(
        pair_rows, Sequence
    ):
        raise TypeError("pair_rows must be a sequence of pair-evidence mappings")
    output: dict[int, list[dict[str, Any]]] = {}
    seen: set[tuple[int, int]] = set()
    for index, raw_row in enumerate(pair_rows):
        if not isinstance(raw_row, Mapping):
            raise TypeError(f"pair_rows[{index}] must be a mapping")
        first = _positive_integer(
            raw_row.get("source_instance_id"),
            f"pair_rows[{index}].source_instance_id",
        )
        second = _positive_integer(
            raw_row.get("target_instance_id"),
            f"pair_rows[{index}].target_instance_id",
        )
        if first == second:
            raise ValueError("pair evidence cannot relate an instance to itself")
        if first not in records or second not in records:
            raise ValueError("pair evidence references an unknown normalized instance")
        key = (min(first, second), max(first, second))
        if key in seen:
            raise ValueError(f"duplicate pair evidence for instances {key}")
        seen.add(key)
        shared_sides = _positive_integer(
            raw_row.get("shared_boundary_side_count"),
            f"pair_rows[{index}].shared_boundary_side_count",
        )
        common = {
            "shared_boundary_side_count": shared_sides,
            "weak_color_boundary_score": raw_row.get("weak_color_boundary_score"),
            "region_mean_color_similarity": raw_row.get(
                "region_mean_color_similarity"
            ),
            "adjacency_definition": raw_row.get("adjacency_definition"),
        }
        output.setdefault(first, []).append(
            {**common, "neighbor_instance_id": second}
        )
        output.setdefault(second, []).append(
            {**common, "neighbor_instance_id": first}
        )
    return output


def _target_order_key(value: Mapping[str, Any]) -> tuple[int, float, int]:
    weak = _optional_fraction(
        value.get("weak_color_boundary_score"),
        "pair_evidence.weak_color_boundary_score",
    )
    return (
        -int(value["shared_boundary_side_count"]),
        -(weak if weak is not None else -1.0),
        int(value["neighbor_instance_id"]),
    )


def _record_index(
    records: Sequence[Mapping[str, Any]],
    instance_map: np.ndarray,
) -> tuple[dict[int, Mapping[str, Any]], dict[int, str]]:
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(
        records, Sequence
    ):
        raise TypeError("records must be a sequence of normalized record mappings")
    by_instance: dict[int, Mapping[str, Any]] = {}
    class_names: dict[int, str] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"records[{index}] must be a mapping")
        instance_id = _positive_integer(
            record.get("instance_id"), f"records[{index}].instance_id"
        )
        if instance_id in by_instance:
            raise ValueError(f"duplicate normalized record instance ID {instance_id}")
        class_id, class_name = _record_class(record)
        previous_name = class_names.setdefault(class_id, class_name)
        if previous_name != class_name:
            raise ValueError(
                f"semantic class ID {class_id} has inconsistent class names"
            )
        by_instance[instance_id] = record

    raster_ids = set(int(value) for value in np.unique(instance_map) if value > 0)
    if raster_ids != set(by_instance):
        raise ValueError("normalized records and instance_map IDs do not agree")
    return by_instance, class_names


def _record_class(record: Mapping[str, Any]) -> tuple[int, str]:
    label = record.get("label")
    if not isinstance(label, Mapping):
        raise ValueError("normalized record requires a label mapping")
    class_id = _nonnegative_integer(label.get("id"), "record.label.id")
    class_name = label.get("name")
    if not isinstance(class_name, str) or not class_name.strip():
        raise ValueError("normalized record requires a non-empty label.name")
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
    raster_shape: tuple[int, int],
) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[1:] != raster_shape or array.shape[0] < 1:
        raise ValueError(
            "semantic_probabilities must have class-first shape "
            "(class_count, height, width)"
        )
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.number):
        raise TypeError("semantic_probabilities must use a numeric dtype")
    result = array.astype(np.float64, copy=False)
    if not np.isfinite(result).all() or np.any((result < 0.0) | (result > 1.0)):
        raise ValueError("semantic_probabilities must be finite and in [0, 1]")
    return result


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config, Mapping):
        raise TypeError("unknown_cleanup config must be a mapping")
    missing = sorted(_CONFIG_KEYS - set(config))
    extra = sorted(set(config) - _CONFIG_KEYS)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise ValueError("unknown_cleanup config has invalid keys: " + "; ".join(details))

    selected = dict(config)
    for name in ("enabled", "auto_apply", "require_not_tile_boundary"):
        if not isinstance(selected[name], bool):
            raise ValueError(f"unknown_cleanup.{name} must be a boolean")
    selected["maximum_segment_area_pixels"] = _positive_integer(
        selected["maximum_segment_area_pixels"],
        "unknown_cleanup.maximum_segment_area_pixels",
    )
    classes = selected["allowed_target_classes"]
    if (
        isinstance(classes, (str, bytes, bytearray))
        or not isinstance(classes, Sequence)
        or not classes
        or any(not isinstance(value, str) or not value for value in classes)
    ):
        raise ValueError(
            "unknown_cleanup.allowed_target_classes must be a non-empty sequence "
            "of class names"
        )
    if len(set(classes)) != len(classes) or UNKNOWN_CLASS_NAME in classes:
        raise ValueError(
            "unknown_cleanup.allowed_target_classes must be unique and exclude unknown"
        )
    selected["allowed_target_classes"] = tuple(classes)
    for name in (
        "minimum_mean_target_probability",
        "minimum_probability_margin",
        "minimum_dominant_pixel_fraction",
        "minimum_shared_boundary_fraction",
        "minimum_weak_color_boundary_score",
        "minimum_region_mean_color_similarity",
    ):
        selected[name] = _fraction(selected[name], f"unknown_cleanup.{name}")
    return selected


def _thresholds(config: Mapping[str, Any]) -> dict[str, Any]:
    return json_safe(
        {
            "maximum_segment_area_pixels": config["maximum_segment_area_pixels"],
            "allowed_target_classes": list(config["allowed_target_classes"]),
            "minimum_mean_target_probability": config[
                "minimum_mean_target_probability"
            ],
            "minimum_probability_margin": config["minimum_probability_margin"],
            "minimum_dominant_pixel_fraction": config[
                "minimum_dominant_pixel_fraction"
            ],
            "minimum_shared_boundary_fraction": config[
                "minimum_shared_boundary_fraction"
            ],
            "minimum_weak_color_boundary_score": config[
                "minimum_weak_color_boundary_score"
            ],
            "minimum_region_mean_color_similarity": config[
                "minimum_region_mean_color_similarity"
            ],
            "require_not_tile_boundary": config["require_not_tile_boundary"],
        }
    )


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
        raise ValueError("normalized instance has no raster perimeter")
    return perimeter


def _touches_tile_boundary(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any())


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


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


__all__ = ["UNKNOWN_CLASS_NAME", "evaluate_unknown_cleanup_candidates"]
