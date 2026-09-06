"""Pure, conservative policy gates for Stage 2 refinement candidates.

The helpers in this module do not mutate rasters or records.  They turn the
available ontology, contact evidence, and semantic probabilities into an
explicit eligibility decision that proposal generation and replay can share.
Scores remain review-ranking evidence; passing these gates never means that a
semantic edit is safe to auto-apply.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


_ROAD = "road"
_AGRICULTURE = "agriculture land"
_UNKNOWN = "unknown"
_CROSS_CLASS_SIZE_UNSAFE = frozenset(
    {_ROAD, _AGRICULTURE, "tree", "buildings"}
)
_UNKNOWN_TARGET_PROBABILITY_MINIMUM = 0.90
_UNKNOWN_TARGET_PROBABILITY_MARGIN_MINIMUM = 0.25
_UNKNOWN_BOUNDARY_FRACTION_MINIMUM = 0.90


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Structured outcome of one candidate policy check."""

    eligible: bool
    reasons: tuple[str, ...]
    hard_vetoes: tuple[str, ...]
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-oriented representation."""

        return {
            "eligible": bool(self.eligible),
            "reasons": list(self.reasons),
            "hard_vetoes": list(self.hard_vetoes),
            "evidence": copy.deepcopy(self.evidence),
        }


def orient_absorb_evidence(
    evidence: Mapping[str, Any],
    candidate_instance_id: int,
    absorb_target_instance_id: int,
) -> dict[str, Any]:
    """Orient a canonical pair-evidence row from candidate to absorb target.

    Pair evidence stores instance IDs in ascending order, so its ``source_*``
    values do not necessarily describe the segment being absorbed.  The
    canonical fields are retained for auditability and explicit
    ``candidate_*`` / ``absorb_target_*`` fields are added for reviewers.
    """

    row = copy.deepcopy(dict(evidence))
    candidate = _positive_integer(candidate_instance_id, "candidate_instance_id")
    target = _positive_integer(absorb_target_instance_id, "absorb_target_instance_id")
    if candidate == target:
        raise ValueError("An absorb candidate and target must be distinct")
    canonical_source = _positive_integer(
        row.get("source_instance_id"), "evidence.source_instance_id"
    )
    canonical_target = _positive_integer(
        row.get("target_instance_id"), "evidence.target_instance_id"
    )
    if {candidate, target} != {canonical_source, canonical_target}:
        raise ValueError(
            "Absorb candidate/target IDs do not match the pair-evidence IDs"
        )

    forward = candidate == canonical_source
    candidate_prefix, target_prefix = (
        ("source", "target") if forward else ("target", "source")
    )
    row.update(
        {
            "absorb_orientation": "canonical_forward" if forward else "canonical_reverse",
            "candidate_instance_id": candidate,
            "neighbor_instance_id": target,
            "absorb_target_instance_id": target,
            "candidate_area_pixels": _nonnegative_integer(
                row.get(f"{candidate_prefix}_area_pixels"),
                f"evidence.{candidate_prefix}_area_pixels",
            ),
            "absorb_target_area_pixels": _nonnegative_integer(
                row.get(f"{target_prefix}_area_pixels"),
                f"evidence.{target_prefix}_area_pixels",
            ),
            "candidate_perimeter_pixel_sides": _positive_integer(
                row.get(f"{candidate_prefix}_perimeter_pixel_sides"),
                f"evidence.{candidate_prefix}_perimeter_pixel_sides",
            ),
            "absorb_target_perimeter_pixel_sides": _positive_integer(
                row.get(f"{target_prefix}_perimeter_pixel_sides"),
                f"evidence.{target_prefix}_perimeter_pixel_sides",
            ),
            "candidate_shared_boundary_fraction": _fraction(
                row.get(f"{candidate_prefix}_shared_boundary_fraction"),
                f"evidence.{candidate_prefix}_shared_boundary_fraction",
            ),
            "absorb_target_shared_boundary_fraction": _fraction(
                row.get(f"{target_prefix}_shared_boundary_fraction"),
                f"evidence.{target_prefix}_shared_boundary_fraction",
            ),
        }
    )
    return row


def classify_hard_vetoes(
    operation_type: str,
    source_record: Mapping[str, Any],
    target_record: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return ontology vetoes which review is not allowed to override."""

    operation = str(operation_type)
    if operation not in {"merge", "absorb"}:
        raise ValueError(f"Policy classification does not support {operation!r}")
    source_class = _class_name(source_record)
    target_class = _class_name(target_record)
    same_class = _class_id(source_record) == _class_id(target_record)
    vetoes: list[str] = []

    if operation == "merge":
        if not same_class:
            vetoes.append("merge_requires_same_semantic_class")
        if source_class != target_class:
            vetoes.append("merge_class_names_disagree")
        if source_class == _ROAD:
            vetoes.append("road_merge_requires_junction_topology")
        elif source_class == _AGRICULTURE:
            vetoes.append("agriculture_merge_may_erase_field_boundary")
        elif source_class == _UNKNOWN:
            vetoes.append("unknown_merge_must_abstain")
        if same_class and not _ontology_allows_merge(cfg, source_class):
            vetoes.append("ontology_disallows_merge")
    else:
        if same_class:
            if source_class == _ROAD:
                vetoes.append("road_absorb_requires_junction_topology")
            elif source_class == _AGRICULTURE:
                vetoes.append("agriculture_absorb_may_erase_field_boundary")
            elif not _ontology_allows_merge(cfg, source_class):
                vetoes.append("ontology_disallows_absorb")
        else:
            if source_class in _CROSS_CLASS_SIZE_UNSAFE:
                vetoes.append(
                    f"cross_class_absorb_forbidden_for_{_reason_slug(source_class)}"
                )
            if target_class == _UNKNOWN:
                vetoes.append("absorb_into_unknown_forbidden")

    return tuple(sorted(set(vetoes)))


def evaluate_merge_candidate(
    cfg: Mapping[str, Any],
    first_record: Mapping[str, Any],
    second_record: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    score: float,
) -> PolicyDecision:
    """Conservatively decide whether a merge may enter human review."""

    row = copy.deepcopy(dict(evidence))
    hard_vetoes = classify_hard_vetoes(
        "merge", first_record, second_record, cfg
    )
    if hard_vetoes:
        return _decision(row, ("hard_policy_veto",), hard_vetoes)

    operation_cfg = _operation_config(cfg, "merge")
    reasons: list[str] = []
    if not bool(operation_cfg.get("enabled", False)):
        reasons.append("merge_disabled")
    candidate_score = _finite_score(score)
    threshold = _fraction(
        operation_cfg.get("score_threshold"),
        "refinement.operations.merge.score_threshold",
    )
    if candidate_score < threshold:
        reasons.append("score_below_configured_threshold")

    shared_sides = _nonnegative_integer(
        row.get("shared_boundary_side_count"),
        "evidence.shared_boundary_side_count",
    )
    minimum_sides = _positive_integer(
        operation_cfg.get("minimum_shared_boundary_pixels"),
        "refinement.operations.merge.minimum_shared_boundary_pixels",
    )
    if shared_sides < minimum_sides:
        reasons.append("shared_boundary_below_configured_minimum")
    contact_fraction = _fraction(
        row.get("minimum_perimeter_shared_boundary_fraction"),
        "evidence.minimum_perimeter_shared_boundary_fraction",
    )
    minimum_fraction = _fraction(
        operation_cfg.get("minimum_shared_boundary_fraction"),
        "refinement.operations.merge.minimum_shared_boundary_fraction",
    )
    if contact_fraction < minimum_fraction:
        reasons.append("shared_boundary_fraction_below_configured_minimum")
    if row.get("adjacency_definition") not in {
        None,
        "shared_pixel_edge_4_connected",
    }:
        reasons.append("adjacency_is_not_shared_pixel_edge_4_connected")

    return _decision(row, tuple(reasons), ())


def evaluate_absorb_candidate(
    cfg: Mapping[str, Any],
    source_record: Mapping[str, Any],
    target_record: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    score: float,
    semantic_probabilities: np.ndarray | None = None,
    candidate_mask: np.ndarray | None = None,
) -> PolicyDecision:
    """Conservatively decide whether an absorb may enter human review.

    Cross-class decisions require probability support.  When the source is
    ``unknown``, the target probability, probability margin, and enclosure
    thresholds are fixed safety floors and cannot be weakened by configuration.
    """

    row = orient_absorb_evidence(
        evidence,
        _instance_id(source_record),
        _instance_id(target_record),
    )
    hard_vetoes = classify_hard_vetoes(
        "absorb", source_record, target_record, cfg
    )
    if hard_vetoes:
        return _decision(row, ("hard_policy_veto",), hard_vetoes)

    operation_cfg = _operation_config(cfg, "absorb")
    reasons: list[str] = []
    if not bool(operation_cfg.get("enabled", False)):
        reasons.append("absorb_disabled")
    candidate_score = _finite_score(score)
    score_threshold = _fraction(
        operation_cfg.get("score_threshold"),
        "refinement.operations.absorb.score_threshold",
    )
    if candidate_score < score_threshold:
        reasons.append("score_below_configured_threshold")

    actual_area = _nonnegative_integer(
        source_record.get("area_pixels"), "source_record.area_pixels"
    )
    row["candidate_record_area_pixels"] = actual_area
    maximum_area = _positive_integer(
        operation_cfg.get("maximum_segment_area_pixels"),
        "refinement.operations.absorb.maximum_segment_area_pixels",
    )
    if actual_area > maximum_area:
        reasons.append("candidate_area_above_configured_maximum")

    boundary_fraction = _fraction(
        row.get("candidate_shared_boundary_fraction"),
        "evidence.candidate_shared_boundary_fraction",
    )
    minimum_boundary = _fraction(
        operation_cfg.get("minimum_shared_boundary_fraction"),
        "refinement.operations.absorb.minimum_shared_boundary_fraction",
    )
    if boundary_fraction < minimum_boundary:
        reasons.append("shared_boundary_fraction_below_configured_minimum")

    neighbor_margin = _neighbor_boundary_margin(row)
    row["best_to_runner_up_boundary_fraction_margin"] = neighbor_margin
    minimum_neighbor_margin = _fraction(
        operation_cfg.get("score_margin_threshold"),
        "refinement.operations.absorb.score_margin_threshold",
    )
    if neighbor_margin < minimum_neighbor_margin:
        reasons.append("neighbor_boundary_margin_below_configured_minimum")

    same_class = _class_id(source_record) == _class_id(target_record)
    if not same_class:
        probability, probability_margin = _target_probability_support(
            row,
            target_class_id=_class_id(target_record),
            semantic_probabilities=semantic_probabilities,
            candidate_mask=candidate_mask,
        )
        row["mean_target_class_probability_on_source"] = probability
        row["target_class_probability_margin_on_source"] = probability_margin
        if probability is None:
            reasons.append("target_class_probability_unavailable")
        if probability_margin is None:
            reasons.append("target_class_probability_margin_unavailable")

        source_class = _class_name(source_record)
        probability_minimum = _UNKNOWN_TARGET_PROBABILITY_MINIMUM
        probability_margin_minimum = _UNKNOWN_TARGET_PROBABILITY_MARGIN_MINIMUM
        cross_class_boundary_minimum = minimum_boundary
        if source_class == _UNKNOWN:
            cross_class_boundary_minimum = max(
                cross_class_boundary_minimum,
                _UNKNOWN_BOUNDARY_FRACTION_MINIMUM,
            )
        if probability is not None and probability < probability_minimum:
            reasons.append("target_class_probability_below_safety_minimum")
        if (
            probability_margin is not None
            and probability_margin < probability_margin_minimum
        ):
            reasons.append("target_class_probability_margin_below_safety_minimum")
        if boundary_fraction < cross_class_boundary_minimum:
            reasons.append("cross_class_boundary_fraction_below_safety_minimum")

    return _decision(row, tuple(reasons), ())


def _target_probability_support(
    evidence: Mapping[str, Any],
    *,
    target_class_id: int,
    semantic_probabilities: np.ndarray | None,
    candidate_mask: np.ndarray | None,
) -> tuple[float | None, float | None]:
    if (semantic_probabilities is None) != (candidate_mask is None):
        raise ValueError(
            "semantic_probabilities and candidate_mask must be supplied together"
        )
    if semantic_probabilities is not None and candidate_mask is not None:
        probabilities = np.asarray(semantic_probabilities, dtype=np.float64)
        mask = np.asarray(candidate_mask)
        if probabilities.ndim != 3 or mask.ndim != 2:
            raise ValueError(
                "semantic_probabilities must be class-first 3-D and candidate_mask 2-D"
            )
        if probabilities.shape[1:] != mask.shape or mask.dtype != np.bool_:
            raise ValueError(
                "candidate_mask must be boolean and match semantic probability dimensions"
            )
        if not mask.any():
            raise ValueError("candidate_mask must select at least one pixel")
        if target_class_id < 0 or target_class_id >= probabilities.shape[0]:
            raise ValueError("target class ID has no semantic probability channel")
        selected = probabilities[:, mask]
        if not np.isfinite(selected).all() or np.any((selected < 0) | (selected > 1)):
            raise ValueError("semantic probabilities must be finite and in [0, 1]")
        class_means = selected.mean(axis=1)
        target_probability = float(class_means[target_class_id])
        competitors = np.delete(class_means, target_class_id)
        runner_up = float(competitors.max()) if competitors.size else 0.0
        return target_probability, float(target_probability - runner_up)

    probability = _optional_fraction(
        evidence.get("mean_target_class_probability_on_source")
    )
    margin_value = evidence.get("target_class_probability_margin_on_source")
    if margin_value is None:
        margin_value = evidence.get("target_vs_runner_up_probability_margin_on_source")
    margin = _optional_probability_margin(margin_value)
    return probability, margin


def _neighbor_boundary_margin(evidence: Mapping[str, Any]) -> float:
    explicit = evidence.get("best_to_runner_up_margin")
    if explicit is not None:
        return _fraction(explicit, "evidence.best_to_runner_up_margin")
    runner_up = evidence.get("runner_up_boundary_fraction")
    if runner_up is None:
        raise ValueError(
            "Absorb evidence requires best_to_runner_up_margin or "
            "runner_up_boundary_fraction"
        )
    best = _fraction(
        evidence.get("candidate_shared_boundary_fraction"),
        "evidence.candidate_shared_boundary_fraction",
    )
    return max(
        0.0,
        best - _fraction(runner_up, "evidence.runner_up_boundary_fraction"),
    )


def _decision(
    evidence: Mapping[str, Any],
    reasons: tuple[str, ...],
    hard_vetoes: tuple[str, ...],
) -> PolicyDecision:
    ordered_reasons = tuple(sorted(set(reasons)))
    ordered_vetoes = tuple(sorted(set(hard_vetoes)))
    return PolicyDecision(
        eligible=not ordered_reasons and not ordered_vetoes,
        reasons=ordered_reasons,
        hard_vetoes=ordered_vetoes,
        evidence=copy.deepcopy(dict(evidence)),
    )


def _operation_config(cfg: Mapping[str, Any], operation: str) -> Mapping[str, Any]:
    operations = cfg.get("operations")
    if not isinstance(operations, Mapping):
        raise ValueError("refinement configuration requires an operations mapping")
    selected = operations.get(operation)
    if not isinstance(selected, Mapping):
        raise ValueError(f"refinement operations require a {operation!r} mapping")
    return selected


def _ontology_allows_merge(cfg: Mapping[str, Any], class_name: str) -> bool:
    ontology = cfg.get("ontology")
    if not isinstance(ontology, Mapping):
        raise ValueError("refinement configuration requires an ontology mapping")
    rule = ontology.get(class_name)
    if not isinstance(rule, Mapping):
        raise ValueError(f"refinement ontology has no rule for {class_name!r}")
    return bool(rule.get("allow_merge", False))


def _class_name(record: Mapping[str, Any]) -> str:
    label = record.get("label")
    if not isinstance(label, Mapping) or not str(label.get("name", "")):
        raise ValueError("Refinement record requires label.name")
    return str(label["name"])


def _class_id(record: Mapping[str, Any]) -> int:
    label = record.get("label")
    if not isinstance(label, Mapping):
        raise ValueError("Refinement record requires a label mapping")
    return _nonnegative_integer(label.get("id"), "record.label.id")


def _instance_id(record: Mapping[str, Any]) -> int:
    return _positive_integer(record.get("instance_id"), "record.instance_id")


def _finite_score(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Candidate score must be a finite number in [0, 1]")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Candidate score must be a finite number in [0, 1]") from exc
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError("Candidate score must be a finite number in [0, 1]")
    return result


def _fraction(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite fraction in [0, 1]")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite fraction in [0, 1]") from exc
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must be a finite fraction in [0, 1]")
    return result


def _optional_fraction(value: Any) -> float | None:
    return None if value is None else _fraction(value, "semantic target probability")


def _optional_probability_margin(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Semantic probability margin must be finite and in [-1, 1]")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Semantic probability margin must be finite and in [-1, 1]"
        ) from exc
    if not math.isfinite(result) or result < -1.0 or result > 1.0:
        raise ValueError("Semantic probability margin must be finite and in [-1, 1]")
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


def _reason_slug(value: str) -> str:
    return value.replace(" ", "_").replace("-", "_")


__all__ = [
    "PolicyDecision",
    "classify_hard_vetoes",
    "evaluate_absorb_candidate",
    "evaluate_merge_candidate",
    "orient_absorb_evidence",
]
