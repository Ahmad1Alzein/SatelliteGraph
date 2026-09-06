"""Pure selective automation for existing refinement proposals.

The automatic policy is intentionally an *abstaining* policy.  It approves a
proposal only when every configured score and raw-evidence gate passes.  A
failed gate produces an ``uncertain`` decision so the proposal remains
available to the human-review workflow; the selector never interprets weak
evidence as proof that a proposal is wrong.

Only same-class, interior ``merge`` and generic ``absorb`` proposals are in
scope.  Connectivity normalization and the independently replay-checked
unknown-cleanup rule retain their embedded decisions.  Semantic splits always
require human review.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .schema import canonical_json, json_safe, make_decision, validate_proposal


AUTOMATIC_POLICY_VERSION = "automatic_selective_v1"
AUTOMATIC_DECISION_SOURCE = AUTOMATIC_POLICY_VERSION

DEFAULT_AUTOMATIC_CONFIG: dict[str, Any] = {
    "policy_version": AUTOMATIC_POLICY_VERSION,
    "protected_classes": [
        "agriculture land",
        "buildings",
        "road",
        "tree",
        "unknown",
    ],
    "merge": {
        "enabled": True,
        "minimum_score": 0.99,
        "minimum_shared_boundary_pixels": 12,
        "minimum_shared_boundary_fraction": 0.25,
        "minimum_weak_color_boundary_score": 0.95,
        "minimum_region_mean_color_similarity": 0.95,
        "require_not_tile_boundary": True,
    },
    "absorb": {
        "enabled": True,
        "minimum_score": 0.99,
        "maximum_source_area_pixels": 20,
        "minimum_shared_boundary_pixels": 8,
        "minimum_shared_boundary_fraction": 0.80,
        "minimum_best_to_runner_up_margin": 0.25,
        "minimum_weak_color_boundary_score": 0.95,
        "minimum_region_mean_color_similarity": 0.95,
        # This is a safety invariant, not an option that may be disabled.
        "require_same_class": True,
        "require_not_tile_boundary": True,
    },
    "maximum_selected_operations_per_tile": 10_000,
}

_TOP_LEVEL_KEYS = frozenset(DEFAULT_AUTOMATIC_CONFIG)
_MERGE_KEYS = frozenset(DEFAULT_AUTOMATIC_CONFIG["merge"])
_ABSORB_KEYS = frozenset(DEFAULT_AUTOMATIC_CONFIG["absorb"])
_SUPPORTED_AUTOMATIC_OPERATIONS = frozenset({"merge", "absorb"})
_PASSTHROUGH_OPERATIONS = frozenset({"split_components"})


@dataclass(slots=True)
class AutomaticProposalEvaluation:
    """Local, deterministic policy result for one proposal.

    Conflict resolution and the per-tile selection cap are deliberately not
    represented by ``eligible``.  They depend on the other proposals in a
    batch and are handled by :func:`select_automatic_decisions`.
    """

    proposal_id: str
    tile_id: str
    operation_type: str
    eligible: bool
    reasons: tuple[str, ...]
    threshold_snapshot: dict[str, Any]
    observations: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-safe representation."""

        return json_safe(
            {
                "proposal_id": self.proposal_id,
                "tile_id": self.tile_id,
                "operation_type": self.operation_type,
                "eligible": self.eligible,
                "reasons": list(self.reasons),
                "threshold_snapshot": copy.deepcopy(self.threshold_snapshot),
                "observations": copy.deepcopy(self.observations),
            }
        )


@dataclass(slots=True)
class AutomaticSelectionResult:
    """Automatic review decisions and their complete selection audit."""

    decisions: list[dict[str, Any]]
    report: dict[str, Any]


def default_automatic_config() -> dict[str, Any]:
    """Return an independent copy of the conservative automatic policy."""

    return copy.deepcopy(DEFAULT_AUTOMATIC_CONFIG)


def select_automatic_config(
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge a partial automatic-policy override with conservative defaults."""

    selected = default_automatic_config()
    if config is not None:
        override = _mapping(config, "automatic")
        _deep_merge(selected, override, "automatic")
    # Class-list order has no policy meaning.  Canonicalizing it makes reports
    # and decision IDs stable for semantically identical configuration.
    protected = selected["protected_classes"]
    if isinstance(protected, list):
        selected["protected_classes"] = sorted(protected)
    validate_automatic_config(selected)
    return selected


def validate_automatic_config(config: Mapping[str, Any]) -> None:
    """Validate a fully selected standalone automatic-policy mapping."""

    cfg = _mapping(config, "automatic")
    _exact_keys(cfg, _TOP_LEVEL_KEYS, "automatic")
    if cfg["policy_version"] != AUTOMATIC_POLICY_VERSION:
        raise ValueError(
            "automatic.policy_version must be "
            f"{AUTOMATIC_POLICY_VERSION!r}"
        )

    protected = cfg["protected_classes"]
    if (
        not isinstance(protected, list)
        or not protected
        or any(not isinstance(value, str) or not value.strip() for value in protected)
    ):
        raise ValueError(
            "automatic.protected_classes must be a non-empty list of class names"
        )
    if len(set(protected)) != len(protected):
        raise ValueError("automatic.protected_classes must not contain duplicates")

    merge = _mapping(cfg["merge"], "automatic.merge")
    _exact_keys(merge, _MERGE_KEYS, "automatic.merge")
    _boolean(merge["enabled"], "automatic.merge.enabled")
    _fraction(merge["minimum_score"], "automatic.merge.minimum_score")
    _positive_integer(
        merge["minimum_shared_boundary_pixels"],
        "automatic.merge.minimum_shared_boundary_pixels",
    )
    for name in (
        "minimum_shared_boundary_fraction",
        "minimum_weak_color_boundary_score",
        "minimum_region_mean_color_similarity",
    ):
        _fraction(merge[name], f"automatic.merge.{name}")
    _boolean(
        merge["require_not_tile_boundary"],
        "automatic.merge.require_not_tile_boundary",
    )

    absorb = _mapping(cfg["absorb"], "automatic.absorb")
    _exact_keys(absorb, _ABSORB_KEYS, "automatic.absorb")
    _boolean(absorb["enabled"], "automatic.absorb.enabled")
    _fraction(absorb["minimum_score"], "automatic.absorb.minimum_score")
    _positive_integer(
        absorb["maximum_source_area_pixels"],
        "automatic.absorb.maximum_source_area_pixels",
    )
    _positive_integer(
        absorb["minimum_shared_boundary_pixels"],
        "automatic.absorb.minimum_shared_boundary_pixels",
    )
    for name in (
        "minimum_shared_boundary_fraction",
        "minimum_best_to_runner_up_margin",
        "minimum_weak_color_boundary_score",
        "minimum_region_mean_color_similarity",
    ):
        _fraction(absorb[name], f"automatic.absorb.{name}")
    if absorb["require_same_class"] is not True:
        raise ValueError(
            "automatic.absorb.require_same_class must remain true; "
            "automatic generic absorbs cannot relabel pixels"
        )
    _boolean(
        absorb["require_not_tile_boundary"],
        "automatic.absorb.require_not_tile_boundary",
    )
    _positive_integer(
        cfg["maximum_selected_operations_per_tile"],
        "automatic.maximum_selected_operations_per_tile",
    )


def evaluate_automatic_proposal(
    proposal: Mapping[str, Any],
    config: Mapping[str, Any] | None = None,
) -> AutomaticProposalEvaluation:
    """Evaluate one proposal using score *and* independently auditable gates.

    This helper is batch-independent and pure.  A caller may replace proposal
    evidence with evidence recomputed from immutable rasters/RGB and call it
    immediately before replay to verify that an automatic approval is still
    justified.
    """

    if not isinstance(proposal, Mapping):
        raise TypeError("proposal must be a mapping")
    validate_proposal(proposal)
    cfg = select_automatic_config(config)
    operation = str(proposal["operation_type"])
    proposal_id = str(proposal["proposal_id"])
    tile_id = str(proposal["tile_id"])

    if operation not in _SUPPORTED_AUTOMATIC_OPERATIONS:
        reason = (
            "semantic_split_requires_human_review"
            if operation == "semantic_split"
            else (
                "split_components_uses_embedded_deterministic_decision"
                if operation == "split_components"
                else "operation_not_supported_by_automatic_policy"
            )
        )
        return AutomaticProposalEvaluation(
            proposal_id=proposal_id,
            tile_id=tile_id,
            operation_type=operation,
            eligible=False,
            reasons=(reason,),
            threshold_snapshot=_threshold_snapshot(cfg, operation),
            observations={},
        )

    if operation == "merge":
        reasons, observations = _evaluate_merge(proposal, cfg)
    else:
        reasons, observations = _evaluate_absorb(proposal, cfg)
    ordered = tuple(sorted(set(reasons)))
    return AutomaticProposalEvaluation(
        proposal_id=proposal_id,
        tile_id=tile_id,
        operation_type=operation,
        eligible=not ordered,
        reasons=ordered,
        threshold_snapshot=_threshold_snapshot(cfg, operation),
        observations=json_safe(observations),
    )


def select_automatic_decisions(
    proposals: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any] | None = None,
) -> AutomaticSelectionResult:
    """Select a deterministic, topology-compatible set of auto approvals.

    Pending merge/absorb proposals that fail a gate or lose a conflict become
    ``uncertain`` decisions.  Multiple absorbs may share one target, matching
    topology replay's reusable-target rule, but a source remains exclusive.
    Proposals with an embedded non-pending decision are not overridden.
    """

    if isinstance(proposals, (str, bytes, bytearray)) or not isinstance(
        proposals, Sequence
    ):
        raise TypeError("proposals must be a sequence of proposal mappings")
    cfg = select_automatic_config(config)
    ordered_proposals: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, value in enumerate(proposals):
        if not isinstance(value, Mapping):
            raise TypeError(f"proposals[{index}] must be a mapping")
        validate_proposal(value)
        proposal_id = str(value["proposal_id"])
        if proposal_id in seen_ids:
            raise ValueError(f"duplicate automatic proposal ID {proposal_id!r}")
        seen_ids.add(proposal_id)
        ordered_proposals.append(copy.deepcopy(dict(value)))
    ordered_proposals.sort(key=lambda value: str(value["proposal_id"]))

    evaluations: dict[str, AutomaticProposalEvaluation] = {}
    final_reasons: dict[str, tuple[str, ...]] = {}
    final_decisions: dict[str, str] = {}
    passthrough: list[dict[str, str]] = []
    eligible: list[dict[str, Any]] = []
    # Reserve participants used by embedded semantic operations before
    # considering external approvals. Otherwise a selected merge could touch
    # the target/source of an existing auto-apply unknown absorb and make the
    # common topology replay reject the whole tile.
    roles: dict[tuple[str, str], list[tuple[str, str]]] = {}

    for proposal in ordered_proposals:
        proposal_id = str(proposal["proposal_id"])
        operation = str(proposal["operation_type"])
        embedded = str(proposal["decision"])
        if operation in _PASSTHROUGH_OPERATIONS or embedded != "pending":
            passthrough.append(
                {
                    "proposal_id": proposal_id,
                    "operation_type": operation,
                    "embedded_decision": embedded,
                }
            )
            if (
                operation in _SUPPORTED_AUTOMATIC_OPERATIONS
                and embedded in {"approved", "auto_apply"}
            ):
                conflicts = _proposal_conflicts(proposal, roles)
                if conflicts:
                    raise ValueError(
                        "Embedded refinement proposals conflict before automatic "
                        f"selection: {sorted(conflicts)} and {proposal_id!r}"
                    )
                _reserve_roles(proposal, roles)
            continue
        evaluation = evaluate_automatic_proposal(proposal, cfg)
        evaluations[proposal_id] = evaluation
        if evaluation.eligible:
            eligible.append(proposal)
        else:
            final_decisions[proposal_id] = "uncertain"
            final_reasons[proposal_id] = evaluation.reasons

    selected_ids: set[str] = set()
    selected_count_by_tile: dict[str, int] = {}
    # Roles are scoped by tile.  Entity IDs normally contain the tile already,
    # but keeping the scope explicit avoids false conflicts in synthetic data.
    maximum = int(cfg["maximum_selected_operations_per_tile"])
    for proposal in sorted(eligible, key=_selection_order_key):
        proposal_id = str(proposal["proposal_id"])
        tile_id = str(proposal["tile_id"])
        if selected_count_by_tile.get(tile_id, 0) >= maximum:
            final_decisions[proposal_id] = "uncertain"
            final_reasons[proposal_id] = (
                "maximum_selected_operations_per_tile_reached",
            )
            continue

        conflicts = _proposal_conflicts(proposal, roles)
        if conflicts:
            final_decisions[proposal_id] = "uncertain"
            final_reasons[proposal_id] = tuple(
                f"conflicts_with_selected_proposal:{value}"
                for value in sorted(conflicts)
            )
            continue

        _reserve_roles(proposal, roles)
        selected_ids.add(proposal_id)
        selected_count_by_tile[tile_id] = selected_count_by_tile.get(tile_id, 0) + 1
        final_decisions[proposal_id] = "approved"
        final_reasons[proposal_id] = ("all_automatic_evidence_gates_passed",)

    decision_records: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    proposal_by_id = {
        str(proposal["proposal_id"]): proposal for proposal in ordered_proposals
    }
    for proposal_id in sorted(final_decisions):
        evaluation = evaluations[proposal_id]
        decision = final_decisions[proposal_id]
        reasons = final_reasons[proposal_id]
        reason_document = {
            "outcome": decision,
            "policy_version": cfg["policy_version"],
            "reasons": list(reasons),
            "threshold_snapshot": evaluation.threshold_snapshot,
        }
        decision_records.append(
            make_decision(
                proposal_id,
                decision,
                decision_source=AUTOMATIC_DECISION_SOURCE,
                reason=canonical_json(reason_document),
            )
        )
        audit = evaluation.to_dict()
        audit.update(
            {
                "final_decision": decision,
                "final_reasons": list(reasons),
                "selected": proposal_id in selected_ids,
                "source_entity_ids": copy.deepcopy(
                    proposal_by_id[proposal_id]["source_entity_ids"]
                ),
                "target_entity_id": proposal_by_id[proposal_id].get(
                    "target_entity_id"
                ),
            }
        )
        audit_rows.append(json_safe(audit))

    counts_by_operation: dict[str, dict[str, int]] = {}
    reason_counts: dict[str, int] = {}
    for row in audit_rows:
        operation_counts = counts_by_operation.setdefault(
            str(row["operation_type"]), {"approved": 0, "uncertain": 0}
        )
        outcome = str(row["final_decision"])
        operation_counts[outcome] = operation_counts.get(outcome, 0) + 1
        for reason in row["final_reasons"]:
            reason_key = str(reason)
            reason_counts[reason_key] = reason_counts.get(reason_key, 0) + 1
    report = json_safe(
        {
            "policy_version": cfg["policy_version"],
            "decision_source": AUTOMATIC_DECISION_SOURCE,
            "proposal_count": len(ordered_proposals),
            "decision_count": len(decision_records),
            "approved_count": len(selected_ids),
            "abstained_count": sum(
                value == "uncertain" for value in final_decisions.values()
            ),
            "passthrough_count": len(passthrough),
            "maximum_selected_operations_per_tile": maximum,
            "selected_count_by_tile": dict(sorted(selected_count_by_tile.items())),
            "selected_proposal_ids": sorted(selected_ids),
            "counts_by_operation": dict(sorted(counts_by_operation.items())),
            "reason_counts": dict(sorted(reason_counts.items())),
            "threshold_snapshots": {
                operation: _threshold_snapshot(cfg, operation)
                for operation in ("absorb", "merge")
            },
            "passthrough_proposals": passthrough,
            "evaluations": audit_rows,
        }
    )
    return AutomaticSelectionResult(decisions=decision_records, report=report)


def _evaluate_merge(
    proposal: Mapping[str, Any], cfg: Mapping[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    operation_cfg = cfg["merge"]
    evidence = proposal["evidence"]
    parameters = proposal["parameters"]
    reasons: list[str] = []
    observations: dict[str, Any] = {}

    if operation_cfg["enabled"] is not True:
        reasons.append("automatic_merge_disabled")
    if len(proposal["source_entity_ids"]) != 2:
        reasons.append("automatic_merge_requires_exactly_two_sources")
    _common_policy_evidence(evidence, reasons, observations)

    class_name = _nonempty_text(parameters.get("class_name"))
    observations["class_name"] = class_name
    if class_name is None:
        reasons.append("class_name_unavailable")
    elif class_name in cfg["protected_classes"]:
        reasons.append("protected_class_requires_human_review")
    if evidence.get("semantic_class_match") is not True:
        reasons.append("same_class_evidence_not_confirmed")

    _score_gate(proposal.get("score"), operation_cfg["minimum_score"], reasons, observations)
    _adjacency_gate(evidence, reasons, observations)
    _minimum_integer_gate(
        evidence,
        "shared_boundary_side_count",
        operation_cfg["minimum_shared_boundary_pixels"],
        "shared_boundary_pixels",
        reasons,
        observations,
    )
    _minimum_fraction_gate(
        evidence,
        "minimum_perimeter_shared_boundary_fraction",
        operation_cfg["minimum_shared_boundary_fraction"],
        "shared_boundary_fraction",
        reasons,
        observations,
    )
    _appearance_gates(evidence, operation_cfg, reasons, observations)
    if operation_cfg["require_not_tile_boundary"]:
        _merge_tile_boundary_gate(evidence, reasons, observations)
    return reasons, observations


def _evaluate_absorb(
    proposal: Mapping[str, Any], cfg: Mapping[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    operation_cfg = cfg["absorb"]
    evidence = proposal["evidence"]
    parameters = proposal["parameters"]
    reasons: list[str] = []
    observations: dict[str, Any] = {}

    if operation_cfg["enabled"] is not True:
        reasons.append("automatic_absorb_disabled")
    _common_policy_evidence(evidence, reasons, observations)
    if parameters.get("automatic_unknown_cleanup") is True:
        reasons.append("automatic_unknown_cleanup_uses_embedded_rule")

    source_class = _nonempty_text(parameters.get("class_name"))
    target_class = _nonempty_text(parameters.get("target_class_name"))
    observations.update(
        {"source_class_name": source_class, "target_class_name": target_class}
    )
    if source_class is None or target_class is None:
        reasons.append("source_or_target_class_name_unavailable")
    else:
        if source_class != target_class:
            reasons.append("automatic_absorb_requires_same_class")
        if source_class in cfg["protected_classes"] or target_class in cfg["protected_classes"]:
            reasons.append("protected_class_requires_human_review")
    if parameters.get("same_class_required") is not True:
        reasons.append("same_class_requirement_not_declared")
    if parameters.get("cross_class_relabel") is not False:
        reasons.append("cross_class_relabel_not_explicitly_false")

    _score_gate(proposal.get("score"), operation_cfg["minimum_score"], reasons, observations)
    _adjacency_gate(evidence, reasons, observations)
    _maximum_integer_gate(
        evidence,
        ("candidate_record_area_pixels", "candidate_area_pixels"),
        operation_cfg["maximum_source_area_pixels"],
        "source_area_pixels",
        reasons,
        observations,
    )
    _minimum_integer_gate(
        evidence,
        "shared_boundary_side_count",
        operation_cfg["minimum_shared_boundary_pixels"],
        "shared_boundary_pixels",
        reasons,
        observations,
    )
    _minimum_fraction_gate(
        evidence,
        "candidate_shared_boundary_fraction",
        operation_cfg["minimum_shared_boundary_fraction"],
        "shared_boundary_fraction",
        reasons,
        observations,
    )
    _minimum_fraction_gate(
        evidence,
        (
            "best_to_runner_up_boundary_fraction_margin",
            "best_to_runner_up_margin",
        ),
        operation_cfg["minimum_best_to_runner_up_margin"],
        "best_to_runner_up_margin",
        reasons,
        observations,
    )
    _appearance_gates(evidence, operation_cfg, reasons, observations)
    if operation_cfg["require_not_tile_boundary"]:
        _single_tile_boundary_gate(evidence, reasons, observations)
    return reasons, observations


def _common_policy_evidence(
    evidence: Mapping[str, Any],
    reasons: list[str],
    observations: dict[str, Any],
) -> None:
    observations["policy_gate_passed"] = evidence.get("policy_gate_passed")
    if evidence.get("policy_gate_passed") is not True:
        reasons.append("proposal_policy_gate_not_confirmed")
    vetoes = evidence.get("hard_vetoes")
    if isinstance(vetoes, Sequence) and not isinstance(
        vetoes, (str, bytes, bytearray)
    ):
        observations["hard_vetoes"] = [str(value) for value in vetoes]
        if vetoes:
            reasons.append("proposal_contains_hard_vetoes")
    else:
        observations["hard_vetoes"] = None
        reasons.append("hard_veto_evidence_unavailable")


def _score_gate(
    raw_score: Any,
    minimum: float,
    reasons: list[str],
    observations: dict[str, Any],
) -> None:
    score = _optional_finite_number(raw_score)
    observations["score"] = score
    if score is None:
        reasons.append("score_unavailable_or_invalid")
    elif score < float(minimum):
        reasons.append("score_below_automatic_minimum")


def _adjacency_gate(
    evidence: Mapping[str, Any],
    reasons: list[str],
    observations: dict[str, Any],
) -> None:
    adjacency = evidence.get("adjacency_definition")
    observations["adjacency_definition"] = adjacency
    if adjacency != "shared_pixel_edge_4_connected":
        reasons.append("adjacency_is_not_shared_pixel_edge_4_connected")


def _appearance_gates(
    evidence: Mapping[str, Any],
    operation_cfg: Mapping[str, Any],
    reasons: list[str],
    observations: dict[str, Any],
) -> None:
    _minimum_fraction_gate(
        evidence,
        "weak_color_boundary_score",
        operation_cfg["minimum_weak_color_boundary_score"],
        "weak_color_boundary_score",
        reasons,
        observations,
    )
    _minimum_fraction_gate(
        evidence,
        "region_mean_color_similarity",
        operation_cfg["minimum_region_mean_color_similarity"],
        "region_mean_color_similarity",
        reasons,
        observations,
    )


def _merge_tile_boundary_gate(
    evidence: Mapping[str, Any],
    reasons: list[str],
    observations: dict[str, Any],
) -> None:
    first = evidence.get("source_touches_tile_boundary")
    second = evidence.get("target_touches_tile_boundary")
    observations["source_touches_tile_boundary"] = first
    observations["target_touches_tile_boundary"] = second
    if not isinstance(first, bool) or not isinstance(second, bool):
        reasons.append("tile_boundary_evidence_unavailable")
    elif first or second:
        reasons.append("tile_boundary_proposal_requires_human_review")


def _single_tile_boundary_gate(
    evidence: Mapping[str, Any],
    reasons: list[str],
    observations: dict[str, Any],
) -> None:
    touches = evidence.get("source_touches_tile_boundary")
    observations["source_touches_tile_boundary"] = touches
    if not isinstance(touches, bool):
        reasons.append("tile_boundary_evidence_unavailable")
    elif touches:
        reasons.append("tile_boundary_proposal_requires_human_review")


def _minimum_integer_gate(
    evidence: Mapping[str, Any],
    raw_keys: str | tuple[str, ...],
    minimum: int,
    observation_name: str,
    reasons: list[str],
    observations: dict[str, Any],
) -> None:
    value = _first_value(evidence, raw_keys)
    observed = _optional_nonnegative_integer(value)
    observations[observation_name] = observed
    if observed is None:
        reasons.append(f"{observation_name}_unavailable_or_invalid")
    elif observed < int(minimum):
        reasons.append(f"{observation_name}_below_automatic_minimum")


def _maximum_integer_gate(
    evidence: Mapping[str, Any],
    raw_keys: str | tuple[str, ...],
    maximum: int,
    observation_name: str,
    reasons: list[str],
    observations: dict[str, Any],
) -> None:
    value = _first_value(evidence, raw_keys)
    observed = _optional_nonnegative_integer(value)
    observations[observation_name] = observed
    if observed is None:
        reasons.append(f"{observation_name}_unavailable_or_invalid")
    elif observed > int(maximum):
        reasons.append(f"{observation_name}_above_automatic_maximum")


def _minimum_fraction_gate(
    evidence: Mapping[str, Any],
    raw_keys: str | tuple[str, ...],
    minimum: float,
    observation_name: str,
    reasons: list[str],
    observations: dict[str, Any],
) -> None:
    value = _first_value(evidence, raw_keys)
    observed = _optional_fraction(value)
    observations[observation_name] = observed
    if observed is None:
        reasons.append(f"{observation_name}_unavailable_or_invalid")
    elif observed < float(minimum):
        reasons.append(f"{observation_name}_below_automatic_minimum")


def _selection_order_key(proposal: Mapping[str, Any]) -> tuple[Any, ...]:
    evidence = proposal["evidence"]
    operation = str(proposal["operation_type"])
    shared_fraction_key = (
        "candidate_shared_boundary_fraction"
        if operation == "absorb"
        else "minimum_perimeter_shared_boundary_fraction"
    )
    return (
        str(proposal["tile_id"]),
        0 if operation == "absorb" else 1,
        -float(evidence[shared_fraction_key]),
        -float(evidence["weak_color_boundary_score"]),
        -float(evidence["region_mean_color_similarity"]),
        -float(proposal["score"]),
        -int(evidence["shared_boundary_side_count"]),
        str(proposal["proposal_id"]),
    )


def _proposal_conflicts(
    proposal: Mapping[str, Any],
    roles: Mapping[tuple[str, str], list[tuple[str, str]]],
) -> set[str]:
    tile_id = str(proposal["tile_id"])
    operation = str(proposal["operation_type"])
    conflicts: set[str] = set()
    for source in proposal["source_entity_ids"]:
        conflicts.update(value for _, value in roles.get((tile_id, str(source)), ()))
    target = proposal.get("target_entity_id")
    if target is not None:
        prior = roles.get((tile_id, str(target)), ())
        if operation != "absorb" or any(role != "absorb_target" for role, _ in prior):
            conflicts.update(value for _, value in prior)
    return conflicts


def _reserve_roles(
    proposal: Mapping[str, Any],
    roles: dict[tuple[str, str], list[tuple[str, str]]],
) -> None:
    tile_id = str(proposal["tile_id"])
    proposal_id = str(proposal["proposal_id"])
    for source in proposal["source_entity_ids"]:
        roles.setdefault((tile_id, str(source)), []).append(
            ("exclusive_source", proposal_id)
        )
    target = proposal.get("target_entity_id")
    if target is not None:
        roles.setdefault((tile_id, str(target)), []).append(
            ("absorb_target", proposal_id)
        )


def _threshold_snapshot(
    cfg: Mapping[str, Any], operation: str
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "policy_version": cfg["policy_version"],
        "protected_classes": copy.deepcopy(cfg["protected_classes"]),
    }
    if operation in _SUPPORTED_AUTOMATIC_OPERATIONS:
        snapshot[operation] = copy.deepcopy(dict(cfg[operation]))
    else:
        snapshot["automatic_approval_supported"] = False
    return json_safe(snapshot)


def _first_value(mapping: Mapping[str, Any], keys: str | tuple[str, ...]) -> Any:
    candidates = (keys,) if isinstance(keys, str) else keys
    for key in candidates:
        if key in mapping:
            return mapping[key]
    return None


def _nonempty_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _optional_finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _optional_fraction(value: Any) -> float | None:
    result = _optional_finite_number(value)
    if result is None or not 0.0 <= result <= 1.0:
        return None
    return result


def _optional_nonnegative_integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], path: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing {missing}")
        if extra:
            details.append(f"unexpected {extra}")
        raise ValueError(f"{path} has invalid keys: {'; '.join(details)}")


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value


def _positive_integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _fraction(value: Any, path: str) -> float:
    result = _optional_fraction(value)
    if result is None:
        raise ValueError(f"{path} must be a finite number between 0 and 1")
    return result


def _deep_merge(
    target: dict[str, Any], override: Mapping[str, Any], path: str
) -> None:
    for key, value in override.items():
        if key not in target:
            raise ValueError(f"Unknown configuration key: {path}.{key}")
        if isinstance(target[key], Mapping):
            nested = _mapping(value, f"{path}.{key}")
            _deep_merge(target[key], nested, f"{path}.{key}")
        else:
            target[key] = copy.deepcopy(value)


__all__ = [
    "AUTOMATIC_DECISION_SOURCE",
    "AUTOMATIC_POLICY_VERSION",
    "AutomaticProposalEvaluation",
    "AutomaticSelectionResult",
    "DEFAULT_AUTOMATIC_CONFIG",
    "default_automatic_config",
    "evaluate_automatic_proposal",
    "select_automatic_config",
    "select_automatic_decisions",
    "validate_automatic_config",
]
