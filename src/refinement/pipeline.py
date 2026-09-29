"""Reversible Stage 2R segmentation refinement orchestration.

The public workflow deliberately separates proposal generation from mutation:

``propose_refinement``
    Builds deterministic topology, merge, and absorption proposals without
    publishing a consumable refined segmentation.

``apply_refinement``
    Replays deterministic connectivity normalization plus explicitly approved
    non-conflicting operations from the immutable Stage 2 ``12_final`` input.

Stage 3 consumes only a completed result from ``13_refinement``.
"""

from __future__ import annotations

import copy
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.config import DEFAULT_CONFIG, load_config, select_refinement_config, validate_config
from src.stage2 import load_stage2_result
from src.stage2.artifacts import checksum, manifest, now, require_fresh
from src.stage2.pipeline import SCHEMA_VERSION as STAGE2_SCHEMA_VERSION
from src.stage2.storage import load_masks, save_masks

from .automatic import select_automatic_decisions
from .config import config_hash, normalize_refinement_mode
from .contextual_cleanup import evaluate_contextual_micro_cleanup
from .evidence import pair_evidence
from .policy import (
    evaluate_absorb_candidate,
    evaluate_merge_candidate,
    orient_absorb_evidence,
)
from .review import save_review_bundle
from .schema import SCHEMA_VERSION, make_proposal, stable_digest, validate_proposal
from .storage import (
    atomic_json,
    atomic_npz,
    load_decisions,
    load_json,
    load_refinement_history,
    load_segmentation_bundle,
    save_decisions,
    save_proposals,
    save_refinement_history,
    save_segmentation_bundle,
    sha256_array,
    sha256_file,
)
from .topology import (
    RefinementResult as CoreRefinementResult,
    apply_reviewed_operations,
    consolidate_micro_components,
    normalize_4_connected_components,
    omit_micro_unknowns,
)
from .unknown_cleanup import evaluate_unknown_cleanup_candidates
from .validation import ValidationReport, validate_refined_segmentation
from .visualization import save_refinement_visualizations


@dataclass(slots=True)
class RefinementProposalResult:
    """Proposal-only result; no completed segmentation has been published."""

    tile_id: str
    output_directory: Path
    proposals: list[dict[str, Any]]
    report: dict[str, Any]
    artifacts: dict[str, str]

    def summary(self) -> dict[str, Any]:
        return {
            "status": "proposed",
            "schema_version": SCHEMA_VERSION,
            "tile_id": self.tile_id,
            "output_directory": str(self.output_directory),
            "proposal_count": len(self.proposals),
            "proposal_counts": copy.deepcopy(self.report.get("proposal_counts", {})),
            "artifacts": copy.deepcopy(self.artifacts),
        }


@dataclass(slots=True)
class RefinementResult:
    """Loaded, validated Stage 2R segmentation result."""

    tile_id: str
    output_directory: Path
    masks: list[np.ndarray]
    segments: list[dict[str, Any]]
    instance_map: np.ndarray
    semantic_map: np.ndarray
    confidence_map: np.ndarray
    source_map: np.ndarray
    valid_mask: np.ndarray
    refinement_operation_map: np.ndarray
    proposals: list[dict[str, Any]]
    decisions: list[dict[str, Any]]
    operations: list[dict[str, Any]]
    lineage: dict[str, Any]
    tile_metadata: dict[str, Any]
    report: dict[str, Any]
    artifacts: dict[str, str]

    @property
    def records(self) -> list[dict[str, Any]]:
        """Alias used by the pure topology helpers."""

        return self.segments

    def summary(self) -> dict[str, Any]:
        comparison = self.report.get("comparison", {})
        return {
            "status": "complete",
            "schema_version": SCHEMA_VERSION,
            "tile_id": self.tile_id,
            "output_directory": str(self.output_directory),
            "segment_count": len(self.segments),
            "operation_count": len(self.operations),
            "proposal_count": len(self.proposals),
            "workflow_mode": self.report.get("workflow_mode", "human_review"),
            "automatic_approved_count": int(
                self.report.get("automatic_selection", {}).get("approved_count", 0)
            ),
            "automatic_abstained_count": int(
                self.report.get("automatic_selection", {}).get("abstained_count", 0)
            ),
            "connectivity_split_count": int(
                self.report.get("operation_counts", {}).get("split_components", 0)
            ),
            "internal_connected_component_count": int(
                self.report.get("object_identity", {}).get(
                    "internal_connected_component_count", len(self.segments)
                )
            ),
            "consolidated_micro_component_count": int(
                self.report.get("object_identity", {}).get(
                    "consolidated_micro_component_count", 0
                )
            ),
            "automatic_unknown_absorb_count": int(
                sum(
                    operation.get("operation_type") == "absorb"
                    and operation.get("parameters", {}).get(
                        "automatic_unknown_cleanup"
                    )
                    is True
                    for operation in self.operations
                )
            ),
            "unknown_pixel_count": comparison.get("refined_unknown_pixel_count"),
            "unknown_pixel_delta": comparison.get("unknown_pixel_delta"),
            "artifacts": copy.deepcopy(self.artifacts),
        }


@dataclass(slots=True)
class _Stage2Source:
    directory: Path
    manifest: dict[str, Any]
    result: Any
    required_inputs: list[Path]
    rgb: np.ndarray | None
    rgb_path: Path | None
    rgb_is_authoritative: bool
    semantic_probabilities: np.ndarray | None
    semantic_probabilities_path: Path | None


def propose_refinement(
    tile_id: str,
    *,
    project_root: str | Path = ".",
    config: Mapping[str, Any] | str | Path | None = None,
    show_progress: bool = True,
) -> RefinementProposalResult:
    """Generate deterministic, non-mutating refinement proposals for one tile."""

    root = Path(project_root).expanduser().resolve()
    project_config = _load_project_config(config)
    refinement_config = select_refinement_config(project_config)
    progress = _RefinementProgress(show_progress, total=3, action="Propose")
    try:
        progress.phase("Load Stage 2")
        source = _load_stage2_source(root, tile_id, refinement_config)
        progress.advance()

        progress.phase("Normalize topology")
        normalized = _normalize(source, refinement_config)
        progress.advance()

        progress.phase("Build review proposals")
        proposals, proposal_report = _build_proposals(
            normalized,
            source.rgb if source.rgb_is_authoritative else None,
            source.semantic_probabilities,
            refinement_config,
            workflow_mode="human_review",
        )
        output_directory = _output_directory(root, source.result.tile_id, refinement_config)
        artifacts, report = _persist_proposal_result(
            output_directory,
            source,
            normalized,
            proposals,
            proposal_report,
            refinement_config,
        )
        progress.advance()
        progress.complete()
    finally:
        progress.close()
    return RefinementProposalResult(
        tile_id=source.result.tile_id,
        output_directory=output_directory,
        proposals=proposals,
        report=report,
        artifacts=artifacts,
    )


def apply_refinement(
    tile_id: str,
    *,
    project_root: str | Path = ".",
    config: Mapping[str, Any] | str | Path | None = None,
    decisions: Mapping[str, Any] | Sequence[Mapping[str, Any]] | str | Path | None = None,
    workflow_mode: str | None = None,
    show_progress: bool = True,
) -> RefinementResult:
    """Replay one tile with human-reviewed or automatic selective decisions.

    Automatic mode is fully unattended but not forceful: candidates which do
    not pass every configured evidence gate are recorded as ``uncertain`` and
    leave their semantic object geometry unchanged. Topology-only
    micro-components are still consolidated under their parent object identity
    before publication in both modes.
    """

    root = Path(project_root).expanduser().resolve()
    project_config = _load_project_config(config)
    refinement_config = select_refinement_config(project_config)
    selected_mode = normalize_refinement_mode(
        refinement_config["mode"] if workflow_mode is None else workflow_mode
    )
    if selected_mode == "automatic" and decisions is not None:
        raise ValueError(
            "Automatic refinement generates its own decisions; omit reviewed decisions "
            "or use workflow_mode='human_review'"
        )
    started = now()
    progress = _RefinementProgress(show_progress, total=4, action="Apply")
    try:
        progress.phase("Load Stage 2")
        source = _load_stage2_source(root, tile_id, refinement_config)
        progress.advance()

        progress.phase("Regenerate proposals")
        normalized = _normalize(source, refinement_config)
        # Human and automatic modes must see exactly the same proposal evidence.
        # A rendered overlay is useful for display but is not authoritative RGB.
        evidence_rgb = source.rgb if source.rgb_is_authoritative else None
        proposals, proposal_report = _build_proposals(
            normalized,
            evidence_rgb,
            source.semantic_probabilities,
            refinement_config,
            workflow_mode=selected_mode,
        )
        output_directory = _output_directory(root, source.result.tile_id, refinement_config)
        automatic_selection_report: dict[str, Any] | None = None
        if selected_mode == "automatic":
            automatic_selection = select_automatic_decisions(
                proposals,
                refinement_config["automatic"],
            )
            decision_records = automatic_selection.decisions
            automatic_selection_report = automatic_selection.report
            automatic_selection_report["source_evidence"] = {
                "rgb_available": source.rgb is not None,
                "rgb_authoritative": source.rgb_is_authoritative,
                "semantic_probabilities_available": (
                    source.semantic_probabilities is not None
                ),
            }
        else:
            current_proposal_checksum = _proposal_document_checksum(
                source,
                proposals,
                refinement_config,
            )
            decision_records = _load_decision_records(
                decisions,
                expected_tile_id=source.result.tile_id,
                expected_proposal_checksum=current_proposal_checksum,
            )
        progress.advance()

        progress.phase("Replay approved edits")
        semantic_candidates = [
            proposal
            for proposal in proposals
            if proposal["operation_type"] != "split_components"
        ]
        proposal_by_id = {
            str(proposal["proposal_id"]): proposal for proposal in proposals
        }
        decision_ids = [
            str(decision.get("proposal_id") or decision.get("operation_id") or "")
            for decision in decision_records
        ]
        if any(not proposal_id for proposal_id in decision_ids):
            raise ValueError("Every refinement decision must identify its proposal")
        if len(set(decision_ids)) != len(decision_ids):
            raise ValueError("Refinement decisions contain duplicate proposal IDs")
        unknown_decisions = sorted(set(decision_ids) - set(proposal_by_id))
        if unknown_decisions:
            raise ValueError(
                "Decisions reference unknown refinement proposals: "
                f"{unknown_decisions}"
            )
        connectivity_decisions = sorted(
            proposal_id
            for proposal_id in decision_ids
            if proposal_by_id[proposal_id]["operation_type"] == "split_components"
        )
        if connectivity_decisions:
            raise ValueError(
                "Deterministic split_components proposals are non-reviewable and "
                f"must not appear in a decisions document: {connectivity_decisions}"
            )
        refined = apply_reviewed_operations(
            normalized,
            semantic_candidates,
            decision_records,
            tile_metadata=source.result.tile_metadata,
            semantic_probabilities=source.semantic_probabilities,
            rgb_image=evidence_rgb,
            policy_config=refinement_config,
        )
        if selected_mode == "automatic":
            refined = omit_micro_unknowns(
                refined, maximum_area_pixels=40,
                semantic_probabilities=source.semantic_probabilities,
                tile_metadata=source.result.tile_metadata,
            )
        refined = consolidate_micro_components(
            refined,
            minimum_object_component_area_pixels=int(
                refinement_config["operations"]["connectivity_split"][
                    "minimum_component_area_pixels"
                ]
            ),
            source_stage2_segment_count=len(source.result.segments),
            tile_metadata=source.result.tile_metadata,
        )
        _attach_ontology(refined.records, refinement_config)
        require_connected = bool(
            refined.report.get("object_identity", {}).get(
                "require_every_object_four_connected", True
            )
        )
        validation = validate_refined_segmentation(
            refined.instance_map,
            refined.semantic_map,
            refined.valid_mask,
            refined.records,
            refined.masks,
            confidence_map=refined.confidence_map,
            source_map=refined.source_map,
            require_four_connected=require_connected,
        )
        progress.advance()

        progress.phase("Persist validated result")
        _persist_complete_result(
            output_directory=output_directory,
            source=source,
            refined=refined,
            proposals=proposals,
            decisions=decision_records,
            proposal_report=proposal_report,
            validation=validation,
            refinement_config=refinement_config,
            workflow_mode=selected_mode,
            automatic_selection_report=automatic_selection_report,
            started=started,
        )
        progress.advance()
        progress.complete()
    finally:
        progress.close()
    return load_refinement_result(output_directory)


def load_refinement_result(output_directory: str | Path) -> RefinementResult:
    """Load and validate one completed, fresh Stage 2R result."""

    directory = Path(output_directory).expanduser().resolve()
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Refinement manifest not found: {manifest_path}")
    public_manifest = load_json(manifest_path)
    if public_manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported refinement result schema: {public_manifest.get('schema_version')!r}"
        )
    if public_manifest.get("status") != "complete":
        raise ValueError(f"Refinement result is not complete: {public_manifest.get('status')!r}")
    if public_manifest.get("stale"):
        raise RuntimeError(f"Refinement result is stale: {directory}")
    _validate_source_inputs(public_manifest.get("source_stage2", {}).get("inputs", []))

    required = {
        "segments": directory / "segments.json",
        "masks": directory / "segments.npz",
        "segmentation_maps": directory / "segmentation.npz",
        "refinement_operation_map": directory / "refinement_operation_map.npz",
        "tile_metadata": directory / "tile_metadata.json",
        "class_registry": directory / "class_registry.json",
        "proposals": directory / "proposals.json",
        "decisions": directory / "decisions.json",
        "operations": directory / "operations.json",
        "lineage": directory / "lineage.json",
        "report": directory / "report.json",
        "refined_segmentation": directory / "refined_segmentation.png",
        "refined_overlay": directory / "refined_overlay.png",
        "before_after": directory / "refinement_before_after.png",
        "phase_manifest": directory / "phase_manifest.json",
    }
    artifacts = {
        str(key): str(value) for key, value in public_manifest.get("artifacts", {}).items()
    }
    for key, path in required.items():
        if key not in artifacts or Path(artifacts[key]).expanduser().resolve() != path.resolve():
            raise ValueError(f"Refinement manifest has an invalid artifact entry for {key!r}")
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Refinement artifact is missing or empty: {path}")
        expected = public_manifest.get("artifact_checksums", {}).get(key)
        if not expected or checksum(path) != expected:
            raise RuntimeError(f"Refinement artifact checksum mismatch: {path}")

    masks, segments = load_masks(required["masks"], required["segments"])
    maps = load_segmentation_bundle(required["segmentation_maps"])
    with np.load(required["refinement_operation_map"], allow_pickle=False) as archive:
        operation_map = np.asarray(archive["refinement_operation_map"], dtype=np.int32)
    proposal_document = load_json(required["proposals"])
    decision_document = load_decisions(required["decisions"])
    operation_document = load_refinement_history(required["operations"])
    lineage = load_json(required["lineage"])
    tile_metadata = load_json(required["tile_metadata"])
    report = load_json(required["report"])
    tile_id = str(public_manifest.get("tile_id", ""))
    if not tile_id or str(tile_metadata.get("tile_id")) != tile_id:
        raise ValueError("Refinement manifest and tile metadata disagree")
    if operation_map.shape != maps["instance_map"].shape:
        raise ValueError("refinement_operation_map shape disagrees with segmentation maps")

    validate_refined_segmentation(
        maps["instance_map"],
        maps["semantic_map"],
        maps["valid_mask"],
        segments,
        masks,
        confidence_map=maps["confidence_map"],
        source_map=maps["source_map"],
        require_four_connected=bool(
            report.get("object_identity", {}).get(
                "require_every_object_four_connected", True
            )
        ),
    )
    artifacts["manifest"] = str(manifest_path.resolve())
    return RefinementResult(
        tile_id=tile_id,
        output_directory=directory,
        masks=masks,
        segments=segments,
        instance_map=maps["instance_map"],
        semantic_map=maps["semantic_map"],
        confidence_map=maps["confidence_map"],
        source_map=maps["source_map"],
        valid_mask=maps["valid_mask"],
        refinement_operation_map=operation_map,
        proposals=[copy.deepcopy(dict(value)) for value in proposal_document["proposals"]],
        decisions=[copy.deepcopy(dict(value)) for value in decision_document["decisions"]],
        operations=[copy.deepcopy(dict(value)) for value in operation_document["operations"]],
        lineage=lineage,
        tile_metadata=tile_metadata,
        report=report,
        artifacts=artifacts,
    )


def validate_refinement_result(
    result_or_directory: RefinementResult | str | Path,
) -> ValidationReport:
    """Validate a loaded result or load and validate a completed directory."""

    result = (
        load_refinement_result(result_or_directory)
        if isinstance(result_or_directory, (str, Path))
        else result_or_directory
    )
    require_connected = bool(
        result.report.get("object_identity", {}).get(
            "require_every_object_four_connected", True
        )
    )
    return validate_refined_segmentation(
        result.instance_map,
        result.semantic_map,
        result.valid_mask,
        result.segments,
        result.masks,
        confidence_map=result.confidence_map,
        source_map=result.source_map,
        require_four_connected=require_connected,
    )


def validate_refinement(
    tile_id: str,
    *,
    project_root: str | Path = ".",
    config: Mapping[str, Any] | str | Path | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """CLI-oriented validation entry point for a configured tile."""

    del show_progress
    root = Path(project_root).expanduser().resolve()
    selected = select_refinement_config(_load_project_config(config))
    directory = _output_directory(root, tile_id, selected)
    result = load_refinement_result(directory)
    validation = validate_refinement_result(result)
    return {
        "status": "valid",
        "schema_version": SCHEMA_VERSION,
        "tile_id": result.tile_id,
        "output_directory": str(result.output_directory),
        "validation": validation.to_dict(),
    }


def _normalize(source: _Stage2Source, cfg: Mapping[str, Any]) -> CoreRefinementResult:
    normalized = normalize_4_connected_components(
        source.result.instance_map,
        source.result.semantic_map,
        source.result.confidence_map,
        source.result.source_map,
        source.result.valid_mask,
        source.result.segments,
        tile_id=source.result.tile_id,
        tile_metadata=source.result.tile_metadata,
    )
    _attach_ontology(normalized.records, cfg)
    return normalized


def _attach_ontology(records: Sequence[dict[str, Any]], cfg: Mapping[str, Any]) -> None:
    for record in records:
        label = record.get("label", {})
        class_name = str(label.get("name", "unknown"))
        if class_name not in cfg["ontology"]:
            raise ValueError(f"Refined record uses an undefined ontology class: {class_name!r}")
        rule = copy.deepcopy(dict(cfg["ontology"][class_name]))
        record["entity_type"] = str(rule["entity_type"])
        record["entity_policy"] = str(rule["policy"])
        record["ontology"] = {
            **rule,
            "target_definition_validated": False,
            "validation_status": "topology_evaluated_object_identity_published",
        }
        if class_name == "road":
            record["segment_subtype"] = "road_reach_candidate"


def _build_proposals(
    normalized: CoreRefinementResult,
    rgb: np.ndarray | None,
    semantic_probabilities: np.ndarray | None,
    cfg: Mapping[str, Any],
    *,
    workflow_mode: str = "human_review",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    proposals: list[dict[str, Any]] = []
    policy_rejection_counts: dict[str, int] = {}

    def record_policy_rejection(reasons: Sequence[str]) -> None:
        for reason in reasons:
            key = str(reason)
            policy_rejection_counts[key] = policy_rejection_counts.get(key, 0) + 1

    connectivity_cfg = cfg["operations"]["connectivity_split"]
    if connectivity_cfg["enabled"]:
        for operation in normalized.operations:
            proposal = make_proposal(
                normalized.report["tile_id"],
                "split_components",
                operation["source_entity_ids"],
                parameters={
                    **copy.deepcopy(operation["parameters"]),
                    "output_entity_ids": copy.deepcopy(operation["output_entity_ids"]),
                    "deterministic_operation_id": operation["operation_id"],
                },
                evidence=copy.deepcopy(operation["evidence"]),
                score=1.0,
                decision=(
                    "auto_apply" if connectivity_cfg["auto_apply"] else "pending"
                ),
            )
            proposals.append(proposal)

    pair_rows: list[dict[str, Any]] = []
    if rgb is not None:
        pair_rows = pair_evidence(normalized.instance_map, rgb)
    by_instance = {int(record["instance_id"]): record for record in normalized.records}
    weights = cfg["evidence"]
    merge_cfg = cfg["operations"]["merge"]
    if merge_cfg["enabled"]:
        for row in pair_rows:
            first = by_instance[int(row["source_instance_id"])]
            second = by_instance[int(row["target_instance_id"])]
            first_label, second_label = first["label"], second["label"]
            same_class = int(first_label["id"]) == int(second_label["id"])
            class_name = str(first_label["name"])
            minimum_fraction = float(row["minimum_perimeter_shared_boundary_fraction"])
            if not same_class:
                continue
            semantic_score = 1.0
            boundary_score = float(row["weak_color_boundary_score"])
            appearance_score = float(row["region_mean_color_similarity"])
            shape_score = min(1.0, minimum_fraction / 0.25)
            score = (
                float(weights["semantic_weight"]) * semantic_score
                + float(weights["boundary_weight"]) * boundary_score
                + float(weights["appearance_weight"]) * appearance_score
                + float(weights["shape_weight"]) * shape_score
            )
            policy = evaluate_merge_candidate(
                cfg,
                first,
                second,
                row,
                score=score,
            )
            if not policy.eligible:
                record_policy_rejection((*policy.reasons, *policy.hard_vetoes))
                continue
            first_mask = normalized.instance_map == int(first["instance_id"])
            second_mask = normalized.instance_map == int(second["instance_id"])
            proposals.append(
                make_proposal(
                    normalized.report["tile_id"],
                    "merge",
                    [str(first["entity_id"]), str(second["entity_id"])],
                    parameters={
                        "class_name": class_name,
                        "entity_policy": first["entity_policy"],
                        "review_only": True,
                        "score_kind": "uncalibrated_review_priority",
                    },
                    evidence={
                        **copy.deepcopy(policy.evidence),
                        "source_touches_tile_boundary": _touches_tile_boundary(
                            first_mask
                        ),
                        "target_touches_tile_boundary": _touches_tile_boundary(
                            second_mask
                        ),
                        "semantic_class_match": True,
                        "hard_vetoes": list(policy.hard_vetoes),
                        "policy_gate_passed": True,
                    },
                    score=score,
                    decision="pending",
                )
            )

    skipped_cross_class_absorb = 0
    unknown_cleanup_evaluations: list[dict[str, Any]] = []
    contextual_cleanup_evaluations: list[dict[str, Any]] = []
    cleanup_cfg = cfg["operations"]["unknown_cleanup"]
    if cleanup_cfg["enabled"]:
        unknown_cleanup_evaluations = evaluate_unknown_cleanup_candidates(
            normalized.records,
            normalized.instance_map,
            pair_rows,
            semantic_probabilities,
            cleanup_cfg,
        )
        for evaluation in unknown_cleanup_evaluations:
            if not evaluation["eligible"]:
                record_policy_rejection(evaluation["reasons"])

    strict_by_source = {
        int(value["source_instance_id"]): value
        for value in unknown_cleanup_evaluations
        if value["eligible_for_auto_apply"]
    }
    contextual_cfg = cfg["operations"]["contextual_cleanup"]
    if contextual_cfg["enabled"]:
        contextual_cleanup_evaluations = evaluate_contextual_micro_cleanup(
            normalized.records,
            normalized.instance_map,
            pair_rows,
            semantic_probabilities,
            contextual_cfg,
        )
        for evaluation in contextual_cleanup_evaluations:
            if evaluation["action"] != "absorb":
                record_policy_rejection(evaluation["reasons"])
                continue
            source_instance_id = int(evaluation["source_instance_id"])
            target_instance_id = int(evaluation["target_instance_id"])
            strict = strict_by_source.get(source_instance_id)
            strict_match = bool(
                strict is not None
                and int(strict["target_instance_id"]) == target_instance_id
            )
            target_row = next(
                row
                for row in evaluation["evidence"]["target_candidates"]
                if int(row["target_instance_id"]) == target_instance_id
            )
            if strict_match:
                rule_version = "micro-unknown-v1"
                thresholds = copy.deepcopy(strict["thresholds"])
                strict_evidence = strict["evidence"]
                probability = float(
                    strict_evidence["mean_target_class_probability_on_source"]
                )
                margin = float(
                    strict_evidence["target_class_probability_margin_on_source"]
                )
                boundary = float(
                    strict_evidence[
                        "selected_target_shared_boundary_fraction_of_source_perimeter"
                    ]
                )
                score = min(
                    1.0, 0.50 * probability + 0.25 * margin + 0.25 * boundary
                )
            else:
                rule_version = "contextual-micro-v1"
                thresholds = copy.deepcopy(
                    evaluation["evidence"]["thresholds"]
                )
                score = max(0.0, 1.0 - float(evaluation["evidence"]["selected_energy"]))
            evidence = {
                **copy.deepcopy(dict(evaluation["evidence"])),
                "candidate_shared_boundary_fraction": float(
                    target_row["source_boundary_fraction"]
                ),
                "weak_color_boundary_score": float(
                    target_row["weak_color_boundary_score"]
                ),
                "region_mean_color_similarity": float(
                    target_row["region_mean_color_similarity"]
                ),
                "mean_target_class_probability_on_source": float(
                    target_row["mean_target_probability_on_source"]
                ),
                "hard_vetoes": [],
                "policy_gate_passed": True,
                "decision_tier": str(evaluation["decision_tier"]),
            }
            proposals.append(
                make_proposal(
                    normalized.report["tile_id"],
                    "absorb",
                    [str(evaluation["source_entity_id"])],
                    target_entity_id=str(evaluation["target"]["entity_id"]),
                    parameters={
                        "class_name": "unknown",
                        "target_class_name": str(evaluation["target"]["class_name"]),
                        "automatic_unknown_cleanup": True,
                        "cleanup_rule_version": rule_version,
                        "thresholds": thresholds,
                        "contextual_evidence_digest": (
                            stable_digest(evidence)
                            if rule_version == "contextual-micro-v1"
                            else None
                        ),
                        "contextual_decision_tier": str(evaluation["decision_tier"]),
                        "review_only": False,
                        "same_class_required": False,
                        "cross_class_relabel": True,
                        "score_kind": (
                            "deterministic_cleanup_gate"
                            if strict_match
                            else "bounded_contextual_energy"
                        ),
                    },
                    evidence=evidence,
                    score=score,
                    decision=("auto_apply" if contextual_cfg["auto_apply"] else "pending"),
                )
            )

    absorb_cfg = cfg["operations"]["absorb"]
    if absorb_cfg["enabled"] and pair_rows:
        contacts_by_source: dict[int, list[dict[str, Any]]] = {}
        for row in pair_rows:
            first, second = int(row["source_instance_id"]), int(row["target_instance_id"])
            forward = orient_absorb_evidence(row, first, second)
            reverse = orient_absorb_evidence(row, second, first)
            contacts_by_source.setdefault(first, []).append(forward)
            contacts_by_source.setdefault(second, []).append(reverse)
        for instance_id, contacts in sorted(contacts_by_source.items()):
            source_record = by_instance[instance_id]
            if str(source_record["label"]["name"]) == "unknown":
                # Unknowns use the independently replay-validated cleanup rule
                # above; the generic pair score is not a semantic classifier.
                continue
            if int(source_record["area_pixels"]) > int(absorb_cfg["maximum_segment_area_pixels"]):
                continue
            contacts.sort(
                key=lambda value: (
                    -float(value["candidate_shared_boundary_fraction"]),
                    -float(value["weak_color_boundary_score"]),
                    int(value["neighbor_instance_id"]),
                )
            )
            best = contacts[0]
            runner_fraction = (
                float(contacts[1]["candidate_shared_boundary_fraction"])
                if len(contacts) > 1
                else 0.0
            )
            best_fraction = float(best["candidate_shared_boundary_fraction"])
            margin = best_fraction - runner_fraction
            best["runner_up_boundary_fraction"] = runner_fraction
            best["best_to_runner_up_margin"] = margin
            target_record = by_instance[int(best["neighbor_instance_id"])]
            source_class_id = int(source_record["label"]["id"])
            target_class_id = int(target_record["label"]["id"])
            cross_class = source_class_id != target_class_id
            score = (
                0.50 * best_fraction
                + 0.25 * float(best["weak_color_boundary_score"])
                + 0.25 * float(best["region_mean_color_similarity"])
            )
            source_mask = normalized.instance_map == instance_id
            probability_values = (
                semantic_probabilities
                if semantic_probabilities is not None
                and 0 <= target_class_id < int(semantic_probabilities.shape[0])
                else None
            )
            policy = evaluate_absorb_candidate(
                cfg,
                source_record,
                target_record,
                best,
                score=score,
                semantic_probabilities=probability_values,
                candidate_mask=source_mask if probability_values is not None else None,
            )
            if not policy.eligible:
                record_policy_rejection((*policy.reasons, *policy.hard_vetoes))
                if cross_class and any(
                    "probability" in reason for reason in policy.reasons
                ):
                    skipped_cross_class_absorb += 1
                continue
            target_mask = normalized.instance_map == int(target_record["instance_id"])
            proposals.append(
                make_proposal(
                    normalized.report["tile_id"],
                    "absorb",
                    [str(source_record["entity_id"])],
                    target_entity_id=str(target_record["entity_id"]),
                    parameters={
                        "class_name": str(source_record["label"]["name"]),
                        "target_class_name": str(target_record["label"]["name"]),
                        "review_only": True,
                        "same_class_required": not cross_class,
                        "cross_class_relabel": cross_class,
                        "score_kind": "uncalibrated_review_priority",
                    },
                    evidence={
                        **copy.deepcopy(policy.evidence),
                        "source_touches_tile_boundary": _touches_tile_boundary(
                            source_mask
                        ),
                        "target_touches_tile_boundary": _touches_tile_boundary(
                            target_mask
                        ),
                        "hard_vetoes": list(policy.hard_vetoes),
                        "policy_gate_passed": True,
                    },
                    score=score,
                    decision="pending",
                )
            )

    proposals.sort(key=lambda value: str(value["proposal_id"]))
    selected_workflow = normalize_refinement_mode(workflow_mode)
    # Proposal generation is shared by both workflows. Automatic mode changes
    # only the decision provider, never which candidates exist.
    cap = int(cfg["review"]["maximum_candidates_per_tile"])
    connectivity = [p for p in proposals if p["operation_type"] == "split_components"]
    automatic = [
        p
        for p in proposals
        if p["operation_type"] != "split_components"
        and p["decision"] == "auto_apply"
    ]
    reviewable = [
        p
        for p in proposals
        if p["operation_type"] != "split_components"
        and p["decision"] != "auto_apply"
    ]
    if len(reviewable) > cap:
        reviewable = sorted(
            reviewable,
            key=lambda value: (
                -(float(value["score"]) if value["score"] is not None else -1.0),
                str(value["proposal_id"]),
            ),
        )[:cap]
        proposals = sorted(
            [*connectivity, *automatic, *reviewable],
            key=lambda value: value["proposal_id"],
        )
    for proposal in proposals:
        validate_proposal(proposal)
    counts: dict[str, int] = {}
    for proposal in proposals:
        operation = str(proposal["operation_type"])
        counts[operation] = counts.get(operation, 0) + 1
    return proposals, {
        "proposal_counts": counts,
        "workflow_mode": selected_workflow,
        "semantic_candidate_cap": cap,
        "pair_evidence_available": rgb is not None,
        "pair_evidence_count": len(pair_rows),
        "skipped_cross_class_absorb_candidate_count": skipped_cross_class_absorb,
        "unknown_cleanup_evaluated_count": len(unknown_cleanup_evaluations),
        "unknown_cleanup_eligible_count": sum(
            bool(value["eligible"]) for value in unknown_cleanup_evaluations
        ),
        "unknown_cleanup_auto_apply_count": sum(
            value["action"] == "absorb" and bool(contextual_cfg["auto_apply"])
            for value in contextual_cleanup_evaluations
        ),
        "strict_unknown_cleanup_eligible_count": len(strict_by_source),
        "contextual_cleanup_evaluated_count": len(contextual_cleanup_evaluations),
        "contextual_cleanup_absorb_count": sum(
            value["action"] == "absorb"
            for value in contextual_cleanup_evaluations
        ),
        "contextual_cleanup_preserved_object_count": sum(
            value["action"] == "preserve"
            for value in contextual_cleanup_evaluations
        ),
        "contextual_cleanup_abstain_count": sum(
            value["action"] == "abstain"
            for value in contextual_cleanup_evaluations
        ),
        "policy_rejection_counts": dict(sorted(policy_rejection_counts.items())),
        "semantic_split_generation_status": "not_enabled_for_automatic_v1; manual_review_contract_only",
        "scores_are_calibrated_probabilities": False,
        "semantic_proposals_auto_applied": any(
            proposal["operation_type"] != "split_components"
            and proposal["decision"] == "auto_apply"
            for proposal in proposals
        ),
    }


def _touches_tile_boundary(mask: np.ndarray) -> bool:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("Refinement masks must be non-empty 2-D arrays")
    return bool(
        values[0, :].any()
        or values[-1, :].any()
        or values[:, 0].any()
        or values[:, -1].any()
    )


def _persist_proposal_result(
    output_directory: Path,
    source: _Stage2Source,
    normalized: CoreRefinementResult,
    proposals: list[dict[str, Any]],
    proposal_report: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, Any]]:
    output_directory.mkdir(parents=True, exist_ok=True)
    proposals_path = output_directory / "proposals.json"
    template_path = output_directory / "decisions.template.json"
    report_path = output_directory / "proposal_report.json"
    proposal_manifest_path = output_directory / "proposal_manifest.json"
    public_manifest_path = output_directory / "manifest.json"
    source_inputs = _source_input_records(source)
    input_checksum = checksum(source.directory / "segmentation.npz")
    save_proposals(
        proposals_path,
        proposals,
        tile_id=source.result.tile_id,
        configuration_hash=config_hash(cfg),
        input_checksum=input_checksum,
        metadata={"source_stage2_inputs": source_inputs},
    )
    proposal_checksum = sha256_file(proposals_path)
    review_bundle = save_review_bundle(
        output_directory,
        tile_id=source.result.tile_id,
        proposal_checksum=proposal_checksum,
        proposals=proposals,
        masks=normalized.masks,
        records=normalized.records,
        instance_map=normalized.instance_map,
        semantic_map=normalized.semantic_map,
        confidence_map=normalized.confidence_map,
        source_map=normalized.source_map,
        valid_mask=normalized.valid_mask,
        image=source.rgb,
        padding=int(cfg["review"]["crop_padding_pixels"]),
        alpha=float(cfg["visualization"]["alpha"]),
        random_seed=int(cfg["visualization"]["random_seed"]),
    )
    save_decisions(
        template_path,
        review_bundle.decision_records,
        tile_id=source.result.tile_id,
        proposal_checksum=proposal_checksum,
        metadata={
            "instructions": (
                "Every pending proposal is prefilled as uncertain. Change a record to "
                "approved or rejected only after inspecting review/index.json and its crop. "
                "The unchanged template applies only deterministic connectivity normalization."
            ),
            "review_index": "review/index.json",
        },
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "proposed",
        "tile_id": source.result.tile_id,
        "workflow_mode": "human_review",
        "input_segment_count": len(source.result.segments),
        "normalized_segment_count": len(normalized.records),
        "proposal_count": len(proposals),
        "review_candidate_count": len(review_bundle.decision_records),
        **copy.deepcopy(dict(proposal_report)),
        "pixels_modified": 0,
        "stage2_immutable": True,
    }
    atomic_json(report_path, report)
    proposal_manifest = manifest(
        proposal_manifest_path,
        "13_refinement_proposals",
        source.required_inputs
        + ([source.rgb_path] if source.rgb_path else [])
        + (
            [source.semantic_probabilities_path]
            if source.semantic_probabilities_path
            else []
        ),
        [
            proposals_path,
            template_path,
            report_path,
            *review_bundle.output_paths,
        ],
        config_hash(cfg),
        now(),
    )
    artifacts = {
        "proposals": str(proposals_path.resolve()),
        "decisions_template": str(template_path.resolve()),
        "proposal_report": str(report_path.resolve()),
        "proposal_manifest": str(proposal_manifest_path.resolve()),
        "review_index": str(review_bundle.index_path),
        "review_segments": str(review_bundle.segments_path),
        "review_masks": str(review_bundle.masks_path),
        "review_segmentation": str(review_bundle.segmentation_path),
    }
    atomic_json(
        public_manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "proposed",
            "stale": True,
            "tile_id": source.result.tile_id,
            "workflow_mode": "human_review",
            "configuration_hash": config_hash(cfg),
            "source_stage2": {
                "schema_version": source.manifest.get("schema_version"),
                "output_directory": str(source.directory.resolve()),
                "inputs": source_inputs,
            },
            "proposal_phase_status": proposal_manifest.get("status"),
            "artifacts": artifacts,
        },
    )
    return artifacts, report


def _persist_complete_result(
    *,
    output_directory: Path,
    source: _Stage2Source,
    refined: CoreRefinementResult,
    proposals: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    proposal_report: Mapping[str, Any],
    validation: ValidationReport,
    refinement_config: Mapping[str, Any],
    workflow_mode: str,
    automatic_selection_report: Mapping[str, Any] | None,
    started: str,
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    public_manifest_path = output_directory / "manifest.json"
    atomic_json(
        public_manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "in_progress",
            "stale": True,
            "tile_id": source.result.tile_id,
            "started_at": started,
        },
    )
    segments_npz = output_directory / "segments.npz"
    segments_json = output_directory / "segments.json"
    save_masks(segments_npz, segments_json, refined.masks, refined.records)
    # Reload mask_index fields exactly as downstream consumers will see them.
    masks, records = load_masks(segments_npz, segments_json)
    refined.masks = masks
    refined.records = records

    maps_path = output_directory / "segmentation.npz"
    save_segmentation_bundle(
        maps_path,
        instance_map=refined.instance_map.astype(np.int32),
        semantic_map=refined.semantic_map.astype(np.uint8),
        confidence_map=np.asarray(refined.confidence_map, dtype=np.float32),
        source_map=np.asarray(refined.source_map, dtype=np.uint8),
        valid_mask=refined.valid_mask.astype(bool),
    )
    operation_map, operation_index = _operation_map(refined)
    operation_map_path = output_directory / "refinement_operation_map.npz"
    atomic_npz(
        operation_map_path,
        refinement_operation_map=operation_map,
        operation_count=np.asarray(len(operation_index), dtype=np.int64),
    )

    proposals_path = output_directory / "proposals.json"
    decisions_path = output_directory / "decisions.json"
    operations_path = output_directory / "operations.json"
    lineage_path = output_directory / "lineage.json"
    input_checksum = checksum(source.directory / "segmentation.npz")
    save_proposals(
        proposals_path,
        proposals,
        tile_id=source.result.tile_id,
        configuration_hash=config_hash(refinement_config),
        input_checksum=input_checksum,
        metadata={"source_stage2_inputs": _source_input_records(source)},
    )
    save_decisions(
        decisions_path,
        decisions,
        tile_id=source.result.tile_id,
        proposal_checksum=sha256_file(proposals_path),
        metadata={
            "workflow_mode": workflow_mode,
            "automatic_selection": (
                None
                if automatic_selection_report is None
                else copy.deepcopy(dict(automatic_selection_report))
            ),
        },
    )
    save_refinement_history(
        operations_path,
        refined.operations,
        tile_id=source.result.tile_id,
        input_checksum=input_checksum,
        output_checksum=sha256_array(refined.instance_map),
        metadata={"operation_index": operation_index},
    )
    atomic_json(lineage_path, refined.lineage)

    tile_metadata_path = output_directory / "tile_metadata.json"
    class_registry_path = output_directory / "class_registry.json"
    atomic_json(tile_metadata_path, source.result.tile_metadata)
    atomic_json(class_registry_path, load_json(source.directory / "class_registry.json"))

    image = source.rgb
    if image is None:
        image = np.zeros((*refined.instance_map.shape, 3), dtype=np.uint8)
    visual_paths = save_refinement_visualizations(
        output_directory,
        image=image,
        instance_map=refined.instance_map,
        semantic_map=refined.semantic_map,
        valid_mask=refined.valid_mask,
        before_instance_map=source.result.instance_map,
        before_semantic_map=source.result.semantic_map,
        alpha=float(refinement_config["visualization"]["alpha"]),
        random_seed=int(refinement_config["visualization"]["random_seed"]),
        show_ids=False,
    )

    operation_counts: dict[str, int] = {}
    for operation in refined.operations:
        name = str(operation["operation_type"])
        operation_counts[name] = operation_counts.get(name, 0) + 1
    automatic_semantic_operations = [
        operation
        for operation in refined.operations
        if operation["operation_type"] in {"absorb", "merge", "semantic_split"}
        and (
            operation.get("decision_source") == "proposal_embedded"
            or str(operation.get("decision_source", "")).startswith("automatic_")
        )
    ]
    raw_unknown_pixels = int(np.count_nonzero(source.result.semantic_map == 8))
    refined_unknown_pixels = int(np.count_nonzero(refined.semantic_map == 8))
    changed_semantic_pixels = int(
        np.count_nonzero(source.result.semantic_map != refined.semantic_map)
    )
    raw_unknown_segments = sum(
        int(record.get("label", {}).get("id", -1)) == 8
        for record in source.result.segments
    )
    refined_unknown_segments = sum(
        int(record.get("label", {}).get("id", -1)) == 8
        for record in refined.records
    )
    comparison = {
        "raw_segment_count": len(source.result.segments),
        "refined_segment_count": len(refined.records),
        "raw_unknown_segment_count": raw_unknown_segments,
        "refined_unknown_segment_count": refined_unknown_segments,
        "raw_unknown_pixel_count": raw_unknown_pixels,
        "refined_unknown_pixel_count": refined_unknown_pixels,
        "unknown_pixel_delta": refined_unknown_pixels - raw_unknown_pixels,
        "changed_semantic_pixel_count": changed_semantic_pixels,
        "semantic_map_equal": changed_semantic_pixels == 0,
        "confidence_map_equal": bool(
            np.array_equal(source.result.confidence_map, refined.confidence_map)
        ),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "tile_id": source.result.tile_id,
        "workflow_mode": workflow_mode,
        "source_stage2_schema_version": source.manifest.get("schema_version"),
        "input_segment_count": len(source.result.segments),
        "output_segment_count": len(refined.records),
        "proposal_count": len(proposals),
        "decision_count": len(decisions),
        "operation_count": len(refined.operations),
        "operation_counts": operation_counts,
        "comparison": comparison,
        "proposal_counts": copy.deepcopy(dict(proposal_report.get("proposal_counts", {}))),
        "every_valid_pixel_exactly_once": True,
        "connectivity": 4,
        "object_identity": copy.deepcopy(
            dict(refined.report.get("object_identity", {}))
        ),
        "stage2_immutable": True,
        "semantic_operations_auto_applied": bool(automatic_semantic_operations),
        "automatic_semantic_operation_count": len(automatic_semantic_operations),
        "automatic_selection": (
            {
                "policy_version": None,
                "approved_count": 0,
                "abstained_count": 0,
                "reason_counts": {},
            }
            if automatic_selection_report is None
            else copy.deepcopy(dict(automatic_selection_report))
        ),
        "unknown_cleanup": {
            "rule_versions": ["micro-unknown-v1", "contextual-micro-v1"],
            "evaluated_count": int(
                proposal_report.get("contextual_cleanup_evaluated_count", 0)
            ),
            "strict_semantic_eligible_count": int(
                proposal_report.get("strict_unknown_cleanup_eligible_count", 0)
            ),
            "auto_apply_count": int(
                proposal_report.get("unknown_cleanup_auto_apply_count", 0)
            ),
            "contextual_absorb_count": int(
                proposal_report.get("contextual_cleanup_absorb_count", 0)
            ),
            "preserved_object_count": int(
                proposal_report.get("contextual_cleanup_preserved_object_count", 0)
            ),
            "abstain_count": int(
                proposal_report.get("contextual_cleanup_abstain_count", 0)
            ),
            "absorbed_pixel_count": -int(comparison["unknown_pixel_delta"]),
            "thresholds": {
                "strict": copy.deepcopy(
                    refinement_config["operations"]["unknown_cleanup"]
                ),
                "contextual": copy.deepcopy(
                    refinement_config["operations"]["contextual_cleanup"]
                ),
            },
        },
        "semantic_split_generation_status": proposal_report.get(
            "semantic_split_generation_status"
        ),
        "validation": validation.to_dict(),
        "operation_index": operation_index,
        "coordinate_reference": copy.deepcopy(
            source.result.report.get("coordinate_reference", {})
        ),
        "limitations": {
            "cross_class_absorb_supported": source.semantic_probabilities is not None,
            "road_junction_decomposition_automatic": False,
            "agriculture_field_split_automatic": False,
            "semantic_split_automatic": False,
            "reason": (
                "Automatic mode additionally permits strictly gated same-class "
                "interior merge/absorb operations for non-protected homogeneous "
                "classes. Road junctions, fields, buildings, tree canopy, semantic "
                "splits, borders, and weak or missing evidence abstain."
            ),
        },
    }
    report_path = output_directory / "report.json"
    atomic_json(report_path, report)

    inputs = (
        source.required_inputs
        + ([source.rgb_path] if source.rgb_path else [])
        + (
            [source.semantic_probabilities_path]
            if source.semantic_probabilities_path
            else []
        )
    )
    outputs = [
        segments_npz,
        segments_json,
        maps_path,
        operation_map_path,
        proposals_path,
        decisions_path,
        operations_path,
        lineage_path,
        tile_metadata_path,
        class_registry_path,
        report_path,
        *visual_paths.values(),
    ]
    phase_manifest_path = output_directory / "phase_manifest.json"
    phase = manifest(
        phase_manifest_path,
        "13_refinement",
        inputs,
        outputs,
        config_hash(refinement_config),
        started,
    )
    if phase.get("status") != "complete":
        raise RuntimeError("Refinement phase manifest did not complete")
    artifacts = {
        "segments": str(segments_json.resolve()),
        "masks": str(segments_npz.resolve()),
        "segmentation_maps": str(maps_path.resolve()),
        "refinement_operation_map": str(operation_map_path.resolve()),
        "tile_metadata": str(tile_metadata_path.resolve()),
        "class_registry": str(class_registry_path.resolve()),
        "proposals": str(proposals_path.resolve()),
        "decisions": str(decisions_path.resolve()),
        "operations": str(operations_path.resolve()),
        "lineage": str(lineage_path.resolve()),
        "report": str(report_path.resolve()),
        "refined_segmentation": str(visual_paths["segmentation"].resolve()),
        "refined_overlay": str(visual_paths["overlay"].resolve()),
        "before_after": str(visual_paths["before_after"].resolve()),
        "phase_manifest": str(phase_manifest_path.resolve()),
    }
    public_manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "stale": False,
        "tile_id": source.result.tile_id,
        "workflow_mode": workflow_mode,
        "automatic_policy_version": (
            None
            if automatic_selection_report is None
            else automatic_selection_report.get("policy_version")
        ),
        "created_at": now(),
        "configuration_hash": config_hash(refinement_config),
        "source_stage2": {
            "schema_version": source.manifest.get("schema_version"),
            "configuration_hash": source.manifest.get("configuration_hash"),
            "output_directory": str(source.directory.resolve()),
            "inputs": _source_input_records(source),
        },
        "segment_count": len(refined.records),
        "operation_count": len(refined.operations),
        "artifacts": artifacts,
        "artifact_checksums": {
            key: checksum(Path(path)) for key, path in artifacts.items()
        },
    }
    atomic_json(public_manifest_path, public_manifest)


def _operation_map(
    refined: CoreRefinementResult,
) -> tuple[np.ndarray, dict[str, int]]:
    operation_ids = [str(value["operation_id"]) for value in refined.operations]
    operation_index = {value: index + 1 for index, value in enumerate(operation_ids)}
    result = np.zeros(refined.instance_map.shape, dtype=np.int32)
    for record, mask in zip(refined.records, refined.masks):
        history = [
            operation_index[value]
            for value in record.get("refinement_history", [])
            if value in operation_index
        ]
        if history:
            result[np.asarray(mask, dtype=bool)] = int(history[-1])
    return result, operation_index


def _load_project_config(
    config: Mapping[str, Any] | str | Path | None,
) -> dict[str, Any]:
    if config is None:
        return load_config(DEFAULT_CONFIG)
    if isinstance(config, (str, Path)):
        return load_config(config)
    result = copy.deepcopy(dict(config))
    validate_config(result)
    return result


def _load_stage2_source(
    project_root: Path,
    requested_tile_id: str,
    cfg: Mapping[str, Any],
) -> _Stage2Source:
    tile_id = _safe_tile_id(requested_tile_id)
    stage2_root = _resolve_root(project_root, str(cfg["stage2_root"]))
    directory = stage2_root / tile_id / "12_final"
    manifest_path = directory / "manifest.json"
    require_fresh(manifest_path)
    public_manifest = load_json(manifest_path)
    if public_manifest.get("schema_version") != STAGE2_SCHEMA_VERSION:
        raise ValueError(
            f"Refinement requires Stage 2 schema {STAGE2_SCHEMA_VERSION!r}; "
            f"found {public_manifest.get('schema_version')!r}"
        )
    if public_manifest.get("status") != "complete":
        raise ValueError(f"Stage 2 result is not complete: {public_manifest.get('status')!r}")
    if public_manifest.get("stale"):
        raise RuntimeError(f"Stage 2 result is stale for {tile_id!r}")
    if str(public_manifest.get("tile_id")) != tile_id:
        raise ValueError("Requested tile ID and Stage 2 manifest tile ID disagree")
    phase_manifest_path = directory / "phase_manifest.json"
    if not phase_manifest_path.is_file():
        raise FileNotFoundError(f"Stage 2 phase manifest not found: {phase_manifest_path}")
    phase_manifest = load_json(phase_manifest_path)
    if phase_manifest.get("status") != "complete" or phase_manifest.get("stale"):
        raise RuntimeError(f"Stage 2 final phase is incomplete or stale: {phase_manifest_path}")
    required = [
        manifest_path,
        phase_manifest_path,
        directory / "segments.npz",
        directory / "segments.json",
        directory / "segmentation.npz",
        directory / "tile_metadata.json",
        directory / "class_registry.json",
        directory / "report.json",
    ]
    for path in required:
        require_fresh(path)
    result = load_stage2_result(directory)
    if not result.report.get("every_valid_pixel_exactly_once", False):
        raise ValueError("Stage 2 report does not guarantee exact valid-pixel coverage")
    rgb, rgb_path, rgb_is_authoritative = _load_rgb(
        directory.parent, directory, result
    )
    semantic_probabilities, semantic_probabilities_path = _load_semantic_probabilities(
        directory.parent,
        result.instance_map.shape,
    )
    return _Stage2Source(
        directory=directory,
        manifest=public_manifest,
        result=result,
        required_inputs=required,
        rgb=rgb,
        rgb_path=rgb_path,
        rgb_is_authoritative=rgb_is_authoritative,
        semantic_probabilities=semantic_probabilities,
        semantic_probabilities_path=semantic_probabilities_path,
    )


def _load_rgb(
    tile_root: Path,
    final_directory: Path,
    stage2_result: Any,
) -> tuple[np.ndarray | None, Path | None, bool]:
    candidates = [
        (tile_root / "02_preprocessing" / "sam_input.npy", True),
        (
            Path(str(stage2_result.tile_metadata.get("image_path", ""))).expanduser(),
            True,
        ),
        (final_directory / "final_overlay.png", False),
    ]
    for path, authoritative in candidates:
        if not str(path) or not path.is_file():
            continue
        if path.suffix.lower() == ".npy":
            image = np.load(path, allow_pickle=False)
        else:
            with Image.open(path) as opened:
                image = np.asarray(opened.convert("RGB"))
        if image.shape[:2] == stage2_result.instance_map.shape:
            return np.asarray(image), path.resolve(), authoritative
    return None, None, False


def _load_semantic_probabilities(
    tile_root: Path,
    shape: tuple[int, int],
) -> tuple[np.ndarray | None, Path | None]:
    path = tile_root / "11_semantic" / "openearthmap.npz"
    if not path.is_file():
        return None, None
    with np.load(path, allow_pickle=False) as archive:
        if "probabilities" not in archive:
            return None, None
        probabilities = np.asarray(archive["probabilities"], dtype=np.float32)
    if probabilities.ndim != 3 or probabilities.shape[1:] != shape:
        raise ValueError(
            "OpenEarthMap probabilities must use class-first shape matching the Stage 2 tile"
        )
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("OpenEarthMap probabilities must be finite and in [0, 1]")
    return probabilities, path.resolve()


def _source_input_records(source: _Stage2Source) -> list[dict[str, str]]:
    paths = [
        *source.required_inputs,
        *([source.rgb_path] if source.rgb_path else []),
        *(
            [source.semantic_probabilities_path]
            if source.semantic_probabilities_path
            else []
        ),
    ]
    return [
        {"path": str(path.resolve()), "sha256": checksum(path)}
        for path in paths
        if path is not None and path.is_file()
    ]


def _validate_source_inputs(records: Any) -> None:
    if not isinstance(records, list) or not records:
        raise ValueError("Refinement manifest has no source Stage 2 checksums")
    for record in records:
        if not isinstance(record, Mapping) or not record.get("path") or not record.get("sha256"):
            raise ValueError("Refinement manifest contains an invalid source input record")
        path = Path(str(record["path"])).expanduser()
        if not path.is_file() or checksum(path) != str(record["sha256"]):
            raise RuntimeError(f"Refinement source changed or disappeared: {path}")


def _load_decision_records(
    value: Mapping[str, Any] | Sequence[Mapping[str, Any]] | str | Path | None,
    *,
    expected_tile_id: str,
    expected_proposal_checksum: str,
) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        document = load_json(value)
        if isinstance(document, Mapping) and document.get("document_type") == "refinement_decisions":
            document = load_decisions(value)
            if str(document.get("tile_id")) != expected_tile_id:
                raise ValueError("Decision document belongs to a different tile")
            if str(document.get("proposal_checksum")) != expected_proposal_checksum:
                raise RuntimeError(
                    "Decision document is not bound to the regenerated proposals; "
                    "rerun refine propose and review the new candidates"
                )
            return [copy.deepcopy(dict(item)) for item in document["decisions"]]
        value = document
    if isinstance(value, Mapping):
        if "decisions" in value:
            nested = value["decisions"]
            if not isinstance(nested, Sequence) or isinstance(nested, (str, bytes)):
                raise ValueError("decisions.decisions must be a list")
            return [copy.deepcopy(dict(item)) for item in nested]
        records = []
        for proposal_id, decision in sorted(value.items(), key=lambda item: str(item[0])):
            if isinstance(decision, Mapping):
                record = copy.deepcopy(dict(decision))
                record.setdefault("proposal_id", str(proposal_id))
            else:
                record = {"proposal_id": str(proposal_id), "decision": str(decision)}
            records.append(record)
        return records
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [copy.deepcopy(dict(item)) for item in value]
    raise ValueError("decisions must be a JSON path, mapping, list, or None")


def _proposal_document_checksum(
    source: _Stage2Source,
    proposals: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
) -> str:
    """Hash the exact deterministic proposal document used by review files."""

    with tempfile.TemporaryDirectory(prefix="satellitegraph-refinement-") as temporary:
        path = Path(temporary) / "proposals.json"
        save_proposals(
            path,
            proposals,
            tile_id=source.result.tile_id,
            configuration_hash=config_hash(cfg),
            input_checksum=checksum(source.directory / "segmentation.npz"),
            metadata={"source_stage2_inputs": _source_input_records(source)},
        )
        return sha256_file(path)


def _output_directory(root: Path, tile_id: str, cfg: Mapping[str, Any]) -> Path:
    output_root = _resolve_root(root, str(cfg["output"]["root_directory"]))
    return output_root / _safe_tile_id(tile_id) / str(cfg["output"]["phase_directory"])


def _resolve_root(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _safe_tile_id(value: str) -> str:
    tile_id = str(value)
    if not tile_id or Path(tile_id).name != tile_id or tile_id in {".", ".."}:
        raise ValueError("tile_id must be one safe path component")
    return tile_id


class _RefinementProgress:
    def __init__(self, enabled: bool, *, total: int, action: str) -> None:
        self.action = action
        self.bar = tqdm(
            total=total,
            unit="phase",
            dynamic_ncols=True,
            disable=not enabled,
            desc=f"Refinement {action}",
        )

    def phase(self, name: str) -> None:
        self.bar.set_description_str(f"Refinement {self.action} | {name}", refresh=True)

    def advance(self) -> None:
        self.bar.update(1)

    def complete(self) -> None:
        self.bar.set_description_str(f"Refinement {self.action} | Complete", refresh=True)

    def close(self) -> None:
        self.bar.close()


__all__ = [
    "RefinementProposalResult",
    "RefinementResult",
    "apply_refinement",
    "load_refinement_result",
    "propose_refinement",
    "validate_refinement",
    "validate_refinement_result",
]
