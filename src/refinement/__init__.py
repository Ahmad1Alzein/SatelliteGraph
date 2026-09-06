"""Public API for reversible Stage 2R segmentation refinement."""

from __future__ import annotations

from typing import Any


__all__ = [
    "RefinementProposalResult",
    "RefinementResult",
    "RefinementBatchResult",
    "AutomaticSelectionResult",
    "apply_reviewed_operations",
    "apply_refinement",
    "discover_refinement_tiles",
    "evaluate_automatic_proposal",
    "load_refinement_result",
    "make_semantic_split_candidate",
    "make_semantic_split_candidates",
    "normalize_4_connected_components",
    "propose_refinement",
    "run_refinement_batch",
    "select_automatic_decisions",
    "validate_semantic_split_partition",
    "validate_refinement",
    "validate_refinement_result",
]


def __getattr__(name: str) -> Any:
    if name in {
        "AutomaticSelectionResult",
        "evaluate_automatic_proposal",
        "select_automatic_decisions",
    }:
        from .automatic import (
            AutomaticSelectionResult,
            evaluate_automatic_proposal,
            select_automatic_decisions,
        )

        return {
            "AutomaticSelectionResult": AutomaticSelectionResult,
            "evaluate_automatic_proposal": evaluate_automatic_proposal,
            "select_automatic_decisions": select_automatic_decisions,
        }[name]
    if name in {
        "RefinementBatchResult",
        "discover_refinement_tiles",
        "run_refinement_batch",
    }:
        from .batch import (
            RefinementBatchResult,
            discover_refinement_tiles,
            run_refinement_batch,
        )

        return {
            "RefinementBatchResult": RefinementBatchResult,
            "discover_refinement_tiles": discover_refinement_tiles,
            "run_refinement_batch": run_refinement_batch,
        }[name]
    pipeline_names = {
        "RefinementProposalResult",
        "RefinementResult",
        "apply_refinement",
        "load_refinement_result",
        "propose_refinement",
        "validate_refinement",
        "validate_refinement_result",
    }
    if name in pipeline_names:
        from .pipeline import (
            RefinementProposalResult,
            RefinementResult,
            apply_refinement,
            load_refinement_result,
            propose_refinement,
            validate_refinement,
            validate_refinement_result,
        )

        return {
            "RefinementProposalResult": RefinementProposalResult,
            "RefinementResult": RefinementResult,
            "apply_refinement": apply_refinement,
            "load_refinement_result": load_refinement_result,
            "propose_refinement": propose_refinement,
            "validate_refinement": validate_refinement,
            "validate_refinement_result": validate_refinement_result,
        }[name]
    if name in {"apply_reviewed_operations", "normalize_4_connected_components"}:
        from .topology import apply_reviewed_operations, normalize_4_connected_components

        return {
            "apply_reviewed_operations": apply_reviewed_operations,
            "normalize_4_connected_components": normalize_4_connected_components,
        }[name]
    if name in {
        "make_semantic_split_candidate",
        "make_semantic_split_candidates",
        "validate_semantic_split_partition",
    }:
        from .splits import (
            make_semantic_split_candidate,
            make_semantic_split_candidates,
            validate_semantic_split_partition,
        )

        return {
            "make_semantic_split_candidate": make_semantic_split_candidate,
            "make_semantic_split_candidates": make_semantic_split_candidates,
            "validate_semantic_split_partition": validate_semantic_split_partition,
        }[name]
    raise AttributeError(name)
