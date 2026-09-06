"""Exact, review-only semantic split candidates and partition validation.

This module deliberately does not guess where a semantic boundary lies.  A
caller supplies an integer child-label raster, and these helpers bind that
exact raster to a deterministic proposal.  The topology replay validates the
same contract again before changing any instance identity.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .schema import make_proposal
from .storage import sha256_array
from .validation import is_four_connected


@dataclass(slots=True)
class SemanticSplitCandidate:
    """One proposal paired with its authoritative canonical partition map."""

    proposal: dict[str, Any]
    partition_map: np.ndarray


def canonical_semantic_split_partition(partition_map: np.ndarray) -> np.ndarray:
    """Return the portable ``int32`` representation used for split checksums."""

    value = np.asarray(partition_map)
    if value.ndim != 2 or any(length <= 0 for length in value.shape):
        raise ValueError("semantic split partition_map must have a non-empty 2-D shape")
    if value.dtype == np.bool_ or not np.issubdtype(value.dtype, np.integer):
        raise TypeError("semantic split partition_map must use an integer dtype")
    if np.any(value < 0):
        raise ValueError("semantic split partition_map cannot contain negative child labels")
    if value.size and int(value.max()) > np.iinfo(np.int32).max:
        raise ValueError("semantic split child labels exceed the int32 range")
    return np.ascontiguousarray(value, dtype=np.int32)


def semantic_split_partition_checksum(partition_map: np.ndarray) -> str:
    """Hash the canonical child-label raster, including its shape and dtype."""

    return sha256_array(canonical_semantic_split_partition(partition_map))


def validate_semantic_split_partition(
    partition_map: np.ndarray,
    parent_mask: np.ndarray,
    *,
    expected_sha256: str | None = None,
    expected_child_count: int | None = None,
    minimum_child_area_pixels: int = 1,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Validate and return a canonical exact partition of one connected parent.

    Label zero is required outside the parent.  Inside it, labels must be the
    contiguous set ``1..K``.  Therefore the returned child masks are disjoint
    by construction and their union is exactly the parent mask.
    """

    partition = canonical_semantic_split_partition(partition_map)
    parent = np.asarray(parent_mask)
    if parent.shape != partition.shape or parent.dtype != np.bool_:
        raise ValueError(
            "semantic split parent_mask must be boolean and match the partition shape"
        )
    if not np.any(parent):
        raise ValueError("semantic split parent_mask must contain at least one pixel")
    if not is_four_connected(parent):
        raise ValueError("semantic split source parent must be one 4-connected region")
    if np.any(partition[~parent] != 0):
        raise ValueError("semantic split partition must be zero outside its source parent")
    if np.any(partition[parent] <= 0):
        raise ValueError("semantic split children must exactly cover every source-parent pixel")

    labels = np.unique(partition[parent]).astype(np.int64, copy=False)
    child_count = int(len(labels))
    if child_count < 2:
        raise ValueError("semantic split must create at least two children")
    expected_labels = np.arange(1, child_count + 1, dtype=np.int64)
    if not np.array_equal(labels, expected_labels):
        raise ValueError(
            "semantic split child labels must be contiguous and start at one; "
            f"got {labels.tolist()}"
        )
    if expected_child_count is not None:
        count = _positive_integer(expected_child_count, "expected_child_count", minimum=2)
        if child_count != count:
            raise ValueError(
                f"semantic split child count disagrees with proposal: {child_count} != {count}"
            )
    minimum_area = _positive_integer(
        minimum_child_area_pixels,
        "minimum_child_area_pixels",
        minimum=1,
    )
    children: list[np.ndarray] = []
    for label in range(1, child_count + 1):
        child = partition == label
        area = int(child.sum())
        if area < minimum_area:
            raise ValueError(
                f"semantic split child {label} has {area} pixels, below minimum {minimum_area}"
            )
        if not is_four_connected(child):
            raise ValueError(f"semantic split child {label} is not 4-connected")
        children.append(child)
    if not np.array_equal(np.logical_or.reduce(children), parent):
        raise ValueError("semantic split children do not exactly reconstruct their source parent")

    actual_sha256 = sha256_array(partition)
    if expected_sha256 is not None and actual_sha256 != str(expected_sha256):
        raise ValueError(
            "semantic split partition checksum disagrees with the reviewed proposal"
        )
    return partition, children


def make_semantic_split_candidate(
    tile_id: str,
    source_entity_id: str,
    parent_mask: np.ndarray,
    partition_map: np.ndarray,
    *,
    minimum_child_area_pixels: int = 1,
    evidence: Mapping[str, Any] | None = None,
    score: float | None = None,
    candidate_source: str = "caller_supplied_partition",
    parameters: Mapping[str, Any] | None = None,
) -> SemanticSplitCandidate:
    """Create a pending proposal bound to one caller-supplied exact partition."""

    source = str(source_entity_id)
    if not source:
        raise ValueError("source_entity_id must be non-empty")
    minimum_area = _positive_integer(
        minimum_child_area_pixels,
        "minimum_child_area_pixels",
        minimum=1,
    )
    partition, children = validate_semantic_split_partition(
        partition_map,
        np.asarray(parent_mask),
        minimum_child_area_pixels=minimum_area,
    )
    supplied_parameters = copy.deepcopy(dict(parameters or {}))
    protected = {
        "partition_sha256",
        "partition_shape",
        "partition_dtype",
        "child_count",
        "child_label_order",
        "minimum_child_area_pixels",
        "review_only",
        "automatic",
    }
    collisions = sorted(protected & set(supplied_parameters))
    if collisions:
        raise ValueError(
            "Caller parameters cannot override semantic split identity fields: "
            f"{collisions}"
        )
    split_parameters = {
        **supplied_parameters,
        "partition_sha256": sha256_array(partition),
        "partition_shape": [int(partition.shape[0]), int(partition.shape[1])],
        "partition_dtype": "int32",
        "child_count": len(children),
        "child_label_order": "ascending_contiguous_from_1",
        "minimum_child_area_pixels": minimum_area,
        "review_only": True,
        "automatic": False,
    }
    split_evidence = copy.deepcopy(dict(evidence or {}))
    split_evidence.setdefault("candidate_source", str(candidate_source))
    split_evidence.setdefault("parent_area_pixels", int(np.asarray(parent_mask).sum()))
    split_evidence.setdefault(
        "child_area_pixels", [int(child.sum()) for child in children]
    )
    proposal = make_proposal(
        str(tile_id),
        "semantic_split",
        [source],
        parameters=split_parameters,
        evidence=split_evidence,
        score=score,
        decision="pending",
    )
    return SemanticSplitCandidate(proposal=proposal, partition_map=partition.copy())


def make_semantic_split_candidates(
    tile_id: str,
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Build deterministic proposals and a proposal-ID keyed partition mapping.

    Each input mapping requires ``source_entity_id``, ``parent_mask``, and
    ``partition_map``.  Optional keyword fields mirror
    :func:`make_semantic_split_candidate`.
    """

    proposals: list[dict[str, Any]] = []
    partitions: dict[str, np.ndarray] = {}
    for index, raw in enumerate(candidates):
        if not isinstance(raw, Mapping):
            raise TypeError(f"semantic split candidate {index} must be a mapping")
        missing = [
            name
            for name in ("source_entity_id", "parent_mask", "partition_map")
            if name not in raw
        ]
        if missing:
            raise ValueError(f"semantic split candidate {index} is missing fields: {missing}")
        candidate = make_semantic_split_candidate(
            tile_id,
            str(raw["source_entity_id"]),
            np.asarray(raw["parent_mask"]),
            np.asarray(raw["partition_map"]),
            minimum_child_area_pixels=raw.get("minimum_child_area_pixels", 1),
            evidence=raw.get("evidence"),
            score=raw.get("score"),
            candidate_source=str(raw.get("candidate_source", "caller_supplied_partition")),
            parameters=raw.get("parameters"),
        )
        proposal_id = str(candidate.proposal["proposal_id"])
        if proposal_id in partitions:
            raise ValueError(f"duplicate semantic split proposal {proposal_id!r}")
        proposals.append(candidate.proposal)
        partitions[proposal_id] = candidate.partition_map
    proposals.sort(key=lambda value: str(value["proposal_id"]))
    return proposals, {str(value["proposal_id"]): partitions[str(value["proposal_id"])] for value in proposals}


def _positive_integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return result


__all__ = [
    "SemanticSplitCandidate",
    "canonical_semantic_split_partition",
    "make_semantic_split_candidate",
    "make_semantic_split_candidates",
    "semantic_split_partition_checksum",
    "validate_semantic_split_partition",
]
