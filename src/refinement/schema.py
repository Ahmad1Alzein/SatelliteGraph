"""Deterministic, JSON-safe schemas for Stage 2 refinement decisions.

The refinement layer deliberately separates *proposals* (which never change
pixels) from *operations* (approved changes which were actually applied).
The helpers in this module contain no filesystem or model dependencies, so a
decision document can be validated and replayed independently of the CLI.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA_VERSION = "satellitegraph.stage2-refinement.v1"
PROPOSAL_SCHEMA_VERSION = f"{SCHEMA_VERSION}.proposal"
DECISION_SCHEMA_VERSION = f"{SCHEMA_VERSION}.decision"
OPERATION_SCHEMA_VERSION = f"{SCHEMA_VERSION}.operation"

OPERATION_TYPES = (
    "split_components",
    "semantic_split",
    "merge",
    "absorb",
    "mark_nodata",
)
PROPOSAL_DECISIONS = (
    "pending",
    "approved",
    "auto_apply",
    "rejected",
    "uncertain",
)
REVIEW_DECISIONS = ("approved", "rejected", "uncertain")
OPERATION_STATUSES = ("applied", "failed", "rolled_back")


def json_safe(value: Any) -> Any:
    """Return a deterministic value accepted by strict JSON encoders.

    NumPy scalars/arrays, paths, dataclasses, tuples, and sets are converted.
    Non-finite floating-point values are rejected because ``NaN`` and
    infinities are not valid JSON and make hashes non-portable.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Refinement JSON values must not contain NaN or infinity")
        return value
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return json_safe(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        clean: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            text_key = str(key)
            if text_key in clean:
                raise ValueError(f"Refinement JSON mapping has duplicate string key {text_key!r}")
            clean[text_key] = json_safe(item)
        return clean
    if isinstance(value, (set, frozenset)):
        clean_values = [json_safe(item) for item in value]
        return sorted(clean_values, key=canonical_json)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_safe(item) for item in value]
    raise TypeError(f"Unsupported refinement JSON value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize ``value`` to the canonical representation used for IDs."""

    return json.dumps(
        json_safe(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_digest(value: Any) -> str:
    """Return a SHA-256 digest of the canonical JSON representation."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def make_proposal(
    tile_id: str,
    operation_type: str,
    source_entity_ids: Sequence[str],
    *,
    target_entity_id: str | None = None,
    parameters: Mapping[str, Any] | None = None,
    evidence: Mapping[str, Any] | None = None,
    score: float | None = None,
    decision: str = "pending",
) -> dict[str, Any]:
    """Build and validate one deterministic, non-mutating proposal."""

    operation = _operation_type(operation_type)
    sources = _canonical_sources(operation, source_entity_ids)
    target = str(target_entity_id) if target_entity_id is not None else None
    clean_parameters = json_safe(dict(parameters or {}))
    clean_evidence = json_safe(dict(evidence or {}))
    identity = {
        "tile_id": str(tile_id),
        "operation_type": operation,
        "source_entity_ids": sources,
        "target_entity_id": target,
        "parameters": clean_parameters,
    }
    document = {
        "schema_version": PROPOSAL_SCHEMA_VERSION,
        "proposal_id": f"{tile_id}::proposal::{operation}::{stable_digest(identity)[:20]}",
        **identity,
        "evidence": clean_evidence,
        "score": _optional_score(score),
        "decision": str(decision),
    }
    validate_proposal(document)
    return document


def make_decision(
    proposal_id: str,
    decision: str,
    *,
    decision_source: str = "human_review",
    reviewer: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic review decision for a proposal."""

    value = str(decision)
    if value not in REVIEW_DECISIONS:
        raise ValueError(f"Unsupported review decision: {value!r}")
    identity = {
        "proposal_id": str(proposal_id),
        "decision": value,
        "decision_source": str(decision_source),
        "reviewer": None if reviewer is None else str(reviewer),
        "reason": None if reason is None else str(reason),
    }
    document = {
        "schema_version": DECISION_SCHEMA_VERSION,
        "decision_id": f"decision::{stable_digest(identity)[:20]}",
        **identity,
    }
    validate_decision(document)
    return document


def make_operation(
    tile_id: str,
    operation_type: str,
    source_entity_ids: Sequence[str],
    output_entity_ids: Sequence[str],
    *,
    target_entity_id: str | None = None,
    source_instance_ids: Sequence[int] = (),
    output_instance_ids: Sequence[int] = (),
    proposal_id: str | None = None,
    parameters: Mapping[str, Any] | None = None,
    evidence: Mapping[str, Any] | None = None,
    decision_source: str = "deterministic_rule",
    invariants: Mapping[str, Any] | None = None,
    status: str = "applied",
) -> dict[str, Any]:
    """Build and validate one replayable record of an applied operation."""

    operation = _operation_type(operation_type)
    sources = _canonical_sources(operation, source_entity_ids)
    outputs = _unique_strings(output_entity_ids, "output_entity_ids")
    target = str(target_entity_id) if target_entity_id is not None else None
    clean_parameters = json_safe(dict(parameters or {}))
    identity = {
        "tile_id": str(tile_id),
        "operation_type": operation,
        "source_entity_ids": sources,
        "target_entity_id": target,
        "output_entity_ids": outputs,
        "parameters": clean_parameters,
        "proposal_id": None if proposal_id is None else str(proposal_id),
    }
    document = {
        "schema_version": OPERATION_SCHEMA_VERSION,
        "operation_id": f"{tile_id}::operation::{operation}::{stable_digest(identity)[:20]}",
        **identity,
        "source_instance_ids": _unique_ints(source_instance_ids, "source_instance_ids"),
        "output_instance_ids": _unique_ints(output_instance_ids, "output_instance_ids"),
        "evidence": json_safe(dict(evidence or {})),
        "decision_source": str(decision_source),
        "invariants": json_safe(dict(invariants or {})),
        "status": str(status),
    }
    validate_operation(document)
    return document


def validate_proposal(document: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` when a proposal violates the public schema."""

    if document.get("schema_version") != PROPOSAL_SCHEMA_VERSION:
        raise ValueError(f"Unsupported refinement proposal schema: {document.get('schema_version')!r}")
    proposal_id = str(document.get("proposal_id", ""))
    if not proposal_id:
        raise ValueError("Refinement proposal_id must be non-empty")
    tile_id = str(document.get("tile_id", ""))
    if not tile_id:
        raise ValueError("Refinement proposal tile_id must be non-empty")
    operation = _operation_type(str(document.get("operation_type", "")))
    sources = _unique_strings(_sequence(document.get("source_entity_ids"), "source_entity_ids"), "source_entity_ids")
    target = document.get("target_entity_id")
    _validate_cardinality(operation, sources, None if target is None else str(target), outputs=None)
    decision = str(document.get("decision", ""))
    if decision not in PROPOSAL_DECISIONS:
        raise ValueError(f"Unsupported proposal decision: {decision!r}")
    _optional_score(document.get("score"))
    parameters = document.get("parameters")
    evidence = document.get("evidence")
    if not isinstance(parameters, Mapping) or not isinstance(evidence, Mapping):
        raise ValueError("Refinement proposal parameters and evidence must be mappings")
    if operation == "mark_nodata" and parameters.get("invalid_only") is not True:
        raise ValueError("mark_nodata proposals must declare parameters.invalid_only=true")
    if operation == "semantic_split":
        _validate_semantic_split_parameters(parameters)
        if decision in {"approved", "auto_apply"}:
            raise ValueError(
                "semantic_split proposals require a separate explicit review decision "
                "and can never be embedded as approved or auto_apply"
            )
    if sources != sorted(sources):
        raise ValueError("Refinement proposal source_entity_ids must be canonically sorted")
    identity = {
        "tile_id": tile_id,
        "operation_type": operation,
        "source_entity_ids": sources,
        "target_entity_id": None if target is None else str(target),
        "parameters": json_safe(dict(parameters)),
    }
    expected_id = f"{tile_id}::proposal::{operation}::{stable_digest(identity)[:20]}"
    if proposal_id != expected_id:
        raise ValueError("Refinement proposal_id does not match its deterministic content")
    json_safe(document)


def validate_decision(document: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` when a review decision is invalid or tampered."""

    if document.get("schema_version") != DECISION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported refinement decision schema: {document.get('schema_version')!r}")
    decision_id = str(document.get("decision_id", ""))
    proposal_id = str(document.get("proposal_id", ""))
    decision = str(document.get("decision", ""))
    decision_source = str(document.get("decision_source", ""))
    if not proposal_id or not decision_source:
        raise ValueError("Refinement decision proposal_id and decision_source must be non-empty")
    if decision not in REVIEW_DECISIONS:
        raise ValueError(f"Unsupported review decision: {decision!r}")
    identity = {
        "proposal_id": proposal_id,
        "decision": decision,
        "decision_source": decision_source,
        "reviewer": None if document.get("reviewer") is None else str(document.get("reviewer")),
        "reason": None if document.get("reason") is None else str(document.get("reason")),
    }
    expected_id = f"decision::{stable_digest(identity)[:20]}"
    if decision_id != expected_id:
        raise ValueError("Refinement decision_id does not match its deterministic content")
    json_safe(document)


def validate_operation(document: Mapping[str, Any]) -> None:
    """Raise ``ValueError`` when an applied operation violates the schema."""

    if document.get("schema_version") != OPERATION_SCHEMA_VERSION:
        raise ValueError(f"Unsupported refinement operation schema: {document.get('schema_version')!r}")
    if not str(document.get("operation_id", "")):
        raise ValueError("Refinement operation_id must be non-empty")
    if not str(document.get("tile_id", "")):
        raise ValueError("Refinement operation tile_id must be non-empty")
    operation = _operation_type(str(document.get("operation_type", "")))
    sources = _unique_strings(_sequence(document.get("source_entity_ids"), "source_entity_ids"), "source_entity_ids")
    outputs = _unique_strings(_sequence(document.get("output_entity_ids"), "output_entity_ids"), "output_entity_ids")
    target = document.get("target_entity_id")
    _validate_cardinality(operation, sources, None if target is None else str(target), outputs=outputs)
    source_instances = _unique_ints(_sequence(document.get("source_instance_ids"), "source_instance_ids"), "source_instance_ids")
    output_instances = _unique_ints(_sequence(document.get("output_instance_ids"), "output_instance_ids"), "output_instance_ids")
    if any(value <= 0 for value in (*source_instances, *output_instances)):
        raise ValueError("Refinement operation instance IDs must be positive")
    if str(document.get("status", "")) not in OPERATION_STATUSES:
        raise ValueError(f"Unsupported refinement operation status: {document.get('status')!r}")
    for name in ("parameters", "evidence", "invariants"):
        if not isinstance(document.get(name), Mapping):
            raise ValueError(f"Refinement operation {name} must be a mapping")
    if operation == "mark_nodata" and document["parameters"].get("invalid_only") is not True:
        raise ValueError("mark_nodata operations must declare parameters.invalid_only=true")
    if operation == "semantic_split":
        _validate_semantic_split_parameters(document["parameters"])
    if sources != sorted(sources):
        raise ValueError("Refinement operation source_entity_ids must be canonically sorted")
    identity = {
        "tile_id": str(document["tile_id"]),
        "operation_type": operation,
        "source_entity_ids": sources,
        "target_entity_id": None if target is None else str(target),
        "output_entity_ids": outputs,
        "parameters": json_safe(dict(document["parameters"])),
        "proposal_id": (
            None if document.get("proposal_id") is None else str(document.get("proposal_id"))
        ),
    }
    expected_id = (
        f"{document['tile_id']}::operation::{operation}::{stable_digest(identity)[:20]}"
    )
    if str(document["operation_id"]) != expected_id:
        raise ValueError("Refinement operation_id does not match its deterministic content")
    json_safe(document)


def _validate_cardinality(
    operation: str,
    sources: Sequence[str],
    target: str | None,
    *,
    outputs: Sequence[str] | None,
) -> None:
    if operation == "split_components":
        if len(sources) != 1 or target is not None:
            raise ValueError("split_components requires exactly one source and no target")
        if outputs is not None and len(outputs) < 2:
            raise ValueError("An applied split_components operation must create at least two outputs")
    elif operation == "semantic_split":
        if len(sources) != 1 or target is not None:
            raise ValueError("semantic_split requires exactly one source and no target")
        if outputs is not None and len(outputs) < 2:
            raise ValueError("An applied semantic_split operation must create at least two outputs")
    elif operation == "merge":
        if len(sources) < 2 or target is not None:
            raise ValueError("merge requires at least two sources and no target")
        if outputs is not None and len(outputs) != 1:
            raise ValueError("An applied merge operation must create exactly one output")
    elif operation == "absorb":
        if len(sources) != 1 or not target or target in sources:
            raise ValueError("absorb requires one source and one distinct target")
        if outputs is not None and len(outputs) != 1:
            raise ValueError("An applied absorb operation must retain exactly one output entity")
    elif operation == "mark_nodata":
        if not sources or target is not None:
            raise ValueError("mark_nodata requires at least one source and no target")
        if outputs is not None and outputs:
            raise ValueError("mark_nodata cannot create output entities")


def _operation_type(value: str) -> str:
    if value not in OPERATION_TYPES:
        raise ValueError(f"Unsupported refinement operation type: {value!r}")
    return value


def _validate_semantic_split_parameters(parameters: Mapping[str, Any]) -> None:
    """Validate the immutable raster-partition identity of a semantic split."""

    required = {
        "partition_sha256",
        "partition_shape",
        "partition_dtype",
        "child_count",
        "child_label_order",
        "minimum_child_area_pixels",
        "review_only",
        "automatic",
    }
    missing = sorted(required - set(parameters))
    if missing:
        raise ValueError(f"semantic_split parameters are missing required fields: {missing}")
    digest = parameters.get("partition_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("semantic_split partition_sha256 must be a lowercase SHA-256 digest")
    shape = parameters.get("partition_shape")
    if (
        not isinstance(shape, Sequence)
        or isinstance(shape, (str, bytes, bytearray))
        or len(shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
    ):
        raise ValueError("semantic_split partition_shape must contain two positive integers")
    if parameters.get("partition_dtype") != "int32":
        raise ValueError("semantic_split partition_dtype must be 'int32'")
    child_count = parameters.get("child_count")
    if isinstance(child_count, bool) or not isinstance(child_count, int) or child_count < 2:
        raise ValueError("semantic_split child_count must be an integer of at least two")
    minimum_area = parameters.get("minimum_child_area_pixels")
    if isinstance(minimum_area, bool) or not isinstance(minimum_area, int) or minimum_area < 1:
        raise ValueError("semantic_split minimum_child_area_pixels must be a positive integer")
    if parameters.get("child_label_order") != "ascending_contiguous_from_1":
        raise ValueError(
            "semantic_split child_label_order must be 'ascending_contiguous_from_1'"
        )
    if parameters.get("review_only") is not True or parameters.get("automatic") is not False:
        raise ValueError("semantic_split must declare review_only=true and automatic=false")


def _canonical_sources(operation: str, values: Sequence[str]) -> list[str]:
    sources = _unique_strings(values, "source_entity_ids")
    # Source order has no semantic meaning for these operations. Sorting makes
    # proposal and operation IDs independent of UI selection order.
    return sorted(sources)


def _unique_strings(values: Sequence[Any], name: str) -> list[str]:
    result = [str(value) for value in values]
    if any(not value for value in result):
        raise ValueError(f"{name} must not contain empty values")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values")
    return result


def _unique_ints(values: Sequence[Any], name: str) -> list[int]:
    result = [int(value) for value in values]
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values")
    return result


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"Refinement {name} must be a sequence")
    return value


def _optional_score(value: Any) -> float | None:
    if value is None:
        return None
    score = float(value)
    if not math.isfinite(score) or score < 0.0 or score > 1.0:
        raise ValueError("Refinement proposal score must be finite and between 0 and 1")
    return score


__all__ = [
    "DECISION_SCHEMA_VERSION",
    "OPERATION_SCHEMA_VERSION",
    "OPERATION_STATUSES",
    "OPERATION_TYPES",
    "PROPOSAL_DECISIONS",
    "PROPOSAL_SCHEMA_VERSION",
    "REVIEW_DECISIONS",
    "SCHEMA_VERSION",
    "canonical_json",
    "json_safe",
    "make_decision",
    "make_operation",
    "make_proposal",
    "stable_digest",
    "validate_operation",
    "validate_decision",
    "validate_proposal",
]
