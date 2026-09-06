"""Atomic, pickle-free persistence helpers for refinement artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


STORAGE_VERSION = "1.0"
SEGMENTATION_KEYS = (
    "instance_map",
    "semantic_map",
    "confidence_map",
    "source_map",
    "valid_mask",
)


def atomic_json(path: str | Path, value: Any, *, indent: int | None = 2) -> Path:
    """Atomically write JSON with stable mapping-key order."""

    target = _path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(
                value,
                stream,
                ensure_ascii=False,
                allow_nan=False,
                indent=indent,
                sort_keys=True,
                separators=(",", ":") if indent is None else None,
                default=_json_default,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return target.resolve()


def load_json(path: str | Path) -> Any:
    """Load one UTF-8 JSON artifact."""

    source = _existing_file(path)
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON artifact {source}: {exc}") from exc


def atomic_npz(path: str | Path, **arrays: np.ndarray) -> Path:
    """Atomically write compressed NumPy arrays without object pickles."""

    if not arrays:
        raise ValueError("At least one array is required")
    clean: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        if not isinstance(key, str) or not key:
            raise ValueError("NPZ array names must be non-empty strings")
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"NPZ array {key!r} may not use object dtype")
        clean[key] = array

    target = _path(path)
    if target.suffix.lower() != ".npz":
        raise ValueError(f"NPZ output path must end in .npz: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=".npz", dir=target.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary_name, **clean)
        os.replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return target.resolve()


def save_segmentation_bundle(
    path: str | Path,
    *,
    instance_map: np.ndarray,
    semantic_map: np.ndarray,
    confidence_map: np.ndarray,
    source_map: np.ndarray,
    valid_mask: np.ndarray,
) -> Path:
    """Persist the complete refined raster state in the Stage 2 map schema."""

    arrays = _segmentation_arrays(
        instance_map=instance_map,
        semantic_map=semantic_map,
        confidence_map=confidence_map,
        source_map=source_map,
        valid_mask=valid_mask,
    )
    return atomic_npz(path, storage_version=np.asarray(STORAGE_VERSION), **arrays)


def load_segmentation_bundle(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate a bundle created by :func:`save_segmentation_bundle`."""

    source = _existing_file(path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            missing = [key for key in (*SEGMENTATION_KEYS, "storage_version") if key not in archive]
            if missing:
                raise ValueError(f"Segmentation bundle is missing arrays: {missing}")
            version = str(np.asarray(archive["storage_version"]).item())
            if version != STORAGE_VERSION:
                raise ValueError(f"Unsupported refinement storage version: {version!r}")
            loaded = {key: np.asarray(archive[key]).copy() for key in SEGMENTATION_KEYS}
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(("Segmentation bundle", "Unsupported")):
            raise
        raise ValueError(f"Could not load segmentation bundle {source}: {exc}") from exc
    return _segmentation_arrays(**loaded)


def save_proposals(
    path: str | Path,
    proposals: Sequence[Mapping[str, Any]],
    *,
    tile_id: str,
    configuration_hash: str,
    input_checksum: str,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save a deterministic proposal-only document.

    Proposals are sorted by ``proposal_id`` when present, then by their
    canonical JSON representation.  This prevents traversal order from
    changing the review artifact.
    """

    records = _sorted_records(proposals, "proposal")
    document: dict[str, Any] = {
        "storage_version": STORAGE_VERSION,
        "document_type": "refinement_proposals",
        "mode": "proposal_only",
        "tile_id": _required_text(tile_id, "tile_id"),
        "configuration_hash": _required_text(configuration_hash, "configuration_hash"),
        "input_checksum": _required_text(input_checksum, "input_checksum"),
        "proposal_count": len(records),
        "proposals": records,
    }
    if metadata is not None:
        document["metadata"] = dict(metadata)
    return atomic_json(path, document)


def load_proposals(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a refinement proposal document."""

    document = _document(load_json(path), "refinement_proposals")
    records = document.get("proposals")
    if not isinstance(records, list):
        raise ValueError("Proposal document 'proposals' must be a list")
    if document.get("proposal_count") != len(records):
        raise ValueError("Proposal document count does not match its records")
    document["proposals"] = _sorted_records(records, "proposal")
    return document


def save_decisions(
    path: str | Path,
    decisions: Sequence[Mapping[str, Any]],
    *,
    tile_id: str,
    proposal_checksum: str,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save deterministic reviewer decisions bound to a proposal checksum."""

    records = _sorted_records(decisions, "decision")
    document: dict[str, Any] = {
        "storage_version": STORAGE_VERSION,
        "document_type": "refinement_decisions",
        "tile_id": _required_text(tile_id, "tile_id"),
        "proposal_checksum": _required_text(proposal_checksum, "proposal_checksum"),
        "decision_count": len(records),
        "decisions": records,
    }
    if metadata is not None:
        document["metadata"] = dict(metadata)
    return atomic_json(path, document)


def load_decisions(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a reviewer-decision document."""

    document = _document(load_json(path), "refinement_decisions")
    records = document.get("decisions")
    if not isinstance(records, list):
        raise ValueError("Decision document 'decisions' must be a list")
    if document.get("decision_count") != len(records):
        raise ValueError("Decision document count does not match its records")
    document["decisions"] = _sorted_records(records, "decision")
    return document


def save_refinement_history(
    path: str | Path,
    operations: Sequence[Mapping[str, Any]],
    *,
    tile_id: str,
    input_checksum: str,
    output_checksum: str,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Save the replayable, ordered history of approved operations."""

    records = [dict(record) for record in operations]
    _require_unique_identifier(records, ("operation_id",), "operation")
    document: dict[str, Any] = {
        "storage_version": STORAGE_VERSION,
        "document_type": "refinement_history",
        "tile_id": _required_text(tile_id, "tile_id"),
        "input_checksum": _required_text(input_checksum, "input_checksum"),
        "output_checksum": _required_text(output_checksum, "output_checksum"),
        "operation_count": len(records),
        # History order is semantically meaningful and is therefore retained.
        "operations": records,
    }
    if metadata is not None:
        document["metadata"] = dict(metadata)
    return atomic_json(path, document)


def load_refinement_history(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a replay history."""

    document = _document(load_json(path), "refinement_history")
    records = document.get("operations")
    if not isinstance(records, list):
        raise ValueError("History document 'operations' must be a list")
    if document.get("operation_count") != len(records):
        raise ValueError("History document count does not match its records")
    _require_unique_identifier(records, ("operation_id",), "operation")
    return document


def sha256_array(array: np.ndarray) -> str:
    """Hash array dtype, shape, and bytes so masks can be audit-bound."""

    value = np.asarray(array)
    if value.dtype.hasobject:
        raise TypeError("Object arrays cannot be hashed as refinement artifacts")
    contiguous = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of an existing file."""

    source = _existing_file(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _segmentation_arrays(**values: np.ndarray) -> dict[str, np.ndarray]:
    missing = [key for key in SEGMENTATION_KEYS if key not in values]
    if missing:
        raise ValueError(f"Segmentation values are missing: {missing}")
    instance_map = np.asarray(values["instance_map"])
    semantic_map = np.asarray(values["semantic_map"])
    confidence_map = np.asarray(values["confidence_map"])
    source_map = np.asarray(values["source_map"])
    valid_mask = np.asarray(values["valid_mask"])
    arrays = {
        "instance_map": instance_map,
        "semantic_map": semantic_map,
        "confidence_map": confidence_map,
        "source_map": source_map,
        "valid_mask": valid_mask,
    }
    shapes = {key: value.shape for key, value in arrays.items()}
    if any(value.ndim != 2 for value in arrays.values()) or len(set(shapes.values())) != 1:
        raise ValueError(f"All segmentation maps must be 2-D with one common shape, got {shapes}")
    if any(length <= 0 for length in instance_map.shape):
        raise ValueError("Segmentation map dimensions must be positive")
    if instance_map.dtype == np.bool_ or not np.issubdtype(instance_map.dtype, np.integer):
        raise TypeError("instance_map must use an integer dtype")
    if np.any(instance_map < 0):
        raise ValueError("instance_map cannot contain negative IDs")
    if np.any(instance_map > np.iinfo(np.int32).max):
        raise ValueError("instance_map IDs must fit in int32")
    if semantic_map.dtype == np.bool_ or not np.issubdtype(semantic_map.dtype, np.integer):
        raise TypeError("semantic_map must use an integer dtype")
    if np.any((semantic_map < 0) | (semantic_map > np.iinfo(np.uint8).max)):
        raise ValueError("semantic_map values must fit in uint8")
    if source_map.dtype == np.bool_ or not np.issubdtype(source_map.dtype, np.integer):
        raise TypeError("source_map must use an integer dtype")
    if np.any((source_map < 0) | (source_map > np.iinfo(np.uint8).max)):
        raise ValueError("source_map values must fit in uint8")
    if not np.issubdtype(confidence_map.dtype, np.floating):
        raise TypeError("confidence_map must use a floating-point dtype")
    if not np.isfinite(confidence_map).all() or np.any((confidence_map < 0) | (confidence_map > 1)):
        raise ValueError("confidence_map values must be finite and between 0 and 1")
    if valid_mask.dtype != np.bool_:
        raise TypeError("valid_mask must use boolean dtype")
    return {
        "instance_map": instance_map.astype(np.int32, copy=False),
        "semantic_map": semantic_map.astype(np.uint8, copy=False),
        "confidence_map": confidence_map.astype(np.float32, copy=False),
        "source_map": source_map.astype(np.uint8, copy=False),
        "valid_mask": valid_mask.astype(bool, copy=False),
    }


def _sorted_records(records: Sequence[Mapping[str, Any]], kind: str) -> list[dict[str, Any]]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError(f"{kind.title()} records must be a sequence")
    copied: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"{kind.title()} record {index} must be a mapping")
        copied.append(dict(record))
    keys = (f"{kind}_id", "proposal_id", "operation_id")
    _require_unique_identifier(copied, keys, kind)
    return sorted(copied, key=lambda item: _record_sort_key(item, keys))


def _record_sort_key(record: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[int, str]:
    for key in keys:
        if record.get(key) is not None:
            return (0, str(record[key]))
    return (1, _canonical_json(record))


def _require_unique_identifier(
    records: Sequence[Mapping[str, Any]], keys: tuple[str, ...], kind: str
) -> None:
    identifiers: list[str] = []
    for record in records:
        value = next((record[key] for key in keys if record.get(key) is not None), None)
        if value is not None:
            identifiers.append(str(value))
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Duplicate {kind} identifiers are not allowed")


def _document(value: Any, expected_type: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{expected_type} artifact must contain a JSON object")
    if value.get("storage_version") != STORAGE_VERSION:
        raise ValueError(f"Unsupported refinement storage version: {value.get('storage_version')!r}")
    if value.get("document_type") != expected_type:
        raise ValueError(f"Expected {expected_type!r}, got {value.get('document_type')!r}")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _json_default(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.name:
        raise ValueError("Artifact path must include a file name")
    return path


def _existing_file(value: str | Path) -> Path:
    path = _path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Refinement artifact not found: {path}")
    return path


def _required_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


__all__ = [
    "SEGMENTATION_KEYS",
    "STORAGE_VERSION",
    "atomic_json",
    "atomic_npz",
    "load_decisions",
    "load_json",
    "load_proposals",
    "load_refinement_history",
    "load_segmentation_bundle",
    "save_decisions",
    "save_proposals",
    "save_refinement_history",
    "save_segmentation_bundle",
    "sha256_array",
    "sha256_file",
]
