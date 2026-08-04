from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, default=_json_default)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(name, **arrays)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phase_dir(root: Path, tile_id: str, phase: str) -> Path:
    return root / tile_id / phase


def manifest(path: Path, phase: str, inputs: Iterable[Path], outputs: Iterable[Path], config_hash: str, started: str, status: str = "complete", failure: str | None = None) -> dict[str, Any]:
    data = {
        "phase": phase, "status": status, "started_at": started, "ended_at": now(),
        "configuration_hash": config_hash,
        "inputs": [{"path": str(p.resolve()), "sha256": checksum(p)} for p in inputs if p.is_file()],
        "outputs": [str(p.resolve()) for p in outputs], "failure": failure,
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "stale": False,
    }
    atomic_json(path, data)
    return data


def mark_downstream_stale(tile_root: Path, after_phase: str) -> list[str]:
    phases = sorted(p for p in tile_root.iterdir() if p.is_dir()) if tile_root.exists() else []
    changed = []
    for directory in phases:
        if directory.name <= after_phase:
            continue
        path = directory / "manifest.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            data["stale"] = True
            data["stale_since"] = now()
            data["stale_reason"] = f"upstream phase {after_phase} was rerun"
            atomic_json(path, data)
            changed.append(directory.name)
    return changed


def require_fresh(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    manifest_path = path.parent / "manifest.json"
    if manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8")).get("stale"):
        raise RuntimeError(f"Required artifact is stale; rerun phase {path.parent.name}: {path}")
