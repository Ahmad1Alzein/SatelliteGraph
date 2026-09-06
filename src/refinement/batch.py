"""Bounded, resumable orchestration for unattended refinement of many tiles."""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from tqdm import tqdm

from src.config import DEFAULT_CONFIG, load_config, select_refinement_config, validate_config
from src.stage2.artifacts import now

from .config import config_hash, normalize_refinement_mode
from .pipeline import apply_refinement, load_refinement_result, propose_refinement
from .storage import atomic_json, load_json


@dataclass(slots=True)
class RefinementBatchResult:
    """Aggregate result for an ordered set of independently processed tiles."""

    workflow_mode: str
    tile_results: list[dict[str, Any]]
    report_path: Path
    started_at: str
    completed_at: str

    def summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for result in self.tile_results:
            status = str(result["status"])
            counts[status] = counts.get(status, 0) + 1
        failed = counts.get("failed", 0)
        return {
            "status": "partial" if failed else "complete",
            "workflow_mode": self.workflow_mode,
            "tile_count": len(self.tile_results),
            "success_count": len(self.tile_results) - failed,
            "failure_count": failed,
            "status_counts": dict(sorted(counts.items())),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "report_path": str(self.report_path),
            "tiles": copy.deepcopy(self.tile_results),
        }


def discover_refinement_tiles(
    *,
    project_root: str | Path = ".",
    config: Mapping[str, Any] | str | Path | None = None,
) -> list[str]:
    """Return sorted tile IDs with a Stage 2 ``12_final/manifest.json``."""

    root = Path(project_root).expanduser().resolve()
    selected = select_refinement_config(_load_project_config(config))
    stage2_root = _resolve_root(root, str(selected["stage2_root"]))
    if not stage2_root.is_dir():
        return []
    return sorted(
        child.name
        for child in stage2_root.iterdir()
        if child.is_dir() and (child / "12_final" / "manifest.json").is_file()
    )


def run_refinement_batch(
    tile_ids: Sequence[str] | None = None,
    *,
    project_root: str | Path = ".",
    config: Mapping[str, Any] | str | Path | None = None,
    workflow_mode: str = "automatic",
    workers: int = 1,
    resume: bool = True,
    continue_on_error: bool = True,
    show_progress: bool = True,
) -> RefinementBatchResult:
    """Refine many tiles with deterministic ordering and per-tile isolation.

    With ``tile_ids=None``, every complete Stage 2 tile below the configured
    artifact root is discovered. Automatic mode publishes validated results;
    human-review mode generates proposal bundles for later review.
    """

    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    root = Path(project_root).expanduser().resolve()
    project_config = _load_project_config(config)
    selected = select_refinement_config(project_config)
    mode = normalize_refinement_mode(workflow_mode)
    discovered = (
        discover_refinement_tiles(project_root=root, config=project_config)
        if tile_ids is None
        else [_safe_tile_id(value) for value in tile_ids]
    )
    ordered_tiles = sorted(set(discovered))
    if not ordered_tiles:
        raise ValueError("No complete Stage 2 tiles were selected for refinement")

    started = now()

    def process(tile_id: str) -> dict[str, Any]:
        try:
            if mode == "automatic" and resume:
                resumed = _resume_result(root, tile_id, selected, mode)
                if resumed is not None:
                    return resumed
            if mode == "automatic":
                result = apply_refinement(
                    tile_id,
                    project_root=root,
                    config=project_config,
                    workflow_mode="automatic",
                    show_progress=False,
                )
            else:
                result = propose_refinement(
                    tile_id,
                    project_root=root,
                    config=project_config,
                    show_progress=False,
                )
            return {
                "tile_id": tile_id,
                "status": "complete" if mode == "automatic" else "proposed",
                "summary": result.summary(),
            }
        except Exception as exc:
            if not continue_on_error:
                raise
            return {
                "tile_id": tile_id,
                "status": "failed",
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }

    results_by_tile: dict[str, dict[str, Any]] = {}
    progress = tqdm(
        total=len(ordered_tiles),
        unit="tile",
        dynamic_ncols=True,
        disable=not show_progress,
        desc=f"Refinement batch | {mode}",
    )
    try:
        if workers == 1:
            for tile_id in ordered_tiles:
                results_by_tile[tile_id] = process(tile_id)
                progress.update(1)
        else:
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="refinement",
            ) as executor:
                futures = {
                    executor.submit(process, tile_id): tile_id
                    for tile_id in ordered_tiles
                }
                for future in as_completed(futures):
                    tile_id = futures[future]
                    results_by_tile[tile_id] = future.result()
                    progress.update(1)
    finally:
        progress.close()

    completed = now()
    ordered_results = [results_by_tile[tile_id] for tile_id in ordered_tiles]
    output_root = _resolve_root(root, str(selected["output"]["root_directory"]))
    report_path = output_root / "_batch" / "refinement_batch_report.json"
    result = RefinementBatchResult(
        workflow_mode=mode,
        tile_results=ordered_results,
        report_path=report_path.resolve(),
        started_at=started,
        completed_at=completed,
    )
    atomic_json(
        report_path,
        {
            **result.summary(),
            "configuration_hash": config_hash(selected),
            "workers": workers,
            "resume": bool(resume),
            "continue_on_error": bool(continue_on_error),
        },
    )
    return result


def _resume_result(
    project_root: Path,
    tile_id: str,
    selected: Mapping[str, Any],
    workflow_mode: str,
) -> dict[str, Any] | None:
    output_root = _resolve_root(
        project_root, str(selected["output"]["root_directory"])
    )
    directory = output_root / tile_id / str(selected["output"]["phase_directory"])
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        document = load_json(manifest_path)
        if document.get("status") != "complete" or document.get("stale"):
            return None
        if str(document.get("workflow_mode", "")) != workflow_mode:
            return None
        if str(document.get("configuration_hash", "")) != config_hash(selected):
            return None
        loaded = load_refinement_result(directory)
    except (OSError, RuntimeError, ValueError):
        return None
    summary = loaded.summary()
    summary["status"] = "skipped"
    return {"tile_id": tile_id, "status": "skipped", "summary": summary}


def _load_project_config(
    value: Mapping[str, Any] | str | Path | None,
) -> dict[str, Any]:
    if value is None:
        return load_config(DEFAULT_CONFIG)
    if isinstance(value, (str, Path)):
        return load_config(value)
    result = copy.deepcopy(dict(value))
    validate_config(result)
    return result


def _resolve_root(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _safe_tile_id(value: Any) -> str:
    tile_id = str(value)
    if not tile_id or Path(tile_id).name != tile_id or tile_id in {".", ".."}:
        raise ValueError("tile_id must be one safe path component")
    return tile_id


__all__ = [
    "RefinementBatchResult",
    "discover_refinement_tiles",
    "run_refinement_batch",
]
