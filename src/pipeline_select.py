"""Selectable native pipelines with persisted options and explicit format adapters.

Both segmentation and graph methods accept metadata-based folders or individual
images. Published-study reproduction remains available through run_experiment.py.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from src.config import DEFAULT_CONFIG
from src.pipeline_state import input_fingerprint, validate_resume

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "pipelines" / "semantic_temporal"
NATIVE = {(s,g,c) for s in ('old','new') for g in ('old','new') for c in ('classical','gnn')}
NEW_GROUP = {"segmentation": "segmentation", "graph": "graph", "classical": "classical", "gnn": "gnn"}


def resolve_options(saved: dict, supplied: dict, *, initial: bool, project_root: Path) -> dict:
    """Only explicit invocation values override effective saved options."""
    explicit = {k: v for k, v in supplied.items() if v is not None}
    merged = {**saved, **explicit}
    if explicit.get('checkpoints') is not None and explicit.get('semantic_models') is None:
        merged['semantic_models']=explicit['checkpoints']
    if explicit.get("registration_json") is not None:
        merged["register_images"] = False
    elif explicit.get("register_images") is True:
        merged.pop("registration_json", None)
    defaults = {"data": project_root / "data/experiment_2013_2023",
                "checkpoints": project_root / "checkpoints/experiment_2013_2023",
                "config": project_root / "config.yaml", "skip_input_check": False,
                "register_images": False, "no_progress": False}
    defaults.update(device='cpu',inference_window=512,inference_overlap=128,tta=True,
                    semantic_models=project_root/'checkpoints/semantic_temporal')
    for key, value in defaults.items():
        merged.setdefault(key, value)
    for key in ("data", "checkpoints", "config", "registration_json", "project_root", "semantic_models", "gnn_checkpoint"):
        if key in merged and not isinstance(merged[key], dict):
            p = Path(merged[key]).expanduser()
            merged[key] = str((p if p.is_absolute() else project_root / p).resolve())
    for key in ("folders", "images", "metadata", "source_rasters"):
        if merged.get(key):
            merged[key] = [str((Path(p) if Path(p).is_absolute() else project_root / p).resolve())
                           if p is not None else None for p in merged[key]]
    return merged


def check_combination(segmentation: str | None, graph: str | None, change: str | None) -> None:
    """All eight combinations use explicit format adapters."""
    for value,allowed in ((segmentation,('old','new')),(graph,('old','new')),(change,('classical','gnn'))):
        if value is not None and value not in allowed: raise ValueError(f'Unknown pipeline method: {value}')


# ----------------------------------------------------------------------------- run ledger
def _ledger(output: Path) -> dict:
    p = output / "run.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _save(output: Path, state: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    tmp = output / "run.json.tmp"
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    tmp.replace(output / "run.json")


def _git_commit() -> str | None:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def _start(output: Path, stage: str, method: str, options: dict) -> dict:
    state = _ledger(output)
    from pipelines.semantic_temporal.runtime.runner import relocate_run
    state=relocate_run(output,state)
    effective = resolve_options(state.get("options", {}), options, initial=not bool(state),
                                project_root=Path(options.get("project_root") or REPO))
    fingerprint = input_fingerprint(effective)
    validate_resume(state, stage, fingerprint)
    previous = state.get("stages", {})
    if stage != "segmentation":
        seg = previous.get("segmentation", {})
        if seg.get("status") != "complete":
            raise ValueError(f"{output} has no completed segmentation stage; run 'pipeline segment' first")
    if stage == "change" and previous.get("graph", {}).get("status") != "complete":
        raise ValueError(f"{output} has no completed graph stage; run 'pipeline graph' first")
    check_combination(previous.get("segmentation", {}).get("method", method if stage == "segmentation" else None),
                      method if stage == "graph" else previous.get("graph", {}).get("method"),
                      method if stage == "change" else None)
    state.setdefault("created", time.strftime("%Y-%m-%d %H:%M:%S"))
    state["git_commit"] = _git_commit()
    state["options"] = effective
    state['output']=str(output)
    state["input_fingerprint"] = fingerprint
    for dependent in ({"segmentation": ("graph", "change"), "graph": ("change",)}.get(stage, ())):
        state.setdefault("stages", {}).pop(dependent, None)
    state.setdefault("stages", {})[stage] = {"method": method, "status": "running", "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    _save(output, state)
    return state


def _finish(output: Path, state: dict, stage: str, **info) -> dict:
    state["stages"][stage].update(status="complete", finished=time.strftime("%Y-%m-%d %H:%M:%S"), **info)
    _save(output, state)
    return {"stage": stage, **state["stages"][stage], "run": str(output / "run.json")}


def _fail(output: Path, state: dict, stage: str, exc: Exception) -> None:
    state["stages"][stage].update(status="failed", error=f"{type(exc).__name__}: {exc}")
    _save(output, state)


# ----------------------------------------------------------------------------- new pipeline (experiment package)
def _run_new(output: Path, until: str, options: dict) -> dict:
    work = output / "work"
    cmd = [sys.executable, str(PACKAGE / "run_experiment.py"), "--work", str(work), "--until", NEW_GROUP[until]]
    if (work / "logs" / "stages.json").exists():
        cmd.append("--resume")
    for key, flag in (("data", "--data"), ("checkpoints", "--checkpoints")):
        if options.get(key):
            cmd += [flag, str(Path(options[key]).resolve())]
    if options.get("skip_input_check"):
        cmd.append("--skip-input-check")
    r = subprocess.run(cmd, cwd=PACKAGE, env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1"))  # readable logs on any console
    stages = json.loads((work / "logs" / "stages.json").read_text()) if (work / "logs" / "stages.json").exists() else {}
    report = json.loads((work / "reproduction_report.json").read_text()) if (work / "reproduction_report.json").exists() else {}
    verification_exit = r.returncode in (1, 2) and bool(report.get("status"))
    if NEW_GROUP[until] not in stages.get("completed", []) or (r.returncode != 0 and not verification_exit):
        raise RuntimeError(f"experiment stage '{until}' did not complete (exit {r.returncode}); see {work / 'logs' / 'run.log'}")
    report = json.loads((work / "reproduction_report.json").read_text()) if (work / "reproduction_report.json").exists() else {}
    return {"work": str(work), "verification": report.get("status"),
            "verification_note": "compared with the published 2013-2023 results (differences are expected for other inputs)"}


# ----------------------------------------------------------------------------- old pipeline (SatelliteGraph stages)
def _years(options: dict) -> list[tuple[int, Path]]:
    if not options.get("folders"):
        raise ValueError("the old pipeline needs --folders EARLIER LATER (tile folders of the two years)")
    folders = [Path(f) for f in options["folders"]]
    years = options.get("years") or [int(f.name) for f in folders]
    if len(folders) != 2 or len(years) != 2:
        raise ValueError("the old pipeline needs exactly two year folders (earlier year first)")
    return list(zip(map(int, years), folders))


def _run_old_annual(output: Path, options: dict, stop_after: str | None) -> dict:
    from src.folder_pipeline import run_folder
    rasters = options.get("source_rasters") or [None, None]
    metas = options.get("metadata") or [None, None]
    out = {}
    for (year, folder), raster, meta in zip(_years(options), rasters, metas):
        out[year] = run_folder(folder, year, output / f"annual_{year}", source_raster=raster, metadata=meta,
                               config=options.get("config") or DEFAULT_CONFIG, project_root=REPO,
                               show_progress=not options.get("no_progress"), stop_after=stop_after)
    return {"annual": {str(y): str(output / f"annual_{y}") for y in out}, "results": out}


def _run_old_classical(output: Path, options: dict) -> dict:
    from src.folder_pipeline import compare_folders
    (y1, _), (y2, _) = _years(options)
    return compare_folders(output / f"annual_{y1}", output / f"annual_{y2}", output / "classical_old",
                           registration_json=options.get("registration_json"), register_images=bool(options.get("register_images")),
                           boundary_tolerance=options.get("boundary_tolerance"), config=options.get("config") or DEFAULT_CONFIG,
                           show_progress=not options.get("no_progress"),
                           project_root=REPO)


# ----------------------------------------------------------------------------- public stage functions
def segment(method: str, output, **options) -> dict:
    output = Path(output).resolve()
    state = _start(output, "segmentation", method, options)
    options = state["options"]
    try:
        info = _execute_stage(output,'segmentation',method,options,state)
    except Exception as exc:
        _fail(output, state, "segmentation", exc); raise
    return _finish(output, state, "segmentation", result=info)


def graph(method: str, output, **options) -> dict:
    output = Path(output).resolve()
    state = _start(output, "graph", method, options)
    options = state["options"]
    try:
        info = _execute_stage(output,'graph',method,options,state)
    except Exception as exc:
        _fail(output, state, "graph", exc); raise
    return _finish(output, state, "graph", result=info)


def change(method: str, output, **options) -> dict:
    output = Path(output).resolve()
    state = _start(output, "change", method, options)
    options = state["options"]
    graph_method = state["stages"]["graph"]["method"]
    try:
        info = _execute_stage(output,'change',method,options,state)
    except Exception as exc:
        _fail(output, state, "change", exc); raise
    result=_finish(output,state,'change',result=info)
    if info.get('status') in ('insufficient_data','training_only'):
        state['stages']['change']['status']=info['status'];_save(output,state);result['status']=info['status']
    return result


def _execute_stage(output,stage,method,options,state):
    from pipelines.semantic_temporal.runtime.runner import execute_stage
    return execute_stage(output,stage,method,options,state)


def run(segmentation: str, graph_method: str, change_method: str, output, *, until='change',resume=False,**options) -> dict:
    if (segmentation, graph_method, change_method) not in NATIVE:
        check_combination(segmentation, graph_method, change_method)
        raise ValueError(f"unsupported combination {segmentation} -> {graph_method} -> {change_method}")
    if until not in ('segmentation','graph','change'): raise ValueError('Unknown final pipeline stage')
    output=Path(output).resolve();state=_ledger(output)
    if resume:
        from pipelines.semantic_temporal.runtime.runner import relocate_run
        state=relocate_run(output,state)
        effective=resolve_options(state.get('options',{}),options,initial=not bool(state),project_root=Path(options.get('project_root') or REPO))
        validate_resume(state,'segmentation',input_fingerprint(effective))
        downstream_changed=any(effective.get(k)!=state.get('options',{}).get(k) for k in ('gnn_checkpoint','gnn_epochs','gnn_hidden','boundary_tolerance'))
        state['options']=effective;_save(output,state)
    else: downstream_changed=False
    results={}
    for stage,method,execute in (('segmentation',segmentation,segment),('graph',graph_method,graph),('change',change_method,change)):
        saved=state.get('stages',{}).get(stage,{})
        if resume and saved.get('status')=='complete' and saved.get('method')==method and not (stage=='change' and downstream_changed): results[stage]=saved
        else: results[stage]=execute(method,output,**options if stage=='segmentation' else {})
        if stage==until: break
    return results
