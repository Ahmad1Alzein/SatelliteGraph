"""Reproduce the 7-tile 2013-2023 experiment with one command.

  python run_experiment.py                         # data in ./data, checkpoints in ./checkpoints, output in ./work
  python run_experiment.py --work D:/runs/repro1 --data D:/my_data --checkpoints D:/my_ckpt

Steps (each stops the run on failure; the full log is in <work>/logs/run.log):
  1 check inputs        every input and checkpoint file against reference/input_manifest.json (SHA-256)
  2 prepare work folder copy pipeline/ into <work>, add reference/shift_samples.csv, record the environment
  3 prepare inputs      georeference the dataset tiles, build the co-registration composites
  4 per-tile pipeline   for each of the 7 tiles: segment 2023 + freeze, co-register + harmonize 2013 (paired 2023 v2
                        release where harmonization blurs 2023), segment 2013 + freeze, region graphs, temporal layer
                        + freeze, spatio-temporal graph + freeze, rule-based change labels, audit montages
  5 audit verdicts      attach the shipped human audit verdicts (guarded by change-table hashes)
  6 strip               stitch the 7 tiles by metadata x0/y0, strip spatio-temporal graph
  7 GNN                 relational GNN, leave-one-tile-out (deterministic CPU training)
  8 figures             thesis figures; GNN prediction figures for tile_00002
  9 freeze strip        checks F1-F5 incl. full rebuild + GNN retraining; read-only release
 10 verify              compare all data outputs and key results with the original experiment
Runtime on the original machine (GTX 1650, 4 CPU threads for the GNN): about 1 h 30 min.

Stage groups (--until stops after a group; --resume continues an existing work folder):
  segmentation  steps 1-3 + the segmentation part of step 4 (14 + paired 2023 v2 frozen releases)
  graph         region graphs, temporal layers, spatio-temporal graphs (step 4) + strip mosaic/graph (step 6)
  classical     rule-based change labels + audit montages (step 4) + audit verdicts (step 5)
  gnn           steps 7-9 (needs classical: its change tables are the GNN's pseudo-labels) + full verification (10)
  python run_experiment.py --work W --until segmentation
  python run_experiment.py --work W --resume --until graph        (later: --until classical, then --until gnn)
Groups run per group for all tiles instead of tile by tile; every step reads only the frozen outputs of the
previous ones, so the results are identical (checked by the full verification of a staged run).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parent
TILES = [f"tile_{i:05d}" for i in range(7)]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_inputs(data: Path, ckpt: Path, strict: bool):
    man = json.loads((PKG / "reference" / "input_manifest.json").read_text())
    missing, wrong = [], []
    for rel, h in man.items():
        top, rest = rel.split("/", 1)
        p = (data if top == "data" else ckpt) / rest
        if not p.exists():
            missing.append(str(p))
        elif sha(p) != h:
            wrong.append(str(p))
    print(f"  {len(man) - len(missing) - len(wrong)}/{len(man)} input files present with the expected SHA-256")
    if missing:
        print("  MISSING:\n    " + "\n    ".join(missing))
        raise SystemExit("required inputs are missing - see docs/frozen-semantic-experiment.md, section 'Inputs'")
    if wrong:
        print("  DIFFERENT CONTENT:\n    " + "\n    ".join(wrong))
        if strict:
            raise SystemExit("inputs differ from the original experiment (use --skip-input-check to run anyway; results will then differ)")


def environment():
    env = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ("torch", "numpy", "pandas", "rasterio", "cv2", "scipy", "shapely", "PIL", "transformers", "ultralytics", "networkx", "matplotlib"):
        try:
            env[m] = __import__(m).__version__
        except Exception as e:  # noqa: BLE001
            env[m] = f"not importable ({type(e).__name__})"
    try:
        from osgeo import gdal
        env["gdal"] = gdal.__version__
    except Exception:  # noqa: BLE001
        env["gdal"] = "not importable"
    try:
        import torch
        env["cuda_available"] = torch.cuda.is_available()
        env["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        env["torch_cuda"] = torch.version.cuda
    except Exception:  # noqa: BLE001
        pass
    return env


def run(step, args, work, log, envv):
    t = time.time()
    print(f"\n=== {step}", flush=True)
    log.write(f"\n===== {step}: {' '.join(args)}\n"); log.flush()
    p = subprocess.Popen([sys.executable, "-u", *args], cwd=work, env=envv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    for line in p.stdout:
        log.write(line)
        if "warning" not in line.lower() and line.strip():
            print("  " + line.rstrip()[:220], flush=True)
    p.wait(); log.flush()
    print(f"--- {step}: {'OK' if p.returncode == 0 else 'FAILED'} ({(time.time() - t) / 60:.1f} min)", flush=True)
    if p.returncode != 0:
        raise SystemExit(f"step '{step}' failed (exit {p.returncode}); see {work / 'logs' / 'run.log'}")


STAGES = ("segmentation", "graph", "classical", "gnn")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, default=PKG / "work", help="output folder (must not exist yet, unless --resume)")
    ap.add_argument("--data", type=Path, default=None, help="folder with dataset/ and aligned/ (default ./data)")
    ap.add_argument("--checkpoints", type=Path, default=None, help="folder with satlens-segformer/ and DelineateAnythingv2.pt (default ./checkpoints)")
    ap.add_argument("--skip-input-check", action="store_true", help="run even if inputs differ from the original experiment")
    ap.add_argument("--until", choices=STAGES, default="gnn", help="last stage group to run (default gnn = the whole experiment)")
    ap.add_argument("--resume", action="store_true", help="continue an existing work folder after its last completed stage group")
    a = ap.parse_args()
    work = a.work.resolve()
    ledger = work / "logs" / "stages.json"
    state = json.loads(ledger.read_text()) if a.resume and ledger.exists() else None
    if a.resume and state is None:
        raise SystemExit(f"--resume needs an existing staged run: {ledger} not found")
    data = (a.data or (Path(state["data"]) if state else PKG / "data")).resolve()
    ckpt = (a.checkpoints or (Path(state["checkpoints"]) if state else PKG / "checkpoints")).resolve()
    t0 = time.time()

    print("=== 1 check inputs")
    check_inputs(data, ckpt, strict=not a.skip_input_check)

    if state is None:
        print("=== 2 prepare work folder")
        if work.exists() and any(work.iterdir()):
            raise SystemExit(f"{work} exists and is not empty: frozen releases are never overwritten - choose a new --work folder (or --resume)")
        shutil.copytree(PKG / "pipeline", work, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
        (work / "coregistration").mkdir()
        shutil.copy2(PKG / "reference" / "shift_samples.csv", work / "coregistration" / "shift_samples.csv")
        (work / "logs").mkdir()
        env = environment()
        (work / "logs" / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
        ref_env = json.loads((PKG / "environment" / "reference_environment.json").read_text())
        diffs = {k: (ref_env[k], env.get(k)) for k in ref_env if k not in ("platform", "gpu") and env.get(k) != ref_env[k]}
        print(f"  python {env['python']} · torch {env.get('torch')} · GPU {env.get('gpu')}")
        if diffs:
            print("  NOTE: environment differs from the original experiment (results may not be byte-identical):")
            for k, (r, g) in diffs.items():
                print(f"    {k}: original {r}, here {g}")
        state = {"data": str(data), "checkpoints": str(ckpt), "completed": []}
    else:
        print(f"=== 2 resume {work} (completed: {', '.join(state['completed']) or 'none'})")
    envv = dict(os.environ, REPRO_PACKAGE=str(PKG), REPRO_DATA=str(data), REPRO_CHECKPOINTS=str(ckpt),
                YOLO_CONFIG_DIR=str(work / ".ultralytics"), PYTHONUTF8="1", PYTHONIOENCODING="utf-8")

    def done(group):
        state["completed"].append(group); state[f"{group}_finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        ledger.write_text(json.dumps(state, indent=2), encoding="utf-8")

    todo = [g for g in STAGES[:STAGES.index(a.until) + 1] if g not in state["completed"]]
    with open(work / "logs" / "run.log", "a", encoding="utf-8") as log:
        if "segmentation" in todo:
            run("3 prepare inputs", ["prepare_inputs.py"], work, log, envv)
            run("4a segmentation (7 tiles)", ["run_tile_pipeline.py", "--stages=segmentation", *TILES], work, log, envv)
            done("segmentation")
        if "graph" in todo:
            run("4b region graphs, temporal layers, spatio-temporal graphs (7 tiles)", ["run_tile_pipeline.py", "--stages=graph", *TILES], work, log, envv)
            run("6 strip mosaic + spatio-temporal graph", ["build_mosaic.py"], work, log, envv)
            done("graph")
        if "classical" in todo:
            run("4c rule-based change detection + audit montages (7 tiles)", ["run_tile_pipeline.py", "--stages=classical", *TILES], work, log, envv)
            run("5 audit verdicts", ["transfer_audit_verdicts.py"], work, log, envv)
            done("classical")
        if "gnn" in todo:
            run("7 GNN change detection", ["gnn_change.py"], work, log, envv)
            run("8a thesis figures", ["thesis_figures.py"], work, log, envv)
            run("8b GNN prediction figures (tile_00002)", ["validation/gnn_predictions_tile_viz.py", "tile_00002"], work, log, envv)
            run("9 freeze strip release", ["freeze_strip.py"], work, log, envv)
            done("gnn")

    sys.path.insert(0, str(PKG))
    import verify_outputs
    full = "gnn" in state["completed"]
    print(f"\n=== 10 verify against the original experiment ({'all outputs' if full else 'outputs produced so far'})")
    code = verify_outputs.main(work, partial=not full)
    print(f"\ntotal {(time.time() - t0) / 60:.1f} min · completed: {', '.join(state['completed'])} · report: {work / 'REPRODUCTION_REPORT.md'}")
    sys.exit(code)


if __name__ == "__main__":
    main()
