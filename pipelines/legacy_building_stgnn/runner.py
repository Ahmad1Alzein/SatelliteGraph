"""Optional legacy building spatio-temporal GNN (frozen 4 Oct 2026 experiment, Shouf 2013 -> 2023).

The scientific code in original/ is a byte-identical copy of the frozen release
(FROZEN_strip_tile_00000-00006_v1/code and run_dataset_2013_2023/validation); this module only
stages verified inputs in the folder layout that code expects, runs it unchanged, and checks the
outputs against the frozen originals. It never imports or alters the multi-class runtime.

Bundle (hash-verified copy of the frozen inputs, built once by `bundle`):
  strip/          frozen strip spatio-temporal graph (no GNN outputs)
  labels/change_stgraph_2013_2023_<tile>/nodes_change.csv, audit/*.csv   rule pseudo-labels and audit verdicts
  dataset/2023/metadata.csv                                              tile placement on the strip
  frozen_gnn/     the original GNN outputs (reference only; reused with --reuse-frozen-predictions)
Run folder:
  work/           original code + staged inputs; outputs exactly where the original code writes them
  figures/        copies of the generated figures
  report.json     evidence type, metrics, environment, verification against the frozen outputs
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ORIGINAL = PACKAGE / "original"
REFERENCE = PACKAGE / "reference"
TILES = [f"tile_{i:05d}" for i in range(7)]
STRIP_NAME = "strip_tile_00000-00006"
FROZEN_STRIP = "FROZEN_strip_tile_00000-00006_v1"
GNN_FILES = ("predictions.csv", "audit_agreement.csv", "summary.json", "history_mlp.json", "history_gnn_spatial.json", "history_gnn_full.json")
STRIP_FIGURES = ("fig5a_gnn_architecture.png", "fig5b_gnn_results.png", "fig5c_gnn_change_map.png")
TILE_FIGURES = ("1_predictions_overlay.png", "2_probability_map.png", "3_gallery_2013.png", "3_gallery_2023.png", "predicted_nodes.csv")
EVIDENCE = ("Agreement with rule-based pseudo-labels (leave-one-tile-out) and with the earlier visual audit; "
            "not accuracy against independent ground truth.")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pixel_sha256(path: Path) -> str:
    """Hash of the decoded image (mode, size, pixels): PNG re-encoding does not change it, any pixel change does."""
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        array = np.asarray(image)
        header = f"{image.mode}|{array.shape}|{array.dtype}".encode()
    return hashlib.sha256(header + np.ascontiguousarray(array).tobytes()).hexdigest()


def _read(name: str) -> dict:
    return json.loads((REFERENCE / name).read_text(encoding="utf-8"))


def _files(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def check_code() -> list[str]:
    """Names of original files whose bytes differ from the recorded frozen code (empty list = intact)."""
    expected = _read("code_manifest.json")["files"]
    actual = _files(ORIGINAL)
    return sorted(k for k in set(expected) | set(actual) if expected.get(k) != actual.get(k) and not k.endswith(".pyc") and "__pycache__" not in k)


def verify_bundle(bundle: Path, manifest: dict | None = None) -> dict:
    manifest = manifest if manifest is not None else _read("input_manifest.json")["files"]
    actual = _files(Path(bundle))
    missing = sorted(set(manifest) - set(actual))
    changed = sorted(k for k in manifest if k in actual and actual[k] != manifest[k])
    if missing or changed:
        raise ValueError(f"Legacy input bundle does not match the frozen inputs: missing={missing[:5]} changed={changed[:5]}")
    return {"files": len(manifest), "status": "verified"}


def bundle_inputs(source: Path, metadata: Path, bundle: Path, manifest: dict | None = None) -> dict:
    """Copy the frozen inputs from an experiment folder (containing graphs/) into a self-contained bundle."""
    source, bundle = Path(source), Path(bundle)
    if bundle.exists() and any(bundle.iterdir()):
        raise FileExistsError(f"Bundle folder is not empty: {bundle}")
    strip = source / "graphs" / FROZEN_STRIP
    copies = {}
    for p in strip.rglob("*"):
        rel = p.relative_to(strip)
        if p.is_file() and rel.parts[0] not in ("gnn", "thesis_figures", "code", "labels"):
            copies[Path("strip") / rel] = p
    for name in GNN_FILES:
        copies[Path("frozen_gnn") / name] = strip / "gnn" / name
    for tile in TILES:
        d = source / "graphs" / f"change_stgraph_2013_2023_{tile}"
        copies[Path("labels") / d.name / "nodes_change.csv"] = d / "nodes_change.csv"
        for p in sorted((d / "audit").glob("*.csv")):
            copies[Path("labels") / d.name / "audit" / p.name] = p
    copies[Path("dataset") / "2023" / "metadata.csv"] = Path(metadata)
    for rel, src in copies.items():
        (bundle / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, bundle / rel)
    return verify_bundle(bundle, manifest)


def prepare_workspace(bundle: Path, output: Path) -> Path:
    """Lay out the original folder structure: <output>/work is the original code's HERE."""
    bundle, output = Path(bundle), Path(output)
    work = output / "work"
    if work.exists():
        raise FileExistsError(f"Run folder already used (legacy runs are never overwritten): {output}")
    shutil.copytree(ORIGINAL, work, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(bundle / "strip", work / "graphs" / STRIP_NAME)
    shutil.copytree(bundle / "labels", work / "graphs", dirs_exist_ok=True)
    (output / "dataset" / "2023").mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundle / "dataset" / "2023" / "metadata.csv", output / "dataset" / "2023" / "metadata.csv")
    (work / "thesis_figures").mkdir()   # done by thesis_figures.py's __main__ (OUT.mkdir), which fig5() alone skips
    return work


def _call(python: str, args: list[str], work: Path, log) -> None:
    log.write(f"\n$ {python} {' '.join(args)}\n"); log.flush()
    result = subprocess.run([python, *args], cwd=work, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Legacy step failed ({' '.join(args)}); see {log.name}")


def environment(python: str) -> dict:
    code = ("import json,platform,torch,numpy,pandas;print(json.dumps({'python':platform.python_version(),"
            "'torch':torch.__version__,'numpy':numpy.__version__,'pandas':pandas.__version__}))")
    out = subprocess.run([python, "-c", code], capture_output=True, text=True)
    return json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else {"error": out.stderr[-500:]}


def run(bundle: Path, output: Path, tiles=None, python: str | None = None, reuse_frozen_predictions: bool = False) -> dict:
    python = python or sys.executable
    tiles = list(tiles or TILES)
    unknown = sorted(set(tiles) - set(TILES))
    if unknown:
        raise ValueError(f"Unknown tiles: {unknown}")
    bad = check_code()
    if bad:
        raise ValueError(f"Original legacy code was modified: {bad}")
    verify_bundle(bundle)
    output = Path(output).resolve()
    work = prepare_workspace(bundle, output)
    gnn = work / "graphs" / STRIP_NAME / "gnn"
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(output / "run.log", "w", encoding="utf-8") as log:
        if reuse_frozen_predictions:
            gnn.mkdir(parents=True)
            for name in GNN_FILES:
                shutil.copy2(Path(bundle) / "frozen_gnn" / name, gnn / name)
        else:
            _call(python, ["gnn_change.py"], work, log)          # training + held-out prediction, unchanged
        _call(python, ["-c", "import thesis_figures as T; T.fig5()"], work, log)
        for tile in tiles:
            _call(python, [str(Path("validation") / "gnn_predictions_tile_viz.py"), tile], work, log)
    figures = output / "figures"
    figures.mkdir()
    for name in STRIP_FIGURES:
        shutil.copy2(work / "thesis_figures" / name, figures / name)
    for tile in tiles:
        shutil.copy2(work / f"gnn_viz_{tile}" / "1_predictions_overlay.png", figures / f"{tile}_predictions_overlay.png")
    summary = json.loads((gnn / "summary.json").read_text(encoding="utf-8"))
    report = {"method": "legacy building spatio-temporal GNN (frozen 2026-10-04)", "status": "complete",
              "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
              "mode": "reused frozen predictions (figures only)" if reuse_frozen_predictions else "retrained (leave-one-tile-out)",
              "tiles_rendered": tiles, "evidence": EVIDENCE, "python": python, "environment": environment(python),
              "reference_environment": _read("output_reference.json")["environment"],
              "metrics": {m: {k: v for k, v in s.items() if k != "seconds"} for m, s in summary["models"].items()},
              "audit_agreement": summary.get("audit_agreement"),
              "labelled_nodes": summary["labelled"], "positives": summary["positives"]}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["verification"] = verify(output)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _summary_core(path: Path) -> dict:
    s = json.loads(Path(path).read_text(encoding="utf-8"))
    for model in s.get("models", {}).values():
        model.pop("seconds", None)
    return s


def verify(output: Path) -> dict:
    """Compare a run folder with the frozen originals (data bytes, summary without timings, decoded figure pixels)."""
    output = Path(output)
    work = output / "work"
    ref = _read("output_reference.json")
    gnn = work / "graphs" / STRIP_NAME / "gnn"
    checks = {}
    for name, digest in ref["gnn_data"].items():
        p = gnn / name
        checks[f"gnn/{name}"] = "identical" if p.is_file() and sha256(p) == digest else ("missing" if not p.is_file() else "DIFFERENT")
    p = gnn / "summary.json"
    checks["gnn/summary.json (excluding timings)"] = ("missing" if not p.is_file() else
                                                      "identical" if _summary_core(p) == ref["summary_without_timings"] else "DIFFERENT")
    for group, base in (("strip_figures", work / "thesis_figures"),):
        for name, digest in ref[group].items():
            p = base / name
            checks[f"thesis_figures/{name}"] = ("missing" if not p.is_file() else "pixel-identical" if pixel_sha256(p) == digest else "DIFFERENT")
    for tile, files in ref["tile_figures"].items():
        base = work / f"gnn_viz_{tile}"
        if not base.is_dir():
            continue
        for name, digest in files["files"].items():
            p = base / name
            same = p.is_file() and (pixel_sha256(p) if name.endswith(".png") else sha256(p)) == digest
            checks[f"gnn_viz_{tile}/{name} [{files['reference']}]"] = ("pixel-identical" if name.endswith(".png") else "identical") if same else ("missing" if not p.is_file() else "DIFFERENT")
    failed = sorted(k for k, v in checks.items() if v not in ("identical", "pixel-identical"))
    return {"status": "identical_to_frozen" if not failed else "differs", "checked": len(checks), "failed": failed, "checks": checks}
