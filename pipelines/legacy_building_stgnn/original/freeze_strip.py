"""Freeze the strip mosaic, strip spatio-temporal graph and GNN results as one read-only release.

  F1 inputs: the 7 frozen tile spatio-temporal graph releases verify against their manifests, and the
     segmentation files they used verify against theirs
  F2 strip build checks (per-year mosaic checks + spatio-temporal graph checks) all pass
  F3 reproducible mosaic + st-graph: a rebuild into a scratch folder is identical (tables byte-identical,
     label rasters equal)
  F4 reproducible GNN: retraining from the frozen inputs gives the same held-out metrics (max |diff|) and
     predictions (GPU message passing is not bit-deterministic, so a tolerance is reported, not assumed)
  F5 labels: the per-tile change tables used as GNN labels are hashed and copied into the release
Release: graphs/FROZEN_strip_tile_00000-00006_v1/ (read-only) with code snapshot + MANIFEST.json.
"""
from __future__ import annotations

from osgeo import gdal  # noqa: F401
import hashlib
import json
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

HERE = Path(__file__).resolve().parent
TILES = [f"tile_{i:05d}" for i in range(7)]
SRC = HERE / "graphs" / "strip_tile_00000-00006"
DEST = HERE / "graphs" / "FROZEN_strip_tile_00000-00006_v1"
SCRATCH = Path(r"C:\Users\USER\AppData\Local\Temp\claude\D--new-DelAnyV2-Experiment\d7494492-e4e6-43c3-9d3b-2d3ffaa38266\scratchpad\strip_rebuild")
CODE = ["build_mosaic.py", "gnn_change.py", "thesis_figures.py", "freeze_strip.py", "build_graph.py", "segment_final.py",
        "segment_v2.py", "temporal_layer.py", "final_strip_qa.py", "refine_merge.py", "semantic_ms.py"]
TABLES = ["2013/nodes.csv", "2013/edges.csv", "2013/seam_decisions.csv", "2023/nodes.csv", "2023/edges.csv", "2023/seam_decisions.csv",
          "temporal_edges.csv", "temporal_components.csv", "stgraph_nodes.csv", "stgraph_edges.csv", "stgraph_node_lineage.csv"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if DEST.exists():
        raise SystemExit(f"{DEST} exists: frozen releases are never overwritten")
    R = []
    chk = lambda i, n, ok, d: R.append({"id": i, "check": n, "result": "PASS" if ok else "FAIL", "detail": d})
    t0 = time.time()

    # F1 inputs
    inputs = {}
    for t in TILES:
        g = HERE / "graphs" / f"FROZEN_stgraph_2013_2023_{t}_v1"
        man = json.loads((g / "MANIFEST.json").read_text())
        for f, h in man["files"].items():
            assert sha(g / f) == h, f"{g.name}/{f} changed"
        inputs[f"{g.name}/MANIFEST.json"] = sha(g / "MANIFEST.json")
        for k in man["inputs"]:
            folder, f = k.split("/", 1)
            if folder.startswith("FROZEN_20"):
                rel = HERE / "final_segmentation" / folder
                rman = json.loads((rel / "MANIFEST.json").read_text())
                assert sha(rel / f) == rman["files"][f], f"{folder}/{f} changed"
                inputs[k] = rman["files"][f]
    chk("F1", "Frozen inputs unchanged", True, f"7 tile spatio-temporal releases + {len(inputs) - 7} segmentation files verified")

    # F2 build checks
    v13 = json.loads((SRC / "2013" / "summary.json").read_text())["checks"]
    v23 = json.loads((SRC / "2023" / "summary.json").read_text())["checks"]
    vst = json.loads((SRC / "stgraph_validation.json").read_text())
    ok2 = all(v13.values()) and all(v23.values()) and vst["status"] == "PASS"
    chk("F2", "Strip build checks", ok2, f"2013 {sum(map(bool, v13.values()))}/{len(v13)}, 2023 {sum(map(bool, v23.values()))}/{len(v23)}, "
                                         f"st-graph {sum(vst['checks'].values())}/{len(vst['checks'])}")

    # F3 mosaic + st-graph reproducibility
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    r = subprocess.run([sys.executable, str(HERE / "build_mosaic.py"), f"--out={SCRATCH}"], cwd=HERE, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    diff = [f for f in TABLES if sha(SRC / f) != sha(SCRATCH / f)]
    rast = all(np.array_equal(rasterio.open(SRC / y / "segment_ids.tif").read(1), rasterio.open(SCRATCH / y / "segment_ids.tif").read(1)) for y in ("2013", "2023"))
    chk("F3", "Mosaic + st-graph reproducible", not diff and rast,
        f"rebuild: {len(TABLES) - len(diff)}/{len(TABLES)} tables byte-identical; segment rasters equal: {rast}" + (f"; differing: {diff}" if diff else ""))

    # F4 GNN reproducibility (rerun on the rebuilt strip)
    r = subprocess.run([sys.executable, str(HERE / "gnn_change.py"), f"--strip={SCRATCH}"], cwd=HERE, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    a = json.loads((SRC / "gnn" / "summary.json").read_text())["models"]; b = json.loads((SCRATCH / "gnn" / "summary.json").read_text())["models"]
    keys = ["ap", "roc_auc", "precision", "recall", "f1"]
    md = max(abs(a[m][k] - b[m][k]) for m in a for k in keys)
    pa, pb = pd.read_csv(SRC / "gnn" / "predictions.csv"), pd.read_csv(SCRATCH / "gnn" / "predictions.csv")
    pdiff = float(np.nanmax(np.abs(pa.prob_gnn_full.values - pb.prob_gnn_full.values)))
    flips = int((pa.pred_gnn_full != pb.pred_gnn_full)[pa.label >= 0].sum())
    chk("F4", "GNN reproducible", md <= 0.01,
        f"retrained 3 models x 7 folds: max metric difference {md:.4f}; spatio-temporal GNN max probability difference {pdiff:.4f}, "
        f"{flips} changed predictions on target nodes (GPU non-determinism tolerance 0.01)")

    # F5 labels snapshot
    labels = {}
    for t in TILES:
        f = HERE / "graphs" / f"change_stgraph_2013_2023_{t}" / "nodes_change.csv"
        labels[t] = sha(f)
    chk("F5", "GNN label inputs recorded", len(labels) == 7, "per-tile change tables hashed and copied to labels/")

    status = "PASS" if all(x["result"] == "PASS" for x in R) else "FAIL"
    if status != "PASS":
        print(json.dumps(R, indent=1)); raise SystemExit("not freezing")

    shutil.copytree(SRC, DEST)
    (DEST / "labels").mkdir()
    for t in TILES:
        shutil.copy2(HERE / "graphs" / f"change_stgraph_2013_2023_{t}" / "nodes_change.csv", DEST / "labels" / f"{t}_nodes_change.csv")
    shutil.copytree(HERE / "thesis_figures", DEST / "thesis_figures")
    (DEST / "code").mkdir()
    for f in CODE:
        shutil.copy2(HERE / f, DEST / "code" / f)
    rep = {"release": DEST.name, "status": status, "checks": R, "validated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1)}
    (DEST / "validation_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    (DEST / "validation_report.md").write_text("\n".join([f"# Validation: {DEST.name}", "", f"**Status: {status}**", "", "| Check | Result | Detail |", "|---|---|---|"]
                                                         + [f"| {x['id']} {x['check']} | {x['result']} | {x['detail']} |" for x in R]) + "\n", encoding="utf-8")
    man = {"release": DEST.name, "created": rep["validated_at"], "inputs": inputs, "label_inputs": labels,
           "files": {str(p.relative_to(DEST)).replace("\\", "/"): sha(p) for p in sorted(DEST.rglob("*")) if p.is_file()}}
    (DEST / "MANIFEST.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    for p in DEST.rglob("*"):
        if p.is_file():
            p.chmod(stat.S_IREAD)
    for x in R:
        print(f"{x['id']} {x['result']} {x['check']:32s} {x['detail']}")
    print(json.dumps({"status": status, "release": DEST.name, "files": len(man["files"])}))


if __name__ == "__main__":
    main()
