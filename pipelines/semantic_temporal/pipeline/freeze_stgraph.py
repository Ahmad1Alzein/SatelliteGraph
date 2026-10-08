"""Freeze the spatio-temporal graph (graphs/stgraph_2013_2023_<tile>) as a read-only release.

  S1 build validation status PASS (stgraph_validation.json)
  S2 reproducible: rebuilding gives identical nodes.csv, edges.csv and tensor arrays
  S3 inputs still match the frozen temporal manifest (spatial graphs, segmentations, temporal layer)
  S4 tensors consistent with tables (node/edge counts, relation ids, joint normalisation stats)
Then: copy to graphs/FROZEN_stgraph_2013_2023_<tile>_v1 + code snapshot + MANIFEST.json, read-only.
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
import torch

import build_stgraph as bs

HERE = Path(__file__).resolve().parent
TILE = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--tile=")), "tile_00000")
SRC = bs.GRAPHS / f"stgraph_2013_2023_{TILE}"
DEST = bs.GRAPHS / f"FROZEN_stgraph_2013_2023_{TILE}_v1"
CODE = ["build_stgraph.py", "freeze_stgraph.py"]
DETERMINISTIC = ["nodes.csv", "edges.csv", "feature_spec.json"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def arrays(p):
    z = np.load(p)
    return {k: z[k] for k in z.files}


def main():
    if DEST.exists():
        raise SystemExit(f"{DEST} exists: frozen releases are never overwritten")
    R = []
    chk = lambda i, n, ok, d: R.append({"id": i, "check": n, "result": "PASS" if ok else "FAIL", "detail": d})

    v = json.loads((SRC / "stgraph_validation.json").read_text())
    chk("S1", "Build validation", v["status"] == "PASS", f"{sum(v['checks'].values())}/{len(v['checks'])} build checks pass")

    before = {f: sha(SRC / f) for f in DETERMINISTIC}
    arr_before = arrays(SRC / "stgraph_homo.npz")
    subprocess.run([sys.executable, str(HERE / "build_stgraph.py"), "--tile", TILE], check=True, capture_output=True)
    after = {f: sha(SRC / f) for f in DETERMINISTIC}
    arr_after = arrays(SRC / "stgraph_homo.npz")
    same_arr = all(np.array_equal(arr_before[k], arr_after[k]) for k in arr_before)
    chk("S2", "Reproducible", before == after and same_arr,
        f"rebuilt: {', '.join(DETERMINISTIC)} byte-identical={before == after}; tensor arrays identical={same_arr}")

    T, n = bs.verify(TILE)
    chk("S3", "Inputs match frozen manifests", True, f"{T.name} files + {n} recorded inputs (spatial graphs, segmentations) verified")

    nodes = pd.read_csv(SRC / "nodes.csv"); edges = pd.read_csv(SRC / "edges.csv")
    homo = torch.load(SRC / "stgraph_homo.pt", weights_only=False)
    het = torch.load(SRC / "stgraph_hetero.pt", weights_only=False)
    rel_ok = [homo["relations"][i] for i in homo["edge_type"].tolist()] == edges.relation.tolist()
    het_edges = sum(r["edge_index"].shape[1] for r in het["relations"].values())
    ok4 = (homo["x"].shape[0] == len(nodes) and homo["edge_index"].shape[1] == len(edges) and rel_ok
           and het_edges == len(edges) and het["x_2013"].shape[0] + het["x_2023"].shape[0] == len(nodes))
    chk("S4", "Tensors consistent with tables", ok4,
        f"homo x {tuple(homo['x'].shape)}, edges {homo['edge_index'].shape[1]}, relation ids match table={rel_ok}; hetero edges {het_edges}")

    status = "PASS" if all(r["result"] == "PASS" for r in R) else "FAIL"
    if status != "PASS":
        print(json.dumps(R, indent=1)); raise SystemExit("not freezing")

    v = json.loads((SRC / "stgraph_validation.json").read_text())
    shutil.copytree(SRC, DEST)
    (DEST / "code").mkdir()
    for f in CODE:
        shutil.copy2(HERE / f, DEST / "code" / f)
    rep = {"release": DEST.name, "status": status, "freeze_checks": R, "build_checks": v["checks"],
           "nodes": v["nodes"], "edges": v["edges"], "validated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    (DEST / "validation_report.json").write_text(json.dumps(rep, indent=2, default=int), encoding="utf-8")
    md = [f"# Validation: {DEST.name}", "", f"**Status: {status}**", "", "| Check | Result | Detail |", "|---|---|---|"]
    md += [f"| {r['id']} {r['check']} | {r['result']} | {r['detail']} |" for r in R]
    md += [f"| B {k} | {'PASS' if ok else 'FAIL'} | build check |" for k, ok in v["checks"].items()]
    (DEST / "validation_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    tman = json.loads((T / "MANIFEST.json").read_text())
    man = {"release": DEST.name, "created": rep["validated_at"],
           "inputs": {**{f"{T.name}/MANIFEST.json": sha(T / "MANIFEST.json")}, **tman["inputs"]},
           "files": {str(p.relative_to(DEST)).replace("\\", "/"): sha(p) for p in sorted(DEST.rglob("*")) if p.is_file()}}
    (DEST / "MANIFEST.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    for p in DEST.rglob("*"):
        if p.is_file():
            p.chmod(stat.S_IREAD)
    for r in R:
        print(f"{r['id']} {r['result']} {r['check']:34s} {r['detail']}")
    print(json.dumps({"status": status, "release": DEST.name, "files": len(man["files"])}))


if __name__ == "__main__":
    main()
