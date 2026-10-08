"""Run the full frozen pipeline for one tile, 2013 vs 2023 (same stages and rules as tile_00002).

  1 segment 2023 + freeze FROZEN_2023_<tile>_v1 (single-year release)
  2 co-register + harmonize 2013 against it; if harmonization blurs 2023, write the harmonized 2023 image,
    segment it and freeze the paired release FROZEN_2023_<tile>_v2
  3 segment 2013 (co-registered, harmonized) + freeze FROZEN_2013_<tile>_v1
  4 spatial graphs   5 temporal layer v3 + freeze   6 spatio-temporal graph + freeze
  7 change detection on the graph   8 audit montages
Stage groups (--stages, default all three, in this order): segmentation = 1-3, graph = 4-6, classical = 7-8.
A later group reads the frozen outputs of the earlier ones, so groups can run in separate calls.
Stops at the first failing stage.
Usage: python run_tile_pipeline.py [--stages=segmentation,graph,classical] tile_00001 [tile_00003 ...]
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
INPUTS = HERE / "final_segmentation" / "inputs"
GROUPS = ("segmentation", "graph", "classical")


def step(log, name, args):
    t = time.time()
    r = subprocess.run([PY, *args], cwd=HERE, capture_output=True, text=True)
    log.write(f"\n===== {name}: {' '.join(args)}\n{r.stdout}\n{r.stderr[-4000:]}\n")
    log.flush()
    ok = r.returncode == 0
    print(f"  {'OK ' if ok else 'FAIL'} {name} ({time.time() - t:.0f}s)", flush=True)
    if not ok:
        tail = [l for l in (r.stdout + r.stderr).splitlines() if l.strip() and "Warning" not in l][-6:]
        raise SystemExit(f"stage '{name}' failed:\n" + "\n".join(tail))
    return r.stdout


def run_tile(tile, stages=GROUPS):
    print(f"=== {tile}", flush=True)
    mode = "w" if "segmentation" in stages else "a"     # a later group appends to the tile's log
    with open(HERE / f"pipeline_{tile}.log", mode, encoding="utf-8") as log:
        if "segmentation" in stages:
            step(log, "segment 2023", ["segment_final.py", "--tile", tile, "--year", "2023", "--image", str(HERE / "inputs" / "2023" / f"{tile}.tif")])
            step(log, "freeze 2023 v1", ["freeze_and_validate.py", "--tile", tile, "--year", "2023", "--version", "v1"])
            step(log, "prepare 2013 pair", ["prepare_pair_tile.py", "--tile", tile, "--reference-release", f"FROZEN_2023_{tile}_v1",
                                            "--write-harmonized-reference"])
        prov = json.loads((INPUTS / f"2013_{tile}_provenance.json").read_text())
        ver23 = "v2" if prov.get("harmonized_2023_reference") else "v1"
        if "segmentation" in stages:
            if ver23 == "v2":
                step(log, "segment 2023 harmonized", ["segment_final.py", "--tile", tile, "--year", "2023", "--image", prov["harmonized_2023_reference"]])
                step(log, "freeze 2023 v2 (paired)", ["freeze_and_validate.py", "--tile", tile, "--year", "2023", "--version", "v2"])
            step(log, "segment 2013", ["segment_final.py", "--tile", tile, "--year", "2013", "--image", prov["output"]])
            step(log, "freeze 2013 v1", ["freeze_and_validate.py", "--tile", tile, "--year", "2013", "--version", "v1"])
        if "graph" in stages:
            for rel in (f"FROZEN_2013_{tile}_v1", f"FROZEN_2023_{tile}_{ver23}"):
                out = step(log, f"graph {rel}", ["build_graph.py", "--release", rel])
                if '"status": "PASS"' not in out:
                    raise SystemExit(f"graph build for {rel} did not pass")
            step(log, "temporal self-test", ["temporal_layer.py", "--tile", tile, "--release-2023", ver23, "--self-test"])
            step(log, "temporal layer", ["temporal_layer.py", "--tile", tile, "--release-2023", ver23])
            step(log, "freeze temporal", ["freeze_temporal.py", f"--tile={tile}", f"--release-2023={ver23}"])
            step(log, "st-graph", ["build_stgraph.py", "--tile", tile])
            step(log, "freeze st-graph", ["freeze_stgraph.py", f"--tile={tile}"])
        if "classical" in stages:
            step(log, "change detection", ["stgraph_change.py", "--tile", tile])
            step(log, "audit montages", ["validation/stgraph_change_audit_tile.py", tile])
    return {"tile": tile, "stages": list(stages), "paired_2023_release": ver23,
            "coregistration": prov["coregistration"]["checks"], "harmonization": prov["harmonization"]}


if __name__ == "__main__":
    args = sys.argv[1:]
    opt = [a for a in args if a.startswith("--stages=")]
    stages = tuple(opt[0].split("=", 1)[1].split(",")) if opt else GROUPS
    unknown = set(stages) - set(GROUPS)
    if unknown:
        raise SystemExit(f"unknown stage group(s) {sorted(unknown)}; choose from {GROUPS}")
    stages = tuple(g for g in GROUPS if g in stages)     # always in pipeline order
    res = [run_tile(t, stages) for t in args if not t.startswith("--")]
    print(json.dumps(res, indent=1))
