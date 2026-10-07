"""Compare a run's outputs with the reference results of the original 2013-2023 experiment.

  python verify_outputs.py <work folder>

Two levels:
  files        SHA-256 of every data output (.csv .tif .graphml .npz .npy .pt) vs reference/expected_outputs.json.
               GeoPackages, JSON/Markdown reports and figures are excluded: they carry timestamps, run times or
               fonts, not results.
  key results  segment / edge counts, validation statuses and GNN metrics vs reference/expected_results.json.
Writes <work>/REPRODUCTION_REPORT.md and reproduction_report.json.
Exit code 0 = every data file byte-identical; 2 = key results identical but some files differ; 1 = results differ.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
SUFFIXES = {".csv", ".tif", ".graphml", ".npz", ".npy", ".pt"}
DIRS = ("inputs", "final_segmentation", "graphs", "gnn_viz_tile_00002")
TILES = [f"tile_{i:05d}" for i in range(7)]
STRIP = "graphs/FROZEN_strip_tile_00000-00006_v1"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def collect(work: Path) -> dict:
    out = {}
    for d in DIRS:
        for p in sorted((work / d).rglob("*")) if (work / d).exists() else []:
            rel = p.relative_to(work)
            if p.is_file() and p.suffix in SUFFIXES and not ({"code", "_scratch_strip_rebuild"} & set(rel.parts)):
                out[rel.as_posix()] = sha(p)
    return out


def key_results(work: Path) -> dict:
    j = lambda p: json.loads((work / p).read_text(encoding="utf-8"))
    r = {"tiles": {}}
    for t in TILES:
        prov = j(f"final_segmentation/inputs/2013_{t}_provenance.json")
        v23 = "v2" if prov.get("harmonized_2023_reference") else "v1"
        tr = {"coregistration": {k: prov["coregistration"]["checks"][k] for k in ("translation_px", "tile_residual_px", "disagreement_with_aoi_px")},
              "release_2023": v23}
        for y, rel in (("2013", f"FROZEN_2013_{t}_v1"), ("2023", f"FROZEN_2023_{t}_{v23}")):
            rep = j(f"final_segmentation/{rel}/validation_report.json")
            tr[f"segments_{y}"] = sum(1 for _ in open(work / "final_segmentation" / rel / "nodes_final.csv", encoding="utf-8")) - 1
            tr[f"checks_{y}"] = f"{sum(c['result'] == 'PASS' for c in rep['checks'])}/{len(rep['checks'])}"
        tr["temporal_edges"] = sum(1 for _ in open(work / f"graphs/FROZEN_temporal_2013_2023_{t}_v3/temporal_edges.csv", encoding="utf-8")) - 1
        tr["stgraph_status"] = j(f"graphs/FROZEN_stgraph_2013_2023_{t}_v1/validation_report.json")["status"]
        r["tiles"][t] = tr
    v = j(f"{STRIP}/stgraph_validation.json")
    r["strip"] = {"stgraph_nodes": v["stgraph_nodes"], "stgraph_edges": v["stgraph_edges"], "by_relation": v["by_relation"], "status": v["status"],
                  "segments_2013": j(f"{STRIP}/2013/summary.json")["segments"], "segments_2023": j(f"{STRIP}/2023/summary.json")["segments"]}
    g = j(f"{STRIP}/gnn/summary.json")
    keys = ("ap", "roc_auc", "precision", "recall", "f1", "tp", "fp", "fn")
    r["gnn"] = {"labelled": g["labelled"], "positives": g["positives"],
                "models": {m: {k: g["models"][m][k] for k in keys} for m in g["models"]},
                "audit_agreement": g["audit_agreement"]}
    fr = j(f"{STRIP}/validation_report.json")
    r["strip_freeze"] = {"status": fr["status"], "checks": {c["id"]: c["result"] for c in fr["checks"]}}
    return r


def flat(d, pre=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from flat(v, f"{pre}{k}.")
        else:
            yield f"{pre}{k}", v


def main(work: Path) -> int:
    exp_files = json.loads((PKG / "reference" / "expected_outputs.json").read_text())
    exp_res = json.loads((PKG / "reference" / "expected_results.json").read_text())
    got_files = collect(work)
    missing = sorted(set(exp_files) - set(got_files)); extra = sorted(set(got_files) - set(exp_files))
    differ = sorted(f for f in exp_files if f in got_files and got_files[f] != exp_files[f])
    same = len(exp_files) - len(missing) - len(differ)
    try:
        got_res, err = key_results(work), None
    except Exception as e:  # incomplete run
        got_res, err = {}, f"{type(e).__name__}: {e}"
    E, G = dict(flat(exp_res)), dict(flat(got_res))
    res_diff = [(k, E[k], G.get(k, "missing")) for k in E if G.get(k, "missing") != E[k]]
    files_ok, res_ok = not missing and not differ, not res_diff and err is None
    status = "EXACT" if files_ok and res_ok else ("RESULTS IDENTICAL, SOME FILES DIFFER" if res_ok else "DIFFERENT")
    code = 0 if status == "EXACT" else (2 if res_ok else 1)
    rep = {"status": status, "files_expected": len(exp_files), "files_identical": same, "missing": missing, "differing": differ,
           "extra_not_in_reference": extra, "key_results_checked": len(E), "key_results_differing": res_diff, "error": err}
    (work / "reproduction_report.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    g = got_res.get("gnn", {}).get("models", {}).get("gnn_full", {})
    lines = [f"# Reproduction report", "", f"**Status: {status}**", "",
             f"- data files byte-identical to the reference: **{same} / {len(exp_files)}** (missing {len(missing)}, differing {len(differ)})",
             f"- key results identical: **{len(E) - len(res_diff)} / {len(E)}**" + (f" (could not read results: {err})" if err else ""), ""]
    if g:
        lines += ["| | reference | this run |", "|---|---|---|"]
        for k in ("ap", "roc_auc", "precision", "recall", "f1"):
            lines.append(f"| spatio-temporal GNN {k} | {E.get(f'gnn.models.gnn_full.{k}')} | {g.get(k)} |")
        lines += [f"| strip st-graph nodes / edges | {E['strip.stgraph_nodes']} / {E['strip.stgraph_edges']} | "
                  f"{G.get('strip.stgraph_nodes')} / {G.get('strip.stgraph_edges')} |",
                  f"| strip freeze | {E['strip_freeze.status']} | {G.get('strip_freeze.status')} |", ""]
    if res_diff:
        lines += ["## Key results that differ", "", "| result | reference | this run |", "|---|---|---|"] + [f"| {k} | {a} | {b} |" for k, a, b in res_diff[:200]] + [""]
    if missing or differ:
        lines += ["## Files", ""] + [f"- missing: `{f}`" for f in missing[:100]] + [f"- differs: `{f}`" for f in differ[:200]] + [""]
    (work / "REPRODUCTION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return code


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve()))
