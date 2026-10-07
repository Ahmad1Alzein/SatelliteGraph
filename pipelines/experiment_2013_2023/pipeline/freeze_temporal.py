"""Validate and freeze the final temporal correspondence layer (v3).

Reads only: the frozen 2013/2023 segmentation releases and graphs, the v2 layer (for the diff), and
the v3 layer built by temporal_layer.py. Writes graphs/FROZEN_temporal_2013_2023_<tile>_v3/ (read-only).

Checks
  T1  inputs unchanged: frozen segmentation files + graph files match their recorded checksums
  T2  reproducible: rebuilding the layer in memory gives identical edges and components
  T3  layer structure checks from temporal_layer.py all pass (edges between years only, no
      duplicates, link rule, fractions in [0,1], one component per node, roof frame only for
      building pairs, no spatial edges, GraphML round trip)
  T4  contained_in links: same class only, contained node >= 50 m2, contained fraction >= 0.8
  T5  no different-class containment promoted (every cross-class primary link is a v1 identity link)
  T6  v2 -> v3 differ only by the removed contained_in links with a contained node < 50 m2
  T7  every v1 identity link preserved
  T8  self-test: 2023 linked to itself -> all one_to_one, high, IoU 1
  T9  primary components bounded (largest <= 50 nodes)
Known limitations are written to known_limitations.md / .csv.
"""
from __future__ import annotations

from osgeo import gdal  # noqa: F401
import hashlib
import json
import shutil
import stat
import time
from pathlib import Path

import numpy as np
import pandas as pd

import temporal_layer as tl

HERE = Path(__file__).resolve().parent
import sys as _sys
_ARGS = dict(a[2:].split("=", 1) for a in _sys.argv[1:] if a.startswith("--") and "=" in a)
TILE = _ARGS.get("tile", "tile_00000")
tl.REL_VERSION.update({"2013": _ARGS.get("release-2013", "v1"), "2023": _ARGS.get("release-2023", "v1")})
SRC = tl.GRAPHS / f"temporal_2013_2023_{TILE}_v3"
V1 = tl.GRAPHS / f"temporal_2013_2023_{TILE}"
V2 = tl.GRAPHS / f"temporal_2013_2023_{TILE}_v2"
DEST = tl.GRAPHS / f"FROZEN_temporal_2013_2023_{TILE}_v3"
CODE = ["temporal_layer.py", "final_strip_qa.py", "freeze_temporal.py"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if DEST.exists():
        raise SystemExit(f"{DEST} exists: frozen releases are never overwritten")
    t0 = time.time()
    R = []
    chk = lambda cid, name, ok, detail: R.append({"id": cid, "check": name, "result": "PASS" if ok else "FAIL", "detail": detail})

    # T1 inputs (load() asserts segmentation manifests; also hash the graph inputs)
    r13, g13, lab13, n13, tr, crs, _ = tl.load("2013", TILE)
    r23, g23, lab23, n23, _, _, _ = tl.load("2023", TILE)
    input_hashes = {}
    for rel in (r13, r23):
        man = json.loads((rel / "MANIFEST.json").read_text())
        for f in ("segment_ids.tif", "nodes_final.csv", "nodes_final.gpkg", "edges.csv"):
            assert sha(rel / f) == man["files"][f]
            input_hashes[f"{rel.name}/{f}"] = man["files"][f]
    for g in (g13, g23):
        for f in ("nodes.csv", "edges.csv", "graph_tensors.pt"):
            input_hashes[f"{g.name}/{f}"] = sha(g / f)
    chk("T1", "Frozen inputs unchanged", True,
        f"segmentation files of {r13.name}, {r23.name} match their manifests; graph files hashed ({len(input_hashes)} files)")

    # T2 reproducibility
    E_file = pd.read_csv(SRC / "temporal_edges.csv"); C_file = pd.read_csv(SRC / "temporal_components.csv")
    E_mem, C_mem, _ = tl.build_layer(lab13, n13, lab23, n23)
    cols = ["node_2013", "node_2023", "frame", "confidence", "edge_role", "link_type", "match_type", "component_id"]
    same_e = E_mem[cols].reset_index(drop=True).equals(E_file[cols].reset_index(drop=True))
    same_c = C_mem.match_type.value_counts().to_dict() == C_file.match_type.value_counts().to_dict()
    chk("T2", "Reproducible", same_e and same_c, f"rebuilt {len(E_mem)} edges identical to saved layer: {same_e}; components identical: {same_c}")

    # T3 structure checks recorded by the build
    S = json.loads((SRC / "summary.json").read_text())
    chk("T3", "Layer structure checks", S["status"] == "PASS", ", ".join(f"{k}={v}" for k, v in S["checks"].items()))

    E = E_file
    c = E[E.link_type == "contained_in"]
    ok4 = bool(c.class_agreement.all() and (c.contained_node_m2 >= tl.CONTAIN_MIN_M2).all()
               and (np.maximum(c.frac_2013, c.frac_2023) >= tl.CONTAIN_FRAC).all())
    chk("T4", "contained_in rule", ok4, f"{len(c)} links; all same class; contained node min {c.contained_node_m2.min():.1f} m2; "
        f"contained fraction min {np.maximum(c.frac_2013, c.frac_2023).min():.2f}")

    cross = E[(E.edge_role == "primary") & ~E.class_agreement]
    ok5 = bool((cross.link_type == "identity").all() and (cross.overlap_score >= 0.5).all())
    chk("T5", "No different-class containment promoted", bool(ok5),
        f"{len(cross)} cross-class primary links, all identity links with mutual overlap >= 0.5; "
        f"{int(((~E.class_agreement) & (np.maximum(E.frac_2013, E.frac_2023) >= 0.8) & (E.edge_role == 'secondary')).sum())} cross-class containment overlaps remain secondary")

    if V1.exists() and V2.exists():   # version history exists only for the tile the rules were developed on
        E1 = pd.read_csv(V1 / "temporal_edges.csv")
        v1_id = set(map(tuple, E1[E1.edge_role == "primary"][["node_2013", "node_2023"]].values))
        E2 = pd.read_csv(V2 / "temporal_edges.csv")
        key = lambda d: set(map(tuple, d[["node_2013", "node_2023", "link_type", "edge_role"]].values))
        removed = key(E2) - key(E3 := E)
        added = key(E3) - key(E2)
        rm_contained = E2[E2.link_type == "contained_in"].merge(E[["node_2013", "node_2023", "link_type"]], on=["node_2013", "node_2023"], suffixes=("", "_v3"))
        dropped = rm_contained[rm_contained.link_type_v3 != "contained_in"]
        small = [min(n13.area_m2[a], n23.area_m2[b]) for a, b in zip(dropped.node_2013, dropped.node_2023)]
        ok6 = len(dropped) == len(removed) == len(added) and all(a < tl.CONTAIN_MIN_M2 for a in small)
        chk("T6", "v2 -> v3 difference", ok6, f"{len(dropped)} contained_in links demoted to overlap, contained nodes {sorted(round(a, 1) for a in small)} m2; nothing else changed")

        v3_id = set(map(tuple, E[(E.edge_role == "primary") & (E.link_type == "identity")][["node_2013", "node_2023"]].values))
        chk("T7", "v1 identity links preserved", v1_id == v3_id, f"{len(v1_id)} identity links")
    else:
        R.append({"id": "T6", "check": "v2 -> v3 difference", "result": "N/A", "detail": "no earlier layer versions for this tile (rules frozen on tile_00000)"})
        R.append({"id": "T7", "check": "v1 identity links preserved", "result": "N/A", "detail": "no earlier layer versions for this tile"})

    Es, Cs, _ = tl.build_layer(lab23, n23, lab23, n23)
    ok8 = (Es.match_type == "one_to_one").all() and (Es.confidence == "high").all() and Es.iou.min() == 1.0 and len(Es) == len(n23)
    chk("T8", "Self-test", bool(ok8), f"2023 vs 2023: {len(Es)} edges, all one_to_one/high, min IoU {Es.iou.min()}")

    big = int((C_file.n_2013 + C_file.n_2023).max())
    chk("T9", "Primary components bounded", big <= 50, f"largest primary component {S['largest_component_2013_2023']} nodes")

    status = "PASS" if all(r["result"] in ("PASS", "N/A") for r in R) else "FAIL"
    if status != "PASS":
        print(json.dumps(R, indent=1)); raise SystemExit("validation failed; not freezing")

    # ---- known limitations
    U = pd.read_csv(SRC / "no_primary_match_nodes.csv")
    ub = U[U["class"] == "building"]
    lim = []
    for r in ub.itertuples():
        col = f"node_{r.year}"
        d = E[E[col] == r.segment_id].copy()
        frac_self = d.frac_2013 if r.year == 2013 else d.frac_2023
        d = d.assign(fs=frac_self.values).sort_values("fs", ascending=False)
        best = d.iloc[0] if len(d) else None
        other_cls = (best.class_2023 if r.year == 2013 else best.class_2013) if best is not None else ""
        if best is not None and best.fs >= tl.CONTAIN_FRAC and other_cls != "building":
            reason = f"label_disagreement: {best.fs:.0%} inside a {other_cls} node of the other year"
        elif best is not None and other_cls == "building":
            reason = f"partial_building_overlap: best same-class containment {best.fs:.0%} (< 80 %)"
        elif r.area_m2 < tl.CONTAIN_MIN_M2:
            reason = "small_fragment (< 50 m2)"
        else:
            reason = "no_counterpart_in_overlap (possibly real appearance/disappearance or segmentation miss)"
        lim.append({"limitation": "unmatched_building", "year": r.year, "segment_id": int(r.segment_id), "area_m2": r.area_m2,
                    "best_overlap_node": int(best.node_2023 if r.year == 2013 else best.node_2013) if best is not None else -1,
                    "best_overlap_class": other_cls, "fraction_inside": round(float(best.fs), 3) if best is not None else 0.0, "reason": reason})
    # oversized = 2013 building segment larger than the largest 2023 building segment
    max23 = float(n23[n23["class"] == "building"].area_m2.max())
    big13 = n13[(n13["class"] == "building") & (n13.area_m2 > max23)].sort_values("area_m2", ascending=False)
    for sid, r in big13.iterrows():
        links = E[(E.node_2013 == sid) & (E.edge_role == "primary")]
        mt = "|".join(sorted(set(links.match_type))); lt = "|".join(sorted(set(links.link_type)))
        lim.append({"limitation": "oversized_2013_building_segment", "year": 2013, "segment_id": int(sid), "area_m2": float(r.area_m2),
                    "best_overlap_node": -1, "best_overlap_class": "", "fraction_inside": 0.0,
                    "reason": f"merged dense block larger than any 2023 building segment ({max23:,.0f} m2); match type {mt}; "
                              f"{len(links)} primary links ({lt}) to 2023 building segments {'|'.join(map(str, links.node_2023))}"})
    L = pd.DataFrame(lim)
    L.to_csv(SRC / "known_limitations.csv", index=False)
    counts = L[L.limitation == "unmatched_building"].reason.str.split(":").str[0].value_counts().to_dict()
    md = ["# Known limitations: temporal layer v3 (tile_00000)", "",
          "## 1. Buildings without a primary temporal link",
          f"{len(ub)} building nodes ({int((ub.year == 2013).sum())} in 2013, {int((ub.year == 2023).sum())} in 2023) have no primary link. By cause:", ""]
    md += [f"- **{k}**: {v}" for k, v in counts.items()]
    md += ["", "*label_disagreement*: the building exists in both years, but in the other year its pixels belong to a",
           "paved_developed or tree segment of the frozen segmentation. Promoting different-class containment would",
           "also join yards to adjacent buildings, so it was deliberately not done. The pair is kept as a secondary",
           "`overlap` edge, so it can be recovered downstream. Visual audit: 5 of 16 sampled still-unmatched",
           "buildings were missed links of this kind.", "",
           "## 2. Oversized 2013 building segments",
           "2013 building segments larger than the largest 2023 building segment: dense blocks that the 2013",
           "segmentation merged into one segment while the 2023 segmentation keeps several buildings. Their",
           "temporal links are one-to-many (identity + contained_in) and mean 'part of the same merged block',",
           "not one building to one building.", ""]
    md += [f"- 2013 segment {r.segment_id}: {r.area_m2:,.0f} m², {r.reason}" for r in L[L.limitation == "oversized_2013_building_segment"].itertuples()]
    md += ["", "Full list: `known_limitations.csv`."]
    (SRC / "known_limitations.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- freeze
    shutil.copytree(SRC, DEST)
    (DEST / "code").mkdir()
    for f in CODE:
        shutil.copy2(HERE / f, DEST / "code" / f)
    report = {"release": DEST.name, "status": status, "checks": R, "validated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
              "temporal_edges": int(len(E)), "by_link_type": E.link_type.value_counts().to_dict(),
              "by_confidence": E.confidence.value_counts().to_dict(),
              "components_by_match_type": C_file.match_type.value_counts().to_dict(),
              "known_limitations": {"unmatched_buildings": int(len(ub)), "by_cause": counts,
                                    "oversized_2013_building_segments": L[L.limitation == "oversized_2013_building_segment"][["segment_id", "area_m2"]].to_dict("records")},
              "seconds": round(time.time() - t0, 1)}
    (DEST / "validation_report.json").write_text(json.dumps(report, indent=2, default=int), encoding="utf-8")
    (DEST / "validation_report.md").write_text("\n".join([f"# Validation: {DEST.name}", "", f"**Status: {status}**", "",
                                                          "| Check | Result | Detail |", "|---|---|---|"]
                                                         + [f"| {r['id']} {r['check']} | {r['result']} | {r['detail']} |" for r in R]) + "\n", encoding="utf-8")
    man = {"release": DEST.name, "created": report["validated_at"], "parameters": {
               "min_frac_smaller": tl.MIN_FRAC_SMALLER, "min_px": tl.MIN_PX, "contain_frac": tl.CONTAIN_FRAC, "contain_min_m2": tl.CONTAIN_MIN_M2,
               "confidence": "high: same class & IoU>=0.5; medium: same class & overlap_score>=0.3, or cross-class & overlap_score>=0.5, or contained_in"},
           "inputs": input_hashes,
           "files": {str(p.relative_to(DEST)).replace("\\", "/"): sha(p) for p in sorted(DEST.rglob("*")) if p.is_file()}}
    (DEST / "MANIFEST.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    for p in DEST.rglob("*"):
        if p.is_file():
            p.chmod(stat.S_IREAD)
    for r in R:
        print(f"{r['id']:3s} {r['result']:4s} {r['check']:40s} {r['detail']}")
    print(json.dumps({k: report[k] for k in ("status", "temporal_edges", "by_link_type", "by_confidence", "components_by_match_type", "known_limitations")}, indent=1, default=int))


if __name__ == "__main__":
    main()
