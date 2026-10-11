"""Multi-class diagnosis: oversized objects, merge artefacts and coarse (whole-object) change labels, all classes, 2013 and 2023.

Read-only on: the frozen study (segmentation_v2/final_segmentation, graphs/change_stgraph_*), OriginalPipeline_Improved
(mc_change v1 nodes, v2 regions) and this repository's multi-class runs (artifacts/shouf3/<tile>_geo).
Base partition: the frozen per-tile segmentations (0.75 m grid, 2013 co-registered to 2023).

 D1 object size per class and year, before vs after refine_merge
 D2 pixel class-transition matrix 2013 -> 2023 (m2)
 D3 how much of each class disagreement is boundary / roof-lean zone / vegetation-internal / built / water,
    and how much survives as compact regions >= 25 m2
 D4 partial change: per object, interior fraction whose class differs in the other year
 D5 change unit of existing labelers: flagged area vs area that actually differs
Outputs artifacts/experiments/multiclass_localised/diagnosis/
"""
from __future__ import annotations

from osgeo import gdal  # noqa: F401
import argparse
import ast
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage

TILES = [f"tile_{i:05d}" for i in range(7)]
CLS = {1: "building", 2: "road", 3: "tree", 4: "grass_shrub", 5: "cropland", 6: "bare_soil", 7: "paved_developed", 8: "water"}
VEG = {3, 4, 5, 6}              # vegetation / agriculture / bare: seasonally and semantically ambiguous between each other
BUILT = {1, 2, 7}
PX = 0.5625
BOUNDARY_PX = 2                 # 1.5 m: registration residual (<= 0.75 px at 1.5 m) + boundary placement
LEAN_PX = 8                     # 6 m: roof displacement measured 3-9 m in this study
MMU_M2 = 25.0
ROOT = Path(__file__).resolve().parents[2]
IMPROVED = Path(r"D:\new\OriginalPipeline_Improved")


def read(p):
    with rasterio.open(p) as s:
        return s.read(1)


def releases(study, tile):
    m = json.loads((study / "graphs" / f"FROZEN_stgraph_2013_2023_{tile}_v1" / "MANIFEST.json").read_text())
    return {k.split("/")[0].split("_")[1]: study / "final_segmentation" / k.split("/")[0] for k in m["inputs"] if k.endswith("segment_ids.tif")}


def boundary(lab, r):
    b = (np.diff(lab, axis=0, prepend=lab[:1]) != 0) | (np.diff(lab, axis=1, prepend=lab[:, :1]) != 0)
    return ndimage.binary_dilation(b, iterations=r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default=r"D:\new\DelAnyV2_Experiment\segmentation_v2")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "experiments" / "multiclass_localised" / "diagnosis"))
    a = ap.parse_args()
    study, out = Path(a.study), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sizes, objs, cat_rows, trans = [], [], [], np.zeros((9, 9))
    for t in TILES:
        rel = releases(study, t)
        lab = {y: read(rel[y] / "segment_ids.tif") for y in ("2013", "2023")}
        pre = {y: read(rel[y] / "segment_ids_before_merge.tif") for y in ("2013", "2023")}
        cls = {y: read(rel[y] / "segment_classes.tif") for y in ("2013", "2023")}
        valid = (cls["2013"] > 0) & (cls["2023"] > 0)
        valid[:20], valid[-20:], valid[:, :20], valid[:, -20:] = False, False, False, False   # tile margins
        # D1 sizes per class / year, before and after merge (class taken from the final map)
        for y in ("2013", "2023"):
            for stage, L in (("before_merge", pre[y]), ("after_merge", lab[y])):
                ids, inv = np.unique(L.ravel(), return_inverse=True)
                area = np.bincount(inv) * PX
                maj = np.bincount(inv * 9 + cls[y].ravel(), minlength=len(ids) * 9).reshape(len(ids), 9).argmax(1)   # majority class
                for c in CLS:
                    s = area[(maj == c) & (ids > 0)]
                    if len(s):
                        sizes.append({"tile": t, "year": y, "stage": stage, "class": CLS[c], "n": len(s), "median_m2": float(np.median(s)),
                                      "p90_m2": float(np.percentile(s, 90)), "max_m2": float(s.max()), "over_1ha": int((s > 1e4).sum()),
                                      "area_m2": float(s.sum()), "area_over_1ha_m2": float(s[s > 1e4].sum())})
        # D2 transition matrix
        c13, c23 = cls["2013"], cls["2023"]
        np.add.at(trans, (c13[valid], c23[valid]), PX)
        # D3 categories of disagreement
        diff = valid & (c13 != c23)
        bnd = boundary(lab["2013"], BOUNDARY_PX) | boundary(lab["2023"], BOUNDARY_PX)
        bld_edge = ndimage.binary_dilation((c13 == 1) | (c23 == 1), iterations=LEAN_PX)
        involves_b = (c13 == 1) | (c23 == 1)
        cat = np.full(c13.shape, "", object)
        cat[diff & bnd] = "boundary_band"
        rest = diff & ~bnd
        cat[rest & involves_b & bld_edge & ~((c13 == 1) & (c23 == 1))] = "building_lean_zone"
        rest &= cat == ""
        vv = np.isin(c13, list(VEG)) & np.isin(c23, list(VEG))
        cat[rest & vv] = "vegetation_internal"
        cat[rest & ~vv & (np.isin(c13, list(BUILT)) | np.isin(c23, list(BUILT)))] = "built_transition"
        cat[rest & (cat == "")] = "water_or_other"
        for k in ("boundary_band", "building_lean_zone", "vegetation_internal", "built_transition", "water_or_other"):
            m = cat == k
            comp, n = ndimage.label(m)
            big = np.bincount(comp.ravel())[1:] * PX >= MMU_M2 if n else np.array([], bool)
            cat_rows.append({"tile": t, "category": k, "area_m2": float(m.sum() * PX), "components": int(n),
                             "components_ge_mmu": int(big.sum()),
                             "area_in_components_ge_mmu_m2": float(np.isin(comp, np.flatnonzero(big) + 1).sum() * PX) if n else 0.0})
        # D4 partial change per object (interior pixels only)
        interior = valid & ~bnd
        for y, other in (("2013", "2023"), ("2023", "2013")):
            L = lab[y]
            ids = np.unique(L[interior]); ids = ids[ids > 0]
            tot = np.bincount(L[interior].ravel(), minlength=L.max() + 1)
            chg = np.bincount(L[interior & (cls[y] != cls[other])].ravel(), minlength=L.max() + 1)
            full = np.bincount(L.ravel(), minlength=L.max() + 1)
            maj = np.bincount(L.ravel() * 9 + cls[y].ravel(), minlength=(L.max() + 1) * 9).reshape(-1, 9).argmax(1)
            for i in ids:
                c = int(maj[i])
                objs.append({"tile": t, "year": y, "segment_id": int(i), "class": CLS.get(c, "none"), "area_m2": float(full[i] * PX),
                             "interior_m2": float(tot[i] * PX), "changed_interior_m2": float(chg[i] * PX),
                             "changed_fraction": float(chg[i] / max(tot[i], 1))})
    S, O, K = pd.DataFrame(sizes), pd.DataFrame(objs), pd.DataFrame(cat_rows)
    S.to_csv(out / "D1_sizes_by_tile.csv", index=False); O.to_csv(out / "D4_objects.csv", index=False); K.to_csv(out / "D3_categories_by_tile.csv", index=False)
    d1 = S.groupby(["year", "stage", "class"]).agg(n=("n", "sum"), over_1ha=("over_1ha", "sum"), max_m2=("max_m2", "max"),
                                                    area_m2=("area_m2", "sum"), area_over_1ha_m2=("area_over_1ha_m2", "sum")).reset_index()
    d1["share_area_over_1ha"] = (d1.area_over_1ha_m2 / d1.area_m2).round(3)
    d1.to_csv(out / "D1_sizes.csv", index=False)
    T = pd.DataFrame(trans[1:, 1:], index=[CLS[i] for i in range(1, 9)], columns=[CLS[i] for i in range(1, 9)]).round(0)
    T.to_csv(out / "D2_transition_matrix_m2.csv")
    d3 = K.groupby("category")[["area_m2", "components", "components_ge_mmu", "area_in_components_ge_mmu_m2"]].sum()
    d3["share_of_disagreement"] = (d3.area_m2 / d3.area_m2.sum()).round(3)
    d3.to_csv(out / "D3_categories.csv")
    O["bin"] = pd.cut(O.changed_fraction, [-0.001, 0.05, 0.25, 0.75, 1.0], labels=["<5%", "5-25%", "25-75%", ">75%"])
    d4 = O.pivot_table(index=["year", "class"], columns="bin", values="segment_id", aggfunc="count", fill_value=0, observed=False)
    d4a = O.pivot_table(index=["year", "class"], columns="bin", values="area_m2", aggfunc="sum", fill_value=0, observed=False)
    d4.to_csv(out / "D4_partial_change_counts.csv"); d4a.round(0).to_csv(out / "D4_partial_change_area_m2.csv")
    # D5 change unit of existing labelers
    d5 = []
    v1 = pd.concat([pd.read_csv(IMPROVED / "improve/out/mc_change/improved" / f"{t}_nodes_mc.csv") for t in TILES])
    f = v1[v1.state > 0]
    d5.append({"labeler": "improved v1 node states (whole node)", "flagged_m2": f.area_m2.sum(), "changed_m2": f.changed_m2.sum()})
    v2 = pd.concat([pd.read_csv(IMPROVED / "improve/out/mc_change/improved_v2" / f"{t}_regions.csv") for t in TILES])
    d5.append({"labeler": "improved v2 change regions (localised)", "flagged_m2": v2[v2.kind == "change"].area_m2.sum(), "changed_m2": v2[v2.kind == "change"].area_m2.sum()})
    lc = pd.concat([pd.read_csv(Path(a.study) / "graphs" / f"change_stgraph_2013_2023_{t}" / "nodes_change.csv") for t in TILES])
    b = lc[(lc.year == 2013) & (lc.change == "built_over")]
    d5.append({"labeler": "legacy rule 2013 built_over (whole node)", "flagged_m2": b.area_m2.sum(), "changed_m2": b.converted_m2.sum()})
    for t in ("tile_00001", "tile_00003", "tile_00006"):
        run = ROOT / "artifacts" / "shouf3" / f"{t}_geo" / "temporal"
        if not run.is_dir():
            continue
        g = json.loads((run / "graph.json").read_text()); area = {n["id"]: float(n["area"]) for n in g["nodes"]}
        ev = json.loads((run / "classical_changes.json").read_text(encoding="utf-8"))["events"]
        src = {s for e in ev if e["status"] == "candidate" for s in (e["source_nodes"] if isinstance(e["source_nodes"], list) else ast.literal_eval(e["source_nodes"]))}
        d5.append({"labeler": f"repo multi-class candidates {t}: event regions", "flagged_m2": sum(float(e["area"]) for e in ev if e["status"] == "candidate"),
                   "changed_m2": sum(float(e["area"]) for e in ev if e["status"] == "candidate"),
                   "source_nodes_m2": sum(area.get(s, 0) for s in src)})
    D5 = pd.DataFrame(d5); D5["changed_share"] = (D5.changed_m2 / D5.flagged_m2).round(3)
    D5.to_csv(out / "D5_change_units.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(d1[d1.stage == "after_merge"].to_string(index=False))
    print(d1.pivot_table(index=["year", "class"], columns="stage", values="over_1ha").to_string())
    print(T.to_string()); print(d3.round(0).to_string()); print(d4.to_string()); print(d4a.round(0).to_string()); print(D5.round(0).to_string(index=False))


if __name__ == "__main__":
    main()
