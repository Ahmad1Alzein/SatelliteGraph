"""Investigation: oversized 2013 land polygons and the localisation of land -> building change.

Read-only on the frozen study (segmentation_v2: final_segmentation/FROZEN_*, graphs/change_stgraph_*, FROZEN_stgraph_*).
Nothing here changes the legacy building ST-GNN baseline.

Questions
 Q1 How large are land polygons, and which step makes them large? (segment map before vs after refine_merge)
 Q2 How much of a 'built_over' 2013 land polygon actually became building? (confirmed new-building evidence pixels)
 Q3 How much would labelling at the pre-merge level localise the change? (pieces holding >= 25 m2 conversion,
    the same threshold as the original rule)
Outputs (artifacts/experiments/oversized_land_2013/)
 sizes_by_tile.csv, built_over_nodes.csv, summary.json, sheet_largest_built_over.png
Usage: python experiments/oversized_land_2013/investigate.py --study D:/new/DelAnyV2_Experiment/segmentation_v2
"""
from __future__ import annotations

from osgeo import gdal  # noqa: F401  (GDAL DLLs before rasterio on Windows)
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio

TILES = [f"tile_{i:05d}" for i in range(7)]
PX = 0.5625                      # m2 per pixel of the 0.75 m segmentation grid
MIN_CONVERTED_M2 = 25.0          # original stgraph_change rule
NEW, EXT = 1, 2                  # change_detection.CODES new_building / extension
ROOT = Path(__file__).resolve().parents[2]


def releases(study: Path, tile: str) -> dict:
    m = json.loads((study / "graphs" / f"FROZEN_stgraph_2013_2023_{tile}_v1" / "MANIFEST.json").read_text())
    rel = {}
    for k in m["inputs"]:
        if k.endswith("segment_ids.tif"):
            name = k.split("/")[0]
            rel[name.split("_")[1]] = study / "final_segmentation" / name
    return rel


def read(path):
    with rasterio.open(path) as s:
        return s.read(1)


def image(release: Path) -> np.ndarray:
    m = json.loads((release / "MANIFEST.json").read_text())
    p = Path(m["input_image"]["path"])
    if not p.is_absolute():
        p = release.parents[1] / p
    with rasterio.open(p) as s:
        rgb = s.read([1, 2, 3]).transpose(1, 2, 0)
    return cv2.resize(rgb, (1024, 1024), interpolation=cv2.INTER_CUBIC) if rgb.shape[0] != 1024 else rgb


def tile_analysis(study: Path, tile: str):
    rel = releases(study, tile)
    r13, r23 = rel["2013"], rel["2023"]
    post13, pre13, cls13 = read(r13 / "segment_ids.tif"), read(r13 / "segment_ids_before_merge.tif"), read(r13 / "segment_classes.tif")
    lab23 = read(r23 / "segment_ids.tif")
    n13 = pd.read_csv(r13 / "nodes_final.csv").set_index("segment_id")
    lc = pd.read_csv(study / "graphs" / f"change_stgraph_2013_2023_{tile}" / "nodes_change.csv")
    change_map = read(study / "graphs" / f"change_stgraph_2013_2023_{tile}" / "evidence_change_map.tif")
    # confirmed conversion pixels, exactly as stgraph_change derives them
    c23 = lc[(lc.year == 2023) & lc.change.isin(["new_building", "extension"]) & (lc.confidence == "high")].segment_id
    gain = np.isin(lab23, c23.values) & np.isin(change_map, [NEW, EXT])
    land = ~np.isin(cls13, [0, 1, 2])          # 2013 land cover (not building / road / nodata)

    def sizes(lab):
        a = np.bincount(lab[land].ravel()) * PX
        return a[a > 0]
    a_pre, a_post = sizes(pre13), sizes(post13)
    size_row = {"tile": tile, "land_before_merge": len(a_pre), "land_after_merge": len(a_post),
                "median_before_m2": round(float(np.median(a_pre))), "median_after_m2": round(float(np.median(a_post))),
                "over_1ha_before": int((a_pre > 1e4).sum()), "over_1ha_after": int((a_post > 1e4).sum()),
                "max_before_m2": round(float(a_pre.max())), "max_after_m2": round(float(a_post.max())),
                "area_share_over_1ha_after": round(float(a_post[a_post > 1e4].sum() / a_post.sum()), 3)}
    rows = []
    bo = lc[(lc.year == 2013) & (lc.change == "built_over")]
    for r in bo.itertuples():
        m = post13 == r.segment_id
        conv = m & gain
        pieces, pc = np.unique(pre13[m], return_counts=True)
        pconv = np.array([(conv & (pre13 == p)).sum() for p in pieces])
        hit = pconv * PX >= MIN_CONVERTED_M2
        rows.append({"tile": tile, "segment_id": int(r.segment_id), "class": n13.loc[r.segment_id, "class"],
                     "area_m2": float(r.area_m2), "converted_m2_rule": float(r.converted_m2),
                     "converted_m2_pixels": round(float(conv.sum() * PX), 1),
                     "converted_fraction": round(float(conv.sum() / m.sum()), 4),
                     "pre_merge_pieces": len(pieces), "edge_split": bool(n13.loc[r.segment_id, "edge_split"]),
                     "pieces_with_conversion": int(hit.sum()),
                     "pre_merge_label_area_m2": round(float(pc[hit].sum() * PX), 1),
                     "largest_piece_share": round(float(pc.max() / pc.sum()), 3)})
    return size_row, rows, {"post13": post13, "pre13": pre13, "gain": gain, "img13": image(r13), "img23": image(r23)}


def sheet(cases, data, out: Path, k=8, crop=300):
    tiles = []
    for c in cases[:k]:
        d = data[c["tile"]]
        m = d["post13"] == c["segment_id"]
        ys, xs = np.nonzero(m)
        cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
        h = max(crop // 2, int(0.6 * max(np.ptp(ys), np.ptp(xs))))
        y0, x0 = max(0, cy - h), max(0, cx - h); y1, x1 = min(1024, cy + h), min(1024, cx + h)
        panels = []
        for key in ("img13", "img23"):
            im = d[key][y0:y1, x0:x1].copy()
            pre = d["pre13"][y0:y1, x0:x1]; mm = m[y0:y1, x0:x1]
            edges = (np.diff(pre, axis=0, prepend=pre[:1]) != 0) | (np.diff(pre, axis=1, prepend=pre[:, :1]) != 0)
            im[edges & mm] = (255, 255, 255)                     # pre-merge pieces inside the polygon
            g = d["gain"][y0:y1, x0:x1] & mm
            im[g] = (0.4 * im[g] + 0.6 * np.array([230, 30, 30])).astype(np.uint8)   # confirmed conversion
            cs, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(im, cs, -1, (255, 220, 0), 2)       # the 2013 node (whole label)
            panels.append(cv2.resize(im, (360, 360), interpolation=cv2.INTER_AREA))
        row = np.hstack([panels[0], np.full((360, 6, 3), 30, np.uint8), panels[1]])
        bar = np.full((44, row.shape[1], 3), 24, np.uint8)
        txt = (f"{c['tile']} #{c['segment_id']} {c['class']}: node {c['area_m2']/1e4:.2f} ha, converted {c['converted_m2_pixels']:.0f} m2 "
               f"({100*c['converted_fraction']:.1f}%), {c['pre_merge_pieces']} pre-merge pieces")
        cv2.putText(bar, txt, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, row]))
    rows = [np.hstack([tiles[i], np.full((tiles[i].shape[0], 14, 3), 255, np.uint8), tiles[i + 1]]) if i + 1 < len(tiles) else tiles[i]
            for i in range(0, len(tiles), 2)]
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 14), (0, w - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    head = np.full((70, w, 3), 24, np.uint8)
    cv2.putText(head, "Largest 2013 land nodes labelled 'built_over' (2013 | 2023)", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(head, "yellow = 2013 node (whole node is the positive label); red = confirmed new-building pixels; white = pieces before refine_merge",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), cv2.cvtColor(np.vstack([head] + rows), cv2.COLOR_RGB2BGR))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default=r"D:\new\DelAnyV2_Experiment\segmentation_v2")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "experiments" / "oversized_land_2013"))
    a = ap.parse_args()
    study, out = Path(a.study), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sizes, rows, data = [], [], {}
    for t in TILES:
        s, r, d = tile_analysis(study, t)
        sizes.append(s); rows += r; data[t] = d
    S, B = pd.DataFrame(sizes), pd.DataFrame(rows).sort_values("area_m2", ascending=False)
    S.to_csv(out / "sizes_by_tile.csv", index=False); B.to_csv(out / "built_over_nodes.csv", index=False)
    summary = {
        "land_polygons_over_1ha_before_after_merge": [int(S.over_1ha_before.sum()), int(S.over_1ha_after.sum())],
        "built_over_nodes": len(B),
        "built_over_label_area_m2": round(float(B.area_m2.sum())),
        "converted_area_m2": round(float(B.converted_m2_pixels.sum())),
        "converted_share_of_label_area": round(float(B.converted_m2_pixels.sum() / B.area_m2.sum()), 3),
        "median_converted_fraction": float(B.converted_fraction.median()),
        "built_over_nodes_from_several_pieces": int((B.pre_merge_pieces > 1).sum()),
        "pre_merge_label_area_m2": round(float(B.pre_merge_label_area_m2.sum())),
        "pre_merge_converted_share": round(float(B.converted_m2_pixels.sum() / max(B.pre_merge_label_area_m2.sum(), 1)), 3),
        "rule_vs_pixel_converted_max_abs_diff_m2": round(float((B.converted_m2_rule - B.converted_m2_pixels).abs().max()), 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    sheet(B.to_dict("records"), data, out / "sheet_largest_built_over.png")
    pd.set_option("display.width", 220)
    print(S.to_string(index=False)); print(B.head(12).to_string(index=False)); print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
