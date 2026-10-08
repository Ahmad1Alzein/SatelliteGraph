"""Audit montages for stgraph_change on any tile (images/segments from the frozen releases it used).
python stgraph_change_audit_tile.py tile_00002"""
from osgeo import gdal  # noqa: F401
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # package: pipeline/ modules

import fonts  # package: portable font paths

import cv2
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parents[1]
TILE = sys.argv[1] if len(sys.argv) > 1 else "tile_00002"
OUT = HERE / "graphs" / f"change_stgraph_2013_2023_{TILE}"
F = ImageFont.truetype(fonts.BOLD, 13)
man = json.loads((HERE / "graphs" / f"FROZEN_stgraph_2013_2023_{TILE}_v1" / "MANIFEST.json").read_text())
rels = {k.split("/")[0] for k in man["inputs"] if k.startswith("FROZEN_20")}
IMG, SEG = {}, {}
for rel in rels:
    y = rel.split("_")[1]
    m = json.loads((HERE / "final_segmentation" / rel / "MANIFEST.json").read_text())
    IMG[y] = cv2.resize(rasterio.open(m["input_image"]["path"]).read([1, 2, 3]).transpose(1, 2, 0), (1024, 1024), interpolation=cv2.INTER_CUBIC)
    SEG[y] = rasterio.open(HERE / "final_segmentation" / rel / "segment_ids.tif").read(1)


def cell(ids13, ids23, title, size=230):
    m_all = np.isin(SEG["2013"], ids13) | np.isin(SEG["2023"], ids23)
    ys, xs = np.nonzero(m_all); cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    half = int(np.clip(max(ys.max() - ys.min(), xs.max() - xs.min()) * 0.8, 70, 400))
    y0, y1, x0, x1 = max(0, cy - half), min(1024, cy + half), max(0, cx - half), min(1024, cx + half)
    pair = []
    for y, ids in (("2013", ids13), ("2023", ids23)):
        c = cv2.resize(IMG[y][y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_CUBIC)
        m = cv2.resize(np.isin(SEG[y][y0:y1, x0:x1], ids).astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
        c[cv2.dilate(m ^ cv2.erode(m, np.ones((3, 3), np.uint8)), np.ones((2, 2), np.uint8)).astype(bool)] = (255, 0, 255)
        p = Image.fromarray(c); ImageDraw.Draw(p).text((4, 2), f"{title} {y}", font=F, fill="yellow", stroke_width=2, stroke_fill="black")
        pair.append(np.array(p))
    return np.concatenate([pair[0], np.zeros((size, 3, 3), np.uint8), pair[1]], 1)


def montage(cells, path, cols=4):
    while len(cells) % cols:
        cells.append(np.full_like(cells[0], 30))
    rows = [np.concatenate(sum([[c, np.full((c.shape[0], 10, 3), 30, np.uint8)] for c in cells[i:i + cols]], [])[:-1], 1) for i in range(0, len(cells), cols)]
    Image.fromarray(np.concatenate(sum([[r, np.full((8, r.shape[1], 3), 30, np.uint8)] for r in rows], [])[:-1], 0)).save(path, quality=88)


(OUT / "audit").mkdir(exist_ok=True)
N = pd.read_csv(OUT / "nodes_change.csv")
hi = N[(N.year == 2023) & N.change.isin(["new_building", "extension"]) & (N.confidence == "high")].sort_values(["change", "segment_id"]).reset_index(drop=True)
hi["item"] = range(1, len(hi) + 1)
montage([cell([], [r.segment_id], f"#{r.item} {'NEW' if r.change == 'new_building' else 'EXT'}") for r in hi.itertuples()], OUT / "audit" / "A_high_confidence_building_changes.jpg")
hi[["item", "segment_id", "change", "area_m2", "evidence_new_area_m2"]].to_csv(OUT / "audit" / "A_high_confidence_building_changes.csv", index=False)
CE = pd.read_csv(OUT / "change_edges_conversion.csv"); CE = CE.sample(min(12, len(CE)), random_state=7).reset_index(drop=True)
CE["item"] = range(len(hi) + 1, len(hi) + 1 + len(CE))
montage([cell([r.node_2013], [r.node_2023], f"#{r.item} {r.class_2013[:6]}->bld {r.converted_m2:.0f}") for r in CE.itertuples()], OUT / "audit" / "B_conversion_edges_sample.jpg")
CE.to_csv(OUT / "audit" / "B_conversion_edges_sample.csv", index=False)
print(hi[["item", "segment_id", "change", "area_m2"]].to_string(index=False)); print(CE[["item", "node_2013", "node_2023", "class_2013", "converted_m2", "via"]].to_string(index=False))
