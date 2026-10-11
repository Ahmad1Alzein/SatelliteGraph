"""Figure for the multi-class diagnosis: where the 2013 and 2023 frozen segmentations disagree, by category, for one tile.
Read-only on the frozen study. Output artifacts/experiments/multiclass_localised/diagnosis/disagreement_<tile>.png
Usage: python experiments/oversized_land_2013/render_disagreement.py tile_00002"""
from __future__ import annotations

from osgeo import gdal  # noqa: F401
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diagnose_multiclass as D  # noqa: E402
import investigate as I  # noqa: E402

COL = {"boundary_band": (150, 150, 150), "building_lean_zone": (255, 150, 30), "vegetation_internal": (60, 200, 90),
       "built_transition": (235, 40, 40)}


def main(tile="tile_00002", study=Path(r"D:\new\DelAnyV2_Experiment\segmentation_v2")):
    rel = D.releases(study, tile)
    lab = {y: D.read(rel[y] / "segment_ids.tif") for y in ("2013", "2023")}
    cls = {y: D.read(rel[y] / "segment_classes.tif") for y in ("2013", "2023")}
    c13, c23 = cls["2013"], cls["2023"]
    diff = (c13 > 0) & (c23 > 0) & (c13 != c23)
    bnd = D.boundary(lab["2013"], D.BOUNDARY_PX) | D.boundary(lab["2023"], D.BOUNDARY_PX)
    lean = ndimage.binary_dilation((c13 == 1) | (c23 == 1), iterations=D.LEAN_PX) & ((c13 == 1) | (c23 == 1)) & ~((c13 == 1) & (c23 == 1))
    vv = np.isin(c13, list(D.VEG)) & np.isin(c23, list(D.VEG))
    masks = {"boundary_band": diff & bnd}
    masks["building_lean_zone"] = diff & ~bnd & lean
    masks["vegetation_internal"] = diff & ~bnd & ~lean & vv
    masks["built_transition"] = diff & ~bnd & ~lean & ~vv
    img13, img23 = I.image(rel["2013"]), I.image(rel["2023"])
    over = (0.45 * img23).astype(np.uint8)
    for k, m in masks.items():
        over[m] = COL[k]
    pane = lambda a: cv2.resize(a, (700, 700), interpolation=cv2.INTER_AREA)
    row = np.hstack([pane(img13), np.full((700, 8, 3), 255, np.uint8), pane(img23), np.full((700, 8, 3), 255, np.uint8), pane(over)])
    head = np.full((96, row.shape[1], 3), 24, np.uint8)
    tot = sum(m.sum() for m in masks.values()) * D.PX
    cv2.putText(head, f"{tile}: 2013 | 2023 | pixels where the two frozen segmentations disagree ({tot/1e4:.1f} ha of {diff.size*D.PX/1e4:.1f} ha)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    x = 10
    for k, c in COL.items():
        cv2.rectangle(head, (x, 52), (x + 22, 74), c, -1)
        lbl = f"{k.replace('_', ' ')} {masks[k].sum()*D.PX/1e4:.1f} ha"
        cv2.putText(head, lbl, (x + 30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        x += 40 + 11 * len(lbl)
    out = D.ROOT / "artifacts" / "experiments" / "multiclass_localised" / "diagnosis" / f"disagreement_{tile}.png"
    cv2.imwrite(str(out), cv2.cvtColor(np.vstack([head, row]), cv2.COLOR_RGB2BGR))
    print(out)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or ["tile_00002"]))
