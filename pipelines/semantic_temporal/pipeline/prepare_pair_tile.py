"""Prepare the 2013 counterpart of a 2023 tile for segmentation with the frozen code.

1. Reference = 2023 shared-grid raster; the 2023 tile pixels must equal the frozen 2023 input.
2. Co-register 2013 (GeoEye) onto the 2023 grid: offsets measured by phase correlation on a context
   window around the tile (tile +- 256 px), robust linear field + smooth local refinement
   (final_strip_2013_2023 functions), residual checked on held-out windows.
3. Harmonize (harmonize.harmonize_pair): 2013 histogram-matched to 2023; the sharper image is
   blurred. If the 2023 reference would be blurred, the frozen 2023 release is no longer the right
   counterpart and the script stops and says so.
4. Write the prepared 2013 tile as a GeoTIFF on the exact 2023 tile grid + a provenance JSON.
"""
from __future__ import annotations

from osgeo import gdal  # noqa: F401
import argparse
import hashlib
import json
from pathlib import Path

import paths  # package: input/checkpoint locations

import numpy as np
import pandas as pd
import rasterio

import final_strip_2013_2023 as fs
import harmonize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
META = paths.META_2023   # package: input location from paths.py
OUT = HERE / "final_segmentation" / "inputs"
CTX = 256


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tile", default="tile_00000")
    ap.add_argument("--reference-release", default="FROZEN_2023_tile_00000_v1")
    ap.add_argument("--write-harmonized-reference", action="store_true",
                    help="if harmonization blurs the 2023 reference, write the harmonized 2023 tile so a paired 2023 release can be segmented from it")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(META).set_index("tile_id").loc[args.tile]
    y0, y1, x0, x1 = int(m.y0), int(m.y1), int(m.x0), int(m.x1)
    with rasterio.open(fs.REFERENCE[1]) as s:
        H, W = s.height, s.width
    cy0, cy1, cx0, cx1 = max(0, y0 - CTX), min(H, y1 + CTX), max(0, x0 - CTX), min(W, x1 + CTX)
    ref_ctx, ctx_tr, crs = fs.read_window(fs.REFERENCE[1], cy0, cy1, cx0, cx1)
    mov_pad, _, _ = fs.read_window(fs.MOVING[1], cy0 - fs.PAD, cy1 + fs.PAD, cx0 - fs.PAD, cx1 + fs.PAD)

    # 1 the 2023 reference must be exactly the frozen 2023 input
    release = HERE / "final_segmentation" / args.reference_release
    frozen_input = Path(json.loads((release / "run_log.json").read_text())["image"])
    with rasterio.open(frozen_input) as s:
        frozen_img = s.read([1, 2, 3]).transpose(1, 2, 0); tile_tr = s.transform
    ty, tx = y0 - cy0, x0 - cx0
    ref_tile = ref_ctx[ty:ty + (y1 - y0), tx:tx + (x1 - x0)]
    same = bool(np.array_equal(ref_tile, frozen_img))
    print("2023 reference tile == frozen 2023 input:", same)
    if not same:
        raise SystemExit("2023 reference differs from the frozen 2023 input")

    # 2 co-registration on the context window
    #   a) translation from the whole context window (robust on small areas where few 256 px windows
    #      correlate across seasons), refined iteratively on the warped image;
    #   b) smooth local refinement only if enough windows correlate confidently;
    #   c) acceptance gate: whole-tile residual <= GATE_PX with a clear response, and agreement with
    #      the independent AOI-wide shift samples near this tile.
    import cv2
    hann = cv2.createHanningWindow((ref_ctx.shape[1], ref_ctx.shape[0]), cv2.CV_32F)
    raw_ctx = mov_pad[fs.PAD:-fs.PAD, fs.PAD:-fs.PAD]
    (dx, dy), resp0 = cv2.phaseCorrelate(fs.lap(ref_ctx), fs.lap(raw_ctx), hann)
    coefs = {"dx": np.array([dx, 0.0, 0.0]), "dy": np.array([dy, 0.0, 0.0])}
    for _ in range(3):
        warped = fs.warp_moving(mov_pad, coefs, ref_ctx.shape[:2])
        (rx, ry), _ = cv2.phaseCorrelate(fs.lap(ref_ctx), fs.lap(warped), hann)
        coefs["dx"][0] += rx; coefs["dy"][0] += ry
    linear = fs.warp_moving(mov_pad, coefs, ref_ctx.shape[:2])
    samples = fs.local_shifts(ref_ctx, linear, step=64, offset=0)
    n_conf = int(((samples.response >= 0.15) & (np.hypot(samples.dx, samples.dy) <= 6)).sum())
    if n_conf >= 8:
        ldx, ldy, n_local = fs.local_field(samples, ref_ctx.shape[:2])
        mov_ctx = fs.warp_moving(mov_pad, coefs, ref_ctx.shape[:2], local=(ldx, ldy))
    else:
        n_local, mov_ctx = 0, linear
    mov_tile = mov_ctx[ty:ty + (y1 - y0), tx:tx + (x1 - x0)]
    th = cv2.createHanningWindow((ref_tile.shape[1], ref_tile.shape[0]), cv2.CV_32F)
    (gx, gy), gresp = cv2.phaseCorrelate(fs.lap(ref_tile), fs.lap(mov_tile), th)
    # independent check against the AOI-wide samples (coregistration/shift_samples.csv)
    samp = pd.read_csv(HERE / "coregistration" / "shift_samples.csv")
    samp = samp[(samp.year == 2013) & (samp.response >= 0.08)].copy()
    samp["d"] = np.hypot(samp.x0 - x0, samp.y0 - y0)
    near = samp.nsmallest(8, "d")
    aoi_dx, aoi_dy = float(near.dx.median()), float(near.dy.median())
    agree = float(np.hypot(coefs["dx"][0] - aoi_dx, coefs["dy"][0] - aoi_dy))
    res = {"translation_px": [round(float(coefs["dx"][0]), 2), round(float(coefs["dy"][0]), 2)],
           "context_response": round(float(resp0), 3),
           "tile_residual_px": [round(float(gx), 2), round(float(gy), 2)], "tile_residual_response": round(float(gresp), 3),
           "aoi_samples_near_tile_px": [round(aoi_dx, 2), round(aoi_dy, 2)], "disagreement_with_aoi_px": round(agree, 2),
           "local_refinement_windows": n_conf, "local_refinement_applied": n_local > 0}
    print("co-registration:", res)
    GATE_PX = 1.0
    if np.hypot(gx, gy) > GATE_PX or gresp < 0.1 or agree > 3.0:
        raise SystemExit(f"Co-registration rejected by the acceptance gate: {res}")
    n_good = n_conf
    ldx = ldy = None

    # 3 harmonization
    ref_h, mov_h, info = harmonize.harmonize_pair(ref_tile, mov_tile)
    print("harmonization:", info)
    ref_path = None
    if (info["blur_sigma_reference"] > 0 or not np.array_equal(ref_h, ref_tile)) and args.write_harmonized_reference:
        ref_path = OUT / f"2023_{args.tile}_harmonized.tif"
        with rasterio.open(ref_path, "w", driver="GTiff", height=ref_h.shape[0], width=ref_h.shape[1], count=3, dtype="uint8",
                           crs=crs, transform=tile_tr, compress="deflate", tiled=True) as d:
            d.write(ref_h.transpose(2, 0, 1))
        print("harmonized 2023 reference written:", ref_path)
    elif info["blur_sigma_reference"] > 0 or not np.array_equal(ref_h, ref_tile):
        raise SystemExit("Harmonization would modify the 2023 reference: the 2023 release must be "
                         "regenerated from the harmonized 2023 image before pairing.")

    # 4 write the prepared 2013 tile on the 2023 tile grid
    path = OUT / f"2013_{args.tile}_coreg_harmonized.tif"
    prof = {"driver": "GTiff", "height": mov_h.shape[0], "width": mov_h.shape[1], "count": 3, "dtype": "uint8",
            "crs": crs, "transform": tile_tr, "compress": "deflate", "tiled": True}
    with rasterio.open(path, "w", **prof) as d:
        d.write(mov_h.transpose(2, 0, 1))
    raw_path = OUT / f"2013_{args.tile}_coreg.tif"
    with rasterio.open(raw_path, "w", **prof) as d:
        d.write(mov_tile.transpose(2, 0, 1))
    prov = {"tile": args.tile, "reference_release": args.reference_release, "reference_input": str(frozen_input),
            "moving_source": str(fs.MOVING[1]), "context_window_rows_cols": [cy0, cy1, cx0, cx1],
            "coregistration": {"model": "translation from whole context window (iteratively refined) + smooth local field if >= 8 confident windows",
                               "dx_coef": [float(v) for v in coefs["dx"]], "dy_coef": [float(v) for v in coefs["dy"]],
                               "confident_windows": int(n_good), "local_samples": int(n_local), "checks": res, "acceptance_gate_px": 1.0},
            "harmonization": info, "output": str(path),
            "harmonized_2023_reference": str(ref_path) if ref_path else None,
            "harmonized_2023_reference_sha256": hashlib.sha256(ref_path.read_bytes()).hexdigest() if ref_path else None,
            "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT / f"2013_{args.tile}_provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
