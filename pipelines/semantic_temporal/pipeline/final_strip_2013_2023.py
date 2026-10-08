"""Final segmentation for change detection: 2013 (GeoEye) vs 2023, validated 7-tile strip.

Steps
1. Read the strip covering tiles tile_00000..tile_00006 (512 x 3284 px, 1.5 m) from the shared-grid
   rasters of both years.
2. Co-register 2013 onto the 2023 grid. The "aligned" 2013 raster is offset by ~20 px (~30 m).
   Local offsets are measured by phase correlation in overlapping windows, outliers rejected, a
   smooth (linear in x, y) shift field fitted, and 2013 resampled with it. The residual is then
   re-measured on the same windows.
3. Harmonize the pair (2013 histogram-matched to 2023; the sharper image blurred to the other's
   sharpness), as validated.
4. Segment both harmonized strips as one seamless mosaic each (segment_v2.segment_image).
5. Write per year: object_ids.tif (0.75 m), objects.gpkg, nodes.csv, edges.csv, plus the
   co-registered/harmonized input images and QA files.
"""
from __future__ import annotations

from osgeo import gdal, osr  # noqa: F401  (GDAL DLLs before torch)
import json
import time
from pathlib import Path

import paths  # package: input/checkpoint locations

import cv2
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.windows import Window

import harmonize
import segment_v2

ALIGNED = Path(__file__).resolve().parent / "inputs" / "aligned"   # run: dataset tiles + full-raster margin
META = paths.META_2023   # package: input location from paths.py
REFERENCE = ("2023", ALIGNED / "shouf_2023_shared_epsg32636_1p5m.tif")
MOVING = ("2013", ALIGNED / "shouf_2013_epsg32636_1p5m.tif")
TILES = [f"tile_{i:05d}" for i in range(7)]
NAME = "strip_tile_00000-00006"
OUT = Path(__file__).resolve().parent / "final" / NAME
PAD = 64           # px read around the strip for the moving image (offsets are ~20 px)
WIN, STEP = 256, 128


def strip_extent():
    m = pd.read_csv(META).set_index("tile_id").loc[TILES]
    return int(m.y0.min()), int(m.y1.max()), int(m.x0.min()), int(m.x1.max())


def read_window(path, row0, row1, col0, col1):
    """Read rows/cols (may extend past the raster; outside is zero-filled)."""
    with rasterio.open(path) as src:
        win = Window(col0, row0, col1 - col0, row1 - row0)
        arr = src.read([1, 2, 3], window=win, boundless=True, fill_value=0)
        transform = src.window_transform(win)
        crs = src.crs
    return arr.transpose(1, 2, 0).astype(np.uint8), transform, crs


def lap(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 1.0), cv2.CV_32F)


def local_shifts(ref, mov, step=STEP, offset=0):
    """Phase-correlation shift of `mov` relative to `ref` in WIN windows (same array size)."""
    hann = cv2.createHanningWindow((WIN, WIN), cv2.CV_32F)
    lr, lm = lap(ref), lap(mov)
    rows = []
    for y in range(offset, ref.shape[0] - WIN + 1, step):
        for x in range(offset, ref.shape[1] - WIN + 1, step):
            (dx, dy), resp = cv2.phaseCorrelate(lr[y:y + WIN, x:x + WIN], lm[y:y + WIN, x:x + WIN], hann)
            rows.append({"cx": x + WIN / 2, "cy": y + WIN / 2, "dx": dx, "dy": dy, "response": resp})
    return pd.DataFrame(rows)


def fit_field(df, min_resp=0.08):
    """Robust linear fit d = a + b*cx + c*cy for dx and dy (iteratively drops >2.5 px residuals)."""
    good = df[df.response >= min_resp].copy()
    coefs = {}
    for comp in ("dx", "dy"):
        d = good
        for _ in range(3):
            A = np.c_[np.ones(len(d)), d.cx, d.cy]
            coef, *_ = np.linalg.lstsq(A, d[comp].values, rcond=None)
            res = np.abs(A @ coef - d[comp].values)
            d = d[res <= max(2.5, np.percentile(res, 90))]
        coefs[comp] = coef
    return coefs, len(good)


def local_field(samples, shape, sigma=160.0, min_resp=0.15, max_abs=6.0):
    """Smooth local residual field (px) from window samples: Gaussian-weighted average of
    confident samples (Nadaraya-Watson), evaluated on a coarse grid and upsampled."""
    good = samples[(samples.response >= min_resp) & (np.hypot(samples.dx, samples.dy) <= max_abs)]
    h, w = shape
    gy, gx = np.mgrid[0:h:16, 0:w:16].astype(np.float32)
    out = []
    for comp in ("dx", "dy"):
        num = np.zeros(gx.shape, np.float64); den = np.full(gx.shape, 1e-6)
        for _, r in good.iterrows():
            wgt = r.response * np.exp(-((gx - r.cx) ** 2 + (gy - r.cy) ** 2) / (2 * sigma ** 2))
            num += wgt * r[comp]; den += wgt
        out.append(cv2.resize((num / den).astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR))
    return out[0], out[1], len(good)


def warp_moving(mov_padded, coefs, shape, local=None):
    """Sample the padded moving image at x + dx(x, y), y + dy(x, y) for every reference pixel."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = coefs["dx"][0] + coefs["dx"][1] * xx + coefs["dx"][2] * yy
    dy = coefs["dy"][0] + coefs["dy"][1] * xx + coefs["dy"][2] * yy
    if local is not None:
        dx, dy = dx + local[0], dy + local[1]
    map_x = (xx + dx + PAD).astype(np.float32)
    map_y = (yy + dy + PAD).astype(np.float32)
    return cv2.remap(mov_padded, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)


def write_rgb(path, img, transform, crs):
    prof = {"driver": "GTiff", "height": img.shape[0], "width": img.shape[1], "count": 3, "dtype": "uint8",
            "crs": crs, "transform": transform, "compress": "deflate", "tiled": True}
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(img.transpose(2, 0, 1))


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    r0, r1, c0, c1 = strip_extent()
    ref, transform, crs = read_window(REFERENCE[1], r0, r1, c0, c1)
    mov_pad, _, _ = read_window(MOVING[1], r0 - PAD, r1 + PAD, c0 - PAD, c1 + PAD)
    report = {"strip_rows": [r0, r1], "strip_cols": [c0, c1], "reference_year": REFERENCE[0], "moving_year": MOVING[0]}

    # --- 2. co-registration
    before = local_shifts(ref, mov_pad[PAD:-PAD, PAD:-PAD])
    coefs, n_good = fit_field(before)
    linear = warp_moving(mov_pad, coefs, ref.shape[:2])
    # Local refinement on the linearly warped image. Fit on windows of the even grid (step 64,
    # offset 0) and evaluate on the interleaved odd grid (offset 32): an honest held-out residual.
    fit_samples = local_shifts(ref, linear, step=64, offset=0)
    ldx, ldy, n_local = local_field(fit_samples, ref.shape[:2])
    mov = warp_moving(mov_pad, coefs, ref.shape[:2], local=(ldx, ldy))
    eval_lin = local_shifts(ref, linear, step=64, offset=32)
    eval_loc = local_shifts(ref, mov, step=64, offset=32)
    after = local_shifts(ref, mov)
    before.to_csv(OUT / "coregistration_shifts_before.csv", index=False)
    after.to_csv(OUT / "coregistration_shifts_after.csv", index=False)
    g_b, g_a = before[before.response >= 0.08], after[after.response >= 0.08]
    mag = lambda d: np.hypot(d.dx, d.dy)
    report["coregistration"] = {
        "model": "dx, dy = a + b*x + c*y (px, strip coordinates)",
        "dx_coef": [round(float(v), 6) for v in coefs["dx"]], "dy_coef": [round(float(v), 6) for v in coefs["dy"]],
        "windows": len(before), "confident_windows": n_good,
        "offset_before_px_median": round(float(mag(g_b).median()), 2),
        "residual_after_px_median": round(float(mag(g_a).median()), 2),
        "residual_after_px_p90": round(float(mag(g_a).quantile(0.9)), 2),
        "residual_after_px_max": round(float(mag(g_a).max()), 2),
        "local_refinement_samples": n_local,
        "local_field_px_range": [round(float(np.hypot(ldx, ldy).min()), 2), round(float(np.hypot(ldx, ldy).max()), 2)],
    }
    for name, ev in (("heldout_linear_only", eval_lin), ("heldout_linear_plus_local", eval_loc)):
        gg = ev[(ev.response >= 0.15) & (mag(ev) <= 8)]
        report["coregistration"][name] = {"windows": len(gg), "median_px": round(float(mag(gg).median()), 2),
                                          "p90_px": round(float(mag(gg).quantile(0.9)), 2)}
    print(json.dumps(report["coregistration"]), flush=True)

    # --- 3. harmonization
    ref_h, mov_h, info = harmonize.harmonize_pair(ref, mov)
    report["harmonization"] = info
    write_rgb(OUT / f"input_{REFERENCE[0]}_harmonized.tif", ref_h, transform, crs)
    write_rgb(OUT / f"input_{MOVING[0]}_coregistered_harmonized.tif", mov_h, transform, crs)
    write_rgb(OUT / f"input_{MOVING[0]}_coregistered.tif", mov, transform, crs)

    # --- 4/5. segmentation
    tr2 = transform * Affine.scale(1 / segment_v2.SCALE)
    report["years"] = {}
    for year, image in ((REFERENCE[0], ref_h), (MOVING[0], mov_h)):
        ts = time.time()
        log = {"tile": NAME, "year": year, "building_mode": segment_v2.P["building_mode"]}
        final, meta, prob = segment_v2.segment_image(image, abs(tr2.a * tr2.e), log)
        out_dir = OUT / year
        out_dir.mkdir(parents=True, exist_ok=True)
        segment_v2.write_outputs(out_dir, NAME, image, final, meta, prob, tr2, crs, log, year=year)
        report["years"][year] = {k: log[k] for k in ("nodes", "edges", "by_type", "non_polygon_geometries")}
        report["years"][year]["seconds"] = round(time.time() - ts, 1)
        print(year, json.dumps(report["years"][year]), flush=True)
    report["settings"] = {k: v for k, v in segment_v2.P.items()}
    report["seconds_total"] = round(time.time() - t0, 1)
    (OUT / "run_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
