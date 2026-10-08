"""Measure the local offset between years on the shared tile grid (phase correlation of Laplacian
images, 512 px tiles, sampled over the whole AOI). Positive dx/dy = the other year's content lies
right/down of 2023."""
from osgeo import gdal  # noqa: F401
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

TILES = Path(r"D:\new\SatelliteGraph\data\tiles")
META = Path(r"D:\new\SatelliteGraph\data\metadata\shouf_2023_shared_tiles.csv")
OUT = Path(__file__).resolve().parent


def lap(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 1.0), cv2.CV_32F)


def shift(a, b):
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(lap(a), lap(b), win)
    return dx, dy, resp


def main(step=21):
    meta = pd.read_csv(META)
    sample = meta.iloc[::step]
    rows = []
    for _, r in sample.iterrows():
        t = r.tile_id
        a = np.array(Image.open(TILES / "shouf_2023_shared" / f"{t}.png").convert("RGB"))
        for year in ("2013", "2017"):
            b = np.array(Image.open(TILES / f"shouf_{year}" / f"{t}.png").convert("RGB"))
            dx, dy, resp = shift(a, b)
            rows.append({"tile_id": t, "year": year, "x0": r.x0, "y0": r.y0, "dx": round(dx, 2), "dy": round(dy, 2), "response": round(resp, 3)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "shift_samples.csv", index=False)
    for year, d in df.groupby("year"):
        good = d[d.response >= 0.08]
        print(f"{year}: {len(d)} tiles, {len(good)} confident | dx median {good.dx.median():.1f} [p10 {good.dx.quantile(.1):.1f}, p90 {good.dx.quantile(.9):.1f}] | "
              f"dy median {good.dy.median():.1f} [p10 {good.dy.quantile(.1):.1f}, p90 {good.dy.quantile(.9):.1f}]")
        # spatial trend: correlation of shift with position
        print(f"   corr(dx, x0)={good.dx.corr(good.x0):.2f} corr(dx, y0)={good.dx.corr(good.y0):.2f} corr(dy, x0)={good.dy.corr(good.x0):.2f} corr(dy, y0)={good.dy.corr(good.y0):.2f}")


if __name__ == "__main__":
    main()
