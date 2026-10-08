"""Prepare the pipeline inputs in the work folder (steps 2-3 of the successful run's setup, unchanged logic).

1. Georeference the dataset PNG tiles from their metadata -> inputs/<year>/<tile>.tif
2. Build the two "aligned" rasters the alignment step reads -> inputs/aligned/: dataset tile pixels wherever the
   tiles exist; the full aligned raster only outside them (the co-registration context margin: the 2013 content of
   the bottom ~19 rows and the right edge of tile_00006 lies outside the 2013 tiles). Asserts the tiles equal the
   full raster where they overlap.
"""
from __future__ import annotations

from osgeo import gdal, osr
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.windows import Window

import paths

RUN = Path(__file__).resolve().parent
DATA = paths.DATASET
FULL = {"2023": paths.ALIGNED_FULL / "shouf_2023_shared_epsg32636_1p5m.tif", "2013": paths.ALIGNED_FULL / "shouf_2013_epsg32636_1p5m.tif"}
COMPOSITE_ROWS, COMPOSITE_COLS = 900, 3700   # covers tiles + 256 px context + 64 px pad used by alignment


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    gdal.UseExceptions()
    report = {}
    # 2 georeference dataset tiles
    for y in ("2013", "2023"):
        meta = pd.read_csv(DATA / y / "metadata.csv")
        out = RUN / "inputs" / y; out.mkdir(parents=True, exist_ok=True)
        for r in meta.itertuples():
            dst = out / f"{r.tile_id}.tif"
            gdal.Translate(str(dst), str(DATA / y / r.tile_name), format="GTiff", creationOptions=["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=2"])
            ds = gdal.Open(str(dst), gdal.GA_Update)
            ds.SetGeoTransform(tuple(float(getattr(r, f"tile_affine_{k}")) for k in ("c", "a", "b", "f", "d", "e")))
            sr = osr.SpatialReference(); sr.SetFromUserInput(r.target_crs); ds.SetProjection(sr.ExportToWkt())
            ds.SetMetadataItem("AREA_OR_POINT", "Area"); ds.FlushCache(); ds = None

    # 3 composite aligned rasters
    report["composites"] = {}
    for y in ("2013", "2023"):
        meta = pd.read_csv(DATA / y / "metadata.csv")
        with rasterio.open(FULL[y]) as s:
            win = Window(0, 0, COMPOSITE_COLS, COMPOSITE_ROWS)
            arr = s.read([1, 2, 3], window=win); tr = s.window_transform(win); crs = s.crs
            assert abs(tr.c - meta.grid_affine_c.iloc[0]) < 1e-6 and abs(tr.f - meta.grid_affine_f.iloc[0]) < 1e-6, "raster origin != grid origin"
        from_tiles = np.zeros(arr.shape[1:], bool); equal = []
        for r in meta.itertuples():
            px = np.array(Image.open(DATA / y / r.tile_name).convert("RGB")).transpose(2, 0, 1)
            sl = (slice(None), slice(int(r.y0), int(r.y0) + px.shape[1]), slice(int(r.x0), int(r.x0) + px.shape[2]))
            equal.append(bool(np.array_equal(arr[sl], px)))
            arr[sl] = px; from_tiles[sl[1:]] = True
        dst = RUN / "inputs" / "aligned" / FULL[y].name; dst.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=3, dtype="uint8",
                           crs=crs, transform=tr, compress="deflate", tiled=True) as d:
            d.write(arr)
        assert all(equal), f"{y}: dataset tiles differ from the full raster"
        report["composites"][y] = {"path": str(dst.relative_to(RUN)), "pixels_from_dataset_tiles": int(from_tiles.sum()),
                                   "pixels_from_full_raster": int((~from_tiles).sum()),
                                   "tiles_equal_full_raster": equal, "full_raster_sha256": sha(FULL[y])}

    (RUN / "inputs" / "prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
