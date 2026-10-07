"""Rasterize OSM building footprints onto each tile's 0.75 m grid as an instance reference.

Also estimates the OSM-vs-imagery shift per tile by cross-correlating the OSM building mask with the
SegFormer building probability (model-agnostic: the same shift is applied for v1 and v2), and
writes an overlay for visual completeness checks.
"""
from osgeo import gdal  # noqa: F401
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio import features
from rasterio.warp import transform_geom
from affine import Affine
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
REF = Path(__file__).resolve().parent / "reference"
TILES = [f"tile_{i:05d}" for i in range(7)]


def osm_polygons():
    data = json.loads((REF / "osm_buildings.json").read_text(encoding="utf-8"))
    polys = []
    for el in data["elements"]:
        if el["type"] == "way" and "geometry" in el:
            ring = [(p["lon"], p["lat"]) for p in el["geometry"]]
            if len(ring) >= 4:
                polys.append({"type": "Polygon", "coordinates": [ring]})
        elif el["type"] == "relation":
            for m in el.get("members", []):
                if m.get("role") == "outer" and "geometry" in m:
                    ring = [(p["lon"], p["lat"]) for p in m["geometry"]]
                    if len(ring) >= 4 and ring[0] == ring[-1]:
                        polys.append({"type": "Polygon", "coordinates": [ring]})
    return [transform_geom("EPSG:4326", "EPSG:32636", g) for g in polys]


def best_shift(ref_mask, prob, max_shift=8):
    """Integer shift (dy, dx) in 2x px that best aligns the OSM mask to the building probability."""
    best, arg = -1e9, (0, 0)
    r = ref_mask.astype(np.float32) - ref_mask.mean()
    p = prob - prob.mean()
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            s = np.roll(np.roll(r, dy, 0), dx, 1)
            score = float((s[16:-16, 16:-16] * p[16:-16, 16:-16]).sum())
            if score > best:
                best, arg = score, (dy, dx)
    return arg


def main():
    import semantic_ms as sm
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    polys = osm_polygons()
    summary = {}
    for tile in TILES:
        with rasterio.open(ROOT / "prepared_input" / "2013" / f"{tile}.tif") as s:
            img = s.read([1, 2, 3]).transpose(1, 2, 0).astype(np.uint8)
            tr = s.transform
        tr2 = tr * Affine.scale(0.5)
        ref = features.rasterize(((g, i + 1) for i, g in enumerate(polys)), out_shape=(1024, 1024),
                                 transform=tr2, fill=0, dtype=np.int32)
        _, p2 = sm.infer(img, scale=2, return_upsampled=True)
        dy, dx = best_shift(ref > 0, p2[8])
        ref_aligned = np.roll(np.roll(ref, dy, 0), dx, 1)
        # zero the wrapped-around border
        if dy > 0: ref_aligned[:dy] = 0
        if dy < 0: ref_aligned[dy:] = 0
        if dx > 0: ref_aligned[:, :dx] = 0
        if dx < 0: ref_aligned[:, dx:] = 0
        np.save(REF / f"{tile}_osm_instances.npy", ref_aligned)
        n = len(np.unique(ref_aligned)) - 1
        summary[tile] = {"osm_buildings": n, "shift_px2x": [dy, dx], "shift_m": [dy * 0.75, dx * 0.75]}
        big = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_CUBIC)
        edge = np.zeros(ref_aligned.shape, bool)
        edge[:, 1:] |= ref_aligned[:, 1:] != ref_aligned[:, :-1]
        edge[1:, :] |= ref_aligned[1:, :] != ref_aligned[:-1, :]
        big[edge] = (255, 255, 0)
        Image.fromarray(big).save(REF / f"{tile}_osm_overlay.jpg", quality=85)
        print(tile, summary[tile], flush=True)
    (REF / "osm_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
