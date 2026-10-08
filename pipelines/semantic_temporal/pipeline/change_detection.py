"""Change detection 2013 -> 2023 on the final strip segmentation.

Design (from the validation and the audit of the cross-year links):
  * Object identity alone is not reliable evidence: split/merge events were mostly differences in
    how touching houses were grouped, and 'gone' buildings were false 2013 detections. So a change
    must be supported by the IMAGE EVIDENCE of both years with a safety margin, and is then
    attributed to graph nodes.
  * Buildings are compared in the roof frame (2013 roofs lean 3-9 m east relative to the ground
    because of the different view angle; the regional roof offset is compensated). Ground classes
    (trees, roads) are compared in the ground-registered frame.
  * low vegetation <-> bare is NOT reported: 2013 is spring, 2023 is dry summer.

Change types (code in change_map.tif):
  1 new_building    2023 building whose footprint has no 2013 building evidence (>= 60% of it)
  2 extension       >= 40 m2 of new building area attached to a building that existed in 2013
  3 demolished      2013 building with no 2023 building evidence (>= 60% of it)
  4 tree_loss       tree in 2013, clearly not tree in 2023 (>= 150 m2, away from buildings)
  5 tree_gain       the reverse
  6 new_road        road in 2023, clearly not road in 2013 (>= 100 m2)
Evidence rule for "present": probability >= PRESENT at the 2x scale used for segmentation.
Evidence rule for "absent":  max probability over 1x/1.5x/2x scales, max-filtered over +-1.5 m,
                             <= ABSENT (no detection anywhere nearby, at any scale).
"""
from __future__ import annotations

from osgeo import gdal, ogr, osr  # noqa: F401  (GDAL DLLs before torch)
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from scipy import ndimage
from shapely import make_valid
from shapely.geometry import shape

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "validation"))
import harmonize  # noqa: E402
import segment_v2  # noqa: E402
import semantic_ms as sm  # noqa: E402
import final_strip_qa as qa  # noqa: E402

D = HERE / "final" / "strip_tile_00000-00006"
OUT = D / "change_2013_2023"
PX = 0.5625                      # m2 per 0.75 m pixel
CODES = {"new_building": 1, "extension": 2, "demolished": 3, "tree_loss": 4, "tree_gain": 5, "new_road": 6}
PARAMS = {
    "PRESENT": 0.5, "ABSENT": 0.2, "absent_window_px": 5,
    "object_fraction_new": 0.6, "extension_min_m2": 40.0, "building_min_m2": 40.0,
    "tree_min_m2": 150.0, "road_min_m2": 100.0, "building_buffer_px": 4,
    "reliable_min_m2": 60.0, "reliable_min_conf": 0.6,
    # added after the visual audit of the first run (see audit/ and README):
    "shadow_gray": 20,              # pixels darker than this carry no evidence (deep shadow)
    "new_building_high_m2": 100.0,  # new area needed for a high-confidence new building
    "extension_high_m2": 100.0,     # new area needed for a high-confidence extension
    "region_high_min_m2": 100.0,    # ...and it must come as ONE connected piece of this size
                                    # (second audit: thin/small extension pieces were mostly false)
}
# Audited precision of the first run on this spring(2013)/summer(2023) pair was ~10-20% for these
# change types, so they are reported but marked 'unverified' and kept out of the headline numbers.
UNVERIFIED = {"demolished", "tree_loss", "tree_gain", "new_road"}
GROUPS = {"tree": [5], "road": [4], "building": [8]}


# ----------------------------------------------------------------------------- evidence
def evidence(image):
    """Per class: 2x probability (shape scale) and max over 1x/1.5x/2x (for 'absent' tests)."""
    _, p2 = sm.infer(image, scale=2, return_upsampled=True)
    size = p2.shape[:0:-1]
    out = {f"{g}_2x": p2[ids].sum(0) for g, ids in GROUPS.items()}
    mx = {g: out[f"{g}_2x"].copy() for g in GROUPS}
    for s in (1.0, 1.5):
        p = sm.infer(image, scale=s)
        for g, ids in GROUPS.items():
            mx[g] = np.maximum(mx[g], cv2.resize(p[ids].sum(0), size, interpolation=cv2.INTER_LINEAR))
    for g in GROUPS:
        out[f"{g}_max"] = mx[g]
    return out


def absent(prob_max):
    w = PARAMS["absent_window_px"]
    return ndimage.maximum_filter(prob_max, size=w) <= PARAMS["ABSENT"]


def roof_remap(arr, fdy, fdx, nearest=False):
    """Sample a 2013 array at q - d(q): moves 2013 roofs into the 2023 roof frame."""
    h, w = arr.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = (xx - fdx[None, :]).astype(np.float32)
    map_y = (yy - fdy[None, :]).astype(np.float32)
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.remap(arr.astype(np.float32), map_x, map_y, interpolation=interp, borderMode=cv2.BORDER_REPLICATE)


def components(mask, min_m2):
    lab, n = ndimage.label(mask, structure=segment_v2.FOUR)
    if n == 0:
        return lab
    sizes = np.bincount(lab.ravel())
    keep = sizes * PX >= min_m2
    keep[0] = False
    return np.where(keep[lab], lab, 0)


# ----------------------------------------------------------------------------- detection
def detect(ev13, ev23, lab13, n13, lab23, n23, fdy, fdx, img13=None, img23=None):
    """Return change_map (codes), per-node tables for both years, and change regions."""
    P = PARAMS
    is_b13 = np.isin(lab13, n13.index[n13.object_type == "building"])
    is_b23 = np.isin(lab23, n23.index[n23.object_type == "building"])
    # roof frame
    b13_2x_al = roof_remap(ev13["building_2x"], fdy, fdx)
    b13_max_al = roof_remap(ev13["building_max"], fdy, fdx)
    lab13_al = roof_remap(lab13, fdy, fdx, nearest=True).astype(np.int64)
    is_b13_al = np.isin(lab13_al, n13.index[n13.object_type == "building"])

    # deep shadow carries no evidence: a dark 2013 pixel cannot prove a building was absent, etc.
    dark13 = dark13_al = dark23 = np.zeros(lab23.shape, bool)
    if img13 is not None:
        up = lambda im: cv2.resize(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), lab23.shape[::-1], interpolation=cv2.INTER_LINEAR)
        dark13 = up(img13) < P["shadow_gray"]
        dark13_al = roof_remap(dark13.astype(np.float32), fdy, fdx) > 0.5
        dark23 = up(img23) < P["shadow_gray"]
    gain = is_b23 & (ev23["building_2x"] >= P["PRESENT"]) & absent(b13_max_al) & ~dark13_al
    loss = is_b13_al & (b13_2x_al >= P["PRESENT"]) & absent(ev23["building_max"]) & ~dark23
    gain = ndimage.binary_opening(gain, structure=segment_v2.FOUR)
    loss = ndimage.binary_opening(loss, structure=segment_v2.FOUR)

    change = np.zeros(lab23.shape, np.uint8)
    rows23, rows13 = [], []
    # --- 2023 buildings: new / extension
    for nid in n23.index[n23.object_type == "building"]:
        m = lab23 == nid
        area = m.sum() * PX
        g = (gain & m).sum() * PX
        status = "unchanged"
        if g >= P["object_fraction_new"] * area and area >= P["building_min_m2"]:
            status = "new_building"
            change[m] = CODES["new_building"]
        elif g >= P["extension_min_m2"]:
            status = "extension"
            change[gain & m] = CODES["extension"]
        node = n23.loc[nid]
        reliable = node.area_m2 >= P["reliable_min_m2"] and node.mean_conf >= P["reliable_min_conf"]
        need = P["new_building_high_m2"] if status == "new_building" else P["extension_high_m2"]
        confidence = "" if status == "unchanged" else ("high" if reliable and g >= need else "low")
        rows23.append({"node_id": int(nid), "object_type": "building", "change": status, "confidence": confidence,
                       "new_area_m2": round(g, 1), "area_m2": round(area, 1),
                       "p2013_building_max": round(float(b13_max_al[m].mean()), 3),
                       "p2023_building": round(float(ev23["building_2x"][m].mean()), 3),
                       "reliability": "high" if reliable else "low"})
    # --- 2013 buildings: demolished (evaluated in the roof frame)
    for nid in n13.index[n13.object_type == "building"]:
        m = lab13_al == nid
        if not m.any():
            continue
        area = m.sum() * PX
        l = (loss & m).sum() * PX
        status = "unchanged"
        if l >= P["object_fraction_new"] * area and area >= P["building_min_m2"]:
            status = "demolished"
            change[m & (change == 0)] = CODES["demolished"]
        node = n13.loc[nid]
        reliable = node.area_m2 >= P["reliable_min_m2"] and node.mean_conf >= P["reliable_min_conf"]
        rows13.append({"node_id": int(nid), "object_type": "building", "change": status,
                       "confidence": "unverified" if status != "unchanged" else "",
                       "lost_area_m2": round(l, 1), "area_m2": round(area, 1),
                       "p2013_building": round(float(b13_2x_al[m].mean()), 3),
                       "p2023_building_max": round(float(ev23["building_max"][m].mean()), 3),
                       "reliability": "high" if reliable else "low"})

    # --- ground classes (ground frame), away from buildings of either year
    near_b = ndimage.binary_dilation(is_b13 | is_b23 | is_b13_al, iterations=P["building_buffer_px"]) | dark23 | dark13
    smooth = lambda a: ndimage.uniform_filter(a, size=5)
    t13, t23 = smooth(ev13["tree_2x"]), smooth(ev23["tree_2x"])
    tree_loss = components((t13 >= 0.6) & absent(ev23["tree_max"]) & ~near_b, P["tree_min_m2"])
    tree_gain = components((t23 >= 0.6) & absent(ev13["tree_max"]) & ~near_b, P["tree_min_m2"])
    r13, r23 = smooth(ev13["road_2x"]), smooth(ev23["road_2x"])
    new_road = components((r23 >= 0.6) & absent(ev13["road_max"]) & ~is_b23, P["road_min_m2"])
    for code, lab in ((CODES["tree_loss"], tree_loss), (CODES["tree_gain"], tree_gain), (CODES["new_road"], new_road)):
        change[(lab > 0) & (change == 0)] = code
    return change, pd.DataFrame(rows23), pd.DataFrame(rows13)


def regions(change, lab13, lab23, n13, n23, nodes23=None):
    """Vectorize change_map into change regions with node attribution (both years)."""
    with rasterio.open(D / "2023" / "object_ids.tif") as s:
        transform, crs = s.transform, s.crs
    inv = {v: k for k, v in CODES.items()}
    region = np.zeros(change.shape, np.int32)
    meta = {}
    nxt = 1
    for code in np.unique(change[change > 0]):
        lab, n = ndimage.label(change == code, structure=segment_v2.FOUR)
        region[lab > 0] = lab[lab > 0] + nxt - 1
        for i in range(n):
            meta[nxt + i] = inv[int(code)]
        nxt += n
    rows, geoms = [], {}
    conf_lookup = dict(zip(nodes23.node_id, nodes23.confidence.fillna(""))) if nodes23 is not None else {}
    for gj, v in features.shapes(region, mask=region > 0, transform=transform, connectivity=4):
        v = int(v); g = make_valid(shape(gj))
        geoms[v] = g if v not in geoms else make_valid(geoms[v].union(g))
    for rid, sl in enumerate(ndimage.find_objects(region), 1):
        if sl is None:
            continue
        m = region[sl] == rid
        ids23, c23 = np.unique(lab23[sl][m], return_counts=True)
        ids13, c13 = np.unique(lab13[sl][m], return_counts=True)
        dom23 = int(ids23[np.argmax(c23)]); dom13 = int(ids13[np.argmax(c13)])
        ys, xs = np.nonzero(m)
        kind = meta[rid]
        if kind in UNVERIFIED:
            conf = "unverified"
        elif nodes23 is not None and dom23 in conf_lookup:
            conf = conf_lookup[dom23] or "low"
            if conf == "high" and m.sum() * PX < PARAMS["region_high_min_m2"]:
                conf = "low"
        else:
            conf = "low"
        rows.append({"region_id": rid, "change": kind, "confidence": conf, "area_m2": round(m.sum() * PX, 1),
                     "node_2023": dom23, "type_2023": n23.object_type.get(dom23, ""),
                     "node_2013": dom13, "type_2013": n13.object_type.get(dom13, ""),
                     "row": int(sl[0].start + ys.mean()), "col": int(sl[1].start + xs.mean())})
    df = pd.DataFrame(rows)
    return df, [geoms[r] for r in df.region_id], transform, crs


# ----------------------------------------------------------------------------- outputs
def write_outputs(change, nodes23, nodes13, reg, geoms, transform, crs, summary):
    OUT.mkdir(parents=True, exist_ok=True)
    prof = {"driver": "GTiff", "height": change.shape[0], "width": change.shape[1], "count": 1, "dtype": "uint8",
            "crs": crs, "transform": transform, "nodata": 0, "compress": "deflate", "tiled": True}
    with rasterio.open(OUT / "change_map.tif", "w", **prof) as dst:
        dst.write(change, 1)
        dst.update_tags(**{f"code_{v}": k for k, v in CODES.items()})
    high = np.zeros_like(change)
    inv = {v: k for k, v in CODES.items()}
    for code in np.unique(change[change > 0]):
        lab, _ = ndimage.label(change == code, structure=segment_v2.FOUR)
        for r in reg[(reg.change == inv[int(code)]) & (reg.confidence == "high")].itertuples():
            rid = lab[r.row, r.col]
            if rid:
                high[lab == rid] = code
    with rasterio.open(OUT / "change_map_high_confidence.tif", "w", **prof) as dst:
        dst.write(high, 1)
    nodes23.to_csv(OUT / "node_changes_2023.csv", index=False)
    nodes13.to_csv(OUT / "node_changes_2013.csv", index=False)
    reg.to_csv(OUT / "change_regions.csv", index=False)
    gpkg = OUT / "changes.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    ds = ogr.GetDriverByName("GPKG").CreateDataSource(str(gpkg))
    srs = osr.SpatialReference(); srs.ImportFromWkt(crs.to_wkt())
    layer = ds.CreateLayer("changes", srs, ogr.wkbUnknown)
    for name, kind in (("change", ogr.OFTString), ("confidence", ogr.OFTString), ("area_m2", ogr.OFTReal), ("node_2023", ogr.OFTInteger),
                       ("type_2023", ogr.OFTString), ("node_2013", ogr.OFTInteger), ("type_2013", ogr.OFTString)):
        layer.CreateField(ogr.FieldDefn(name, kind))
    layer.StartTransaction()
    for r, g in zip(reg.itertuples(), geoms):
        f = ogr.Feature(layer.GetLayerDefn())
        for k in ("change", "confidence", "area_m2", "node_2023", "type_2023", "node_2013", "type_2013"):
            f.SetField(k, getattr(r, k))
        f.SetGeometry(ogr.CreateGeometryFromWkb(g.wkb))
        layer.CreateFeature(f)
    layer.CommitTransaction()
    ds = None
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def summarize(change, nodes23, nodes13, reg, extent_km2):
    s = {"params": PARAMS, "area_km2": round(extent_km2, 3),
         "regions": {f"{a}|{b}": int(v) for (a, b), v in reg.groupby(["change", "confidence"]).size().items()} if len(reg) else {},
         "area_m2_by_change": reg.groupby("change").area_m2.sum().round(0).to_dict() if len(reg) else {}}
    s["buildings_2023"] = nodes23.groupby(["change", "confidence"]).size().rename(lambda x: str(x)).to_dict() if len(nodes23) else {}
    s["buildings_2013"] = nodes13.groupby(["change", "confidence"]).size().rename(lambda x: str(x)).to_dict() if len(nodes13) else {}
    s["buildings_2023"] = {str(k): int(v) for k, v in s["buildings_2023"].items()}
    s["buildings_2013"] = {str(k): int(v) for k, v in s["buildings_2013"].items()}
    return s


def load_year(year):
    with rasterio.open(D / year / "object_ids.tif") as s:
        lab = s.read(1).astype(np.int64)
    return lab, pd.read_csv(D / year / "nodes.csv").set_index("node_id")


def main():
    img23 = rasterio.open(D / "input_2023_harmonized.tif").read().transpose(1, 2, 0)
    img13 = rasterio.open(D / "input_2013_coregistered_harmonized.tif").read().transpose(1, 2, 0)
    lab13, n13 = load_year("2013")
    lab23, n23 = load_year("2023")
    b13 = np.isin(lab13, n13.index[n13.object_type == "building"])
    b23 = np.isin(lab23, n23.index[n23.object_type == "building"])
    fdy, fdx, _ = qa.roof_offset_field(b13, b23)
    ev13, ev23 = evidence(img13), evidence(img23)
    change, nodes23, nodes13 = detect(ev13, ev23, lab13, n13, lab23, n23, fdy, fdx, img13, img23)
    reg, geoms, transform, crs = regions(change, lab13, lab23, n13, n23, nodes23)
    extent = change.size * PX / 1e6
    summary = summarize(change, nodes23, nodes13, reg, extent)
    write_outputs(change, nodes23, nodes13, reg, geoms, transform, crs, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "params"}, indent=1))


if __name__ == "__main__":
    main()
