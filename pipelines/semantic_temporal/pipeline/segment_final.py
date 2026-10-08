"""Final object segmentation for the graph: every pixel belongs to exactly one segment (= one node).

Node classes
  1 building         validated v3 building step (2x components + radius-2 neck split)
  2 road             road network split into street segments at intersections
  3 tree             trees / forest
  4 grass_shrub      rangeland: grass, shrubs, sparse vegetation
  5 cropland         agriculture: fields, orchards, terraces under cultivation
  6 bare_soil        bare land / soil / rock
  7 paved_developed  developed open space: yards, plazas, parking, paved lots
  8 water            water bodies including pools

Pipeline (on the 2x / 0.75 m grid)
  1. OpenEarthMap SegFormer probabilities at 1x, 1.5x, 2x (segment_v2 / semantic_ms).
  2. Buildings: segment_v2.building_instances_simple (validated, unchanged).
  3. Roads and water from the unsmoothed multi-scale class map (thin features survive); roads are
     split at junctions of their skeleton, short spurs are absorbed.
  4. Other land: probabilities smoothed over ~5 m, class regions; confident Delineate Anything
     parcels (fields, terraces, lots) become segments first.
  5. Regions larger than `split_above_m2` are divided along image edges (marker watershed on the
     colour gradient, markers on a grid with ~`target_region_m2` spacing), so land nodes are
     parcel-sized instead of tile-sized.
  3b. Two RGB rules where the OpenEarthMap model is blind in this landscape (checked on all 7 tiles):
     - open water / pools: hue 80-100 (OpenCV), saturation >= 110, value >= 100. The model labels
       pools as building; blue roofs (hue ~105) are excluded.
     - bare soil: no vegetation (excess-green index), warm soil colour (redness >= 0.08), not shadow,
       smoothed over ~5 m; only where the model said grass/shrub, cropland or paved. The model never
       predicts its own bareland class on these tiles. In a dry-summer image dead grass and bare soil
       look alike in RGB, so this class means "bare / unvegetated ground".
  7. Final refinement (refine_merge.py): adjacent same-class segments that are one continuous real
     region are merged (no visible boundary, same appearance, broad contact, same semantics);
     separate real objects stay separate (buildings never merge, streets meet at real junctions).
  6. Fragments below the class minimum are merged into the neighbour with the longest shared
     boundary (same class preferred, never into a building unless enclosed). Every segment is
     one 4-connected region; every pixel is assigned (checked).
"""
from __future__ import annotations

from osgeo import gdal, ogr, osr  # noqa: F401  (GDAL DLLs before torch)
import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio import features
from scipy import ndimage
from shapely import make_valid
from shapely.geometry import shape

import refine_merge
import segment_v2 as v2
import semantic_ms as sm

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = Path(__file__).resolve().parent / "final_segmentation"
FOUR = v2.FOUR

CLASSES = {1: "building", 2: "road", 3: "tree", 4: "grass_shrub", 5: "cropland",
           6: "bare_soil", 7: "paved_developed", 8: "water"}
COLORS = {1: (230, 57, 70), 2: (90, 90, 100), 3: (27, 120, 55), 4: (150, 200, 90), 5: (238, 200, 70),
          6: (196, 150, 100), 7: (170, 140, 200), 8: (40, 120, 230)}
# OpenEarthMap index -> node class (0 = background, folded into the best other class)
OEM_TO_CLASS = {1: 6, 2: 4, 3: 7, 4: 2, 5: 3, 6: 8, 7: 5, 8: 1}
LAND = (3, 4, 5, 6, 7)

P = {
    "smooth_px": 7,                 # land-cover probability smoothing window (2x px, ~5 m)
    "road_min_m2": 100.0,
    "road_spur_max_m": 15.0,        # skeleton branches shorter than this (dead ends) are absorbed
    "water_min_m2": 10.0,           # keeps swimming pools
    "water_prob": 0.4,
    "pool_hue": (80, 100), "pool_sat": 110, "pool_val": 100, "pool_min_m2": 6.0,
    "bare_exg": 0.04, "bare_veg_frac": 0.05, "bare_redness": 0.08, "bare_val": 100, "bare_share": 0.5,
    "land_min_region_m2": 100.0,    # land class regions smaller than this are absorbed
    "parcel_conf": 0.20, "parcel_purity": 0.65, "parcel_min_m2": 150.0,
    "split_above_m2": 8000.0,       # land regions larger than this are split along image edges
    "snake_above_m2": 3000.0,       # ...and so are long thin regions above this size with
    "snake_compactness": 0.06,      #    compactness (4*pi*A/P^2) below this
    "target_region_m2": 3000.0,     # approximate size of split pieces
    "min_segment_m2": {1: 27.0, 2: 40.0, 3: 50.0, 4: 50.0, 5: 50.0, 6: 50.0, 7: 30.0, 8: 10.0},
}


# ----------------------------------------------------------------------------- semantics
def semantics(image):
    _, p2 = sm.infer(image, scale=2, return_upsampled=True)
    size = p2.shape[:0:-1]
    pms = p2.copy()
    for s in (1.0, 1.5):
        p = sm.infer(image, scale=s)
        pms += np.stack([cv2.resize(c, size, interpolation=cv2.INTER_LINEAR) for c in p])
    pms /= 3.0
    # class probabilities in node-class order (index = class id), background folded out
    cp = np.zeros((9, *p2.shape[1:]), np.float32)
    for oem, c in OEM_TO_CLASS.items():
        cp[c] += pms[oem]
    return p2, cp


# ----------------------------------------------------------------------------- RGB rules
def pool_water(image2x):
    hsv = cv2.cvtColor(image2x, cv2.COLOR_RGB2HSV)
    m = ((hsv[:, :, 0] >= P["pool_hue"][0]) & (hsv[:, :, 0] <= P["pool_hue"][1])
         & (hsv[:, :, 1] >= P["pool_sat"]) & (hsv[:, :, 2] >= P["pool_val"]))
    m = ndimage.binary_opening(m, structure=FOUR)
    return v2.remove_small(m, max(4, int(P["pool_min_m2"] / 0.5625)))


def bare_ground(image2x):
    x = image2x.astype(np.float32)
    exg = (2 * x[:, :, 1] - x[:, :, 0] - x[:, :, 2]) / (x.sum(2) + 1)
    veg_frac = ndimage.uniform_filter((exg > P["bare_exg"]).astype(np.float32), size=P["smooth_px"])
    red = (x[:, :, 0] - x[:, :, 2]) / (x[:, :, 0] + x[:, :, 2] + 1)
    raw = (veg_frac < P["bare_veg_frac"]) & (red >= P["bare_redness"]) & (x.max(2) >= P["bare_val"])
    return ndimage.uniform_filter(raw.astype(np.float32), size=P["smooth_px"]) >= P["bare_share"]


# ----------------------------------------------------------------------------- roads
def thin(mask):
    """Zhang-Suen thinning (vectorized)."""
    img = mask.astype(np.uint8).copy()
    while True:
        changed = False
        for step in (0, 1):
            Pd = np.pad(img, 1)
            p2, p3, p4, p5 = Pd[:-2, 1:-1], Pd[:-2, 2:], Pd[1:-1, 2:], Pd[2:, 2:]
            p6, p7, p8, p9 = Pd[2:, 1:-1], Pd[2:, :-2], Pd[1:-1, :-2], Pd[:-2, :-2]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            B = sum(x.astype(np.int16) for x in nb[:8])
            A = sum(((nb[i] == 0) & (nb[i + 1] == 1)).astype(np.int16) for i in range(8))
            c1 = p2 * p4 * p6 if step == 0 else p2 * p4 * p8
            c2 = p4 * p6 * p8 if step == 0 else p2 * p6 * p8
            m = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & (c1 == 0) & (c2 == 0)
            if m.any():
                img[m] = 0
                changed = True
        if not changed:
            return img.astype(bool)


def split_roads(road, px_m):
    """Street segments: skeleton branches between junctions; every road pixel joins its nearest branch."""
    skel = thin(road)
    Pd = np.pad(skel.astype(np.uint8), 1)
    nbs = [Pd[:-2, 1:-1], Pd[:-2, 2:], Pd[1:-1, 2:], Pd[2:, 2:], Pd[2:, 1:-1], Pd[2:, :-2], Pd[1:-1, :-2], Pd[:-2, :-2]]
    crossings = sum(((nbs[i] == 0) & (nbs[(i + 1) % 8] == 1)).astype(np.int16) for i in range(8))
    junction = skel & (crossings >= 3)
    width = ndimage.distance_transform_edt(road)
    # clear a disk around each junction so branches separate (radius = local half-width + 1)
    jzone = np.zeros_like(road)
    for y, x in zip(*np.nonzero(junction)):
        r = int(width[y, x]) + 1
        jzone[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = True
    branches, n = ndimage.label(skel & ~jzone, structure=np.ones((3, 3), bool))
    if n == 0:
        return ndimage.label(road, structure=FOUR)[0]
    # absorb short dead-end spurs (they are road widenings, not streets)
    lengths = np.bincount(branches.ravel()) * px_m
    touches_j = np.zeros(n + 1, bool)
    near_j = ndimage.binary_dilation(jzone, iterations=2)
    touches_j[np.unique(branches[near_j])] = True
    spur = (lengths < P["road_spur_max_m"]) & touches_j
    spur[0] = False
    if (~spur[1:]).any():  # never remove every branch of a network
        branches[spur[branches]] = 0
    # each road pixel -> nearest branch pixel
    _, idx = ndimage.distance_transform_edt(branches == 0, return_indices=True)
    seg = branches[idx[0], idx[1]]
    seg[~road] = 0
    return v2.relabel_sequential(seg)


# ----------------------------------------------------------------------------- land regions
def gradient_image(image2x):
    lab = cv2.cvtColor(cv2.GaussianBlur(image2x, (0, 0), 1.5), cv2.COLOR_RGB2LAB).astype(np.float32)
    g = np.zeros(image2x.shape[:2], np.float32)
    for c in range(3):
        g += ndimage.sobel(lab[:, :, c], 0) ** 2 + ndimage.sobel(lab[:, :, c], 1) ** 2
    return np.sqrt(g)


def split_large(region, grad, px_area):
    """Split one large region along image edges into ~target-sized pieces."""
    target_px = P["target_region_m2"] / px_area
    S = max(8, int(math.sqrt(target_px)))
    ys, xs = np.nonzero(region)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    reg = region[y0:y1, x0:x1]
    dist = ndimage.distance_transform_edt(reg)
    markers = np.zeros(reg.shape, np.int32)
    k = 0
    for yy in range(S // 2, reg.shape[0], S):
        for xx in range(S // 2, reg.shape[1], S):
            # snap the grid point to the deepest interior pixel nearby
            sy0, sy1 = max(0, yy - S // 3), min(reg.shape[0], yy + S // 3 + 1)
            sx0, sx1 = max(0, xx - S // 3), min(reg.shape[1], xx + S // 3 + 1)
            win = dist[sy0:sy1, sx0:sx1]
            if win.max() < 1.5:
                continue
            dy, dx = np.unravel_index(np.argmax(win), win.shape)
            k += 1
            markers[sy0 + dy, sx0 + dx] = k
    if k < 2:
        return None
    g = grad[y0:y1, x0:x1]
    elev = np.clip(g / max(np.percentile(g[reg], 95), 1e-6), 0, 1)
    elev = (elev * 60000).astype(np.uint16)
    elev[~reg] = 65535
    ws = ndimage.watershed_ift(elev, ndimage.grey_dilation(markers, size=3) * reg, structure=FOUR).astype(np.int32)
    ws[~reg] = 0
    out = np.zeros(region.shape, np.int32)
    out[y0:y1, x0:x1] = ws
    return out


# ----------------------------------------------------------------------------- cleanup
def shared_boundaries(labels):
    counts = Counter()
    for a, b, *_ in v2.pair_boundaries(labels):
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        u, c = np.unique(lo.astype(np.int64) * 10_000_000 + hi, return_counts=True)
        for kk, cc in zip(u, c):
            counts[(int(kk // 10_000_000), int(kk % 10_000_000))] += int(cc)
    return counts


def merge_small(labels, cls, px_area, max_iter=8):
    """Merge segments below their class minimum into the neighbour with the longest shared
    boundary (same class preferred; buildings only absorb fully enclosed specks)."""
    for _ in range(max_iter):
        areas = np.bincount(labels.ravel()) * px_area
        small = [i for i in range(1, len(areas)) if areas[i] > 0 and areas[i] < P["min_segment_m2"][cls[i]]]
        if not small:
            break
        nb = defaultdict(list)
        for (a, b), n in shared_boundaries(labels).items():
            nb[a].append((b, n)); nb[b].append((a, n))
        lut = np.arange(len(areas))
        for i in sorted(small, key=lambda j: areas[j]):
            cand = [(b, n) for b, n in nb[i] if lut[b] == b]
            if not cand:
                continue
            non_b = [(b, n) for b, n in cand if cls[b] != 1]
            pool = non_b if (non_b and cls[i] != 1) else cand
            same = [(b, n) for b, n in pool if cls[b] == cls[i]]
            target = max(same or pool, key=lambda t: t[1])[0]
            lut[i] = target
        # resolve chains
        for i in range(len(lut)):
            j = i
            while lut[j] != j:
                j = lut[j]
            lut[i] = j
        labels = lut[labels]
    return labels


# ----------------------------------------------------------------------------- main segmentation
def segment(image, px_area=0.5625, log=None):
    log = {} if log is None else log
    t0 = time.time()
    h, w = image.shape[:2]
    image2x = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    p2, cp = semantics(image)
    px_m = math.sqrt(px_area)

    # 1 buildings (validated step)
    bld = v2.building_instances_simple(p2, px_area)
    is_b = bld > 0

    # 2 roads and water from the unsmoothed class map
    raw = cp.argmax(0)
    raw[raw == 0] = 6
    road = ndimage.binary_closing(raw == 2, structure=np.ones((5, 5), bool)) & ~is_b
    road = v2.remove_small(road, int(P["road_min_m2"] / px_area))
    roads = split_roads(road, px_m)
    pools = pool_water(image2x)
    bld[pools] = 0                      # the model buries pools inside building masks
    is_b = bld > 0
    road &= ~pools
    water = (((raw == 8) & (cp[8] >= P["water_prob"])) | pools) & ~is_b & ~road
    water = v2.remove_small(water, max(4, int(P["water_min_m2"] / px_area)))
    log["pool_pixels"] = int(pools.sum())
    waters = ndimage.label(water, structure=FOUR)[0]

    # 3 land classes from smoothed probabilities
    sp = np.stack([ndimage.uniform_filter(cp[c], size=P["smooth_px"]) for c in range(9)])
    land_cls = np.array(LAND)[np.argmax(sp[list(LAND)], axis=0)]
    bare = bare_ground(image2x) & np.isin(land_cls, (4, 5, 7))
    land_cls[bare] = 6
    log["bare_rule_pixels"] = int(bare.sum())
    free = ~is_b & ~road & ~water

    labels = np.zeros((2 * h, 2 * w), np.int64)
    cls = [0]
    src = [""]

    def add(mask, c, source):
        labels[mask] = len(cls)
        cls.append(c); src.append(source)

    for i in range(1, bld.max() + 1):
        m = bld == i
        if m.any():
            add(m, 1, "building_v3")
    for i in range(1, roads.max() + 1):
        m = roads == i
        if m.any():
            add(m, 2, "road_segment")
    for i in range(1, waters.max() + 1):
        add(waters == i, 8, "water")

    # 3a confident DelAny parcels
    pm, pc = v2.delany_parcel_candidates(image, image2x, P["parcel_conf"])
    n_parcels = 0
    for k in np.argsort([m.sum() for m in pm]):
        m = v2.largest_component(pm[k] & free & (labels == 0))
        if m.sum() * px_area < P["parcel_min_m2"]:
            continue
        cnt = np.bincount(land_cls[m], minlength=9)
        c = int(cnt.argmax())
        if cnt[c] / cnt.sum() < P["parcel_purity"]:
            continue
        add(m, c, "delany_parcel"); n_parcels += 1

    # 3b class regions
    remaining = free & (labels == 0)
    for c in LAND:
        comp, n = ndimage.label(remaining & (land_cls == c), structure=FOUR)
        sizes = np.bincount(comp.ravel()) * px_area
        for i in np.flatnonzero(sizes >= P["land_min_region_m2"]):
            if i:
                add(comp == i, c, "class_region")
    # 3c leftovers grow from neighbouring non-building segments, then from anything
    for allow_building in (False, True):
        for _ in range(200):
            un = labels == 0
            if not un.any():
                break
            src_lab = labels if allow_building else np.where(np.isin(labels, np.flatnonzero(np.array(cls) == 1)), 0, labels)
            dil = ndimage.grey_dilation(src_lab, footprint=FOUR)
            take = un & (dil > 0)
            if not take.any():
                break
            labels[take] = dil[take]
    log["unassigned_after_growth"] = int((labels == 0).sum())

    # 4 split large land regions along image edges
    grad = gradient_image(image2x)
    areas = np.bincount(labels.ravel()) * px_area
    perim = np.zeros(len(areas))
    for a, b, *_ in v2.pair_boundaries(labels):
        perim += np.bincount(a, minlength=len(areas))[:len(areas)] + np.bincount(b, minlength=len(areas))[:len(areas)]
    perim *= px_m
    compact = 4 * np.pi * areas / np.maximum(perim, 1e-6) ** 2
    n_split = 0
    for i in range(1, len(cls)):
        big = areas[i] > P["split_above_m2"]
        snake = areas[i] > P["snake_above_m2"] and compact[i] < P["snake_compactness"]
        if cls[i] in LAND and (big or snake):
            parts = split_large(labels == i, grad, px_area)
            if parts is None:
                continue
            for j in range(2, parts.max() + 1):
                m = parts == j
                if m.any():
                    labels[m] = len(cls); cls.append(cls[i]); src.append(src[i] + "+edge_split")
            n_split += 1
    log["regions_split"] = n_split

    # 5 cleanup: connectivity, small fragments, connectivity again
    labels = v2.enforce_connectivity(labels.astype(np.int32), min_px=1).astype(np.int64)
    cls_arr = np.array(cls)
    labels = merge_small(labels, cls_arr, px_area)
    labels = v2.enforce_connectivity(labels.astype(np.int32), min_px=1).astype(np.int64)
    # relabel sequentially and carry classes/sources
    ids = np.unique(labels)
    ids = ids[ids > 0]
    lut = np.zeros(labels.max() + 1, np.int64)
    lut[ids] = np.arange(1, len(ids) + 1)
    final = lut[labels].astype(np.int32)
    rule_support = np.zeros(labels.shape, np.float32)
    rule_support[pools] = 1.0
    rule_support[bare & ~pools] = 1.0
    seg_cls = {int(lut[i]): int(cls_arr[i]) for i in ids}
    seg_src = {int(lut[i]): src[i] for i in ids}
    # 7 final refinement: merge same-class neighbours that are one continuous real region
    pre_merge = final.copy()
    final, seg_cls, seg_src, merge_log = refine_merge.refine(final.astype(np.int64), seg_cls, seg_src, image2x, cp, grad, px_m)
    log["segments_before_merge"] = int(pre_merge.max())
    log["merges"] = int((merge_log.decision == "merge").sum()) if len(merge_log) else 0
    log["merges_by_class"] = {CLASSES[int(k)] if str(k).isdigit() else str(k): int(v)
                              for k, v in merge_log[merge_log.decision == "merge"]["class"].value_counts().items()} if len(merge_log) else {}
    log.update({"delany_parcels": n_parcels, "segments": int(final.max()),
                "unassigned_pixels": int((final == 0).sum()), "seconds": round(time.time() - t0, 1)})
    return final, seg_cls, seg_src, cp, image2x, rule_support, pre_merge, merge_log


# ----------------------------------------------------------------------------- outputs
def node_table(final, seg_cls, seg_src, cp, image2x, transform, tile, year, rule_support=None):
    geoms = {}
    for gj, v in features.shapes(final, mask=final > 0, transform=transform, connectivity=4):
        v = int(v); g = make_valid(shape(gj))
        geoms[v] = g if v not in geoms else make_valid(geoms[v].union(g))
    hsv = cv2.cvtColor(image2x, cv2.COLOR_RGB2HSV)
    rows = []
    for nid, sl in enumerate(ndimage.find_objects(final), 1):
        if sl is None:
            continue
        m = final[sl] == nid
        g = geoms[nid]
        rgb = image2x[sl][m].astype(np.float32)
        hv = hsv[sl][m].astype(np.float32)
        probs = cp[:, sl[0], sl[1]][:, m].mean(1)
        c = g.centroid; b = g.bounds
        row = {"node_id": nid, "node_key": f"{year}:{tile}:{nid}", "class_id": seg_cls[nid], "class": CLASSES[seg_cls[nid]],
               "source": seg_src[nid], "area_m2": round(g.area, 2), "geom_parts": 1 if g.geom_type == "Polygon" else len(g.geoms),
               **v2.shape_metrics(g),
               "centroid_x": round(c.x, 2), "centroid_y": round(c.y, 2),
               "bbox_minx": round(b[0], 2), "bbox_miny": round(b[1], 2), "bbox_maxx": round(b[2], 2), "bbox_maxy": round(b[3], 2),
               "mean_r": round(float(rgb[:, 0].mean()), 2), "mean_g": round(float(rgb[:, 1].mean()), 2), "mean_b": round(float(rgb[:, 2].mean()), 2),
               "std_r": round(float(rgb[:, 0].std()), 2), "std_g": round(float(rgb[:, 1].std()), 2), "std_b": round(float(rgb[:, 2].std()), 2),
               "mean_h": round(float(hv[:, 0].mean()), 2), "mean_s": round(float(hv[:, 1].mean()), 2), "mean_v": round(float(hv[:, 2].mean()), 2)}
        for cid, name in CLASSES.items():
            row[f"p_{name}"] = round(float(probs[cid]), 4)
        if seg_cls[nid] in (6, 8) and rule_support is not None:
            # rule-based classes: the model gives them ~0 probability, so report rule support
            row["class_confidence"] = round(float(rule_support[sl][m].mean()), 4)
            row["source"] = row["source"] + "+rgb_rule"
        else:
            row["class_confidence"] = row[f"p_{CLASSES[seg_cls[nid]]}"]
        rows.append(row)
    return pd.DataFrame(rows), geoms


def edge_table(final, nodes, px_m):
    cent = nodes.set_index("node_id")[["centroid_x", "centroid_y"]]
    cl = nodes.set_index("node_id")["class"]
    rows = []
    for (a, b), n in sorted(shared_boundaries(final).items()):
        rows.append({"source": a, "target": b, "relation": "adjacent",
                     "class_pair": "|".join(sorted((cl[a], cl[b]))),
                     "shared_boundary_m": round(n * px_m, 2),
                     "centroid_distance_m": round(math.dist(cent.loc[a], cent.loc[b]), 2)})
    return pd.DataFrame(rows)


def write_outputs(out, final, nodes, edges, geoms, transform, crs, log):
    out.mkdir(parents=True, exist_ok=True)
    prof = {"driver": "GTiff", "height": final.shape[0], "width": final.shape[1], "count": 1, "dtype": "int32",
            "crs": crs, "transform": transform, "nodata": 0, "compress": "deflate", "tiled": True}
    with rasterio.open(out / "segment_ids.tif", "w", **prof) as d:
        d.write(final, 1)
    cls_map = np.zeros(final.shape, np.uint8)
    lut = np.zeros(final.max() + 1, np.uint8); lut[nodes.node_id] = nodes.class_id
    cls_map = lut[final]
    prof.update(dtype="uint8")
    with rasterio.open(out / "segment_classes.tif", "w", **prof) as d:
        d.write(cls_map, 1)
        d.write_colormap(1, {**{0: (0, 0, 0, 0)}, **{k: (*v, 255) for k, v in COLORS.items()}})
    nodes.to_csv(out / "nodes.csv", index=False)
    edges.to_csv(out / "edges.csv", index=False)
    gpkg = out / "segments.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    ds = ogr.GetDriverByName("GPKG").CreateDataSource(str(gpkg))
    srs = osr.SpatialReference(); srs.ImportFromWkt(crs.to_wkt())
    layer = ds.CreateLayer("segments", srs, ogr.wkbPolygon)
    fields = [("node_id", ogr.OFTInteger), ("node_key", ogr.OFTString), ("class", ogr.OFTString),
              ("source", ogr.OFTString), ("area_m2", ogr.OFTReal), ("class_conf", ogr.OFTReal)]
    for name, kind in fields:
        layer.CreateField(ogr.FieldDefn(name, kind))
    layer.StartTransaction()
    for r in nodes.to_dict("records"):
        f = ogr.Feature(layer.GetLayerDefn())
        f.SetField("node_id", int(r["node_id"])); f.SetField("node_key", r["node_key"]); f.SetField("class", r["class"])
        f.SetField("source", r["source"]); f.SetField("area_m2", float(r["area_m2"])); f.SetField("class_conf", float(r["class_confidence"]))
        f.SetGeometry(ogr.CreateGeometryFromWkb(geoms[r["node_id"]].wkb))
        layer.CreateFeature(f)
    layer.CommitTransaction()
    ds = None
    (out / "run_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tile", default="tile_00000")
    ap.add_argument("--image", type=Path, default=None, help="GeoTIFF (default: prepared_input/2013/<tile>.tif, which is 2023 imagery)")
    ap.add_argument("--year", default="2023")
    args = ap.parse_args()
    path = args.image or ROOT / "prepared_input" / "2013" / f"{args.tile}.tif"
    with rasterio.open(path) as s:
        image = s.read([1, 2, 3]).transpose(1, 2, 0).astype(np.uint8)
        transform, crs = s.transform, s.crs
    tr2 = transform * Affine.scale(0.5)
    log = {"tile": args.tile, "year": args.year, "image": str(path), "params": P}
    final, seg_cls, seg_src, cp, image2x, rule_support, pre_merge, merge_log = segment(image, abs(tr2.a * tr2.e), log)
    nodes, geoms = node_table(final, seg_cls, seg_src, cp, image2x, tr2, args.tile, args.year, rule_support)
    edges = edge_table(final, nodes, abs(tr2.a))
    log["edges"] = len(edges)
    log["segments_by_class"] = nodes["class"].value_counts().to_dict()
    log["multi_part_segments"] = int((nodes.geom_parts > 1).sum())
    out = OUT_ROOT / f"{args.year}_{args.tile}"
    write_outputs(out, final, nodes, edges, geoms, tr2, crs, log)
    merge_log["class"] = [CLASSES.get(c, c) if isinstance(c, (int, np.integer)) else c for c in merge_log["class"]]
    merge_log.to_csv(out / "merge_log.csv", index=False)
    with rasterio.open(out / "segment_ids.tif") as src_ds:
        prof = src_ds.profile
    with rasterio.open(out / "segment_ids_before_merge.tif", "w", **prof) as d:
        d.write(pre_merge.astype(np.int32), 1)
    print(json.dumps({k: v for k, v in log.items() if k != "params"}, indent=1))


if __name__ == "__main__":
    main()
