"""Segmentation v2: scale-corrected, instance-first object segmentation.

Why a v2
--------
The v1 chain (semantic_hybrid -> object_first -> topology -> SAM2 split) ran both
models at the native 1.5 m GSD:

* SegFormer/OpenEarthMap was trained at 0.25-0.5 m and the HF processor forced
  every tile to 512x512, so gaps between neighbouring houses disappeared and
  building masks fused into multi-house blobs. Later stages tried to cut those
  blobs apart after the fact.
* Delineate Anything at 1x found only a few coarse polygons on these tiles, and
  the 2500 m2 minimum-area filter then removed almost all of them.

v2 works on a 2x upsampled grid (0.75 m):

1. SegFormer, sliding 512 windows, 4-flip TTA -> class probabilities.
2. Delineate Anything at 2x -> raw instance masks (roofs, lots, terraces).
3. Buildings: building mask from semantics, split by marker-controlled watershed.
   Markers = DelAny instances that are mostly building, plus distance-transform
   peaks where no DelAny marker exists. Splits with weak evidence are merged back.
4. Roads: connected semantic road network.
5. Land cover: DelAny parcels that are class-pure, then smoothed class regions
   (tree / low vegetation / bare-developed / water). Leftover pixels are absorbed
   by region growing from adjacent context objects (never by buildings).
6. Every object is one 4-connected region; graph nodes/edges are written.

All outputs go to segmentation_v2/outputs/<tile>; inputs are read-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import paths  # package: input/checkpoint locations

from osgeo import gdal, ogr, osr  # must load before torch (Windows DLL order)
import cv2
import numpy as np
import rasterio
import torch
from affine import Affine
from rasterio import features
from scipy import ndimage
from shapely import make_valid
from shapely.geometry import shape

import semantic_ms

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "prepared_input" / "2013"
METADATA = ROOT / "2013" / "metadata.csv"
DELANY_CKPT = paths.DELANY_CKPT   # package: checkpoint location from paths.py
OUT_ROOT = ROOT / "segmentation_v2" / "outputs"

SCALE = 2
FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
EIGHT = np.ones((3, 3), dtype=bool)

# OpenEarthMap classes -> object groups
SEM_BUILDING, SEM_ROAD, SEM_WATER = 8, 4, 6
GROUP_OF_SEMANTIC = {0: "bare", 1: "bare", 3: "bare", 2: "low_vegetation", 7: "low_vegetation",
                     5: "tree", 6: "water", 4: "road", 8: "building"}
LANDCOVER_GROUPS = ["bare", "low_vegetation", "tree", "water"]
TYPE_COLORS = {
    "building": (0, 238, 255), "road": (255, 70, 45), "tree": (40, 170, 70),
    "low_vegetation": (150, 230, 80), "bare": (255, 190, 40), "water": (50, 120, 255),
}

# Thresholds expressed in native-pixel units, converted to the 2x grid below.
P = {
    # "simple2x" (default): 2x semantic mask -> connected components. Validated as the most
    # accurate (OSM F1) and most stable under year-to-year perturbations.
    # "watershed": DelAny/peak-marker watershed split (fewer merges, but markers flicker between
    # acquisitions and repartition houses -> 1.5x more false changes). Kept for comparison only.
    "building_mode": "simple2x",
    "min_building_m2": 27.0,
    "neck_split_radius_px": 2,       # simple2x: split necks narrower than 2r px (0 = off)
    "neck_min_core_m2": 20.0,        # ~12 native px; smaller blobs go to context
    "delany_conf": 0.10,
    "delany_building_frac": 0.55,   # DelAny instance counts as a roof marker
    "building_prob_2x": 0.40,       # 2x building probability defining roof shape
    "ms_growth_px2x": 1,            # max growth of 2x roofs into the multi-scale mask
    "roof_support_prob": 0.35,     # mean max(1x,2x) building prob for promoting a DelAny roof
    "roof_max_m2": 1500.0,          # larger DelAny instances are parcels, not roofs
    "delany_parcel_purity": 0.65,   # DelAny instance counts as a land parcel
    "delany_parcel_min_m2": 150.0,
    "delany_parcel_conf": 0.20,     # low-confidence DelAny parcels overlap and are box-clipped
    "landcover_min_m2": 300.0,      # smaller land-cover pieces are absorbed
    "road_min_m2": 150.0,
    "peak_min_depth_px2x": 2.5,     # EDT depth (2x px) for a building peak marker
    "merge_contrast": 14.0,         # RGB contrast below which a split is "weak"
    "merge_gap_prob": 0.80,         # mean building prob on boundary above which no gap is visible
    "merge_union_solidity": 0.88,   # weak split is merged only if union stays compact
}


# ----------------------------------------------------------------------------- io
def read_tile(tile: str):
    with rasterio.open(INPUT_DIR / f"{tile}.tif") as src:
        image = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.uint8)
        transform, crs = src.transform, src.crs
    return image, transform, crs


def tile_year(tile: str) -> str:
    with METADATA.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["tile_id"] == tile:
                return row["year"]
    return "unknown"


# ------------------------------------------------------------------- inference
_delany = None


def delany_instances(image: np.ndarray, out_size: int | None = None, conf: float | None = None):
    """Raw Delineate Anything instance masks for `image` (bool masks, confidences), optionally
    nearest-resized to out_size x out_size (to put 1x masks on the 2x grid)."""
    global _delany
    if _delany is None:
        os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
        from ultralytics import YOLO
        _delany = YOLO(str(DELANY_CKPT))
    result = _delany.predict(image[:, :, ::-1], conf=P["delany_conf"] if conf is None else conf, imgsz=image.shape[0],
                             retina_masks=True, verbose=False, half=False, max_det=1500,
                             device=0 if torch.cuda.is_available() else "cpu")[0]
    if result.masks is None:
        size = out_size or image.shape[0]
        return np.zeros((0, size, size), bool), np.zeros(0)
    masks = result.masks.data.cpu().numpy() > 0.5
    if out_size and out_size != masks.shape[1]:
        masks = np.stack([cv2.resize(m.astype(np.uint8), (out_size, out_size), interpolation=cv2.INTER_NEAREST) > 0 for m in masks])
    conf = result.boxes.conf.cpu().numpy()
    torch.cuda.empty_cache()
    return masks, conf


def delany_parcel_candidates(image, image2x, conf, window=512, step=462):
    """DelAny parcel candidates for an image of any size, on the 2x grid.

    Runs DelAny at 2x and 1x in windows of `window` native px with `step` (= the shared tile grid,
    so on a tile mosaic every window is exactly an original tile, as in validation). Masks below
    `conf`, or flat-clipped by their box on a side that is NOT the outer image border, are dropped
    per window; duplicates from overlapping windows are removed (IoU > 0.5, keep higher conf).
    For a single 512 px tile this is the whole-tile behaviour used during validation."""
    h, w = image.shape[:2]
    H, W = image2x.shape[:2]
    k = H // h
    masks, confs, win_ids = [], [], []
    starts = semantic_ms._window_starts
    win = 0
    for y0 in starts(h, window, step):
        for x0 in starts(w, window, step):
            ws = min(window, h - y0), min(window, w - x0)
            crop1 = image[y0:y0 + ws[0], x0:x0 + ws[1]]
            crop2 = image2x[k * y0:k * (y0 + ws[0]), k * x0:k * (x0 + ws[1])]
            exempt = (y0 == 0, y0 + ws[0] == h, x0 == 0, x0 + ws[1] == w)
            win += 1
            for m_set, c_set in (delany_instances(crop2, conf=conf),
                                 delany_instances(crop1, out_size=crop2.shape[0], conf=conf)):
                for m, c in zip(m_set, c_set):
                    if not m.any() or bbox_truncated(m, exempt=exempt):
                        continue
                    full = np.zeros((H, W), bool)
                    full[k * y0:k * y0 + m.shape[0], k * x0:k * x0 + m.shape[1]] = m
                    masks.append(full); confs.append(float(c)); win_ids.append(win)
    # duplicates between DIFFERENT windows only (within one window, 2x and 1x detections are both
    # kept and resolved by the parcel step, exactly as in the validated single-tile pipeline)
    keep = []
    for i in np.argsort(confs, kind="stable")[::-1]:
        dup = False
        for j in keep:
            if win_ids[i] == win_ids[j]:
                continue
            inter = np.logical_and(masks[i], masks[j]).sum()
            if inter and inter / np.logical_or(masks[i], masks[j]).sum() > 0.5:
                dup = True
                break
        if not dup:
            keep.append(i)
    keep.sort()
    if not keep:
        return np.zeros((0, H, W), bool), np.zeros(0)
    return np.stack([masks[i] for i in keep]), np.array([confs[i] for i in keep])


# ------------------------------------------------------------------- helpers
def remove_small(mask: np.ndarray, min_px: int, structure=FOUR) -> np.ndarray:
    lab, n = ndimage.label(mask, structure=structure)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


def solidity(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    area = float(mask.sum())
    hull = cv2.convexHull(np.vstack(contours))
    return area / max(cv2.contourArea(hull) + 0.5 * cv2.arcLength(hull, True) + 1.0, 1.0)


def pair_boundaries(labels: np.ndarray):
    """Yield (a, b, idx_a, idx_b) arrays for horizontally and vertically adjacent different labels."""
    for a, b, sl_a, sl_b in (
        (labels[:, :-1], labels[:, 1:], (slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        (labels[:-1, :], labels[1:, :], (slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ):
        diff = (a != b) & (a > 0) & (b > 0)
        yield a[diff], b[diff], sl_a, sl_b, diff


def boundary_evidence(labels, image_f, bprob):
    """Per adjacent pair: shared length, mean RGB contrast, mean min building-prob across the boundary."""
    stats = defaultdict(lambda: [0, 0.0, 0.0])
    for a, b, sl_a, sl_b, diff in pair_boundaries(labels):
        ca = image_f[sl_a][diff]
        cb = image_f[sl_b][diff]
        contrast = np.linalg.norm(ca - cb, axis=1)
        pb = np.minimum(bprob[sl_a][diff], bprob[sl_b][diff])
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        key = lo.astype(np.int64) * 10_000_000 + hi
        uniq, inv = np.unique(key, return_inverse=True)
        cnt = np.bincount(inv)
        csum = np.bincount(inv, weights=contrast)
        psum = np.bincount(inv, weights=pb)
        for k, n, c, p in zip(uniq, cnt, csum, psum):
            s = stats[(int(k // 10_000_000), int(k % 10_000_000))]
            s[0] += int(n); s[1] += float(c); s[2] += float(p)
    return {k: (n, c / n, p / n) for k, (n, c, p) in stats.items()}


def relabel_sequential(labels):
    values = np.unique(labels)
    values = values[values > 0]
    lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    lut[values] = np.arange(1, len(values) + 1)
    return lut[labels]


# ------------------------------------------------------------------- buildings
def building_instances_simple(prob2x, px_area2x):
    """Buildings = 4-connected components of the 2x building mask (argmax and p >= 0.4, opened,
    components smaller than min_building_m2 dropped). Deliberately no instance splitting: this is
    the variant that validated as most stable for change detection."""
    mask = (prob2x.argmax(0) == SEM_BUILDING) & (prob2x[SEM_BUILDING] >= P["building_prob_2x"])
    mask = ndimage.binary_opening(mask, structure=FOUR)
    mask = remove_small(mask, int(round(P["min_building_m2"] / px_area2x)))
    return neck_split(mask, P["neck_split_radius_px"], px_area2x)


def neck_split(mask, r, px_area2x):
    """Split buildings joined by a neck narrower than 2r px, using only the mask's own geometry
    (erode by a disk, keep cores >= neck_min_core_m2, partition the mask among cores by watershed on
    the distance transform). No external markers, so it does not flicker between acquisitions.
    Validated: r=2 lowers OSM merges 19% -> 16% (held-out) with unchanged stability; r>=3 costs
    stability."""
    comps = ndimage.label(mask, structure=FOUR)[0]
    if r == 0:
        return comps.astype(np.int32)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    cores = cv2.erode(mask.astype(np.uint8), disk).astype(bool)
    cores = remove_small(cores, int(P["neck_min_core_m2"] / px_area2x))
    markers = ndimage.label(cores, structure=FOUR)[0]
    n = markers.max()
    no_core = mask & ~np.isin(comps, np.unique(comps[cores]))  # never delete a building
    extra = ndimage.label(no_core, structure=FOUR)[0]
    markers[extra > 0] = extra[extra > 0] + n
    dist = ndimage.distance_transform_edt(mask)
    elev = (65000 - np.clip(dist * 2000, 0, 64000)).astype(np.uint16)
    elev[~mask] = 65535
    ws = ndimage.watershed_ift(elev, markers.astype(np.int32), structure=FOUR).astype(np.int32)
    ws[~mask] = 0
    return enforce_connectivity(ws, min_px=int(round(P["min_building_m2"] / px_area2x)), allow_drop=False)


def building_instances(image2x, prob_ms, prob2x, dl_masks, dl_conf, px_area2x, log):
    """Building instances on the 2x grid.

    prob_ms : multi-scale (1x, 1.5x, 2x) mean probabilities -> decides WHAT is building (recall).
    prob2x  : 2x-only probabilities -> decide roof SHAPE and WHERE buildings separate (gaps between
              roofs are only resolved at 2x).
    """
    argmax = prob_ms.argmax(0)
    bprob2x = prob2x[SEM_BUILDING]
    # Shape from 2x (gaps between roofs are open); multi-scale only adds recall:
    #  * whole multi-scale building components that 2x missed entirely, and
    #  * at most `ms_growth_px2x` px of growth of 2x roofs into the multi-scale mask
    #    (restores roof edges without bridging the gaps between neighbouring houses).
    base = (prob2x.argmax(0) == SEM_BUILDING) & (bprob2x >= P["building_prob_2x"])
    base = ndimage.binary_opening(base, structure=FOUR)
    ms = ndimage.binary_opening(argmax == SEM_BUILDING, structure=FOUR)
    ms_comp, _ = ndimage.label(ms, structure=FOUR)
    touched = np.unique(ms_comp[base])
    isolated = ms & ~np.isin(ms_comp, touched[touched > 0])
    grow = ms & ndimage.binary_dilation(base, structure=FOUR, iterations=P["ms_growth_px2x"])
    building = base | isolated | grow
    log["building_px_isolated_ms_added"] = int((isolated & ~base).sum())

    # Recall: flat light concrete roofs are under-detected by SegFormer. A DelAny instance is
    # promoted to a roof if multi-scale building probability supports it and it is not mostly
    # tree/road/water. DelAny supplies the shape, SegFormer the evidence.
    max_roof_px = int(P["roof_max_m2"] / px_area2x)
    min_px = int(round(P["min_building_m2"] / px_area2x))
    promoted = 0
    for k in np.argsort(-dl_conf):
        m = dl_masks[k]
        area = int(m.sum())
        if area < min_px or area > max_roof_px:
            continue
        if building[m].mean() >= P["delany_building_frac"]:
            continue  # already a building; becomes a marker below
        if prob_ms[SEM_BUILDING][m].mean() < P["roof_support_prob"]:
            continue
        if np.isin(argmax[m], (5, SEM_ROAD, SEM_WATER)).mean() > 0.35:
            continue
        add = m & ~building & np.isin(argmax, (0, 1, 3, SEM_BUILDING))
        if add.sum() + (m & building).sum() < P["delany_building_frac"] * area:
            continue
        building |= add
        promoted += 1
    log["building_roofs_promoted_from_delany"] = promoted
    # Fill tiny holes (courtyard noise) only.
    holes = ndimage.binary_fill_holes(building) & ~building
    building |= holes & ~remove_small(holes, int(20 / px_area2x))

    # Separation cores: where the 2x model is confident, gaps between adjacent roofs are open.
    core2x = building & (bprob2x >= 0.5)

    # --- markers from DelAny roofs
    markers = np.zeros(building.shape, np.int32)
    next_id = 1
    roof_marker_ids = set()
    for k in np.argsort(-dl_conf):
        m = dl_masks[k]
        area = int(m.sum())
        if area < min_px:
            continue
        inside = m & building
        if inside.sum() / area < P["delany_building_frac"]:
            continue
        core = ndimage.binary_erosion(inside, structure=EIGHT, iterations=2)
        if core.sum() < 6:
            core = ndimage.binary_erosion(inside, structure=FOUR, iterations=1)
        core &= markers == 0
        core = largest_component(core)
        if core.sum() < 4:
            continue
        markers[core] = next_id
        roof_marker_ids.add(next_id)
        next_id += 1

    # --- distance peaks of the 2x cores for roofs no DelAny marker explains
    dist = ndimage.distance_transform_edt(core2x)
    smooth = ndimage.gaussian_filter(dist, 1.0)
    peaks = (smooth == ndimage.maximum_filter(smooth, size=9)) & (dist >= P["peak_min_depth_px2x"])
    peaks &= ~(ndimage.distance_transform_edt(markers == 0) < 8)
    peak_lab, n_peaks = ndimage.label(ndimage.binary_dilation(peaks, structure=EIGHT), structure=EIGHT)
    for pid in range(1, n_peaks + 1):
        sel = (peak_lab == pid) & core2x & (markers == 0)
        if sel.any():
            markers[sel] = next_id
            next_id += 1

    # --- components without any marker become a single instance
    comp, n_comp = ndimage.label(building, structure=FOUR)
    has_marker = np.zeros(n_comp + 1, bool)
    has_marker[np.unique(comp[markers > 0])] = True
    for cid in np.flatnonzero(~has_marker[1:]) + 1:
        markers[comp == cid] = next_id
        next_id += 1

    # --- watershed inside the building mask: colour gradient + 2x gap evidence + core depth.
    # Non-building pixels get maximum elevation and NO marker, so roof pixels are always
    # reached from inside their own building first (no edge erosion by a background marker).
    lab_img = cv2.cvtColor(cv2.GaussianBlur(image2x, (3, 3), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    grad = np.zeros(building.shape, np.float32)
    for c in range(3):
        grad += ndimage.sobel(lab_img[:, :, c], 0) ** 2 + ndimage.sobel(lab_img[:, :, c], 1) ** 2
    grad = np.sqrt(grad)
    grad /= max(np.percentile(grad[building], 99), 1e-6) if building.any() else 1.0
    elev = np.clip(0.6 * grad + 0.4 * (1 - np.clip(dist / 8.0, 0, 1)) + 0.6 * (1 - bprob2x), 0, 2)
    elev_u16 = (elev / 2 * 60000).astype(np.uint16)
    elev_u16[~building] = 65535
    ws = ndimage.watershed_ift(elev_u16, markers, structure=FOUR).astype(np.int32)
    ws[~building] = 0
    bprob = bprob2x

    # --- merge weak splits (only splits not separated by two DelAny roofs)
    image_f = image2x.astype(np.float32)
    merges = 0
    for _ in range(4):
        evidence = boundary_evidence(ws, image_f, bprob)
        parent = {}

        def find(v):
            while parent.get(v, v) != v:
                v = parent[v]
            return v
        changed = False
        for (a, b), (shared, contrast, gap_prob) in sorted(evidence.items(), key=lambda kv: -kv[1][0]):
            if a in roof_marker_ids and b in roof_marker_ids:
                continue  # two independent roof detections: keep separate
            ra, rb = find(a), find(b)
            if ra == rb:
                continue
            weak = contrast < P["merge_contrast"] and gap_prob > P["merge_gap_prob"]
            if not weak:
                continue
            union = (ws == ra) | (ws == rb)
            if solidity(union) < P["merge_union_solidity"]:
                continue
            parent[rb] = ra
            if rb in roof_marker_ids:
                roof_marker_ids.add(ra)
            changed = True
            merges += 1
        if not changed:
            break
        for child in list(parent):
            ws[ws == child] = find(child)

    # --- per-instance connectivity, tiny fragments into neighbour or context
    ws = enforce_connectivity(ws, min_px, allow_drop=True)
    log["building_markers_delany"] = len([i for i in roof_marker_ids])
    log["building_markers_peaks"] = int(n_peaks)
    log["building_weak_merges"] = merges
    return relabel_sequential(ws)


def largest_component(mask):
    lab, n = ndimage.label(mask, structure=FOUR)
    if n <= 1:
        return mask
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    return lab == sizes.argmax()


def enforce_connectivity(labels, min_px, allow_drop=False):
    """Keep largest 4-component per label; detached/tiny pieces join the neighbour with the longest
    shared boundary, or are dropped (set 0) if they touch nothing and allow_drop."""
    out = labels.copy()
    for _ in range(6):
        changed = False
        orphan = np.zeros(out.shape, bool)
        objs = ndimage.find_objects(out)
        for lid, sl in enumerate(objs, 1):
            if sl is None:
                continue
            sub = out[sl] == lid
            lab, k = ndimage.label(sub, structure=FOUR)
            sizes = np.bincount(lab.ravel()); sizes[0] = 0
            keep = sizes.argmax()
            if k > 1:
                orphan[sl] |= sub & (lab != keep)
            if sizes[keep] < min_px:
                orphan[sl] |= sub & (lab == keep)
        if not orphan.any():
            break
        olab, on = ndimage.label(orphan, structure=FOUR)
        for oid, sl in enumerate(ndimage.find_objects(olab), 1):
            sl2 = tuple(slice(max(s.start - 1, 0), s.stop + 1) for s in sl)
            piece = olab[sl2] == oid
            ring = ndimage.binary_dilation(piece, structure=FOUR) & ~piece
            neigh = out[sl2][ring]
            own = out[sl2][piece][0]
            neigh = neigh[(neigh > 0) & (neigh != own)]
            if neigh.size:
                out[sl2][piece] = Counter(neigh.tolist()).most_common(1)[0][0]
                changed = True
            elif allow_drop:
                out[sl2][piece] = 0
                changed = True
        if not changed:
            break
    return out


# ------------------------------------------------------------------- context objects
def bbox_truncated(mask, min_side_px=20, max_fill=0.6, exempt=None):
    """True if a YOLO mask was clipped flat by its own bounding box (straight artificial edge).
    `exempt` = (top, bottom, left, right) booleans: sides on which a flat edge is legitimate
    because the object continues outside the image. Default: whichever sides lie on the border."""
    ys, xs = np.nonzero(mask)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    h, w = mask.shape
    if exempt is None:
        exempt = (True, True, True, True)
    on_border = (exempt[0] and y0 == 0, exempt[1] and y1 == h - 1, exempt[2] and x0 == 0, exempt[3] and x1 == w - 1)
    sides = ((mask[y0, x0:x1 + 1], x1 - x0), (mask[y1, x0:x1 + 1], x1 - x0),
             (mask[y0:y1 + 1, x0], y1 - y0), (mask[y0:y1 + 1, x1], y1 - y0))
    return any(length >= min_side_px and line.mean() >= max_fill and not border
               for (line, length), border in zip(sides, on_border))


def landcover_objects(prob2x, occupied, dl_masks, dl_conf, px_area2x, log):
    """Land-cover objects for all pixels not occupied by buildings/roads."""
    argmax = prob2x.argmax(0)
    group_idx = {g: i for i, g in enumerate(LANDCOVER_GROUPS)}
    # Group probabilities, then majority smoothing to remove texture speckle.
    gp = np.zeros((len(LANDCOVER_GROUPS), *argmax.shape), np.float32)
    for sem, g in GROUP_OF_SEMANTIC.items():
        if g in group_idx:
            gp[group_idx[g]] += prob2x[sem]
    gp = np.stack([ndimage.uniform_filter(c, size=9) for c in gp])
    group_map = gp.argmax(0)

    labels = np.zeros(argmax.shape, np.int32)
    meta = {}
    next_id = 1
    free = ~occupied
    # 1) DelAny parcels that are class-pure (terraces, fields, lots, gardens)
    parcel_min = int(P["delany_parcel_min_m2"] / px_area2x)
    order = np.argsort([-m.sum() for m in dl_masks]) if len(dl_masks) else []
    rejected = 0
    for k in order[::-1]:  # small first so nested small parcels are not swallowed
        if dl_conf[k] < P["delany_parcel_conf"] or bbox_truncated(dl_masks[k]):
            rejected += 1
            continue
        m = dl_masks[k] & free & (labels == 0)
        m = largest_component(m)
        if m.sum() < parcel_min:
            continue
        counts = np.bincount(group_map[m], minlength=len(LANDCOVER_GROUPS))
        g = int(counts.argmax())
        if counts[g] / counts.sum() < P["delany_parcel_purity"]:
            continue
        labels[m] = next_id
        meta[next_id] = {"object_type": LANDCOVER_GROUPS[g], "source": "delany_parcel"}
        next_id += 1
    n_parcels = next_id - 1

    # 2) remaining pixels: connected class regions above the size threshold
    lc_min = int(P["landcover_min_m2"] / px_area2x)
    remaining = free & (labels == 0)
    for g, name in enumerate(LANDCOVER_GROUPS):
        comp, n = ndimage.label(remaining & (group_map == g), structure=FOUR)
        if n == 0:
            continue
        sizes = np.bincount(comp.ravel())
        for cid in np.flatnonzero(sizes >= (lc_min if name != "water" else lc_min // 4)):
            if cid == 0:
                continue
            labels[comp == cid] = next_id
            meta[next_id] = {"object_type": name, "source": "semantic_region"}
            next_id += 1

    # 3) leftovers: region growing from adjacent land-cover objects (blocked by buildings/roads)
    unassigned = free & (labels == 0)
    grown = 0
    while unassigned.any():
        dil = ndimage.grey_dilation(labels, footprint=FOUR)
        take = unassigned & (dil > 0)
        if not take.any():
            break
        labels[take] = dil[take]
        grown += int(take.sum())
        unassigned &= ~take
    log["landcover_delany_parcels"] = n_parcels
    log["landcover_delany_rejected_lowconf_or_truncated"] = rejected
    log["landcover_pixels_absorbed"] = grown
    log["landcover_isolated_pixels"] = int(unassigned.sum())
    return labels, meta, unassigned


# ------------------------------------------------------------------- full tile
def segment_tile(tile: str, out_root: Path = OUT_ROOT):
    t0 = time.time()
    log = {"tile": tile}
    image, transform, crs = read_tile(tile)
    transform2x = transform * Affine.scale(1 / SCALE)
    final, meta, prob_ms = segment_image(image, abs(transform2x.a * transform2x.e), log)
    log["seconds"] = round(time.time() - t0, 1)
    out_dir = out_root / tile
    out_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(out_dir, tile, image, final, meta, prob_ms, transform2x, crs, log)
    return final, meta, log


def segment_image(image: np.ndarray, px_area2x: float = 0.5625, log: dict | None = None):
    """Segment an RGB uint8 array. Returns (object labels on the 2x grid, {id: meta}, prob_ms)."""
    log = {} if log is None else log
    h, w = image.shape[:2]
    image2x = cv2.resize(image, (w * SCALE, h * SCALE), interpolation=cv2.INTER_CUBIC)

    # Multi-scale semantics on the 2x grid: 2x resolves gaps between roofs, 1x/1.5x keep recall.
    _, prob2x = semantic_ms.infer(image, scale=SCALE, return_upsampled=True)
    size2x = prob2x.shape[:0:-1]
    prob_ms = prob2x.copy()
    for s in (1.0, 1.5):
        p = semantic_ms.infer(image, scale=s)
        prob_ms += np.stack([cv2.resize(c, size2x, interpolation=cv2.INTER_LINEAR) for c in p])
    prob_ms /= 3.0
    # DelAny at 2x: small objects (roofs, lots, terraces). At 1x: large fields (its training scale).
    # In simple2x mode DelAny is only used for parcels, which require conf >= delany_parcel_conf, so
    # it is run at that threshold directly (identical result; avoids hundreds of low-confidence
    # full-resolution masks that overflow 4 GB VRAM and spill into very slow shared memory).
    if P["building_mode"] == "simple2x":
        parcel_masks, parcel_conf = delany_parcel_candidates(image, image2x, P["delany_parcel_conf"])
        dl_masks, dl_conf = parcel_masks, parcel_conf
        dl1_masks = np.zeros((0,), bool)
    else:
        if image.shape[0] != image.shape[1] or image.shape[0] > 512:
            raise ValueError("watershed building mode supports single 512 px tiles only")
        dl_masks, dl_conf = delany_instances(image2x)
        dl1_masks, dl1_conf = delany_instances(image, out_size=image2x.shape[0])
    log["delany_instances_2x"] = int(len(dl_masks))
    log["delany_instances_1x"] = int(len(dl1_masks))
    if P["building_mode"] != "simple2x":
        parcel_masks = np.concatenate([dl_masks, dl1_masks]) if len(dl1_masks) else dl_masks
        parcel_conf = np.concatenate([dl_conf, dl1_conf]) if len(dl1_masks) else dl_conf

    # Buildings
    if P["building_mode"] == "simple2x":
        bld = building_instances_simple(prob2x, px_area2x)
    else:
        bld = building_instances(image2x, prob_ms, prob2x, dl_masks, dl_conf, px_area2x, log)
    log["building_mode"] = P["building_mode"]
    # Roads
    road = (prob_ms.argmax(0) == SEM_ROAD) & (bld == 0)
    road = ndimage.binary_closing(road, structure=np.ones((5, 5), bool)) & (bld == 0)
    road = remove_small(road, int(P["road_min_m2"] / px_area2x))
    road_lab, n_road = ndimage.label(road, structure=FOUR)

    occupied = (bld > 0) | (road_lab > 0)
    lc, lc_meta, isolated = landcover_objects(prob_ms, occupied, parcel_masks, parcel_conf, px_area2x, log)

    # Compose final label map: buildings, roads, land cover.
    final = np.zeros((h * SCALE, w * SCALE), np.int32)
    meta = {}
    nid = 1
    for lid in range(1, int(bld.max()) + 1):
        sel = bld == lid
        if sel.any():
            final[sel] = nid; meta[nid] = {"object_type": "building", "source": "watershed_instance"}; nid += 1
    for lid in range(1, n_road + 1):
        final[road_lab == lid] = nid; meta[nid] = {"object_type": "road", "source": "semantic_network"}; nid += 1
    for lid, m in lc_meta.items():
        sel = lc == lid
        if sel.any():
            final[sel] = nid; meta[nid] = m; nid += 1
    # Pockets fully enclosed by buildings/roads with no land-cover neighbour: give to a road
    # neighbour if any, else the adjacent building (they are courtyard/roof-gap pixels).
    if isolated.any():
        dil = ndimage.grey_dilation(final, footprint=FOUR)
        for _ in range(50):
            take = (final == 0) & (dil > 0)
            if not take.any():
                break
            final[take] = dil[take]
            dil = ndimage.grey_dilation(final, footprint=FOUR)

    # Connectivity guarantee for every object (tiny pieces absorbed by neighbour).
    final = enforce_connectivity(final, min_px=1)
    final, meta = relabel_meta(final, meta)
    return final, meta, prob_ms


def relabel_meta(labels, meta):
    values = [int(v) for v in np.unique(labels) if v > 0]
    lut = np.zeros(int(labels.max()) + 1, np.int32)
    new_meta = {}
    for i, v in enumerate(values, 1):
        lut[v] = i
        new_meta[i] = meta[v]
    return lut[labels], new_meta


# ------------------------------------------------------------------- outputs
def shape_metrics(geom):
    area, perim = geom.area, geom.length
    rect = geom.minimum_rotated_rectangle
    xs = list(rect.exterior.coords)
    sides = [math.dist(xs[i], xs[i + 1]) for i in range(4)]
    major, minor = max(sides), max(min(sides), 1e-9)
    e = max(range(4), key=lambda i: sides[i])
    orient = (math.degrees(math.atan2(xs[e + 1][1] - xs[e][1], xs[e + 1][0] - xs[e][0])) + 180) % 180
    return {
        "perimeter_m": round(perim, 3),
        "compactness": round(4 * math.pi * area / max(perim ** 2, 1e-9), 5),
        "solidity": round(area / max(geom.convex_hull.area, 1e-9), 5),
        "elongation": round(major / minor, 4),
        "orientation_deg": round(orient, 2),
    }


def write_outputs(out_dir, tile, image, final, meta, prob_ms, transform2x, crs, log, year=None):

    year = year or tile_year(tile)
    image2x = cv2.resize(image, final.shape[::-1], interpolation=cv2.INTER_CUBIC)
    sem2x = prob_ms.argmax(0)
    conf2x = prob_ms.max(0)

    profile = {"driver": "GTiff", "height": final.shape[0], "width": final.shape[1], "count": 1,
               "dtype": "int32", "crs": crs, "transform": transform2x, "nodata": 0,
               "compress": "deflate", "tiled": True}
    with rasterio.open(out_dir / "object_ids.tif", "w", **profile) as dst:
        dst.write(final, 1)

    geoms = {}
    for gj, v in features.shapes(final, mask=final > 0, transform=transform2x, connectivity=4):
        v = int(v)
        g = make_valid(shape(gj))
        geoms[v] = g if v not in geoms else make_valid(geoms[v].union(g))

    objs = ndimage.find_objects(final)
    rows = []
    for nid, sl in enumerate(objs, 1):
        if sl is None or nid not in geoms:
            continue
        mask = final[sl] == nid
        g = geoms[nid]
        rgb = image2x[sl][mask].astype(np.float32)
        sem_vals = sem2x[sl][mask]
        c = g.centroid
        b = g.bounds
        rows.append({
            "node_id": nid, "node_key": f"{year}:{tile}:{nid}", "tile_id": tile, "source_year": year,
            "object_type": meta[nid]["object_type"], "source": meta[nid]["source"],
            "dominant_semantic": int(np.bincount(sem_vals, minlength=9).argmax()),
            "building_fraction": round(float((sem_vals == SEM_BUILDING).mean()), 4),
            "mean_conf": round(float(conf2x[sl][mask].mean()), 4),
            "area_m2": round(g.area, 3), **shape_metrics(g),
            "geom_type": g.geom_type,
            "centroid_x": round(c.x, 3), "centroid_y": round(c.y, 3),
            "bbox_minx": round(b[0], 3), "bbox_miny": round(b[1], 3), "bbox_maxx": round(b[2], 3), "bbox_maxy": round(b[3], 3),
            "mean_r": round(float(rgb[:, 0].mean()), 2), "mean_g": round(float(rgb[:, 1].mean()), 2), "mean_b": round(float(rgb[:, 2].mean()), 2),
            "std_r": round(float(rgb[:, 0].std()), 2), "std_g": round(float(rgb[:, 1].std()), 2), "std_b": round(float(rgb[:, 2].std()), 2),
        })
    write_csv(out_dir / "nodes.csv", rows)

    # Edges: touching (shared boundary) + kNN proximity among buildings.
    shared = Counter()
    for a, b_, *_ in pair_boundaries(final):
        lo, hi = np.minimum(a, b_), np.maximum(a, b_)
        key = lo.astype(np.int64) * 10_000_000 + hi
        u, n = np.unique(key, return_counts=True)
        for k, cnt in zip(u, n):
            shared[(int(k // 10_000_000), int(k % 10_000_000))] += int(cnt)
    pix = abs(transform2x.a)
    cent = {r["node_id"]: (r["centroid_x"], r["centroid_y"]) for r in rows}
    edges = [{"source": s, "target": t, "edge_type": "touching", "shared_boundary_m": round(n * pix, 2),
              "centroid_distance_m": round(math.dist(cent[s], cent[t]), 2)} for (s, t), n in sorted(shared.items())]
    from scipy.spatial import cKDTree
    bids = [r["node_id"] for r in rows if r["object_type"] == "building"]
    if len(bids) > 1:
        pts = np.array([cent[i] for i in bids])
        tree = cKDTree(pts)
        seen = set(shared)
        for i, nid in enumerate(bids):
            d, j = tree.query(pts[i], k=min(6, len(bids)), distance_upper_bound=60)
            for dist, jj in zip(np.atleast_1d(d), np.atleast_1d(j)):
                if not np.isfinite(dist) or jj == i:
                    continue
                pair = tuple(sorted((nid, bids[jj])))
                if pair not in seen:
                    seen.add(pair)
                    edges.append({"source": pair[0], "target": pair[1], "edge_type": "proximity",
                                  "shared_boundary_m": 0.0, "centroid_distance_m": round(float(dist), 2)})
    write_csv(out_dir / "edges.csv", edges)

    gpkg = out_dir / "objects.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    ds = ogr.GetDriverByName("GPKG").CreateDataSource(str(gpkg))
    srs = osr.SpatialReference(); srs.ImportFromWkt(crs.to_wkt())
    layer = ds.CreateLayer("objects", srs, ogr.wkbUnknown)
    for name, kind in (("node_id", ogr.OFTInteger), ("node_key", ogr.OFTString), ("obj_type", ogr.OFTString),
                       ("source", ogr.OFTString), ("area_m2", ogr.OFTReal), ("solidity", ogr.OFTReal),
                       ("mean_conf", ogr.OFTReal)):
        layer.CreateField(ogr.FieldDefn(name, kind))
    defn = layer.GetLayerDefn()
    layer.StartTransaction()
    for r in rows:
        f = ogr.Feature(defn)
        f.SetField("node_id", r["node_id"]); f.SetField("node_key", r["node_key"])
        f.SetField("obj_type", r["object_type"]); f.SetField("source", r["source"])
        f.SetField("area_m2", r["area_m2"]); f.SetField("solidity", r["solidity"]); f.SetField("mean_conf", r["mean_conf"])
        f.SetGeometry(ogr.CreateGeometryFromWkb(geoms[r["node_id"]].wkb))
        layer.CreateFeature(f)
    layer.CommitTransaction()
    ds = None

    log["nodes"] = len(rows)
    log["edges"] = len(edges)
    log["by_type"] = dict(Counter(r["object_type"] for r in rows))
    log["non_polygon_geometries"] = sum(r["geom_type"] != "Polygon" for r in rows)
    log["grid_resolution_m"] = pix
    log["source_year_from_metadata"] = year
    (out_dir / "run_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tiles", nargs="*", default=[f"tile_{i:05d}" for i in range(7)])
    parser.add_argument("--building-mode", choices=["simple2x", "watershed"], default=P["building_mode"])
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()
    P["building_mode"] = args.building_mode
    for tile in args.tiles:
        _, _, log = segment_tile(tile, args.out_root)
        print(json.dumps(log))


if __name__ == "__main__":
    main()
