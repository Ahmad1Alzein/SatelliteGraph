"""Generalized region segmentation derived from segment_final, with per-run settings."""
import math,time
from collections import Counter,defaultdict
import numpy as np
import cv2
from scipy import ndimage
from . import parcels as v2, refinement as refine_merge
from .semantic import semantics
FOUR=v2.FOUR
CLASSES = {1: "building", 2: "road", 3: "tree", 4: "grass_shrub", 5: "cropland",
           6: "bare_soil", 7: "paved_developed", 8: "water"}
LAND = (3, 4, 5, 6, 7)
DEFAULT_SETTINGS = {
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

def pool_water(image2x, *, P):
    hsv = cv2.cvtColor(image2x, cv2.COLOR_RGB2HSV)
    m = (hsv[:, :, 0] >= P['pool_hue'][0]) & (hsv[:, :, 0] <= P['pool_hue'][1]) & (hsv[:, :, 1] >= P['pool_sat']) & (hsv[:, :, 2] >= P['pool_val'])
    m = ndimage.binary_opening(m, structure=FOUR)
    return v2.remove_small(m, max(4, int(P['pool_min_m2'] / P['pixel_area'])))

def bare_ground(image2x, *, P):
    x = image2x.astype(np.float32)
    exg = (2 * x[:, :, 1] - x[:, :, 0] - x[:, :, 2]) / (x.sum(2) + 1)
    veg_frac = ndimage.uniform_filter((exg > P['bare_exg']).astype(np.float32), size=P['smooth_px'])
    red = (x[:, :, 0] - x[:, :, 2]) / (x[:, :, 0] + x[:, :, 2] + 1)
    raw = (veg_frac < P['bare_veg_frac']) & (red >= P['bare_redness']) & (x.max(2) >= P['bare_val'])
    return ndimage.uniform_filter(raw.astype(np.float32), size=P['smooth_px']) >= P['bare_share']

def thin(mask):
    """Zhang-Suen thinning (vectorized)."""
    img = mask.astype(np.uint8).copy()
    while True:
        changed = False
        for step in (0, 1):
            Pd = np.pad(img, 1)
            p2, p3, p4, p5 = (Pd[:-2, 1:-1], Pd[:-2, 2:], Pd[1:-1, 2:], Pd[2:, 2:])
            p6, p7, p8, p9 = (Pd[2:, 1:-1], Pd[2:, :-2], Pd[1:-1, :-2], Pd[:-2, :-2])
            nb = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            B = sum((x.astype(np.int16) for x in nb[:8]))
            A = sum((((nb[i] == 0) & (nb[i + 1] == 1)).astype(np.int16) for i in range(8)))
            c1 = p2 * p4 * p6 if step == 0 else p2 * p4 * p8
            c2 = p4 * p6 * p8 if step == 0 else p2 * p6 * p8
            m = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & (c1 == 0) & (c2 == 0)
            if m.any():
                img[m] = 0
                changed = True
        if not changed:
            return img.astype(bool)

def split_roads(road, px_m, *, P):
    """Street segments: skeleton branches between junctions; every road pixel joins its nearest branch."""
    skel = thin(road)
    Pd = np.pad(skel.astype(np.uint8), 1)
    nbs = [Pd[:-2, 1:-1], Pd[:-2, 2:], Pd[1:-1, 2:], Pd[2:, 2:], Pd[2:, 1:-1], Pd[2:, :-2], Pd[1:-1, :-2], Pd[:-2, :-2]]
    crossings = sum((((nbs[i] == 0) & (nbs[(i + 1) % 8] == 1)).astype(np.int16) for i in range(8)))
    junction = skel & (crossings >= 3)
    width = ndimage.distance_transform_edt(road)
    jzone = np.zeros_like(road)
    for y, x in zip(*np.nonzero(junction)):
        r = int(width[y, x]) + 1
        jzone[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = True
    branches, n = ndimage.label(skel & ~jzone, structure=np.ones((3, 3), bool))
    if n == 0:
        return ndimage.label(road, structure=FOUR)[0]
    lengths = np.bincount(branches.ravel()) * px_m
    touches_j = np.zeros(n + 1, bool)
    near_j = ndimage.binary_dilation(jzone, iterations=2)
    touches_j[np.unique(branches[near_j])] = True
    spur = (lengths < P['road_spur_max_m']) & touches_j
    spur[0] = False
    if (~spur[1:]).any():
        branches[spur[branches]] = 0
    _, idx = ndimage.distance_transform_edt(branches == 0, return_indices=True)
    seg = branches[idx[0], idx[1]]
    seg[~road] = 0
    return v2.relabel_sequential(seg)

def gradient_image(image2x, sigma):
    lab = cv2.cvtColor(cv2.GaussianBlur(image2x, (0, 0), sigma), cv2.COLOR_RGB2LAB).astype(np.float32)
    g = np.zeros(image2x.shape[:2], np.float32)
    for c in range(3):
        g += ndimage.sobel(lab[:, :, c], 0) ** 2 + ndimage.sobel(lab[:, :, c], 1) ** 2
    return np.sqrt(g)

def split_large(region, grad, px_area, *, P):
    """Split one large region along image edges into ~target-sized pieces."""
    target_px = P['target_region_m2'] / px_area
    S = max(8, int(math.sqrt(target_px)))
    ys, xs = np.nonzero(region)
    y0, y1, x0, x1 = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
    reg = region[y0:y1, x0:x1]
    dist = ndimage.distance_transform_edt(reg)
    markers = np.zeros(reg.shape, np.int32)
    k = 0
    for yy in range(S // 2, reg.shape[0], S):
        for xx in range(S // 2, reg.shape[1], S):
            sy0, sy1 = (max(0, yy - S // 3), min(reg.shape[0], yy + S // 3 + 1))
            sx0, sx1 = (max(0, xx - S // 3), min(reg.shape[1], xx + S // 3 + 1))
            win = dist[sy0:sy1, sx0:sx1]
            if win.max() < 1.5:
                continue
            dy, dx = np.unravel_index(np.argmax(win), win.shape)
            k += 1
            markers[sy0 + dy, sx0 + dx] = k
    if k < 2:
        return None
    g = grad[y0:y1, x0:x1]
    elev = np.clip(g / max(np.percentile(g[reg], 95), 1e-06), 0, 1)
    elev = (elev * 60000).astype(np.uint16)
    elev[~reg] = 65535
    ws = ndimage.watershed_ift(elev, ndimage.grey_dilation(markers, size=3) * reg, structure=FOUR).astype(np.int32)
    ws[~reg] = 0
    out = np.zeros(region.shape, np.int32)
    out[y0:y1, x0:x1] = ws
    return out

def shared_boundaries(labels):
    counts = Counter()
    for a, b, *_ in v2.pair_boundaries(labels):
        lo, hi = (np.minimum(a, b), np.maximum(a, b))
        u, c = np.unique(lo.astype(np.int64) * 10000000 + hi, return_counts=True)
        for kk, cc in zip(u, c):
            counts[int(kk // 10000000), int(kk % 10000000)] += int(cc)
    return counts

def merge_small(labels, cls, px_area, max_iter=8, *, P):
    """Merge segments below their class minimum into the neighbour with the longest shared
    boundary (same class preferred; buildings only absorb fully enclosed specks)."""
    for _ in range(max_iter):
        areas = np.bincount(labels.ravel()) * px_area
        small = [i for i in range(1, len(areas)) if areas[i] > 0 and areas[i] < P['min_segment_m2'][cls[i]]]
        if not small:
            break
        nb = defaultdict(list)
        for (a, b), n in shared_boundaries(labels).items():
            nb[a].append((b, n))
            nb[b].append((a, n))
        lut = np.arange(len(areas))
        for i in sorted(small, key=lambda j: areas[j]):
            cand = [(b, n) for b, n in nb[i] if lut[b] == b]
            if not cand:
                continue
            non_b = [(b, n) for b, n in cand if cls[b] != 1]
            pool = non_b if non_b and cls[i] != 1 else cand
            same = [(b, n) for b, n in pool if cls[b] == cls[i]]
            target = max(same or pool, key=lambda t: t[1])[0]
            lut[i] = target
        for i in range(len(lut)):
            j = i
            while lut[j] != j:
                j = lut[j]
            lut[i] = j
        labels = lut[labels]
    return labels

def segment(image, px_area=0.5625, log=None, *, P, models, valid, building_settings):
    log = {} if log is None else log
    t0 = time.time()
    h, w = image.shape[:2]
    image2x = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    p2, cp = semantics(image, models)
    p2[:,~valid]=0
    cp[:,~valid]=0
    px_m = math.sqrt(px_area)
    bld = v2.building_instances_simple(p2, px_area, building_settings)
    is_b = bld > 0
    raw = cp.argmax(0)
    raw[raw == 0] = 6
    road = ndimage.binary_closing(raw == 2, structure=np.ones((P['road_close_px'], P['road_close_px']), bool)) & ~is_b
    road = v2.remove_small(road, int(P['road_min_m2'] / px_area))
    roads = split_roads(road, px_m, P=P)
    pools = pool_water(image2x, P=P)
    bld[pools] = 0
    is_b = bld > 0
    road &= ~pools
    water = ((raw == 8) & (cp[8] >= P['water_prob']) | pools) & ~is_b & ~road
    water = v2.remove_small(water, max(4, int(P['water_min_m2'] / px_area)))
    log['pool_pixels'] = int(pools.sum())
    waters = ndimage.label(water, structure=FOUR)[0]
    sp = np.stack([ndimage.uniform_filter(cp[c], size=P['smooth_px']) for c in range(9)])
    land_cls = np.array(LAND)[np.argmax(sp[list(LAND)], axis=0)]
    bare = bare_ground(image2x, P=P) & np.isin(land_cls, (4, 5, 7))
    land_cls[bare] = 6
    log['bare_rule_pixels'] = int(bare.sum())
    free = valid & ~is_b & ~road & ~water
    labels = np.zeros((2 * h, 2 * w), np.int64)
    cls = [0]
    src = ['']

    def add(mask, c, source):
        labels[mask] = len(cls)
        cls.append(c)
        src.append(source)
    for i in range(1, bld.max() + 1):
        m = bld == i
        if m.any():
            add(m, 1, 'building_v3')
    for i in range(1, roads.max() + 1):
        m = roads == i
        if m.any():
            add(m, 2, 'road_segment')
    for i in range(1, waters.max() + 1):
        add(waters == i, 8, 'water')
    pm, pc = models.parcels(image, image2x.shape[:2], P['parcel_conf'])
    n_parcels = 0
    for k in np.argsort([m.sum() for m in pm]):
        m = v2.largest_component(pm[k] & free & (labels == 0))
        if m.sum() * px_area < P['parcel_min_m2']:
            continue
        cnt = np.bincount(land_cls[m], minlength=9)
        c = int(cnt.argmax())
        if cnt[c] / cnt.sum() < P['parcel_purity']:
            continue
        add(m, c, 'delany_parcel')
        n_parcels += 1
    remaining = free & (labels == 0)
    for c in LAND:
        comp, n = ndimage.label(remaining & (land_cls == c), structure=FOUR)
        sizes = np.bincount(comp.ravel()) * px_area
        for i in np.flatnonzero(sizes >= P['land_min_region_m2']):
            if i:
                add(comp == i, c, 'class_region')
    for c in LAND:
        components,n=ndimage.label(free & (labels==0) & (land_cls==c),structure=FOUR)
        for ident in range(1,n+1):
            piece=components==ident
            if not np.any(ndimage.binary_dilation(piece,structure=FOUR) & (labels>0)):
                add(piece,int(c),'small_isolated_class_region')
    for allow_building in (False, True):
        for _ in range(200):
            un = valid & (labels == 0)
            if not un.any():
                break
            src_lab = labels if allow_building else np.where(np.isin(labels, np.flatnonzero(np.array(cls) == 1)), 0, labels)
            dil = ndimage.grey_dilation(src_lab, footprint=FOUR)
            take = un & (dil > 0)
            if not take.any():
                break
            labels[take] = dil[take]
    log['unassigned_after_growth'] = int((labels == 0).sum())
    grad = gradient_image(image2x, 1.125 / px_m)
    areas = np.bincount(labels.ravel()) * px_area
    perim = np.zeros(len(areas))
    for a, b, *_ in v2.pair_boundaries(labels):
        perim += np.bincount(a, minlength=len(areas))[:len(areas)] + np.bincount(b, minlength=len(areas))[:len(areas)]
    perim *= px_m
    compact = 4 * np.pi * areas / np.maximum(perim, 1e-06) ** 2
    n_split = 0
    for i in range(1, len(cls)):
        big = areas[i] > P['split_above_m2']
        snake = areas[i] > P['snake_above_m2'] and compact[i] < P['snake_compactness']
        if cls[i] in LAND and (big or snake):
            parts = split_large(labels == i, grad, px_area, P=P)
            if parts is None:
                continue
            for j in range(2, parts.max() + 1):
                m = parts == j
                if m.any():
                    labels[m] = len(cls)
                    cls.append(cls[i])
                    src.append(src[i] + '+edge_split')
            n_split += 1
    log['regions_split'] = n_split
    labels = v2.enforce_connectivity(labels.astype(np.int32), min_px=1).astype(np.int64)
    cls_arr = np.array(cls)
    labels = merge_small(labels, cls_arr, px_area, P=P)
    labels = v2.enforce_connectivity(labels.astype(np.int32), min_px=1).astype(np.int64)
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
    pre_merge = final.copy()
    final, seg_cls, seg_src, merge_log = refine_merge.refine(final.astype(np.int64), seg_cls, seg_src, image2x, cp, grad, px_m)
    log['segments_before_merge'] = int(pre_merge.max())
    log['merges'] = int((merge_log.decision == 'merge').sum()) if len(merge_log) else 0
    log['merges_by_class'] = {CLASSES[int(k)] if str(k).isdigit() else str(k): int(v) for k, v in merge_log[merge_log.decision == 'merge']['class'].value_counts().items()} if len(merge_log) else {}
    log.update({'delany_parcels': n_parcels, 'segments': int(final.max()), 'unassigned_pixels': int((final == 0).sum()), 'seconds': round(time.time() - t0, 1)})
    return (final, seg_cls, seg_src, cp, image2x, rule_support, pre_merge, merge_log)
