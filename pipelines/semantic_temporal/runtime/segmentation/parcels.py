"""Building and parcel geometry extracted from the frozen study methods."""
import math
from collections import Counter
import numpy as np
import cv2
from scipy import ndimage
FOUR=np.array([[0,1,0],[1,1,1],[0,1,0]],bool)
SEM_BUILDING=8

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

def relabel_sequential(labels):
    values = np.unique(labels)
    values = values[values > 0]
    lut = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    lut[values] = np.arange(1, len(values) + 1)
    return lut[labels]

def building_instances_simple(prob2x, px_area2x, P):
    """Buildings = 4-connected components of the 2x building mask (argmax and p >= 0.4, opened,
    components smaller than min_building_m2 dropped). Deliberately no instance splitting: this is
    the variant that validated as most stable for change detection."""
    mask = (prob2x.argmax(0) == SEM_BUILDING) & (prob2x[SEM_BUILDING] >= P["building_prob_2x"])
    mask = ndimage.binary_opening(mask, structure=FOUR)
    mask = remove_small(mask, int(round(P["min_building_m2"] / px_area2x)))
    return neck_split(mask, P["neck_split_radius_px"], px_area2x, P)

def neck_split(mask, r, px_area2x, P):
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
