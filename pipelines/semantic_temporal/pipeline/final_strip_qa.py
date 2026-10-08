"""QA and cross-year links for the final 2013 vs 2023 strip.

Outputs (in final/<strip>/):
  qa_osm_2023.json        2023 buildings vs OSM (independent check that the mosaic behaves like
                          the validated tiles)
  object_links.csv        every overlapping (2013 object, 2023 object) pair: pixels, IoU, fractions
  building_events.csv     one row per connected group of overlapping buildings, classified as
                          stable / shape / split / merge / complex / new_2023 / gone_2023
  qa_years_zoom.png       2013 | 2023 objects side by side for a few windows
These are inputs for change detection, not change detection itself: residual co-registration error
(median ~1 px) and the season difference still produce some false events (validated rate for
a 1 px misregistration: ~15 false events per 100 buildings).
"""
from osgeo import gdal  # noqa: F401
import json
import sys
from collections import defaultdict
from pathlib import Path

import fonts  # package: portable font paths

import cv2
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio import features
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "validation"))
import osm_reference  # noqa: E402

D = HERE / "final" / "strip_tile_00000-00006"
YEARS = ("2013", "2023")


def load(year):
    with rasterio.open(D / year / "object_ids.tif") as s:
        lab, tr = s.read(1), s.transform
    nodes = pd.read_csv(D / year / "nodes.csv").set_index("node_id")
    return lab, tr, nodes


def mask_iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def roof_offset_field(b13, b23, seg=1024, search=12, smooth_sigma=1.0):
    """Regional roof displacement (2x px) of 2013 buildings relative to 2023 along the strip:
    best mask-overlap shift per `seg`-px segment, then smoothed and linearly interpolated per column.
    Co-registration aligns the ground; roofs additionally lean by view angle x height."""
    centers, dys, dxs = [], [], []
    W = b23.shape[1]
    for x0 in range(0, W - seg // 2, seg // 2):
        s13, s23 = b13[:, x0:x0 + seg], b23[:, x0:x0 + seg]
        if s23.sum() < 500:
            continue
        best = max(((mask_iou(np.roll(np.roll(s13, dy, 0), dx, 1)[16:-16, 16:-16], s23[16:-16, 16:-16]), dy, dx)
                    for dy in range(-search, search + 1) for dx in range(-search, search + 1)))
        centers.append(x0 + min(seg, W - x0) / 2); dys.append(best[1]); dxs.append(best[2])
    from scipy.ndimage import gaussian_filter1d
    dys = gaussian_filter1d(np.array(dys, float), smooth_sigma); dxs = gaussian_filter1d(np.array(dxs, float), smooth_sigma)
    cols = np.arange(W)
    return np.interp(cols, centers, dys), np.interp(cols, centers, dxs), list(zip(centers, dys.round(2), dxs.round(2)))


def offset_building_labels(lab13, bids13, b23, fdy, fdx, local=4):
    """Move every 2013 building by the regional roof offset at its centroid, refined per building
    within +-`local` px to maximise overlap with the 2023 building mask."""
    from scipy import ndimage
    out = np.zeros_like(lab13)
    H, W = lab13.shape
    objs = ndimage.find_objects(lab13)
    shifts = {}
    for bid in sorted(bids13, key=lambda b: -(lab13 == b).sum() if False else 0):
        sl = objs[bid - 1] if bid - 1 < len(objs) else None
        if sl is None:
            continue
        ys, xs = np.nonzero(lab13[sl] == bid)
        ys = ys + sl[0].start; xs = xs + sl[1].start
        cx = int(round(xs.mean()))
        base_dy, base_dx = int(round(fdy[cx])), int(round(fdx[cx]))
        # maximise overlap; on ties prefer the smallest departure from the regional offset
        # (otherwise a building next to another can slide onto its neighbour with an equal score)
        best, bdy, bdx = (-1, 0.0), base_dy, base_dx
        for dy in range(base_dy - local, base_dy + local + 1):
            for dx in range(base_dx - local, base_dx + local + 1):
                yy, xx = ys + dy, xs + dx
                ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
                score = (int(b23[yy[ok], xx[ok]].sum()), -float(np.hypot(dy - base_dy, dx - base_dx)))
                if score > best:
                    best, bdy, bdx = score, dy, dx
        yy, xx = ys + bdy, xs + bdx
        ok = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
        out[yy[ok], xx[ok]] = bid
        shifts[bid] = (bdy, bdx)
    return out, shifts


def osm_check(lab23, tr2, nodes23):
    import evaluate as ev
    import semantic_ms as sm
    polys = osm_reference.osm_polygons()
    ref = features.rasterize(((g, i + 1) for i, g in enumerate(polys)), out_shape=lab23.shape,
                             transform=tr2, fill=0, dtype=np.int32)
    img = rasterio.open(D / "input_2023_harmonized.tif").read().transpose(1, 2, 0)
    p1 = sm.infer(img, scale=1)  # 1x model for alignment (not the 2x model v3 uses)
    p1u = cv2.resize(p1[8], lab23.shape[::-1], interpolation=cv2.INTER_LINEAR)
    dy, dx = osm_reference.best_shift(ref > 0, p1u)
    ref = np.roll(np.roll(ref, dy, 0), dx, 1)
    b = np.where(np.isin(lab23, nodes23.index[nodes23.object_type == "building"]), lab23, 0)
    out = {"osm_shift_px2x": [int(dy), int(dx)], **ev.osm_metrics(b, ref)}
    (D / "qa_osm_2023.json").write_text(json.dumps(out, indent=2))
    return out


def links_and_events(lab13, n13, lab23, n23, lab13_buildings=None, suffix=""):
    m = (lab13 > 0) & (lab23 > 0)
    key = lab13[m].astype(np.int64) * 10_000_000 + lab23[m]
    u, c = np.unique(key, return_counts=True)
    a13, a23 = np.bincount(lab13.ravel()), np.bincount(lab23.ravel())
    links = pd.DataFrame({"node_2013": u // 10_000_000, "node_2023": u % 10_000_000, "shared_px": c})
    links["type_2013"] = n13.loc[links.node_2013, "object_type"].values
    links["type_2023"] = n23.loc[links.node_2023, "object_type"].values
    links["frac_of_2013"] = (links.shared_px / a13[links.node_2013]).round(4)
    links["frac_of_2023"] = (links.shared_px / a23[links.node_2023]).round(4)
    links["iou"] = (links.shared_px / (a13[links.node_2013] + a23[links.node_2023] - links.shared_px)).round(4)
    links["shared_m2"] = (links.shared_px * 0.5625).round(2)
    if not suffix:
        links.to_csv(D / "object_links.csv", index=False)
    if lab13_buildings is not None:
        # re-link buildings using the roof-offset-compensated 2013 building map
        mb = (lab13_buildings > 0) & (lab23 > 0)
        kb = lab13_buildings[mb].astype(np.int64) * 10_000_000 + lab23[mb]
        ub, cb = np.unique(kb, return_counts=True)
        a13 = np.bincount(lab13_buildings.ravel(), minlength=len(a13))
        links = pd.DataFrame({"node_2013": ub // 10_000_000, "node_2023": ub % 10_000_000, "shared_px": cb})
        links["type_2013"] = n13.loc[links.node_2013, "object_type"].values
        links["type_2023"] = n23.loc[links.node_2023, "object_type"].values
        links["iou"] = (links.shared_px / (a13[links.node_2013] + a23[links.node_2023] - links.shared_px)).round(4)

    # building events: same linking rule as the validation (overlap >= 25% of the smaller object)
    b13 = n13.index[n13.object_type == "building"].values
    b23 = n23.index[n23.object_type == "building"].values
    bl = links[(links.type_2013 == "building") & (links.type_2023 == "building")]
    bl = bl[(bl.shared_px >= 0.25 * np.minimum(a13[bl.node_2013], a23[bl.node_2023])) & (bl.shared_px >= 6)]
    idx13 = {v: i for i, v in enumerate(b13)}
    idx23 = {v: i + len(b13) for i, v in enumerate(b23)}
    n = len(b13) + len(b23)
    g = coo_matrix((np.ones(len(bl)), ([idx13[v] for v in bl.node_2013], [idx23[v] for v in bl.node_2023])), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    groups = defaultdict(lambda: ([], []))
    for v, i in idx13.items(): groups[comp[i]][0].append(int(v))
    for v, i in idx23.items(): groups[comp[i]][1].append(int(v))
    iou = {(r.node_2013, r.node_2023): r.iou for r in bl.itertuples()}
    rows = []
    for gid, (A, B) in enumerate(groups.values(), 1):
        if len(A) == 1 and len(B) == 1:
            event = "stable" if iou.get((A[0], B[0]), 0) >= 0.5 else "shape"
        elif A and not B:
            event = "gone_2023"
        elif B and not A:
            event = "new_2023"
        elif len(A) == 1:
            event = "split"
        elif len(B) == 1:
            event = "merge"
        else:
            event = "complex"
        rows.append({"group": gid, "event": event, "nodes_2013": "|".join(map(str, A)), "nodes_2023": "|".join(map(str, B)),
                     "area_2013_m2": round(float(a13[A].sum()) * 0.5625, 1) if A else 0.0,
                     "area_2023_m2": round(float(a23[B].sum()) * 0.5625, 1) if B else 0.0})
    ev = pd.DataFrame(rows)
    ev.to_csv(D / f"building_events{suffix}.csv", index=False)
    return links, ev


def render(lab13, n13, lab23, n23):
    imgs = {y: rasterio.open(D / ("input_2023_harmonized.tif" if y == "2023" else "input_2013_coregistered_harmonized.tif")).read().transpose(1, 2, 0) for y in YEARS}
    font = ImageFont.truetype(fonts.BOLD, 18)
    rng = np.random.default_rng(3)
    windows = [(40, 280, 128), (300, 1460, 128), (160, 2860, 128), (380, 600, 128)]
    tiles = []
    for (y, x, w) in windows:
        row = []
        for year, lab, nodes in (("2013", lab13, n13), ("2023", lab23, n23)):
            img = cv2.resize(imgs[year][y:y + w, x:x + w], (512, 512), interpolation=cv2.INTER_CUBIC)
            l = cv2.resize(lab[2 * y:2 * (y + w), 2 * x:2 * (x + w)].astype(np.float32), (512, 512), interpolation=cv2.INTER_NEAREST).astype(np.int64)
            is_b = np.zeros(lab.max() + 1, bool); is_b[nodes.index[nodes.object_type == "building"]] = True
            col = rng.uniform(70, 255, (lab.max() + 1, 3))
            out = img.astype(np.float32)
            bm = is_b[l]
            out[bm] = 0.45 * out[bm] + 0.55 * col[l][bm]
            e = np.zeros(l.shape, bool); e[:, 1:] |= l[:, 1:] != l[:, :-1]; e[1:] |= l[1:] != l[:-1]
            out[e & bm] = 255; out[e & ~bm] = (255, 200, 0)
            pil = Image.fromarray(out.clip(0, 255).astype(np.uint8)); ImageDraw.Draw(pil).text((10, 8), year, font=font, fill="white")
            row.append(np.array(pil))
        tiles.append(np.concatenate([row[0], np.full((512, 8, 3), 20, np.uint8), row[1]], 1))
    Image.fromarray(np.concatenate([np.vstack([t, np.full((8, t.shape[1], 3), 20, np.uint8)]) for t in tiles], 0)).save(D / "qa_years_zoom.png", optimize=True)


def main():
    lab13, tr, n13 = load("2013")
    lab23, tr23, n23 = load("2023")
    assert tr == tr23 and lab13.shape == lab23.shape
    osm = osm_check(lab23, tr23, n23)
    print("OSM 2023:", json.dumps(osm))
    links, ev = links_and_events(lab13, n13, lab23, n23, suffix="_uncompensated")
    print("links:", len(links), "| building events, ground-registered only:", ev.event.value_counts().to_dict())
    bids13 = n13.index[n13.object_type == "building"].values
    b13 = np.isin(lab13, bids13); b23 = np.isin(lab23, n23.index[n23.object_type == "building"].values)
    fdy, fdx, prof = roof_offset_field(b13, b23)
    lab13b, shifts = offset_building_labels(lab13, bids13, b23, fdy, fdx)
    pd.DataFrame([{"node_2013": k, "roof_dy_px2x": v[0], "roof_dx_px2x": v[1]} for k, v in shifts.items()]).to_csv(D / "building_roof_offsets_2013.csv", index=False)
    _, ev2 = links_and_events(lab13, n13, lab23, n23, lab13_buildings=lab13b, suffix="")
    print("regional roof offset profile (x_center, dy, dx in 2x px):", prof)
    print("building events, roof-offset compensated:", ev2.event.value_counts().to_dict())
    (D / "qa_events.json").write_text(json.dumps({"uncompensated": ev.event.value_counts().to_dict(),
                                                  "roof_offset_compensated": ev2.event.value_counts().to_dict(),
                                                  "roof_offset_profile": [[float(a), float(b), float(c)] for a, b, c in prof]}, indent=2))
    ev3 = add_reliability()
    print("building events by reliability:", ev3.groupby(["event", "reliability"]).size().to_dict())
    render(lab13, n13, lab23, n23)



# ----------------------------------------------------------------------------- event reliability
SMALL_M2, LOW_CONF = 60.0, 0.60


def add_reliability():
    """Flag building events whose objects are small or low-confidence. A visual audit showed the
    false 'gone_2023' events are almost all small low-confidence 2013 detections (shadows, clearings)."""
    ev = pd.read_csv(D / "building_events.csv")
    nodes = {y: pd.read_csv(D / y / "nodes.csv").set_index("node_id") for y in YEARS}
    cols = {"max_area_m2": [], "mean_conf": [], "reliability": []}
    for r in ev.itertuples():
        ids = [(y, int(v)) for y in YEARS for v in str(getattr(r, f"nodes_{y}")).split("|") if v not in ("", "nan")]
        area = np.array([nodes[y].loc[i, "area_m2"] for y, i in ids]); conf = np.array([nodes[y].loc[i, "mean_conf"] for y, i in ids])
        wconf = float((area * conf).sum() / area.sum())
        cols["max_area_m2"].append(round(float(area.max()), 1)); cols["mean_conf"].append(round(wconf, 3))
        cols["reliability"].append("low" if area.max() < SMALL_M2 or wconf < LOW_CONF else "high")
    for k, v in cols.items():
        ev[k] = v
    ev.to_csv(D / "building_events.csv", index=False)
    return ev


if __name__ == "__main__":
    main()
