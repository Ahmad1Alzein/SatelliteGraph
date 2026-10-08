"""Combine frozen tile segmentations into one strip per year (placement from metadata x0, y0) and build
the strip region graph, the strip temporal layer and the strip spatio-temporal graph.

Placement   tile pixel (r, c) -> strip pixel (2*y0 + r, 2*x0 + c) on the 0.75 m grid; the strip transform is
            the shared grid affine (metadata grid_affine_*) at half the pixel size. Checked against every
            frozen tile's own georeferencing.
Overlaps    tiles overlap by 50 px (1.5 m grid). Each strip pixel is taken from the covering tile whose centre
            is nearest (a Voronoi cut through the middle of each overlap).
Seams       an object cut by an ownership boundary is re-joined when the two tile segments touching across the
            boundary have the same class and are the same object inside the overlap (pixel overlap
            >= SEAM_MATCH x the smaller of the two footprints inside the overlap).
Cleanup     pieces of a tile segment separated by the cut become their own segments; pieces below the class
            minimum are merged with segment_final.merge_small (frozen rule).
Nodes       class from the source segment; geometry, shape and colour recomputed on the strip; class
            probabilities / confidence are pixel-weighted means of the source tile nodes; lineage kept.
Edges       touches (raster adjacency) and near (building-building <= 25 m), build_graph rules.
Temporal    temporal_layer.build_layer (frozen rules) on the two strip label maps.
Output      graphs/strip_<name>/ : <year>/ (segment_ids.tif, image_2x.tif, nodes.csv, edges.csv, graph.graphml),
            temporal_edges.csv, stgraph nodes/edges + tensors, validation.json
"""
from __future__ import annotations

from osgeo import gdal, ogr, osr  # noqa: F401
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import paths  # package: input/checkpoint locations

import cv2
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
import torch
from affine import Affine
from rasterio import features
from scipy import ndimage
from shapely import make_valid
from shapely.geometry import shape

import build_graph as bg
import segment_final as sf
import segment_v2 as v2
import temporal_layer as tl

HERE = Path(__file__).resolve().parent
META = paths.META_2023   # package: input location from paths.py
TILES = [f"tile_{i:05d}" for i in range(7)]
NAME = "strip_tile_00000-00006"
OUT = HERE / "graphs" / f"strip_{NAME.split('_', 1)[1]}"
if len(sys.argv) > 1 and sys.argv[1].startswith("--out="):
    OUT = Path(sys.argv[1].split("=", 1)[1])
SEAM_MATCH = 0.5
PX_AREA, PX_M = 0.5625, 0.75
STRIDE = 100_000


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def releases(tile):
    """Per-year frozen segmentation release used by this tile's frozen spatio-temporal graph."""
    man = json.loads((HERE / "graphs" / f"FROZEN_stgraph_2013_2023_{tile}_v1" / "MANIFEST.json").read_text())
    out = {}
    for k in man["inputs"]:
        f = k.split("/")[0]
        for y in ("2013", "2023"):
            if f.startswith(f"FROZEN_{y}_{tile}_"):
                out[y] = f
    return out


def load_tile(tile, year, rel):
    d = HERE / "final_segmentation" / rel
    man = json.loads((d / "MANIFEST.json").read_text())
    for f in ("segment_ids.tif", "nodes_final.csv", "nodes.csv"):
        assert sha(d / f) == man["files"][f], f"{rel}/{f} changed"
    with rasterio.open(d / "segment_ids.tif") as s:
        lab, tr = s.read(1).astype(np.int64), s.transform
    nf = pd.read_csv(d / "nodes_final.csv").set_index("segment_id")
    nn = pd.read_csv(d / "nodes.csv").set_index("node_id")
    img = rasterio.open(man["input_image"]["path"]).read([1, 2, 3]).transpose(1, 2, 0)
    img2 = cv2.resize(img, (lab.shape[1], lab.shape[0]), interpolation=cv2.INTER_CUBIC)
    return lab, tr, nf, nn, img2


def build_year(year, meta, rel_of):
    H = 2 * int(meta.y1.max()); W = 2 * int(meta.x1.max())
    g = meta.iloc[0]
    strip_tr = Affine(g.grid_affine_a / 2, g.grid_affine_b, g.grid_affine_c, g.grid_affine_d, g.grid_affine_e / 2, g.grid_affine_f)
    dist = np.full((H, W), np.inf, np.float32)
    M = np.zeros((H, W), np.int64)          # source key = tile_index*STRIDE + local id (owner tile)
    full = {}                              # per tile: label in strip coordinates (for overlap evidence)
    img = np.zeros((H, W, 3), np.uint8)
    attrs, checks = {}, {}
    yy, xx = np.mgrid[0:1024, 0:1024]
    for ti, tile in enumerate(meta.index):
        r = meta.loc[tile]
        lab, tr, nf, nn, img2 = load_tile(tile, year, rel_of[tile][year])
        r0, c0 = 2 * int(r.y0), 2 * int(r.x0)
        checks[tile] = bool(abs(tr.c - (strip_tr.c + c0 * strip_tr.a)) < 1e-6 and abs(tr.f - (strip_tr.f + r0 * strip_tr.e)) < 1e-6 and abs(tr.a - strip_tr.a) < 1e-9)
        cy, cx = r0 + lab.shape[0] / 2, c0 + lab.shape[1] / 2
        d = np.hypot(yy + r0 - cy, xx + c0 - cx).astype(np.float32)
        sl = (slice(r0, r0 + lab.shape[0]), slice(c0, c0 + lab.shape[1]))
        take = d < dist[sl]
        dist[sl][take] = d[take]
        M[sl][take] = ti * STRIDE + lab[take]
        img[sl][take] = img2[take]
        full[ti] = (sl, lab)
        for sid, row in nf.iterrows():
            k = ti * STRIDE + sid
            a = nn.loc[sid]
            attrs[k] = {"class_id": int(row.class_id), "class": row["class"], "confidence": float(row.confidence),
                        "rgb_rule": bool(row.rgb_rule), "source": row.source, "tile": tile, "tile_segment": int(sid),
                        **{f"p_{c}": float(a[f"p_{c}"]) for c in sf.CLASSES.values()}}
    assert all(checks.values()), f"tile georeferencing inconsistent with metadata placement: {checks}"

    # ---- seam re-joining
    parent = {}
    def find(k):
        while parent.get(k, k) != k:
            parent[k] = parent.get(parent[k], parent[k]); k = parent[k]
        return k
    owner = M // STRIDE
    pairs = Counter()
    for a, b, sl_a, sl_b, diff in v2.pair_boundaries(M):
        oa, ob = owner[sl_a][diff], owner[sl_b][diff]
        cross = oa != ob
        for x, y in zip(a[cross], b[cross]):
            pairs[(min(x, y), max(x, y))] += 1
    seam_rows = []
    for (ka, kb), n in pairs.items():
        ta, tb = ka // STRIDE, kb // STRIDE
        if attrs[ka]["class_id"] != attrs[kb]["class_id"]:
            seam_rows.append({"a": ka, "b": kb, "contact_px": n, "decision": "keep", "reason": "different class"}); continue
        (sla, laba), (slb, labb) = full[ta], full[tb]
        # overlap window in strip coordinates
        r0, r1 = max(sla[0].start, slb[0].start), min(sla[0].stop, slb[0].stop)
        c0, c1 = max(sla[1].start, slb[1].start), min(sla[1].stop, slb[1].stop)
        A = laba[r0 - sla[0].start:r1 - sla[0].start, c0 - sla[1].start:c1 - sla[1].start] == ka % STRIDE
        B = labb[r0 - slb[0].start:r1 - slb[0].start, c0 - slb[1].start:c1 - slb[1].start] == kb % STRIDE
        inter, small = int((A & B).sum()), int(min(A.sum(), B.sum()))
        ok = small > 0 and inter >= SEAM_MATCH * small
        seam_rows.append({"a": ka, "b": kb, "contact_px": n, "overlap_px": inter, "smaller_footprint_px": small,
                          "decision": "merge" if ok else "keep", "reason": "same object in overlap" if ok else "different objects in overlap"})
        if ok:
            ra, rb = find(ka), find(kb)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    keys = np.unique(M)
    root_arr = np.array([find(int(k)) for k in keys], np.int64)
    R = root_arr[np.searchsorted(keys, M)]
    # ---- connectivity: every root region split into 4-connected components (per bounding box)
    rk_vals, Rc = np.unique(R, return_inverse=True); Rc = Rc.reshape(R.shape) + 1
    lab = np.zeros(M.shape, np.int64); comp_root = {}; nxt = 1
    for i, sl in enumerate(ndimage.find_objects(Rc), 1):
        sel = Rc[sl] == i
        cl, n = ndimage.label(sel, structure=v2.FOUR)
        sub = lab[sl]
        sub[sel] = cl[sel] + nxt - 1
        for c in range(n):
            comp_root[nxt + c] = int(rk_vals[i - 1])
        nxt += n
    cls_arr = np.zeros(nxt, np.int64)
    for i, rk in comp_root.items():
        cls_arr[i] = attrs[rk]["class_id"]
    pre = int(lab.max())
    lab = sf.merge_small(lab, cls_arr, PX_AREA)
    lab = v2.enforce_connectivity(lab.astype(np.int32), min_px=1).astype(np.int64)
    ids = np.unique(lab); lut = np.zeros(lab.max() + 1, np.int64); lut[ids] = np.arange(1, len(ids) + 1)
    lab = lut[lab]
    final_cls = {int(lut[i]): int(cls_arr[i]) for i in ids if i > 0}

    # ---- node attributes: pixel-weighted source attributes + recomputed geometry/colour
    S = np.searchsorted(keys, M)
    n = int(lab.max())
    cnt = np.bincount(lab.ravel(), minlength=n + 1).astype(np.float64)
    def wmean(vals):
        v = np.asarray(vals, np.float64)[S]
        return np.bincount(lab.ravel(), weights=v.ravel(), minlength=n + 1) / np.maximum(cnt, 1)
    karr = list(keys)
    P = {c: wmean([attrs[k][f"p_{c}"] for k in karr]) for c in sf.CLASSES.values()}
    conf = wmean([attrs[k]["confidence"] for k in karr])
    rule = wmean([float(attrs[k]["rgb_rule"]) for k in karr])
    lineage = defaultdict(Counter)
    flat_l, flat_s = lab.ravel(), S.ravel()
    pair_key = flat_l * len(karr) + flat_s
    u, c = np.unique(pair_key, return_counts=True)
    for kk, cc in zip(u, c):
        lineage[int(kk // len(karr))][karr[int(kk % len(karr))]] += int(cc)
    geoms = {}
    for gj, v in features.shapes(lab.astype(np.int32), mask=lab > 0, transform=strip_tr, connectivity=4):
        v = int(v); gg = make_valid(shape(gj)); geoms[v] = gg if v not in geoms else make_valid(geoms[v].union(gg))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    rows = []
    for nid, sl in enumerate(ndimage.find_objects(lab), 1):
        m = lab[sl] == nid
        g = geoms[nid]; rgb = img[sl][m].astype(np.float32); hv = hsv[sl][m].astype(np.float32)
        ip = g.representative_point(); ce = g.centroid
        cid = final_cls[nid]; name = sf.CLASSES[cid]
        lin = lineage[nid]
        src_tiles = sorted({attrs[k]["tile"] for k in lin})
        rows.append({"segment_id": nid, "class_id": cid, "class": name, "area_m2": round(g.area, 2),
                     "geom_parts": 1 if g.geom_type == "Polygon" else len(g.geoms), **v2.shape_metrics(g),
                     "centroid_x": round(ce.x, 3), "centroid_y": round(ce.y, 3), "node_x": round(ip.x, 3), "node_y": round(ip.y, 3),
                     "mean_r": rgb[:, 0].mean(), "mean_g": rgb[:, 1].mean(), "mean_b": rgb[:, 2].mean(),
                     "std_r": rgb[:, 0].std(), "std_g": rgb[:, 1].std(), "std_b": rgb[:, 2].std(),
                     "mean_h": hv[:, 0].mean(), "mean_s": hv[:, 1].mean(), "mean_v": hv[:, 2].mean(),
                     **{f"p_{c}": round(float(P[c][nid]), 4) for c in sf.CLASSES.values()},
                     "confidence": round(float(conf[nid]), 4), "rgb_rule": bool(rule[nid] >= 0.5),
                     "source": "seam_merged" if len({attrs[k]["tile"] for k in lin}) > 1 else attrs[max(lin, key=lin.get)]["source"],
                     "source_tiles": "|".join(src_tiles),
                     "lineage": "|".join(f"{attrs[k]['tile']}:{attrs[k]['tile_segment']}:{v}" for k, v in lin.most_common())})
    nodes = pd.DataFrame(rows)
    touch = sf.edge_table(lab.astype(np.int32), nodes.rename(columns={"segment_id": "node_id"}), PX_M)
    E = bg.build_edges(nodes, touch, geoms)
    info = {"tiles": len(meta), "strip_px": [H, W], "source_segments": int(len(keys)), "seam_pairs": len(seam_rows),
            "seam_merges": int(sum(r["decision"] == "merge" for r in seam_rows)), "components_before_cleanup": pre,
            "segments": n, "edges": len(E), "edges_by_type": E.edge_type.value_counts().to_dict(),
            "georef_consistent": checks}
    return lab, img, strip_tr, nodes, E, geoms, pd.DataFrame(seam_rows), info, owner


def write_year(out, year, lab, img, tr, nodes, E, seams, info, crs):
    d = out / year; d.mkdir(parents=True, exist_ok=True)
    prof = {"driver": "GTiff", "height": lab.shape[0], "width": lab.shape[1], "crs": crs, "transform": tr, "compress": "deflate", "tiled": True}
    with rasterio.open(d / "segment_ids.tif", "w", count=1, dtype="int32", nodata=0, **prof) as s:
        s.write(lab.astype(np.int32), 1)
    with rasterio.open(d / "image_2x.tif", "w", count=3, dtype="uint8", **prof) as s:
        s.write(img.transpose(2, 0, 1))
    nodes.to_csv(d / "nodes.csv", index=False); E.to_csv(d / "edges.csv", index=False); seams.to_csv(d / "seam_decisions.csv", index=False)
    G = nx.Graph(year=year, strip=NAME)
    for r in nodes.drop(columns=["lineage"]).to_dict("records"):
        G.add_node(int(r["segment_id"]), **{k: (float(v) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (int, np.integer, bool)) else str(v))) for k, v in r.items() if k != "segment_id"})
    for r in E.to_dict("records"):
        G.add_edge(int(r["source"]), int(r["target"]), **{k: (float(v) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (int, np.integer)) else str(v))) for k, v in r.items() if k not in ("source", "target")})
    nx.write_graphml(G, d / "graph.graphml")
    chk = {"every_pixel_assigned": bool((lab > 0).all()), "segments_single_polygon": bool((nodes.geom_parts == 1).all()),
           "graph_connected": nx.is_connected(G), "areas_sum_to_strip": abs(nodes.area_m2.sum() - lab.size * PX_AREA) < 1.0,
           "georef_consistent_all_tiles": all(info["georef_consistent"].values())}
    info["checks"] = chk
    (d / "summary.json").write_text(json.dumps(info, indent=2, default=int), encoding="utf-8")
    return G


def main():
    meta = pd.read_csv(META).set_index("tile_id").loc[TILES]
    rel_of = {t: releases(t) for t in TILES}
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for year in ("2013", "2023"):
        lab, img, tr, nodes, E, geoms, seams, info, owner = build_year(year, meta, rel_of)
        crs = rasterio.open(HERE / "final_segmentation" / rel_of[TILES[0]][year] / "segment_ids.tif").crs
        G = write_year(OUT, year, lab, img, tr, nodes, E, seams, info, crs)
        res[year] = (lab, nodes, E, img, tr, owner)
        print(year, json.dumps({k: v for k, v in info.items() if k != "georef_consistent"}, default=int), flush=True)
    np.save(OUT / "tile_owner.npy", res["2023"][5].astype(np.int8))

    # ---- temporal layer on the strip (frozen rules)
    lab13, n13 = res["2013"][0], res["2013"][1].set_index("segment_id")
    lab23, n23 = res["2023"][0], res["2023"][1].set_index("segment_id")
    T, C, roof = tl.build_layer(lab13, n13, lab23, n23)
    T.to_csv(OUT / "temporal_edges.csv", index=False); C.to_csv(OUT / "temporal_components.csv", index=False)

    # ---- spatio-temporal graph (same relations/features as build_stgraph)
    bounds = (res["2023"][4].c, res["2023"][4].f + res["2023"][4].e * lab23.shape[0], res["2023"][4].c + res["2023"][4].a * lab23.shape[1], res["2023"][4].f)
    Fs, N = [], []
    for y in ("2013", "2023"):
        nd = res[y][1].copy(); nd["year"] = y
        F = bg.node_features(nd.assign(confidence=nd.confidence, rgb_rule=nd.rgb_rule), bounds)
        Fs.append(F); N.append(nd)
    nodes = pd.concat(N, ignore_index=True); nodes.insert(0, "gid", np.arange(len(nodes)))
    nodes.insert(1, "key", [f"{y}:{NAME}:{s}" for y, s in zip(nodes.year, nodes.segment_id)])
    gid = {(y, s): g for g, y, s in zip(nodes.gid, nodes.year, nodes.segment_id)}
    feat_names = [c for c in Fs[0].columns if c != "segment_id"]
    X = np.concatenate([f[feat_names].values for f in Fs]).astype(np.float32)
    cont = [i for i, nme in enumerate(feat_names) if not nme.startswith("class_") and nme != "rule_based_class"]
    mu, sd = X[:, cont].mean(0), X[:, cont].std(0) + 1e-6
    Xn = X.copy(); Xn[:, cont] = (X[:, cont] - mu) / sd
    Xn = np.concatenate([Xn, (nodes.year == "2023").values[:, None].astype(np.float32)], 1)
    rows = []
    for y in ("2013", "2023"):
        for r in res[y][2].to_dict("records"):
            rows.append({"relation": f"spatial_{r['edge_type']}", "layer": "spatial", "year": y, "src": gid[(y, r["source"])], "dst": gid[(y, r["target"])],
                         **{k: r[k] for k in bg_spat()}})
    conf_ord = {"high": 2, "medium": 1, "low": 0}
    for r in T.to_dict("records"):
        rows.append({"relation": f"temporal_{r['link_type']}", "layer": "temporal", "year": "2013->2023",
                     "src": gid[("2013", r["node_2013"])], "dst": gid[("2023", r["node_2023"])], "confidence": r["confidence"],
                     "iou": r["iou"], "frac_2013": r["frac_2013"], "frac_2023": r["frac_2023"], "overlap_score": r["overlap_score"],
                     "shared_m2": r["shared_m2"], "class_agreement": int(r["class_agreement"]), "confidence_ord": conf_ord[r["confidence"]],
                     "frame_roof": int(r["frame"] == "roof"), "roof_shift_dx_m": r["roof_shift_dx_m"], "roof_shift_dy_m": r["roof_shift_dy_m"]})
    E = pd.DataFrame(rows); E.insert(0, "eid", np.arange(len(E)))
    nodes.drop(columns=["lineage"]).to_csv(OUT / "stgraph_nodes.csv", index=False)
    nodes[["gid", "key", "lineage"]].to_csv(OUT / "stgraph_node_lineage.csv", index=False)
    E.to_csv(OUT / "stgraph_edges.csv", index=False)
    rels = ["spatial_touches", "spatial_near", "temporal_identity", "temporal_contained_in", "temporal_overlap"]
    torch.save({"x": torch.from_numpy(Xn), "node_feature_names": feat_names + ["year_2023"], "edge_index": torch.from_numpy(E[["src", "dst"]].values.T.astype(np.int64)),
                "edge_type": torch.from_numpy(E.relation.map({r: i for i, r in enumerate(rels)}).values.astype(np.int64)), "relations": rels,
                "y_class": torch.from_numpy((nodes.class_id.values - 1).astype(np.int64)), "year": torch.from_numpy((nodes.year == "2023").values.astype(np.int64)),
                "pos": torch.from_numpy(nodes[["node_x", "node_y"]].values), "keys": nodes.key.tolist()}, OUT / "stgraph_tensors.pt")
    yr = nodes.year.values
    Et, Es = E[E.layer == "temporal"], E[E.layer == "spatial"]
    G = nx.Graph(); G.add_nodes_from(nodes.gid); G.add_edges_from(zip(E.src, E.dst))
    v = {"stgraph_nodes": len(nodes), "stgraph_edges": len(E), "by_relation": E.relation.value_counts().to_dict(),
         "temporal_by_match_type": C.match_type.value_counts().to_dict(),
         "checks": {"spatial_within_year": bool((yr[Es.src] == yr[Es.dst]).all()),
                    "temporal_2013_to_2023": bool(((yr[Et.src] == "2013") & (yr[Et.dst] == "2023")).all()),
                    "no_duplicates": not E.duplicated(["src", "dst"]).any(), "features_finite": bool(np.isfinite(Xn).all()),
                    "connected": nx.is_connected(G), "every_node_in_one_temporal_component": int(C.n_2013.sum()) == len(n13) and int(C.n_2023.sum()) == len(n23)}}
    v["status"] = "PASS" if all(v["checks"].values()) else "FAIL"
    (OUT / "stgraph_validation.json").write_text(json.dumps(v, indent=2, default=int), encoding="utf-8")
    print(json.dumps(v, indent=1, default=int))


def bg_spat():
    return ["shared_boundary_m", "gap_m", "distance_m", "boundary_frac_source", "boundary_frac_target", "color_contrast", "same_class", "dir_sin", "dir_cos"]


if __name__ == "__main__":
    main()
