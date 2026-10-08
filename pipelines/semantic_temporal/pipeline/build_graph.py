"""Build the region graph from a FROZEN segmentation release (read-only input).

Nodes  one per segment (nodes_final.csv), position = node_x/node_y (inside the segment).
Edges  touches : segments sharing a boundary (from the frozen edges.csv, verified against the raster)
       near    : building-building pairs that do not touch but lie within NEAR_M (edge-to-edge),
                 at most NEAR_K per building; captures the arrangement of houses separated by yards
                 and streets. Separate edge type, can be dropped.

Outputs (graphs/<release>/)
  nodes.csv, edges.csv          raw node / edge attributes and features (one row per node / undirected edge)
  graph.graphml                 NetworkX graph with all attributes (Gephi, QGIS, networkx)
  graph.gpkg                    layers: nodes (points), segments (polygons), edges (lines)
  graph_tensors.pt / .npz       PyTorch-Geometric layout: x, edge_index (both directions), edge_attr,
                                edge_type, y, pos, node ids, feature names, normalisation stats
  feature_spec.json             definition of every node and edge feature
  graph_validation.json         graph checks; graph_overview.png
"""
from __future__ import annotations

from osgeo import gdal, ogr, osr  # noqa: F401
import argparse
import hashlib
import json
import math
from pathlib import Path

import fonts  # package: portable font paths

import cv2
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
import torch
from PIL import Image, ImageDraw, ImageFont
from shapely import STRtree, from_wkt
from shapely.geometry import LineString, Point

HERE = Path(__file__).resolve().parent
FROZEN_ROOT = HERE / "final_segmentation"
OUT_ROOT = HERE / "graphs"
CLASSES = {1: "building", 2: "road", 3: "tree", 4: "grass_shrub", 5: "cropland", 6: "bare_soil", 7: "paved_developed", 8: "water"}
COLORS = {1: (230, 57, 70), 2: (120, 120, 130), 3: (27, 140, 60), 4: (150, 200, 90), 5: (238, 200, 70),
          6: (196, 150, 100), 7: (170, 140, 200), 8: (40, 120, 230)}
NEAR_M, NEAR_K = 25.0, 6
EDGE_TYPES = {"touches": 0, "near": 1}


def sha256(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()


def load_release(release: Path):
    nodes = pd.read_csv(release / "nodes_final.csv")
    feats = pd.read_csv(release / "nodes.csv").rename(columns={"node_id": "segment_id"})
    keep = ["segment_id", "compactness", "solidity", "elongation", "orientation_deg", "mean_r", "mean_g", "mean_b",
            "std_r", "std_g", "std_b", "mean_h", "mean_s", "mean_v"] + [f"p_{c}" for c in CLASSES.values()]
    nodes = nodes.merge(feats[keep], on="segment_id", how="left", validate="one_to_one")
    edges = pd.read_csv(release / "edges.csv")
    return nodes.sort_values("segment_id").reset_index(drop=True), edges


def node_features(nodes, bounds):
    x0, y0, x1, y1 = bounds
    f = pd.DataFrame({"segment_id": nodes.segment_id})
    for cid, name in CLASSES.items():
        f[f"class_{name}"] = (nodes.class_id == cid).astype(np.float32)
    f["log_area"] = np.log(nodes.area_m2)
    f["log_perimeter"] = np.log(nodes.perimeter_m)
    for c in ("compactness", "solidity"):
        f[c] = nodes[c]
    f["log_elongation"] = np.log(nodes.elongation.clip(lower=1.0))
    th = np.deg2rad(nodes.orientation_deg) * 2          # orientation is axial (0 == 180 deg)
    f["orient_sin2"], f["orient_cos2"] = np.sin(th), np.cos(th)
    for c in ("mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b", "mean_s", "mean_v"):
        f[c] = nodes[c] / 255.0
    h = np.deg2rad(nodes.mean_h * 2)                    # OpenCV hue 0-180 -> circular
    f["hue_sin"], f["hue_cos"] = np.sin(h), np.cos(h)
    for name in CLASSES.values():
        f[f"p_{name}"] = nodes[f"p_{name}"]
    f["confidence"] = nodes.confidence
    f["rule_based_class"] = nodes.rgb_rule.astype(np.float32)
    f["pos_x"] = (nodes.node_x - x0) / (x1 - x0)
    f["pos_y"] = (nodes.node_y - y0) / (y1 - y0)
    return f


def build_edges(nodes, edges_touch, geoms):
    pos = nodes.set_index("segment_id")[["node_x", "node_y"]]
    per = nodes.set_index("segment_id").perimeter_m
    rgb = nodes.set_index("segment_id")[["mean_r", "mean_g", "mean_b"]]
    rows = []
    for e in edges_touch.itertuples():
        rows.append({"source": int(e.source), "target": int(e.target), "edge_type": "touches",
                     "shared_boundary_m": e.shared_boundary_m, "gap_m": 0.0})
    # near edges between buildings
    bids = nodes.segment_id[nodes.class_id == 1].tolist()
    bgeoms = [geoms[i] for i in bids]
    tree = STRtree(bgeoms)
    touching = {(min(a, b), max(a, b)) for a, b in zip(edges_touch.source, edges_touch.target)}
    added = set()
    for i, b in enumerate(bids):
        cand = []
        for j in tree.query(bgeoms[i].buffer(NEAR_M)):
            o = bids[j]
            if o == b or (min(b, o), max(b, o)) in touching:
                continue
            d = bgeoms[i].distance(bgeoms[j])
            if d <= NEAR_M:
                cand.append((d, o))
        for d, o in sorted(cand)[:NEAR_K]:
            key = (min(b, o), max(b, o))
            if key not in added:
                added.add(key)
                rows.append({"source": key[0], "target": key[1], "edge_type": "near", "shared_boundary_m": 0.0, "gap_m": round(d, 2)})
    E = pd.DataFrame(rows)
    ps, pt = pos.loc[E.source].values, pos.loc[E.target].values
    d = np.hypot(*(pt - ps).T)
    ang = np.arctan2(pt[:, 1] - ps[:, 1], pt[:, 0] - ps[:, 0])
    E["distance_m"] = d.round(2)
    E["dir_sin"], E["dir_cos"] = np.sin(ang).round(4), np.cos(ang).round(4)
    E["boundary_frac_source"] = (E.shared_boundary_m / per.loc[E.source].values).round(4)
    E["boundary_frac_target"] = (E.shared_boundary_m / per.loc[E.target].values).round(4)
    E["color_contrast"] = (np.linalg.norm(rgb.loc[E.source].values - rgb.loc[E.target].values, axis=1) / 255.0).round(4)
    cl = nodes.set_index("segment_id")["class"]
    E["class_source"], E["class_target"] = cl.loc[E.source].values, cl.loc[E.target].values
    E["same_class"] = (E.class_source == E.class_target).astype(int)
    # buildings meeting only at a corner point: no shared boundary (not 4-adjacent), gap 0
    E["corner_contact"] = ((E.edge_type == "near") & (E.gap_m <= 0)).astype(int)
    E.insert(0, "edge_id", np.arange(1, len(E) + 1))
    return E


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", default="FROZEN_2023_tile_00000_v1")
    args = ap.parse_args()
    release = FROZEN_ROOT / args.release
    manifest = json.loads((release / "MANIFEST.json").read_text())
    report = json.loads((release / "validation_report.json").read_text())
    if report["status"] != "PASS":
        raise SystemExit("Release did not pass validation; refusing to build a graph from it.")
    # provenance: inputs must match the frozen manifest
    for f in ("nodes_final.csv", "nodes.csv", "edges.csv", "segment_ids.tif"):
        if sha256(release / f) != manifest["files"][f]:
            raise SystemExit(f"{f} does not match the frozen manifest")
    out = OUT_ROOT / args.release.replace("FROZEN_", "graph_")
    out.mkdir(parents=True, exist_ok=True)

    nodes, edges_touch = load_release(release)
    geoms = {r.segment_id: from_wkt(r.geometry_wkt) for r in nodes.itertuples()}
    with rasterio.open(release / "segment_ids.tif") as s:
        b = s.bounds; crs_wkt = s.crs.to_wkt(); seg = s.read(1)
    bounds = (b.left, b.bottom, b.right, b.top)

    F = node_features(nodes, bounds)
    E = build_edges(nodes, edges_touch, geoms)

    # ---- tensors (PyG layout)
    feat_names = [c for c in F.columns if c != "segment_id"]
    X = F[feat_names].values.astype(np.float32)
    cont = [i for i, n in enumerate(feat_names) if not n.startswith("class_") and n != "rule_based_class"]
    mu = X[:, cont].mean(0); sd = X[:, cont].std(0) + 1e-6
    Xn = X.copy(); Xn[:, cont] = (X[:, cont] - mu) / sd
    idx = {sid: i for i, sid in enumerate(nodes.segment_id)}
    s = E.source.map(idx).values; t = E.target.map(idx).values
    edge_index = np.stack([np.r_[s, t], np.r_[t, s]]).astype(np.int64)
    efeat = ["shared_boundary_m", "gap_m", "distance_m", "boundary_frac_source", "boundary_frac_target", "color_contrast", "same_class"]
    EA = E[efeat].values.astype(np.float32)
    EA_rev = EA.copy(); EA_rev[:, [3, 4]] = EA[:, [4, 3]]                      # swap the per-node fractions
    dirs = E[["dir_sin", "dir_cos"]].values.astype(np.float32)
    edge_attr = np.r_[np.c_[EA, dirs], np.c_[EA_rev, -dirs]]
    edge_type = np.r_[E.edge_type.map(EDGE_TYPES).values, E.edge_type.map(EDGE_TYPES).values].astype(np.int64)
    y = (nodes.class_id.values - 1).astype(np.int64)
    pos = nodes[["node_x", "node_y"]].values.astype(np.float64)
    tensors = {"x": torch.from_numpy(Xn), "x_raw": torch.from_numpy(X), "edge_index": torch.from_numpy(edge_index),
               "edge_attr": torch.from_numpy(edge_attr), "edge_type": torch.from_numpy(edge_type), "y": torch.from_numpy(y),
               "pos": torch.from_numpy(pos), "segment_id": torch.from_numpy(nodes.segment_id.values),
               "node_feature_names": feat_names, "edge_feature_names": efeat + ["dir_sin", "dir_cos"],
               "edge_types": EDGE_TYPES, "class_names": list(CLASSES.values()),
               "norm_mean": mu.tolist(), "norm_std": sd.tolist(), "normalized_feature_idx": cont,
               "release": args.release}
    torch.save(tensors, out / "graph_tensors.pt")
    np.savez_compressed(out / "graph_tensors.npz", x=Xn, x_raw=X, edge_index=edge_index, edge_attr=edge_attr,
                        edge_type=edge_type, y=y, pos=pos, segment_id=nodes.segment_id.values)

    # ---- tables
    node_out = nodes[["segment_id", "node_key", "tile_id", "year", "class_id", "class", "area_m2", "perimeter_m",
                      "centroid_x", "centroid_y", "node_x", "node_y", "confidence", "confidence_type", "source"]].merge(F, on="segment_id", suffixes=("", "_feat"))
    deg = pd.concat([E.source, E.target]).value_counts()
    deg_t = pd.concat([E[E.edge_type == "touches"].source, E[E.edge_type == "touches"].target]).value_counts()
    node_out["degree"] = node_out.segment_id.map(deg).fillna(0).astype(int)
    node_out["degree_touches"] = node_out.segment_id.map(deg_t).fillna(0).astype(int)
    node_out.to_csv(out / "nodes.csv", index=False)
    E.to_csv(out / "edges.csv", index=False)

    # ---- networkx / graphml
    G = nx.Graph(release=args.release, crs="EPSG:32636", tile=nodes.tile_id.iloc[0], year=str(nodes.year.iloc[0]))
    for r in node_out.to_dict("records"):
        G.add_node(int(r["segment_id"]), **{k: (float(v) if isinstance(v, (np.floating, float)) else (int(v) if isinstance(v, (np.integer, bool)) else str(v)))
                                           for k, v in r.items() if k != "segment_id"})
    for r in E.to_dict("records"):
        G.add_edge(int(r["source"]), int(r["target"]), **{k: (float(v) if isinstance(v, (np.floating, float)) else (int(v) if isinstance(v, np.integer) else str(v)))
                                                         for k, v in r.items() if k not in ("source", "target")})
    nx.write_graphml(G, out / "graph.graphml")

    # ---- gpkg (nodes points, segments polygons, edges lines)
    gpkg = out / "graph.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    ds = ogr.GetDriverByName("GPKG").CreateDataSource(str(gpkg))
    srs = osr.SpatialReference(); srs.ImportFromWkt(crs_wkt)
    for lname, gtype, table, geom_fn in (
            ("nodes", ogr.wkbPoint, node_out[["segment_id", "class", "area_m2", "confidence", "source", "degree"]],
             lambda r: Point(r["node_x"], r["node_y"])),
            ("segments", ogr.wkbPolygon, node_out[["segment_id", "class", "area_m2", "confidence", "source", "degree"]],
             lambda r: geoms[r["segment_id"]]),
            ("edges", ogr.wkbLineString, E[["edge_id", "source", "target", "edge_type", "shared_boundary_m", "gap_m", "distance_m", "same_class"]],
             None)):
        lyr = ds.CreateLayer(lname, srs, gtype)
        for c, dt in table.dtypes.items():
            lyr.CreateField(ogr.FieldDefn(c, ogr.OFTInteger if dt.kind in "iub" else (ogr.OFTReal if dt.kind == "f" else ogr.OFTString)))
        lyr.StartTransaction()
        P = node_out.set_index("segment_id")[["node_x", "node_y"]]
        for r in table.to_dict("records"):
            f = ogr.Feature(lyr.GetLayerDefn())
            for c, v in r.items():
                f.SetField(c, int(v) if isinstance(v, (np.integer, bool)) else v)
            if geom_fn is None:
                g = LineString([tuple(P.loc[r["source"]]), tuple(P.loc[r["target"]])])
            else:
                g = geom_fn(node_out.loc[node_out.segment_id == r["segment_id"]].iloc[0])
            f.SetGeometry(ogr.CreateGeometryFromWkb(g.wkb))
            lyr.CreateFeature(f)
        lyr.CommitTransaction()
    ds = None

    # ---- validation of the graph against the frozen release
    touch = E[E.edge_type == "touches"]
    checks = {
        "nodes_equal_segments": G.number_of_nodes() == len(nodes) == int(seg.max()),
        "touch_edges_equal_frozen_adjacency": len(touch) == len(edges_touch) and
            set(zip(touch.source, touch.target)) == set(zip(edges_touch.source, edges_touch.target)),
        "no_self_loops": nx.number_of_selfloops(G) == 0,
        "no_duplicate_edges": len(E) == G.number_of_edges(),
        "connected": nx.is_connected(G),
        "no_isolated_nodes": nx.number_of_isolates(G) == 0,
        "near_edges_only_between_buildings": bool((E[E.edge_type == "near"][["class_source", "class_target"]] == "building").all().all()),
        "near_edges_not_in_frozen_adjacency": not (set(zip(E[E.edge_type == "near"].source, E[E.edge_type == "near"].target))
                                                   & {(min(a, b), max(a, b)) for a, b in zip(edges_touch.source, edges_touch.target)}),
        "near_edges_share_no_boundary": bool((E[E.edge_type == "near"].shared_boundary_m == 0).all()),
        "features_finite": bool(np.isfinite(Xn).all() and np.isfinite(edge_attr).all()),
        "edge_index_in_range": bool(edge_index.min() >= 0 and edge_index.max() < len(nodes)),
        "edge_index_symmetric": edge_index.shape[1] == 2 * len(E),
        "graphml_roundtrip": nx.read_graphml(out / "graph.graphml").number_of_edges() == len(E),
    }
    degs = np.array([d for _, d in G.degree()])
    summary = {"release": args.release, "release_validation": report["status"],
               "nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
               "edges_by_type": E.edge_type.value_counts().to_dict(),
               "node_features": len(feat_names), "edge_features": len(efeat) + 2,
               "degree": {"min": int(degs.min()), "median": float(np.median(degs)), "mean": round(float(degs.mean()), 2), "max": int(degs.max())},
               "density": round(nx.density(G), 5), "avg_clustering": round(nx.average_clustering(G), 3),
               "diameter_hops": nx.diameter(G) if nx.is_connected(G) else None,
               "same_class_edge_share": round(float(E.same_class.mean()), 3),
               "near_edges_corner_contact": int(E.corner_contact.sum()),
               "class_pairs_touches_top10": touch.assign(p=[" | ".join(sorted(x)) for x in zip(touch.class_source, touch.class_target)]).p.value_counts().head(10).to_dict(),
               "checks": checks, "status": "PASS" if all(checks.values()) else "FAIL"}
    (out / "graph_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    spec = {"node_features": feat_names, "normalized": "z-score on continuous features (stats in graph_tensors.pt); one-hot class and rule flag unscaled",
            "edge_features": efeat + ["dir_sin", "dir_cos"], "edge_types": EDGE_TYPES, "near_edge_rule": f"building-building, not touching, edge-to-edge <= {NEAR_M} m, <= {NEAR_K} per building",
            "label_y": "class_id - 1 (0..7) in order " + ", ".join(CLASSES.values()), "pos": "node_x, node_y (EPSG:32636, inside each segment)"}
    (out / "feature_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")

    render(release, out, node_out, E, seg)
    print(json.dumps({k: v for k, v in summary.items() if k != "class_pairs_touches_top10"}, indent=1))


def render(release, out, node_out, E, seg):
    run_log = json.loads((release / "run_log.json").read_text())
    with rasterio.open(run_log["image"]) as s:
        img = s.read([1, 2, 3]).transpose(1, 2, 0)
    with rasterio.open(release / "segment_ids.tif") as s:
        tr = s.transform
    S = 2048
    base = cv2.resize(img, (S, S), interpolation=cv2.INTER_CUBIC)
    base = (0.55 * base + 0.45 * 255 * 0.15).astype(np.uint8)
    segu = cv2.resize(seg, (S, S), interpolation=cv2.INTER_NEAREST)
    e = np.zeros(segu.shape, bool); e[:, 1:] |= segu[:, 1:] != segu[:, :-1]; e[1:] |= segu[1:] != segu[:-1]
    base[e] = (90, 90, 90)
    im = Image.fromarray(base); d = ImageDraw.Draw(im)
    k = S / seg.shape[0]
    P = {r.segment_id: (((r.node_x - tr.c) / tr.a) * k, ((r.node_y - tr.f) / tr.e) * k) for r in node_out.itertuples()}
    for r in E.itertuples():
        col = (255, 255, 255) if r.edge_type == "touches" else (255, 60, 220)
        d.line([P[r.source], P[r.target]], fill=col, width=2 if r.edge_type == "touches" else 1)
    for r in node_out.itertuples():
        x, y = P[r.segment_id]; rad = 4 + 2.2 * math.log10(max(r.area_m2, 10))
        d.ellipse((x - rad, y - rad, x + rad, y + rad), fill=COLORS[r.class_id], outline=(0, 0, 0))
    f = ImageFont.truetype(fonts.BOLD, 26)
    d.rectangle((0, 0, S, 44), fill=(20, 20, 24))
    d.text((14, 8), f"{len(node_out)} nodes · {len(E)} edges   (white = touches, magenta = near, building-building ≤ {NEAR_M:.0f} m)", font=f, fill="white")
    im.save(out / "graph_overview.png", optimize=True)


if __name__ == "__main__":
    main()
