"""Spatio-temporal graph from the frozen layers (no change classification).

Inputs (verified, read only)
  spatial 2013 : graphs/graph_2013_<tile>_v1   (built from FROZEN_2013_<tile>_v1)
  spatial 2023 : graphs/graph_2023_<tile>_v1   (built from FROZEN_2023_<tile>_v1)
  temporal     : graphs/FROZEN_temporal_2013_2023_<tile>_v3
  The spatial graph files must equal the hashes recorded as inputs in the frozen temporal manifest,
  and the segmentation releases must match their own manifests.

Graph
  nodes      every segment of both years; global index gid (2013 first); key "<year>:<tile>:<segment_id>"
  relations  spatial_touches, spatial_near            (within one year; from the spatial graphs)
             temporal_identity, temporal_contained_in  (primary temporal edges)
             temporal_overlap                          (secondary temporal edges; drop for a strict graph)
  node features: the 37 spatial-graph features, re-normalised JOINTLY over both years (z-score of
  continuous features; one-hot class and rule flag unscaled) + year flag.
Outputs (graphs/stgraph_2013_2023_<tile>/)
  nodes.csv, edges.csv, stgraph.graphml, stgraph.gpkg (nodes_2013, nodes_2023, spatial_edges,
  temporal_edges), stgraph_hetero.pt, stgraph_homo.pt/.npz, feature_spec.json, stgraph_validation.json,
  stgraph_overview.png, stgraph_zoom.png
"""
from __future__ import annotations

from osgeo import gdal, ogr, osr  # noqa: F401
import argparse
import hashlib
import json
from pathlib import Path

import fonts  # package: portable font paths

import cv2
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
import torch
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import LineString, Point

HERE = Path(__file__).resolve().parent
GRAPHS = HERE / "graphs"
FROZEN = HERE / "final_segmentation"
RELATIONS = ["spatial_touches", "spatial_near", "temporal_identity", "temporal_contained_in", "temporal_overlap"]
TEMP_EFEAT = ["iou", "frac_2013", "frac_2023", "overlap_score", "shared_m2", "class_agreement", "confidence_ord",
              "frame_roof", "roof_shift_dx_m", "roof_shift_dy_m"]
SPAT_EFEAT = ["shared_boundary_m", "gap_m", "distance_m", "boundary_frac_source", "boundary_frac_target",
              "color_contrast", "same_class", "dir_sin", "dir_cos"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


VER = {"2013": "v1", "2023": "v1"}


def versions_from_manifest(tman, tile):
    """Per-year segmentation release version recorded as input of the frozen temporal layer."""
    out = {}
    for key in tman["inputs"]:
        folder = key.split("/", 1)[0]
        for y in ("2013", "2023"):
            if folder.startswith(f"FROZEN_{y}_{tile}_"):
                out[y] = folder.rsplit("_", 1)[1]
    return out


def verify(tile):
    T = GRAPHS / f"FROZEN_temporal_2013_2023_{tile}_v3"
    tman = json.loads((T / "MANIFEST.json").read_text())
    assert json.loads((T / "validation_report.json").read_text())["status"] == "PASS"
    for f, h in tman["files"].items():
        assert sha(T / f) == h, f"temporal release file changed: {f}"
    n_checked = 0
    for key, h in tman["inputs"].items():
        folder, f = key.split("/", 1)
        base = GRAPHS / folder if folder.startswith("graph_") else FROZEN / folder
        assert sha(base / f) == h, f"input changed since the temporal layer was frozen: {key}"
        n_checked += 1
    VER.update(versions_from_manifest(tman, tile))
    return T, n_checked


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tile", default="tile_00000")
    args = ap.parse_args()
    tile = args.tile
    T, n_inputs = verify(tile)
    out = GRAPHS / f"stgraph_2013_2023_{tile}"
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- nodes
    N, Sp, X_raw, feat_names = [], [], [], None
    for year in ("2013", "2023"):
        g = GRAPHS / f"graph_{year}_{tile}_{VER[year]}"
        nd = pd.read_csv(g / "nodes.csv")
        t = torch.load(g / "graph_tensors.pt", weights_only=False)
        assert np.array_equal(t["segment_id"].numpy(), nd.segment_id.values)
        feat_names = feat_names or t["node_feature_names"]
        assert feat_names == t["node_feature_names"]
        X_raw.append(t["x_raw"].numpy())
        nd = nd.assign(year=year)
        N.append(nd)
        e = pd.read_csv(g / "edges.csv").assign(year=year)
        Sp.append(e)
    nodes = pd.concat(N, ignore_index=True)
    nodes.insert(0, "gid", np.arange(len(nodes)))
    nodes.insert(1, "key", [f"{y}:{tile}:{s}" for y, s in zip(nodes.year, nodes.segment_id)])
    gid = {(y, s): g for g, y, s in zip(nodes.gid, nodes.year, nodes.segment_id)}
    X = np.concatenate(X_raw).astype(np.float32)
    cont = [i for i, n in enumerate(feat_names) if not n.startswith("class_") and n != "rule_based_class"]
    mu, sd = X[:, cont].mean(0), X[:, cont].std(0) + 1e-6
    Xn = X.copy(); Xn[:, cont] = (X[:, cont] - mu) / sd
    year_flag = (nodes.year == "2023").values.astype(np.float32)[:, None]
    Xn = np.concatenate([Xn, year_flag], 1); X = np.concatenate([X, year_flag], 1)
    feat_all = feat_names + ["year_2023"]

    # temporal node attributes (structure only)
    C = pd.read_csv(T / "temporal_components.csv")
    mt = {}
    for r in C.itertuples():
        for y, col in (("2013", r.nodes_2013), ("2023", r.nodes_2023)):
            for s in str(col).split("|"):
                if s not in ("", "nan"):
                    mt[(y, int(s))] = r.match_type
    nodes["temporal_match_type"] = [mt[(y, s)] for y, s in zip(nodes.year, nodes.segment_id)]

    # ---------------- edges
    rows = []
    sp = pd.concat(Sp, ignore_index=True)
    for r in sp.to_dict("records"):
        rows.append({"relation": f"spatial_{r['edge_type']}", "layer": "spatial", "year": r["year"],
                     "src": gid[(r["year"], r["source"])], "dst": gid[(r["year"], r["target"])],
                     **{k: r[k] for k in SPAT_EFEAT}})
    te = pd.read_csv(T / "temporal_edges.csv")
    conf_ord = {"high": 2, "medium": 1, "low": 0}
    for r in te.to_dict("records"):
        rel = {"identity": "temporal_identity", "contained_in": "temporal_contained_in", "overlap": "temporal_overlap"}[r["link_type"]]
        rows.append({"relation": rel, "layer": "temporal", "year": "2013->2023",
                     "src": gid[("2013", r["node_2013"])], "dst": gid[("2023", r["node_2023"])],
                     "temporal_edge_id": r["temporal_edge_id"], "confidence": r["confidence"], "edge_role": r["edge_role"],
                     "match_type": r["match_type"], "frame": r["frame"], "contained_side": r.get("contained_side", ""),
                     "iou": r["iou"], "frac_2013": r["frac_2013"], "frac_2023": r["frac_2023"], "overlap_score": r["overlap_score"],
                     "shared_m2": r["shared_m2"], "class_agreement": int(r["class_agreement"]), "confidence_ord": conf_ord[r["confidence"]],
                     "frame_roof": int(r["frame"] == "roof"), "roof_shift_dx_m": r["roof_shift_dx_m"], "roof_shift_dy_m": r["roof_shift_dy_m"]})
    E = pd.DataFrame(rows)
    E.insert(0, "eid", np.arange(len(E)))
    E["src_key"] = nodes.key.values[E.src]; E["dst_key"] = nodes.key.values[E.dst]

    # ---------------- tables
    keep = ["gid", "key", "year", "segment_id", "class_id", "class", "area_m2", "perimeter_m", "node_x", "node_y",
            "centroid_x", "centroid_y", "confidence", "confidence_type", "source", "degree", "temporal_match_type"]
    nodes_out = nodes[keep].copy()
    nodes_out["degree_temporal_primary"] = nodes_out.gid.map(pd.concat([E[E.relation.isin(["temporal_identity", "temporal_contained_in"])].src,
                                                                         E[E.relation.isin(["temporal_identity", "temporal_contained_in"])].dst]).value_counts()).fillna(0).astype(int)
    nodes_out.to_csv(out / "nodes.csv", index=False)
    E.to_csv(out / "edges.csv", index=False)

    # ---------------- tensors: homogeneous
    rel_id = {r: i for i, r in enumerate(RELATIONS)}
    ei = E[["src", "dst"]].values.T.astype(np.int64)
    homo = {"x": torch.from_numpy(Xn), "x_raw": torch.from_numpy(X), "y": torch.from_numpy((nodes.class_id.values - 1).astype(np.int64)),
            "pos": torch.from_numpy(nodes[["node_x", "node_y"]].values.astype(np.float64)),
            "year": torch.from_numpy((nodes.year == "2023").values.astype(np.int64)),
            "edge_index": torch.from_numpy(ei), "edge_type": torch.from_numpy(E.relation.map(rel_id).values.astype(np.int64)),
            "relations": RELATIONS, "node_feature_names": feat_all, "node_keys": nodes.key.tolist(),
            "norm_mean": mu.tolist(), "norm_std": sd.tolist(), "normalized_feature_idx": cont,
            "note": "edge_index stores each edge once (spatial: source<target within a year; temporal: 2013->2023); add reverse edges for message passing"}
    torch.save(homo, out / "stgraph_homo.pt")
    np.savez_compressed(out / "stgraph_homo.npz", x=Xn, x_raw=X, edge_index=ei, edge_type=homo["edge_type"].numpy(),
                        y=homo["y"].numpy(), pos=homo["pos"].numpy(), year=homo["year"].numpy())

    # ---------------- tensors: heterogeneous (PyG HeteroData layout as a dict)
    off = {"2013": 0, "2023": int((nodes.year == "2013").sum())}
    hetero = {"node_types": ["2013", "2023"], "node_feature_names": feat_all, "relations": {}, "release": T.name}
    for y in ("2013", "2023"):
        m = (nodes.year == y).values
        hetero[f"x_{y}"] = torch.from_numpy(Xn[m]); hetero[f"y_{y}"] = homo["y"][m]; hetero[f"pos_{y}"] = homo["pos"][m]
        hetero[f"segment_id_{y}"] = torch.from_numpy(nodes.segment_id.values[m])
    for rel in RELATIONS:
        sub = E[E.relation == rel]
        if rel.startswith("spatial"):
            for y in ("2013", "2023"):
                s2 = sub[sub.year == y]
                idx = (s2[["src", "dst"]].values - off[y]).T.astype(np.int64)
                hetero["relations"][f"{y}|{rel}|{y}"] = {"edge_index": torch.from_numpy(idx),
                                                         "edge_attr": torch.from_numpy(s2[SPAT_EFEAT].values.astype(np.float32)),
                                                         "edge_attr_names": SPAT_EFEAT}
        else:
            idx = np.stack([sub.src.values - off["2013"], sub.dst.values - off["2023"]]).astype(np.int64)
            hetero["relations"][f"2013|{rel}|2023"] = {"edge_index": torch.from_numpy(idx),
                                                       "edge_attr": torch.from_numpy(sub[TEMP_EFEAT].values.astype(np.float32)),
                                                       "edge_attr_names": TEMP_EFEAT}
    torch.save(hetero, out / "stgraph_hetero.pt")

    # ---------------- graphml
    G = nx.Graph(tile=tile, years="2013,2023", temporal_release=T.name, relations=",".join(RELATIONS))
    for r in nodes_out.to_dict("records"):
        G.add_node(r["key"], **{k: (float(v) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (int, np.integer)) else str(v)))
                                for k, v in r.items() if k != "key"})
    for r in E.to_dict("records"):
        G.add_edge(r["src_key"], r["dst_key"], **{k: (float(v) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (int, np.integer)) else str(v)))
                                                  for k, v in r.items() if k not in ("src_key", "dst_key") and not (isinstance(v, float) and np.isnan(v))})
    nx.write_graphml(G, out / "stgraph.graphml")

    # ---------------- gpkg
    gpkg = out / "stgraph.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    with rasterio.open(FROZEN / f"FROZEN_2023_{tile}_{VER['2023']}" / "segment_ids.tif") as s:
        crs_wkt, tr = s.crs.to_wkt(), s.transform
    ds = ogr.GetDriverByName("GPKG").CreateDataSource(str(gpkg)); srs = osr.SpatialReference(); srs.ImportFromWkt(crs_wkt)
    P = nodes.set_index("gid")[["node_x", "node_y"]]
    def layer(name, gtype, df, geom):
        lyr = ds.CreateLayer(name, srs, gtype)
        for c, dt in df.dtypes.items():
            lyr.CreateField(ogr.FieldDefn(c, ogr.OFTInteger if dt.kind in "iub" else (ogr.OFTReal if dt.kind == "f" else ogr.OFTString)))
        lyr.StartTransaction()
        for r in df.to_dict("records"):
            f = ogr.Feature(lyr.GetLayerDefn())
            for c, v in r.items():
                if not (isinstance(v, float) and np.isnan(v)):
                    f.SetField(c, int(v) if isinstance(v, (np.integer, bool)) else v)
            f.SetGeometry(ogr.CreateGeometryFromWkb(geom(r).wkb)); lyr.CreateFeature(f)
        lyr.CommitTransaction()
    ncols = ["gid", "key", "segment_id", "class", "area_m2", "temporal_match_type", "degree_temporal_primary"]
    for y in ("2013", "2023"):
        layer(f"nodes_{y}", ogr.wkbPoint, nodes_out[nodes_out.year == y][ncols], lambda r: Point(P.loc[r["gid"]]))
    line = lambda r: LineString([tuple(P.loc[r["src"]]), tuple(P.loc[r["dst"]])])
    layer("spatial_edges", ogr.wkbLineString, E[E.layer == "spatial"][["eid", "relation", "year", "src", "dst", "shared_boundary_m", "distance_m"]], line)
    layer("temporal_edges", ogr.wkbLineString, E[E.layer == "temporal"][["eid", "relation", "src", "dst", "confidence", "match_type", "frame", "iou", "overlap_score"]], line)
    ds = None

    # ---------------- validation
    te_keys = set(zip(te.node_2013, te.node_2023))
    Et = E[E.layer == "temporal"]; Es = E[E.layer == "spatial"]
    yr = nodes.year.values
    checks = {
        "inputs_match_frozen_manifests": True,  # verify() asserts; reaching here means all matched
        "node_count": len(nodes) == sum(len(pd.read_csv(GRAPHS / f"graph_{y}_{tile}_{VER[y]}" / "nodes.csv")) for y in ("2013", "2023")) and nodes.key.is_unique,
        "spatial_edges_equal_per_year_graphs": all(
            set(zip(Es[Es.year == y].src - off[y], Es[Es.year == y].dst - off[y])) ==
            set(zip(pd.read_csv(GRAPHS / f"graph_{y}_{tile}_{VER[y]}" / "edges.csv").source.map(lambda s, y=y: gid[(y, s)] - off[y]),
                    pd.read_csv(GRAPHS / f"graph_{y}_{tile}_{VER[y]}" / "edges.csv").target.map(lambda s, y=y: gid[(y, s)] - off[y])))
            for y in ("2013", "2023")),
        "temporal_edges_equal_frozen_layer": len(Et) == len(te) and set(zip(nodes.segment_id.values[Et.src], nodes.segment_id.values[Et.dst])) == te_keys,
        "spatial_edges_within_one_year": bool((yr[Es.src] == yr[Es.dst]).all()),
        "temporal_edges_cross_years_2013_to_2023": bool(((yr[Et.src] == "2013") & (yr[Et.dst] == "2023")).all()),
        "no_self_loops_or_duplicates": bool((E.src != E.dst).all() and not E.duplicated(["src", "dst"]).any()),
        "features_finite": bool(np.isfinite(Xn).all() and all(np.isfinite(v["edge_attr"].numpy()).all() for v in hetero["relations"].values())),
        "joint_normalisation": bool(np.allclose(Xn[:, cont].mean(0), 0, atol=1e-4)),
        "hetero_indices_in_range": all(v["edge_index"].min() >= 0 and v["edge_index"][0].max() < (off["2023"] if k.startswith("2013") else len(nodes) - off["2023"])
                                       and v["edge_index"][1].max() < (off["2023"] if k.endswith("2013") else len(nodes) - off["2023"]) for k, v in hetero["relations"].items()),
        "graphml_roundtrip": nx.read_graphml(out / "stgraph.graphml").number_of_edges() == len(E),
    }
    G_strict = G.edge_subgraph([(u, v) for u, v, d in G.edges(data=True) if d["relation"] != "temporal_overlap"])
    summary = {"tile": tile, "inputs": {"temporal_release": T.name, "inputs_verified": n_inputs,
                                        "spatial_graphs": [f"graph_2013_{tile}_{VER['2013']}", f"graph_2023_{tile}_{VER['2023']}"]},
               "nodes": {"total": len(nodes), "2013": int((nodes.year == "2013").sum()), "2023": int((nodes.year == "2023").sum())},
               "edges": {"total": len(E), **E.relation.value_counts().to_dict()},
               "spatial_edges_by_year": Es.year.value_counts().to_dict(),
               "temporal_edges_by_confidence": Et.confidence.value_counts().to_dict(),
               "node_features": len(feat_all), "spatial_edge_features": len(SPAT_EFEAT), "temporal_edge_features": len(TEMP_EFEAT),
               "connected_components_full": nx.number_connected_components(G),
               "connected_components_without_temporal_overlap": nx.number_connected_components(G_strict.copy()) if G_strict.number_of_nodes() else None,
               "nodes_without_primary_temporal_edge": int((nodes_out.degree_temporal_primary == 0).sum()),
               "checks": checks, "status": "PASS" if all(checks.values()) else "FAIL"}
    (out / "stgraph_validation.json").write_text(json.dumps(summary, indent=2, default=int), encoding="utf-8")
    spec = {"node_features": feat_all, "normalisation": "joint z-score over both years on continuous features; stats in stgraph_homo.pt",
            "relations": RELATIONS, "spatial_edge_features": SPAT_EFEAT, "temporal_edge_features": TEMP_EFEAT,
            "confidence_ord": {"high": 2, "medium": 1, "low": 0}, "label_y": "class_id-1",
            "warning": "class one-hot and p_* features encode the label; drop them if y is a prediction target"}
    (out / "feature_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    render(out, nodes, E, tile, tr)
    render_zoom(out, nodes, E, tile, tr)
    print(json.dumps(summary, indent=1, default=int))


def render_zoom(out, nodes, E, tile, tr, win=(0, 0, 340)):
    """Same stacked view for one window (2x-grid row, col, size), with full-resolution imagery."""
    r0, c0, S = win
    COL = {1: (230, 57, 70), 2: (130, 130, 140), 3: (27, 140, 60), 4: (150, 200, 90), 5: (238, 200, 70),
           6: (196, 150, 100), 7: (170, 140, 200), 8: (40, 120, 230)}
    W = Hp = 820; gap, sh = 170, 0.35
    canvas = Image.new("RGB", (W + int(sh * Hp) + 80, 2 * Hp + gap + 130), (18, 18, 22))
    def plane_xy(year, px, py):
        return 40 + px * W + (1 - py) * sh * Hp, 60 + (0 if year == "2023" else Hp + gap) + py * Hp
    for y in ("2023", "2013"):
        man = json.loads((FROZEN / f"FROZEN_{y}_{tile}_{VER[y]}" / "MANIFEST.json").read_text())
        img = rasterio.open(man["input_image"]["path"]).read([1, 2, 3]).transpose(1, 2, 0)
        crop = cv2.resize(cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_CUBIC)[r0:r0 + S, c0:c0 + S], (W, Hp), interpolation=cv2.INTER_CUBIC)
        crop = (0.6 * crop + 0.4 * 30).astype(np.uint8)
        src = np.float32([[0, 0], [W, 0], [0, Hp]]); dst = np.float32([plane_xy(y, 0, 0), plane_xy(y, 1, 0), plane_xy(y, 0, 1)])
        M = cv2.getAffineTransform(src, dst)
        warped = cv2.warpAffine(crop, M, canvas.size); mask = cv2.warpAffine(np.ones((Hp, W), np.uint8), M, canvas.size)
        arr = np.array(canvas); arr[mask > 0] = warped[mask > 0]; canvas = Image.fromarray(arr)
    d = ImageDraw.Draw(canvas)
    def norm(x, yv):
        return ((x - tr.c) / tr.a - c0) / S, ((yv - tr.f) / tr.e - r0) / S
    inside = {r.gid: norm(r.node_x, r.node_y) for r in nodes.itertuples()}
    inside = {g: v for g, v in inside.items() if 0 <= v[0] <= 1 and 0 <= v[1] <= 1}
    yr = dict(zip(nodes.gid, nodes.year)); cls = dict(zip(nodes.gid, nodes.class_id))
    pos = {g: plane_xy(yr[g], *v) for g, v in inside.items()}
    styles = {"spatial_touches": ((210, 210, 210), 1), "spatial_near": ((255, 80, 220), 1),
              "temporal_overlap": ((120, 70, 70), 1), "temporal_identity": ((80, 230, 255), 2), "temporal_contained_in": ((255, 210, 60), 2)}
    for rel in styles:
        for r in E[E.relation == rel].itertuples():
            if r.src in pos and r.dst in pos:
                d.line([pos[r.src], pos[r.dst]], fill=styles[rel][0], width=styles[rel][1])
    for g, (x, y) in pos.items():
        d.ellipse((x - 6, y - 6, x + 6, y + 6), fill=COL[cls[g]], outline=(0, 0, 0))
    f = ImageFont.truetype(fonts.BOLD, 24); fs = ImageFont.truetype(fonts.REGULAR, 17)
    d.text((20, 15), f"Spatio-temporal graph, zoom ({S * 0.75:.0f} m window): {sum(1 for g in pos if yr[g]=='2013')} + {sum(1 for g in pos if yr[g]=='2023')} nodes", font=f, fill="white")
    d.text(plane_xy("2023", 0, 0), " 2023", font=f, fill="yellow"); d.text(plane_xy("2013", 0, 0), " 2013", font=f, fill="yellow")
    d.text((20, canvas.height - 45), "grey spatial touches · magenta spatial near · cyan temporal identity · yellow temporal contained-in · dark red temporal overlap (secondary)", font=fs, fill=(220, 220, 220))
    canvas.save(out / "stgraph_zoom.png", optimize=True)


def render(out, nodes, E, tile, tr):
    """Oblique stacked view: 2013 plane (bottom), 2023 plane (top), primary temporal edges between."""
    COL = {1: (230, 57, 70), 2: (130, 130, 140), 3: (27, 140, 60), 4: (150, 200, 90), 5: (238, 200, 70),
           6: (196, 150, 100), 7: (170, 140, 200), 8: (40, 120, 230)}
    imgs = {}
    for y in ("2013", "2023"):
        man = json.loads((FROZEN / f"FROZEN_{y}_{tile}_{VER[y]}" / "MANIFEST.json").read_text())
        imgs[y] = rasterio.open(man["input_image"]["path"]).read([1, 2, 3]).transpose(1, 2, 0)
    W, Hp, gap, sh = 1400, 560, 160, 0.35
    H = 2 * Hp + gap + 120
    canvas = Image.new("RGB", (W + int(sh * Hp) + 80, H), (18, 18, 22)); d = ImageDraw.Draw(canvas)
    def plane_xy(year, px, py):  # px,py in 0..1
        y0 = 60 + (0 if year == "2023" else Hp + gap)
        return 40 + px * W + (1 - py) * sh * Hp, y0 + py * Hp
    for y in ("2023", "2013"):
        im = cv2.resize(imgs[y], (W, Hp), interpolation=cv2.INTER_AREA)
        im = (0.55 * im + 0.45 * 30).astype(np.uint8)
        src = np.float32([[0, 0], [W, 0], [0, Hp]]); dst = np.float32([plane_xy(y, 0, 0), plane_xy(y, 1, 0), plane_xy(y, 0, 1)])
        warped = cv2.warpAffine(im, cv2.getAffineTransform(src, dst), canvas.size, borderValue=(0, 0, 0))
        mask = cv2.warpAffine(np.ones((Hp, W), np.uint8), cv2.getAffineTransform(src, dst), canvas.size)
        arr = np.array(canvas); arr[mask > 0] = warped[mask > 0]; canvas = Image.fromarray(arr); d = ImageDraw.Draw(canvas)
    norm = lambda x, yv: ((x - tr.c) / (tr.a * 1024), (yv - tr.f) / (tr.e * 1024))
    pos = {r.gid: plane_xy(r.year, *norm(r.node_x, r.node_y)) for r in nodes.itertuples()}
    for r in E[E.layer == "spatial"].itertuples():
        d.line([pos[r.src], pos[r.dst]], fill=(200, 200, 200) if r.relation == "spatial_touches" else (255, 80, 220), width=1)
    tcol = {"temporal_identity": (80, 230, 255), "temporal_contained_in": (255, 210, 60)}
    for r in E[E.relation.isin(tcol)].itertuples():
        d.line([pos[r.src], pos[r.dst]], fill=tcol[r.relation], width=1)
    for r in nodes.itertuples():
        x, y = pos[r.gid]; d.ellipse((x - 3, y - 3, x + 3, y + 3), fill=COL[r.class_id])
    f = ImageFont.truetype(fonts.BOLD, 24); fs = ImageFont.truetype(fonts.REGULAR, 18)
    d.text((20, 15), f"Spatio-temporal graph {tile}: {len(nodes)} nodes, {len(E)} edges", font=f, fill="white")
    d.text(plane_xy("2023", 0, 0), " 2023", font=f, fill="yellow"); d.text(plane_xy("2013", 0, 0), " 2013", font=f, fill="yellow")
    d.text((20, H - 40), "grey = spatial touches · magenta = spatial near · cyan = temporal identity · yellow = temporal contained-in · (secondary temporal overlap edges not drawn)", font=fs, fill=(220, 220, 220))
    canvas.save(out / "stgraph_overview.png", optimize=True)


if __name__ == "__main__":
    main()
