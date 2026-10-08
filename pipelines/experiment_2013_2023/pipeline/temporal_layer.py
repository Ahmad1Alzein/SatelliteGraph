"""Temporal correspondence layer between two frozen region graphs (no change classification).

Inputs  FROZEN_2013_<tile>_v1 and FROZEN_2023_<tile>_v1 (validated, checksum-verified) and their
        spatial graphs (graphs/graph_<year>_<tile>_v1). Nothing in those folders is modified.
Output  graphs/temporal_2013_2023_<tile>/ : ONLY temporal edges (2013 node <-> 2023 node).
        Spatial edges remain in the per-year graphs and are not copied here.

Temporal edge = overlap link between a 2013 node and a 2023 node on the shared 0.75 m grid:
  shared area >= 25 % of the smaller node and >= 6 px.
  frame 'roof'   building <-> building, after moving each 2013 building by the regional roof
                 displacement (view-angle lean) refined per building (ties -> smallest shift);
  frame 'ground' every other pair (any class combination) on the co-registered grid.
Edge attributes: shared area, IoU, fraction of each node covered, overlap_score =
  sqrt(frac_2013 * frac_2023), class of both ends, class agreement, roof shift applied.
Confidence (correspondence only, not change):
  high    same class and IoU >= 0.5
  medium  same class and overlap_score >= 0.3, or different class but overlap_score >= 0.5
          (the same region, labelled differently)
  low     everything else (weak or cross-class contact overlap)
  v2 adds link type 'contained_in' (after the v1 visual audit: 6 of 16 sampled unmatched buildings
  existed in both years but lay inside a much larger same-class node of the other year): same class
  and >= CONTAIN_FRAC of one node inside the other -> medium, primary.
  v3 (final): the contained node must also be >= CONTAIN_MIN_M2 (v2 re-audit: questionable
  contained_in links were 29-41 m2 fragments). Different-class containment is never promoted.
link_type: identity (high/medium by the rules above), contained_in, overlap (secondary).
Edge role: primary = high/medium (identity correspondence); secondary = low (overlap context).
Both partitions cover every pixel, so secondary overlaps chain almost all land nodes into one
component (444 x 439 on tile_00000); match types are therefore computed on PRIMARY edges only:
  one_to_one, one_to_many (1 x 2013 : n x 2023), many_to_one, many_to_many; nodes without a
  primary edge are no_primary_match_2013 / no_primary_match_2023 (they may still have secondary
  edges). Secondary edges carry match_type 'secondary_overlap'.
"""
from __future__ import annotations

from osgeo import gdal, ogr, osr  # noqa: F401
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import fonts  # package: portable font paths

import cv2
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw, ImageFont
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import LineString

import final_strip_qa as qa

HERE = Path(__file__).resolve().parent
FROZEN = HERE / "final_segmentation"
GRAPHS = HERE / "graphs"
PX = 0.5625
MIN_FRAC_SMALLER, MIN_PX = 0.25, 6
REL_VERSION = {"2013": "v1", "2023": "v1"}   # frozen segmentation release / graph used per year
CONTAIN_FRAC = 0.8
CONTAIN_MIN_M2 = 50.0
VERSION = "v3"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load(year, tile):
    rel = FROZEN / f"FROZEN_{year}_{tile}_{REL_VERSION[year]}"
    man = json.loads((rel / "MANIFEST.json").read_text())
    assert json.loads((rel / "validation_report.json").read_text())["status"] == "PASS"
    for f in ("segment_ids.tif", "nodes_final.csv"):
        assert sha256(rel / f) == man["files"][f], f"{rel.name}/{f} changed since freezing"
    g = GRAPHS / f"graph_{year}_{tile}_{REL_VERSION[year]}"
    assert json.loads((g / "graph_validation.json").read_text())["status"] == "PASS"
    with rasterio.open(rel / "segment_ids.tif") as s:
        lab = s.read(1).astype(np.int64); tr, crs = s.transform, s.crs
    nodes = pd.read_csv(g / "nodes.csv").set_index("segment_id")
    return rel, g, lab, nodes, tr, crs, Path(man["input_image"]["path"])


def overlaps(lab_a, lab_b, ids_a=None, ids_b=None):
    m = (lab_a > 0) & (lab_b > 0)
    if ids_a is not None:
        m &= np.isin(lab_a, ids_a) & np.isin(lab_b, ids_b)
    key = lab_a[m] * 10_000_000 + lab_b[m]
    u, c = np.unique(key, return_counts=True)
    aa, ab = np.bincount(lab_a.ravel()), np.bincount(lab_b.ravel())
    d = pd.DataFrame({"node_2013": u // 10_000_000, "node_2023": u % 10_000_000, "shared_px": c})
    d["area_2013_px"], d["area_2023_px"] = aa[d.node_2013], ab[d.node_2023]
    return d[(d.shared_px >= MIN_FRAC_SMALLER * np.minimum(d.area_2013_px, d.area_2023_px)) & (d.shared_px >= MIN_PX)]


def build_layer(lab13, n13, lab23, n23):
    b13 = n13.index[n13["class"] == "building"].values
    b23 = n23.index[n23["class"] == "building"].values
    fdy, fdx, prof = qa.roof_offset_field(np.isin(lab13, b13), np.isin(lab23, b23), seg=512)
    lab13_roof, shifts = qa.offset_building_labels(lab13, b13, np.isin(lab23, b23), fdy, fdx)
    roof = overlaps(lab13_roof, lab23, b13, b23); roof["frame"] = "roof"
    ground = overlaps(lab13, lab23)
    ground = ground[~(np.isin(ground.node_2013, b13) & np.isin(ground.node_2023, b23))]; ground["frame"] = "ground"
    E = pd.concat([roof, ground], ignore_index=True)
    E["shared_m2"] = (E.shared_px * PX).round(2)
    E["frac_2013"] = (E.shared_px / E.area_2013_px).round(4)
    E["frac_2023"] = (E.shared_px / E.area_2023_px).round(4)
    E["iou"] = (E.shared_px / (E.area_2013_px + E.area_2023_px - E.shared_px)).round(4)
    E["overlap_score"] = np.sqrt(E.frac_2013 * E.frac_2023).round(4)
    E["class_2013"] = n13.loc[E.node_2013, "class"].values
    E["class_2023"] = n23.loc[E.node_2023, "class"].values
    E["class_agreement"] = E.class_2013 == E.class_2023
    sh = {k: v for k, v in shifts.items()}
    E["roof_shift_dx_m"] = [round(sh[i][1] * 0.75, 2) if f == "roof" else 0.0 for i, f in zip(E.node_2013, E.frame)]
    E["roof_shift_dy_m"] = [round(-sh[i][0] * 0.75, 2) if f == "roof" else 0.0 for i, f in zip(E.node_2013, E.frame)]
    E["confidence"] = np.where((E.iou >= 0.5) & E.class_agreement, "high",
                      np.where((E.class_agreement & (E.overlap_score >= 0.3)) | (~E.class_agreement & (E.overlap_score >= 0.5)), "medium", "low"))
    E["link_type"] = np.where(E.confidence == "low", "overlap", "identity")
    a13 = n13.loc[E.node_2013, "area_m2"].values; a23 = n23.loc[E.node_2023, "area_m2"].values
    E["contained_node_m2"] = np.where(E.frac_2013 >= E.frac_2023, a13, a23).round(2)
    contained = (E.class_agreement & (np.maximum(E.frac_2013, E.frac_2023) >= CONTAIN_FRAC)
                 & (E.contained_node_m2 >= CONTAIN_MIN_M2) & (E.confidence == "low"))
    E.loc[contained, "confidence"] = "medium"
    E.loc[contained, "link_type"] = "contained_in"
    E["contained_side"] = np.where(~contained, "", np.where(E.frac_2013 >= E.frac_2023, "2013_in_2023", "2023_in_2013"))
    E["edge_role"] = np.where(E.confidence == "low", "secondary", "primary")
    PR = E[E.edge_role == "primary"]

    # components of the temporal bipartite graph -> match type
    ids13, ids23 = n13.index.values, n23.index.values
    i13 = {v: i for i, v in enumerate(ids13)}; i23 = {v: i + len(ids13) for i, v in enumerate(ids23)}
    n = len(ids13) + len(ids23)
    A = coo_matrix((np.ones(len(PR)), ([i13[v] for v in PR.node_2013], [i23[v] for v in PR.node_2023])), shape=(n, n))
    _, comp = connected_components(A, directed=False)
    c13 = Counter(comp[[i13[v] for v in ids13]]); c23 = Counter(comp[[i23[v] for v in ids23]])
    def mtype(c):
        a, b = c13.get(c, 0), c23.get(c, 0)
        if a == 1 and b == 1: return "one_to_one"
        if a == 1 and b > 1: return "one_to_many"
        if a > 1 and b == 1: return "many_to_one"
        if a > 1 and b > 1: return "many_to_many"
        return "no_primary_match_2013" if a else "no_primary_match_2023"
    E["component_id"] = [int(comp[i13[v]]) for v in E.node_2013]
    E["match_type"] = [mtype(c) if role == "primary" else "secondary_overlap" for c, role in zip(E.component_id, E.edge_role)]
    E["component_size"] = [f"{c13.get(c, 0)}:{c23.get(c, 0)}" if role == "primary" else "" for c, role in zip(E.component_id, E.edge_role)]
    E.loc[E.edge_role == "secondary", "component_id"] = -1
    E = E.sort_values(["component_id", "node_2013", "node_2023"]).reset_index(drop=True)
    E.insert(0, "temporal_edge_id", np.arange(1, len(E) + 1))
    comp_rows = []
    for c in sorted(set(comp)):
        nodes13 = [int(v) for v in ids13 if comp[i13[v]] == c] if c13.get(c) else []
        nodes23 = [int(v) for v in ids23 if comp[i23[v]] == c] if c23.get(c) else []
        comp_rows.append({"component_id": int(c), "match_type": mtype(c), "n_2013": len(nodes13), "n_2023": len(nodes23),
                          "nodes_2013": "|".join(map(str, nodes13)), "nodes_2023": "|".join(map(str, nodes23)),
                          "classes_2013": "|".join(sorted(set(n13.loc[nodes13, "class"]))) if nodes13 else "",
                          "classes_2023": "|".join(sorted(set(n23.loc[nodes23, "class"]))) if nodes23 else ""})
    C = pd.DataFrame(comp_rows)
    roof_info = {"regional_profile_x_dy_dx_px2x": [[float(a), float(b), float(c)] for a, b, c in prof]}
    return E.drop(columns=["area_2013_px", "area_2023_px"]), C, roof_info


def validate(E, C, n13, n23):
    checks = {
        "edges_only_between_years": bool(E.node_2013.isin(n13.index).all() and E.node_2023.isin(n23.index).all()),
        "no_duplicate_edges": not E.duplicated(["node_2013", "node_2023"]).any(),
        "link_rule_satisfied": bool(((E.shared_m2 / PX) >= MIN_PX - 1e-9).all()),
        "fractions_in_range": bool(E[["frac_2013", "frac_2023", "iou", "overlap_score"]].apply(lambda s: s.between(0, 1)).all().all()),
        "every_node_in_exactly_one_component": int(C.n_2013.sum()) == len(n13) and int(C.n_2023.sum()) == len(n23),
        "primary_components_bounded": int((C.n_2013 + C.n_2023).max()) <= 50,
        "roof_frame_only_building_pairs": bool((E[E.frame == "roof"][["class_2013", "class_2023"]] == "building").all().all()),
    }
    return checks


def render(out, img13, img23, E, n13, n23, tr):
    S = 2048; k = S / 1024
    base = cv2.resize(img23, (S, S), interpolation=cv2.INTER_CUBIC)
    base = (0.45 * base + 0.55 * 40).astype(np.uint8)
    im = Image.fromarray(base); d = ImageDraw.Draw(im)
    px = lambda x, y: (((x - tr.c) / tr.a) * k, ((y - tr.f) / tr.e) * k)
    col = {"high": (60, 220, 90), "medium": (250, 200, 40), "low": (240, 60, 60)}
    for conf in ("low", "medium", "high"):
        for r in E[E.confidence == conf].itertuples():
            a = px(n13.node_x[r.node_2013], n13.node_y[r.node_2013]); b = px(n23.node_x[r.node_2023], n23.node_y[r.node_2023])
            d.line([a, b], fill=(150, 60, 60) if conf == "low" else col[conf], width={"low": 1, "medium": 3, "high": 4}[conf])
    for (n, c) in ((n13, (0, 200, 255)), (n23, (255, 255, 255))):
        for r in n.itertuples():
            x, y = px(r.node_x, r.node_y); d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=c, outline=(0, 0, 0))
    f = ImageFont.truetype(fonts.BOLD, 26)
    d.rectangle((0, 0, S, 46), fill=(20, 20, 24))
    d.text((12, 9), f"{len(E)} temporal edges   green high  yellow medium (primary)  thin red low (secondary)   ·   cyan = 2013 node, white = 2023 node (on 2023 image)", font=f, fill="white")
    im.save(out / "temporal_overview.png", optimize=True)
    # zoom panel: 2013 image | 2023 image with temporal lines
    rows = []
    for (r0, c0) in ((80, 560), (560, 60), (600, 640)):
        tiles = []
        for img, label in ((img13, "2013"), (img23, "2023")):
            t = cv2.resize(img[r0 // 2:r0 // 2 + 160, c0 // 2:c0 // 2 + 160], (640, 640), interpolation=cv2.INTER_CUBIC)
            p = Image.fromarray(t); dd = ImageDraw.Draw(p); kk = 640 / 320
            pz = lambda x, y: ((((x - tr.c) / tr.a) - c0) * kk, (((y - tr.f) / tr.e) - r0) * kk)
            if label == "2023":
                for r in E.itertuples():
                    a = pz(n13.node_x[r.node_2013], n13.node_y[r.node_2013]); b = pz(n23.node_x[r.node_2023], n23.node_y[r.node_2023])
                    if -50 < a[0] < 690 and -50 < a[1] < 690:
                        if r.confidence == "low":
                            dd.line([a, b], fill=(150, 70, 70), width=1)
                        else:
                            dd.line([a, b], fill=col[r.confidence], width=4)
                for n, c in ((n13, (0, 200, 255)), (n23, (255, 255, 255))):
                    for q in n.itertuples():
                        x, y = pz(q.node_x, q.node_y)
                        if 0 <= x < 640 and 0 <= y < 640:
                            dd.ellipse((x - 5, y - 5, x + 5, y + 5), fill=c, outline=(0, 0, 0))
            dd.text((8, 6), label, font=f, fill="yellow", stroke_width=2, stroke_fill="black")
            tiles.append(np.array(p))
        rows.append(np.concatenate([tiles[0], np.full((640, 8, 3), 25, np.uint8), tiles[1]], 1))
    Image.fromarray(np.concatenate(sum([[r, np.full((8, r.shape[1], 3), 25, np.uint8)] for r in rows], [])[:-1], 0)).save(out / "temporal_zooms.png", optimize=True)


def render_match_maps(out, img13, img23, E, C, tile):
    """Segments of each year coloured by the match type of their primary component."""
    colors = {"one_to_one": (60, 200, 90), "one_to_many": (60, 140, 255), "many_to_one": (170, 90, 255),
              "many_to_many": (255, 150, 40), "no_primary_match_2013": (90, 90, 90), "no_primary_match_2023": (90, 90, 90)}
    panels = []
    for year, img in (("2013", img13), ("2023", img23)):
        with rasterio.open(FROZEN / f"FROZEN_{year}_{tile}_{REL_VERSION[year]}" / "segment_ids.tif") as s:
            seg = s.read(1)
        lut = np.zeros((seg.max() + 1, 3), np.float32)
        col = "nodes_2013" if year == "2013" else "nodes_2023"
        for r in C.itertuples():
            for i in str(getattr(r, col)).split("|"):
                if i not in ("", "nan"):
                    lut[int(i)] = colors[r.match_type]
        base = cv2.resize(img, seg.shape[::-1], interpolation=cv2.INTER_CUBIC).astype(np.float32)
        out_img = (0.45 * base + 0.55 * lut[seg]).astype(np.uint8)
        e = np.zeros(seg.shape, bool); e[:, 1:] |= seg[:, 1:] != seg[:, :-1]; e[1:] |= seg[1:] != seg[:-1]
        out_img[e] = (20, 20, 20)
        p = Image.fromarray(out_img); d = ImageDraw.Draw(p)
        d.text((10, 8), year, font=ImageFont.truetype(fonts.BOLD, 30), fill="white", stroke_width=2, stroke_fill="black")
        panels.append(np.array(p))
    canvas = np.concatenate([panels[0], np.full((panels[0].shape[0], 12, 3), 25, np.uint8), panels[1]], 1)
    leg = Image.new("RGB", (canvas.shape[1], 46), (25, 25, 25)); d = ImageDraw.Draw(leg); x = 12
    f = ImageFont.truetype(fonts.REGULAR, 19)
    for name, c in (("one-to-one", colors["one_to_one"]), ("one-to-many (1 x 2013 : n x 2023)", colors["one_to_many"]),
                    ("many-to-one", colors["many_to_one"]), ("many-to-many", colors["many_to_many"]), ("no primary match", (90, 90, 90))):
        d.rectangle((x, 13, x + 20, 33), fill=c); d.text((x + 28, 23), name, font=f, fill="white", anchor="lm"); x += 60 + int(d.textlength(name, font=f))
    Image.fromarray(np.concatenate([np.array(leg), canvas], 0)).save(out / "temporal_match_types.png", optimize=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tile", default="tile_00000")
    ap.add_argument("--self-test", action="store_true", help="link the 2023 graph to itself (must be all one_to_one/high)")
    ap.add_argument("--release-2013", default="v1"); ap.add_argument("--release-2023", default="v1")
    args = ap.parse_args()
    REL_VERSION.update({"2013": args.release_2013, "2023": args.release_2023})
    r13, g13, lab13, n13, tr, crs, ip13 = load("2013", args.tile)
    r23, g23, lab23, n23, tr23, _, ip23 = load("2023", args.tile)
    assert tr == tr23 and lab13.shape == lab23.shape
    if args.self_test:
        E, C, _ = build_layer(lab23, n23, lab23, n23)
        print(json.dumps({"self_test_edges": len(E), "nodes": len(n23), "match_types": E.match_type.value_counts().to_dict(),
                          "confidence": E.confidence.value_counts().to_dict(), "min_iou": float(E.iou.min())}))
        return
    E, C, roof = build_layer(lab13, n13, lab23, n23)
    out = GRAPHS / f"temporal_2013_2023_{args.tile}_{VERSION}"; out.mkdir(parents=True, exist_ok=True)
    E.insert(1, "key_2013", [f"2013:{args.tile}:{i}" for i in E.node_2013])
    E.insert(2, "key_2023", [f"2023:{args.tile}:{i}" for i in E.node_2023])
    E.to_csv(out / "temporal_edges.csv", index=False)
    C.to_csv(out / "temporal_components.csv", index=False)
    un = pd.concat([pd.DataFrame({"year": "2013", "segment_id": C[C.match_type == "no_primary_match_2013"].nodes_2013.astype(int)}),
                    pd.DataFrame({"year": "2023", "segment_id": C[C.match_type == "no_primary_match_2023"].nodes_2023.astype(int)})])
    un["class"] = [(n13 if y == "2013" else n23).loc[i, "class"] for y, i in zip(un.year, un.segment_id)]
    un["area_m2"] = [(n13 if y == "2013" else n23).loc[i, "area_m2"] for y, i in zip(un.year, un.segment_id)]
    sec = E[E.edge_role == "secondary"]
    un["secondary_edges"] = [int(((sec.node_2013 if y == "2013" else sec.node_2023) == i).sum()) for y, i in zip(un.year, un.segment_id)]
    un.to_csv(out / "no_primary_match_nodes.csv", index=False)

    # bipartite graph: nodes of both years, ONLY temporal edges
    G = nx.Graph(layer="temporal", tile=args.tile, years="2013,2023",
                 spatial_graphs=f"{g13.name},{g23.name}")
    for y, n in (("2013", n13), ("2023", n23)):
        for i, r in n.iterrows():
            G.add_node(f"{y}:{args.tile}:{i}", year=y, segment_id=int(i), cls=r["class"], x=float(r.node_x), y=float(r.node_y), bipartite=int(y == "2023"))
    for r in E.to_dict("records"):
        G.add_edge(r["key_2013"], r["key_2023"], edge_layer="temporal", **{k: (bool(v) if isinstance(v, (bool, np.bool_)) else (float(v) if isinstance(v, (float, np.floating)) else (int(v) if isinstance(v, (np.integer, int)) else str(v))))
                                                                         for k, v in r.items() if k not in ("key_2013", "key_2023")})
    nx.write_graphml(G, out / "temporal_layer.graphml")

    gpkg = out / "temporal_layer.gpkg"
    if gpkg.exists(): gpkg.unlink()
    ds = ogr.GetDriverByName("GPKG").CreateDataSource(str(gpkg))
    srs = osr.SpatialReference(); srs.ImportFromWkt(crs.to_wkt())
    lyr = ds.CreateLayer("temporal_edges", srs, ogr.wkbLineString)
    cols = ["temporal_edge_id", "node_2013", "node_2023", "frame", "shared_m2", "iou", "overlap_score", "class_2013", "class_2023", "match_type", "confidence", "edge_role", "link_type"]
    for c in cols:
        lyr.CreateField(ogr.FieldDefn(c, ogr.OFTInteger if c in ("temporal_edge_id", "node_2013", "node_2023") else (ogr.OFTReal if c in ("shared_m2", "iou", "overlap_score") else ogr.OFTString)))
    lyr.StartTransaction()
    for r in E.to_dict("records"):
        f = ogr.Feature(lyr.GetLayerDefn())
        for c in cols:
            f.SetField(c, int(r[c]) if isinstance(r[c], (np.integer,)) else r[c])
        g = LineString([(n13.node_x[r["node_2013"]], n13.node_y[r["node_2013"]]), (n23.node_x[r["node_2023"]], n23.node_y[r["node_2023"]])])
        f.SetGeometry(ogr.CreateGeometryFromWkb(g.wkb)); lyr.CreateFeature(f)
    lyr.CommitTransaction(); ds = None

    img13 = rasterio.open(ip13).read([1, 2, 3]).transpose(1, 2, 0); img23 = rasterio.open(ip23).read([1, 2, 3]).transpose(1, 2, 0)
    render(out, img13, img23, E, n13, n23, tr)
    render_match_maps(out, img13, img23, E, C, args.tile)

    checks = validate(E, C, n13, n23)
    checks["graphml_roundtrip"] = nx.read_graphml(out / "temporal_layer.graphml").number_of_edges() == len(E)
    checks["no_spatial_edges_in_layer"] = all(d.get("edge_layer") == "temporal" for *_, d in G.edges(data=True))
    comp_types = C.match_type.value_counts().to_dict()
    summary = {
        "inputs": {"2013": [r13.name, g13.name], "2023": [r23.name, g23.name]},
        "nodes": {"2013": len(n13), "2023": len(n23)},
        "temporal_edges": len(E),
        "temporal_edges_by_frame": E.frame.value_counts().to_dict(),
        "temporal_edges_by_confidence": E.confidence.value_counts().to_dict(),
        "version": VERSION, "contain_frac": CONTAIN_FRAC, "contain_min_m2": CONTAIN_MIN_M2,
        "temporal_edges_by_role": E.edge_role.value_counts().to_dict(),
        "temporal_edges_by_link_type": E.link_type.value_counts().to_dict(),
        "contained_in_by_class": E[E.link_type == "contained_in"].class_2023.value_counts().to_dict(),
        "temporal_edges_by_match_type": E.match_type.value_counts().to_dict(),
        "primary_edges_cross_class": int(((E.edge_role == "primary") & ~E.class_agreement).sum()),
        "components_by_match_type": comp_types,
        "nodes_by_match_type": {"2013": {t: int(C[C.match_type == t].n_2013.sum()) for t in comp_types},
                                "2023": {t: int(C[C.match_type == t].n_2023.sum()) for t in comp_types}},
        "largest_component_2013_2023": [int(C.n_2013.max()), int(C.n_2023.max())],
        "class_agreement_share": round(float(E.class_agreement.mean()), 3),
        "confidence_by_class_pair_same": E[E.class_agreement].groupby("class_2023").confidence.value_counts().unstack(fill_value=0).to_dict("index"),
        "roof_offset": roof,
        "checks": checks, "status": "PASS" if all(checks.values()) else "FAIL",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=int), encoding="utf-8")
    print(json.dumps(summary, indent=1, default=int))


if __name__ == "__main__":
    main()
