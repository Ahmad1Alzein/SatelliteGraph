"""Change detection on the frozen spatio-temporal graph (2013 -> 2023).

Correspondence comes ONLY from the frozen graph's temporal edges (no re-matching). Image evidence
(change_detection.detect: present in one year, absent at every scale within +-1.5 m in the other,
shadow = no evidence, roof lean compensated) decides whether a change is real. Topology without
evidence is never reported as change.

Node / edge change labels
  2023 building : new_building   evidence new  AND no primary temporal edge from a 2013 building
                  extension      evidence new area AND >= 1 primary 2013-building predecessor
                  (evidence and topology disagree -> confidence 'conflict')
                  persistent / segmentation_difference / unexplained_no_evidence otherwise
  2013 building : demolished (unverified) / persistent / unexplained_no_evidence
  change edge   : conversion_land_to_building (DERIVED relation, not part of the frozen temporal layer):
                  2013 land node -> confirmed 2023 building wherever evidence pixels inside both nodes
                  >= 25 m2; via = 'temporal_edge' if the frozen layer links the pair, else 'evidence_only'
                  (partial conversions often lack a temporal edge: link rule needs >= 25 % of the smaller node)
  temporal edge : persistence, conversion (if it carries a conversion), class_transition_candidate
                  (primary cross-class, non-building; unverified), segmentation_link, context (secondary)
  2013 land     : built_over if converted area >= 100 m2 or >= 30 % of the node (high)
  2023 land     : tree_loss / tree_gain / new_road evidence candidates (unverified)
  persisting 2023 building: neighbourhood Jaccard across years (via temporal edges) and number of
                  confirmed new buildings among its spatial neighbours (densification)
Output: graphs/change_stgraph_2013_2023_<tile>/ (derived; the frozen graph is not modified).
"""
from __future__ import annotations

from osgeo import gdal  # noqa: F401
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import fonts  # package: portable font paths

import cv2
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw, ImageFont

import change_detection as cd
import final_strip_qa as qa

HERE = Path(__file__).resolve().parent
GRAPHS = HERE / "graphs"
FROZEN = HERE / "final_segmentation"
PX = 0.5625
PRIMARY = ("identity", "contained_in")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- core (in-memory)
def detect_on_graph(img13, lab13, n13, img23, lab23, n23, T, spatial=None):
    """n13/n23: index segment_id, columns class, area_m2, confidence.
    T: temporal edges with node_2013, node_2023, link_type, edge_role, class_2013, class_2023, shared_m2.
    spatial: optional dict year -> DataFrame(source, target) of spatial edges (for neighbourhood change)."""
    b13 = n13.index[n13["class"] == "building"].values
    b23 = n23.index[n23["class"] == "building"].values
    fdy, fdx, _ = qa.roof_offset_field(np.isin(lab13, b13), np.isin(lab23, b23), seg=512)
    to_cd = lambda n: n.rename(columns={"class": "object_type", "confidence": "mean_conf"})
    ev13, ev23 = cd.evidence(img13), cd.evidence(img23)
    change_map, eb23, eb13 = cd.detect(ev13, ev23, lab13, to_cd(n13), lab23, to_cd(n23), fdy, fdx, img13, img23)
    eb23 = eb23.set_index("node_id"); eb13 = eb13.set_index("node_id")
    P = T[T.link_type.isin(PRIMARY)]
    pred_b = defaultdict(list); succ_b = defaultdict(list)
    for r in P[(P.class_2013 == "building") & (P.class_2023 == "building")].itertuples():
        pred_b[r.node_2023].append(r.node_2013); succ_b[r.node_2013].append(r.node_2023)
    match23 = defaultdict(set); match13 = defaultdict(set)
    for r in P.itertuples():
        match23[r.node_2023].add(r.node_2013); match13[r.node_2013].add(r.node_2023)

    # ---- 2023 nodes
    rows23 = []
    for i, r in n23.iterrows():
        row = {"segment_id": int(i), "class": r["class"], "area_m2": float(r.area_m2), "change": "persistent", "confidence": "",
               "evidence": "", "evidence_new_area_m2": 0.0, "n_primary_building_predecessors": len(pred_b.get(i, []))}
        if r["class"] == "building":
            ev, evc, new = eb23.loc[i, "change"], eb23.loc[i, "confidence"], float(eb23.loc[i, "new_area_m2"])
            row.update(evidence=ev, evidence_new_area_m2=new)
            has_pred = len(pred_b.get(i, [])) > 0
            if ev == "new_building":
                row.update(change="new_building", confidence=evc if not has_pred else "conflict")
            elif ev == "extension":
                row.update(change="extension", confidence=evc if has_pred else "conflict")
            elif not match23.get(i):
                row.update(change="unexplained_no_evidence")
            elif has_pred and (len(pred_b[i]) > 1 or any(len(succ_b[p]) > 1 for p in pred_b[i])):
                row.update(change="segmentation_difference")
        else:
            codes, cnt = np.unique(change_map[lab23 == i], return_counts=True)
            cand = {k: n * PX for k, n in zip(codes, cnt) if k in (cd.CODES["tree_loss"], cd.CODES["tree_gain"], cd.CODES["new_road"])}
            if cand:
                k, a = max(cand.items(), key=lambda t: t[1])
                if a >= 100 or a >= 0.3 * r.area_m2:
                    row.update(change={v: k2 for k2, v in cd.CODES.items()}[k], confidence="unverified", evidence_new_area_m2=round(a, 1))
        rows23.append(row)
    N23 = pd.DataFrame(rows23).set_index("segment_id")

    # ---- temporal edge labels + land conversion
    confirmed = set(N23.index[N23.change.isin(["new_building", "extension"]) & (N23.confidence == "high")])
    gain_hi = np.isin(lab23, list(confirmed))
    if confirmed:
        # evidence pixels only: re-derive the 'gain' part of confirmed nodes from the change map
        gain_hi &= np.isin(change_map, [cd.CODES["new_building"], cd.CODES["extension"]])
    # derived conversion edges from evidence pixels (land 2013 x confirmed building 2023)
    land13 = set(n13.index[n13["class"] != "building"])
    k = lab13[gain_hi] * 10_000_000 + lab23[gain_hi]
    u, c = np.unique(k, return_counts=True)
    tpairs = set(zip(T.node_2013, T.node_2023))
    CE = pd.DataFrame({"node_2013": u // 10_000_000, "node_2023": u % 10_000_000, "converted_m2": (c * PX).round(1)})
    CE = CE[CE.node_2013.isin(land13) & (CE.converted_m2 >= 25)].copy()
    CE["class_2013"] = n13.loc[CE.node_2013, "class"].values
    CE["via"] = ["temporal_edge" if (a, b) in tpairs else "evidence_only" for a, b in zip(CE.node_2013, CE.node_2023)]
    conv_pairs = dict(zip(zip(CE.node_2013, CE.node_2023), CE.converted_m2))
    Tl = T.copy()
    conv = []
    lab_e = []
    for r in Tl.itertuples():
        a = conv_pairs.get((r.node_2013, r.node_2023), 0.0)
        conv.append(a)
        if a > 0:
            lab_e.append("conversion_land_to_building"); continue
        if r.link_type in PRIMARY:
            if r.class_2013 == r.class_2023:
                lab_e.append("persistence" if not (r.class_2023 == "building" and N23.loc[r.node_2023, "change"] == "segmentation_difference") else "segmentation_link")
            elif "building" in (r.class_2013, r.class_2023):
                lab_e.append("building_label_link")
            else:
                lab_e.append("class_transition_candidate")
        else:
            lab_e.append("context")
    Tl["converted_m2"] = np.round(conv, 1); Tl["change_label"] = lab_e
    Tl["change_confidence"] = Tl.change_label.map({"conversion_land_to_building": "high", "class_transition_candidate": "unverified"}).fillna("")

    # ---- 2013 nodes
    rows13 = []
    for i, r in n13.iterrows():
        row = {"segment_id": int(i), "class": r["class"], "area_m2": float(r.area_m2), "change": "persistent", "confidence": "", "converted_m2": 0.0}
        if r["class"] == "building":
            if i in eb13.index and eb13.loc[i, "change"] == "demolished":
                row.update(change="demolished", confidence="unverified")
            elif not match13.get(i):
                row.update(change="unexplained_no_evidence")
        else:
            a = float(CE[CE.node_2013 == i].converted_m2.sum())
            row["converted_m2"] = round(a, 1)
            if a >= 100 or a >= 0.3 * r.area_m2:
                row.update(change="built_over", confidence="high")
        rows13.append(row)
    N13 = pd.DataFrame(rows13).set_index("segment_id")

    # ---- neighbourhood change of persisting buildings (needs spatial edges)
    if spatial is not None:
        nb = {}
        for y, n in (("2013", n13), ("2023", n23)):
            bset = set(n.index[n["class"] == "building"])
            d = defaultdict(set)
            for s_, t_ in zip(spatial[y].source, spatial[y].target):
                if s_ in bset and t_ in bset:
                    d[s_].add(t_); d[t_].add(s_)
            nb[y] = d
        jac, dens = {}, {}
        for b in b23:
            if N23.loc[b, "change"] not in ("persistent", "segmentation_difference") or not pred_b.get(b):
                continue
            old = set()
            for p in pred_b[b]:
                for q in nb["2013"].get(p, ()):
                    old |= set(succ_b.get(q, ()))
            old.discard(b)
            new = nb["2023"].get(b, set())
            jac[b] = len(old & new) / max(len(old | new), 1)
            dens[b] = len(new & confirmed)
        N23["neighbour_jaccard"] = pd.Series(jac).round(3)
        N23["confirmed_new_neighbours"] = pd.Series(dens)
    return {"N23": N23, "N13": N13, "T": Tl, "CE": CE, "change_map": change_map}


# ----------------------------------------------------------------------------- real run
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tile", default="tile_00000")
    args = ap.parse_args()
    tile = args.tile
    S = GRAPHS / f"FROZEN_stgraph_2013_2023_{tile}_v1"
    man = json.loads((S / "MANIFEST.json").read_text())
    for f, h in man["files"].items():
        assert sha(S / f) == h, f"frozen st-graph changed: {f}"
    Tz = GRAPHS / f"FROZEN_temporal_2013_2023_{tile}_v3"
    assert sha(Tz / "MANIFEST.json") == man["inputs"][f"{Tz.name}/MANIFEST.json"]
    nodes = pd.read_csv(S / "nodes.csv"); edges = pd.read_csv(S / "edges.csv")
    VER = {}
    for key in man["inputs"]:
        folder = key.split("/", 1)[0]
        for y in ("2013", "2023"):
            if folder.startswith(f"FROZEN_{y}_{tile}_"):
                VER[y] = folder.rsplit("_", 1)[1]

    lab, img, nyear = {}, {}, {}
    for y in ("2013", "2023"):
        rel = FROZEN / f"FROZEN_{y}_{tile}_{VER[y]}"
        m = json.loads((rel / "MANIFEST.json").read_text())
        assert sha(rel / "segment_ids.tif") == m["files"]["segment_ids.tif"]
        assert sha(m["input_image"]["path"]) == m["input_image"]["sha256"]
        lab[y] = rasterio.open(rel / "segment_ids.tif").read(1).astype(np.int64)
        img[y] = rasterio.open(m["input_image"]["path"]).read([1, 2, 3]).transpose(1, 2, 0).astype(np.uint8)
        nyear[y] = nodes[nodes.year == int(y)].set_index("segment_id")[["class", "area_m2", "confidence", "gid", "key", "node_x", "node_y", "temporal_match_type"]]
    seg_of = dict(zip(nodes.gid, nodes.segment_id))
    te = edges[edges.layer == "temporal"].copy()
    te["node_2013"] = te.src.map(seg_of); te["node_2023"] = te.dst.map(seg_of)
    te["link_type"] = te.relation.str.replace("temporal_", "", regex=False)
    te["class_2013"] = nyear["2013"].loc[te.node_2013, "class"].values; te["class_2023"] = nyear["2023"].loc[te.node_2023, "class"].values
    spatial = {y: edges[(edges.layer == "spatial") & (edges.year.astype(str) == y)].assign(source=lambda d: d.src.map(seg_of), target=lambda d: d.dst.map(seg_of)) for y in ("2013", "2023")}

    res = detect_on_graph(img["2013"], lab["2013"], nyear["2013"], img["2023"], lab["2023"], nyear["2023"], te, spatial)
    out = GRAPHS / f"change_stgraph_2013_2023_{tile}"; out.mkdir(parents=True, exist_ok=True)
    N23 = res["N23"].join(nyear["2023"][["gid", "key", "temporal_match_type"]]); N13 = res["N13"].join(nyear["2013"][["gid", "key", "temporal_match_type"]])
    nodes_change = pd.concat([N13.reset_index().assign(year=2013), N23.reset_index().assign(year=2023)], ignore_index=True)
    cols = ["gid", "key", "year", "segment_id", "class", "area_m2", "temporal_match_type", "change", "confidence"]
    nodes_change = nodes_change[cols + [c for c in nodes_change.columns if c not in cols]].sort_values("gid")
    nodes_change.to_csv(out / "nodes_change.csv", index=False)
    T = res["T"][["eid", "relation", "src", "dst", "node_2013", "node_2023", "class_2013", "class_2023", "confidence", "shared_m2", "converted_m2", "change_label", "change_confidence"]]
    T.to_csv(out / "temporal_edges_change.csv", index=False)
    CE = res["CE"].copy()
    CE["key_2013"] = nyear["2013"].loc[CE.node_2013, "key"].values; CE["key_2023"] = nyear["2023"].loc[CE.node_2023, "key"].values
    CE["change_confidence"] = "high"
    CE.to_csv(out / "change_edges_conversion.csv", index=False)
    with rasterio.open(FROZEN / f"FROZEN_2023_{tile}_{VER['2023']}" / "segment_ids.tif") as s:
        prof = s.profile; tr = s.transform
    prof.update(dtype="uint8", nodata=0)
    with rasterio.open(out / "evidence_change_map.tif", "w", **prof) as d:
        d.write(res["change_map"].astype(np.uint8), 1)

    # change-annotated spatio-temporal graph (graphml): frozen structure + change attributes
    G = nx.Graph(tile=tile, source_release=S.name)
    lab_node = nodes_change.set_index("gid")
    for r in nodes.to_dict("records"):
        a = lab_node.loc[r["gid"]]
        G.add_node(r["key"], year=str(r["year"]), segment_id=int(r["segment_id"]), cls=str(r["class"]),
                   area_m2=float(r["area_m2"]), change=str(a.change), change_confidence=str(a.confidence))
    tlab = T.set_index("eid")
    for r in edges.itertuples():
        attrs = {"relation": r.relation, "layer": r.layer}
        if r.layer == "temporal":
            attrs.update(change_label=str(tlab.loc[r.eid, "change_label"]), change_confidence=str(tlab.loc[r.eid, "change_confidence"]),
                         converted_m2=float(tlab.loc[r.eid, "converted_m2"]))
        G.add_edge(nodes.key[r.src], nodes.key[r.dst], **attrs)
    for r in CE.itertuples():
        if not G.has_edge(r.key_2013, r.key_2023):   # temporal-edge conversions are already labelled on that edge
            G.add_edge(r.key_2013, r.key_2023, relation="change_conversion", layer="change", via=r.via,
                       converted_m2=float(r.converted_m2), change_label="conversion_land_to_building", change_confidence="high")
    nx.write_graphml(G, out / "change_stgraph.graphml")

    b23 = N23[N23["class"] == "building"]
    summary = {
        "source_release": S.name,
        "buildings_2023": {f"{a}|{b}": int(v) for (a, b), v in b23.groupby(["change", "confidence"]).size().items()},
        "buildings_2013": {f"{a}|{b}": int(v) for (a, b), v in N13[N13["class"] == "building"].groupby(["change", "confidence"]).size().items()},
        "land_2013_built_over": int((N13.change == "built_over").sum()),
        "land_built_over_m2_by_class": N13[N13.change == "built_over"].groupby("class").converted_m2.sum().round(0).to_dict(),
        "land_2023_evidence_candidates": N23[N23["class"] != "building"].change.value_counts().drop("persistent", errors="ignore").to_dict(),
        "temporal_edges_by_change_label": T.change_label.value_counts().to_dict(),
        "conversion_edges": {"total": len(CE), **CE.via.value_counts().to_dict()},
        "converted_m2_by_class_2013": CE.groupby("class_2013").converted_m2.sum().round(0).to_dict(),
        "confirmed_new_building_area_m2": round(float(b23[(b23.change == "new_building") & (b23.confidence == "high")].area_m2.sum()), 1),
        "persisting_buildings": int(b23.neighbour_jaccard.notna().sum()),
        "neighbour_jaccard_median": round(float(b23.neighbour_jaccard.median()), 3),
        "persisting_buildings_with_confirmed_new_neighbour": int((b23.confirmed_new_neighbours.fillna(0) > 0).sum()),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=int), encoding="utf-8")
    render(out, img, lab, N13, N23, res["T"], nyear, tr, res["CE"], res["change_map"])
    print(json.dumps(summary, indent=1, default=int))


def render(out, img, lab, N13, N23, T, nyear, tr, CE, change_map):
    S = 1024
    hi23 = N23[N23.confidence == "high"]
    bo = N13[N13.change == "built_over"].index
    panels = []
    for y in ("2013", "2023"):
        base = cv2.resize(img[y], (S, S), interpolation=cv2.INTER_CUBIC).copy()
        if y == "2013":
            m = np.zeros(lab["2013"].shape, bool)
            for r in CE.itertuples():
                m |= (lab["2013"] == r.node_2013) & (lab["2023"] == r.node_2023) & np.isin(change_map, [1, 2])
            base[m] = (0.45 * base[m] + 0.55 * np.array([0, 200, 255])).astype(np.uint8)
        for kind, col in (("new_building", (255, 40, 40)), ("extension", (255, 165, 0))):
            mm = np.isin(lab["2023"], hi23.index[hi23.change == kind]).astype(np.uint8)
            e = cv2.dilate(mm ^ cv2.erode(mm, np.ones((3, 3), np.uint8)), np.ones((2, 2), np.uint8)).astype(bool)
            base[e] = col
        p = Image.fromarray(base); d = ImageDraw.Draw(p)
        if y == "2023":   # draw conversion temporal edges as arrows from 2013 land node position
            for r in CE.itertuples():
                a = nyear["2013"].loc[r.node_2013]; b = nyear["2023"].loc[r.node_2023]
                pa = ((a.node_x - tr.c) / tr.a, (a.node_y - tr.f) / tr.e); pb = ((b.node_x - tr.c) / tr.a, (b.node_y - tr.f) / tr.e)
                d.line([pa, pb], fill=(0, 220, 255), width=2); d.ellipse((pa[0] - 4, pa[1] - 4, pa[0] + 4, pa[1] + 4), fill=(0, 220, 255))
        d.text((10, 8), y, font=ImageFont.truetype(fonts.BOLD, 26), fill="white", stroke_width=2, stroke_fill="black")
        panels.append(np.array(p))
    canvas = np.concatenate([panels[0], np.full((S, 10, 3), 25, np.uint8), panels[1]], 1)
    head = Image.new("RGB", (canvas.shape[1], 40), (25, 25, 25))
    ImageDraw.Draw(head).text((12, 9), "high confidence: red = new building, orange = extension · cyan (2013) = land converted to building; cyan lines (2023) = conversion change edges",
                              font=ImageFont.truetype(fonts.REGULAR, 17), fill="white")
    Image.fromarray(np.concatenate([np.array(head), canvas], 0)).save(out / "change_overview.png", optimize=True)


if __name__ == "__main__":
    main()
