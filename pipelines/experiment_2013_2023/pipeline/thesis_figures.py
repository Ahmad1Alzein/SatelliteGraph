"""Defense figures, one per pipeline step (written to segmentation_v2/thesis_figures/).

  fig0_pipeline.png            the chain with the numbers of this run
  fig1_tile_segmentation.png   one tile (tile_00002, 2023): image | segment boundaries | segments by class
  fig2_tile_graph.png          the same tile as a region graph
  fig3_strip_mosaic.png        7 tiles placed by metadata x0/y0, overlaps resolved, seam objects re-joined + strip graph
  fig3b_seam_zoom.png          one seam: tile A | tile B | strip (re-joined objects outlined)
  fig4_spatiotemporal.png      2013 and 2023 strip graphs linked by temporal edges (window over tile_00002)
  fig5a_gnn_architecture.png   relational GNN on the spatio-temporal graph
  fig5b_gnn_results.png        held-out (leave-one-tile-out) AP / F1 / ROC-AUC: MLP vs spatial GNN vs spatio-temporal GNN
  fig5c_gnn_change_map.png     held-out GNN predictions on the strip vs rule labels (TP / FP / FN)
"""
from __future__ import annotations

from osgeo import gdal  # noqa: F401
import json
import math
from pathlib import Path

import fonts  # package: portable font paths

import paths  # package: input/checkpoint locations

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw, ImageFont

import segment_final as sf

HERE = Path(__file__).resolve().parent
OUT = HERE / "thesis_figures"
STRIP = HERE / "graphs" / "strip_tile_00000-00006"
META = paths.META_2023   # package: input location from paths.py
TILES = [f"tile_{i:05d}" for i in range(7)]
SURF, INK, INK2, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d4"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"     # validated categorical slots 1-3 (all-pairs safe)
BG = (24, 24, 28)


def font(sz, bold=True):
    return ImageFont.truetype(fonts.BOLD if bold else fonts.REGULAR, sz)


def boundaries(lab):
    e = np.zeros(lab.shape, bool); e[:, 1:] |= lab[:, 1:] != lab[:, :-1]; e[1:] |= lab[1:] != lab[:-1]; return e


def class_panel(img, lab, cls_of):
    base = np.array([sf.COLORS.get(c, (0, 0, 0)) for c in range(9)], np.float32)
    rng = np.random.default_rng(1)
    n = int(max(lab.max(), max(cls_of))) + 1
    cls = np.zeros(n, np.int64)
    for k, v in cls_of.items():
        cls[int(k)] = int(v)
    shade = rng.uniform(-35, 35, (n, 1)).astype(np.float32)
    col = np.clip(base[cls] + shade, 0, 255)[lab]
    out = (0.25 * img + 0.75 * col).astype(np.uint8); out[boundaries(lab)] = (20, 20, 20)
    return out


def titled(arr, title, sub=None, h=54):
    head = Image.new("RGB", (arr.shape[1], h), BG); d = ImageDraw.Draw(head)
    d.text((14, 8), title, font=font(22), fill="white")
    if sub:
        d.text((14, 32), sub, font=font(15, False), fill=(200, 200, 200))
    return np.concatenate([np.array(head), arr], 0)


def legend(width, counts=None):
    bar = Image.new("RGB", (width, 40), BG); d = ImageDraw.Draw(bar); x = 14
    for cid, name in sf.CLASSES.items():
        d.rectangle((x, 11, x + 18, 29), fill=sf.COLORS[cid])
        t = name.replace("_", " ") + (f" ({counts.get(name, 0)})" if counts is not None else "")
        d.text((x + 24, 20), t, font=font(15, False), fill=(235, 235, 235), anchor="lm"); x += 36 + int(d.textlength(t, font=font(15, False)))
    return np.array(bar)


def hstack(arrs, gap=10):
    h = max(a.shape[0] for a in arrs)
    arrs = [np.pad(a, ((0, h - a.shape[0]), (0, 0), (0, 0)), constant_values=24) for a in arrs]
    out = [arrs[0]]
    for a in arrs[1:]:
        out += [np.full((h, gap, 3), 24, np.uint8), a]
    return np.concatenate(out, 1)


def vstack(arrs, gap=10):
    w = max(a.shape[1] for a in arrs)
    arrs = [np.pad(a, ((0, 0), (0, w - a.shape[1]), (0, 0)), constant_values=24) for a in arrs]
    out = [arrs[0]]
    for a in arrs[1:]:
        out += [np.full((gap, w, 3), 24, np.uint8), a]
    return np.concatenate(out, 0)


def draw_graph(img, nodes, edges, tr, scale=1.0, temporal=None, node_r=None):
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    P = {r.segment_id: (((r.node_x - tr.c) / tr.a) * scale, ((r.node_y - tr.f) / tr.e) * scale) for r in nodes.itertuples()}
    for r in edges.itertuples():
        d.line([P[r.source], P[r.target]], fill=(240, 240, 240) if r.edge_type == "touches" else (255, 70, 220), width=1)
    for r in nodes.itertuples():
        x, y = P[r.segment_id]; rad = node_r or (2 + 1.5 * math.log10(max(r.area_m2, 10)))
        d.ellipse((x - rad, y - rad, x + rad, y + rad), fill=sf.COLORS[r.class_id], outline=(0, 0, 0))
    return np.array(im)


# ----------------------------------------------------------------------------- figures
def fig1_fig2():
    rel = HERE / "final_segmentation" / "FROZEN_2023_tile_00002_v2"
    man = json.loads((rel / "MANIFEST.json").read_text())
    img = cv2.resize(rasterio.open(man["input_image"]["path"]).read([1, 2, 3]).transpose(1, 2, 0), (1024, 1024), interpolation=cv2.INTER_CUBIC)
    with rasterio.open(rel / "segment_ids.tif") as s:
        lab, tr = s.read(1), s.transform
    nf = pd.read_csv(rel / "nodes_final.csv")
    cls = dict(zip(nf.segment_id, nf.class_id))
    b = img.copy(); b[boundaries(lab)] = (255, 255, 0)
    c = class_panel(img, lab, cls)
    S = 700
    panels = [cv2.resize(p, (S, S), interpolation=cv2.INTER_AREA) for p in (img, b, c)]
    row = hstack([titled(panels[0], "Input image"), titled(panels[1], "Segment boundaries"), titled(panels[2], "Segments by class")])
    fig = vstack([titled(row, f"Step 1 · Segmentation of one tile (tile_00002, 2023, 768 × 768 m)",
                         f"{len(nf)} segments · every pixel in exactly one segment · 15/15 validation checks"),
                  legend(row.shape[1], nf["class"].value_counts().to_dict())], 0)
    Image.fromarray(fig).save(OUT / "fig1_tile_segmentation.png", optimize=True)

    g = HERE / "graphs" / "graph_2023_tile_00002_v2"
    gn = pd.read_csv(g / "nodes.csv"); ge = pd.read_csv(g / "edges.csv")
    base = (0.5 * img + 0.5 * 30).astype(np.uint8); base[boundaries(lab)] = (90, 90, 90)
    gimg = draw_graph(base, gn, ge, tr)
    gimg = cv2.resize(gimg, (1000, 1000), interpolation=cv2.INTER_AREA)
    zoom = draw_graph(cv2.resize(base[300:556, 300:556], (744, 744), interpolation=cv2.INTER_CUBIC),
                      gn.assign(node_x=gn.node_x - 300 * tr.a, node_y=gn.node_y - 300 * tr.e), ge, tr, scale=744 / 256, node_r=7)
    v = json.loads((g / "graph_validation.json").read_text())
    fig = vstack([titled(hstack([titled(gimg, "Whole tile"), titled(zoom, "Zoom (192 m)")]),
                         "Step 2 · Region graph of the tile: one node per segment",
                         f"{v['nodes']} nodes · {v['edges']} edges (touches {v['edges_by_type']['touches']}, near {v['edges_by_type'].get('near', 0)}) · "
                         f"37 node features, 9 edge features · white = shares a boundary, magenta = buildings ≤ 25 m apart"),
                  legend(1754)], 0)
    Image.fromarray(fig).save(OUT / "fig2_tile_graph.png", optimize=True)


def fig3():
    meta = pd.read_csv(META).set_index("tile_id").loc[TILES]
    y = "2023"
    with rasterio.open(STRIP / y / "segment_ids.tif") as s:
        lab, tr = s.read(1), s.transform
    img = rasterio.open(STRIP / y / "image_2x.tif").read().transpose(1, 2, 0)
    nodes = pd.read_csv(STRIP / y / "nodes.csv"); edges = pd.read_csv(STRIP / y / "edges.csv")
    seams = pd.read_csv(STRIP / y / "seam_decisions.csv"); summ = json.loads((STRIP / y / "summary.json").read_text())
    k = 0.5; W = int(lab.shape[1] * k); H = int(lab.shape[0] * k)
    a = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA).copy()
    im = Image.fromarray(a); d = ImageDraw.Draw(im)
    cols = [(255, 220, 0), (0, 220, 255)]
    for i, (t, r) in enumerate(meta.iterrows()):
        x0, x1 = int(2 * r.x0 * k), int(2 * r.x1 * k) - 1
        d.rectangle((x0, 2 + 4 * (i % 2), x1, H - 3 - 4 * (i % 2)), outline=cols[i % 2], width=3)
        d.text((x0 + 8, 10 + 4 * (i % 2)), f"{t}  x0={int(r.x0)}", font=font(16), fill=cols[i % 2], stroke_width=2, stroke_fill="black")
    p1 = np.array(im)
    cls = dict(zip(nodes.segment_id, nodes.class_id))
    p2 = class_panel(img, lab, cls)
    merged = nodes[nodes.source == "seam_merged"].segment_id.values
    m = np.isin(lab, merged).astype(np.uint8)
    e = cv2.dilate((m ^ cv2.erode(m, np.ones((3, 3), np.uint8))), np.ones((4, 4), np.uint8)).astype(bool)
    p2[e] = (255, 255, 255)
    p2 = cv2.resize(p2, (W, H), interpolation=cv2.INTER_AREA)
    base = (0.5 * img + 0.5 * 30).astype(np.uint8)
    p3 = draw_graph(cv2.resize(base, (W, H), interpolation=cv2.INTER_AREA), nodes, edges[edges.edge_type == "touches"], tr, scale=k, node_r=4)
    fig = vstack([titled(p1, "Tiles placed on the shared grid from metadata x0, y0 (50 px overlaps)"),
                  titled(p2, f"Strip segmentation: overlaps cut at the middle; {summ['seam_merges']} objects cut by a seam re-joined (white outline)"),
                  titled(p3, f"Strip region graph (touches edges shown; {int((edges.edge_type == 'near').sum())} near edges omitted for readability)"), legend(W)])
    fig = titled(fig, "Step 3 · Combining the 7 tiles into one strip (2023, 4.9 × 0.8 km)",
                 f"{summ['source_segments']} tile segments → {summ['segments']} strip segments · {summ['edges']} edges · "
                 f"every pixel assigned, every segment one polygon, graph connected, georeferencing consistent with all tiles")
    Image.fromarray(fig).save(OUT / "fig3_strip_mosaic.png", optimize=True)

    # seam zoom between tile_00002 and tile_00003
    r2, r3 = meta.loc["tile_00002"], meta.loc["tile_00003"]
    seam = (2 * int(r3.x0) + 2 * int(r2.x1)) // 2
    c0, c1, rr0, rr1 = seam - 160, seam + 160, 300, 620
    tiles = {}
    for t in ("tile_00002", "tile_00003"):
        man = json.loads((HERE / "graphs" / f"FROZEN_stgraph_2013_2023_{t}_v1" / "MANIFEST.json").read_text())
        rel = next(k_.split("/")[0] for k_ in man["inputs"] if k_.startswith("FROZEN_2023"))
        tiles[t] = (rasterio.open(HERE / "final_segmentation" / rel / "segment_ids.tif").read(1),
                    dict(zip(*pd.read_csv(HERE / "final_segmentation" / rel / "nodes_final.csv")[["segment_id", "class_id"]].values.T)))
    panels = []
    for t, (tl_, cl_) in tiles.items():
        off = 2 * int(meta.loc[t].x0)
        sub = np.zeros((rr1 - rr0, c1 - c0), np.int64); vis = np.zeros(sub.shape, bool)
        lo, hi = max(c0, off), min(c1, off + 1024)
        sub[:, lo - c0:hi - c0] = tl_[rr0:rr1, lo - off:hi - off]; vis[:, lo - c0:hi - c0] = True
        p = class_panel(img[rr0:rr1, c0:c1], sub, cl_); p[~vis] = 40
        p[:, seam - c0] = (255, 0, 0)
        panels.append(titled(cv2.resize(p, (640, 640), interpolation=cv2.INTER_NEAREST), t, "red = seam (overlap midline)"))
    ps = class_panel(img[rr0:rr1, c0:c1], lab[rr0:rr1, c0:c1], cls)
    sub = lab[rr0:rr1, c0:c1]; mm = np.isin(sub, merged).astype(np.uint8)
    ps[cv2.dilate(mm ^ cv2.erode(mm, np.ones((3, 3), np.uint8)), np.ones((2, 2), np.uint8)).astype(bool)] = (255, 255, 255)
    ps[:, seam - c0] = (255, 0, 0)
    panels.append(titled(cv2.resize(ps, (640, 640), interpolation=cv2.INTER_NEAREST), "Strip", "white = object re-joined across the seam"))
    Image.fromarray(titled(hstack(panels), "Step 3 · Seam handling between tile_00002 and tile_00003 (240 m window)",
                           "each tile segments its overlap independently; the strip keeps each pixel from the nearest tile centre and re-joins objects that both tiles saw in the overlap")).save(OUT / "fig3b_seam_zoom.png", optimize=True)


def fig4():
    nodes = pd.read_csv(STRIP / "stgraph_nodes.csv"); E = pd.read_csv(STRIP / "stgraph_edges.csv")
    v = json.loads((STRIP / "stgraph_validation.json").read_text())
    tr = rasterio.open(STRIP / "2023" / "segment_ids.tif").transform
    r0, c0, Sz = 380, 2250, 360
    W = Hp = 760; gap, sh = 190, 0.35
    canvas = Image.new("RGB", (W + int(sh * Hp) + 80, 2 * Hp + gap + 70), BG)
    def pxy(year, px, py):
        return 40 + px * W + (1 - py) * sh * Hp, 30 + (0 if year == "2023" else Hp + gap) + py * Hp
    for y in ("2023", "2013"):
        img = rasterio.open(STRIP / y / "image_2x.tif").read().transpose(1, 2, 0)[r0:r0 + Sz, c0:c0 + Sz]
        img = (0.6 * cv2.resize(img, (W, Hp), interpolation=cv2.INTER_CUBIC) + 0.4 * 30).astype(np.uint8)
        Mx = cv2.getAffineTransform(np.float32([[0, 0], [W, 0], [0, Hp]]), np.float32([pxy(y, 0, 0), pxy(y, 1, 0), pxy(y, 0, 1)]))
        warped = cv2.warpAffine(img, Mx, canvas.size); mask = cv2.warpAffine(np.ones((Hp, W), np.uint8), Mx, canvas.size)
        arr = np.array(canvas); arr[mask > 0] = warped[mask > 0]; canvas = Image.fromarray(arr)
    d = ImageDraw.Draw(canvas)
    nodes["px"] = ((nodes.node_x - tr.c) / tr.a - c0) / Sz; nodes["py"] = ((nodes.node_y - tr.f) / tr.e - r0) / Sz
    ins = nodes[(nodes.px.between(0, 1)) & (nodes.py.between(0, 1))]
    pos = {r.gid: pxy(str(r.year), r.px, r.py) for r in ins.itertuples()}
    sty = {"spatial_touches": ((210, 210, 210), 1), "spatial_near": ((255, 80, 220), 1), "temporal_overlap": ((110, 60, 60), 1),
           "temporal_identity": ((80, 230, 255), 2), "temporal_contained_in": ((255, 210, 60), 2)}
    for rel, (col, w) in sty.items():
        for r in E[E.relation == rel].itertuples():
            if r.src in pos and r.dst in pos:
                d.line([pos[r.src], pos[r.dst]], fill=col, width=w)
    for r in ins.itertuples():
        x, y = pos[r.gid]; d.ellipse((x - 6, y - 6, x + 6, y + 6), fill=sf.COLORS[r.class_id], outline=(0, 0, 0))
    d.text(pxy("2023", 0, 0), " 2023", font=font(24), fill="yellow"); d.text(pxy("2013", 0, 0), " 2013", font=font(24), fill="yellow")
    d.text((20, canvas.height - 34), "grey spatial touches · magenta spatial near · cyan temporal identity · yellow temporal contained-in · dark red temporal overlap (secondary)",
           font=font(15, False), fill=(220, 220, 220))
    br = v["by_relation"]
    fig = titled(np.array(canvas), "Step 4 · Spatio-temporal graph of the strip (window 270 m, tile_00002 area)",
                 f"{v['stgraph_nodes']} nodes (2013 + 2023) · {v['stgraph_edges']} edges: spatial {br['spatial_touches'] + br['spatial_near']}, "
                 f"temporal identity {br['temporal_identity']}, contained-in {br['temporal_contained_in']}, overlap {br['temporal_overlap']} · frozen temporal rules · checks PASS")
    Image.fromarray(fig).save(OUT / "fig4_spatiotemporal.png", optimize=True)


def fig5():
    s = json.loads((STRIP / "gnn" / "summary.json").read_text())
    # --- architecture
    fig, ax = plt.subplots(figsize=(15, 4.4), facecolor=SURF); ax.set_facecolor(SURF); ax.axis("off"); ax.set_xlim(0, 15); ax.set_ylim(0, 4.4)
    boxes = [("Spatio-temporal\ngraph", "7,020 nodes\n27,056 edges\n38 node features\nedge attributes"),
             ("Relational\nlayer 1", "per relation r:\nW_r h_j + U_r e_ij\nmean over neighbours\n+ W_0 h_i, ReLU"),
             ("Relational\nlayer 2", "64 hidden units\n6 relations: touches,\nnear, temporal primary\n& overlap (2 directions)"),
             ("Node head", "MLP → change logit\nweighted BCE\n(class imbalance)"),
             ("Output", "P(change) per node\n2023: new / extended\nbuilding\n2013: land built over")]
    bw, step = 2.45, 2.95
    for i, (t, sub) in enumerate(boxes):
        x = 0.2 + i * step
        ax.add_patch(FancyBboxPatch((x, 0.95), bw, 2.65, boxstyle="round,pad=0.05,rounding_size=0.08", fc="white", ec=RULE, lw=1.2))
        ax.text(x + bw / 2, 3.1, t, ha="center", va="center", fontsize=11, weight="bold", color=INK, linespacing=1.15)
        ax.text(x + bw / 2, 1.85, sub, ha="center", va="center", fontsize=9, color=INK2, linespacing=1.4)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + step - 0.02, 2.3), xytext=(x + bw + 0.08, 2.3), arrowprops=dict(arrowstyle="->", color=INK2, lw=1.4))
    ax.text(0.2, 4.15, "Step 5 · Relational graph neural network for change detection", fontsize=14, weight="bold", color=INK)
    ax.text(0.2, 0.45, "Labels: audited high-confidence rule-based changes (pseudo-labels); low / conflict / unverified masked; "
            "targets: 2023 buildings and 2013 land nodes.", fontsize=9, color=INK2)
    ax.text(0.2, 0.15, "Evaluation: leave-one-tile-out (each tile predicted by a model that never saw its labels).", fontsize=9, color=INK2)
    fig.savefig(OUT / "fig5a_gnn_architecture.png", dpi=150, facecolor=SURF, bbox_inches="tight"); plt.close(fig)

    # --- results: one axis 0-1, models as categories (slots 1-3), direct labels
    models = [("mlp", "MLP (node features only)", S1), ("gnn_spatial", "GNN, spatial edges only", S2), ("gnn_full", "GNN, spatial + temporal", S3)]
    metrics = [("ap", "Average precision"), ("f1", "F1"), ("roc_auc", "ROC-AUC")]
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURF); ax.set_facecolor(SURF)
    w = 0.26
    for i, (k, lab, col) in enumerate(models):
        vals = [s["models"][k][m] for m, _ in metrics]
        xs = np.arange(len(metrics)) + (i - 1) * (w + 0.02)
        ax.bar(xs, vals, width=w, color=col, label=lab, zorder=2, edgecolor=SURF, linewidth=2)
        for xx, vv in zip(xs, vals):
            ax.text(xx, vv + 0.015, f"{vv:.2f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(np.arange(len(metrics))); ax.set_xticklabels([m for _, m in metrics], color=INK, fontsize=10)
    ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=RULE, lw=0.8, zorder=0)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.legend(frameon=False, loc="upper left", fontsize=9, labelcolor=INK)
    ax.set_title(f"Held-out change detection ({s['labelled']:,} labelled nodes, {s['positives']} changes; leave-one-tile-out)", loc="left", fontsize=11, color=INK)
    fig.savefig(OUT / "fig5b_gnn_results.png", dpi=150, facecolor=SURF, bbox_inches="tight"); plt.close(fig)

    # --- change map (held-out predictions, full model)
    P = pd.read_csv(STRIP / "gnn" / "predictions.csv")
    k = 0.5; rows = []
    for y in ("2013", "2023"):
        lab = rasterio.open(STRIP / y / "segment_ids.tif").read(1)
        img = rasterio.open(STRIP / y / "image_2x.tif").read().transpose(1, 2, 0)
        base = (0.75 * img + 0.25 * 255 * 0.1).astype(np.uint8)
        q = P[P.year == int(y)]
        groups = {"TP": q[(q.label == 1) & (q.pred_gnn_full == 1)].segment_id, "FP": q[(q.label == 0) & (q.pred_gnn_full == 1)].segment_id,
                  "FN": q[(q.label == 1) & (q.pred_gnn_full == 0)].segment_id}
        colhex = {"TP": S1, "FP": S2, "FN": S3}
        for g_, ids in groups.items():
            m = np.isin(lab, ids.values).astype(np.uint8)
            c = tuple(int(colhex[g_][i:i + 2], 16) for i in (1, 3, 5))
            base[m.astype(bool)] = (0.45 * base[m.astype(bool)] + 0.55 * np.array(c)).astype(np.uint8)
            base[cv2.dilate(m ^ cv2.erode(m, np.ones((3, 3), np.uint8)), np.ones((3, 3), np.uint8)).astype(bool)] = c
        what = "2013 land nodes built over by 2023" if y == "2013" else "2023 new / extended buildings"
        rows.append(titled(cv2.resize(base, (int(base.shape[1] * k), int(base.shape[0] * k)), interpolation=cv2.INTER_AREA), f"{y}: {what}"))
    leg = Image.new("RGB", (rows[0].shape[1], 40), BG); d = ImageDraw.Draw(leg); x = 14
    for g_, t in (("TP", "true positive: GNN and rules agree on change"), ("FP", "false positive: GNN only"), ("FN", "false negative: rules only")):
        c = {"TP": S1, "FP": S2, "FN": S3}[g_]; d.rectangle((x, 11, x + 18, 29), fill=c)
        d.text((x + 24, 20), t, font=font(15, False), fill=(235, 235, 235), anchor="lm"); x += 44 + int(d.textlength(t, font=font(15, False)))
    m = s["models"]["gnn_full"]
    fig = titled(vstack(rows + [np.array(leg)]), "Step 5 · Held-out GNN change detection on the strip (2013 → 2023)",
                 f"spatio-temporal GNN: precision {m['precision']:.2f}, recall {m['recall']:.2f}, F1 {m['f1']:.2f} "
                 f"(2023 buildings F1 {m['by_year']['2023']['f1']:.2f}); every tile predicted by a model trained on the other tiles")
    Image.fromarray(fig).save(OUT / "fig5c_gnn_change_map.png", optimize=True)


def fig0():
    s = json.loads((STRIP / "gnn" / "summary.json").read_text()); v = json.loads((STRIP / "stgraph_validation.json").read_text())
    s13 = json.loads((STRIP / "2013" / "summary.json").read_text()); s23 = json.loads((STRIP / "2023" / "summary.json").read_text())
    steps = [("1  Tile segmentation", "7 tiles × 2 years\nfrozen, 15/15 checks each\n8 land-cover classes"),
             ("2  Tile graphs", "node = segment\nedges: touches, near\n37 node features"),
             ("3  Strip mosaic", f"placed by metadata x0, y0\n{s13['seam_merges'] + s23['seam_merges']} seam objects re-joined\n"
                                 f"{s13['segments']:,} / {s23['segments']:,} segments"),
             ("4  Spatio-temporal\ngraph", f"{v['stgraph_nodes']:,} nodes\n{v['stgraph_edges']:,} edges\nfrozen temporal rules"),
             ("5  GNN change\ndetection", f"held-out F1 {s['models']['gnn_full']['f1']:.2f}\nAP {s['models']['gnn_full']['ap']:.2f} "
                                         f"(MLP {s['models']['mlp']['ap']:.2f})\ntemporal edges are essential")]
    fig, ax = plt.subplots(figsize=(14, 3.3), facecolor=SURF); ax.set_facecolor(SURF); ax.axis("off"); ax.set_xlim(0, 14); ax.set_ylim(0, 3.3)
    for i, (t, sub) in enumerate(steps):
        x = 0.2 + i * 2.8
        ax.add_patch(FancyBboxPatch((x, 0.35), 2.35, 2.2, boxstyle="round,pad=0.05,rounding_size=0.08", fc="white", ec=RULE, lw=1.2))
        ax.text(x + 1.17, 2.15, t, ha="center", va="center", fontsize=11, weight="bold", color=INK, linespacing=1.2)
        ax.text(x + 1.17, 1.15, sub, ha="center", va="center", fontsize=9, color=INK2, linespacing=1.45)
        if i < 4:
            ax.annotate("", xy=(x + 2.75, 1.45), xytext=(x + 2.42, 1.45), arrowprops=dict(arrowstyle="->", color=INK2, lw=1.4))
    ax.text(0.2, 3.0, "From satellite images to a GNN change graph: Shouf strip, 2013 vs 2023", fontsize=14, weight="bold", color=INK)
    fig.savefig(OUT / "fig0_pipeline.png", dpi=150, facecolor=SURF, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    import sys
    todo = [globals()[n] for n in sys.argv[1:]] or [fig0, fig1_fig2, fig3, fig4, fig5]
    for f in todo:
        f(); print("done", f.__name__, flush=True)
