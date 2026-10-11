"""Show the GNN's own held-out change predictions on the imagery of one tile (read-only; nothing is retrained).

Reads graphs/strip_tile_00000-00006/gnn/predictions.csv (spatio-temporal GNN, leave-one-tile-out: the predictions
of tile k come from the model that never saw tile k's labels) and the strip rasters/images.
Writes gnn_viz_<tile>/:
  1_predictions_overlay.png    2013 | 2023 side by side, GNN-predicted changed nodes filled, P written on each,
                               the other year's predictions as thin outlines, rule changes the GNN missed dashed
  2_probability_map.png        P(change) of every scored node (log scale), 2013 | 2023, fold threshold marked
  3_gallery_2023.png / 3_gallery_2013.png   every predicted node: 2013 crop | 2023 crop, P, rule label, audit verdict
  predicted_nodes.csv
Usage: python validation/gnn_predictions_tile_viz.py tile_00002
"""
from __future__ import annotations

from osgeo import gdal  # noqa: F401
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parents[1]
STRIP = HERE / "graphs" / "strip_tile_00000-00006"
META = HERE.parent / "dataset" / "2023" / "metadata.csv"
TILE = sys.argv[1] if len(sys.argv) > 1 else "tile_00002"
OUT = HERE / f"gnn_viz_{TILE}"
# validated categorical slots (reference palette): 1 blue = 2013 land node predicted built over, 2 orange = 2023 building predicted new/extended
C13, C23 = (0x2a, 0x78, 0xd6), (0xeb, 0x68, 0x34)
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
BG, SURF, INK, INK2 = (24, 24, 28), (252, 252, 251), (11, 11, 11), (82, 81, 78)


def font(sz, bold=True):
    return ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf", sz)


def edge(m, w=2):
    m = m.astype(np.uint8)
    return cv2.dilate(m ^ cv2.erode(m, np.ones((3, 3), np.uint8)), np.ones((w, w), np.uint8)).astype(bool)


def dashed(mask_edge, period=10):
    yy, xx = np.indices(mask_edge.shape)
    return mask_edge & (((xx + yy) // period) % 2 == 0)


def titled(arr, title, sub=None, dark=True):
    h = 64 if sub else 40
    bg, fg, fg2 = (BG, "white", (200, 200, 200)) if dark else (SURF, INK, INK2)
    head = Image.new("RGB", (arr.shape[1], h), bg); d = ImageDraw.Draw(head)
    d.text((14, 8), title, font=font(21), fill=fg)
    if sub:
        d.text((14, 36), sub, font=font(15, False), fill=fg2)
    return np.concatenate([np.array(head), arr], 0)


def hstack(arrs, gap=10, fill=24):
    h = max(a.shape[0] for a in arrs)
    arrs = [np.pad(a, ((0, h - a.shape[0]), (0, 0), (0, 0)), constant_values=fill) for a in arrs]
    out = [arrs[0]]
    for a in arrs[1:]:
        out += [np.full((h, gap, 3), fill, np.uint8), a]
    return np.concatenate(out, 1)


def vstack(arrs, gap=10, fill=24):
    w = max(a.shape[1] for a in arrs)
    arrs = [np.pad(a, ((0, 0), (0, w - a.shape[1]), (0, 0)), constant_values=fill) for a in arrs]
    out = [arrs[0]]
    for a in arrs[1:]:
        out += [np.full((gap, w, 3), fill, np.uint8), a]
    return np.concatenate(out, 0)


def label_text(d, xy, txt, col, sz=14):
    d.text(xy, txt, font=font(sz), fill=col, anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0))


def main():
    OUT.mkdir(exist_ok=True)
    m = pd.read_csv(META).set_index("tile_id").loc[TILE]
    c0, c1, r0, r1 = 2 * int(m.x0), 2 * int(m.x1), 2 * int(m.y0), 2 * int(m.y1)
    P = pd.read_csv(STRIP / "gnn" / "predictions.csv")
    q = P[P.tile == TILE].copy()
    scored = q[q.prob_gnn_full.notna()]
    thr_lo = scored[scored.pred_gnn_full == 1].prob_gnn_full.min(); thr_hi = scored[scored.pred_gnn_full == 0].prob_gnn_full.max()
    # audit verdicts (2023 high-confidence building changes), mapped to strip nodes by the GNN script
    au = pd.read_csv(STRIP / "gnn" / "audit_agreement.csv")
    au = au[au.tile == TILE]
    verdict = {int(g): v for g, v in zip(au.strip_node, au.verdict) if g >= 0}
    img, lab = {}, {}
    for y in ("2013", "2023"):
        img[y] = rasterio.open(STRIP / y / "image_2x.tif").read([1, 2, 3]).transpose(1, 2, 0)[r0:r1, c0:c1].copy()
        lab[y] = rasterio.open(STRIP / y / "segment_ids.tif").read(1)[r0:r1, c0:c1]
    pred = scored[scored.pred_gnn_full == 1].copy()
    miss = scored[(scored.pred_gnn_full == 0) & (scored.label == 1)]
    # pixel centroid inside the window, for labels and crops
    def cen(y, sid):
        yy, xx = np.nonzero(lab[y] == sid)
        return (float(xx.mean()), float(yy.mean()), int(xx.min()), int(xx.max()), int(yy.min()), int(yy.max())) if len(xx) else None
    pred["c"] = [cen(str(r.year), r.segment_id) for r in pred.itertuples()]
    pred = pred[pred.c.notna()]
    rule = {1: "rule: change", 0: "rule: no change"}

    # ---- 1 overlay
    panels = []
    for y, col, other in (("2013", C13, "2023"), ("2023", C23, "2013")):
        base = img[y].astype(np.float32) * 0.82
        own = pred[pred.year == int(y)]
        for r in own.itertuples():
            mk = lab[y] == r.segment_id
            base[mk] = 0.5 * base[mk] + 0.5 * np.array(col)
        base = base.astype(np.uint8)
        for r in own.itertuples():
            e = edge(lab[y] == r.segment_id, 3); base[e] = col
        # the other year's predicted nodes as thin outlines (same place, other date)
        ocol = C23 if y == "2013" else C13
        for r in pred[pred.year == int(other)].itertuples():
            e = edge(lab[other] == r.segment_id, 2); base[e] = ocol
        for r in miss[miss.year == int(y)].itertuples():
            base[dashed(edge(lab[y] == r.segment_id, 2))] = (255, 255, 255)
        im = Image.fromarray(base); d = ImageDraw.Draw(im)
        for r in own.itertuples():
            label_text(d, (r.c[0], r.c[1]), f"{r.prob_gnn_full:.2f}" if r.prob_gnn_full >= 0.01 else f"{r.prob_gnn_full:.3f}", (255, 255, 255))
        what = "2013 land nodes the GNN predicts are built over by 2023" if y == "2013" else "2023 building nodes the GNN predicts are new / extended"
        panels.append(titled(np.array(im), f"{y}  ·  {len(own)} predicted changed", what))
    leg = Image.new("RGB", (panels[0].shape[1] * 2 + 10, 72), BG); d = ImageDraw.Draw(leg); x = 14
    items = [(C13, "fill", "2013 node predicted changed"), (C23, "fill", "2023 node predicted changed"),
             ((255, 255, 255), "dash", f"rule-labelled change the GNN missed ({len(miss)})")]
    for col, kind, t in items:
        if kind == "fill":
            d.rectangle((x, 12, x + 20, 30), fill=tuple(int(0.5 * c + 40) for c in col), outline=col, width=3)
        else:
            for k in range(0, 20, 6):
                d.line((x + k, 21, x + k + 3, 21), fill=col, width=2)
        d.text((x + 28, 21), t, font=font(15, False), fill=(235, 235, 235), anchor="lm"); x += 60 + int(d.textlength(t, font=font(15, False)))
    d.text((14, 50), "number on a node = GNN P(change); thin outline = the other year's predicted node at the same place; "
                     f"fold decision threshold ≈ {thr_lo:.3f} (F1-optimal on the validation tile)", font=font(14, False), fill=(200, 200, 200), anchor="lm")
    tp = int(((scored.pred_gnn_full == 1) & (scored.label == 1)).sum()); fp = int(((scored.pred_gnn_full == 1) & (scored.label == 0)).sum())
    fig = titled(vstack([hstack(panels), np.array(leg)], gap=0),
                 f"Spatio-temporal GNN change predictions, {TILE} (768 × 768 m), held out: the model never saw this tile's labels",
                 f"{len(pred)} nodes predicted changed out of {len(scored)} scored nodes · vs rule labels: {tp} agree, {fp} GNN-only, {len(miss)} missed")
    Image.fromarray(fig).save(OUT / "1_predictions_overlay.png", optimize=True)

    # ---- 2 probability map (log10 P from 1e-4 to 1)
    lo = -4
    seq = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] for h in SEQ], np.float32)
    def ramp(p):
        t = np.clip((np.log10(max(p, 1e-6)) - lo) / -lo, 0, 1) * (len(seq) - 1)
        i = int(np.floor(t)); f = t - i
        return seq[i] if i >= len(seq) - 1 else (1 - f) * seq[i] + f * seq[i + 1]
    pm = []
    for y in ("2013", "2023"):
        canvas = np.full(lab[y].shape + (3,), 236, np.float32)
        ids = scored[scored.year == int(y)]
        lut = {int(r.segment_id): ramp(r.prob_gnn_full) for r in ids.itertuples()}
        for sid, c in lut.items():
            canvas[lab[y] == sid] = c
        canvas = canvas.astype(np.uint8)
        bnd = np.zeros(lab[y].shape, bool); bnd[:, 1:] |= lab[y][:, 1:] != lab[y][:, :-1]; bnd[1:] |= lab[y][1:] != lab[y][:-1]
        canvas[bnd] = (255, 255, 255)
        for r in pred[pred.year == int(y)].itertuples():
            canvas[edge(lab[y] == r.segment_id, 3)] = INK
        what = "scored: 2013 land nodes (grass, tree, bare soil, …)" if y == "2013" else "scored: 2023 building nodes"
        pm.append(titled(canvas, f"{y}  ·  {len(ids)} scored nodes", what + " · grey = not scored", dark=False))
    W = pm[0].shape[1] * 2 + 10
    cb = Image.new("RGB", (W, 86), SURF); d = ImageDraw.Draw(cb); x0, x1 = 60, W - 60
    for k in range(x1 - x0):
        p = 10 ** (lo + (-lo) * k / (x1 - x0 - 1)); d.line((x0 + k, 14, x0 + k, 34), fill=tuple(int(v) for v in ramp(p)))
    for e in range(lo, 1):
        xx = x0 + (e - lo) / -lo * (x1 - x0 - 1); d.line((xx, 34, xx, 40), fill=INK2)
        d.text((xx, 48), f"{10 ** e:g}", font=font(14, False), fill=INK2, anchor="mt")
    xt = x0 + (np.log10(thr_lo) - lo) / -lo * (x1 - x0 - 1)
    d.line((xt, 8, xt, 40), fill=INK, width=2); d.text((xt + 6, 64), f"fold threshold ≈ {thr_lo:.3f}: nodes right of it are predicted changed (black outline)", font=font(14, False), fill=INK, anchor="lm")
    fig2 = titled(vstack([hstack(pm, fill=252), np.array(cb)], gap=0, fill=252),
                  f"GNN P(change) for every scored node, {TILE} (log scale)",
                  "the full model output, before thresholding · light = unlikely, dark = likely change", dark=False)
    Image.fromarray(fig2).save(OUT / "2_probability_map.png", optimize=True)

    # ---- 3 galleries: 2013 crop | 2023 crop per predicted node
    rows_out = []
    for yr, col in ((2023, C23), (2013, C13)):
        g = pred[pred.year == yr].sort_values("prob_gnn_full", ascending=False)
        cells = []
        for i, r in enumerate(g.itertuples(), 1):
            cx, cy, xa, xb, ya, yb = r.c
            half = int(max(80, 0.65 * max(xb - xa, yb - ya)))
            a, b = int(cy) - half, int(cx) - half
            pair = []
            for y in ("2013", "2023"):
                big = np.pad(img[y], ((half, half), (half, half), (0, 0)))
                crop = big[a + half:a + 3 * half, b + half:b + 3 * half].copy()
                mk = np.pad(lab[str(yr)] == r.segment_id, half)[a + half:a + 3 * half, b + half:b + 3 * half]
                crop[edge(mk, 2)] = col
                crop = cv2.resize(crop, (230, 230), interpolation=cv2.INTER_CUBIC)
                im = Image.fromarray(crop); d = ImageDraw.Draw(im); d.text((6, 4), y, font=font(14), fill="white", stroke_width=2, stroke_fill="black")
                pair.append(np.array(im))
            v = verdict.get(int(r.gid), "")
            sub = f"{r._5} {r.area_m2:.0f} m² · {rule.get(int(r.label), '')}" + (f" · audit: {v}" if v else "")
            cell = titled(hstack(pair, gap=4), f"#{i}  P = {r.prob_gnn_full:.3f}", sub)
            cells.append(cell)
            rows_out.append({"year": yr, "item": i, "segment_id": r.segment_id, "class": r._5, "area_m2": r.area_m2, "prob_gnn_full": r.prob_gnn_full,
                             "rule_label": int(r.label), "audit_verdict": v, "crop_side_m": round(2 * half * 0.75)})
        grid = [hstack(cells[k:k + 4]) for k in range(0, len(cells), 4)]
        what = "2023 buildings predicted new / extended" if yr == 2023 else "2013 land nodes predicted built over by 2023"
        fig3 = titled(vstack(grid), f"{TILE} · {what} · {len(cells)} nodes, sorted by GNN probability",
                      "each cell: 2013 | 2023 at the same place, the predicted node outlined in both; crop side ≥ 120 m")
        Image.fromarray(fig3).save(OUT / f"3_gallery_{yr}.png", optimize=True)
    pd.DataFrame(rows_out).to_csv(OUT / "predicted_nodes.csv", index=False)
    print(f"scored {len(scored)}, predicted {len(pred)}, missed {len(miss)}, threshold between {thr_hi} and {thr_lo}; tp {tp} fp {fp}")


if __name__ == "__main__":
    main()
