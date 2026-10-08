"""Visualization of a final segmentation: original | segment boundaries | segments coloured by class,
plus zoom windows and per-class counts."""
from osgeo import gdal  # noqa: F401
import argparse
import json
from pathlib import Path

import fonts  # package: portable font paths

import cv2
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw, ImageFont

from segment_final import CLASSES, COLORS, OUT_ROOT, ROOT


def font(size, bold=True):
    return ImageFont.truetype(fonts.BOLD if bold else fonts.REGULAR, size)


def boundaries(lab):
    e = np.zeros(lab.shape, bool)
    e[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    e[1:, :] |= lab[1:, :] != lab[:-1, :]
    return e


def panels(img2x, seg, nodes):
    rng = np.random.default_rng(1)
    cls = np.zeros(seg.max() + 1, np.int32); cls[nodes.node_id] = nodes.class_id
    base = np.array([COLORS.get(c, (0, 0, 0)) for c in range(9)], np.float32)
    shade = rng.uniform(-38, 38, (seg.max() + 1, 1)).astype(np.float32)
    seg_col = np.clip(base[cls] + shade, 0, 255)[seg]
    edge = boundaries(seg)
    p_bound = img2x.copy(); p_bound[edge] = (255, 255, 0)
    p_cls = (0.25 * img2x + 0.75 * seg_col).astype(np.uint8); p_cls[edge] = (20, 20, 20)
    return p_bound, p_cls


def legend_bar(width, nodes, total):
    bar = Image.new("RGB", (width, 46), (24, 24, 28)); d = ImageDraw.Draw(bar)
    x = 14
    counts = nodes["class"].value_counts()
    for cid, name in CLASSES.items():
        d.rectangle((x, 14, x + 18, 32), fill=COLORS[cid])
        txt = f"{name} ({counts.get(name, 0)})"
        d.text((x + 24, 23), txt, font=font(15, False), fill=(235, 235, 235), anchor="lm")
        x += 34 + int(d.textlength(txt, font=font(15, False)))
    d.text((width - 14, 23), f"{total} segments", font=font(16), fill=(255, 255, 255), anchor="rm")
    return np.array(bar)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="2023_tile_00000")
    ap.add_argument("--zooms", default="[[0,0,256],[256,128,256],[60,300,200]]", help="[[row,col,size] native px]")
    args = ap.parse_args()
    out = OUT_ROOT / args.run
    log = json.loads((out / "run_log.json").read_text())
    with rasterio.open(log["image"]) as s:
        img = s.read([1, 2, 3]).transpose(1, 2, 0).astype(np.uint8)
    seg = rasterio.open(out / "segment_ids.tif").read(1)
    nodes = pd.read_csv(out / "nodes.csv")
    img2x = cv2.resize(img, seg.shape[::-1], interpolation=cv2.INTER_CUBIC)
    pb, pc = panels(img2x, seg, nodes)
    head = 48
    W = seg.shape[1]
    canvas = Image.new("RGB", (3 * W + 20, seg.shape[0] + head), (24, 24, 28))
    for i, (p, t) in enumerate(((img2x, "ORIGINAL"), (pb, "SEGMENT BOUNDARIES"), (pc, "SEGMENTS BY CLASS"))):
        canvas.paste(Image.fromarray(p), (i * (W + 10), head))
        ImageDraw.Draw(canvas).text((i * (W + 10) + W / 2, head / 2), t, font=font(22), fill="white", anchor="mm")
    full = np.concatenate([np.array(canvas), legend_bar(canvas.width, nodes, len(nodes))], 0)
    Image.fromarray(full).save(out / "viz_full.png", optimize=True)
    rows = []
    for r0, c0, s in json.loads(args.zooms):
        sl = (slice(2 * r0, 2 * (r0 + s)), slice(2 * c0, 2 * (c0 + s)))
        tiles = [cv2.resize(x[sl], (600, 600), interpolation=cv2.INTER_NEAREST if x is not img2x else cv2.INTER_CUBIC) for x in (img2x, pb, pc)]
        rows.append(np.concatenate([tiles[0], np.full((600, 10, 3), 24, np.uint8), tiles[1], np.full((600, 10, 3), 24, np.uint8), tiles[2]], 1))
    z = np.concatenate(sum([[r, np.full((10, rows[0].shape[1], 3), 24, np.uint8)] for r in rows], [])[:-1], 0)
    z = np.concatenate([z, legend_bar(z.shape[1], nodes, len(nodes))], 0)
    Image.fromarray(z).save(out / "viz_zooms.png", optimize=True)
    print("saved", out / "viz_full.png", out / "viz_zooms.png")


if __name__ == "__main__":
    main()
