"""Final merging/refinement of adjacent same-class segments (region adjacency graph).

Goal: one continuous real region / object = one segment. Adjacent segments of the same class are
merged only when the evidence says they are ONE region; separate real objects stay separate.

Rules by class
  building      never merged: the validated neck split separated touching houses on purpose.
  road          two road segments are merged only if they meet where no third road segment meets
                (a false junction from skeleton noise); real intersections keep streets separate.
  land / paved  best-first agglomerative merging; a pair merges only if ALL hold:
  / water         1. no visible boundary: mean gradient along the shared boundary is not clearly
                     stronger than the gradient inside the two segments (edge_ratio);
                  2. same appearance: Lab colour means differ little relative to their spread
                     (appearance_d), and texture (Lab std) is similar (texture_ratio);
                  3. broad contact, not a neck: shared boundary length >= neck_factor * sqrt(area of
                     the smaller segment) and >= min_contact_m;
                  4. same semantic profile: L1 distance of mean class-probability vectors.
                Delineate Anything parcels (fields/terraces/lots) need stricter evidence because
                their boundaries are usually real.
Every candidate pair and the decision with its evidence is logged (merge_log.csv).
"""
from __future__ import annotations

import math
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage

import segment_v2 as v2

M = {
    "edge_ratio_max": 1.25,
    "appearance_d_max": 0.9,
    "texture_ratio_max": 1.6,
    "semantic_l1_max": 0.45,
    "neck_factor": 0.35,
    "min_contact_m": 4.0,
    # stricter for Delineate Anything parcels (their boundaries are usually real field edges)
    "parcel_edge_ratio_max": 1.05,
    "parcel_appearance_d_max": 0.5,
    # roads: a contact is a false junction if no third road segment lies within this radius
    "road_junction_radius_m": 8.0,
}
NEVER = {1}          # building
ROAD = 2


class RAG:
    def __init__(self, labels, cls, src, image2x, cp, grad, px_m):
        self.labels = labels
        self.cls = dict(cls)
        self.src = dict(src)
        self.px_m = px_m
        lab = cv2.cvtColor(image2x, cv2.COLOR_RGB2LAB).astype(np.float64)
        n = int(labels.max()) + 1
        flat = labels.ravel()
        self.cnt = np.bincount(flat, minlength=n).astype(np.float64)
        self.s1 = np.stack([np.bincount(flat, weights=lab[:, :, c].ravel(), minlength=n) for c in range(3)], 1)
        self.s2 = np.stack([np.bincount(flat, weights=(lab[:, :, c] ** 2).ravel(), minlength=n) for c in range(3)], 1)
        self.sg = np.bincount(flat, weights=grad.ravel(), minlength=n)
        self.sp = np.stack([np.bincount(flat, weights=cp[c].ravel(), minlength=n) for c in range(cp.shape[0])], 1)
        # pair statistics: boundary pixel pairs, gradient along boundary, contact centroid
        self.pair = defaultdict(lambda: np.zeros(4))  # count, grad_sum, y_sum, x_sum
        self.perim = np.zeros(n)
        H, W = labels.shape
        for a, b, sl_a, sl_b, diff in v2.pair_boundaries(labels):
            ga = grad[sl_a][diff]; gb = grad[sl_b][diff]
            ys, xs = np.nonzero(diff)
            g = (ga + gb) / 2
            lo, hi = np.minimum(a, b), np.maximum(a, b)
            key = lo.astype(np.int64) * 10_000_000 + hi
            u, inv = np.unique(key, return_inverse=True)
            c = np.bincount(inv); gs = np.bincount(inv, weights=g)
            yy = np.bincount(inv, weights=ys); xx = np.bincount(inv, weights=xs)
            for k, ci, gi, yi, xi in zip(u, c, gs, yy, xx):
                self.pair[(int(k // 10_000_000), int(k % 10_000_000))] += (ci, gi, yi, xi)
            self.perim += np.bincount(a, minlength=n) + np.bincount(b, minlength=n)
        border = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
        self.perim += np.bincount(border, minlength=n)
        self.parent = np.arange(n)
        self.members = {i: [i] for i in range(1, n) if self.cnt[i] > 0}

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def neighbours(self):
        out = defaultdict(set)
        for a, b in self.pair:
            out[a].add(b); out[b].add(a)
        return out

    def features(self, a, b):
        c, gsum, ysum, xsum = self.pair[(min(a, b), max(a, b))]
        mean = lambda i: self.s1[i] / self.cnt[i]
        std = lambda i: np.sqrt(np.maximum(self.s2[i] / self.cnt[i] - mean(i) ** 2, 1e-6))
        inner = (self.sg[a] + self.sg[b]) / (self.cnt[a] + self.cnt[b])
        edge_ratio = (gsum / c) / max(inner, 1e-6)
        pooled = np.sqrt((std(a) ** 2 + std(b) ** 2) / 2)
        appearance_d = float(np.linalg.norm((mean(a) - mean(b)) / np.maximum(pooled, 1.0)) / math.sqrt(3))
        ta, tb = std(a).mean(), std(b).mean()
        texture_ratio = max(ta, tb) / max(min(ta, tb), 1e-6)
        pa, pb = self.sp[a] / self.cnt[a], self.sp[b] / self.cnt[b]
        semantic_l1 = float(np.abs(pa - pb).sum())
        contact_m = c * self.px_m
        small_area_m2 = min(self.cnt[a], self.cnt[b]) * self.px_m ** 2
        neck_ok = contact_m >= max(M["min_contact_m"], M["neck_factor"] * math.sqrt(small_area_m2))
        return {"contact_m": round(contact_m, 2), "edge_ratio": round(float(edge_ratio), 3),
                "appearance_d": round(appearance_d, 3), "texture_ratio": round(float(texture_ratio), 3),
                "semantic_l1": round(semantic_l1, 3), "neck_ok": bool(neck_ok),
                "contact_yx": (ysum / c, xsum / c)}

    def merge(self, a, b):
        """Merge b into a (both roots); update statistics and pair table."""
        self.parent[b] = a
        for arr in (self.cnt, self.sg, self.perim):
            arr[a] += arr[b]
        self.s1[a] += self.s1[b]; self.s2[a] += self.s2[b]; self.sp[a] += self.sp[b]
        shared = self.pair.pop((min(a, b), max(a, b)))
        self.perim[a] -= 2 * shared[0]
        for (x, y) in [k for k in self.pair if b in k]:
            other = y if x == b else x
            vals = self.pair.pop((x, y))
            self.pair[(min(a, other), max(a, other))] += vals
        self.members[a] += self.members.pop(b)
        self.src[a] = self.src[a] if self.src[a] == self.src[b] else f"{self.src[a]}+merged"


def decide(f, is_parcel_pair):
    er_max = M["parcel_edge_ratio_max"] if is_parcel_pair else M["edge_ratio_max"]
    ad_max = M["parcel_appearance_d_max"] if is_parcel_pair else M["appearance_d_max"]
    reasons = []
    if not f["neck_ok"]:
        reasons.append("narrow_contact")
    if f["edge_ratio"] > er_max:
        reasons.append("visible_boundary")
    if f["appearance_d"] > ad_max:
        reasons.append("different_appearance")
    if f["texture_ratio"] > M["texture_ratio_max"]:
        reasons.append("different_texture")
    if f["semantic_l1"] > M["semantic_l1_max"]:
        reasons.append("different_semantics")
    return not reasons, reasons


def refine(labels, cls, src, image2x, cp, grad, px_m):
    """Return (new labels, new cls, new src, merge log DataFrame)."""
    g = RAG(labels, cls, src, image2x, cp, grad, px_m)
    log = []
    # --- roads: merge only at false junctions
    road_ids = [i for i, c in g.cls.items() if c == ROAD]
    road_mask_lab = np.where(np.isin(labels, road_ids), labels, 0)
    r_px = int(round(M["road_junction_radius_m"] / px_m))
    objs = ndimage.find_objects(labels)
    for (a, b) in list(g.pair):
        if g.cls.get(a) != ROAD or g.cls.get(b) != ROAD:
            continue
        f = g.features(a, b)
        # the whole contact line between a and b, dilated by the junction radius: if any third road
        # segment reaches that zone, the contact is a real junction (wide intersections included)
        sa, sb = objs[a - 1], objs[b - 1]
        y0 = max(0, min(sa[0].start, sb[0].start) - r_px); y1 = max(sa[0].stop, sb[0].stop) + r_px
        x0 = max(0, min(sa[1].start, sb[1].start) - r_px); x1 = max(sa[1].stop, sb[1].stop) + r_px
        sub = labels[y0:y1, x0:x1]
        ma, mb = sub == a, sub == b
        contact = (ma & ndimage.binary_dilation(mb, structure=v2.FOUR)) | (mb & ndimage.binary_dilation(ma, structure=v2.FOUR))
        zone = ndimage.binary_dilation(contact, iterations=r_px)
        segs = set(np.unique(road_mask_lab[y0:y1, x0:x1][zone]).tolist()) - {0}
        merge = segs <= {a, b}
        log.append({"a": a, "b": b, "class": "road", **{k: v for k, v in f.items() if k != "contact_yx"},
                    "decision": "merge" if merge else "keep",
                    "reason": "false_junction (no third road segment)" if merge else f"junction with {len(segs)} road segments"})
    for row in [r for r in log if r["decision"] == "merge"]:
        ra, rb = g.find(row["a"]), g.find(row["b"])
        if ra != rb and (min(ra, rb), max(ra, rb)) in g.pair:
            g.merge(ra, rb)

    # --- land / paved / water: best-first agglomeration
    decided = {}
    while True:
        best = None
        for (a, b) in g.pair:
            ca, cb = g.cls.get(a), g.cls.get(b)
            if ca != cb or ca in NEVER or ca == ROAD:
                continue
            f = g.features(a, b)
            is_parcel = "delany_parcel" in g.src[a] or "delany_parcel" in g.src[b]
            ok, reasons = decide(f, is_parcel)
            key = (min(a, b), max(a, b))
            decided[key] = (f, ok, reasons, ca, is_parcel)
            if ok:
                score = f["edge_ratio"] + f["appearance_d"]
                if best is None or score < best[0]:
                    best = (score, a, b)
        if best is None:
            break
        _, a, b = best
        f, _, _, ca, is_parcel = decided[(min(a, b), max(a, b))]
        log.append({"a": a, "b": b, "class": ca, **{k: v for k, v in f.items() if k != "contact_yx"},
                    "decision": "merge", "reason": "one continuous region" + (" (parcel, strict)" if is_parcel else "")})
        keep, gone = (a, b) if g.cnt[a] >= g.cnt[b] else (b, a)
        g.merge(keep, gone)
    for key, (f, ok, reasons, ca, is_parcel) in decided.items():
        if key in g.pair and not ok:
            log.append({"a": key[0], "b": key[1], "class": ca, **{k: v for k, v in f.items() if k != "contact_yx"},
                        "decision": "keep", "reason": ",".join(reasons)})
    # building pairs: logged as kept by rule
    for (a, b) in g.pair:
        if g.cls.get(a) == g.cls.get(b) == 1:
            log.append({"a": a, "b": b, "class": 1, "decision": "keep", "reason": "buildings are never merged (validated split)"})

    # relabel
    n = len(g.parent)
    root = np.array([g.find(i) for i in range(n)])
    new = root[labels]
    ids = np.unique(new); ids = ids[ids > 0]
    lut = np.zeros(n, np.int64); lut[ids] = np.arange(1, len(ids) + 1)
    out = lut[new].astype(np.int32)
    new_cls = {int(lut[i]): g.cls[i] for i in ids}
    new_src = {int(lut[i]): g.src[i] for i in ids}
    return out, new_cls, new_src, pd.DataFrame(log)
