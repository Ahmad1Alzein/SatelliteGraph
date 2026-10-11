"""GNN change detection on the strip spatio-temporal graph (plain PyTorch, relational message passing).

Labels (pseudo-labels from the audited rule-based change detection, transferred to strip nodes by pixel lineage)
  1  2023 building that is new or extended (high confidence)   |  2013 land node built over (high confidence)
  0  persistent / segmentation difference / no evidence (both years)
  masked: low / conflict / unverified candidates, and nodes whose pixels are mostly such candidates;
          nodes the change type does not apply to (2023 non-buildings, 2013 buildings) are not targets (run 2;
          run 1 scored them as negatives, see summary_run1_all_nodes_scored.json)
Model   R-GCN style: h_i' = ReLU(W0 h_i + sum_r mean_{j in N_r(i)} (W_r h_j + U_r e_ij)), 2 layers, 64 hidden.
        relations: spatial_touches, spatial_near (both directions), temporal primary (identity+contained_in)
        forward/backward, temporal overlap forward/backward. Edge attributes enter through U_r.
Ablations  mlp (no edges), gnn_spatial (spatial relations only), gnn_full (spatial + temporal).
Evaluation leave-one-tile-out over the 7 strip tiles (node tile = tile contributing most pixels);
           validation tile = next tile (threshold + early stopping); metrics pooled over test folds.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
S = HERE / "graphs" / "strip_tile_00000-00006"
import sys
if len(sys.argv) > 1 and sys.argv[1].startswith("--strip="):
    S = Path(sys.argv[1].split("=", 1)[1])
OUT = S / "gnn"
TILES = [f"tile_{i:05d}" for i in range(7)]
POS23 = {"new_building", "extension"}
NEG = {"persistent", "segmentation_difference", "unexplained_no_evidence", "unchanged"}
# CPU + deterministic algorithms: GPU scatter-add (index_add_) is not bit-deterministic and changed held-out
# predictions between identical runs (freeze check F4), so results are produced deterministically on CPU.
DEV = "cpu"
torch.use_deterministic_algorithms(True)
torch.set_num_threads(4)
SEED = 0


# ----------------------------------------------------------------------------- data
def labels(nodes, lineage):
    tab = {}
    for t in TILES:
        d = pd.read_csv(HERE / "graphs" / f"change_stgraph_2013_2023_{t}" / "nodes_change.csv")
        for r in d.itertuples():
            if r.year == 2023:
                lab = 1 if (r.change in POS23 and r.confidence == "high") else (0 if r.change in NEG else -1)
            else:
                lab = 1 if r.change == "built_over" else (0 if r.change in NEG else -1)
            tab[(str(r.year), t, int(r.segment_id))] = lab
    y = np.full(len(nodes), -1, np.int64); tile_of = []
    for i, (yr, lin) in enumerate(zip(nodes.year.astype(str), lineage)):
        votes = {1: 0, 0: 0, -1: 0}; per_tile = {}
        for part in lin.split("|"):
            t, s, px = part.split(":")
            votes[tab[(yr, t, int(s))]] += int(px); per_tile[t] = per_tile.get(t, 0) + int(px)
        tot = sum(votes.values())
        y[i] = 1 if votes[1] >= 0.5 * tot else (-1 if votes[-1] >= 0.5 * tot else 0)
        # run 2: a node is a target only where the change type applies to it
        #   2023 -> building nodes (new / extended?), 2013 -> non-building nodes (built over?)
        is_b = nodes["class"].iloc[i] == "building"
        if (yr == "2023" and not is_b) or (yr == "2013" and is_b):
            y[i] = -1
        tile_of.append(max(per_tile, key=per_tile.get))
    return y, np.array([TILES.index(t) for t in tile_of])


def relations(E, mode):
    """Directed relation edge lists with attributes (padded to 10 dims)."""
    spat = ["shared_boundary_m", "gap_m", "distance_m", "boundary_frac_source", "boundary_frac_target", "color_contrast", "same_class", "dir_sin", "dir_cos"]
    temp = ["iou", "frac_2013", "frac_2023", "overlap_score", "shared_m2", "class_agreement", "confidence_ord", "frame_roof", "roof_shift_dx_m", "roof_shift_dy_m"]
    out = {}
    def add(name, src, dst, attr):
        a = np.nan_to_num(attr.astype(np.float32))
        a = (a - a.mean(0)) / (a.std(0) + 1e-6)
        a = np.pad(a, ((0, 0), (0, 10 - a.shape[1])))
        out[name] = (torch.tensor(np.stack([src, dst])), torch.tensor(a))
    if mode == "mlp":
        return out
    for rel in ("spatial_touches", "spatial_near"):
        e = E[E.relation == rel]
        s, d, a = e.src.values, e.dst.values, e[spat].values
        add(rel, np.r_[s, d], np.r_[d, s], np.r_[a, a])
    if mode == "gnn_full":
        for name, rels in (("t_primary", ["temporal_identity", "temporal_contained_in"]), ("t_overlap", ["temporal_overlap"])):
            e = E[E.relation.isin(rels)]
            add(name + "_fwd", e.src.values, e.dst.values, e[temp].values)
            add(name + "_bwd", e.dst.values, e.src.values, e[temp].values)
    return out


class RGNN(nn.Module):
    def __init__(self, din, rels, hidden=64, layers=2, drop=0.3):
        super().__init__()
        self.rels = list(rels); self.drop = drop
        dims = [din] + [hidden] * layers
        self.self_w = nn.ModuleList([nn.Linear(dims[k], dims[k + 1]) for k in range(layers)])
        self.rel_w = nn.ModuleList([nn.ModuleDict({r: nn.Linear(dims[k], dims[k + 1], bias=False) for r in self.rels}) for k in range(layers)])
        self.edge_w = nn.ModuleList([nn.ModuleDict({r: nn.Linear(10, dims[k + 1], bias=False) for r in self.rels}) for k in range(layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(drop), nn.Linear(hidden, 1))

    def forward(self, x, R):
        h = x
        for k in range(len(self.self_w)):
            out = self.self_w[k](h)
            for r in self.rels:
                (ei, ea) = R[r]
                msg = self.rel_w[k][r](h[ei[0]]) + self.edge_w[k][r](ea)
                agg = torch.zeros_like(out).index_add_(0, ei[1], msg)
                deg = torch.zeros(out.shape[0], device=out.device).index_add_(0, ei[1], torch.ones(ei.shape[1], device=out.device)).clamp(min=1)
                out = out + agg / deg[:, None]
            h = F.dropout(F.relu(out), self.drop, self.training)
        return self.head(h).squeeze(-1)


def ap_score(y, p):
    o = np.argsort(-p); y = y[o]; tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(y.sum(), 1))


def auc_score(y, p):
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = pd.Series(np.r_[pos, neg]).rank().values
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def prf(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9), tp, fp, fn


def train_fold(x, R, y, train, val, epochs=300, lr=5e-3):
    torch.manual_seed(SEED)
    m = RGNN(x.shape[1], R.keys()).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=5e-4)
    pw = torch.tensor((y[train] == 0).sum() / max((y[train] == 1).sum(), 1), device=DEV)
    yt = torch.tensor(y, dtype=torch.float32, device=DEV)
    tr_idx = torch.tensor(np.flatnonzero(train), device=DEV)
    best, best_state, hist = -1, None, []
    for ep in range(epochs):
        m.train(); opt.zero_grad()
        logit = m(x, R)
        loss = F.binary_cross_entropy_with_logits(logit[tr_idx], yt[tr_idx], pos_weight=pw)
        loss.backward(); opt.step()
        if ep % 5 == 4:
            m.eval()
            with torch.no_grad():
                p = torch.sigmoid(m(x, R)).cpu().numpy()
            vap = ap_score(y[val], p[val]) if (y[val] == 1).any() else float("nan")
            hist.append({"epoch": ep + 1, "loss": float(loss), "val_ap": vap, "train_ap": ap_score(y[train], p[train])})
            score = vap if not np.isnan(vap) else -float(loss)
            if score > best:
                best, best_state = score, {k: v.detach().clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad():
        p = torch.sigmoid(m(x, R)).cpu().numpy()
    # threshold maximising F1 on validation (fallback 0.5)
    thr = 0.5
    if (y[val] == 1).any():
        cand = np.unique(np.round(p[val], 3))
        f1s = [prf(y[val], (p[val] >= t).astype(int))[2] for t in cand]
        thr = float(cand[int(np.argmax(f1s))])
    return p, thr, hist


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    nodes = pd.read_csv(S / "stgraph_nodes.csv"); lin = pd.read_csv(S / "stgraph_node_lineage.csv").lineage.values
    E = pd.read_csv(S / "stgraph_edges.csv")
    T = torch.load(S / "stgraph_tensors.pt", weights_only=False)
    x = T["x"].to(DEV)
    y, tile = labels(nodes, lin)
    lab = y >= 0
    summary = {"nodes": len(nodes), "labelled": int(lab.sum()), "positives": int((y == 1).sum()),
               "positives_2023": int(((y == 1) & (nodes.year == 2023)).sum()), "positives_2013": int(((y == 1) & (nodes.year == 2013)).sum()),
               "masked": int((~lab).sum()), "device": DEV, "models": {}}
    preds = {}
    for mode in ("mlp", "gnn_spatial", "gnn_full"):
        R = {k: (ei.to(DEV), ea.to(DEV)) for k, (ei, ea) in relations(E, mode).items()}
        P = np.full(len(nodes), np.nan); PRED = np.zeros(len(nodes), int); hist_all = {}; t0 = time.time()
        for k in range(7):
            v = (k + 1) % 7
            test, val = lab & (tile == k), lab & (tile == v)
            train = lab & ~np.isin(tile, [k, v])
            p, thr, hist = train_fold(x, R, y, train, val)
            P[test] = p[test]; PRED[test] = (p[test] >= thr).astype(int); hist_all[TILES[k]] = hist
        ev = lab & ~np.isnan(P)
        pr, rc, f1, tp, fp, fn = prf(y[ev], PRED[ev])
        per_year = {}
        for yr in (2013, 2023):
            s = ev & (nodes.year.values == yr)
            a, b, c, *_ = prf(y[s], PRED[s]); per_year[yr] = {"precision": round(a, 3), "recall": round(b, 3), "f1": round(c, 3), "ap": round(ap_score(y[s], P[s]), 3)}
        summary["models"][mode] = {"ap": round(ap_score(y[ev], P[ev]), 3), "roc_auc": round(auc_score(y[ev], P[ev]), 3),
                                   "precision": round(pr, 3), "recall": round(rc, 3), "f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn,
                                   "by_year": per_year, "seconds": round(time.time() - t0, 1)}
        preds[mode] = (P, PRED)
        json.dump(hist_all, open(OUT / f"history_{mode}.json", "w"))
        print(mode, json.dumps(summary["models"][mode]), flush=True)
    df = nodes[["gid", "key", "year", "segment_id", "class", "area_m2", "node_x", "node_y"]].copy()
    df["tile"] = [TILES[t] for t in tile]; df["label"] = y
    for mode, (P, PRED) in preds.items():
        df[f"prob_{mode}"] = np.round(P, 4); df[f"pred_{mode}"] = PRED
    df.to_csv(OUT / "predictions.csv", index=False)

    # agreement with the human audit (audited high-confidence building changes, tiles 0-5)
    rows = []
    for t in TILES:
        a = HERE / "graphs" / f"change_stgraph_2013_2023_{t}" / "audit"
        f = a / ("A_high_confidence_building_changes.csv" if (a / "A_high_confidence_building_changes.csv").exists() else "A_building_changes.csv")
        if not f.exists():
            continue
        d = pd.read_csv(f)
        if "confidence" in d.columns:
            d = d[d.confidence == "high"]
        if "verdict" not in d.columns:
            continue
        for r in d.itertuples():
            rows.append({"tile": t, "segment_id": int(r.segment_id), "verdict": r.verdict})
    au = pd.DataFrame(rows)
    look = {}
    for i, (yr, l) in enumerate(zip(nodes.year, lin)):
        if yr != 2023:
            continue
        for part in l.split("|"):
            t, s_, px = part.split(":")
            look.setdefault((t, int(s_)), []).append((int(px), i))
    au["strip_node"] = [max(look[(t, s_)])[1] if (t, s_) in look else -1 for t, s_ in zip(au.tile, au.segment_id)]
    P, PRED = preds["gnn_full"]
    au["gnn_prob"] = [P[i] if i >= 0 else np.nan for i in au.strip_node]
    au["gnn_detected"] = [int(PRED[i]) if i >= 0 else -1 for i in au.strip_node]
    au.to_csv(OUT / "audit_agreement.csv", index=False)
    summary["audit_agreement"] = {v: {"n": int((au.verdict == v).sum()), "detected_by_gnn": int(((au.verdict == v) & (au.gnn_detected == 1)).sum()),
                                      "median_prob": round(float(au[au.verdict == v].gnn_prob.median()), 3)} for v in au.verdict.unique()}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=int), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("labelled", "positives", "positives_2023", "positives_2013", "audit_agreement")}, indent=1, default=int))


if __name__ == "__main__":
    main()
