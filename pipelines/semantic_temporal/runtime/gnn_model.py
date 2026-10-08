"""Study RGNN architecture with an explicit edge feature dimension."""
import torch
from torch import nn
from torch.nn import functional as F

class RGNN(nn.Module):
    def __init__(self, din, rels, hidden=64, layers=2, drop=0.3, edge_dim=9):
        super().__init__()
        self.rels = list(rels); self.drop = drop
        dims = [din] + [hidden] * layers
        self.self_w = nn.ModuleList([nn.Linear(dims[k], dims[k + 1]) for k in range(layers)])
        self.rel_w = nn.ModuleList([nn.ModuleDict({r: nn.Linear(dims[k], dims[k + 1], bias=False) for r in self.rels}) for k in range(layers)])
        self.edge_w = nn.ModuleList([nn.ModuleDict({r: nn.Linear(edge_dim, dims[k + 1], bias=False) for r in self.rels}) for k in range(layers)])
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
