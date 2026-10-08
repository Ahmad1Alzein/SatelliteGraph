"""Attach the human audit verdicts (real / questionable / unclear) to this run's audit tables.

The verdicts were assigned once, by visual review of the high-confidence building changes of the original
2013-2023 run; a re-run regenerates the montages but cannot regenerate the human judgement. They are shipped in
reference/audit_verdicts.csv. The transfer is only valid if this run's change tables are byte-identical to the
audited ones: the recorded SHA-256 per tile is asserted, and every audited item must match on (segment_id, change).
The verdicts are used only for the GNN's audit-agreement report, never for training or labels.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

import paths

HERE = Path(__file__).resolve().parent
TILES = [f"tile_{i:05d}" for i in range(7)]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

V = pd.read_csv(paths.REFERENCE / "audit_verdicts.csv")
H = json.loads((paths.REFERENCE / "audited_change_tables_sha256.json").read_text())
prov = {"source": "reference/audit_verdicts.csv", "tiles": {}}
for t in TILES:
    d = HERE / "graphs" / f"change_stgraph_2013_2023_{t}"
    v = V[V.tile == t]
    if v.empty:
        prov["tiles"][t] = "not audited"
        continue
    got = sha(d / "nodes_change.csv")
    assert got == H[t], f"{t}: change table differs from the audited one ({got[:12]} != {H[t][:12]}): verdicts not transferable"
    f = d / "audit" / "A_high_confidence_building_changes.csv"
    new = pd.read_csv(f)
    cols = ["segment_id", "change", "verdict"] + (["note"] if v.note.notna().any() else [])   # tile_00000 was audited with notes
    m = new.merge(v[cols], on=["segment_id", "change"], how="left")
    assert len(m) == len(new) == len(v) and m.verdict.notna().all(), f"{t}: audited items do not match"
    m.to_csv(f, index=False)
    prov["tiles"][t] = {"items": len(m), "verdicts": m.verdict.value_counts().to_dict()}
(HERE / "graphs" / "audit_verdicts_provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
print(json.dumps(prov["tiles"], indent=1))
