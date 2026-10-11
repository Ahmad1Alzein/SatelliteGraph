import cv2
import numpy as np
import pandas as pd
import streamlit as st

from dashboard import data as D
from dashboard import ui

ui.setup()
st.title("Multi-class object-level spatio-temporal change detection")
st.markdown("**Al Shouf, Lebanon · 2013 → 2023** · MSc thesis dashboard for the SatelliteGraph pipeline")
ui.flow(["Satellite images", "Multi-class segmentation", "Spatial graphs", "Temporal matching", "Spatio-temporal GNN",
         "Change detection", "Evaluation"])
cols = st.columns(7)
links = ["imagery", "segmentation", "graphs", "temporal", "gnn", "changes", "evaluation"]
names = ["Images", "Segmentation", "Graphs", "Matching", "GNN", "Changes", "Evaluation"]
for c, l, n in zip(cols, links, names):
    c.page_link(f"dashboard/pages/{l}.py", label=n)

av = D.available()
if not av["bundle"]:
    st.error("The frozen study bundle (data/legacy_building_stgnn) is missing. Create it with "
             "`python -m src legacy-building-gnn bundle …` (see Reproduce).")
    st.stop()

st.subheader("Study area")
t = D.tiles()
img = D.strip_image("2023")
small = cv2.resize(img, (img.shape[1] // 3, img.shape[0] // 3), interpolation=cv2.INTER_AREA)
for r in t.itertuples():
    x = r.col0 // 3
    cv2.line(small, (x, 0), (x, small.shape[0]), (255, 255, 255), 1)
    cv2.putText(small, r.tile.replace("tile_0000", "T"), (x + 6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
st.image(small, caption="2023 imagery of the 7-tile strip (frozen study, 0.75 m grid resampled from 1.5 m; tile boundaries in white; "
                        "tiles overlap by 50 source pixels).", width="stretch")

v = D.strip_validation()
s13, s23 = D.strip_summary("2013"), D.strip_summary("2023")
c = st.columns(5)
with c[0]: ui.card("7", "tiles · 4.9 × 0.77 km")
with c[1]: ui.card(f"{s13['segments']:,}", "2013 objects (segments)", "descriptive")
with c[2]: ui.card(f"{s23['segments']:,}", "2023 objects (segments)", "descriptive")
with c[3]: ui.card(f"{len(D.temporal_edges()):,}", "temporal edges 2013 ↔ 2023", "descriptive")
with c[4]: ui.card(f"{s13['edges'] + s23['edges']:,}", "spatial edges (both years)", "descriptive")

st.subheader("Headline results and their evidence")
rows = []
if av["legacy_run"]:
    rep = D.legacy_report()
    m = rep["metrics"]["gnn_full"]
    rows.append({"Result": "Legacy building ST-GNN, held-out (leave-one-tile-out)", "Value": f"AP {m['ap']:.2f} · F1 {m['f1']:.3f} · ROC-AUC {m['roc_auc']:.3f}",
                 "Evidence": "pseudo"})
    rows.append({"Result": "Ablation: same GNN without temporal edges / without edges",
                 "Value": f"AP {rep['metrics']['gnn_spatial']['ap']:.2f} / {rep['metrics']['mlp']['ap']:.3f}", "Evidence": "pseudo"})
    au = rep["audit_agreement"]["real"]
    rows.append({"Result": "Audited 'real' building changes detected by the GNN", "Value": f"{au['detected_by_gnn']} / {au['n']}", "Evidence": "visual"})
    rows.append({"Result": "Reproduction of the frozen GNN (data + figures)",
                 "Value": f"{rep['verification']['checked'] - len(rep['verification']['failed'])} / {rep['verification']['checked']} identical", "Evidence": "descriptive"})
ext = D.external_eval() if av["external_eval"] else {}
if "buildings" in ext:
    b = ext["buildings"].set_index("method")
    rows.append({"Result": "Building instances vs OSM 2023, tiles 1/3/6 (F1 @ IoU 0.5), current pipeline segmentation",
                 "Value": f"{b.loc['repo new segmentation (SegFormer + DA v2 runtime)', 'f1']:.3f}", "Evidence": "osm"})
rows.append({"Result": "Accuracy of multi-class change on Al Shouf", "Value": "unmeasured", "Evidence": "none"})
for r in rows:
    a, b_, e = st.columns([5, 3, 3])
    a.markdown(r["Result"]); b_.markdown(f"**{r['Value']}**"); e.markdown(ui.badge(r["Evidence"]), unsafe_allow_html=True)

with st.expander("How to read the evidence badges", expanded=False):
    for k, (label, fg, bg, tip) in ui.EVIDENCE.items():
        st.markdown(f"{ui.badge(k)} {tip}", unsafe_allow_html=True)

st.subheader("What each workflow page shows")
st.dataframe(pd.DataFrame([
    ["1 · Satellite images", "2013 / 2023 imagery per tile with swipe and blink comparison", "frozen study bundle"],
    ["2 · Multi-class segmentation", "8-class object maps, class areas, object explorer with attributes", "frozen study bundle"],
    ["3 · Spatial graphs", "interactive region adjacency graphs (touches / near) per tile and year", "frozen study bundle"],
    ["4 · Temporal matching", "2013 ↔ 2023 object links, split / merge / n:m components, per-object lineage", "frozen study bundle"],
    ["5 · Spatio-temporal GNN", "legacy building ST-GNN (verified reproduction) and the multi-class RGNN head status", "legacy run + pipeline runs"],
    ["6 · Change detection", "legacy strip change map, 7 tile overlays, multi-class change events with filters and export", "legacy run + pipeline runs"],
    ["7 · Evaluation", "every metric grouped by evidence type, including what is still unmeasured", "all of the above"],
], columns=["Page", "Content", "Source"]), hide_index=True, width="stretch")
