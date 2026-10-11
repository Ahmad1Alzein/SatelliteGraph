import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import data as D
from dashboard import ui

ui.setup()
st.title("7 · Evaluation")
st.markdown("Results are grouped by **what they can prove**. Only an independent human reference can measure accuracy "
            "on Al Shouf; that reference does not exist yet.")

st.subheader("Evidence hierarchy")
st.dataframe(pd.DataFrame([
    [ui.EVIDENCE["none"][0], "Accuracy of building and multi-class change on Al Shouf", "not available: annotation kit prepared, not yet annotated"],
    [ui.EVIDENCE["osm"][0], "Building segmentation quality (2023 instances vs OpenStreetMap)", "tiles 1 / 3 / 6, see below"],
    [ui.EVIDENCE["external"][0], "Whether graph edges help change ranking on an annotated dataset", "documented LsSCD-Ex results (not re-verified here)"],
    [ui.EVIDENCE["visual"][0], "Plausibility of high-confidence building changes", "legacy audit, 43 'real' changes"],
    [ui.EVIDENCE["pseudo"][0], "How well the GNN reproduces the rule layer on held-out tiles; edge ablations", "legacy building ST-GNN"],
    [ui.EVIDENCE["descriptive"][0], "Counts and areas of objects, links and change events", "all workflow pages"],
], columns=["Evidence type", "What it can show", "Available here"]), hide_index=True, width="stretch")

st.subheader("Legacy building ST-GNN")
if ui.need("legacy_run", "A verified legacy building ST-GNN run"):
    ui.badges("pseudo")
    rep = D.legacy_report()
    P = D.legacy_predictions()
    rows = []
    for t in D.TILES:
        s = P[(P.tile == t) & (P.label >= 0)]
        tp = int(((s.label == 1) & (s.pred_gnn_full == 1)).sum()); fp = int(((s.label == 0) & (s.pred_gnn_full == 1)).sum())
        fn = int(((s.label == 1) & (s.pred_gnn_full == 0)).sum())
        rows.append({"tile": t, "labelled": len(s), "positives": int((s.label == 1).sum()), "TP": tp, "FP": fp, "FN": fn,
                     "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3)})
    T = pd.DataFrame(rows)
    l, r = st.columns([3, 2])
    with l:
        f = px.bar(T.melt("tile", ["precision", "recall"]), x="tile", y="value", color="variable", barmode="group", height=330,
                   color_discrete_sequence=["#2563eb", "#f59e0b"], title="Per held-out tile (each tile predicted by a model trained on the others)")
        f.update_layout(yaxis_range=[0, 1], margin=dict(l=0, r=0, t=40, b=0), legend_title=None); st.plotly_chart(f, use_container_width=True)
    with r:
        st.dataframe(T, hide_index=True, width="stretch")
    st.caption(f"Pooled over the 7 test tiles: precision {rep['metrics']['gnn_full']['precision']}, recall "
               f"{rep['metrics']['gnn_full']['recall']}, F1 {rep['metrics']['gnn_full']['f1']}, AP {rep['metrics']['gnn_full']['ap']}. "
               "Positives per tile are few (see the table), so per-tile values vary strongly.")
    st.markdown(ui.badge("visual"), unsafe_allow_html=True)
    au = D.legacy_audit()
    st.dataframe(au.groupby("verdict").agg(n=("verdict", "size"), detected_by_gnn=("gnn_detected", lambda x: int((x == 1).sum())),
                                           median_probability=("gnn_prob", "median")).round(3), width="stretch")

st.subheader("Building segmentation vs OpenStreetMap 2023")
ext = D.external_eval() if D.available()["external_eval"] else {}
if "buildings" in ext:
    ui.badges("osm", note="OSM is partial and 2023-only; an independent but incomplete reference for building instances.")
    b = ext["buildings"].rename(columns={"method": "segmentation"})
    b["segmentation"] = b["segmentation"].replace({"improved": "improved (native 0.5 m buildings, OriginalPipeline_Improved)"})
    st.dataframe(b, hide_index=True, width="stretch")
    st.caption(f"Source: {D.paths()['external_eval']} (instance matching at IoU ≥ 0.5, tiles 1, 3, 6).")
else:
    st.info("OSM comparison tables not found (SG_EXTERNAL_EVAL).")

st.subheader("Still unmeasured")
st.markdown(ui.badge("none"), unsafe_allow_html=True)
st.markdown("- Accuracy of building change and of multi-class change on Al Shouf (needs independent human annotation).\n"
            "- Recall of real changes that the rule layer itself misses (pseudo-labels cannot reveal them).\n"
            "- Multi-class GNN performance on Al Shouf (no head has a held-out evaluation on the available runs).")
