import json

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import data as D
from dashboard import ui

ui.setup()
st.title("5 · Spatio-temporal GNN")
tab_legacy, tab_mc = st.tabs(["Legacy building ST-GNN (frozen baseline)", "Multi-class RGNN (current pipeline)"])

with tab_legacy:
    if not ui.need("legacy_run", "A verified legacy building ST-GNN run"):
        st.stop()
    rep = D.legacy_report()
    st.markdown("Relational GNN (R-GCN style, 2 layers, 64 hidden) on the 7-tile strip spatio-temporal graph. "
                "It predicts **2023 new / extended buildings** and **2013 land built over by 2023**. "
                "Each tile is predicted by a model trained on the other tiles (leave-one-tile-out).")
    ui.badges("pseudo", note="Targets are rule-based pseudo-labels; scores measure agreement with those rules.")
    v = rep["verification"]
    ok = v["status"] == "identical_to_frozen"
    (st.success if ok else st.error)(
        f"Reproduction check: {v['checked'] - len(v['failed'])} of {v['checked']} outputs identical to the frozen originals "
        f"({rep['mode']}, {rep['finished']}, torch {rep['environment'].get('torch', '?')}).")
    m = pd.DataFrame([{"model": {"mlp": "MLP (no edges)", "gnn_spatial": "GNN, spatial edges", "gnn_full": "GNN, spatial + temporal"}[k],
                       **{x: v_[x] for x in ("ap", "roc_auc", "precision", "recall", "f1", "tp", "fp", "fn")}}
                      for k, v_ in rep["metrics"].items()])
    l, r = st.columns([3, 2])
    with l:
        long = m.melt("model", ["ap", "f1", "roc_auc"], var_name="metric", value_name="value")
        f = px.bar(long, x="metric", y="value", color="model", barmode="group", text="value", height=360,
                   color_discrete_sequence=["#94a3b8", "#f59e0b", "#059669"], title="Held-out agreement with pseudo-labels")
        f.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        f.update_layout(yaxis_range=[0, 1.05], margin=dict(l=0, r=0, t=40, b=0), legend_title=None)
        st.plotly_chart(f, use_container_width=True)
    with r:
        st.dataframe(m, hide_index=True, width="stretch")
        by = pd.DataFrame(rep["metrics"]["gnn_full"]["by_year"]).T
        st.caption("Full GNN by target type")
        st.dataframe(by.rename(index={"2013": "2013 land built over", "2023": "2023 new / extended building"}), width="stretch")
    au = rep["audit_agreement"]
    st.markdown(ui.badge("visual") + f" Earlier visual audit of high-confidence building changes: the GNN detects "
                f"**{au['real']['detected_by_gnn']} of {au['real']['n']}** 'real', {au['questionable']['detected_by_gnn']} of "
                f"{au['questionable']['n']} 'questionable', {au['unclear']['detected_by_gnn']} of {au['unclear']['n']} 'unclear'.",
                unsafe_allow_html=True)
    st.subheader("Architecture")
    st.image(str(D.legacy_figure("fig5a_gnn_architecture.png")), width="stretch")
    st.subheader("Ablation")
    st.image(str(D.legacy_figure("fig5b_gnn_results.png")), width="stretch")
    st.caption(f"Labelled nodes {rep['labelled_nodes']:,}; positives {rep['positives']}. Figures are the frozen originals, "
               "regenerated and pixel-verified.")

with tab_mc:
    runs = D.multiclass_runs()
    st.markdown("The current pipeline extends the RGNN to **16 tasks** (gain / loss for 8 classes). A head produces scores "
                "only when it has enough supported examples; otherwise it reports `insufficient_data`.")
    ui.badges("descriptive", "external")
    if not runs:
        st.info("No multi-class pipeline runs found (SG_RUNS).")
    else:
        rows = []
        for name in runs:
            g = D.run_gnn(name)
            hs = g.get("head_status", {})
            st_ = pd.Series([h.get("status") if isinstance(h, dict) else h for h in hs.values()]).value_counts().to_dict()
            rows.append({"run": D.run_label(name), "GNN status": g.get("status"), "positive targets": g.get("positive_targets"),
                         "negative targets": g.get("negative_targets"), "checkpoint": "external" if g.get("checkpoint") and "lsscd" in str(g.get("checkpoint")) else ("trained in run" if g.get("checkpoint") else "none"),
                         "heads": ", ".join(f"{k}: {v}" for k, v in st_.items())})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        pick = st.selectbox("Head status for run", list(runs), format_func=D.run_label)
        hs = D.run_gnn(pick).get("head_status", {})
        st.dataframe(pd.DataFrame([{"task": k, **(v if isinstance(v, dict) else {"status": v})} for k, v in hs.items()]),
                     hide_index=True, width="stretch", height=420)
        st.warning("No multi-class head has a held-out evaluation on these Al Shouf tiles: almost all heads report "
                   "insufficient_data (a single image pair yields too few reliable rule targets), and the only trained head "
                   "(training_only, tile_00006) was trained and scored on the same image pair. The externally trained LsSCD-Ex "
                   "checkpoint is not present on this machine. No multi-class GNN scores are therefore presented as results.")
        st.markdown(ui.badge("external") + " Reported in `docs/all-class-change-results.md` (artifacts not on this machine, "
                    "not re-verified here): on the external LsSCD-Ex test split the full temporal-difference RGNN reached "
                    "macro AP 0.559 vs 0.304 without edges; 4 of 16 directions passed validation.", unsafe_allow_html=True)
