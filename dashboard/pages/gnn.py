import json

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import data as D
from dashboard import ui

ui.setup()
st.title("5 · Spatio-temporal GNN")
tab_legacy, tab_mc = st.tabs(["Original thesis ST-GNN (preserved results)", "Selectable pipeline runs"])

def preserved_results():
    if not ui.need("legacy_run", "A verified legacy building ST-GNN run"):
        return
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

with tab_legacy:
    preserved_results()

with tab_mc:
    runs = D.multiclass_runs()
    st.markdown('Selectable runs use the same original binary ST-GNN implementation shown above. '
        'It scores later buildings and earlier land built over. Fits on new inputs are separate experiments; '
        'the frozen results above retain their original predictions and evaluation.')
    st.code('pipelines/legacy_building_stgnn/original/gnn_change.py',language=None)
    ui.badges('descriptive')
    if not runs:
        st.info('No pipeline runs found. Use Run pipelines & results to create a comparison.')
    else:
        rows = []
        for name in runs:
            g = D.run_gnn(name)
            current=bool(g.get('model_source'))
            rows.append({'run':D.run_label(name),'GNN status':g.get('status') if current or g.get('status')=='not_selected' else 'retired experimental run',
                'positive targets':g.get('positive_targets') if current else None,
                'negative targets':g.get('negative_targets') if current else None})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.page_link('dashboard/pages/pipelines.py',label='Run pipelines & inspect results')
