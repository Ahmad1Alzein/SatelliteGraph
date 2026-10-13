import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard import data as D
from dashboard import ui

ui.setup()
st.title("6 · Change detection")
tab_strip, tab_tiles, tab_mc = st.tabs(["Building change · 7-tile strip (legacy GNN)", "Building change · tile overlays (legacy GNN)",
                                        "Multi-class change events (current pipeline)"])

with tab_strip:
    if ui.need("legacy_run", "A verified legacy building ST-GNN run"):
        ui.badges("pseudo", note="Colours compare GNN predictions with rule pseudo-labels (not ground truth).")
        st.image(str(D.legacy_figure("fig5c_gnn_change_map.png")), width="stretch",
                 caption="Primary thesis figure (frozen original, pixel-verified). Top: 2013 land built over by 2023; "
                         "bottom: 2023 new / extended buildings. Blue = GNN and rules agree, orange = GNN only, green = rules only.")
        st.caption("Known limitation: the 2013 band colours whole land objects, although on average only 7.6 % of a "
                   "built-over land object actually changed (oversized land polygons). This is under investigation in a "
                   "separate experiment and is not corrected in this frozen figure.")
        st.download_button("Download fig5c (PNG)", D.legacy_figure("fig5c_gnn_change_map.png").read_bytes(),
                           file_name="fig5c_gnn_change_map.png", mime="image/png")

with tab_tiles:
    if ui.need("legacy_run", "A verified legacy building ST-GNN run"):
        tile = st.select_slider("Tile", options=D.TILES, value="tile_00002", key="ov_tile")
        d = D.legacy_tile_dir(tile)
        ref = {t: v["reference"] for t, v in json.loads((D.ROOT / "pipelines" / "legacy_building_stgnn" / "reference" /
                                                          "output_reference.json").read_text())["tile_figures"].items()}
        ui.badges("pseudo", note=f"Held-out tile: the model never saw its labels. Verified against: {ref.get(tile, '?')}.")
        view = st.radio("View", ["Prediction overlay", "Probability map", "Gallery 2023", "Gallery 2013"], horizontal=True, key="ov_view")
        f = {"Prediction overlay": "1_predictions_overlay.png", "Probability map": "2_probability_map.png",
             "Gallery 2023": "3_gallery_2023.png", "Gallery 2013": "3_gallery_2013.png"}[view]
        st.image(str(d / f), width="stretch")
        pn = pd.read_csv(d / "predicted_nodes.csv")
        with st.expander(f"Predicted changed nodes ({len(pn)})"):
            st.dataframe(pn, hide_index=True, width="stretch")
        c1, c2 = st.columns(2)
        c1.download_button("Download this image (PNG)", (d / f).read_bytes(), file_name=f"{tile}_{f}", mime="image/png")
        c2.download_button("Download predicted nodes (CSV)", pn.to_csv(index=False).encode(), file_name=f"{tile}_predicted_nodes.csv", mime="text/csv")
        P = D.legacy_predictions()
        s = P[(P.tile == tile) & (P.label >= 0)]
        tp, fp, fn = int(((s.label == 1) & (s.pred_gnn_full == 1)).sum()), int(((s.label == 0) & (s.pred_gnn_full == 1)).sum()), int(((s.label == 1) & (s.pred_gnn_full == 0)).sum())
        m = st.columns(4)
        with m[0]: ui.card(f"{len(s):,}", "labelled nodes in tile")
        with m[1]: ui.card(tp, "agree with rules (TP)", "pseudo")
        with m[2]: ui.card(fp, "GNN only (FP vs rules)", "pseudo")
        with m[3]: ui.card(fn, "rules only (FN vs rules)", "pseudo")

with tab_mc:
    runs = D.multiclass_runs()
    if not runs:
        st.info('No pipeline runs found. Use Run pipelines & results to create a comparison.')
    else:
        name = st.selectbox('Run', list(runs), format_func=D.run_label, key='mc_run')
        run = runs[name]
        state = D.run_state(name)
        saved = state.get('stages', {}).get('change', {})
        if saved.get('render_status') != 'complete':
            st.info('This run has no completed current change visualization. Open the pipeline workspace for its status.')
        else:
            from dashboard.run_results import change_view_status, show_archived_results
            try:
                if change_view_status(run, state) == 'archived':
                    show_archived_results(run, state)
                else:
                    st.image(str(run/'results/change_report.png'), width='stretch')
                    st.download_button('Download change report', (run/'results/change_report.png').read_bytes(),
                        file_name=name.replace('/', '_')+'_changes.png', mime='image/png')
                    events = D.run_events(name)
                    st.dataframe(events.drop(columns='geometry'), hide_index=True, width='stretch')
                    st.download_button('Download events (CSV)', events.drop(columns='geometry').to_csv(index=False),
                        file_name=name.replace('/', '_')+'_events.csv', mime='text/csv')
            except (ValueError, OSError) as exc:
                st.error(f'Results could not be verified: {exc}')
        if st.button('Select this run for local crops and stage images'):
            st.session_state.selected_run = str(run)
        st.page_link('dashboard/pages/pipelines.py', label='Open pipeline workspace: changes, segmentation, graphs and matching')
