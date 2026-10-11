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
    if not runs or not D.available()["bundle"]:
        st.info("Multi-class runs or the frozen study bundle are not available.")
    else:
        st.markdown("Change **events** of the current all-class pipeline: localised polygons where the two years' classes "
                    "differ and image evidence supports a change. `season_sensitive` events involve classes whose appearance "
                    "changes with season (spring 2013 vs summer 2023).")
        ui.badges("descriptive", "none", note="Events are review candidates, not confirmed changes.")
        c = st.columns([3, 2, 2, 2])
        name = c[0].selectbox("Run", list(runs), format_func=D.run_label, key="mc_run")
        E = D.run_events(name)
        status = c[1].multiselect("Status", sorted(E.status.unique()), default=["candidate"] if "candidate" in set(E.status) else sorted(E.status.unique()))
        min_a = c[2].number_input("Minimum area (m²)", 0.0, value=25.0, step=25.0)
        color_by = c[3].selectbox("Colour by", ["target class", "source class", "status"])
        trans = st.multiselect("Transitions", sorted(E.transition.unique()), default=[], placeholder="All transitions")
        F = E[E.status.isin(status) & (E.area_m2 >= min_a)]
        if trans:
            F = F[F.transition.isin(trans)]
        tile = D.run_tile(name)
        x0, x1 = D.tile_window(tile)
        base = st.radio("Background", ["2023", "2013"], horizontal=True, key="mc_bg")
        fig = ui.image_figure((D.strip_image(base)[:, x0:x1] * 0.7).astype("uint8"), height=760)
        key = {"target class": "target_class", "source class": "source_class", "status": "status"}[color_by]
        palette = {"candidate": "#ef4444", "season_sensitive": "#22c55e"}
        for val, grp in F.groupby(key):
            color = palette.get(val, "#a3a3a3") if key == "status" else D.hexcolor(val)
            xs, ys, cd = [], [], []
            for r in grp.itertuples():
                for ring in D.geometry_to_pixels(r.geometry, x0):
                    xs += list(ring[:, 0]) + [None]; ys += list(ring[:, 1]) + [None]
                    cd += [[r.transition, round(r.area_m2, 1), r.status, r.related_tasks]] * len(ring) + [[None] * 4]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself", name=f"{D.LABELS.get(val, val)} ({len(grp)})",
                                     line=dict(color=color, width=2.5), fillcolor=color, opacity=0.85, customdata=cd,
                                     hovertemplate="%{customdata[0]}<br>%{customdata[1]} m²<br>%{customdata[2]}<br>%{customdata[3]}<extra></extra>"))
        st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})
        m = st.columns(4)
        with m[0]: ui.card(f"{len(F):,}", "events shown", "descriptive")
        with m[1]: ui.card(f"{F.area_m2.sum() / 1e4:.2f} ha", "event area", "descriptive")
        with m[2]: ui.card(f"{(E.status == 'season_sensitive').mean() * 100:.0f}%", "of all events season-sensitive", "descriptive")
        with m[3]: ui.card(f"{len(E):,}", "events in run", "descriptive")
        l, r = st.columns(2)
        with l:
            t = F.groupby("transition").area_m2.sum().sort_values().tail(15)
            b = px.bar(x=t.values, y=t.index, orientation="h", height=420, title="Event area by transition (m²)", color_discrete_sequence=["#dc2626"])
            b.update_layout(margin=dict(l=0, r=0, t=40, b=0), yaxis_title=None, xaxis_title=None); st.plotly_chart(b, use_container_width=True)
        with r:
            g = pd.concat([F.groupby("target_class").area_m2.sum().rename("gain"), -F.groupby("source_class").area_m2.sum().rename("loss")], axis=1).fillna(0)
            g.index = [D.LABELS.get(i, i) for i in g.index]
            gl = g.reset_index().melt("index", var_name="direction", value_name="m2")
            b2 = px.bar(gl, x="m2", y="index", color="direction", orientation="h", height=420, barmode="relative",
                        color_discrete_map={"gain": "#16a34a", "loss": "#b91c1c"}, title="Gross class gain / loss in shown events (m²)")
            b2.update_layout(margin=dict(l=0, r=0, t=40, b=0), yaxis_title=None, xaxis_title=None, legend_title=None); st.plotly_chart(b2, use_container_width=True)
        table = F.drop(columns=["geometry"])
        st.dataframe(table.sort_values("area_m2", ascending=False), hide_index=True, width="stretch", height=300)
        gj = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:32636"}},
              "features": [{"type": "Feature", "geometry": r.geometry, "properties": {k: getattr(r, k) for k in table.columns}} for r in F.itertuples()]}
        d1, d2 = st.columns(2)
        d1.download_button("Download events (CSV)", table.to_csv(index=False).encode(), file_name=f"{name}_events.csv", mime="text/csv")
        d2.download_button("Download events (GeoJSON, EPSG:32636)", json.dumps(gj, default=str).encode(), file_name=f"{name}_events.geojson", mime="application/geo+json")
        st.page_link("streamlit_app.py", label="Inspect local before/after crops of these events in the original review tool →")
