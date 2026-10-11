import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import data as D
from dashboard import ui

ui.setup()
st.title("2 · Multi-class segmentation")
st.caption("Frozen object segmentation of the study: SegFormer (OpenEarthMap classes) + building instances + "
           "Delineate-Anything parcels + rules + merge. Each object is one graph node.")
ui.badges("descriptive", note="Classes are model outputs; they have not been checked against ground truth.")
if not ui.need("bundle", "The frozen study bundle"):
    st.stop()

c = st.columns([2, 2, 3, 3])
with c[0]:
    tile = ui.tile_picker("seg_tile")
year = c[1].radio("Year", ["2013", "2023"], horizontal=True, key="seg_year")
alpha = c[2].slider("Overlay opacity", 0.0, 1.0, 0.55, 0.05)
show_b = c[3].checkbox("Object boundaries", value=True)
keep = st.multiselect("Classes shown", list(D.CLASSES), default=list(D.CLASSES), format_func=lambda k: D.LABELS[D.CLASSES[k]])
x0, x1 = D.tile_window(tile)
rgb = D.strip_image(year)[:, x0:x1]
cls = D.class_raster(year)[:, x0:x1]
seg = D.strip_segments(year)[:, x0:x1]
nodes = D.nodes(year)
tn = nodes[(nodes.px >= x0) & (nodes.px < x1)].copy()

left, right = st.columns([3, 2])
with left:
    view = ui.overlay(rgb, cls, alpha, keep=keep, boundaries=seg if show_b else None)
    sel = st.session_state.get("seg_obj")
    if sel is not None and sel in set(tn.segment_id):
        view = ui.outline(view, seg == sel, color=(255, 230, 0), width=3)
    st.image(view, width="stretch", caption=f"{tile} · {year} · {len(tn):,} objects")
    ui.class_legend()
with right:
    area = pd.DataFrame({"class": [D.LABELS[D.CLASSES[k]] for k in D.CLASSES]})
    for y in ("2013", "2023"):
        cr = D.class_raster(y)[:, x0:x1]
        area[y] = [round(float((cr == k).sum()) * D.PIXEL_M ** 2 / 1e4, 2) for k in D.CLASSES]
    long = area.melt("class", var_name="year", value_name="ha")
    fig = px.bar(long, x="ha", y="class", color="year", barmode="group", orientation="h",
                 color_discrete_map={"2013": "#64748b", "2023": "#2563eb"}, height=380, title="Class area in this tile (ha)")
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), yaxis_title=None, legend_title=None)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Area differences mix real change with class inconsistency between the two segmentations "
               "(about half of the area changes class between years in the frozen study).")

st.subheader("Object explorer")
f = st.columns([3, 2, 2])
cls_f = f[0].multiselect("Class", sorted(tn["class"].unique()), default=sorted(tn["class"].unique()), key="seg_cls")
min_a = f[1].number_input("Minimum area (m²)", 0.0, value=0.0, step=50.0)
sort = f[2].selectbox("Sort by", ["area_m2", "confidence", "compactness"])
cols = ["segment_id", "class", "area_m2", "confidence", "compactness", "solidity", "elongation", "source", "source_tiles"]
tab = tn[tn["class"].isin(cls_f) & (tn.area_m2 >= min_a)][cols].sort_values(sort, ascending=False)
ev = st.dataframe(tab, hide_index=True, width="stretch", height=320, on_select="rerun", selection_mode="single-row", key="seg_table")
rows = ev.selection.rows if ev and ev.selection else []
if rows:
    chosen = int(tab.iloc[rows[0]].segment_id)
    if st.session_state.get("seg_obj") != chosen:
        st.session_state.seg_obj = chosen
        st.rerun()
if st.session_state.get("seg_obj") is not None:
    st.caption(f"Selected object {st.session_state.seg_obj} is outlined in yellow on the map. Click another row to change it.")
else:
    st.caption("Select a row to outline that object on the map.")
st.download_button("Download object table (CSV)", tab.to_csv(index=False).encode(), file_name=f"{tile}_{year}_objects.csv", mime="text/csv")
