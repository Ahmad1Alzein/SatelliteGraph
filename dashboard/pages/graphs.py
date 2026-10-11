import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard import data as D
from dashboard import ui

ui.setup()
st.title("3 · Spatial graphs")
st.caption("Region adjacency graph per year: nodes are segmented objects (positioned at an interior point), edges join "
           "objects that touch (shared boundary) or are near each other (gap ≤ threshold).")
ui.badges("descriptive")
if not ui.need("bundle", "The frozen study bundle"):
    st.stop()

c = st.columns([2, 2, 3, 2])
with c[0]:
    tile = ui.tile_picker("g_tile")
year = c[1].radio("Year", ["2013", "2023"], horizontal=True, key="g_year")
kinds = c[2].multiselect("Edge types", ["touches", "near"], default=["touches", "near"])
size_by = c[3].selectbox("Node size", ["area", "degree"])
x0, x1 = D.tile_window(tile)
n = D.nodes(year)
e = D.edges(year)
tn = n[(n.px >= x0) & (n.px < x1)].set_index("segment_id")
te = e[e.source.isin(tn.index) & e.target.isin(tn.index) & e.edge_type.isin(kinds)]
deg = pd.concat([te.source, te.target]).value_counts()
tn["degree"] = deg.reindex(tn.index).fillna(0).astype(int)

fig = ui.image_figure((D.strip_image(year)[:, x0:x1] * 0.65).astype(np.uint8), height=760)
style = {"touches": ("rgba(255,255,255,0.75)", 1.2), "near": ("rgba(255,200,0,0.8)", 1.0)}
for k in kinds:
    sub = te[te.edge_type == k]
    xs, ys = [], []
    for s, t in zip(sub.source.values, sub.target.values):
        xs += [tn.at[s, "px"] - x0, tn.at[t, "px"] - x0, None]; ys += [tn.at[s, "py"], tn.at[t, "py"], None]
    fig.add_trace(go.Scattergl(x=xs, y=ys, mode="lines", line=dict(color=style[k][0], width=style[k][1]),
                               name=f"{k} ({len(sub):,})", hoverinfo="skip"))
for cl in sorted(tn["class"].unique()):
    sub = tn[tn["class"] == cl]
    size = np.clip(np.sqrt(sub.area_m2) / 3, 4, 22) if size_by == "area" else np.clip(3 + 1.5 * sub.degree, 4, 22)
    fig.add_trace(go.Scattergl(
        x=sub.px - x0, y=sub.py, mode="markers", name=D.LABELS[cl],
        marker=dict(color=D.hexcolor(cl), size=size, line=dict(color="white", width=0.8)),
        customdata=np.c_[sub.index, sub.area_m2.round(1), sub.degree, sub.confidence.round(3)],
        hovertemplate="object %{customdata[0]}<br>" + D.LABELS[cl] + "<br>%{customdata[1]} m²<br>degree %{customdata[2]}"
                      "<br>confidence %{customdata[3]}<extra></extra>"))
st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

m = st.columns(4)
with m[0]: ui.card(f"{len(tn):,}", "nodes in tile")
with m[1]: ui.card(f"{len(te):,}", "edges shown")
with m[2]: ui.card(f"{tn.degree.mean():.2f}", "mean degree")
with m[3]: ui.card(f"{(tn.degree == 0).sum():,}", "isolated nodes")
l, r = st.columns(2)
with l:
    h = px.histogram(tn, x="degree", nbins=30, height=300, title="Degree distribution", color_discrete_sequence=["#2563eb"])
    h.update_layout(margin=dict(l=0, r=0, t=40, b=0)); st.plotly_chart(h, use_container_width=True)
with r:
    pairs = te.assign(pair=[" – ".join(sorted([a, b])) for a, b in zip(te.class_source, te.class_target)]).pair.value_counts().head(12)
    b = px.bar(x=pairs.values, y=pairs.index, orientation="h", height=300, title="Most frequent class pairs on edges",
               color_discrete_sequence=["#0f766e"])
    b.update_layout(margin=dict(l=0, r=0, t=40, b=0), yaxis_title=None, xaxis_title="edges"); st.plotly_chart(b, use_container_width=True)
st.download_button("Download tile edges (CSV)", te.to_csv(index=False).encode(), file_name=f"{tile}_{year}_edges.csv", mime="text/csv")
