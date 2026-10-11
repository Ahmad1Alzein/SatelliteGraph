import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import data as D
from dashboard import ui

ui.setup()
st.title("4 · Temporal matching")
st.caption("2013 objects are linked to 2023 objects by overlap (buildings in the roof frame, other classes in the ground "
           "frame). Primary links (identity, contained-in) form components: one-to-one, split, merge or many-to-many.")
ui.badges("descriptive", note="Links describe geometric correspondence between two segmentations, not verified identity.")
if not ui.need("bundle", "The frozen study bundle"):
    st.stop()

T = D.temporal_edges()
C = D.temporal_components()
n13, n23 = D.nodes("2013").set_index("segment_id"), D.nodes("2023").set_index("segment_id")
T = T.assign(tile=n13.tile.reindex(T.node_2013).values)

c = st.columns([2, 5])
with c[0]:
    tile = ui.tile_picker("t_tile")
tT = T[T.tile == tile]
with c[1]:
    m = st.columns(4)
    with m[0]: ui.card(f"{len(tT):,}", "temporal edges in tile")
    with m[1]: ui.card(f"{(tT.edge_role == 'primary').sum():,}", "primary links")
    with m[2]: ui.card(f"{tT.class_agreement.mean() * 100:.0f}%", "of all links (incl. secondary) keep the class")
    with m[3]: ui.card(f"{(tT.frame == 'roof').sum():,}", "links in the roof frame")

l, r = st.columns(2)
with l:
    mt = tT[tT.edge_role == "primary"].drop_duplicates("component_id").match_type.value_counts()
    f = px.bar(x=mt.index, y=mt.values, height=320, title="Primary link components by match type", color_discrete_sequence=["#2563eb"])
    f.update_layout(margin=dict(l=0, r=0, t=40, b=0), xaxis_title=None, yaxis_title="components"); st.plotly_chart(f, use_container_width=True)
with r:
    ct = pd.crosstab(tT.class_2013, tT.class_2023, values=tT.shared_m2, aggfunc="sum").fillna(0) / 1e4
    h = px.imshow(ct.round(2), text_auto=True, aspect="auto", height=320, color_continuous_scale="Blues",
                  title="Shared area of linked objects, 2013 class → 2023 class (ha)")
    h.update_layout(margin=dict(l=0, r=0, t=40, b=0)); st.plotly_chart(h, use_container_width=True)

st.subheader("Object lineage")
x0, x1 = D.tile_window(tile)
cands = n13[(n13.tile == tile)].sort_values("area_m2", ascending=False)
f = st.columns([2, 3])
cls = f[0].selectbox("2013 class", sorted(cands["class"].unique()), index=sorted(cands["class"].unique()).index("building"))
pick = f[1].selectbox("2013 object", cands[cands["class"] == cls].index[:400],
                      format_func=lambda i: f"{i} · {cands.at[i, 'area_m2']:.0f} m²")
links = tT[tT.node_2013 == pick].sort_values("shared_m2", ascending=False)
seg13, seg23 = D.strip_segments("2013")[:, x0:x1], D.strip_segments("2023")[:, x0:x1]
a = ui.outline(D.strip_image("2013")[:, x0:x1], seg13 == pick, (255, 220, 0), 3)
b = D.strip_image("2023")[:, x0:x1]
for row in links.itertuples():
    col = (0, 230, 255) if row.edge_role == "primary" else (255, 120, 0)
    b = ui.outline(b, seg23 == row.node_2023, col, 3 if row.edge_role == "primary" else 2)
ys, xs = np.nonzero(seg13 == pick)
if len(ys):
    cy, cx = int(ys.mean()), int(xs.mean()); hw = max(110, int(1.2 * max(np.ptp(ys), np.ptp(xs))))
    y0, y1, xa, xb = max(0, cy - hw), min(1024, cy + hw), max(0, cx - hw), min(x1 - x0, cx + hw)
    a, b = a[y0:y1, xa:xb], b[y0:y1, xa:xb]
p, q = st.columns(2)
p.image(a, caption=f"2013 · object {pick} ({cls})", width="stretch")
q.image(b, caption="2023 · linked objects (cyan = primary link, orange = secondary overlap)", width="stretch")
if links.empty:
    st.info("This 2013 object has no temporal link: no 2023 object overlaps it enough (unmatched).")
else:
    st.dataframe(links[["node_2023", "class_2023", "link_type", "edge_role", "match_type", "iou", "frac_2013", "frac_2023",
                        "shared_m2", "frame", "roof_shift_dx_m", "roof_shift_dy_m", "confidence"]], hide_index=True, width="stretch")
st.download_button("Download tile temporal edges (CSV)", tT.to_csv(index=False).encode(), file_name=f"{tile}_temporal_edges.csv", mime="text/csv")
