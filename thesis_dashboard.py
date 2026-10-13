"""MSc thesis dashboard: multi-class object-level spatio-temporal change detection, Al Shouf 2013 -> 2023.

Run:  python -m streamlit run thesis_dashboard.py
The selectable pipeline workspace is part of this dashboard.
"""
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(page_title="SatelliteGraph · MSc thesis dashboard", page_icon="🛰️", layout="wide")

P = "dashboard/pages/"
pages = {
    "Thesis": [
        st.Page(P + "overview.py", title="Overview", icon="🏠", default=True),
    ],
    "Workflow": [
        st.Page(P + "imagery.py", title="1 · Satellite images", icon="🛰️"),
        st.Page(P + "segmentation.py", title="2 · Multi-class segmentation", icon="🗺️"),
        st.Page(P + "graphs.py", title="3 · Spatial graphs", icon="🕸️"),
        st.Page(P + "temporal.py", title="4 · Temporal matching", icon="🔗"),
        st.Page(P + "gnn.py", title="5 · Spatio-temporal GNN", icon="🧠"),
        st.Page(P + "changes.py", title="6 · Change detection", icon="🔍"),
        st.Page(P + "evaluation.py", title="7 · Evaluation", icon="📏"),
    ],
    "Tools": [
        st.Page(P + "pipelines.py", title="Run pipelines & results", icon="▶️", url_path="image_comparison"),
        st.Page(P + "reproduce.py", title="Reproduce & data", icon="🧾"),
    ],
}
nav = st.navigation(pages)
with st.sidebar:
    st.caption("Al Shouf, Lebanon · 2013 → 2023 · 7 tiles (4.9 km × 0.77 km strip)")
    st.caption("Every metric is labelled with its evidence type. Pseudo-label agreement is not accuracy.")
nav.run()
