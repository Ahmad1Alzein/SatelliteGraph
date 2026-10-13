import subprocess

import pandas as pd
import streamlit as st

from dashboard import data as D
from dashboard import ui

ui.setup()
st.title("Reproduce & data")
try:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=D.ROOT, capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=D.ROOT, capture_output=True, text=True).stdout.strip()
except OSError:
    commit = branch = "unknown"
st.caption(f"Repository {D.ROOT} · branch {branch} · commit {commit}")

p, av = D.paths(), D.available()
st.subheader("Data sources used by this dashboard")
st.dataframe(pd.DataFrame([
    ["Frozen study bundle", str(p["bundle"]), av["bundle"], "SG_LEGACY_BUNDLE"],
    ["Legacy building ST-GNN run", str(p["legacy_run"]), av["legacy_run"], "SG_LEGACY_RUN"],
    ["Multi-class pipeline runs", str(p["runs"]), av["runs"], "SG_RUNS"],
    ["OSM comparison tables (optional)", str(p["external_eval"]), av["external_eval"], "SG_EXTERNAL_EVAL"],
], columns=["Source", "Path", "Found", "Override"]), hide_index=True, width="stretch")

st.subheader("Commands")
st.code("""# dashboard
python -m streamlit run thesis_dashboard.py

# legacy building ST-GNN: inputs once, full reproduction, verification
python -m src legacy-building-gnn bundle --source D:\\new\\DelAnyV2_Experiment\\segmentation_v2 --metadata D:\\new\\DelAnyV2_Experiment\\dataset\\2023\\metadata.csv
python -m src legacy-building-gnn run --output artifacts/legacy_building_stgnn/<run>
python -m src legacy-building-gnn verify --output artifacts/legacy_building_stgnn/<run>

# selectable pipeline; each method can be changed independently
python -m src pipeline run --segmentation new --graph new --change gnn --images <before.tif> <after.tif> --years 2013 2023 --device cpu --max-report-pixels 256000000 --output artifacts/image_comparisons/<run>""",
        language="powershell")
st.subheader("Documentation")
for doc in ["docs/pipeline-options.md", "docs/legacy-building-stgnn.md", "docs/thesis-dashboard.md", "docs/pipeline-validation.md", "docs/research-sources-for-thesis.md"]:
    st.markdown(f"- `{doc}`" + ("" if (D.ROOT / doc).is_file() else " (missing)"))
