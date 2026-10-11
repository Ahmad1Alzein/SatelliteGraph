# MSc thesis dashboard

```powershell
python -m pip install -r requirements.txt          # streamlit and plotly
python -m streamlit run thesis_dashboard.py
```

The dashboard is a multipage Streamlit app in front of the existing results. It reads the files and recomputes no
results. The original image-comparison tool (`streamlit_app.py`) is unchanged: it is mounted as the page
*Run a new image pair* and can still be run on its own.

## Pages
| Page | Content | Source |
|---|---|---|
| Overview | workflow, study-area strip, headline results with evidence badges | all |
| 1 · Satellite images | 2013 / 2023 per tile with swipe, blink or side-by-side comparison; tile metadata; PNG export | frozen study bundle |
| 2 · Multi-class segmentation | 8-class object overlay (opacity, class filter, boundaries); class areas per year; object explorer (select a row to outline the object); CSV export | bundle |
| 3 · Spatial graphs | interactive adjacency graph over the image (touches / near; hover attributes); degree and class-pair statistics; CSV export | bundle |
| 4 · Temporal matching | link counts, match-type components, class-to-class shared area; object lineage view (2013 object → linked 2023 objects); CSV export | bundle |
| 5 · Spatio-temporal GNN | legacy building ST-GNN: verification status, metrics and ablation, audit agreement, frozen fig5a / fig5b; multi-class RGNN head status per run | legacy run, pipeline runs |
| 6 · Change detection | fig5c strip map; the 7 per-tile overlays (overlay, probability map, galleries, predicted nodes); multi-class change events as an interactive map with filters, charts, CSV and GeoJSON export | legacy run, pipeline runs |
| 7 · Evaluation | evidence hierarchy; per-tile legacy GNN agreement; visual audit; building instances vs OSM 2023; what is still unmeasured | legacy run, OSM comparison |
| Run a new image pair | the original tool (upload, run pipeline, review crops, reviewed labels) | `streamlit_app.py` |
| Reproduce & data | data sources found, commands, git commit | — |

## Evidence badges
Every metric carries one badge:

| Badge | Meaning |
|---|---|
| Pseudo-label agreement | agreement with our rule labels |
| Visual audit (not independent) | visual judgements made during development |
| OSM 2023 reference (partial) | OpenStreetMap buildings |
| External annotated benchmark | LsSCD-Ex, reported in the docs |
| Descriptive (no accuracy claim) | counts and areas |
| Independent ground truth: not yet available | no independent reference exists yet |

Pseudo-label scores are never presented as accuracy.

## Data
Paths can be overridden with environment variables:

| Variable | Default | Content |
|---|---|---|
| `SG_LEGACY_BUNDLE` | `data/legacy_building_stgnn` | frozen study: strip imagery, segmentations, graphs, temporal links (`python -m src legacy-building-gnn bundle …`) |
| `SG_LEGACY_RUN` | newest `artifacts/legacy_building_stgnn/*/report.json` | verified legacy GNN run (`… legacy-building-gnn run …`) |
| `SG_RUNS` | `artifacts/shouf3` | multi-class pipeline runs (`python -m src pipeline run …`) |
| `SG_EXTERNAL_EVAL` | `data/external_eval`, falling back to `D:\new\OriginalPipeline_Improved\improve\out\compare_repo` | optional OSM building comparison tables |
| `SATELLITEGRAPH_RUN` | — | run folder preselected in the original tool |

Missing sources produce a notice on the affected page, not an error.

## Tests
```powershell
python scripts/run_project_tests.py tests/test_thesis_dashboard.py tests/test_preview_browser.py
```
- **Dashboard tests:** every page renders without exceptions on the real artifacts (skipped when the bundle is
  absent). They also check that evidence badges are present, that object selection works, and that the original
  tool stays mounted.
- **Original tool:** `test_preview_browser.py` covers the original tool.
