# MSc thesis dashboard

```powershell
python -m pip install -r requirements.txt          # streamlit and plotly
python -m streamlit run thesis_dashboard.py
```

The dashboard preserves the original thesis result pages and includes **Run pipelines & results**
(`dashboard/pages/pipelines.py`). Launch `thesis_dashboard.py` and choose each stage independently:

- Segmentation: SAM 2.1 + OpenEarthMap (old), or SegFormer + Delineate Anything (new).
- Spatial graph: adjacency only (old), or adjacency plus nearby relations (new).
- Change: original Stage 5 rules (old), or the original thesis building ST-GNN.

All eight combinations share the segmentation, graph, matching and paired-crop result viewer.
Upload two images, enter local image paths, or supply two tile folders with placement metadata.
Set chronological years and the processing device; known physical pixel size and preprocessing
options are optional. Each launch creates a new run under `artifacts/image_comparisons/`.
Completed runs are listed on Change detection and Spatio-temporal GNN, and can be opened in
the workspace by folder path. Rule-based runs need no GNN artifact, and empty results are supported.

The retained model is project code in `pipelines/legacy_building_stgnn/original/gnn_change.py`
(class `RGNN`), integrated by contributor Roaa Elmir. It is not downloaded pretrained weights.
It learns binary building/built-over targets; it does not save a weights checkpoint. The former
experimental multi-class GNN and its training controls were removed. New-input fitting does not
replace or inherit the evaluation of the preserved seven-tile experiment.

## Pages
| Page | Content | Source |
|---|---|---|
| Overview | workflow, study-area strip, headline results with evidence badges | all |
| 1 · Satellite images | 2013 / 2023 per tile with swipe, blink or side-by-side comparison; tile metadata; PNG export | frozen study bundle |
| 2 · Multi-class segmentation | 8-class object overlay (opacity, class filter, boundaries); class areas per year; object explorer (select a row to outline the object); CSV export | bundle |
| 3 · Spatial graphs | interactive adjacency graph over the image (touches / near; hover attributes); degree and class-pair statistics; CSV export | bundle |
| 4 · Temporal matching | link counts, match-type components, class-to-class shared area; object lineage view (2013 object → linked 2023 objects); CSV export | bundle |
| 5 · Spatio-temporal GNN | original building ST-GNN: preserved verification, metrics, audit and figures; separate new-input run status | legacy run, pipeline runs |
| 6 · Change detection | preserved strip and tile overlays; selectable pipeline reports and CSV/PNG export; link to local crop viewer | legacy run, pipeline runs |
| 7 · Evaluation | evidence hierarchy; per-tile legacy GNN agreement; visual audit; building instances vs OSM 2023; what is still unmeasured | legacy run, OSM comparison |
| Run pipelines & results | independent stage choices; upload/path/folder inputs; common crop and stage viewer | `dashboard/pages/pipelines.py` |
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

Local change crops display the saved **GNN change score** when available, alongside
the evidence status. Higher values indicate greater model support for change;
strong image evidence, weak candidates and seasonal differences retain separate
labels. Unscored events show `unavailable`. Segmentation confidence is not shown
or used as a filter in the crop browser. New-input GNN fits are trained on the
current area and do not provide independently tested confidence or accuracy.

## Data
Paths can be overridden with environment variables:

| Variable | Default | Content |
|---|---|---|
| `SG_LEGACY_BUNDLE` | `data/legacy_building_stgnn` | frozen study: strip imagery, segmentations, graphs, temporal links (`python -m src legacy-building-gnn bundle …`) |
| `SG_LEGACY_RUN` | newest `artifacts/legacy_building_stgnn/*/report.json` | verified legacy GNN run (`… legacy-building-gnn run …`) |
| `SG_RUNS` | `artifacts/shouf3` plus `artifacts/image_comparisons` | pipeline runs; an explicit override scans only that folder |
| `SG_EXTERNAL_EVAL` | `data/external_eval`, falling back to `D:\new\OriginalPipeline_Improved\improve\out\compare_repo` | optional OSM building comparison tables |
| `SATELLITEGRAPH_RUN` | — | run folder preselected in the pipeline workspace |

Missing sources produce a notice on the affected page, not an error.

## Tests
```powershell
python scripts/run_project_tests.py tests/test_thesis_dashboard.py tests/test_preview_browser.py
```
- **Dashboard tests:** every page renders without exceptions on the real artifacts (skipped when the bundle is
  absent). They also check that evidence badges are present, that object selection works, and that the original
  tool stays mounted.
- **Original tool:** `test_preview_browser.py` covers the original tool.
