# Reproducible experiment: object-based spatio-temporal graphs for change detection, Shouf 2013 → 2023

This folder contains the exact code of the final 7-tile experiment. Provide the input data and two model
checkpoints, run one command, and the experiment is rebuilt from the raw tiles. The command then compares every
data output with the original results and writes a reproduction report.

```
python run_experiment.py
```

## What it reproduces

| Step | Output | Original result |
|---|---|---|
| Input validation and alignment | 2013 co-registered to 2023 per tile (phase correlation + acceptance gate), harmonized | residual ≤ 0.75 px, 7/7 tiles accepted |
| Segmentation | 14 frozen releases (7 tiles × 2 years; paired 2023 v2 for tiles 2 and 3), 15/15 checks each | 4,123 segments (2013), 3,651 (2023) |
| Region graphs, temporal layer | per-tile spatial graphs, 2013↔2023 temporal correspondence, frozen | 8,697 temporal edges |
| Strip | 7 tiles stitched by metadata x0/y0, strip spatio-temporal graph | 7,020 nodes, 27,056 edges |
| GNN change detection | relational GNN, leave-one-tile-out | AP 0.53, ROC-AUC 0.906, F1 0.549 (MLP AP 0.084, spatial-only GNN AP 0.10) |
| Audit agreement | GNN vs 43 human-audited real changes | 33 / 43 detected |

## 1. Environment

The experiment was run on Windows 10 with Python 3.11, PyTorch 2.11 (CUDA 12.8) and an NVIDIA GTX 1650 (4 GB).
The full versions are in `environment/reference_environment.json`.

| File | Use |
|---|---|
| `environment/environment_windows_exact.yml` | the exact environment: `conda env create -f environment/environment_windows_exact.yml` |
| `environment/requirements.txt` | the pinned key packages, for other platforms (GDAL from conda-forge, see the header of the file) |
| `environment/pip_freeze_full.txt` | the full package list of the original environment, for reference |

The segmentation runs on the GPU when one is available (it also runs on the CPU, but much slower). The GNN always
trains on the CPU in deterministic mode.

## 2. Inputs

Put the files in the layout below, or point to other folders with `--data` and `--checkpoints`. Before anything
runs, every file is checked against `reference/input_manifest.json` (SHA-256).

```
data/
  dataset/2013/   tile_00000.png … tile_00006.png, metadata.csv, shouf_2013_tiles.csv
  dataset/2023/   tile_00000.png … tile_00006.png, metadata.csv, shouf_2023_shared_tiles.csv
  aligned/        shouf_2013_epsg32636_1p5m.tif, shouf_2023_shared_epsg32636_1p5m.tif
checkpoints/
  DelineateAnythingv2.pt                     Delineate Anything v2 (YOLO) field/parcel instances
  satlens-segformer/                         SegFormer land-cover model (OpenEarthMap classes)
    config.json, model.safetensors, preprocessor_config.json
```

- **`dataset/`**: the 7 tiles per year (512 × 512 px, 1.5 m, EPSG:32636) and their metadata. The `x0`/`y0` and
  affine columns define the shared grid used for stitching.
- **`aligned/`**: the full 1.5 m rasters of both years. They supply only the margin the tiles do not cover. The
  2013 image is shifted by about 20 px (about 30 m) against 2023, so aligning the tile strip needs 2013 content
  from just outside the tiles: about 19 rows below the strip and the right edge of tile_00006. Inside the tiles,
  the dataset pixels are used (and checked to be identical to the rasters).

## 3. Run

```
python run_experiment.py                                   # writes ./work
python run_experiment.py --work D:/runs/repro1 --data D:/inputs --checkpoints D:/models
```

The run takes about 1 h 30 min on the original machine. It prints every stage, and stops at the first failing
stage. The full log is in `<work>/logs/run.log`. The work folder must not exist yet, because frozen releases are
never overwritten. To re-run, choose a new `--work` folder.

## 4. Outputs (in the work folder)

| Path | Content |
|---|---|
| `final_segmentation/FROZEN_<year>_<tile>_<v>/` | read-only segmentation releases: `segment_ids.tif`, `nodes_final.csv`, `segments.gpkg`, validation report, MANIFEST |
| `graphs/graph_<year>_<tile>_<v>/` | per-tile region graphs (nodes, edges, `graph.graphml`) |
| `graphs/FROZEN_temporal_2013_2023_<tile>_v3/`, `graphs/FROZEN_stgraph_2013_2023_<tile>_v1/` | temporal layers and per-tile spatio-temporal graphs (frozen) |
| `graphs/change_stgraph_2013_2023_<tile>/` | rule-based change labels (the GNN's pseudo-labels) and audit montages |
| `graphs/FROZEN_strip_tile_00000-00006_v1/` | **final release**: strip mosaic of both years, strip spatio-temporal graph, GNN results (`gnn/summary.json`, `gnn/predictions.csv`), labels, figures, validation F1–F5 |
| `thesis_figures/` | one defense figure per pipeline step |
| `gnn_viz_tile_00002/` | the GNN's own predictions on the tile_00002 imagery (2013 / 2023 side by side, P(change) per node) |
| `REPRODUCTION_REPORT.md` | comparison with the original experiment |

## 5. Verification

`verify_outputs.py`, run automatically as the last step, checks 2 things:

- **Files:** the SHA-256 of every data output (551 files: `.csv`, `.tif`, `.graphml`, `.npz`, `.npy`, `.pt`) is
  compared with `reference/expected_outputs.json`. GeoPackages, JSON/Markdown reports and figures are not
  compared, because they contain timestamps, run times and fonts.
- **Key results:** 121 values (segment counts, checks per release, temporal edge counts, strip graph size,
  all GNN metrics, audit agreement, freeze status) are compared with `reference/expected_results.json`.

The status is **EXACT** (exit 0), **RESULTS IDENTICAL, SOME FILES DIFFER** (exit 2) or **DIFFERENT** (exit 1).
You can re-check a finished run with `python verify_outputs.py <work folder>`.

### Expected determinism

- **Same environment and GPU model:** byte-identical. This was tested by re-running the package from a fresh copy;
  see the table at the end.
- **Different GPU, driver or CUDA/cuDNN version:** the segmentation models run on the GPU, and GPU convolutions are
  not guaranteed bit-identical across hardware. A few boundary pixels can then differ, which propagates to segment
  ids and everything downstream. The report shows how far the results moved. The GNN itself is deterministic: it
  trains on the CPU with `torch.use_deterministic_algorithms(True)`.
- **CPU only:** this runs, but expect small segmentation differences for the same reason.

## 6. What is fixed in the experiment (and why)

- **Frozen rules:** all segmentation, graph, temporal-link and change-label rules are the frozen ones. No
  parameter can be changed from the command line. The tile-0 change detection rules are the baseline and are not
  tuned.
- **Human audit verdicts** (`reference/audit_verdicts.csv`, 58 items: 43 real, 13 questionable, 2 unclear): these
  are the visual judgements made once on the original run. A re-run cannot regenerate human judgement, so the
  verdicts are shipped and attached only if the run's change tables are byte-identical to the audited ones (hashes
  in `reference/audited_change_tables_sha256.json`). They are used only for the GNN's audit-agreement report,
  never for training.
- **`reference/shift_samples.csv`**: AOI-wide 2013→2023 offsets measured once on the whole study area
  (`reference/measure_shift.py`, which needs the full AOI tile set). The alignment step uses them only as an
  independent check: its result must agree within 3 px.

## 7. Code provenance

- `pipeline/` is the code of the successful run. That code differed from the frozen release code snapshots only
  in input paths (`reference/successful_run_vs_frozen_release.diff`).
- For this package, only input/checkpoint locations (`pipeline/paths.py`) and font paths (`pipeline/fonts.py`;
  Arial on Windows as originally, DejaVu Sans elsewhere, which affects figures only) were changed. These edits are
  listed line by line in `PATCHES.diff`.
- `pipeline/prepare_inputs.py` and `pipeline/transfer_audit_verdicts.py` are the input-preparation and verdict
  steps of the successful run, reading from the package locations.

## 8. Troubleshooting

- **`DLL load failed` when importing GDAL or rasterio:** GDAL must be imported before torch. The pipeline scripts
  already do this, so check that GDAL comes from conda-forge.
- **CUDA out of memory:** segmentation needs about 3.5 GB of GPU memory at the 2× scale used.
- **`exists and is not empty`:** choose a new `--work` folder.
- **Input check fails:** your files differ from the original inputs. `--skip-input-check` runs anyway, but the
  verification will then report differences.

## Package layout

```
run_experiment.py        one command: check inputs → run all steps → verify
verify_outputs.py        output and result comparison with the original experiment
pipeline/                experiment code (copied into the work folder and run there)
reference/               input manifest, expected outputs and results, audit verdicts, AOI shift samples, diffs
environment/             exact and minimal environment specifications
data/, checkpoints/      put the inputs here (see section 2)
PATCHES.diff             package edits vs the code of the successful run
```
