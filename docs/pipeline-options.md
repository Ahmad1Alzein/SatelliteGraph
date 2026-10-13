# Semantic Temporal pipeline

The general pipeline lives in `pipelines/semantic_temporal/runtime/`. It accepts one or two years, arbitrary folder names and tile counts, or individual images. The existing `src` stages and `image`/`folder` commands remain available. `pipelines/semantic_temporal/run_experiment.py` separately preserves the frozen 2013-2023 study; it is never invoked by the general commands.

| Selection | Method |
|---|---|
| Segmentation `old` | Existing SAM 2.1 + OpenEarthMap FasterSeg, automatic Stage 2R refinement |
| Segmentation `new` | Local SegFormer, Delineate Anything v2, multi-scale evidence and region refinement |
| Graph `old` | Existing Stage 3 adjacency and Stage 4 overlap ownership |
| Graph `new` | Connected regions, adjacency, nearby buildings and nearby areas |
| Change `classical` | Original Stage 5 polygon change rules, independent of graph choice, adapted to current reports |
| Change `gnn` | Original thesis binary building ST-GNN from `pipelines/legacy_building_stgnn/original/gnn_change.py` |

All eight segmentation x graph x change combinations are supported through explicit adapters. Old graph objects may remain multipart; the new graph adapter splits disconnected objects and records lineage. New segmentation uses a method-defined 2x grid: this does not increase physical detail.

## Native PowerShell commands

From the repository root, use the project environment:

```powershell
$python = '..\.venv\Scripts\python.exe'
& $python pipelines/semantic_temporal/check_environment.py
& $python -m src pipeline run --segmentation new --graph new --change classical --folders tiles17_23/2017 tiles17_23/2023 --output artifacts/runs/new_folders --device cpu
```

One image versus one image:

```powershell
& $python -m src pipeline run --segmentation new --graph new --change classical --images tiles17_23/2017/tile_00000.png tiles17_23/2023/tile_00000.png --output artifacts/runs/new_pair --device cpu
```

Use `--years 2017 2023` when directory names are not years. One input supports segmentation and graph only: add `--until graph`. Change needs exactly two chronological years.

Separate commands inherit every saved input and inference option from `run.json`:

```powershell
& $python -m src pipeline segment --method new --folders tiles17_23/2017 tiles17_23/2023 --output artifacts/runs/staged --device cpu
& $python -m src pipeline graph --method new --output artifacts/runs/staged
& $python -m src pipeline change --method classical --output artifacts/runs/staged
```

Old methods through the same general input contract:

```powershell
& $python -m src pipeline run --segmentation old --graph old --change classical --folders tiles17_23/2017 tiles17_23/2023 --output artifacts/runs/old_folders --device cpu
```

Standalone entry point (same options and implementation):

```powershell
& $python pipelines/semantic_temporal/run_pipeline.py run --segmentation new --graph new --change classical --images tiles17_23/2017/tile_00000.png tiles17_23/2023/tile_00000.png --output artifacts/runs/standalone
```

The first run hashes input files and model assets. CPU is the default and can be slow for full folders. `--device cuda` requires an available CUDA device; `auto` selects it when available. `--inference-window 64 --inference-overlap 16 --no-tta` is useful for small smoke runs; use the default 512/128 and TTA for normal processing.

## Input placement and scale

PNG folders require `metadata.csv` with one row per image: `tile_name,x0,y0`. Offsets are original image pixel coordinates. Optional width/height and valid_width/valid_height are checked; alpha and raster masks exclude missing pixels. Filenames never establish positions.

A matching georeferenced RGB source GeoTIFF supplies the original affine transform, CRS and resolution. PNG pixels must match its windows. The example folders already have metadata and `selection_<year>.tif`. External files use `--metadata FIRST.csv SECOND.csv --source-rasters FIRST.tif SECOND.tif`. Alternatively, complete `tile_affine_a` through `tile_affine_f` and `target_crs` metadata supplies georeferencing. Direct georeferenced RGB GeoTIFF images need no separate CSV. uint16 RGB is converted using the full 0 to 65535 range, rather than an image-dependent contrast stretch.

PNGs with placement metadata but no CRS are accepted in original-image coordinates. Metric segmentation needs `--pixel-size METRES_PER_PIXEL` (one value per input). Cross-year original-pixel alignment needs `--register-images` or `--registration-json`. Without physical scale, comparisons are limited to correspondences; metric event rules and building-near edges are withheld. Old segmentation can still operate in pixels. PNGs without placement metadata are rejected because their original positions cannot be recovered reliably.

Mapped inputs are projected onto a shared metric grid, anchored to the later/reference image. Use `--target-crs` and `--target-resolution` to override it. Coordinate georeferencing alone is recorded as alignment not independently verified. `--register-images` estimates current-data SIFT/RANSAC alignment, with coverage/inlier checks; failure is actionable rather than silently accepting a shift. Control JSON contains `moving_xy` and `reference_xy` arrays of corresponding world-coordinate points. Roof compensation uses current observations; historic study shifts and audits are never reused.

Tile counts and dimensions can differ; temporal comparisons use actual overlap and annual pixel ownership. Missing or unknown coverage cannot become confident positive or negative training labels.

## Resume and override behavior

```powershell
& $python -m src pipeline run --segmentation new --graph new --change classical --output artifacts/runs/new_pair --resume
```

Omitted flags inherit saved values, including custom model paths, source paths, registration and booleans. Explicit boolean resets use `--no-register-images`, `--no-skip-input-check`, `--tta` or `--no-tta`. `--skip-input-check` is retained for compatibility and persists; it does not bypass the general placement, mask and scale checks. General runs do not compare files against the historic input manifest.

Changed upstream inputs, segmentation model assets or preprocessing options invalidate a run: use a fresh output folder. Changing a stage method rebuilds its dependent stages. Completed artifacts are hash-verified on resume. An interrupted new segmentation resumes at verified completed tiles. Saved outputs can be moved; original source/model files must remain accessible. Older reproduction-format saved runs need a fresh general output folder.

Explicit memory budgets protect large inputs: `--max-pixels` defaults to 16 million prepared output pixels per tile, and `--max-comparison-pixels` to 4 million mosaic pixels. Large or spatially separated inputs should be split into meaningful overlapping areas, or budgets raised to fit available memory. Full-image reads and probability/evidence tensors still require RAM; arbitrary folder support is not an unlimited-memory guarantee.

## GNN behavior

Choose `--change gnn` to use the original thesis ST-GNN. The experimental sixteen-head RGNN, temporal-difference enhancement and external LsSCD-Ex training implementation have been removed. No downloaded or experimental checkpoint is loaded. `--gnn-checkpoint` is rejected explicitly; the original model does not save trained weights. `--gnn-hidden` is retained only for CLI compatibility: the original architecture always uses 64 hidden units. `--gnn-epochs` defaults to 300 and uses at least 5 epochs because the original training function selects a state every 5 epochs.

`pipelines/legacy_building_stgnn/adapter.py` converts either segmentation/graph combination into the original feature and relation contract. It calls the original `relations` and `train_fold` functions, with the original binary `RGNN`. Extra context-near edges are excluded from this model's inputs. Targets retain the original roles: later buildings that are new/extended and earlier non-building land built over. Unknown, low-confidence and unobserved nodes are masked. Reliable changed and unchanged targets are both required; otherwise the report says `insufficient_data` and exports no scores. New-input fitting is labeled `training_only`, with no held-out accuracy claim. Other classes remain image-evidence candidates rather than extra GNN heads.

New-input pseudo-labels come from the current comparison catalog, not the frozen
tile labels. Earlier land is a positive built-over target when at least half its
area intersects reliably changed later buildings; eligible unchanged land is
negative and ambiguous land is masked. This conservative rule is specific to
new-input runs and does not alter the preserved study's audited target labels.

The preserved 2013–2023 study is separate: use `python -m src legacy-building-gnn run` to reproduce its original seven-fold experiment, or `verify` to check existing results. The frozen source, bundle, predictions and figures are never overwritten by general runs. See [the original GNN](legacy-building-stgnn.md).

Use `--change-policy policy.json` for validated partial evidence thresholds, `--reviewed-targets reviewed.csv` for eligible reviewed labels, and `--max-report-pixels` for the complete report budget (default 128000000). Policy/label changes rebuild change only. Budget/style changes rerender unchanged verified science. `run.json` distinguishes scientific and render status; a render failure retains completed scientific artifacts.

## Models, requirements and outputs

```powershell
& $python scripts/download_pipeline_models.py
& $python -m pip install -r requirements.txt
```

The downloader explicitly acquires and hash-verifies required old and new model assets; inference never downloads implicitly. Existing differing files are preserved and reported. Assets live under `checkpoints/`, with new assets in `checkpoints/semantic_temporal/`. The current environment and all seven required files were verified; no additional downloads are needed here. General processing uses Rasterio and does not require `osgeo`. Frozen study reproduction separately needs its recorded GDAL environment and original dataset.

Outputs include `run.json`, resolved context/registration and segmentation model paths/hashes, canonical segmentation rasters and lineage, per-tile and annual graph JSON/NPZ, temporal graph, classical events/targets, optional GNN predictions/report, `validation.json`, `preview_index.json` and `previews/`. Areas use recorded units; GeoJSON is exported to WGS84 only when a CRS exists. Two-year previews show separate actual-year panels.

See [validation evidence](pipeline-validation.md). Scientific accuracy on arbitrary new imagery still requires independent assessment; historic published scores are not transferred to new runs.

## All-class PNG results and registration

The report format includes segmentation/graph PNGs, landmark/object matching, plain-language legends and one combined change report. Start at `results/index.json`. Empty coverage is excluded internally. [Earlier all-class results](all-class-change-results.md) document the retired experimental model only; their checkpoint/training commands no longer apply.

Mapped-image automatic registration reuses the existing phase-guided SIFT/RANSAC method. `--registration-max-residual` explicitly changes the default 2-unit landmark tolerance; inspect fitted residuals and the matching PNG. `--harmonize-images` matches paired brightness/sharpness before segmentation, while reports preserve original RGB.
