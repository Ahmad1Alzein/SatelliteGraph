# Native validation evidence

Execution date: 2026-10-08. Python 3.11, Windows, CPU PyTorch 2.12.0; CUDA is unavailable. The reviewed implementation was merged locally into `main` at `49d2cf2`. Run commands from the repository root.

## Verified models and dependencies

Read-only preflight reports `ready`. All seven required local asset files match the manifest hashes: SAM 2.1 Hiera-Large, FasterSeg architecture and weights, Delineate Anything v2, and the three SegFormer files. New assets were downloaded from pinned publisher revisions. Old assets were reused; their local hashes are recorded honestly because publisher checksums were not available. Both new models and both old backends completed real CPU inference. No additional model download is required in this workspace; no universal project-specific GNN checkpoint exists.

General processing needs no `osgeo`; original study reproduction does. The shared environment was not replaced. pytest is installed under ignored `.tools/python` and invoked with `scripts/run_project_tests.py`.

## Real imagery runs

Crops came from the actual 2017 and 2023 selection rasters, at original offsets (128,128), shape 64 by 96 pixels, EPSG:32636, 1.5 metres per original pixel. Metadata and source windows were checked. No original imagery was modified.

| Run | Methods | Inputs | Annual nodes/edges | Recorded stage seconds | Validation |
|---|---|---|---|---|---|
| `artifacts/native_validation/new_pair` | new/new/classical | 2017 and 2023, one selected PNG each | 2017: 15/30; 2023: 20/44 | segmentation 33; graph 5; change 5 | pass |
| `artifacts/native_validation/old_pair_verified` | old/old/classical | same selected pair | 2017: 5/6; 2023: 3/2 | segmentation 41; graph 2; change 1 | pass |
| `artifacts/native_validation/unequal_pair` | new/new/classical | one earlier tile, two overlapping later tiles | earlier: 15/30; later: 31/65 | segmentation 58; graph 4; change 1 | pass |

The unequal case uses test labels 2018/2024 for the same 2017/2023 source imagery; these labels test year independence, not new acquisition dates. Its second later crop starts at (192,128). Areas and distances are recorded in metres, and overlap ownership is explicit.

The new pair was segmented first, then graph and change were run in separate native CLI processes with only method/output supplied. Saved paths, CPU/window/overlap/TTA settings persisted. The small new runs use window 64, overlap 16, TTA off. The old smoke uses the actual models with one SAM sampling point, no crop layer, FasterSeg input size 64 and TTA off; this verifies compatibility, not default-setting accuracy. Two-year previews were inspected visually. After review fixes, fresh runs `new_pair_final` and `old_pair_final` also passed: new stage times 36/5/4 seconds; old 52/5/4 seconds. Both contexts record all seven model files, and the completed new pair resumed with hash verification and no repeated stages.

The real pair exposed an invalid-border evidence slicing bug caused by resampling; a failing regression reproduced it and passed after source/destination clipping was added.

## Regression evidence

The pre-change baseline passed 270 tests and 114 subtests. New targeted checks cover input placement/masks, shared grids, real model inference, adapters, both graph methods, ownership, evidence/roof shifts, unknown/missing coverage gates, all eight selections, saved options, moved outputs, interrupted resume, stage-method changes and hash tampering. Deterministic staged and one-pass graph tensors match. GNN tests cover insufficient labels, training-only status, saved weights, independent groups and training-only normalization.

The first full implementation suite passed 322 tests and 114 subtests in 216.20 seconds. After independent review fixes, a further full suite passed 343 tests and 114 subtests in 285.19 seconds. Both reported the same pre-existing Rasterio warning. The final complete tree passed 346 tests and 114 subtests in 193.88 seconds, including real semantic CPU inference, with the same single existing warning. Known-shift current-image registration and connected building-extension checks also pass. Original algorithms/reference files: all 37 files remain byte-identical to the baseline after the package rename.

## Practical limits

Full published 2013-2023 reproduction is unverified: original inputs and the recorded GDAL environment are absent. General runs never import its historic shifts, audits or expected scores. Real crop validation checks execution and artifact integrity; it does not establish segmentation/change accuracy on arbitrary imagery.

PNG placement metadata is required. Metric new segmentation requires physical scale. Cross-year inputs without a CRS require explicit or estimated registration. Missing coverage and unknown observations cannot become confident change labels. Explicit per-tile and comparison-mosaic memory budgets remain; use spatial subsets for large/gapped inputs. GNN evaluation requires sufficient labels and independent spatial groups. A single pair does not establish a held-out accuracy score.

The [final review record](pipeline-review.md) lists every confirmed defect, its regression and the execution rulings. New evidence fields use earlier/later names; tree/road change regions and JSON reports agree; construction targets require later-year building endpoints. Model, metadata, source and saved-GNN content changes are fingerprinted or verified before reuse.

## Local main integration

After the approved fast-forward merge, the full suite passed again from the repository root: 346 tests and 114 subtests in 207.07 seconds, with the same existing Rasterio warning. The test log is `artifacts/native_validation/verification/main-merged-tests.log`. No remote push was performed.

Validation outputs and local test dependencies were preserved in the main workspace. Output/input/model path records were relocated from the temporary worktree to the repository root; scientific arrays were unchanged. All 37 canonical artifact manifests were verified after migration. Both `new_pair_final` and `old_pair_final` resumed successfully from the root. Original JSON/YAML path records are archived in `verification/original_path_records.zip`, with the migration report alongside them.
