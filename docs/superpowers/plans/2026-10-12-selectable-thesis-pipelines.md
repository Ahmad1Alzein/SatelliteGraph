# Selectable thesis pipelines

The user requested independent old/new choices for segmentation, spatial graphs,
and change detection, and confirmed that the retained GNN is the original building
ST-GNN displayed by the thesis dashboard. The experimental all-class RGNN and its
training entry points are to be removed. Work is performed in the requested workspace.

1. Add regression tests for dashboard method selection, all eight combinations,
   classical-only runs, empty results, and the original GNN adapter.
2. Reuse the byte-identical original GNN's features, relations, model and training
   function through a new adapter. Keep its building/built-over target roles;
   label new-input fits as training only. Never modify frozen code or results.
   Reject experimental checkpoints explicitly rather than load different weights.
3. Expose independent method selectors and input options in the dashboard's run
   page, using a fresh output per launch. Use the common verified result browser
   for every combination. Remove experimental training controls and source modules.
   The user subsequently requested removal of the standalone app: move its
   workspace into `dashboard/pages/pipelines.py` and mount it in `thesis_dashboard.py`.
4. Make old Stage 5 change detection independent of the spatial graph choice and
   adapt its events to the current report contract. Keep raw old reports as well.
5. Make dashboard result loaders tolerate classical runs and empty events; show
   selected methods and genuine GNN provenance. Retain frozen study pages/results.
6. Run pipeline matrix, Streamlit page tests, full suite and frozen integrity
   checks. Record any environment-dependent failures and verification limits.

Review focus: switching stage methods must invalidate dependent results; old class
IDs must be converted once; absent GNN reports must not crash classical runs;
new-input GNN training must never be described as the frozen study's accuracy;
original code, input bundle, and existing verified predictions stay untouched.

Documentation cleanup requested by the user: keep one project `Readme.md`, merge
private-data instructions into it, and preserve detailed historical reproduction
instructions under `docs/`. Upstream library READMEs remain with vendored code.
Keep `SHA256SUMS.txt` because it verifies the private data package; remove its
obsolete `README_DATA.md` entry after consolidation. All 137 remaining entries
were verified identical, with no missing files.

Verification so far: original GNN verification passed all 44 checks, including
byte-identical predictions and pixel-identical figures. All eight stage
combinations passed the common viewer tests before app relocation; those tests
are repeated through the integrated dashboard. Focused startup, selector, cache
refresh and GNN-to-classical report tests passed after relocation. Frozen figure
regeneration requires GDAL, absent from the current project interpreter; its
optional test now records that dependency as a skip, without altering frozen code.

Final verification: 397 regression tests and 114 subtests passed; one optional
GDAL figure-regeneration test skipped. All eight combinations were executed
through the integrated dashboard viewer. An additional 18 visualization tests
passed after fixing earlier-land built-over predictions in major-change focus.
The final frozen-output check again passed all 44 outputs. The private package's
137 checksums match, current documentation links resolve, and `git diff --check`
is clean. No original scientific code, frozen input or preserved result changed.

Follow-up verification (2026-10-13): 624 tests and 114 subtests passed, one optional
GDAL test skipped, and three real-model tests were excluded to respect the user's
request to avoid extra SAM jobs. Archive browsing now handles partial historical
exports without claiming manifest verification. Registration retains local RANSAC
inliers before deduplication, and its algorithm version always invalidates cached
PNG alignment. The exact uploaded pair passed registration with 35/48 final
inliers and 1.013 m fitted-control RMSE.

The crop browser now omits segmentation confidence from cards, filters and its
change table. It shows saved GNN change probabilities when available and preserves
strong/weak/seasonal evidence labels; unscored events remain unavailable. Both new
UI regression cases failed against the previous display before the fix. All 63
focused dashboard, pipeline-matrix and report tests passed after the change.
The final integrity check again matched all 44 preserved GNN outputs, all 137
private-package checksums matched, and independent review found no crop-score
issues. Original GNN code, frozen data and saved predictions remain untouched.
