# General-purpose SatelliteGraph Pipelines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver separate old and semantic-temporal pipelines supporting metadata-based folders and single-image pairs, arbitrary chronological years and tile counts, persistent stage options, verified model assets and interoperable graph formats.

**Architecture:** Keep the existing old stages and the frozen experiment algorithms available. Rename the experiment package to semantic_temporal and build a general runtime with explicit input/context/artifact contracts, extracted reusable algorithms and tested adapters. Both the shared CLI and standalone runner use this runtime; published reproduction remains an explicit separate runner.

**Tech Stack:** Python 3.11, NumPy, SciPy, OpenCV, Rasterio, Shapely, PyTorch, Transformers, Ultralytics, NetworkX, Pillow, pytest; existing SAM 2/OpenEarthMap runtimes.

**Spec:** `docs/superpowers/specs/2026-10-08-general-purpose-pipelines-design.md` (approved).

## Global Constraints

- Image folders must contain metadata locating images within their original source image. Do not invent a layout from filenames.
- Rename `pipelines/experiment_2013_2023` professionally; the selected name is `pipelines/semantic_temporal`.
- Preserve the old pipeline and expose the two methods independently.
- Include a one-image-per-year mode, not only whole folders.
- Valid transparent/nodata pixels must remain excluded from observed coverage.
- General runs must not import historic shifts, human verdicts or original-study expected results.
- CPU execution is supported. Do not automatically replace the user's existing PyTorch or install a system CUDA toolkit.
- Download only missing or invalid model assets; never overwrite existing differing checkpoints silently or commit model binaries.
- Preserve unknown classes, object lineage, coordinate units and the distinction between physical changes and missing observations.
- Do not generate modified Python source through textual year substitutions.
- Model-load and real inference checks are required before claiming the runtime is usable.
- Original-study exact reproduction is only claimed after its inputs and output verification are available.

## Review Focus

1. Resuming after changing a source raster/configuration must not reuse graphs derived from old pixels (Task 2).
2. Same basenames in distinct years, Unicode/space-containing Windows paths and relocated legacy run paths must resolve without collisions (Tasks 2, 4, 12).
3. Cropped, rectangular and partially transparent tiles must retain original offsets and never create changed objects in missing coverage (Tasks 4, 6, 9).
4. Overlapping tiles must not leak the same observation into different GNN evaluation groups (Task 10).
5. Original pixel coordinates with no physical scale must never produce metre-labelled areas, distance thresholds or georeferenced exports (Tasks 4, 5, 8).

## File map and execution conventions

Preserve existing `src/stage2`, `src/refinement`, `src/stage3`, `src/stage4`, `src/stage5` and existing folder/image commands. Modify shared wiring in `src/cli.py`, `src/pipeline_select.py` and, only where necessary, `src/folder_pipeline.py`.

New files live under `pipelines/semantic_temporal/runtime/`: `contracts.py`, `context.py`, `inputs.py`, `registration.py`, `storage.py`, `models.py`, `segmentation/{__init__,semantic,parcels,regions,refinement}.py`, `adapters.py`, `graphs.py`, `temporal.py`, `gnn.py`, `visualization.py`, `preflight.py`, `runner.py`. Add package `__init__.py` files and standalone `run_pipeline.py`/`check_environment.py` in the owning tasks.

Existing frozen `pipeline/`, `reference/`, environment specifications, `run_experiment.py` and `verify_outputs.py` move under the renamed package. Their study-specific behavior remains exclusive to reproduction.

Tests use actual small raster/mask/graph artifacts and deterministic injected model outputs where inference is not under test. Mocking subprocess dispatch alone is insufficient. Use the already downloaded models for marked real smoke tests.

For each task, record the red test failure, green result and commit before moving on. Commands below use `$python`, the selected Python 3.11 environment. If pytest is missing, install it into an ignored workspace-local dependency directory and invoke tests through `scripts/run_project_tests.py`; the script adds that directory to sys.path before importing pytest. Do not modify the shared parent virtual environment merely to add tests.

### Task 1: Regression-tested persistent options

**Files:** Modify `src/cli.py`, `src/pipeline_select.py`; test `tests/test_pipeline_select.py`, new `tests/test_pipeline_options.py`; create `scripts/run_project_tests.py`; modify `.gitignore` for `/.tools/`.

**Interfaces:** Produce `resolve_options(saved: dict, supplied: dict, *, initial: bool, project_root: Path) -> dict` in `src/pipeline_select.py`. Preserve the existing segment/graph/change/run public signatures.

- [x] Add `test_separate_cli_stages_inherit_inputs_and_flags`: call segment with custom data/checkpoints/config/skip-input-check, then graph and change without those options; assert all three effective dictionaries match. Add explicit true-to-false override tests and registration mutual-exclusion inheritance tests.
- [x] Run `& $python scripts/run_project_tests.py tests/test_pipeline_options.py -q`; confirm failures describe reset options, not missing test dependencies.
- [x] Make omitted CLI values None, including booleans; add inverse switches such as `--no-skip-input-check` and `--no-register-images`. Apply initial defaults only after inheritance. Serialize resolved absolute paths and effective flags; segment also receives resolved options.
- [x] Run option tests and existing `tests/test_pipeline_select.py`; update parsing assertions to test effective initial defaults rather than requiring parser defaults.
- [x] Commit `fix: preserve selectable pipeline options between stages` with these exact files.

### Task 2: Reliable stage ledger and invalidation

**Files:** Create `src/pipeline_state.py`; modify `src/pipeline_select.py`; test `tests/test_pipeline_state.py`.

**Interfaces:** Produce `stage_fingerprint(inputs: list[Path], settings: dict) -> str`, `validate_resume(state: dict, stage: str, fingerprint: str) -> None`, `record_stage(output: Path, stage: str, status: str, **details) -> dict`. Atomic JSON remains the persistence format.

- [x] Add tests `test_changed_input_rejects_resume`, `test_failed_process_cannot_use_stale_complete_ledger`, `test_graph_change_failure_invalidates_dependents`, `test_legacy_options_load_with_space_and_unicode_paths`; assert explicit mismatch diagnostics and no falsely complete stage.
- [x] Run `& $python scripts/run_project_tests.py tests/test_pipeline_state.py -q`; confirm red assertions.
- [x] Hash source files, metadata, model manifests and effective per-stage settings; exclude pure presentation/progress flags from upstream scientific fingerprints. Reject changed upstream inputs with a new-output/recompute instruction. Check real subprocess return codes; treat reproduction verification exits 1/2 explicitly rather than conflating them with a failed stage or exact reproduction.
- [x] Run state/option tests; verify atomic replacement and recovery from a corrupt or missing ledger produce actionable diagnostics.
- [x] Commit `fix: validate saved pipeline state and resume fingerprints`.

### Task 3: Rename and preserve reproduction

**Files:** Move `pipelines/experiment_2013_2023/` to `pipelines/semantic_temporal/`; modify `src/pipeline_select.py`, `Readme.md`, `docs/pipeline-options.md`, `.gitignore`, `tests/test_pipeline_select.py`; create `pipelines/__init__.py`, `pipelines/semantic_temporal/__init__.py`; test `tests/test_reproduction_package.py`.

**Interfaces:** Produce `REPRODUCTION_PACKAGE: Path` and a distinct general runner entry point reserved for Task 12. Continue accepting explicit --data/--checkpoints in `run_experiment.py`.

- [x] Add tests asserting the new package path, availability of reproduction --help, unchanged 24-input/551-output manifests and old pipeline imports. Record a pre-move SHA-256 list for frozen pipeline/reference files.
- [x] Run the rename tests and confirm red path expectations.
- [x] Move files with checked workspace-contained source/destination paths. Update active references and ignore rules; keep historical spec/plan text as history. Preserve algorithm/reference bytes and explain reproduction versus general mode in the renamed README.
- [x] Compare frozen-file hashes before/after; run package tests and `& $python pipelines/semantic_temporal/run_experiment.py --help`. Verify that model binaries remain ignored.
- [x] Commit `refactor: name the new package semantic temporal`.

### Task 4: Explicit input and artifact contracts

**Files:** Create `runtime/{__init__,contracts,inputs,context,storage}.py`; create `tests/test_semantic_inputs.py`, `tests/semantic_fixtures.py`.

**Interfaces:** Produce immutable `GridSpec(crs, affine, units, physical_pixel_size)`, `TileInput(key, year, image, metadata, source_id, offset_xy, shape_hw, grid, valid_extent_hw)`, `YearInput(year, tiles, source_raster)`, `RunContext(output, years, options, models, fingerprint)`. Produce `SegmentationArtifact(year, tile_key, directory, grid, shape_hw, paths, lineage)`, `GraphArtifact(year, method, directory, grid, paths)`, `TemporalArtifact(years, directory, paths)`; paths is dict[str, Path]. Produce `inspect_inputs(options: dict, project_root: Path) -> tuple[YearInput, ...]`, `build_context(options: dict, output: Path) -> RunContext`, and `read_tile(tile: TileInput) -> tuple[np.ndarray, np.ndarray]` (RGB and boolean valid mask).

Contract field types: paths/directories/images/output are Path; optional metadata/source_raster are Path | None; year is int; key/source_id/tile_key/method/fingerprint are str. GridSpec.crs is str | None, affine is a six-float tuple, units is Literal['metres', 'pixels'], physical_pixel_size is tuple[float, float] | None. Offsets/shapes/extents are two-int tuples. YearInput.tiles is tuple[TileInput, ...]; RunContext.years is tuple[YearInput, ...], options/models are dict[str, object]. TemporalArtifact.years is tuple[int, int]; lineage is dict[str, list[dict[str, object]]]. Canonical class IDs are 1..8 in the original new pipeline's order, 0=unknown and 255=nodata; the separate valid mask distinguishes unknown observations from missing pixels.

- [x] Add tests for required metadata, duplicate/missing rows, exact x0/y0 source offsets, ambiguous raster discovery, one selected PNG among multiple files, independent duplicate basenames, differing year tile counts, and a direct georeferenced image. Assert a rectangular RGBA tile's valid mask and offset exactly match its metadata.
- [x] Run `& $python scripts/run_project_tests.py tests/test_semantic_inputs.py -q` and confirm missing contracts/behavior.
- [x] Normalize CSV placement and complete affine metadata or source GeoTIFF georeferencing. Use source windows for exact pixel checks. Keep original_pixel mode when map information is absent. Require physical scale before metric new segmentation; retain valid extents and lineage. Accept uint8 RGB/RGBA; for uint16 GeoTIFFs use recorded full-range scaling to uint8, consistently in tile/source checks, never per-tile percentile stretching.
- [x] Run input tests including geographic/rotated affines, a mismatched declared image dimension, negative/invalid offsets, nodata and explicit year ordering. Validate serialized context round trips.
- [x] Commit `feat: normalize metadata based folder and image inputs`.

### Task 5: Shared-grid preparation and registration

**Files:** Create `runtime/registration.py`; modify `runtime/inputs.py`; test `tests/test_semantic_registration.py`.

**Interfaces:** Produce `prepare_comparison(context: RunContext) -> RunContext` and `prepare_tile(tile: TileInput, context: RunContext) -> tuple[np.ndarray, np.ndarray, GridSpec]`. Save shared-grid/transformation/registration details in context.options and persisted provenance. Produce `coverage_pairs(context: RunContext) -> list[tuple[str, str]]` based on observed geographic/source-grid overlap.

- [x] Add deterministic shifted-image tests with different names/offsets and disjoint coverage; assert recovered alignment, no unmatched-coverage disappearance and unchanged input pixels. Add original-pixel mode tests with explicit physical scale, rejected unsupported grid identity and supplied transforms.
- [x] Run registration tests to red.
- [x] Reuse `src/stage5/registration.py` algorithms where applicable; introduce local-pixel registration with explicit source-frame transforms rather than fake CRSs. For projected comparisons choose a metric target and shared resolution; derive affine/area correctly including determinant and rotation. Use current imagery/control points, never historic shift samples. Process large imagery with windows rather than the old 4096-pixel tile refusal.
- [x] Run tests for low-confidence registration rejection, transparent margins, partially overlapping years, non-square windows and missing physical scale; ensure no metre-labelled original-pixel exports without scale.
- [x] Commit `feat: prepare arbitrary yearly grids for comparison`.

### Task 6: General new segmentation with actual models

**Files:** Create `runtime/models.py`, `runtime/segmentation/{__init__,semantic,parcels,regions,refinement}.py`; test `tests/test_semantic_segmentation.py`, `tests/test_semantic_models.py`; add real_semantic marker in `pytest.ini`.

**Interfaces:** Produce `ModelBundle` with lazy SegFormer/YOLO loaders, `load_models(context: RunContext) -> ModelBundle`, and `segment_tile(tile: TileInput, context: RunContext, *, models: ModelBundle | None = None) -> SegmentationArtifact`. Persist segment_ids.tif, segment_classes.tif, valid_mask.tif, probabilities.npy, nodes.csv, lineage.json and validation.json via storage.

- [x] Add model-injected tests asserting full valid-pixel assignment, unassigned invalid pixels, connected regions, non-square small-window inference, scale-dependent physical area and stable year/tile identity. Include a transparent island and a padded edge tile.
- [x] Run segmentation tests to red; run marked loader checks separately against the verified local checkpoints.
- [x] Extract the established segment_final semantics/regions, segment_v2 building/parcel methods and refine_merge logic into focused runtime modules with explicit scale/device/masks. Derive spatial kernels from physical scale and output grid; use sliding inference for arbitrary dimensions. Remove general-runtime dependence on GDAL and legacy mutable globals; keep original files untouched for reproduction. Avoid treating upsampling as added physical detail.
- [x] Run segmentation tests plus a marked real CPU inference at a small configured size. Assert class-channel mapping and valid-mask dimensions match actual outputs; check cuda/auto/cpu behavior.
- [x] Commit `feat: generalize semantic temporal segmentation`.

### Task 7: Old/new segmentation adapters

**Files:** Create `runtime/adapters.py`; modify `src/folder_pipeline.py` only for reusable stage access; test `tests/test_pipeline_adapters.py`.

**Interfaces:** Produce `load_old_segmentation(directory: Path, tile: TileInput) -> SegmentationArtifact`, `connected_region_view(artifact: SegmentationArtifact) -> SegmentationArtifact`, `export_old_refinement_contract(artifact: SegmentationArtifact, context: RunContext) -> Path`, `run_old_segmentation(context: RunContext) -> list[SegmentationArtifact]`.

- [x] Add adapter tests using real persisted old Stage 2/2R fixtures and new masks. Assert exact pixel conservation, mapped class IDs, unknown preservation, stable IDs, original object lineage after component splits and correct output affine after resampling.
- [x] Run adapter tests to red.
- [x] Map the existing eight-class/unknown registries explicitly. Retain semantic confidence/probabilities and evidence-availability masks. Export required validated Stage 2R structures for the old graph without replaying old refinement over new segmentation. Make old runtime accept selected-image and original-pixel context while preserving existing folder/image APIs.
- [x] Run adapter tests and existing folder/refinement tests; validate exported structures through the real old loaders and schema validators.
- [x] Commit `feat: adapt old and new segmentation artifacts`.

### Task 8: Distinct spatial graph methods and annual combination

**Files:** Create `runtime/graphs.py`; extend `runtime/adapters.py`; test `tests/test_semantic_graphs.py`.

**Interfaces:** Produce `build_tile_graph(artifact: SegmentationArtifact, method: str, context: RunContext) -> GraphArtifact`, `combine_year_graphs(graphs: list[GraphArtifact], context: RunContext) -> GraphArtifact`, `export_old_graph_contract(graph: GraphArtifact, context: RunContext) -> Path`. Persist a canonical graph.json with stable IDs/units plus nodes.csv, edges.csv, feature_spec.json and tensor files.

- [x] Add raster-derived tests for side versus corner adjacency, old touches-only versus new building-near edges, overlap deduplication with preserved members, disconnected extents and original-pixel units. Assert finite tensors for one-node/zero-edge graphs, unknown nodes and zero positional extent.
- [x] Run graph tests to red.
- [x] Reuse old Stage 3/4 algorithms through validated adapters; port new node/edge feature calculations from build_graph. Generate features from actual imagery/geometry/probabilities with explicit missing-evidence flags. Use actual transforms and metadata placement, deterministic overlap ownership and source lineage; avoid fixed 1024 grids and fixed-ID packing that can collide. Use memory estimates/windowed or disk-backed raster handling for large combined extents.
- [x] Validate both old graph schemas and canonical tensors; run new graph tests and existing Stage 3/4 tests. Test metric pixel-size changes and no-CRS pixel exports.
- [x] Commit `feat: build old and semantic spatial graphs from shared artifacts`.

### Task 9: Dynamic temporal graphs and classical labels

**Files:** Create `runtime/temporal.py`; extend `runtime/adapters.py`; test `tests/test_semantic_temporal.py`.

**Interfaces:** Produce `build_temporal_graph(graphs: tuple[GraphArtifact, GraphArtifact], context: RunContext) -> TemporalArtifact`, `detect_classical(graph: TemporalArtifact, context: RunContext) -> dict`. Preserve named classical events and expose training_targets.csv with gid/label/confidence/source fields.

- [x] Add tests using 2017/2023 and other years with unmatched/no-observation margins, persistent/constructed/extended buildings, unknown-class regions, split/merge lineage and nonidentical tile coverage. Assert all output records contain actual years and that nodata never becomes physical change.
- [x] Run temporal tests to red.
- [x] Route old graphs through Stage 5 correspondence/events via adapters; port new temporal_layer and stgraph_change rules with explicit dimensions, physical scales and years. Preserve source-rule provenance and observation confidence. Old+GNN targets map old named construction/extension/built-over events conservatively into the common building-change target, never substitute unknown/low-confidence labels with negatives.
- [x] Run temporal tests and existing Stage 5 tests. Ensure general mode cannot access historic audits/reference outputs and output hashes bind actual upstream inputs.
- [x] Commit `feat: generalize temporal correspondence and classical change labels`.

### Task 10: Dynamic GNN training and reusable checkpoints

**Files:** Create `runtime/gnn.py`; test `tests/test_semantic_gnn.py`.

**Interfaces:** Produce `run_gnn(graph: TemporalArtifact, context: RunContext, *, checkpoint: Path | None = None) -> dict`, `evaluation_groups(context: RunContext) -> dict[str, str]`, `save_gnn_checkpoint(path: Path, model, schema: dict, normalization: dict, settings: dict) -> None`. Return status, checkpoint, predictions, evaluation and audit statuses.

- [x] Add tests for arbitrary group counts, grouped overlapping observations, empty/no-positive targets, a single pair, absent audit records, checkpoint round trip, incompatible features and training-only results without held-out metrics.
- [x] Run GNN tests to red.
- [x] Extract RGNN relational message passing with seed=0 and deterministic CPU training. Fit node/edge normalization only on each training group, record fold membership, use validation only for thresholds/early stopping and apply test isolation consistently. Keep overlapping observations in one group and exclude cross-partition message-passing edges during independent held-out evaluation. Dynamic >=3-group evaluation requires usable train/validation/test target support. Save model/state/schema/normalization/target definitions; load custom GNN tensor state with weights_only=True and JSON metadata. Smaller datasets support saved-model inference or labelled training-only results, with no invented evaluation; insufficient targets return counts and an actionable status.
- [x] Run tests with tiny graphs and a short test-only epoch budget; verify saved-model predictions match before/after load. Keep historical human audit metrics optional and separate from pseudo-label agreement.
- [x] Commit `feat: support dynamic GNN training and checkpoint inference`.

### Task 11: Dynamic visualizations and run validation

**Files:** Create `runtime/visualization.py`; extend `runtime/storage.py`; test `tests/test_semantic_outputs.py`.

**Interfaces:** Produce `validate_run(context: RunContext, stage: str) -> dict`, `render_outputs(context: RunContext, stage: str) -> dict[str, Path]`. Save validation.json and preview_index.json; all images enumerate actual years/tiles and preserve aspect ratios.

- [x] Add tests for a rectangular single-pair preview, one-node/zero-edge graph, masks with gaps, absent optional audits, changed artifact hashes and coordinate-unit labels. Assert preview paths exist and legends use actual years.
- [x] Run output tests to red.
- [x] Validate coverage, connectivity, geometry, lineage, feature ranges and file hashes per actual run. Render only observed comparisons, use affine-based image coordinates and explicit units, and keep general reports distinct from reproduction reports. Add GeoJSON exports only when a valid CRS conversion is available.
- [x] Run output tests and visually inspect representative segmentation, annual and temporal overlays for shape/offset errors. Confirm no fixed strip names or tile_00002 requirements in the general runtime.
- [x] Commit `feat: validate and visualize arbitrary semantic temporal runs`.

### Task 12: Unified and standalone stage orchestration

**Files:** Create `runtime/runner.py`, `pipelines/semantic_temporal/run_pipeline.py`; modify `src/cli.py`, `src/pipeline_select.py`; test `tests/test_general_pipeline.py`.

**Interfaces:** Produce `run_stage(stage: str, method: str, output: Path, **options) -> dict` and `run_pipeline(segmentation: str, graph: str, change: str, output: Path, **options) -> dict`. Consumers use Tasks 1/2 options and stage state, Task 4 context and Tasks 5-11 artifacts. Keep the original selector public API as a delegation layer.

- [x] Add actual-artifact integration tests covering all eight old/new graph/segmentation and classical/GNN selections with usable fixture targets, separate stages versus one-pass, interrupted resume, one input year for segment/graph, single-image selection and a relocated saved run. Assert --images/--folders exclusion and chronological two-year requirement for change.
- [x] Run orchestration tests to red; use injected model predictions but real adapters/graph/temporal/GNN artifact processing.
- [x] Add mutually exclusive --images/--folders, explicit --years, metadata/source overrides, --pixel-size, --device, --gnn-checkpoint and configurable testable inference settings. Pass folders to both methods. Persist resolved contexts and verified outputs. Do not mark GNN insufficient-data results as completed predictions; expose their distinct status while preserving completed preceding stages.
- [x] Run integrations plus existing selector/folder/image tests. Verify standalone general --help and --until/--resume support; new commands no longer call reproduction implicitly. Check old reproduction-only --data semantics with explicit mode and saved-run migration.
- [x] Commit `feat: expose folder and single image modes for both pipelines`.

### Task 13: Model management, preflight and dependency instructions

**Files:** Create `runtime/preflight.py`, `pipelines/semantic_temporal/check_environment.py`, `scripts/download_pipeline_models.py`; modify `requirements.txt`, renamed package README, `.gitignore`; test `tests/test_pipeline_preflight.py`, `tests/test_model_downloads.py`.

**Interfaces:** Produce `check_environment(context: RunContext | None = None) -> dict` and `ensure_models(destination: Path, *, methods: tuple[str, ...] = ('old', 'new')) -> dict`. Downloader uses the verified pinned repository revisions and manifest hashes from the approved spec.

- [x] Add tests for missing libraries/models/metadata, wrong hashes, interrupted partial downloads, existing valid assets and preservation of existing differing files. Assert no implicit downloads during pipeline inference and no shared-parent-environment modification.
- [x] Run preflight/download tests to red using local HTTP fixtures and known tiny hashes rather than downloading models in every test.
- [x] Record verified publishers/revisions/size/hashes and CPU/GPU capability. Reuse the already downloaded 4 semantic assets and existing old assets. Add the missing general runtime requirements (NetworkX, Transformers, Ultralytics and safetensors) while keeping existing old requirements. General runtime uses Rasterio/Shapely and requires no osgeo; reproduction preflight separately checks GDAL and documents its exact environment. Keep custom GNN acquisition as training/inference rather than inventing a downloadable universal checkpoint.
- [x] Run preflight, fresh hash verification and actual local model loads. Check old checkpoint compatibility through existing backends; report any import incompatibility precisely. Verify all downloads/reports/cache directories are ignored.
- [x] Commit `feat: check and acquire pipeline model assets`.

### Task 14: Documentation, full validation and review

**Files:** Modify `Readme.md`, `docs/pipeline-options.md`, `RUN_TWO_YEAR_PIPELINE.txt`, `pipelines/semantic_temporal/README.md`; create `docs/pipeline-validation.md`; update relevant tests and the plan checkboxes with actual outcomes.

**Interfaces:** Deliver documented working commands and reproducible test evidence for the final implementation.

- [x] Run required new tests, then the full existing suite with real checkpoint smoke tests marked separately. Run no further broad repeats unless failures/new edits justify them.
- [x] Execute a real small 2017/2023 single-pair run with new models, then segmentation-only followed by graph/change continuation with only --output/method supplied. Execute a small multi-tile run with different years/counts and old-method regressions. Use fresh outputs and record counts, masks, unit checks, failures and runtime; actual input-image crops must retain metadata offsets.
- [x] Document both independent pipelines, all supported combinations, folder and single-pair commands, standalone commands, metadata/scale requirements, resume/override rules, reproduction mode and GNN data/checkpoint requirements. Repair malformed existing README command examples and PowerShell continuations.
- [x] Self-review requirement-to-test coverage and obtain the independent review appropriate to the selected execution method. Fix any demonstrated defects and rerun only affected checks. Original-study reproduction remains explicitly unverified if original inputs are unavailable.
- [x] Commit `docs: document and verify general purpose pipeline usage`; use finishing-a-development-branch workflow to present integration choices and disclose any remaining blocked checks.

## Planning-time assets and handoff

The new segmentation assets are already downloaded into `checkpoints/semantic_temporal/`; all four hashes match the original reference and both models completed small CPU inference. Their report is `checkpoints/semantic_temporal/download_report.json`. At planning time, no runtime code had been changed yet. The approved spec is committed as `749f176`.

Recommend native execution in this session because the normalization, adapters and stage orchestration share closely coupled interfaces. Follow executing-plans for implementation and arrange the final review permitted by that workflow. Subagent-driven execution remains an available user choice; do not launch implementation agents before that execution choice and written-plan approval.

## Execution evidence

Tasks 1-13 were implemented and committed in the approved isolated branch. The Windows ledger retained individual RED/GREEN evidence; equivalent behavior tests and the existing stage suites cover the contracts. Task 14 adds native real new/old pair execution, unequal counts, direct/rotated/geographic inputs, current-image registration, resumed-stage equivalence and independent review corrections. Exact original-study reproduction remains unverified because original inputs and GDAL are absent. Final checks and integration status are recorded in docs/pipeline-validation.md and docs/pipeline-review.md.
