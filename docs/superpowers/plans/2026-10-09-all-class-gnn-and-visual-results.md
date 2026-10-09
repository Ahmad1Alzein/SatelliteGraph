# All-class RGNN and visual results implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing RGNN to all eight known classes and observable change categories, and deliver independently inspectable stage PNGs, matching PNGs, and one combined before/after change image.

**Architecture:** Keep the original RGNN encoder, paired-preprocessing approach, directional temporal relations, two-sided image evidence, and pixel lineage. Add sixteen task heads, an exhaustive class-conversion/event catalog, and focused renderers sharing coordinate-safe raster sampling and plain-language legends. Preserve original `src` image/folder pipelines and the frozen experiment; integrate through the existing general runtime and CLI.

**Tech Stack:** Existing Python environment, NumPy, PyTorch, Rasterio, OpenCV, Shapely, SciPy, pandas, Pillow, and pytest. No new dependencies or pretrained model downloads.

**Spec:** `docs/superpowers/specs/2026-10-09-multiclass-gnn-change-reports-design.md` (approved 2026-10-09, including subsequent wording and separate-stage PNG clarifications).

**Execution:** Native, as selected by the user. Implement tasks in order in an isolated development worktree using executing-plans, preserving access to root input/model assets. Follow that skill's final-review requirements. Do not push or merge automatically; the prior local-main merge authorization concerned the previous completed branch.

## Global Constraints

- Keep the RGNN message-passing equation and training approach; do not substitute another GNN architecture or an unrelated pretrained detector.
- The full known-class registry is: 1 building, 2 road, 3 tree, 4 grass_shrub, 5 cropland, 6 bare_soil, 7 paved_developed, 8 water.
- Class 0 unknown and 255 nodata remain explicit unassessed observations, never confident land classes.
- Introduce sixteen ordered tasks, in ascending class-ID order and gain before loss: `<class_name>_gain` and `<class_name>_loss` for each known class.
- Support every ordered transition between distinct known classes: 8 times 7, or 56 conversion categories.
- A conversion is one canonical event linking its source-class loss and destination-class gain, not two independently counted changes.
- The frozen `pipeline/` implementation remains unchanged and runnable separately.
- Every node appears in output tables and coverage maps, including ineligible nodes.
- Train a task only when it has both reliable positive and negative labels.
- Pseudo-label agreement is never presented as independent human accuracy.
- Keep derived change-conversion edges in the annotated result graph, separate from RGNN input relations.
- Default crop panels are 224 by 224 pixels with two event pairs per row.
- The combined PNG has a 128-million-pixel budget.
- Standalone `pipeline segment` and `pipeline graph` commands produce their PNGs without needing `pipeline change`.
- Use the supplied 2013 and 2023 samples for validation; arbitrary folders, chronological years, unequal tile layouts, and single-image pairs remain supported.
- No additional model downloads are required for this extension.

## Review Focus

1. Partial conversion can be smaller than the temporal-link threshold: report the pixel-supported conversion once without promoting a weak link or leaking derived labels into GNN input (Tasks 3, 5).
2. A trained shared encoder must not expose random scores from an unsupported head; render unavailable scores distinctly from zero probability (Tasks 6, 9).
3. Overlapping, rotated, rectangular, or partly transparent tiles must retain matching crop coordinates, consistent preprocessing, and ownership without duplicate events (Tasks 2, 4, 7, 8).
4. Tiny images, empty graphs, and long class/change names must retain complete readable legends outside the image canvas (Tasks 7, 9).
5. Moving a run, changing an external checkpoint/label file at the same path, or failing the PNG budget must not leave stale reports looking current (Tasks 6, 10).

## File map and execution conventions

New runtime modules: `change_contracts.py` (class/tasks/policy/labels), `preprocessing.py` (paired model inputs), `comparison.py` (owned common-frame sampling), `all_class_evidence.py` (evidence masks), `change_catalog.py` (events/targets/structural observations), `gnn_training.py` (task training/evaluation), and `reports/{common,stages,matching,changes}.py` (rendering). Add `reports/__init__.py` with the owning renderer task.

Modify current runtime `registration.py`, `segmentation/__init__.py`, `adapters.py`, `graphs.py`, `temporal.py`, `gnn_model.py`, `gnn.py`, `visualization.py`, `storage.py`, and `runner.py` only where their task requires it. Wire options and invalidation in `src/cli.py`, `src/pipeline_select.py`, and `src/pipeline_state.py`. Add the sample preparation script and documentation in Task 11.

Below, `R` means the exact directory `pipelines/semantic_temporal/runtime/`.
`$python` means `..\.venv\Scripts\python.exe` when running at the root checkout;
from an isolated worktree use the same absolute Python path. Test commands use
`scripts/run_project_tests.py`, which loads workspace-local pytest dependencies.

Every task records its red failure, green result, and commit. Use genuine tiny
rasters and finite probability fixtures for algorithm tests; model stubs are
allowed only outside real inference verification. Do not weaken confidence or
coverage gates to make the 2013/2023 example train. Do not edit frozen scripts to
make tests pass. Preserve previously completed source/model data and runs.

### Task 1: Class, task, policy, and display contracts

**Files:** Create `R/change_contracts.py`, `tests/test_all_class_contracts.py`; read `R/contracts.py` without changing its segmentation schema unnecessarily.

**Interfaces:** Produce `TASKS: tuple[str, ...]`, `RELATIONS: tuple[str, ...]`, `CLASS_LABELS: dict[int, str]`, `EDGE_LABELS: dict[str, str]`, and `load_change_policy(path: Path | None) -> dict`. Policy keys are named probabilities, fractions, distances in metres, areas by class name, and shadow cutoff; reject unknown keys and invalid ranges. Export `change_label(event: dict) -> str` and `status_label(status: str) -> str` for all renderers.

- [ ] **Write failing contract tests.** In `test_registry_covers_every_class_and_direction`, assert `len(TASKS) == 16`, `TASKS[0] == 'building_gain'`, `TASKS[-1] == 'water_loss'`, and `len(conversions) == 56`. Known IDs are exactly `range(1, 9)`. Add `test_invalid_policy_is_rejected` for malformed thresholds, NaN, unknown class keys, and negative areas, using `pytest.raises(ValueError)`.
- [ ] **Run red:** `& $python scripts/run_project_tests.py tests/test_all_class_contracts.py -q`; expect missing contract implementation.
- [ ] **Implement contracts.** Copy exact policy defaults, relation order, and plain-language labels from the spec. Use known classes only for trainable tasks. Keep evidence status separate from display change kind and retain technical IDs in records.
- [ ] **Run green:** the contract tests and `tests/test_pipeline_options.py`; existing defaults remain valid.
- [ ] **Commit:** `feat: define all-class change and display contracts` with this task's files.

### Task 2: Saved controls and paired model preprocessing

**Files:** Create `R/preprocessing.py`, `tests/test_paired_preprocessing.py`; modify `R/registration.py`, `R/segmentation/__init__.py`, `R/adapters.py`, `src/cli.py`, `src/pipeline_select.py`, `src/pipeline_state.py`.

**Interfaces:** Produce `prepare_model_inputs(context: RunContext) -> RunContext`. Returned options contain a `prepared_inputs` mapping by tile key with original RGB, model RGB, valid-mask paths, and grid. Extend `prepare_tile(tile, context)` to consume this mapping when present while preserving its `(rgb, valid, grid)` return contract. Persist registration controls with tile keys, native/world coordinates, inlier flags, residuals, and units in `prepared/registration_controls.json`.

- [ ] **Write failing tests.** Use a known brightness transform and blur on overlapping RGBA tiles. Assert common pixel values remain equal across extracted overlaps, invalid pixels cannot affect fitted statistics, original RGB is unchanged, and model inputs use the modified reference when reference blur is necessary. Assert omitted/inverse harmonization flags inherit/reset correctly and changing them rejects upstream resume. For a known affine registration, assert saved control IDs, inliers, and residual units match the measured transform.
  Required cases include `test_masked_pair_preserves_overlap` with `assert np.array_equal(left_overlap, right_overlap)`, `test_original_rgb_is_preserved` with `assert np.array_equal(original_before, original_after)`, and `test_saved_controls_match_registration` with `assert controls['units'] == 'metres'`.
- [ ] **Run red:** `& $python scripts/run_project_tests.py tests/test_paired_preprocessing.py tests/test_semantic_registration.py -q`; expect missing preprocessing or saved controls.
- [ ] **Implement paired preprocessing.** Adapt histogram matching and blur-to-sharpness from frozen `harmonize.py`, using only common observed, non-shadow pixels for statistics and masked filtering. Apply one consistent per-year transform on the owned prepared mosaic before tile extraction. Keep original RGB for display/shadow checks. Default harmonization off; use 0.02 tolerance, sigma cap 3.0, 14 bisection steps. Add persisted BooleanOptionalAction flags and bind enabled preprocessing to upstream fingerprints; omitted/false harmonization must remain hash-equivalent to existing unharmonized runs. Hash-verify cached preparation before reuse. Both old/new segmentation adapters consume the same prepared model pixels.
- [ ] **Run green:** preprocessing, registration, option inheritance, and segmentation tests; unchanged default processing remains equivalent.
- [ ] **Commit:** `feat: preserve paired preprocessing and measured registration controls` with the listed files.

### Task 3: Spatial context and original directional temporal relations

**Files:** Modify `R/graphs.py`, `R/temporal.py`; create `tests/test_all_class_relations.py`; extend `tests/test_semantic_graphs.py`.

**Interfaces:** Keep `build_tile_graph`, `combine_year_graphs`, and `build_temporal_graph` signatures. Graph records gain relation-profile version and primary-component metadata. Produce `temporal_components.json` and relation-aware temporal tensors: raw node features plus `is_later`, ten-dimensional edge attributes, and `edge_type` indices in Task 1's seven-relation order. Record feature names separately for spatial and temporal relations.

- [ ] **Write failing tests.** Assert non-building and mixed-class non-touching neighbors get `context_near`, original building-near neighbors stay intact, unknown nodes get no proximity edges, and duplicate undirected pairs are absent. Assert forward/backward temporal fractions and roof shifts reverse correctly; primary components classify split/merge/many-to-many without secondary links chaining unrelated components. Assert derived conversion edges never appear in training tensors.
  In `test_directional_temporal_features`, assert `backward_fraction_source == forward_fraction_target` and `backward_shift_x == -forward_shift_x`. In `test_change_edges_are_not_model_inputs`, assert `'change_conversion' not in input_relations`.
- [ ] **Run red:** `& $python scripts/run_project_tests.py tests/test_all_class_relations.py tests/test_semantic_graphs.py -q`.
- [ ] **Implement relations.** Preserve 25 metre/six-neighbor building proximity; add separate 25 metre/six-neighbor context proximity. Annual spatial tensors retain their nine spatial attributes; temporal assembly pads them to ten and constructs original temporal evidence directly from canonical links. Primary identity/containment and secondary overlap use separate forward/backward channels. Calculate components on primary edges only. Bind profile/version and role feature into the temporal feature schema. Preserve old graph selection and its configured matching policy.
- [ ] **Run green:** relation/graph tests and existing temporal correspondence tests, including known roof displacement and zero-edge graphs.
- [ ] **Commit:** `feat: restore directional temporal evidence and all-class spatial context`.

### Task 4: Owned comparison frame and symmetric class evidence

**Files:** Create `R/comparison.py`, `R/all_class_evidence.py`, `tests/test_all_class_evidence.py`; modify the evidence-loading portions of `R/temporal.py` and reuse `R/change_evidence.py` building logic without editing frozen sources.

**Interfaces:** `load_comparison(graph: TemporalArtifact, context: RunContext) -> dict` returns `grid`, `years`, and per-year `labels`, `classes`, `valid`, `rgb_original`, `rgb_model`, `shape_probabilities`, `max_probabilities`, and node-ID lookup. `detect_evidence(frame: dict, policy: dict) -> dict` returns class/direction masks keyed by Task 1 tasks, evidence/status masks, vegetation aggregate masks, and an encoded transition raster. Encoding: `16 * source_class + target_class` for known observations, 0 for unknown/unassessed, 65535 for unobserved pixels; save the encoding in metadata.

- [ ] **Write failing tests.** Parameterize all eight classes and both directions with probability fixtures above/below presence/absence limits. Assert all 56 conversions encode correctly, tree-to-grass leaves aggregate vegetation stable, non-observation/shadow cannot become positive evidence, roof compensation affects buildings only, and overlapping pixels are owned once. Assert a rotated rectangular tile samples a known world landmark correctly and a comparison over budget raises before allocating evidence tensors.
  In `test_known_transition_encoding`, assert `transition[y, x] == 16 * source_class + target_class` for the 56 cases. In `test_missing_observations_are_masked`, assert `not evidence['masks'][task][~common_valid].any()` for every task.
- [ ] **Run red:** `& $python scripts/run_project_tests.py tests/test_all_class_evidence.py -q`.
- [ ] **Implement masks.** Use recorded shared transforms, observed masks, and annual ownership; reproject windows when a simple slice is insufficient. Read saved probabilities for all canonical classes. Old confidence-only artifacts obtain independent multiscale model evidence through the existing local model loader. Preserve original building evidence and use class-specific minimum areas and physical margins from policy. Exclude original-image shadows; retain seasonal-sensitive candidates with separate flags. Allocate masks lazily and obey comparison budget.
- [ ] **Run green:** evidence tests plus existing invalid-border, unknown-coverage, and tree-gain regression tests adapted to the new evidence contract.
- [ ] **Commit:** `feat: derive two-sided image evidence for every known class`.

### Task 5: Event taxonomy, conversions, structural observations, and targets

**Files:** Create `R/change_catalog.py`, `tests/test_all_class_catalog.py`; modify `R/temporal.py` and its existing tests.

**Interfaces:** `build_change_catalog(graph: TemporalArtifact, frame: dict, evidence: dict, policy: dict, *, reviewed_targets: Path | None = None) -> dict` returns `events`, `structural_observations`, `node_changes`, `targets`, `transition_matrix`, `coverage`, and `years`. Target rows have `gid,task,label,confidence,source,year,eligible,reason`. Event rows have stable ID, classes, kind/descriptors, geometry, area/units, source/target node IDs, related tasks, evidence status, and component lineage. `detect_classical(graph, context) -> dict` remains the stage entry point and persists the catalog.

- [ ] **Write failing tests.** Assert each of 56 conversions is one event with two linked task records, including partial conversion lacking a primary graph edge. Assert new/expanded/contracted/reconfigured geometry types and road widening/narrowing use evidence, not unmatched IDs. Assert pure split/merge differences supply no positive training targets; connectivity uses observed class masks rather than proximity links. Assert unchanged labels need stability evidence; seasonal/unknown/coverage-limited rows stay masked. Assert reviewed rows reject duplicates, unknown IDs/tasks, invalid confidence, and cannot override unknown coverage.
  In `test_partial_conversion_is_one_event`, assert `len(catalog['events']) == 1` and `len(catalog['events'][0]['related_tasks']) == 2`. In `test_structural_difference_is_not_a_positive`, assert `not any(row['label'] == 1 for row in catalog['targets'])` on an unchanged-image split fixture.
- [ ] **Run red:** `& $python scripts/run_project_tests.py tests/test_all_class_catalog.py -q`.
- [ ] **Implement catalog.** Associate owned evidence components to annual nodes using pixel lineage, use primary matching for structural descriptors, and create separate derived conversion edges without rematching. Persist `classical_changes.json`, `training_targets.csv`, `class_transition_map.tif`, `transition_matrix.json`, and `change_graph.json`. Keep the former method-specific result under `legacy_classical_changes.json`; preserve Stage 5 configuration behavior when producing the old-graph legacy result. The unified report uses `engine='all_class_image_evidence'`, with `legacy_engine` and `legacy_report` pointing to preserved provenance. Do not change standalone old `src` stage behavior.
- [ ] **Run green:** catalog and existing classical tests; update building-only assertions to select the building task explicitly, while retaining equivalent legacy-engine regression coverage.
- [ ] **Commit:** `feat: catalog all-class changes and reliable task targets`.

### Task 6: Existing RGNN with sixteen supported output heads

**Files:** Modify `R/gnn_model.py`, `R/gnn.py`; create `R/gnn_training.py`, `tests/test_all_class_gnn.py`; extend `tests/test_semantic_gnn.py`.

**Interfaces:** Extend `RGNN(..., tasks: tuple[str, ...] | None = None)` so the default keeps the original single-head state-dict layout and requested tasks add ordered heads to the same encoder. Add `train_tasks(x, relations, labels, eligibility, groups, settings) -> dict` in `gnn_training.py`; labels/eligibility are node-by-task arrays in Task 1 order, labels -1 are masked, and groups are one identifier per node. Preserve `run_gnn(graph, context, *, checkpoint=None) -> dict` as orchestrator. Long-form predictions use `gid,task,probability,prediction,eligible,reason,status` with null score and decision -1 for unavailable outputs. Checkpoint metadata includes target-schema version, task order, relation profile, features, normalization, head support/status, thresholds, settings, and SHA-256. Fit edge normalization separately for each relation on training edges; use mean 0/std 1 when a relation has none.

- [ ] **Write failing tests.** Train tiny deterministic fixtures with support for multiple classes; assert unsupported heads have null scores, supported heads round-trip exactly, and target eligibility matches class and year direction. Assert single-class support returns insufficient data per head, mixed supported/unsupported tasks can finish, shared training excludes masked labels, and external checkpoint tampering fails verification. Assert fold normalization is train-only and no held-out message edges remain. Assert original default RGNN weights/outputs remain identical for the original edge dimension/relations, while incompatible legacy graph/checkpoint profiles fail clearly.
  In `test_unsupported_head_has_no_score`, assert `predictions.loc[unavailable, 'probability'].isna().all()` and `(predictions.loc[unavailable, 'prediction'] == -1).all()`. In `test_checkpoint_roundtrip`, use `np.testing.assert_allclose(first_scores, loaded_scores)` for deployment scores.
- [ ] **Run red:** `& $python scripts/run_project_tests.py tests/test_all_class_gnn.py -q`.
- [ ] **Implement training.** Keep two layers, hidden 64, dropout 0.3, Adam 0.005/0.0005, seed 0, CPU deterministic aggregation, and configurable epochs/hidden size. Class-weight each supported task and mask unsupported losses. Retain grouped normalization/isolation; validation-select each supported task's threshold and best state per fold, retaining per-head snapshots when best epochs differ. Require both classes in each evaluated task's relevant partitions; otherwise mark training-only. Persist out-of-fold scores separately in `evaluation_predictions.csv`. As in the current general runtime, train a deployment model on all reliable targets for compatible saved inference, using fixed threshold 0.5 when there is no independent deployment validation; never describe that threshold as validation-calibrated. Mark deployment scores separately from held-out scores and prefer held-out scores in current-run views where available. Strictly load compatible weights-only models; legacy inference requires compatible saved legacy graph schemas. Produce per-task/overall status, support counts, predictions, and hash-bound artifacts.
- [ ] **Run green:** all new/existing GNN tests and original-encoder numerical equality check. Update binary-only tests to keep explicit legacy coverage and add all-class equivalents.
- [ ] **Commit:** `feat: extend the existing RGNN to sixteen class-change tasks`.

### Task 7: Coordinate-safe rendering and separate stage verification PNGs

**Files:** Create `R/reports/__init__.py`, `R/reports/common.py`, `R/reports/stages.py`, `tests/test_stage_result_pngs.py`; modify `R/visualization.py`.

**Interfaces:** `sample_rgb(frame: dict, year: int, bounds: tuple[float, float, float, float], size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]` returns RGB and validity using one shared world window. `render_stage_results(context: RunContext, stage: str) -> dict[str, Path]` writes stage-specific PNGs/indexes. Shared helpers draw geometry including holes, preserve aspect ratio, wrap legends in an expanded canvas, and use portable fonts with a readable fallback.

- [ ] **Write failing tests.** Run genuine tiny segmentation/graph fixtures through rendering. Assert one PNG per image plus annual graph PNGs, actual years/units, segment-type and relevant edge legends, zero-edge status, and original RGB landmark placement. Test rectangular/tiny images, long labels, polygon holes, alpha/nodata display, and different method/year paths without collisions. Verify changing stages does not overwrite earlier-stage PNGs.
  In `test_separate_stage_pngs_are_retained`, assert `all(path.is_file() for path in segmentation_paths)` before and after graph rendering. In `test_tiny_image_legend_is_outside_pixels`, assert `rendered.height > image.height` and `legend_bounds[1] >= image_bounds[3]` for a below-image legend.
- [ ] **Run red:** `& $python scripts/run_project_tests.py tests/test_stage_result_pngs.py -q`.
- [ ] **Implement stage PNGs.** Use the exact spec paths and plain-language labels; retain full-resolution scientific rasters. Draw spatial edges by color and line style with nodes/segments over original RGB. Class colors remain consistent. Produce separate per-stage indexes so updating the root index cannot invalidate earlier-stage image manifests. Avoid fixed square assumptions or legends pasted over image content.
- [ ] **Run green:** stage PNG tests, then open small/rectangular and empty-graph outputs for visual inspection; fix clipped text or invisible edges.
- [ ] **Commit:** `feat: render independent segmentation and graph verification images`.

### Task 8: Image landmarks, temporal matches, and conversion PNGs

**Files:** Create `R/reports/matching.py`, `tests/test_matching_pngs.py`; use Task 2 saved controls, Task 3 components, and Task 7 sampling.

**Interfaces:** `render_matching_results(graph: TemporalArtifact, context: RunContext, frame: dict, catalog: dict) -> dict[str, Path]` writes registration matches, temporal matches, match-type map, derived-conversion view, and component zoom galleries, plus per-coverage-pair views for folders and a full match table.

- [ ] **Write failing tests.** Assert line endpoints correspond to known control/world points on before/after panels, invalid areas are distinct, all matching component types appear correctly, and conversion edges do not masquerade as primary matches. Assert georeference-only runs explicitly show no feature registration, unequal filenames pair by coverage, and over-300-link displays disclose total/displayed counts without truncating scientific tables.
  In `test_link_display_cap_preserves_full_table`, assert `displayed_count <= 300`, `table_count == total_count`, and `omitted_count == total_count - displayed_count`. In `test_control_endpoint_projection`, use `np.testing.assert_allclose(rendered_endpoint, expected_endpoint, atol=1)`.
- [ ] **Run red:** `& $python scripts/run_project_tests.py tests/test_matching_pngs.py -q`.
- [ ] **Implement matching PNGs.** Follow original temporal overview/match-type/zoom conventions with current bounds and actual years. Use `Share a border`, `Nearby buildings`, `Nearby areas`, `Main match across years`, `Partial overlap across years`, and `Class change` only where relevant. Draw direction with arrows/year labels. Select at most 300 strongest spatially distributed links per panel, retaining the complete table and explicit omitted count. Sample crop bounds from actual components, not original fixed coordinates.
- [ ] **Run green:** matching tests and open rendered registration, temporal, and conversion examples to verify endpoints, legends, and status text.
- [ ] **Commit:** `feat: visualize image and object matching with plain-language legends`.

### Task 9: All-class GNN views, transition matrix, and single change gallery

**Files:** Create `R/reports/changes.py`, `tests/test_change_report_pngs.py`; consume Task 5 catalog, Task 6 predictions, and Task 7 sampling.

**Interfaces:** `render_change_results(graph: TemporalArtifact, context: RunContext, frame: dict, catalog: dict, gnn_result: dict | None) -> dict[str, Path]`. Produce spec-named predictions, probabilities, class-transition matrix, and `change_report.png`, with an event-to-crop index. `--max-report-pixels` controls the default 128,000,000-pixel budget.

- [ ] **Write failing tests.** Assert the combined PNG includes every canonical event exactly once with paired same-window crops and both years. Check an edge-of-image landmark has matching position without slice wrapping, unsupported GNN scores render unavailable rather than zero, all eight classes remain visible in all-masked results, and the 8-by-8 matrix retains zero-count categories. Test empty results, long change names, seasonal badges, structural-only sections, compound descriptors without double counts, and required-size error above the report budget.
  In `test_combined_report_has_every_event_once`, assert `set(crop_ids) == set(event_ids)` and `len(crop_ids) == len(set(crop_ids))`. In `test_report_budget_fails_without_truncation`, require `pytest.raises(ValueError, match='required.*pixels')` and assert scientific catalog files still exist.
- [ ] **Run red:** `& $python scripts/run_project_tests.py tests/test_change_report_pngs.py -q`.
- [ ] **Implement reports.** Use two 224-pixel panels per event and two event pairs per row, with unchanged aspect ratios and separate original RGB validity. Show a summary/legend, all sixteen task statuses, class and readable change names, area/units, evidence badge, and available GNN scores. Use clear examples from the spec, including New building, Building grew, Road got wider, Less tree cover, and Water expanded. Keep uncertain and structural observations visibly distinct. Include every event or fail explicitly at the budget; do not silently drop crops or substitute per-event files.
- [ ] **Run green:** report tests and open multi-class, no-change, long-label, and insufficient-training images at readable scale.
- [ ] **Commit:** `feat: render complete all-class change results in one image`.

### Task 10: Native orchestration, manifests, resume, and CLI output

**Files:** Modify `R/runner.py`, `R/visualization.py`, `R/storage.py`, `R/context.py`, `src/cli.py`, `src/pipeline_select.py`, `src/pipeline_state.py`; create `tests/test_all_class_pipeline.py`; extend existing option/state/general-pipeline tests.

**Interfaces:** Add persisted `change_policy`, `reviewed_targets`, and `max_report_pixels` options, with CLI flags from the spec and Task 9. Introduce `graph_fingerprint(options: dict) -> str` and `render_fingerprint(options: dict, stage: str) -> str` alongside `change_fingerprint`; each binds its own algorithm/profile and relevant values/files. Add `render_completed_stage(output: Path, stage: str, state: dict) -> dict` in the runner for verified render-only retries. Preserve existing public stage/run signatures. Produce root `results/index.json` pointing to stage indexes and scientific artifacts, with render status and primary PNG paths. Result manifests store scientific input hashes in a relocation-safe form, with external inputs explicitly identified.

- [ ] **Write failing integration tests.** For every old/new segmentation/graph combination and classical/GNN selection, assert standalone segmentation/graph PNGs exist before change and remain after change/resume. Assert stage-only options inherit, graph-profile changes rebuild graph/change only, target/checkpoint/policy changes rebuild change only, and harmonization changes reject stale upstream reuse. Assert moved-run manifests validate with relocated internal references, same-path external file replacement invalidates results, and image-budget failure leaves scientific outputs verifiable with an explicit failed render status.
  In `test_change_only_override_keeps_segmentation`, assert `segmentation_hash_after == segmentation_hash_before` and `change_fingerprint_after != change_fingerprint_before`. In `test_render_failure_keeps_science_complete`, assert `state['stages']['change']['scientific_status'] == 'complete'` and `state['stages']['change']['render_status'] == 'failed'`.
- [ ] **Run red:** `& $python scripts/run_project_tests.py tests/test_all_class_pipeline.py -q`.
- [ ] **Implement orchestration.** Preparation precedes segmentation; graph and temporal assembly use completed verified artifacts; image evidence/catalog precede GNN; matching and final rendering follow supported scientific outputs. Persist stage/root indexes atomically, hash-bind PNGs/indexes and scientific inputs, and extend validation for new tensor/task/event schemas. Print progress, per-task support, unavailability explanations, and absolute result paths. Record `scientific_status` and `render_status` separately; render failure sets overall `status='render_failed'` while retaining verified scientific results. On `--resume`, unchanged verified science with failed/outdated rendering calls `render_completed_stage` only. Report-budget/label-style changes affect rendering, not inference. Do not accept stale pre-extension change reports as current.
- [ ] **Run green:** all new integration tests and existing pipeline/state/adapter tests. Confirm all eight method combinations still have valid scientific outputs and accurate statuses.
- [ ] **Commit:** `feat: integrate all-class results with native stages and safe resume`.

### Task 11: Audited 2013/2023 sample preparation and real demonstrations

**Files:** Create `scripts/prepare_2013_2023_samples.py`, `tests/test_sample_placement_recovery.py`; extend `docs/pipeline-validation.md`; generated data/logs/PNGs remain ignored under `artifacts/`.

**Interfaces:** `prepare_samples(project_root: Path, output: Path) -> dict` returns derived 2013/2023 metadata paths and a placement-audit JSON path. Script accepts `--output` and optional `--project-root`; it validates the known sample hashes before using saved 2013 transforms. It never overwrites source metadata or creates an original source raster.

- [ ] **Write failing tests.** Use sample-shaped metadata fixtures with a known manifest hash map. Assert every tile affine agrees with source affine plus its offset, wrong PNG hashes or inconsistent saved coordinates stop recovery, and sources remain byte-identical after preparation. Verify the produced CSV is accepted by general `inspect_inputs`.
  In `test_recovered_placement_matches_saved_source`, use `np.testing.assert_allclose(tuple(tile_affine)[:6], tuple(source_affine * Affine.translation(x0, y0))[:6])`. In `test_recovery_never_modifies_sources`, assert `source_hashes_after == source_hashes_before`.
- [ ] **Run red:** `& $python scripts/run_project_tests.py tests/test_sample_placement_recovery.py -q`.
- [ ] **Implement and run preparation.** For actual samples, require all fourteen image hashes to match the experiment manifest, validate all seven saved 2013 metadata records, and verify 2023 source-window pixels. Save derived CSV/audit under `artifacts/all_class_validation/inputs/`. Record original CSV/metadata and derived-file hashes.
- [ ] **Run real validation.** Check environment/assets, then execute new segmentation -> new graph -> GNN for tile_00002 and for all seven folders, with CPU, registration, harmonization, explicit metadata overrides, and `--max-comparison-pixels 8000000`. Use `artifacts/all_class_validation/pair/` and `artifacts/all_class_validation/folders/` beneath the root workspace, even when executing worktree code; supply absolute root input/model/project/output paths so results do not depend on a temporary checkout. Use normal inference defaults for final evidence. Open separate segmentation/graph PNGs, matching images, probability maps, and combined reports. Record actual support, unavailable tasks, timing, counts, paths, and validation; do not claim all heads trained if evidence is insufficient.
- [ ] **Commit:** `test: verify all-class outputs on the real 2013 and 2023 samples` with script/tests/documented evidence, excluding generated assets.

### Task 12: Documentation, complete tests, and native final review

**Files:** Update `docs/pipeline-options.md`, `docs/pipeline-validation.md`, `Readme.md`, `RUN_TWO_YEAR_PIPELINE.txt`, and `pipelines/semantic_temporal/README.md`; add `docs/all-class-change-results.md` for the plain-language legend and result examples. Preserve frozen-study descriptions as historical behavior.

**Interfaces:** Document the existing general and standalone entry points, new flags, exact result paths, class/change/edge meanings, legacy checkpoint compatibility, stage-resume behavior, reviewed-label format, and honest task statuses. Explain that specific subtypes are image/graph-derived while RGNN scores class gains/losses.

- [ ] **Update user instructions.** Provide PowerShell commands for sample preparation, one-image pair, full folders, separate stages, resume, and compatible checkpoint inference. Include plain-language edge/change examples and the separate-stage PNG locations. Distinguish candidate change evidence, GNN predictions, structural matches, and unknown coverage without requiring users to understand tensor identifiers.
- [ ] **Run complete checks:** `& $python scripts/run_project_tests.py tests -q --tb=short`, then `git diff --check`; require passing output and record exact counts. Re-run only checks justified by subsequent fixes. Verify frozen source files are unchanged and all seven model assets still pass existing preflight.
- [ ] **Perform final native review** using the executing-plans/requesting-code-review skills' required reviewer workflow. Review against all spec sections and five Review Focus cases, and inspect actual rendered artifacts. Fix material findings with regression tests and repeat affected checks.
- [ ] **Commit documentation and verified fixes.** Record final commit/test/run evidence. Do not report full independent accuracy, perfect recall, or universal subtype inference; report observed results and limitations.
- [ ] **Deliver concrete results.** Link primary combined PNG, separate-stage PNG directories, matching PNGs, and run instructions. State which tasks trained or lacked support and the current branch/integration status, then follow the finishing skill for integration choices.

## Plan self-review

- Scope coverage: Task 1 covers registry/wording/policy; 2 covers original preprocessing/controls; 3 covers graph relations/components; 4 covers all-class evidence; 5 covers taxonomy/targets/conversions; 6 covers the existing RGNN extension/checkpoints/evaluation; 7-9 cover every specified visual output; 10 covers orchestration/persistence; 11 covers real sample placement/execution; 12 covers documentation/full verification/review.
- Review Focus coverage: partial conversion and leakage in 3/5; unsupported head scores in 6/9; rotated/overlapping masked imagery in 2/4/7/8; small/empty/long-label canvases in 7/9; relocation, same-path inputs, and budget failure in 6/10.
- Interfaces use one Task 1 registry/order throughout; shared comparison sampling comes from 4/7; legacy model profile compatibility is explicit; relative report references survive relocation.
- Documentation and setup are folded into their deliverables; no independent scaffolding-only task is required.
- No product code changes are part of this plan-writing stage. Native execution begins after this written plan is reviewed and approved.
