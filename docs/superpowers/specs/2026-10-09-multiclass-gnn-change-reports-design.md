# Multiclass GNN change reports and image matching

Date: 2026-10-09

Status: Conversational design approved; written specification awaiting review.

## Intent and approved constraints

Continue the project through meaningful, inspectable change detection, using the
GNN phase already supplied with the new pipeline. Produce one PNG containing
before/after crops of every reported change, with highlights and change types.
Also produce PNGs explaining matching between the two images. Include buildings,
roads, and vegetation. Use the supplied 2013 and 2023 samples for real validation.
Execute natively, as previously selected by the user. Preserve both existing
pipelines and arbitrary-folder and one-image-versus-one-image support.

Vegetation means trees, grass/shrubs, and cropland. An apparent vegetation change
does not establish its cause: seasonal differences must not be described as
deforestation or permanent land conversion without additional evidence.

## Existing phase to extend

The original phase is `pipelines/semantic_temporal/pipeline/gnn_change.py`.
It provides the plain-PyTorch RGNN equation with self-node, neighbor-node, and
edge-feature projections, relation-wise mean aggregation, two 64-dimensional
hidden layers, dropout 0.3, Adam learning rate 0.005, weight decay 0.0005,
class-weighted binary cross entropy, deterministic CPU training, pseudo labels,
and validation-selected thresholds. Its visualization companion is
`pipeline/validation/gnn_predictions_tile_viz.py`; its audit montage companion is
`pipeline/validation/stgraph_change_audit_tile.py`.

The general implementation already adapts this phase in `runtime/gnn_model.py`
and `runtime/gnn.py`. Extend these modules and their current touches/near/temporal
relations. Keep the RGNN message-passing equation and training approach; do not
substitute another GNN architecture or an unrelated pretrained detector.
Retain the general implementation's training-only normalization and protected
spatial evaluation groups. Do not restore historic hard-coded years, tile IDs,
audit verdicts, or experimental evaluation scores.

The frozen `pipeline/` implementation remains unchanged and runnable separately.
Its module-level paths and optional GDAL imports are not dependencies of the
general runtime. Existing single-output RGNN checkpoints remain explicitly
legacy building-construction checkpoints; they cannot score new targets.

## Verified sample inventory and placement

- `tiles17_23/2013`: seven 512 by 512 RGBA PNGs and placement `metadata.csv`.
- `tiles17_23/2023`: seven corresponding PNGs, placement CSV, and
  `selection_2023.tif` with georeferencing.
- All fourteen PNG hashes match the supplied experiment input manifest.
- All seven 2023 PNGs pass the general input loader's source-window check.
- The original 2013 source GeoTIFF is unavailable at its recorded path.
- Each 2013 tile's saved `artifacts/annual_2013_updated/stage2/.../13_refinement/
  tile_metadata.json` supplies its CRS and affine transform. All seven agree
  with the placement CSV and the saved source transform: EPSG:32636, 1.5 metre
  pixels, source affine `(1.5, 0, 740574, 0, -1.5, 3744507)`.

Prepare a separate, derived 2013 placement CSV containing the supported
`tile_affine_a` through `tile_affine_f` and `target_crs` columns. Record hashes of
the PNGs, original CSV, saved metadata, and derived CSV in a preparation audit.
Do not overwrite source metadata or fabricate a replacement original raster.
Use the supported metadata override when running the general pipeline.

Estimate current-image registration for this pair; do not transplant historic
shift samples. Save accepted control pairs, residuals, coordinate units, and
registration method for visualization and reproducibility. Saved georeferencing
is placement evidence, not an independent registration accuracy assessment.

## Evidence, event, and target contracts

Extend metric semantic evidence to symmetric gain/loss rules for road and
vegetation families. Retain building construction, extension, and loss evidence.
Vegetation-family evidence sums tree, grass/shrub, and crop probabilities before
testing presence or absence, so grass-to-tree alone is not vegetation gain.
Record the contributing category separately.

Use full observed masks, known semantic evidence, physical area thresholds,
boundary tolerance, and shadow exclusion. Separate ambiguous or weak evidence
from reliable training labels. Apply eligibility checks consistently to raster
events, object records, training targets, predictions, and report summaries.
Unknown pixels, uncovered areas, alignment uncertainty, and absence of a graph
match must not become confident change or persistence labels.

Introduce six ordered output tasks: `building_gain`, `building_loss`,
`road_gain`, `road_loss`, `vegetation_gain`, `vegetation_loss`.
Gain tasks apply to later-year nodes of the relevant family; loss tasks apply to
earlier-year nodes of that family. Building gain covers construction/extension;
retain those more specific types only when image/geometry evidence supports
them. A GNN-only positive is a family/direction prediction, not proof of a
specific construction mechanism. Preserve unknown targets as masked values.

Persist task-specific target rows with node ID, task, binary label, confidence,
year, evidence source, and eligibility reason. Persistence negatives require
positive evidence of stability, not merely failure to detect a positive event.
Keep candidate geometry and evidence provenance independent of GNN confidence.
Persist stable event IDs and linked source/target node IDs so report rows can be
traced back to scientific artifacts without double-counting shared tile pixels.

## GNN extension and checkpoint behavior

Retain the existing RGNN encoder and add task-specific binary output heads.
Each task uses its own eligibility mask, class weights, support counts, threshold,
and status. Train a task only when it has both reliable positive and negative
labels. Unsupported heads are excluded from the loss and cannot emit usable
probabilities or predictions through incidental shared-encoder training.

Preserve deterministic CPU behavior and existing configurable epochs/hidden size.
Select thresholds and model state using validation evidence where independent
groups support it. Each evaluated task needs both labels in its train, validation,
and test partitions; otherwise report that task as training-only or insufficient.
Exclude messages across held-out groups and fit normalization on training data.
Pseudo-label agreement is never presented as independent human accuracy.

Version checkpoint metadata with task names/order, feature schema, eligibility
definition, normalization, head availability, thresholds, settings, and hashes.
Use weights-only loading and exact compatibility checks. Keep legacy inference
explicitly limited to its original building-gain definition. A legacy checkpoint
cannot silently initialize road, vegetation, or loss predictions.

Produce long-form predictions keyed by node ID and task, including probability,
decision, eligibility reason, model status, and provenance. An insufficient task
does not prevent supported tasks or the candidate report from completing.
If every task is insufficient, still produce reports showing that outcome.
No additional pretrained checkpoint can supply these project-specific heads;
they are trained from available reliable targets or loaded from compatible runs.

## Image matching PNGs

Create `results/registration_matches.png` showing the two original image views
with accepted landmark pairs, match counts, residual units, and method. A saved
control-point registration uses those controls. Georeference-only runs must
state that feature registration was not performed; do not draw invented matches.

Create `results/temporal_matches.png` showing corresponding object boundaries
and correspondence links on the prepared before/after RGB imagery. Distinguish
primary identity/containment links from secondary overlaps and show unmatched
regions separately from changed regions. Matching is not a change prediction.
Include year labels, a legend, registration status, and displayed/total counts.

For folders, also write named per-coverage-pair matching PNGs where the overview
would be unreadable. Coverage and coordinates determine pairs, not filenames.
Use stable identifiers; retain the full match table alongside any displayed
subset. Display at most 300 strongest, spatially distributed links per matching
panel and disclose the number omitted. Scientific matching remains uncapped.

## One combined change PNG

Write `results/change_report.png` as the primary result. Start with before/after
RGB overviews, a legend, counts by change type and evidence status, and per-task
GNN training/inference status. Follow with a contact sheet containing every
reported change as one numbered before/after crop pair.

Use the same geographic crop bounds, scale, and padding for both dates. Preserve
aspect ratio, show invalid areas distinctly, and draw the relevant geometry on
both panels without replacing the underlying pixels. Each pair shows years,
event ID, change type, area and units, evidence status, and applicable GNN score
or an explicit unavailable status. Keep rule candidates and GNN predictions
visually distinguishable. A probability is a model score, not calibrated accuracy.

Use clipped raster windows and annual ownership masks; do not wrap array slices
at image edges or show overlap duplicates as distinct changes. Consolidate only
duplicate representations of the same event, keeping component lineage. Do not
merge nearby unrelated events or silently drop events to fit the image.

Default crop panels are 224 by 224 pixels with two event pairs per row. The
combined PNG has a 128-million-pixel budget. If all events do not fit, stop report
generation with the required size and an actionable budget/subset instruction;
preserve completed scientific artifacts. Do not silently truncate or substitute
one PNG per event. Empty results still produce an informative combined PNG.

Write `results/index.json` linking images, events, matches, predictions, task
statuses, and counts. Hash-bind report outputs to their scientific inputs and
include them in validation. Print stage progress, task support counts, and the
primary PNG path in the CLI. Do not require users to discover numbered previews.

## Integration and compatibility

Generate matching outputs after graph/change construction and the final report
after classical evidence and optional GNN inference. Both classical and GNN runs
receive the combined report, with their evidence sources stated.

Preserve existing stage commands, saved input options, adapters, and segmentation
artifacts. Version change/report processing in the downstream fingerprint so
resume recomputes affected stages instead of accepting obsolete reports. A model
or target change must not unnecessarily rerun segmentation. Registration changes
correctly invalidate prepared segmentation and require a fresh run.

Use existing NumPy, PyTorch, Rasterio, OpenCV, Shapely, SciPy, pandas, and Pillow
dependencies. No additional model downloads are required for this extension.

## Verification and real execution

Test symmetric family changes, vegetation category swaps, unchanged regions,
partial coverage, unknown evidence, shadows, tile overlaps, masked heads,
single-class support, mixed task availability, checkpoint round trips, legacy
restrictions, fold isolation, and resume invalidation.

Test crop placement with known spatial landmarks, edge clipping, unequal tile
layouts, valid masks, geometry holes, non-square images, readable labels, empty
reports, absent registration landmarks, and report-budget failure. Verify event
counts and IDs across raster/vector/table/PNG index artifacts. Open and visually
inspect rendered PNGs; file existence alone is insufficient.

Run the project's complete test suite after the extension. Then execute the real
2013-versus-2023 tile_00002 pair, followed by the seven-tile folders, using new
output directories, audited placement, current-image registration, and the
existing local model assets. The full prepared strip is approximately 6.73
million pixels; explicitly use an eight-million-pixel comparison budget rather
than silently bypassing the current four-million default. Record actual support,
training statuses, timing, and generated PNG paths. Do not claim a trained head
or measured accuracy when the observed data cannot support it.
