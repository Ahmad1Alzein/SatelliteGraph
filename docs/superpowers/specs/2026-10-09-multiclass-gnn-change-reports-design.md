# Multiclass GNN change reports and image matching

Date: 2026-10-09

Status: Conversational design approved; written specification awaiting review.

## Intent and approved constraints

Continue the project through meaningful, inspectable change detection, using the
GNN phase already supplied with the new pipeline. Produce one PNG containing
before/after crops of every reported change, with highlights and change types.
Also produce PNGs explaining matching between the two images. Include every
known semantic class, every ordered class transition, and every supported
observable change category. Use the supplied 2013 and 2023 samples for validation.
Execute natively, as previously selected by the user. Preserve both existing
pipelines and arbitrary-folder and one-image-versus-one-image support.

The full known-class registry is: 1 building, 2 road, 3 tree, 4 grass_shrub,
5 cropland, 6 bare_soil, 7 paved_developed, 8 water. Class 0 unknown and 255
nodata remain explicit unassessed observations, never confident land classes.
Every node appears in output tables and coverage maps, including ineligible nodes.
Vegetation summaries aggregate classes 3, 4, and 5 without removing individual
class predictions or conversions.

"All possible" means exhaustive coverage of this registry and the observable
taxonomy below. It is not a guarantee that every physical change is detectable
from two images. An apparent class change does not establish its cause: seasonal
differences must not be described as deforestation, flooding, development, or
permanent conversion without additional evidence.

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
and `runtime/gnn.py`. Extend these modules. Restore the original distinction
between forward/backward primary and secondary temporal messages and their
temporal edge evidence, rather than collapsing all temporal links into one
relation. Keep the RGNN message-passing equation and training approach; do not
substitute another GNN architecture or an unrelated pretrained detector.
Retain the general implementation's training-only normalization and protected
spatial evaluation groups. Do not restore historic hard-coded years, tile IDs,
audit verdicts, or experimental evaluation scores.

The frozen `pipeline/` implementation remains unchanged and runnable separately.
Its module-level paths and optional GDAL imports are not dependencies of the
general runtime. Existing single-output RGNN checkpoints remain explicitly
legacy building-construction checkpoints; they cannot score new targets.

Preserve the source-level sequence in `run_tile_pipeline.py` and
`prepare_pair_tile.py` (registration and paired preprocessing), `build_graph.py`
(spatial relations), `temporal_layer.py` (correspondence and primary components),
`build_stgraph.py` (joint graph), `stgraph_change.py` (two-sided evidence and
derived conversions), `build_mosaic.py` (ownership and pixel lineage), and
`gnn_change.py` with its visualization companion (predictions and crop galleries).
The original binary task includes later new/extended buildings and earlier
non-building nodes built over. Carry that two-sided evidence attribution into
per-class gain/loss tasks.

Original assumptions such as suppressing low-vegetation/bare-soil observations,
reporting tree/road candidates as unverified, fixed 0.75 metre pixels, global
normalization, and hard-coded seven-fold evaluation are not universal rules.
Use current physical scale, explicit uncertainty, and protected evaluation.

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

## Paired preprocessing from the original phase

Adapt `pipeline/harmonize.py` behind persisted `--harmonize-images` /
`--no-harmonize-images` options, default off for compatibility and explicitly on
for the 2013/2023 demonstration. After registration, match earlier radiometry
to the later reference and blur whichever image is sharper, using the original
0.02 relative sharpness tolerance, maximum sigma 3.0, and 14 bisection steps.
Do not sharpen a blurred image. Estimate shared statistics on common observed
coverage, exclude nodata/shadows, and apply consistent transforms across owned
tiles so overlaps do not receive contradictory preprocessing.

Retain registered original RGB separately from model-input RGB. Use original
radiometry for final crop displays and shadow eligibility, harmonized imagery
for model inference, and record the mapping, sharpness, and blur for each date.
If harmonization modifies the later reference, segment that modified image as
well, as in `prepare_pair_tile.py`. Changing preprocessing invalidates upstream
artifacts; do not pair old segmentation with newly harmonized imagery.

## Evidence, event, and target contracts

Extend metric semantic evidence to symmetric gain/loss rules for all eight known
classes, preserving multiscale presence/absence, building roof compensation,
ground registration for other classes, and two-sided pixel lineage.
Vegetation-family evidence sums tree, grass/shrub, and crop probabilities only
for the additional aggregate summary. Tree-to-grass is a per-class conversion,
but is not net vegetation-family loss or gain.

Use full observed masks, known semantic evidence, physical area thresholds,
boundary tolerance, and shadow exclusion. Separate ambiguous or weak evidence
from reliable training labels. Apply eligibility checks consistently to raster
events, object records, training targets, predictions, and report summaries.
Unknown pixels, uncovered areas, alignment uncertainty, and absence of a graph
match must not become confident change or persistence labels.

Record a versioned change policy with these initial conservative defaults:
presence probability 0.6 (retain the original building-specific 0.5 presence
rule), multiscale absence maximum 0.2, absence-window width 3.75 metres,
original grayscale shadow cutoff 20, reliable node confidence 0.6, and observed
and known-counterpart fractions 0.95. Minimum connected candidate area is
40 square metres for building, 100 for road, 150 for tree/grass_shrub/cropland,
and 100 for bare_soil/paved_developed/water. Building high-confidence gain keeps
the original 100-square-metre reliable connected evidence requirement.
For other classes a positive pseudo target needs at least its minimum connected
evidence area and 30 percent of the eligible node area; weaker regions remain
candidates. Persistence negatives require supported same-class correspondence,
at least 95 percent stable known evidence, and no qualifying changed component.
These are heuristic defaults, not audited accuracy for the added classes.
Accept an optional `--change-policy PATH` JSON overriding named thresholds,
validate finite ranges/units, save the resolved policy, and bind its bytes and
effective values to the change-stage fingerprint.

Introduce sixteen ordered tasks, in ascending class-ID order and gain before
loss: `<class_name>_gain` and `<class_name>_loss` for each known class. Gain tasks
apply to later-year nodes of that class; loss tasks apply to earlier-year nodes
of that class. Both dates' nodes participate in graph context.
Building gain covers construction/extension;
retain those more specific types only when image/geometry evidence supports
them. A GNN-only positive is a class/direction prediction, not proof of a
specific construction mechanism. Preserve unknown targets as masked values.

Support every ordered transition between distinct known classes: 8 times 7,
or 56 conversion categories. A conversion is one canonical event linking its
source-class loss and destination-class gain, not two independently counted
changes. Preserve both task scores and their availability. Derive conversion
geometry from common observed evidence pixels, including partial conversions
too small to create a primary temporal link. Use a derived conversion edge as
in `stgraph_change.py`; do not promote correspondence confidence because a
change candidate exists.

Include the complete eight-by-eight class transition matrix, persistence on the
diagonal, zero-count supported transitions, and unknown/uncovered observations
separately. Areas count each owned pixel once. Grass/shrub, crop, and bare-soil
observations remain visible even where seasonal sensitivity prevents reliable
training labels. Mark these candidates `season_sensitive`, retain them for
review, and mask them from training unless reviewed evidence supports the target.
Absence of a trained head does not remove a class from reports.

Accept optional `--reviewed-targets PATH` CSV with `gid,task,label,confidence,source`
for user-reviewed evidence on the current graph. Require known IDs/tasks, labels
0 or 1, confidence in [0,1], and a nonempty source; reject duplicate/conflicting
rows. Reviewed labels may resolve seasonal ambiguity but cannot override missing
coverage or unknown semantics. Save provenance and hash-bind the file to change
settings; do not import historic audit verdicts automatically.

Persist task-specific target rows with node ID, task, binary label, confidence,
year, evidence source, and eligibility reason. Persistence negatives require
positive evidence of stability, not merely failure to detect a positive event.
Keep candidate geometry and evidence provenance independent of GNN confidence.
Persist stable event IDs and linked source/target node IDs so report rows can be
traced back to scientific artifacts without double-counting shared tile pixels.

## Observable change taxonomy

Apply this taxonomy to every known class where geometry and evidence allow it.
Persist kind, class IDs, source/target nodes, correspondence kind, evidence
status, and geometry, with a stable event ID and readable display name.

| Kind | Required interpretation |
| --- | --- |
| Appearance / gain | Later class evidence appears; inspect former coverage/class before naming its cause. |
| Disappearance / loss | Earlier class evidence disappears within observed coverage. |
| Expansion | A matched class component has supported added area beyond boundary tolerance. |
| Contraction | A matched class component has supported lost area beyond boundary tolerance. |
| Reconfiguration | A matched component has supported gains and losses; retain both geometries under one parent event. |
| Class conversion | A supported ordered source-to-destination class transition, including partial conversion. |
| Split correspondence | One primary earlier component links to several later components; physical change needs separate image evidence. |
| Merge correspondence | Several primary earlier components link to one later component; physical change needs separate image evidence. |
| Many-to-many correspondence | Primary matching is structurally complex; show members and any separately supported changes. |
| Connectivity / neighborhood change | Derived class-contiguity or neighborhood change; state its relation and supporting footprint evidence. |
| Persistent | Sufficient positive evidence of no material class or footprint change. |
| Uncertain / unassessed | Unknown semantics, weak evidence, coverage gaps, shadow, seasonal sensitivity, alignment conflict, or insufficient model support. |

For buildings, retain new building, extension, contraction, and demolition
candidate names. For roads, expose added/removed areas, widening/narrowing, and
reconfiguration where matched road geometry supports those names. Vegetation
has individual category changes/conversions plus its aggregate summary. For
water, show mapped-water appearance/disappearance and expansion/contraction;
do not infer flood, drought, or reservoir operations. Bare soil and
paved/developed land receive the same class-generic categories.

Physical connectivity derives from evidence-supported contiguous class masks
within observed regions; proximity edges alone cannot establish a road
connection. Neighborhood metrics describe graph context and cannot prove
physical change without footprint evidence. Pure segmentation split/merge
differences remain structural observations and cannot supply positive targets.
Apparent displacement is reconfiguration or alignment uncertainty, not a claim
that a building, road, or water body physically relocated.

Semantic conversion and geometry descriptors can coexist on one event. Do not
count a bare-soil-to-building conversion again as an independent appearance.
Specific subtypes are image/graph-derived, as in the original binary GNN phase;
the GNN supplies relevant class gain/loss scores. Do not label derived subtypes
as independently learned GNN subtype predictions.

## Spatial and temporal graph relations

Preserve touching edges for all node types and the original building-near edges
(25 metre edge-to-edge distance, six selected neighbors per building). Add a
separate `context_near` relation for non-touching mixed-class or non-building
known-class neighbors within 25 metres, six selected neighbors per node. Exclude
unknown nodes from new proximity edges, deduplicate undirected pairs, and do not
replace original building-near neighbors with unrelated nearer land nodes.
Unknown nodes can retain observed touching/overlap context.

Preserve primary identity/containment links and secondary overlaps. Compute
one-to-one, one-to-many, many-to-one, many-to-many, and unmatched components
using primary edges only, as in `temporal_layer.py`.

RGNN relation names are `touches`, `near`, `context_near`, `t_primary_fwd`,
`t_primary_bwd`, `t_overlap_fwd`, and `t_overlap_bwd`. Spatial edges are
bidirectional. Temporal forward messages run earlier to later; backward messages
reverse direction, swap source/target fractions, and negate directional roof
shifts. Restore the original ten-dimensional temporal evidence: IoU, source and
target fractions, overlap score, shared physical area, class agreement,
confidence ordinal, roof-frame flag, and roof-shift x/y. Pad nine-dimensional
spatial attributes to ten dimensions and record relation-specific feature names.
Add an explicit earlier/later node-role feature without fixed calendar years.

Keep derived change-conversion edges in the annotated result graph, separate
from RGNN input relations: feeding labels back as input edges would leak the
answer. Version graph contracts, relation definitions, and tensor indices together.

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
definition, graph-relation profile, normalization, head availability, thresholds,
settings, and hashes.
Use weights-only loading and exact compatibility checks. Keep legacy inference
explicitly limited to its original building-gain definition. A legacy checkpoint
cannot silently initialize other classes or loss predictions. Legacy inference
requires its original compatible relation/feature profile; reject incompatible
new graphs clearly rather than silently collapsing relations or padding weights.

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

Also write `results/temporal_match_types.png` with the original phase's
one-to-one, split, merge, many-to-many, and unmatched coloring on both RGB images.
Provide matching zoom galleries using current component geometry rather than
fixed pixel locations. Include derived conversions in a separate annotated view
so they cannot be confused with correspondence links.

For folders, also write named per-coverage-pair matching PNGs where the overview
would be unreadable. Coverage and coordinates determine pairs, not filenames.
Use stable identifiers; retain the full match table alongside any displayed
subset. Display at most 300 strongest, spatially distributed links per matching
panel and disclose the number omitted. Scientific matching remains uncapped.

## Complete visual outputs

Provide named PNGs in the result index, with actual source years and units:

- `segmentation_overview.png`: original RGB, semantic colors, and region
  boundaries per year; all eight classes and unknown/nodata in the legend.
- `spatial_graph.png`: RGB node/region overlays and colored touching,
  building-near, and context-near relations, with class and edge counts.
- Registration, temporal correspondence, match-type, conversion, and zoom PNGs
  described above.
- `gnn_predictions.png`: eligible class-specific decisions; unavailable and
  ineligible regions distinguished from predicted negatives.
- `gnn_probabilities.png`: before-class-loss and after-class-gain panels for all
  eight classes, thresholds, task status, and unavailable-head indicators.
- `class_transitions.png`: exhaustive known-class transition count/area matrix,
  persistence, and separate unknown/uncovered coverage totals.
- `change_report.png`: the single comprehensive cropped-change contact sheet.

Use overviews plus named tile/coverage-pair views where needed for readability.
Tables retain every node and edge even when a panel explicitly samples links.
All-masked or insufficient-data runs still write visual status outputs; never
display untrained scores or hide unavailable classes from legends.

## One combined change PNG

Write `results/change_report.png` as the primary result. Start with before/after
RGB overviews, a legend, counts by change type and evidence status, and per-task
GNN training/inference status for all sixteen tasks. Follow with a contact sheet
containing every reported change as one numbered before/after crop pair, grouped
by source/destination class and kind. Include supported changes and uncertain
candidates with distinct status markings. Pure matching differences belong in a
separate structural-observation section and cannot inflate physical-change totals.

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
artifacts. Version graph relations and change/report processing in their stage
fingerprints so resume rebuilds affected graph/change stages without rerunning
unchanged segmentation. A model or target change must not unnecessarily rebuild
spatial graphs. Registration changes
correctly invalidate prepared segmentation and require a fresh run.

Use existing NumPy, PyTorch, Rasterio, OpenCV, Shapely, SciPy, pandas, and Pillow
dependencies. No additional model downloads are required for this extension.

## Verification and real execution

Parameterize tests over all eight classes and both gain/loss directions. Test
all 56 ordered conversions, category swaps with stable aggregate vegetation,
appearance/disappearance, expansion/contraction, reconfiguration, road widening
and narrowing, split/merge/many-to-many correspondence, connectivity and
neighborhood observations, unchanged regions, seasonal sensitivity,
partial coverage, unknown evidence, shadows, tile overlaps, masked heads,
single-class support, mixed task availability, checkpoint round trips, legacy
restrictions, temporal direction and relation-specific features, new proximity
edges, derived-edge leakage exclusion, fold isolation, and resume invalidation.
Test reviewed-target validation and seasonal overrides, change-policy validation,
masked paired harmonization, consistent overlap preprocessing, unchanged source
RGB displays, and correct upstream invalidation when preprocessing changes.

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
