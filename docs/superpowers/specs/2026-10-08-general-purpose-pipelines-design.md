# General-purpose SatelliteGraph pipelines

Date: 2026-10-08

## Objective and approved direction

Keep two distinct pipelines in SatelliteGraph: the existing SAM 2.1/OpenEarthMap
pipeline and a general-purpose SegFormer/Delineate Anything semantic-temporal
pipeline. Both must support any valid pair of years, arbitrary tile counts and
dimensions, folder processing, and one-image-versus-one-image comparison. Fix
stage option inheritance. Download and verify the required segmentation model
assets. Preserve the original experiment as an explicit reproduction mode.

The user approved the conceptual design and added these requirements:

- Image folders must contain metadata locating images within their original
  source image. Do not invent a layout from filenames.
- Download all needed checkpoints, retaining existing usable checkpoints.
- Rename `pipelines/experiment_2013_2023` professionally.
- Preserve the old pipeline and expose the two methods independently.
- Include a one-image-per-year mode, not only whole folders.

This specification concretizes that approved direction. Implementation starts
after review of this written specification and the written implementation plan.

## Current findings

The working tree was clean when inspected. The current selector supports only
old/old/classical, new/new/classical, and new/new/gnn. New stages delegate to a
frozen experiment runner. Study-specific assumptions occur in 25 pipeline
scripts, including literal years, seven-tile lists, strip names, 512/1024 pixel
dimensions, 1.5/0.75 metre grids, historic registration samples, audit verdicts,
reference output hashes, and seven-fold training loops.

The option bug originates in CLI defaults: omitted options are still passed as
default paths or false booleans, and overwrite saved run options. This affects
custom data/checkpoint paths, skip-input-check, registration, configuration and
progress flags. Tests currently replace actual selectable runners with fakes;
they do not exercise this inheritance regression.

The local `../.venv` has Python 3.11 and CPU-only PyTorch. SegFormer, Ultralytics,
Rasterio and most runtime dependencies are installed. GDAL Python bindings and
pytest are absent. The old segmentation checkpoints exist. The two new model
weights are absent. The supplied 2017/2023 folders have placement metadata and
source GeoTIFFs; metadata records alpha, valid extents and original x0/y0.

## Package and execution structure

Rename the new package to `pipelines/semantic_temporal`. Update imports,
subprocess paths, documentation, tests and ignore rules together. Keep existing
algorithm and reference snapshots available in this package for reproduction;
avoid silently changing published experiment semantics.

Expose a general `run_pipeline.py` runner in the renamed package, backed by
small modules for run context, input normalization, staging, validation and
inference. Extract reusable algorithms as necessary; do not generate modified
Python source through textual year substitutions. The existing
`run_experiment.py` continues to run the fixed published study explicitly.

Keep `src/stage2`, `src/refinement`, `src/stage3`, `src/stage4`, `src/stage5` and
the old folder/image commands available. The shared selector routes old/new
choices and persists a common run description. Method choices affect algorithms,
not which years or filenames are accepted.

## Input contract

Support RGB/RGBA PNG tiles with placement metadata and georeferenced RGB
GeoTIFFs. Reject unreadable images and duplicate or ambiguous metadata before
loading models. Discover tile names dynamically and match each selected PNG to
exactly one metadata row. Single-image mode selects only the requested image
even when its directory contains other images.

The minimum placement CSV has `tile_name`, `x0`, `y0`; dimensions are read from
the image and checked against any declared `width`, `height`, `x1`, `y1`.
Validate integer offsets, valid extents and alpha masks. Metadata describes
placement in an original source, not segmentation output coordinates. Retain
source identity, offsets, transforms and input hashes in every exported stage.

Two coordinate modes are supported:

1. Georeferenced: a source GeoTIFF or complete CRS and affine metadata defines
   each original grid. Explicit source/metadata arguments override discovery.
   Reject multiple ambiguous source rasters. With a source raster, check tile
   pixels against their recorded windows. Transform different yearly grids to
   a shared projected metric grid and save transformation/registration evidence.
2. Original source pixels: PNG placement metadata without map georeferencing
   permits source-coordinate annual layouts and individual/tile-pair graphs.
   Record units as pixels. Two source grids are not assumed identical; require
   sufficient image registration or an explicitly supplied grid transform for
   comparison. Metre-based algorithms require supplied physical pixel size or
   verified georeferencing. Never label pixel distances as metres or issue
   georeferenced exports without an actual CRS.

Raster inputs carry location internally. A PNG selected in single-image mode
requires its placement row, or an explicit equivalent affine/source-grid
description. The earlier allowance for PNG-only processing without any placement
metadata is superseded by the user's latest requirement.

Require two distinct chronological years for comparison; allow folder names
unrelated to years when `--years` is supplied. A single-year segmentation or
graph stage need not load or fabricate a second year. Discover overlapping
coverage geometrically rather than assuming identical filenames or tile counts.
Unmatched coverage is reported as unobserved, not classified as disappearance.

Preserve valid-data masks through reprojection, segmentation, graph generation
and comparison. Transparent/nodata pixels never become fabricated observations.
Partial tiles, non-square dimensions, gaps and overlapping windows are valid.

## General new pipeline

Build one explicit run context containing years, tile list, coordinate frame,
resolution, paths, model hashes, segmentation settings, registration and output
schema. Pass it to every stage; avoid study-specific global constants.

Prepare inputs from their actual extent and transforms. Registration uses
current imagery or supplied control points; historic shift samples apply only
in reproduction mode. Registration confidence and residual gates remain active.
Provide clear diagnostics when alignment cannot be established.

Retain the new segmentation methods: multi-scale SegFormer, Delineate Anything,
class-specific region construction and refinement. Make inference dimensions
dynamic. Derive metric areas, lengths and raster smoothing kernels from the
actual output affine. Spatial heuristics are documented model settings, not
claims of landscape-independent accuracy. Respect device=auto/cpu/cuda, with
auto falling back to CPU and explicit unavailable CUDA producing a clear error.

Build per-tile spatial graphs and annual combined graphs, using metadata for
placement and evidence for reconciling objects across overlap. Allocate combined
layouts from actual extents, check memory before large allocations and use
windowed staging where needed. Edge/feature tensors retain class and lineage.

Temporal layers compare validated shared coverage. Use actual year values in
records, paths and figures. Run classical labels before GNN when those labels
provide its training targets. Figures and audits enumerate actual tiles rather
than requiring tile_00002 or a fixed strip name.

General validation checks coverage, geometry, lineage, adjacency, feature schema
and saved hashes for this run. It does not compare arbitrary imagery with thesis
metrics or import historic human judgements. Reproduction mode alone performs
the original input/output comparisons.

## Interoperability

Introduce explicit segmentation/graph adapters to enable old/new combinations.
Do not imply interoperability by simply removing the selector's refusal checks.

Normalize class IDs, valid-data masks, region IDs, confidence, coordinates and
lineage into a versioned intermediate contract. Preserve old unknown regions as
unknown; never invent a known label. When the new graph requires connected
regions, split multi-part objects into connected components with a mapping back
to their original identity. Compute feature tensors from actual geometry,
imagery and probabilities, marking unavailable evidence explicitly.

The old graph adapter exports the existing Stage 3/4 contract without applying
old refinement a second time to already-refined new segments. The GNN adapter
validates feature schema and target definitions before execution. Every accepted
method combination must have an integration test showing real converted
artifacts, not only successful dispatch.

## GNN training and inference

Make the tile/group list and all year-dependent target selection dynamic.
Preserve the study's building-change target as a documented default; do not
claim that its binary predictor recognizes every land-cover change type.
Classical outputs retain their existing broader supported event types.

For training datasets with at least three independent groups and adequate
positive/negative examples, use dynamic grouped train/validation/test folds.
Overlapping tiles of the same underlying observations must not leak across
evaluation groups. Record fold membership and target provenance.

For a single pair or smaller datasets, support inference with a compatible
saved GNN checkpoint. If training is requested, permit clearly identified
training-only results when there are usable targets but no independent test
groups; do not publish invented held-out metrics. With no usable targets or
only one target class, report `insufficient_training_data` with counts and a
remedy; never silently substitute a random network or classical probabilities.

Save trained weights, preprocessing statistics, feature schema, class/target
definitions, seed and configuration for reuse. Inference checks compatibility.
There is no identified universal pretrained checkpoint for the custom relational
GNN; it is trained from this project's graphs. Download pretrained segmentation
assets, and create GNN checkpoints only through actual training when data permit.

Human audit data are optional. Missing/empty audits produce an explicit
`not_available` audit result and never crash the run. Keep independent audit
metrics distinct from agreement with classical pseudo-labels.

## Option persistence and resume semantics

Distinguish omitted arguments from explicit values using None/unset CLI
defaults. Resolve options in this order: explicit invocation, saved run,
initial-run defaults. Support explicit inverse boolean switches where needed.
Resolve paths relative to the invocation project root once, then save absolute
paths. Preserve custom data, models, configuration, registration and flags
between segment, graph and change invocations.

Persist effective options before execution. Stage failures include the actual
subprocess status and log location. A stale completed ledger must not override
a new failing process. Verification differences in reproduction mode have
explicit statuses and are not silently misrepresented as exact reproduction.

Fingerprint input data, models and stage settings. Refuse reuse after a changed
upstream input unless dependent stages are explicitly invalidated/recomputed.
Failed stages resume using verified completed artifacts; never overwrite frozen
outputs silently. Add migration handling for saved runs referencing the former
package path or checkpoint directory.

## Proposed user-facing commands

Folder mode:

```powershell
python -m src pipeline run --segmentation new --graph new --change classical `
  --folders tiles17_23/2017 tiles17_23/2023 --years 2017 2023 `
  --register-images --output artifacts/runs/semantic_2017_2023
```

Single pair:

```powershell
python -m src pipeline run --segmentation new --graph new --change classical `
  --images tiles17_23/2017/tile_00000.png tiles17_23/2023/tile_00000.png `
  --years 2017 2023 --register-images --output artifacts/runs/semantic_pair
```

`--images` and `--folders` are mutually exclusive. Metadata/source discovery
and explicit `--metadata`/`--source-rasters` work in both modes. Subsequent stage
commands need only method and output unless overriding a setting. The standalone
new runner exposes the same input/context semantics. Reproduction is a separate
explicit command under the renamed package.

## Checkpoints and environment

Reuse existing SAM 2.1 Hiera-Large and FasterSeg architecture/weights. Verify
their load compatibility and record their hashes without unnecessary downloads.
Download only missing or invalid assets to ignored local checkpoint directories;
never commit model binaries. Use partial files and publish final names only
after size/hash validation. Preserve existing differing files rather than
overwriting them silently. Store publisher, repository revision, URL, length and
SHA-256 in a local asset report.

Verified new publishers and exact revisions:

- `MykolaL/DelineateAnything`, revision
  `46f98c05ece1d57047e36952a8b440583de38f0d`, file
  `DelineateAnythingv2.pt`, 124747297 bytes, SHA-256
  `46700b8a279b07922953a11adaeb5e658d9a2384b6334c8e0a3090886218915a`.
- `boehnen/satlens-segformer`, revision
  `19cdd879e535d76e7fcbbd1e8b54dac9c1da90d6`, files config.json,
  preprocessor_config.json, model.safetensors. The weights are 14891980 bytes,
  SHA-256 `68f1c70396d3e1b5a9b752408cadbafa90c655da816564ce7d9646b112392ac0`.
  Validate all three against the shipped input manifest.

Sources: https://huggingface.co/MykolaL/DelineateAnything and
https://huggingface.co/boehnen/satlens-segformer . Both weight hashes match the
project's original reference manifest. Additional Delineate variants and the
unused SAM v1 model are not required downloads.

Use `checkpoints/semantic_temporal/` for new assets. Register model paths in the
general context; the reproduction runner accepts this directory explicitly.
CPU execution is supported. Do not automatically replace the user's existing
PyTorch or install a system CUDA toolkit. Add necessary runtime requirements
for the general runner and tests, choosing a workspace-local environment if
installation would modify the shared parent environment. Retain the exact
historic environment specification solely for reproduction.

Minimize dependency changes: reuse Rasterio/Fiona/Shapely for new geospatial I/O
when possible; if extracted algorithms still require osgeo, install compatible
GDAL bindings in an isolated environment and verify imports before claiming the
pipeline is runnable. Do not treat an installed package listing as an import or
model-load test. A preflight command reports missing models, libraries, metadata
and coordinate information before expensive work.

## Acceptance and verification

- Option regression: explicit custom paths and true/false flags persist across
  separate CLI processes; explicit overrides take effect; incompatible input
  changes are detected.
- Real normalized fixtures: arbitrary folder names, years other than 2013/2023,
  1/2/3+ tile pairs, unequal year tile counts, different tile names, rectangular
  and partial tiles, source offsets, overlap, gaps and nodata.
- Metadata rejection: missing/duplicate rows, invalid extents, ambiguous raster,
  pixel/window disagreement and unsupported coordinate comparisons fail early.
- One-image mode processes only selected images and uses their original offset;
  it never discovers and processes the rest of either folder.
- Test old/new adapter artifact contracts and each supported combination.
- Compare staged execution with one-pass execution using deterministic small
  fixtures. Test interrupted/resumed runs and fingerprint invalidation.
- GNN tests cover dynamic groups, absent audits, one-class/empty targets,
  training-only status, saved weights and incompatible feature schemas.
- Run existing old folder/image/refinement/graph tests to detect regressions.
- Parse/import checks for renamed runners and updated CLI help/documentation.
- Verify downloaded hashes and load all required actual segmentation models.
- Run a real small single-pair smoke test with downloaded new models; run a
  small multi-tile case when resources permit. Record actual results and runtime.
- Full published reproduction is only claimed when its original inputs are
  available and output verification has completed; those inputs are currently
  absent and cannot be manufactured from the 2017 folders.

## Delivery

Deliver both working pipeline choices, folder and single-pair commands,
standalone semantic-temporal methods, resumable stages, converted artifact
contracts, downloaded verified assets, installation/preflight instructions,
updated documentation and validation evidence. Distinguish completed checks
from any unavailable original-study verification. Do not claim universal
segmentation accuracy or meaningful GNN evaluation without supporting data.
