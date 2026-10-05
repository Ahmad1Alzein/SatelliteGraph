# SatelliteGraph

SatelliteGraph turns satellite imagery into semantic instance segmentations
and tile-local adjacency graphs. The current workflow is:

1. Stage 1 divides a source raster into overlapping image tiles.
2. Stage 2 creates one complete semantic instance segmentation for a tile.
3. Stage 2R refines segment topology and replays reviewed corrections.
4. Stage 3 converts the refined segments into a graph for that tile.
5. Stage 4 combines any selected set of tile graphs in source coordinates and
   reconciles duplicate objects in their overlaps.
6. Stage 5 treats each complete Stage 4 graph as one year, converts it to shared
   world coordinates, and adds temporal region-correspondence and split/merge
   edges between years. It also records class-specific appearance/disappearance,
   construction/removal, expansion/contraction, and classification-uncertainty
   events. See [Stage 5 spatiotemporal graphs](docs/stage5-spatiotemporal-graph.md).
   For a professor-facing explanation of matching and every class-specific
   change edge, see [Temporal edge methodology](docs/temporal-edge-methodology-for-review.md).
   A thesis/article citation ledger is in [Research sources](docs/research-sources-for-thesis.md).

Stage 2 runs two models in one pipeline:

- **SAM 2.1 Hiera-Large** proposes detailed, class-agnostic segment geometry.
- **OpenEarthMap FasterSeg** supplies the named land-cover labels.
- Deterministic fusion resolves overlaps, fills uncovered valid pixels, and uses
  `unknown` when no OpenEarthMap class is confident enough.

The Stage 2 result is complete for one tile: every valid pixel belongs to
exactly one instance and has exactly one semantic value. Stage 2R temporarily
decomposes objects into four-connected pieces for evidence and replay, then
consolidates topology-only secondary pieces through 40 pixels back into their
original object identity before graph export. This prevents speckle from
creating hundreds of new nodes while preserving every pixel and immutable
Stage 2 lineage. A hard count budget also prevents larger topology-only pieces
from raising the published object count above Stage 2; only an explicit,
approved semantic split can add a corresponding allowance. It offers two
decision paths over the same deterministic proposals:

- `human_review` produces checksum-bound crops and accepts only explicit
  reviewer decisions for generic merge, absorb, and supplied semantic-split
  operations; and
- `automatic` runs unattended, approving only conservative same-class,
  interior merge/absorb proposals whose score **and** raw evidence gates all
  pass. It records every other pending proposal as an abstention and leaves it
  unchanged.

Both paths also replay the independently gated, at-most-40-pixel unknown
cleanup rules. Valid pixels are never deleted. Stage 3 creates one node per
refined segment and an undirected edge whenever two segments share a pixel
side. Corner-only contact is not adjacency.

See [Stage 2 full segmentation](docs/stage2-segmentation.md),
[Stage 2R refinement](docs/stage2-refinement.md), and
[Stage 3 graph building](docs/stage3-graph-building.md), and
[Stage 4 graph combination](docs/stage4-graph-combination.md) for their complete
artifact and metadata contracts. A presentation-oriented explanation of the
refinement decisions and similarity formulas is available in
[Refinement teacher discussion](docs/refinement-teacher-discussion.md).

Stages 2, 2R, and 3 operate on one tile at a time. Stage 4 performs the
cross-tile reconciliation and graph combination.

## Setup

The repository has one dependency file and one project configuration file:

```powershell
python -m pip install -r requirements.txt
```

Edit `config.yaml` to configure project paths, both models, preprocessing,
filtering, fusion, refinement, graph construction, visualization, and output. There is no
separate per-stage configuration and no Streamlit application.

The configured model assets are local and are not downloaded while the pipeline
runs:

```text
checkpoints/sam2/sam2.1_hiera_large.pt
checkpoints/openearthmap/fasterseg/arch_1.pt
checkpoints/openearthmap/fasterseg/weights1.pt
```

## Where to put the image

Put the source GeoTIFF at `data/raw/<source>.tif`. This location is controlled
by `project.paths.raw_directory` in `config.yaml`. Stage 1 prepares the two
artifacts that Stage 2 actually consumes:

```text
data/tiles/<source>/tile_XXXXX.png
data/metadata/<source>.csv
```

Run Stage 2 with the canonical tile ID `<source>__tile_XXXXX`; `run_stage2` and
the CLI take a `tile_id`, not an image path.

## Tile an image and optionally change its ground resolution

Keep the original pixel grid:

```powershell
python -m src stage1 tile --source data/raw/scene.tif --keep-resolution
```

Or change the pixel size to 1.5 metres before tiling:

```powershell
python -m src stage1 tile --source "D:/satellite graph/data/raw/image_2023.tif" `
  --data-directory "D:/satellite graph/data" --resolution 1.5 `
  --resolution-mode preserve --tile-size 512 --overlap 50
```

`--resolution` is the requested **metres per pixel**, not degrees. The default
`--resolution-mode preserve` resizes the full image in its original CRS and
preserves its footprint. It does not crop, filter tiles, or create reprojection
margins. The affine transform is scaled to the new dimensions. For geographic
sources, the requested metre size is approximate at the image centre and varies
with location; integer output dimensions also require slight rounding. The
manifest records the actual pixel-axis lengths in metres at the centre.

For an exact projected pixel size, explicitly use `--resolution-mode projected`
and optionally `--target-crs EPSG:32636`. This reprojects to a metre-based grid
and can introduce masked margins around the transformed footprint. It is a
different operation from resizing the original rectangular image.

The source must have valid georeferencing. Average resampling is the default for
downsampling continuous imagery; `--resampling` also accepts bilinear, cubic,
or nearest. This path supports uint8/uint16 RGB bands without intensity stretching.

The original GeoTIFF is preserved. Resampled output is named
`raw/image_2023_1p5m.tif`, tiles go to `tiles/image_2023_1p5m/`, and the CSV and
run manifest go to `metadata/`. Use `--name` for a different output identifier;
existing outputs are not overwritten. Streaming windows keep memory bounded
for large rasters.

The entire converted GeoTIFF is completed and closed **before** tiling starts.
By default, tiles are exactly 512×512 with stride 462 (50-pixel overlap). The
final row/column shift inward to meet the image boundary, so their overlap can
be larger than 50 pixels. This covers every resized image pixel without padding
or removing boundary tiles. The source must be at least one tile wide and high.
Exact 50-pixel overlap everywhere, full-size edge tiles, and no padding cannot
all be guaranteed for arbitrary image dimensions.

`--edge-mode pad` explicitly selects a regular grid with transparent padding.
PNG alpha preserves any pre-existing source nodata; missing source observations
are never invented. The CSV records exact offsets, `valid_width`, `valid_height`,
`alpha_band`, and `valid_pixel_count`. The manifest records the complete grid
offsets. Downstream placement uses these offsets, including the shifted final
tiles, and the converted raster's affine transform.

Defaults are in `project.tiling` in `config.yaml`: `target_resolution_m: null`
preserves the source resolution; set it to a positive value to resample.
`--keep-resolution` explicitly overrides that setting. Resolution changes alter
pixel detail, not the geographic footprint in preserve mode.

The CLI and Python API use the same tiling method for both steps:

```python
from src.io.tile_image import tile_image

result = tile_image(
    source_path="data/raw/scene.tif",
    data_directory="data",
    target_resolution=1.5,  # None skips conversion
    tile_size=512,
    overlap=50,
)
```

Legacy in-memory array calls retain their pixel-only behavior. Metre conversion
requires `source_path` because an array alone does not contain georeferencing.
The resampling and affine scaling follow the
[Rasterio resampling guidance](https://rasterio.readthedocs.io/en/stable/topics/resampling.html).

## Run Stage 2 for one tile

From Python, use the single public runner:

```python
from src.stage2 import run_stage2

result = run_stage2("scene__tile_00000")
print(result.output_directory)
print(result.segments)
```

Or use the single full-pipeline CLI command:

```powershell
python -m src stage2 run --tile-id scene__tile_00000
```

The runner shows a `tqdm` phase progress bar. Add `--no-progress` to the CLI
command when progress output is not wanted.

Both forms use `config.yaml` by default and write the final product to
`artifacts/stage2/<tile_id>/12_final/`. The returned `Stage2Result` includes the
individual masks and records, instance and semantic maps, semantic confidence,
per-pixel provenance, the valid-data mask, normalized tile metadata, and the
artifact paths.

The final records preserve tile-local pixels, source/global pixels, projected
coordinates, and WGS84 coordinates when the source raster contains usable
georeferencing. GeoTIFF and GeoJSON exports are included alongside the NumPy and
JSON artifacts.

## Refine the segmentation in either mode

Stage 2R always regenerates proposals from immutable `12_final` data, replays
the selected operations, validates the result, and writes
`artifacts/stage2/<tile_id>/13_refinement/`. `refinement.mode` defaults to
`human_review`, and a CLI mode flag can override it for one run.

For a tile where a person should inspect buildings, canopy fragments, and
other proposed corrections:

```powershell
python -m src refine propose --tile-id scene__tile_00000
# Inspect review/index.json and edit decisions.template.json.
python -m src refine apply `
  --tile-id scene__tile_00000 `
  --mode human_review `
  --decisions artifacts/stage2/scene__tile_00000/13_refinement/decisions.template.json
python -m src refine validate --tile-id scene__tile_00000
```

The template starts at `uncertain`. An omitted or all-`uncertain` document
authorizes no review-gated edit, although internal connectivity analysis,
micro-component identity consolidation, and the embedded unknown-cleanup rules
still run. Proposal mode writes
`review/index.json`, exact normalized masks/maps, and one
original/before/proposed crop per pending merge or absorb.

For one tile with no reviewer:

```powershell
python -m src refine auto --tile-id scene__tile_00000
python -m src refine validate --tile-id scene__tile_00000
```

`refine auto` is equivalent to `refine apply --mode automatic`; reviewed
decision files are intentionally not accepted in this mode. The
`automatic_selective_v1` policy can approve generic merges only for supported
same-class, non-border homogeneous regions, and generic absorbs only when the
small source and target have the same non-protected class. It protects
`road`, `agriculture land`, `buildings`, `tree`, and `unknown` from these
generic automatic operations. A high combined score is insufficient by
itself: exact four-neighbor adjacency, shared-boundary support, boundary and
region color agreement, class policy, source area where applicable, and all
configured thresholds must pass. Missing or weak evidence, a protected class,
a tile-border contact, or a proposal conflict produces `uncertain`, not a
forced edit. Color evidence in both modes must come from the Stage 2
preprocessed tile array or the original tile path; a rendered segmentation
overlay is never accepted as source-image evidence. This guarantees that human
and automatic runs generate the same proposal set. Without an authoritative
image, the run still completes conservatively but does not invent image-based
operations.

The separately replay-validated `micro-unknown-v1` and
`contextual-micro-v1` rules remain available in both modes. They are bounded to
unknown regions of at most 40 pixels, never delete valid pixels, and apply
their own OpenEarthMap, adjacency, border, and protected-object gates. This is
why a contextual unknown may be assigned to a road/building/field only under
its specialized safeguards even though generic automatic operations protect
those classes.

For hundreds of complete Stage 2 tiles, omit `--tile-id` to discover all tiles
with a `12_final/manifest.json` and use bounded workers:

```powershell
python -m src refine batch --mode automatic --workers 4
```

Repeat `--tile-id` to select an explicit subset. Automatic batch runs resume by
default: a completed, non-stale result is skipped only when its mode,
configuration hash, manifest, checksums, and loaded artifacts remain valid.
Use `--force` to recompute those compatible results. By default one failed tile
does not stop the others; the per-run failure ledger at
`artifacts/stage2/_batch/refinement_batch_report.json` records ordered tile
statuses (`complete`, `skipped`, or `failed`) and each failure's exception type
and message. Add `--fail-fast` when immediate termination is preferred.

The same batch command with `--mode human_review` generates review bundles for
all selected tiles but does not apply their pending review decisions. Connected
semantic splits use an exact checksum-bound partition through the Python core
API; neither automatic mode nor the CLI guesses building, road-junction,
canopy, or field boundaries in this release. See the Stage 2R guide for the
full policy, reports, and research roadmap.

## Build the tile graph with Stage 3

Stage 3 does not silently fall back to `12_final` and does not rerun either
segmentation model. Once Stage 2 is complete, run unattended refinement and
build the graph with one command:

```powershell
python -m src stage3 run `
  --tile-id IKONOS_2005-002__tile_00003 `
  --refine-first `
  --refinement-mode automatic
```

From the beginning, use:

```powershell
python -m src stage2 run --tile-id IKONOS_2005-002__tile_00003
python -m src stage3 run `
  --tile-id IKONOS_2005-002__tile_00003 `
  --refine-first `
  --refinement-mode automatic
```

For reviewed decisions, use `--refinement-mode human_review` together with
`--refinement-decisions <path>`. Automatic mode generates its own decisions and
does not accept a human decisions document. Use separate `refine propose`,
`refine apply`, and `refine validate` commands when you want to inspect the
review workflow explicitly.

Or call the public Python API:

```python
from src.stage3 import run_stage3

result = run_stage3("scene__tile_00000")
print(result.summary())
```

The default output is `artifacts/stage3/<tile_id>/01_graph/`. It includes the
authoritative `graph.json`/`graph_arrays.npz`, explicit low-noise
`graph_analysis.json`/`graph_analysis_arrays.npz`, adjacency CSV, node polygons,
reports/manifests, and views of the segmentation and spatial graph. Each tile is compact from creation: nodes keep IDs, semantic class/confidence,
shape/size/location, and graph topology; edges keep adjacency measurements.
Exact tile-pixel polygons are stored once per node. SAM/semantic-model
diagnostics, provenance, repeated coordinate variants, and copied refinement
histories are omitted from graph records. Original audit data remains in the
Stage 2/2R artifacts referenced by the graph. Re-run Stage 3 to compact older
tile graphs without rerunning the models.

`refinement_graph_before_after.png` places the graph reconstructed from
immutable `12_final` beside the refined graph, both over semantic-class
backgrounds. Purple is therefore always the explicit unresolved `unknown`
class in this comparison.

Adjacency is exact four-neighbor shared-edge adjacency over the refined
`instance_map`. A one-pixel shared side creates an edge; diagonal contact alone
does not. The authoritative exports keep every node and every one-side edge.
For the normal low-noise view, the default threshold of 41 excludes remaining
unknown nodes through 40 pixels and edges with fewer than two shared sides,
then routes each included edge through its measured contact centroid.
Topology-split micro-fragments through 40 pixels have already been consolidated
into their original objects and therefore are not authoritative nodes.
Every `graph.json`
node and edge carries `analysis_view.included` and exclusion reasons; full graph
counts remain authoritative, while `graph_analysis.json` is the checksum-bound
selected subgraph and `report.json.analysis_graph` records its counts.
See the Stage 3 document for the precise metadata and artifact schema.

## Combine selected tile graphs with Stage 4

After Stage 3 exists for the tiles you want, repeat `--tile-id` for any
selection size:

```powershell
python -m src stage4 combine `
  --tile-id IKONOS_2005-002__tile_00000 `
  --tile-id IKONOS_2005-002__tile_00001 `
  --tile-id IKONOS_2005-002__tile_00002 `
  --name IKONOS_first_three_tiles
```

Stage 4 uses Stage 1 global pixel offsets for exact placement. Within overlaps,
same-class mutual-best masks must pass IoS, IoU, bidirectional-coverage, and
minimum-intersection gates before their identities merge. Every shared source
pixel is then owned once, preferring a tile that has a valid prediction and
then the tile with the greatest distance from its image edge, with semantic
confidence and stable tile order as tie-breakers. The final adjacency graph is
rebuilt from the resulting non-overlapping geometries.

Outputs are written to `artifacts/stage4/<selection>/01_combined_graph/`,
including the combined JSON/NumPy graph, GeoJSON, adjacency CSV, overlap match
ledger, report, checksummed manifest, spatial graph image, and
`combined_graph_overlay.png` drawn over the ownership-resolved tile mosaic.

The normal combined overlay uses `graph_analysis.json`: unresolved unknowns
below 41 pixels and edges with fewer than two shared sides are omitted. The
full graph remains in `graph.json` and `combined_graph_overlay_audit.png`.
This display selection is separate from real semantic absorption in Stage 2R.

## Run all stages for complete yearly tile folders

The `folder run` command takes every PNG directly inside one folder. It executes
real SAM 2 and OpenEarthMap segmentation, automatic full refinement (including
unknown and contextual noise cleanup), refinement validation, one spatial graph
per tile, and one combined graph for the whole year. It saves the combined graph
over the complete selected image as `full_image_with_graph.png` inside the output
folder. That image shows every graph node and edge; the less crowded analysis
view is `full_image_with_analysis_graph.png`. These are rendered previews of the
whole tile mosaic, not native-resolution rasters. The full graph, analysis graph,
overlap ledger, GeoJSON, and other audit files remain under
`stage4/year_<year>/01_combined_graph/`.

Every folder needs `metadata.csv` with each PNG's exact `tile_name`, `x0`, and
`y0`, plus one georeferenced GeoTIFF (or explicit `--metadata` and
`--source-raster` arguments). Without those inputs the command stops: PNG
filenames do not establish global location. The valid PNG pixels must match
the RGB values in the declared GeoTIFF windows; padded edge pixels are excluded
using the CSV's `valid_width` and `valid_height`. The files for the supplied 2017
and 2023 folders are already present. Their `georeference_audit.json` records
how the coordinates were matched to the source rasters. The two folders each
contain **7** PNG tiles, so these commands process all **14** available tiles.

From the repository root, with the existing sibling virtual environment in
PowerShell:

```powershell
$python = '..\.venv\Scripts\python.exe'
& $python -m src folder run --folder 'tiles17_23/2017' --year 2017 --output 'artifacts/tiles17_23_runs/2017'
& $python -m src folder run --folder 'tiles17_23/2023' --year 2023 --output 'artifacts/tiles17_23_runs/2023'
& $python -m src folder compare --first 'artifacts/tiles17_23_runs/2017' --second 'artifacts/tiles17_23_runs/2023' --registration-json 'tiles17_23/registration_2017_2023.json' --boundary-tolerance 6 --output 'artifacts/tiles17_23_runs/spatiotemporal'
```

For a completed `image run` output folder from each year, use `image compare`
with `--first`, `--second`, and `--output`. To inspect just one saved PNG from
two multi-tile annual outputs, add `--tile tile_00000.png`:

```powershell
& $python -m src image compare --first 'artifacts/annual_2013_updated' --second 'artifacts/annual_2023_updated' --tile 'tile_00000.png' --output 'artifacts/temporal_tile_00000_2013_2023'
```

Both comparison commands render the earlier imagery above the later imagery
in the same mapped horizontal frame. Small node markers and matched-region
links run fully between the two images. `all_edge_types.png` uses one color per
class-specific outcome with a legend; `outcome_graph.png` gives nearby links
different colors, and `change_<class_outcome>.png` isolates each detected
physical event type. The raw correspondence, split, merge, and same-year
adjacency records remain in the graph but are not shown as temporal outcomes. Weak or
distant links are screened using world-coordinate overlap and local changed
area. Unknown and unchanged observations do not create change edges; the
rejection counts are in `report.json`.

The third command reads the two completed annual `run.json` files and their
checksummed Stage 4 graphs. It saves `01_temporal_graph/graph.json`,
`temporal_graph.png`, `changes.geojson`, reports, and an indexed graph array in
the comparison output folder. Its `run.json` reports common coverage and the
spatial, temporal, and change-edge counts. The supplied registration file uses
cross-year image controls; its held-out patch check had 2.3 m median and 5.3 m 95th
percentile feature disagreement. The 6 m change-boundary tolerance is a
conservative screening threshold, not proof of real-world change accuracy.
Inspect the overlays and candidate change geometries before interpreting
construction, demolition, or disappearance as physical facts.

The yearly command can resume a run with identical inputs and configuration:
completed tiles are validated and reused, while incomplete tiles run all their
stages. Changing source files, metadata, or the configuration requires a new
output folder. A failed comparison can be retried with the same inputs and
output folder; completed comparisons are kept immutable. The comparison checks
that the annual graphs contain the recorded tiles and have common observed
coverage after registration. `folder compare` accepts `--register-images`
instead of `--registration-json` for other datasets when automatic image matching
has enough reliable correspondences using the saved PNGs and georeference; the
original source GeoTIFFs need not still be present. With `--project-root`, relative
config paths are resolved from that project root.
