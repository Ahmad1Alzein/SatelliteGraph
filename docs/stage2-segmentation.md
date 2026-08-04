# Stage 2 — full semantic instance segmentation

Stage 2 runs SAM 2.1 and OpenEarthMap together and produces one complete,
labelled segmentation for one Stage 1 tile. It is the end of the segmentation
phase: there is no separate landscape-completion workflow and no Streamlit
interface.

For every valid tile pixel, the final contract guarantees:

- one positive, contiguous instance ID;
- one semantic value;
- no overlap between final instances; and
- recorded geometry and label provenance.

Invalid input pixels remain outside the segmentation with instance ID `0` and
semantic value `255` (`nodata`). The product is deliberately **per tile**.
Overlapping tiles can describe the same real-world object independently, so
cross-tile deduplication, edge matching, stitching, and graph construction must
be performed in a later stage.

## Installation and configuration

Install all project, SAM 2, OpenEarthMap, geospatial, and test dependencies from
the repository's single requirements file:

```powershell
python -m pip install -r requirements.txt
```

`config.yaml` is the single configuration source for the whole project. Its
`project` section defines paths and tiling, while `stage2` contains:

- `instance_model`: SAM 2 checkpoint, model type, device, and automatic-mask
  settings;
- `semantic_model`: OpenEarthMap FasterSeg source, architecture, weights,
  device, and inference settings;
- `preprocessing`, `filtering`, `deduplication`, and `coverage`;
- `fusion`: confidence threshold, component sizes, and the mandatory `unknown`
  fallback that guarantees complete valid-pixel coverage;
- `visualization`; and
- `output`.

There is no `configs/stage2.yaml`, no second landscape YAML, and no secondary
requirements file. Relative model and output paths are resolved from the
project root. The default local assets are:

```text
checkpoints/sam2/sam2.1_hiera_large.pt
checkpoints/openearthmap/fasterseg/arch_1.pt
checkpoints/openearthmap/fasterseg/weights1.pt
```

The pipeline never downloads model files at runtime. Install a CPU or CUDA
PyTorch build suitable for the target machine and place the configured assets
at their local paths before running a real inference.

## Input tile contract

Stage 2 consumes existing Stage 1 artifacts:

```text
data/tiles/<source>/tile_XXXXX.png
data/metadata/<source>.csv
```

The CSV must identify the tile and its source-image pixel extent, including
`tile_id`, `tile_name`, `x0`, `y0`, `x1`, `y1`, `width`, and `height`. Use the
canonical globally unique ID `<source>__tile_XXXXX`, for example
`scene__tile_00000`.

### Where to put the image

Put the source GeoTIFF at `data/raw/<source>.tif`. The raw-image directory is
configured by `project.paths.raw_directory` in `config.yaml`. Stage 2 does not
take this image path directly; it consumes the Stage 1 tile and metadata at
`data/tiles/<source>/tile_XXXXX.png` and `data/metadata/<source>.csv`.

Pass the canonical `<source>__tile_XXXXX` value as `tile_id` to `run_stage2` or
as `--tile-id` to the CLI. Neither runner takes an image path.

When `data/raw/<source>.tif` or `data/raw/<source>.tiff` exists as one exact,
unambiguous match, Stage 2 reads its metadata (not its pixel bands) to recover
the source affine transform, CRS, raster size, dtype, nodata value, projected
bounds, and WGS84 bounds. The segmentation pixels still come from the Stage 1
PNG. If the source raster or usable georeferencing is unavailable, spatial
fields stay explicitly unavailable with a reason; the pipeline does not invent
a CRS or transform.

## Run the complete Stage 2

The Python API has one full-pipeline method:

```python
from src.stage2 import run_stage2

result = run_stage2(
    "scene__tile_00000",
    project_root=".",
    config="config.yaml",
)

print(result.output_directory)
print(len(result.segments))
print(result.report["coverage"])
```

`run_stage2` returns a `Stage2Result` with:

- `tile_id` and `output_directory`;
- `masks` and their `segments` metadata records;
- `instance_map`, `semantic_map`, `confidence_map`, and `source_map`;
- `valid_mask` and normalized `tile_metadata`;
- the final `report`, artifact paths, and phase results.

The equivalent CLI surface is one command:

```powershell
python -m src stage2 run --tile-id scene__tile_00000
```

The runner displays a `tqdm` phase progress bar by default. Use
`--no-progress` to disable it:

```powershell
python -m src stage2 run --tile-id scene__tile_00000 --no-progress
```

It uses `config.yaml` by default and prints a JSON summary. To select an explicit
configuration or repository location:

```powershell
python -m src stage2 run `
  --tile-id scene__tile_00000 `
  --config config.yaml `
  --project-root .
```

There is no web application to launch. Both models, fusion, validation, and
final export are part of this one call.

## What each model contributes

SAM 2 and OpenEarthMap have different responsibilities; their outputs are not
interchangeable:

1. The tile is validated and normalized to an RGB model input with a valid-data
   mask.
2. SAM 2.1 Hiera-Large proposes class-agnostic masks. These masks are normalized,
   filtered, and deduplicated to provide detailed candidate geometry.
3. OpenEarthMap FasterSeg predicts an eight-channel land-cover probability map.
4. Fusion assigns OpenEarthMap labels to retained SAM geometry, resolves all
   overlap into a single instance map, and creates OpenEarthMap-derived connected
   components for valid pixels not covered by retained SAM geometry.
5. Remaining low-confidence valid pixels are grouped under the explicit
   `unknown` fusion label, so valid-pixel coverage is complete.

SAM 2 never supplies a class name. A segment can therefore have geometry from
SAM 2 and a label from OpenEarthMap. Segments introduced to complete uncovered
areas use OpenEarthMap for both their geometry and label. `unknown` comes from a
Stage 2 fusion rule, not from either model.

On native Windows or CPU-only installations, SAM 2's optional compiled `_C`
extension is normally unavailable. The Stage 2 runner automatically disables
that predictor hook and applies SAM 2's OpenCV small-hole/small-island cleanup
once to each generated mask instead. The model health report records either
`sam2_compiled_extension` or `opencv_cpu_fallback` as the cleanup backend; the
core SAM 2 inference is unchanged.

The semantic registry is fixed and versioned:

| ID | Name | Label source |
|---:|---|---|
| 0 | `bareland` | OpenEarthMap FasterSeg |
| 1 | `rangeland` | OpenEarthMap FasterSeg |
| 2 | `developed space` | OpenEarthMap FasterSeg |
| 3 | `road` | OpenEarthMap FasterSeg |
| 4 | `tree` | OpenEarthMap FasterSeg |
| 5 | `water` | OpenEarthMap FasterSeg |
| 6 | `agriculture land` | OpenEarthMap FasterSeg |
| 7 | `buildings` | OpenEarthMap FasterSeg |
| 8 | `unknown` | Stage 2 low-confidence fusion rule |
| 255 | `nodata` | Input valid-data mask |

The `source_map` records the per-pixel construction path:

| Value | Meaning |
|---:|---|
| 1 | SAM 2 geometry with an OpenEarthMap label |
| 2 | OpenEarthMap-derived geometry and OpenEarthMap label |
| 3 | Geometry and `unknown` label created by the fusion fallback |
| 255 | Invalid/nodata input pixel |

Every segment record repeats this distinction in its `geometry_source` and
`label` provenance. `report.json` also states the model roles, and
`manifest.json` records the configured backends, model types, checkpoints, and
class names. This prevents a semantic prediction from being misrepresented as
a SAM label.

## Final artifact contract

Intermediate audit artifacts occupy phases `01_validation` through
`11_semantic`. The authoritative full product is:

```text
artifacts/stage2/<tile_id>/12_final/
  segments.npz
  segments.json
  segmentation.npz
  class_registry.json
  tile_metadata.json
  report.json
  segments.geojson
  instance_map.tif
  semantic_map.tif
  final_segmentation.png
  final_overlay.png
  manifest.json
  phase_manifest.json
```

The final schema identifier is
`satellitegraph.stage2.full-segmentation.v1`.

### Segments and dense maps

- `segments.npz` stores the final boolean masks in compressed, bit-packed form.
- `segments.json` stores one record per mask, including the stable segment ID,
  source tile, area, bounds, centroid, semantic label and confidence, geometry
  provenance, label provenance, mask index, and coordinate metadata.
- `segmentation.npz` stores five aligned arrays:
  - `instance_map` (`int32`): `0` for invalid pixels and contiguous IDs `1..N`
    for valid pixels;
  - `semantic_map` (`uint8`): class IDs `0..8`, with `255` for nodata;
  - `confidence_map`: OpenEarthMap semantic confidence associated with the final
    assignment;
  - `source_map` (`uint8`): the provenance codes listed above; and
  - `valid_mask` (`bool`): the valid input-data footprint.
- `class_registry.json` defines every semantic ID, name, source, and
  explanation.
- `tile_metadata.json` preserves the normalized tile and source-raster metadata.
- `report.json` summarizes coverage, class counts, fusion decisions, coordinate
  availability, and the distinct roles of both models.

The stored masks and `instance_map` are equivalent: mask index `i` represents
pixels whose instance value is `i + 1`. Final instance IDs are contiguous, final
masks do not overlap, every valid pixel has a positive instance ID, and each
mask's semantic value agrees with its segment record.

### GeoTIFF, GeoJSON, and previews

- `instance_map.tif` is the `int32` instance raster with nodata value `0`.
- `semantic_map.tif` is the `uint8` class raster with nodata value `255`.
- `segments.geojson` polygonizes the actual final instance masks and includes
  instance/segment IDs, label ID/name/source, geometry source, and pixel area.
- `final_segmentation.png` is a semantic color preview.
- `final_overlay.png` overlays semantic color and instance boundaries on the
  normalized tile.

When an affine transform and CRS are available, the GeoTIFFs carry them and the
GeoJSON geometry is transformed to WGS84 when possible. If WGS84 conversion is
not possible, GeoJSON truthfully identifies its source-CRS coordinate space. If
there is no georeferencing, exports use source-image pixel coordinates and no
CRS is claimed.

`manifest.json` is the stable artifact index and model/configuration record.
`phase_manifest.json` records checksummed inputs and outputs for the final
phase.

## Coordinate spaces

Pixel coordinates are zero-based and use `(x, y) = (column, row)`. Bounding
boxes are half-open: `[x_min, y_min, x_max, y_max)`. Centroids refer to pixel
centres, so the centre of pixel index `(x, y)` is `(x + 0.5, y + 0.5)`.

Each segment's `coordinates` object makes the available spaces explicit:

- `tile_pixel`: local coordinates inside the current tile, including the local
  bounding box and centroid;
- `source_pixel`: local coordinates plus the tile's `global_x0/global_y0`
  offset in the original source image;
- `global_pixel`: an intentional alias of `source_pixel`, retained because both
  terms are used by the Stage 1/Stage 2 interface;
- `projected`: the bounding footprint and centroid after applying the tile or
  source affine transform, expressed in the source CRS;
- `world`: an alias of `projected`; and
- `wgs84`: the projected values transformed to longitude/latitude in
  `EPSG:4326`.

The projected and WGS84 coordinate objects contain `available`, `crs`,
`centroid_xy`, `bbox_xyxy`, and a polygon `footprint`. When a space cannot be
derived, those values are `null` and `unavailable_reason` explains why. The
coordinate footprint in `segments.json` is based on the segment bounding box;
the detailed mask boundary is the polygonized geometry in `segments.geojson`.

The coordinate hierarchy can be summarized as:

```text
tile-local pixel
  + (global_x0, global_y0)
  -> source/global pixel
  + source affine transform
  -> projected/source-CRS coordinate
  + CRS transformation
  -> WGS84 longitude/latitude
```

These offsets and spatial fields make later cross-tile reconciliation possible,
but Stage 2 itself does not use them to merge neighboring results.

## Verification and limitations

Run the automated suite with:

```powershell
python -m pytest -q
```

The model outputs are predictions rather than ground truth. SAM boundaries can
be too coarse, fragmented, or nested; OpenEarthMap can be affected by sensor,
season, resolution, and radiometric differences; and low-confidence pixels may
legitimately become `unknown`. Inspect `final_overlay.png`, class counts,
coverage, and provenance before downstream use.

Common failures are explicit:

- A missing checkpoint or architecture path means the configured local model
  asset has not been installed.
- Selecting CUDA when the installed PyTorch build has no CUDA support fails
  rather than silently changing the model.
- A missing or ambiguous exact source GeoTIFF prevents projected/WGS84 recovery
  but does not invent spatial metadata.
- A stale or missing intermediate artifact identifies the upstream phase that
  must be regenerated.

Stage 2 does not stitch overlapping tiles, reconcile segments across tile
borders, extract graph features, or build a local/global graph. Those operations
consume the per-tile `12_final` products in later stages.
