# Stage 2 segmentation phase: implementation and continuation guide

This document explains the segmentation phase as it is implemented in this
repository. It covers the real inputs to both models, their outputs, mask
processing, deterministic fusion, the final segment schema, artifacts,
quality checks, current limitations, and the correct place to continue the
project.

The most important idea is:

> SAM 2 proposes class-agnostic shapes. OpenEarthMap predicts land-cover
> classes per pixel. Stage 2 combines those independent predictions into one
> exclusive, gap-free semantic-instance map for one tile.

The authoritative output is `artifacts/stage2/<tile_id>/12_final/`. The
earlier `10_final/` directory contains only processed SAM candidates and is not
the completed two-model segmentation.

## 1. What Stage 2 produces

Stage 2 produces a semantic-instance segmentation for one Stage 1 image tile.
For every valid tile pixel, a successful result guarantees:

- exactly one positive instance ID;
- exactly one semantic class ID;
- no overlap between final instances;
- no unassigned valid pixels; and
- traceable geometry and label provenance.

Invalid input pixels remain outside the segmentation:

- `instance_map = 0`;
- `semantic_map = 255` (`nodata`); and
- `source_map = 255`.

"Instance" in this contract means one exclusive connected output region. It
does not necessarily mean one independently verified real-world object. A SAM
region can be an object-like instance, while an OpenEarthMap completion region
can be a connected land-cover patch spanning several physical objects.

Stage 2 is deliberately tile-local. It does not reconcile objects in the
50-pixel overlap between neighboring tiles, assign source-wide IDs, stitch
boundaries, or construct a graph.

### Pipeline at a glance

```text
Stage 1 PNG tile + CSV row + optional source-TIFF metadata
                         |
                         v
             validated, normalized uint8 RGB
                    + valid-data mask
                    /               \
                   v                 v
       SAM 2.1 automatic masks    OpenEarthMap FasterSeg
       (geometry, no classes)     (8-class probabilities)
                   |                 |
                   v                 v
       normalize/filter/dedupe    label + confidence maps
                    \               /
                     v             v
                 deterministic fusion
                 1. prioritize SAM geometry
                 2. label SAM components
                 3. fill with confident OEM components
                 4. label every residual pixel unknown
                             |
                             v
              exclusive masks + dense maps + metadata
                    + GeoTIFF/GeoJSON/previews
```

## 2. Input contract

The public runner takes a tile ID, not an image path:

```text
<source_image_id>__<tile_filename_without_extension>
```

Example:

```text
IKONOS_2005-002__tile_00003
```

### Required and optional files

| Input | Required | How it is used |
|---|---:|---|
| `data/tiles/<source>/tile_XXXXX.png` | Yes | Supplies every pixel passed to both models. |
| `data/metadata/<source>.csv` | Yes | Resolves the tile name, source-pixel offset, and expected dimensions. |
| `data/raw/<source>.tif` or `.tiff` | No for inference; strongly recommended | Only its metadata is read to recover dtype, nodata, affine transform, CRS, source dimensions, and bounds. Its raster bands are not used by Stage 2. |
| `config.yaml` | Yes | Selects model assets and all preprocessing/fusion behavior. |
| SAM 2 checkpoint | Yes for the real backend | `checkpoints/sam2/sam2.1_hiera_large.pt` by default. |
| FasterSeg architecture and weights | Yes for the real backend | `arch_1.pt` and `weights1.pt` by default. |

The Stage 1 producer writes these CSV columns:

```text
tile_id, tile_name, x0, y0, x1, y1, width, height
```

Treat all eight as the project contract. In the current validator,
`tile_name`, `x0`, and `y0` are operationally essential; `width` and `height`
are checked when present. `x1` and `y1` are currently not cross-validated,
which is listed later as an improvement.

The source TIFF lookup is exact. For source ID `scene`, the code checks only
`data/raw/scene.tif` and `data/raw/scene.tiff`. If neither exists, or both
exist, spatial metadata is marked unavailable rather than guessed.

### What image is actually passed to the models

Both models receive the same `sam_input.npy` array created from the Stage 1
PNG. Despite the filename, it is shared by SAM 2 and OpenEarthMap.

The array contract is:

```text
shape: (tile_height, tile_width, 3)
layout: HWC
channel meaning: RGB
dtype: uint8
range: 0..255
```

Construction is deterministic:

1. OpenCV reads the PNG and Stage 2 converts OpenCV BGR ordering to RGB.
2. The three configured 1-based `rgb_bands` are selected.
3. A pixel is valid only when all three selected values are finite and none
   equals the recovered nodata value, when a nodata value is available.
4. The default `percentile` normalization pools all valid selected RGB values,
   computes the 2nd and 98th percentiles, clips to that range, and scales to
   `0..255`. `per_band_percentile` is a different supported mode.
5. Invalid pixels receive the configured `nodata_fill`, currently `0`.

The resulting RGB image is saved to:

```text
02_preprocessing/sam_input.npy
02_preprocessing/sam_input.png
```

The aligned Boolean footprint is saved as:

```text
02_preprocessing/valid_pixels.npz -> valid_pixels
```

The valid mask is not an input channel to either neural network. Both models
see the RGB fill value at invalid locations; the valid mask is applied later
during SAM filtering and final fusion.

Important limitation: Stage 2 does not read a native TIFF window for model
inference. Therefore, original multispectral bands, explicit alpha/mask bands,
padding information, and some radiometric fidelity may already have been lost
when Stage 1 created the PNG. The current Stage 1 loader reads source bands
1-3 only, so Stage 2 `rgb_bands` selects among PNG channels rather than
arbitrary original GeoTIFF bands.

## 3. Models and their real runtime inputs/outputs

Neither model receives a text prompt. Neither model receives the other
model's output. They run independently on the same normalized RGB tile.

| Model | Role | What the pipeline passes | Prompt behavior | Direct output |
|---|---|---|---|---|
| SAM 2.1 Hiera-Large | Candidate instance geometry | One HWC `uint8` RGB image | No user prompt is passed. `SAM2AutomaticMaskGenerator` internally creates a regular grid of positive point prompts, also over configured image crops. | A list of class-agnostic mask dictionaries. |
| OpenEarthMap FasterSeg | Per-pixel land-cover evidence | The same HWC `uint8` RGB image | Image only; no point, box, mask, or text prompt. | Eight probability planes, an argmax label map, and a maximum-probability confidence map. |

### 3.1 SAM 2.1 Hiera-Large

The configured backend builds Meta's SAM 2.1 Hiera-Large checkpoint and calls:

```python
automatic_mask_generator.generate(image)
```

The application itself supplies no click, bounding box, text, or previous
mask. The automatic generator creates its point prompts internally. With the
current configuration it samples a 32 by 32 grid on the full tile and also
uses one crop layer. These are generated prompts, not user-provided semantic
instructions.

Each raw SAM dictionary normally contains:

| Field | Meaning |
|---|---|
| `segmentation` | Boolean mask or configured RLE representation. |
| `area` | SAM's reported pixel area before Stage 2 normalization/fusion. |
| `bbox` | SAM's raw `[x, y, width, height]` box. |
| `predicted_iou` | SAM's predicted mask-quality score; not measured ground-truth IoU. |
| `stability_score` | Stability under mask-threshold perturbation. |
| `point_coords` | Internally generated point prompt that produced the mask. |
| `crop_box` | Image crop in which the proposal was generated. |

SAM does not output `tree`, `road`, `building`, or any other class name.

On CPU or native Windows, SAM 2's optional compiled connected-component hook
is commonly unavailable. The backend then disables that generator hook and
applies the official OpenCV small-hole/small-island cleanup once to each
returned mask. The inference report records either
`sam2_compiled_extension`, `opencv_cpu_fallback`, or `disabled` as the
postprocessing backend.

### 3.2 OpenEarthMap FasterSeg

FasterSeg receives the shared RGB image, then its backend:

1. resizes it to `1024 x 1024` by default;
2. converts RGB values to floating point and divides by 255;
3. applies the configured implementation's fixed RGB mean/std normalization;
4. runs the network;
5. optionally runs a horizontally flipped copy and combines the logits;
6. applies softmax across the eight channels; and
7. bilinearly resizes each probability plane back to the original tile shape
   and renormalizes the planes if needed.

Its direct outputs are:

| Output | Shape | Meaning |
|---|---|---|
| `probabilities` | `(8, H, W)` | Probability for every OpenEarthMap class at every pixel. |
| `label_map` | `(H, W)` | `argmax(probabilities)`; one class ID `0..7` per pixel. |
| `confidence` | `(H, W)` | `max(probabilities)`; confidence of the argmax class. |

The probabilities are normally saved in `11_semantic/openearthmap.npz`, but
the current fusion implementation reads only `label_map` and `confidence`.
It does not yet use the full eight-class distribution.

### OpenEarthMap class order

The channel/ID order is fixed in code because it must match the checkpoint:

| ID | Name |
|---:|---|
| 0 | `bareland` |
| 1 | `rangeland` |
| 2 | `developed space` |
| 3 | `road` |
| 4 | `tree` |
| 5 | `water` |
| 6 | `agriculture land` |
| 7 | `buildings` |

`unknown` is not a FasterSeg channel. It is added by the fusion rules as ID
`8`. `nodata` is not a model class either; it is stored as ID `255`.

## 4. Execution phases

`python -m src stage2 run` performs the following sequence.

| Directory/phase | Main input | Main output | Purpose |
|---|---|---|---|
| `01_validation` | Stage 1 PNG/CSV and optional TIFF metadata | normalized tile metadata, validation report, preview | Resolve a unique tile, check dimensions/offsets, and recover georeferencing. |
| `02_preprocessing` | validated PNG | shared RGB model input and valid mask | Select RGB, normalize, fill invalid pixels. |
| `04_raw_inference` | shared RGB input | raw SAM masks and raw metadata | Run automatic SAM inference. |
| `05_normalized` | raw SAM output | normalized masks/records | Recompute mask geometry and reject empty/wrong-shaped masks. |
| `06_filtering` | normalized SAM masks | accepted and rejected masks | Apply area, quality, valid-pixel, fill, and border rules. |
| `07_deduplication` | accepted SAM masks | retained candidate masks | Remove near-duplicate masks and audit nested masks. |
| `08_coverage` | retained SAM masks + valid mask | coverage/overlap maps and audit masks | Measure SAM-only gaps and overlap. This is not final fusion coverage. |
| `09_visualization` | SAM audit artifacts | overlays and inspection JSON | Make the SAM proposal process inspectable. |
| `10_final` | processed SAM candidates | candidate `segments.npz/json` | Stable input to fusion. Despite its historical name, it is not the completed segmentation. |
| `11_semantic` | shared RGB input | OEM label/confidence/probability maps | Run OpenEarthMap FasterSeg. |
| `12_final` | SAM candidates + OEM maps + valid mask + tile metadata | authoritative complete segmentation | Fuse, validate, georeference, and export. |

There is no per-tile `03_model` directory in a normal full run. The separate
`stage2 check-model` diagnostic writes to `_model/03_model`; at present it
checks only the SAM backend, not FasterSeg.

Each phase manifest records timestamps, a relevant configuration hash, input
checksums, output paths, software basics, and staleness. Rerunning an upstream
phase marks later phase manifests stale. The full runner currently recomputes
the pipeline; it does not resume automatically from the newest fresh phase.

## 5. How SAM proposals are prepared

Before model fusion, Stage 2 turns raw SAM proposals into auditable candidates.

### Normalization

For every non-empty, correctly shaped mask, Stage 2 computes:

- stable candidate ID `seg_XXXXX`;
- exact area from the Boolean mask;
- tile-local bounding box and centroid;
- predicted IoU and stability score copied from SAM;
- fraction of mask pixels inside the valid-data footprint;
- whether the mask touches the tile boundary; and
- provenance linking back to `raw_XXXXX` and SAM's raw area/bbox.

### Filtering

With the current defaults, a proposal is rejected if any applicable rule
fails, including:

- fewer than 40 pixels;
- more than 95% of the tile area;
- less than 50% valid pixels or more than 50% nodata;
- fill ratio below 1%; or
- the configured border policy.

The optional minimum predicted-IoU and stability filters are currently `null`,
so the generator's own thresholds perform those quality gates by default.
Accepted and rejected masks are both retained as audit artifacts.

### Deduplication and nesting

Pairs with intersecting bounding boxes are compared using mask IoU,
intersection-over-smaller-mask containment, and area similarity. The default
behavior removes near-identical masks and keeps the one with higher predicted
IoU. Ordinary nested parent/child proposals remain because the default nested
policy is `remove_near_identical_only`.

### SAM-only coverage audit

Phase 08 counts how many retained SAM masks cover every valid pixel. The default
`leave_unassigned` policy adds no background mask. Overlaps and gaps seen here
are expected; phase 12 resolves them. Do not use phase 08 or phase 10 coverage
as the final product's coverage.

## 6. Exact fusion algorithm

Fusion is deterministic. Given the same masks, semantic maps, metadata, and
configuration, the ordering and output IDs are reproducible.

### Step 1: initialize exclusive maps

The algorithm creates empty `instance_map`, `semantic_map`, `confidence_map`,
`source_map`, and an `occupied` mask. Only pixels in `valid_mask` may be
assigned.

### Step 2: order SAM masks

SAM candidates are sorted by:

1. `predicted_iou`, descending;
2. `stability_score`, descending;
3. reported `area_pixels`, descending;
4. `segment_id`, ascending; and
5. original input index, ascending.

Missing quality values rank below finite values.

### Step 3: make SAM geometry exclusive

For each sorted SAM mask:

```text
available = SAM mask AND valid mask AND NOT already occupied
```

The available remainder is split into deterministic 8-connected components.
Components smaller than the configured SAM fusion minimum (40 pixels by
default) are discarded at this step. They are not necessarily lost: those
pixels remain unoccupied and can later be recovered by OEM completion or the
unknown fallback.

This means a lower-priority SAM mask never overwrites a higher-priority mask.
It can be shortened, split into several final segments, or disappear.

### Step 4: label each retained SAM component

Within one SAM component, an OEM pixel is allowed to vote only if:

- its label ID is valid;
- its confidence is usable; and
- confidence is at least the semantic threshold (`0.45` by default).

The confident pixels cast an unweighted class vote. The class with the most
pixels wins. Ties go to the lowest ID in the fixed OEM class order.

The entire SAM component receives the winning class, even if some individual
pixels have a different OEM argmax class. If no pixel is confident enough to
vote, the SAM geometry is retained but its label becomes `unknown`.

The final provenance for these segments is:

```text
geometry_source.model = sam2
label.source_model     = openearthmap_fasterseg, or stage2_fusion_rule
source_code            = 1
```

The last case is subtle: source code `1` can contain an `unknown` label when
SAM supplied valid geometry but OEM supplied no confident vote. Code `1` does
not always imply a named OEM label.

### Step 5: fill unoccupied pixels with confident OEM components

For each of the eight OEM classes, fusion selects pixels that are:

```text
valid AND unoccupied AND OEM argmax == class AND confidence >= threshold
```

It splits those pixels into 8-connected components. A component is retained
only when it meets the class-specific area minimum:

| Class | Default minimum pixels |
|---|---:|
| bareland | 100 |
| rangeland | 100 |
| developed space | 80 |
| road | 30 |
| tree | 40 |
| water | 80 |
| agriculture land | 100 |
| buildings | 20 |

For these regions, OpenEarthMap supplies both geometry and label. They use
source code `2`.

### Step 6: make every residual valid pixel unknown

After SAM and confident OEM assignment:

```text
remainder = valid mask AND NOT occupied
```

Every 8-connected remainder component becomes a final `unknown` segment with
source code `3`. No residual pixel is dropped, including one-pixel components.

Residual unknown pixels can represent:

- OEM confidence below the threshold;
- a confident OEM component smaller than its class-specific minimum;
- invalid OEM labels/confidences when `fuse_segmentations` is called directly;
- small fragments removed from SAM geometry that were not recovered by OEM;
  or
- any other valid pixel left after accepted assignments.

The top-level runner validates OEM output before fusion, so invalid OEM
labels/confidences should not normally reach this step.

### Step 7: enforce final invariants

Fusion fails rather than writing a partial result if it detects:

- a valid gap;
- an overlap;
- an assigned nodata pixel; or
- disagreement between masks, maps, IDs, and record labels.

Final instance IDs are contiguous `1..N`. Record/mask index `i` corresponds to
instance ID `i + 1`.

### A small conceptual example

Suppose the highest-ranked SAM mask covers a region containing 80 confident
`tree` pixels and 20 confident `rangeland` pixels. The final SAM component is
uniformly labeled `tree`; it is not split by those semantic classes. A later
SAM mask overlapping it loses the occupied pixels and may split into multiple
components. After all SAM masks, an uncovered confident road strip becomes an
OEM component. Any remaining low-confidence islands become `unknown`. All
resulting components are mutually exclusive in the dense instance map.

## 7. Semantic and provenance registries

### Semantic map

| Value | Name | Source of the label definition |
|---:|---|---|
| 0 | `bareland` | OpenEarthMap FasterSeg |
| 1 | `rangeland` | OpenEarthMap FasterSeg |
| 2 | `developed space` | OpenEarthMap FasterSeg |
| 3 | `road` | OpenEarthMap FasterSeg |
| 4 | `tree` | OpenEarthMap FasterSeg |
| 5 | `water` | OpenEarthMap FasterSeg |
| 6 | `agriculture land` | OpenEarthMap FasterSeg |
| 7 | `buildings` | OpenEarthMap FasterSeg |
| 8 | `unknown` | Stage 2 fusion rule |
| 255 | `nodata` | Input validity mask |

### Source map

| Value | Stored source name | Exact meaning |
|---:|---|---|
| 1 | `sam2_geometry_openearthmap_label` | Geometry came from SAM. The label usually comes from OEM majority voting, but can be fusion `unknown` when no confident vote exists. |
| 2 | `openearthmap_geometry_and_label` | Geometry and named label came from an uncovered confident OEM connected component. |
| 3 | `fusion_unknown` | Geometry and `unknown` label came from the residual gap-free fusion rule. |
| 255 | `nodata` | Pixel is outside the valid-data footprint. |

Geometry provenance and label provenance are intentionally separate. Always
read `geometry_source` and `label.source_model`; do not infer one from the
other or from the segment ID alone.

## 8. Output artifacts

The authoritative directory is:

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

The schema identifier is:

```text
satellitegraph.stage2.full-segmentation.v1
```

This version is enforced through the final `manifest.json`. The current
`segments.json` wrapper itself contains only `mask_storage`, `count`, and
`records`; it does not repeat `schema_version`.

### 8.1 Dense arrays in `segmentation.npz`

| Array | Persisted dtype | Meaning |
|---|---|---|
| `instance_map` | `int32` | `0` for invalid pixels; contiguous `1..N` for final instances. |
| `semantic_map` | `uint8` | IDs `0..8`; `255` for nodata. Every pixel in one instance has one semantic ID. |
| `confidence_map` | `float16` | Confidence associated with the final assigned label; loaded as `float32` by the API. |
| `source_map` | `uint8` | Per-pixel provenance codes `1`, `2`, `3`, or `255`. |
| `valid_mask` | `bool` | Valid input footprint used by fusion. |

`confidence_map` needs careful interpretation:

- For an OEM-geometry segment, it is OEM's maximum-class probability at each
  component pixel.
- For a named SAM-geometry segment, it keeps OEM maximum-class confidence
  where that pixel's OEM argmax agrees with the segment's winning class and
  writes `0` where it disagrees. Agreement pixels below the voting threshold
  can still carry their OEM confidence.
- For a SAM segment labeled unknown, and for residual unknown, it is `0`.
- For nodata it is `0`.

Therefore zero means unavailable/not supporting the final label, not a
calibrated probability of zero. Mean confidence should not be compared across
provenance types without accounting for these rules.

### 8.2 Masks and records

`segments.npz` stores final Boolean masks in bit-packed compressed form.
`segments.json` stores one metadata record per mask. The two files must be read
together with `src.stage2.storage.load_masks`, or through
`load_stage2_result`.

The equivalence is exact:

```text
masks[i] == (instance_map == i + 1)
segments[i].mask_index == i
segments[i].instance_id == i + 1
```

### 8.3 Geospatial and visual exports

- `instance_map.tif` is a compressed `int32` GeoTIFF with nodata `0`.
- `semantic_map.tif` is a compressed `uint8` GeoTIFF with nodata `255`.
- `segments.geojson` polygonizes actual final mask boundaries and stores a
  compact property set: instance/segment IDs, label, provenance models, and
  pixel area.
- `final_segmentation.png` is a semantic color preview.
- `final_overlay.png` blends semantic color with the normalized RGB tile and
  draws instance boundaries in white.

When usable affine/CRS metadata exists, GeoTIFFs carry it and GeoJSON is
transformed to WGS84 when possible. Without georeferencing, GeoTIFF/GeoJSON
coordinates are source-image pixels offset by `global_x0/global_y0`, and no CRS
is claimed.

`segments.json` contains bounding-box coordinate footprints. Use
`segments.geojson` when the detailed polygonized mask boundary is required.

### 8.4 Reports and manifests

- `class_registry.json` is the machine-readable semantic registry.
- `tile_metadata.json` preserves the normalized Stage 1/source-raster context.
- `report.json` records fusion rules, class/source counts, confidence summary,
  exact-coverage checks, per-SAM processing audit, OEM completion audit, model
  roles, and coordinate availability.
- `manifest.json` is the public artifact index and model/config summary.
- `phase_manifest.json` records the final phase's checksummed inputs and output
  paths.

In the report, `class_pixel_counts` counts pixels and
`class_segment_counts` counts segments. The later compatibility field
`semantic_class_counts` also counts segments despite its less precise name.

## 9. Per-segment metadata

The record schema is a structured dictionary rather than one rigid dataclass;
some evidence fields depend on whether geometry came from SAM, OEM, or the
unknown fallback. The following fields form the final common contract.

### Identity and storage

| Field | Meaning |
|---|---|
| `segment_id` | Deterministic origin-aware ID such as `sam_...`, `oem_...`, or `unknown_...`; unique within the tile result. |
| `instance_id` | Positive dense-map ID, `1..N`. |
| `mask_index` | Zero-based index in `segments.npz`; equals `instance_id - 1`. |
| `source_tile_id` | Canonical `<source>__tile_XXXXX` ID. |
| `source_raw_mask_id` | Original SAM raw ID for SAM-derived records; `null` for OEM/residual records. |

IDs are only tile-local. Do not use `instance_id` or `segment_id` as a
source-wide/global object ID without prefixing and later reconciliation.

### Final geometry

| Field | Meaning |
|---|---|
| `area_pixels` | Exact count of pixels in the final post-fusion mask. It is not square metres. |
| `bbox_xywh` | Tile-local `[x, y, width, height]`. |
| `bbox_xyxy` | Tile-local half-open `[x_min, y_min, x_max, y_max)`. |
| `centroid_rc` | Mean mask pixel index in `[row, column]` order. |
| `centroid_xy` | Pixel-centre coordinate `[column + 0.5, row + 0.5]`. |
| `touches_tile_boundary` | Whether the final mask reaches any tile edge. This is a strong cross-tile reconciliation signal, not proof of truncation. |
| `valid_pixel_fraction` | `1.0` for every final component because fusion clips to valid data. |

### Label and construction source

| Field | Meaning |
|---|---|
| `semantic_class_id`, `class_id` | Compatibility aliases of final semantic ID. |
| `semantic_class`, `class_name` | Compatibility aliases of final class name. |
| `label_source` | Model/rule credited with the final label. |
| `label` | Nested label decision, evidence, and explanation. |
| `geometry_source` | Nested model/rule and operation that created the final geometry. |
| `source`, `source_code` | Compact construction path matching `source_map`. |
| `mean_semantic_confidence` | Mean of the final confidence-map values across the segment, using the provenance-specific confidence rule above. |
| `explanation` | Human-readable combined geometry/label explanation. |

The duplicated class fields are compatibility aliases. New downstream code
should prefer `label.id` and `label.name`, while still validating that the
aliases agree.

### Quality and provenance

| Field | Meaning |
|---|---|
| `predicted_iou` | SAM predicted quality for SAM-derived geometry; `null` otherwise. It is not a final semantic accuracy score. |
| `stability_score` | SAM stability for SAM-derived geometry; `null` otherwise. |
| `metrics` | Final area ratio relative to the whole tile (not only valid pixels), valid/nodata fractions, bounding-box fill ratio, and boundary flag; SAM records can also retain SAM quality values. |
| `accepted` | Always `true` for a persisted final segment. |
| `rejection_reasons` | Empty for final segments; populated only in earlier rejected-mask artifacts. |
| `provenance` | Machine-readable raw-mask and fusion operation lineage. |

For SAM-derived records, the original quality scores describe the source SAM
proposal. Fusion may subtract overlap and split it, so those scores were not
recomputed for the final component.

### The nested `label` object

All final label objects contain:

| Field | Meaning |
|---|---|
| `id`, `name` | Final class. |
| `source_model` | `openearthmap_fasterseg` or `stage2_fusion_rule`. |
| `method` | Exact decision method. |
| `component_pixel_count` | Final component size. |
| `explanation` | Human-readable decision explanation. |
| `mean_final_label_confidence` | Mean final confidence under the map rules above. |
| `evidence` | Copy of method-specific evidence for machine consumption. |

Method-specific evidence includes:

- For a labeled SAM component: threshold, confident voter count, supporting
  count/fraction, vote count for all eight classes, mean supporting confidence,
  tied IDs, and tie-break rule.
- For an OEM component: threshold, supporting/component count, and mean/min/max
  source confidence.
- For unknown: mean OEM maximum confidence and reason counts. A SAM unknown
  specifically records that there were no confident votes; residual unknown
  records low-confidence/small-component/invalid-output counts.

### Coordinate metadata

Every record contains `coordinates` with explicit spaces:

| Space | Meaning |
|---|---|
| `tile_pixel` | Local coordinates inside this tile. |
| `source_pixel` | Tile coordinates plus `global_x0/global_y0`. |
| `global_pixel` | Intentional alias of `source_pixel`. |
| `projected` | Affine-transformed world coordinates; normally in the source CRS, but `crs` can be `null` if an affine exists without a CRS. |
| `world` | Intentional alias of `projected`. |
| `wgs84` | Longitude/latitude in `EPSG:4326`, when transformable. |

Pixel conventions are:

```text
origin: zero-based upper-left
x: column
y: row
bbox_xyxy: half-open [x_min, y_min, x_max, y_max)
pixel centre: (x + 0.5, y + 0.5)
```

Projected/WGS84 objects contain `available`, `unavailable_reason`, `crs`,
`centroid_xy`, `bbox_xyxy`, and a bounding-box `footprint`. Projected metadata
also records which affine/pixel space was used. An unavailable coordinate
space uses explicit `null` values and a reason; the code does not invent a CRS.
The per-record `projected.available` flag can be true for unreferenced affine
coordinates with `crs=null`; the report's projected-availability summary is
stricter and requires both affine and CRS. Treat CRS-less affine values as
unreferenced world coordinates, not safely projected GIS coordinates.

The coordinate chain is:

```text
tile pixel
  + (global_x0, global_y0)
  -> source/global pixel
  + source or tile affine transform
  -> projected/source-CRS coordinate
  + CRS transformation
  -> WGS84 longitude/latitude
```

## 10. Tile metadata

`tile_metadata.json` records, where available:

- canonical/local tile IDs and source image ID;
- absolute PNG and CSV paths;
- tile width/height and source offsets;
- assumed padding and selected/RGB bands;
- source raster path, dtype(s), dimensions, band count, and nodata;
- source and tile affine transforms;
- CRS in compact/WKT form;
- tile and source projected/WGS84 bounds;
- explicit availability flags/reasons; and
- compatibility notes explaining information Stage 1 did not preserve.

The raw TIFF contributes metadata only. The segmentation remains aligned to
the Stage 1 PNG pixels.

## 11. Running, inspecting, and consuming Stage 2

### Install and verify assets

```powershell
python -m pip install -r requirements.txt
```

Confirm these configured local files exist before real inference:

```text
checkpoints/sam2/sam2.1_hiera_large.pt
checkpoints/openearthmap/fasterseg/arch_1.pt
checkpoints/openearthmap/fasterseg/weights1.pt
```

`stage2 check-model` currently validates only SAM. A successful full run, or
the health record in `11_semantic/report.json`, is the current FasterSeg load
check.

### Run one tile

```powershell
python -m src stage2 run --tile-id scene__tile_00000
```

Optional arguments:

```powershell
python -m src stage2 run `
  --tile-id scene__tile_00000 `
  --config config.yaml `
  --project-root . `
  --no-progress
```

The supported recovery path after a failure is to correct the reported input,
asset, device, or configuration problem and rerun `stage2 run`. Individual SAM
phase CLI commands exist for diagnostics, but semantic/fusion are completed by
the unified `run` command and there is no automatic resume planner yet.

### Python API

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

To load an existing completed result without rerunning either model:

```python
from src.stage2 import load_stage2_result

result = load_stage2_result(
    "artifacts/stage2/scene__tile_00000/12_final"
)
```

### Relating a pixel, instance, and record

```python
y, x = 100, 200
instance_id = int(result.instance_map[y, x])

if instance_id == 0:
    print("nodata")
else:
    record = result.segments[instance_id - 1]
    assert record["instance_id"] == instance_id
    print(record["label"]["name"])
    print(record["geometry_source"])
    print(record["coordinates"]["wgs84"])
```

### Batch all tiles of one source

There is no supported all-tiles CLI command yet. A simple Python driver can
call the one-tile public API:

```python
from pathlib import Path
from src.stage2 import run_stage2

source = "scene"
for image_path in sorted((Path("data/tiles") / source).glob("*.png")):
    tile_id = f"{source}__{image_path.stem}"
    run_stage2(tile_id, project_root=".", config="config.yaml")
```

Plan GPU memory and disk use before parallelizing; both large models are loaded
for every call in this simple driver.

## 12. Quality-control checklist

Coverage is a structural guarantee, not proof of segmentation accuracy. Before
downstream use, check:

1. `final_overlay.png`: inspect boundaries, mixed-class SAM regions, thin
   roads, small buildings, water edges, and tile borders.
2. `report.json -> every_valid_pixel_exactly_once`: must be `true`.
3. `report.json -> coverage`: must report 100%, zero gaps, zero overlaps.
4. `class_pixel_counts` and `class_segment_counts`: look for implausibly absent
   or dominant classes.
5. `unknown_reason_pixel_counts`: separate low confidence from confident
   components rejected for small area.
6. `source_pixel_counts`: understand how much geometry came from SAM, OEM, and
   residual fallback.
7. `sam_processing`: inspect proposals lost to overlap or the final SAM area
   threshold.
8. boundary-touching segments and neighboring-tile overlaps: expect duplicate
   or truncated descriptions until a later stitching stage.
9. `coordinate_reference` and representative segment coordinates: verify CRS
   and source offsets before spatial analysis.
10. compare a labeled validation sample against ground truth. Model confidence
    is not a substitute for accuracy measurement.

Run deterministic tests without the optional real-model smoke tests:

```powershell
python -m pytest -q -m "not real_sam and not real_landscape"
```

The real-model marks can be run separately when the machine has the required
assets and compute capacity.

## 13. How to continue after segmentation

The current supported segmentation phase ends at `12_final`. No implemented
stage in this repository consumes that fused product to build the satellite
graph.

The `src/landscape` package is legacy compatibility code. It reads
`10_final` SAM candidates, excludes tree regions, and predates the unified
gap-free fusion. It is not the next stage after the current Stage 2 and should
not be used as a consumer of `12_final`.

A sound continuation sequence is:

1. **Run and gate every tile.** Require the structural invariants and establish
   project-specific thresholds for unknown fraction and class plausibility.
2. **Reconcile overlapping tiles.** Transform candidates to source-pixel or
   projected coordinates; match same-class regions in the overlap using mask
   IoU, boundary agreement, confidence, and provenance; retain a deterministic
   owner for duplicate pixels.
3. **Create source-wide objects.** Merge accepted matches, split conflicting
   regions when necessary, and assign stable global segment IDs. Preserve all
   contributing tile/segment IDs.
4. **Derive graph nodes.** Compute polygon geometry, projected area, perimeter,
   compactness, class distribution/confidence, provenance, and source-image
   lineage for every global segment.
5. **Derive graph edges.** Use adjacency/shared boundary, distance, containment,
   road connectivity, or other explicitly defined relations. The dense
   `instance_map` is convenient for exact local adjacency; GeoJSON is
   convenient for spatial operations.
6. **Validate at source scale.** Check seams, duplicated objects, global
   coverage, topology, and representative ground truth before graph analysis.

Prefer `12_final/segmentation.npz`, `segments.json`, and `segments.npz` as the
lossless continuation inputs. Use GeoTIFF/GeoJSON for GIS interoperability,
but remember that the current GeoJSON property set is deliberately smaller
than the full JSON record.

## 14. Current limitations and recommended improvements

These are recommendations, not descriptions of behavior that already exists.

### Highest priority

| Current limitation | Recommended improvement | Why it matters |
|---|---|---|
| No cross-tile reconciliation or global IDs | Implement an overlap-aware stitching stage consuming `12_final` | Prevents duplicate/truncated objects and is required before a reliable global graph. |
| Model input comes from PNG while source TIFF pixels/masks are ignored | Preserve tiled GeoTIFF/COG windows plus an explicit validity mask and band metadata, or read source windows directly | Avoids radiometric, nodata, padding, band-identity, and multispectral information loss. |
| Thresholds are configuration defaults, not dataset-calibrated decisions | Build representative ground truth and tune/report per-class precision/recall/IoU, boundary F-score, panoptic quality, calibration, and unknown rate | Structural 100% coverage can hide poor semantic or boundary accuracy. |
| Fusion discards the saved eight-channel probability tensor | Aggregate class distributions over SAM masks, use calibrated confidence-weighted evidence, and persist entropy/margin/full class distribution | Gives better labels and interpretable uncertainty, especially for mixed SAM regions. |
| A whole SAM component receives one majority class | Detect semantic heterogeneity and split a SAM component along strong semantic boundaries, or reject/flag ambiguous regions | Avoids forcing one label over a mask that spans several land-cover types. |

### Model and confidence integrity

- Calibrate OEM probabilities on the target sensor, season, resolution, and
  geography. The current confidence threshold should not be interpreted as a
  universal accuracy probability.
- The horizontal-flip TTA implementation sums two logits before softmax. This
  changes probability sharpness relative to averaging logits/probabilities.
  Evaluate and calibrate the combination because fusion thresholds depend on
  these values.
- Preserve aspect ratio for non-square Stage 1 edge tiles. FasterSeg currently
  stretches every whole tile to a square; use letterboxing or tested
  sliding-window inference instead.
- FasterSeg loading currently accepts any non-empty intersection of checkpoint
  keys and model keys, then keeps the model's current values for missing keys.
  Require full expected key/shape coverage and report missing/unexpected keys
  to prevent partially initialized inference.
- Consider OEM-derived boxes/points or a controlled prompt mode for classes
  where the automatic SAM grid repeatedly misses fine geometry. Keep automatic
  and prompted provenance distinct.
- Preserve SAM `point_coords`, `crop_box`, and postprocessing lineage into final
  provenance when detailed reproducibility/debugging is needed; they currently
  remain mainly in raw artifacts.

### Fusion and region quality

- Use per-class SAM/OEM compatibility rather than allowing every accepted SAM
  geometry to outrank all OEM geometry unconditionally.
- Add an ambiguity flag based on vote fraction, runner-up margin, entropy, and
  spatial disagreement. `predicted_iou` measures SAM mask quality, not label
  certainty.
- Revisit minimum component areas in ground units or resolution-aware units.
  Pixel thresholds change physical meaning across sensors.
- Residual unknown currently has no minimum size because gap-free coverage is
  mandatory. To reduce thousands of tiny unknown instances without losing
  pixels, merge small unknown regions into a compatible adjacent unknown or
  explicitly model a deterministic neighbor-assignment rule rather than
  deleting them.
- The configured `deduplication.bbox_overlap_threshold` is not used by the
  current pairwise implementation. Either implement its intended behavior or
  remove it from the public configuration.

### Geospatial and metadata correctness

- Validate CSV `x1/y1` against `x0/y0 + width/height`, and preserve real edge
  padding/unpadded dimensions rather than hardcoding zero padding.
- Record a true Stage 1 validity mask. The current finite/nodata reconstruction
  from PNG can mark pixels valid when the original mask information was lost.
- Fix the `report.coordinate_reference.bbox_convention` wording: final segment
  coordinates use half-open `bbox_xyxy`, while `bbox_xywh` is a separate field.
- Add projected area/perimeter and units. Current `area_pixels` is not a ground
  area, and a bounding-box footprint is not the detailed segment polygon.
- Enrich GeoJSON properties or provide an explicit join schema to the full
  `segments.json` metadata.

### Reproducibility and operations

- Make `stage2 check-model` validate both models, their architecture/weight
  compatibility, device selection, and a minimal inference shape.
- Define and validate a formal versioned schema for `segments.json` and all
  five dense arrays. Extend the loader to check confidence/source shapes and
  values, record `instance_id`/`mask_index`, and coordinate consistency.
- Make the report's declared dtypes agree with persistence. It currently
  describes the fusion-time `uint32`/`float32` maps while the authoritative NPZ
  stores `instance_map` as `int32` and `confidence_map` as `float16`.
- Hash model files and the FasterSeg architecture in manifests; recording only
  paths/config strings is insufficient if a file changes in place.
- Add automatic resume/reuse of fresh phases and a first-class batch command.
  The current one-call runner repeats all work.
- Reuse loaded models across tiles and add bounded GPU-aware concurrency.
- Avoid the current `float16` semantic disk round-trip before fusion, then
  persist scientific confidence arrays as `float32` or quantify/document the
  precision loss near decision thresholds.
- Implement or remove `save_raw_masks`, `save_rejected_masks`,
  `save_intermediate_previews`, and `compressed_masks`; they are currently
  configured but the pipeline always writes/bit-packs those products.
- Move repeated coordinate conventions, affine data, and CRS/WKT to a shared
  tile/document section and reference it from records. The current schema
  repeats large common metadata for every segment.
- Rename or clearly deprecate historical `10_final` and legacy landscape
  surfaces so downstream code cannot mistake SAM-only candidates for the fused
  product.
- Add integration tests for non-square edge tiles, explicit nodata masks,
  ambiguous/missing CRS, mixed-class SAM masks, checkpoint completeness,
  cross-tile overlap, and real target-domain accuracy.

## 15. Current repository artifact status

At the time this guide was prepared:

- `IKONOS_2005-002__tile_00003/12_final` is a completed fused artifact. Its
  report records 262,144 valid pixels, 100% exclusive coverage, 893 segments,
  and 10,074 unknown pixels. It is useful as a schema/example artifact, not as
  ground truth.
- `IKONOS_2005-002__tile_00000` stops at the historical `10_final` SAM-only
  output. It must be rerun with the unified `stage2 run` command before being
  consumed as a full segmentation.

Always require `12_final/manifest.json` with schema
`satellitegraph.stage2.full-segmentation.v1`; directory existence or a
`10_final` manifest is not enough.

## 16. Implementation source map

The main source-of-truth files are:

- [`src/stage2/pipeline.py`](../src/stage2/pipeline.py): public runner,
  semantic/fusion phases, final exports, and result validation.
- [`src/stage2/backend.py`](../src/stage2/backend.py): SAM 2 backend and raw
  automatic-mask contract.
- [`src/stage2/semantic.py`](../src/stage2/semantic.py): FasterSeg loading,
  input transformation, TTA, and probability outputs.
- [`src/stage2/masks.py`](../src/stage2/masks.py): SAM normalization,
  filtering, deduplication, and coverage audit.
- [`src/stage2/fusion.py`](../src/stage2/fusion.py): exact priority, voting,
  completion, unknown, confidence, and invariant rules.
- [`src/stage2/tiles.py`](../src/stage2/tiles.py): Stage 1 discovery,
  validation, and source-raster metadata recovery.
- [`src/stage2/preprocessing.py`](../src/stage2/preprocessing.py): model RGB
  input and valid-mask creation.
- [`src/stage2/coordinates.py`](../src/stage2/coordinates.py): coordinate
  conventions and transformations.
- [`src/stage2/storage.py`](../src/stage2/storage.py): mask/record persistence.
- [`src/config.py`](../src/config.py) and [`config.yaml`](../config.yaml):
  validated configuration and current defaults.
- [`tests/test_full_stage2.py`](../tests/test_full_stage2.py): full-product,
  coordinate, persistence, and unknown-fallback invariants.
