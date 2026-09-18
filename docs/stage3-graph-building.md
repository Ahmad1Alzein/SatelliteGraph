# Stage 3 tile graph building

## 1. Purpose and scope

Stage 3 transforms the completed Stage 2R refinement for one tile into a
tile-local region-adjacency graph:

- every final refined segment becomes exactly one graph node;
- every pair of segments that shares at least one pixel side becomes one
  undirected graph edge;
- nodes retain useful graph features; model diagnostics and refinement audit
  records stay in their original upstream artifacts;
- exact segment polygons and coordinate metadata are exported; and
- spatial graph visualizations make the result inspectable beside the
  segmentation.

Stage 3 is a deterministic geometry and topology step. It runs no model and
does not change refined segment boundaries or semantic labels.

The exported graph and the rendered graph view have deliberately different
jobs. `graph.json`, `graph_arrays.npz`, `adjacency.csv`, and `nodes.geojson`
retain the complete exact graph. A low-noise analysis/display view excludes
bounded micro-unknown nodes and one-side contacts from normal inspection so
segmentation noise does not dominate the picture; its inclusion flags never
delete or redefine graph data.

The output schema is:

```text
satellitegraph.stage3.tile-adjacency-graph.v3
```

The loader still accepts existing v2 graphs and v1 analysis views. Newly built
tiles use v3 graphs and v2 analysis views. Run Stage 3 again to compact existing
tiles; no segmentation/model rerun is needed.

This stage handles one tile only. It does not reconcile overlapping tiles,
merge duplicate objects, assign source-wide object IDs, or create cross-tile
edges. Those operations belong to Stage 4.

## 2. Authoritative input contract

The input is the authoritative, validated Stage 2R product:

```text
artifacts/stage2/<tile_id>/13_refinement/
```

The actual Stage 2 root comes from `stage2.output.root_directory` in
`config.yaml`. Stage 3 derives that value and requires
`stage3.input.source: refinement`; there is no duplicate input path under the
`stage3` configuration.

Stage 3 requires the completed, fresh refinement `manifest.json` and its
schema:

```text
satellitegraph.stage2-refinement.v1
```

It consumes lossless refinement data rather than trying to reconstruct
segments from a preview image:

| Refinement artifact | Stage 3 use |
|---|---|
| `manifest.json` | Schema gate, tile identity, artifact index, and provenance. |
| `segmentation.npz` | Dense `instance_map`, semantic map, confidence/source maps, and valid-data mask. |
| `segments.json` + `segments.npz` | Complete refined records and their exact masks. |
| `tile_metadata.json` | Tile offset, affine transform, CRS, and coordinate availability. |
| `operations.json`, `lineage.json`, and `refinement_operation_map.npz` | Reversible operation history and ancestry. |
| `refined_segmentation.png` / `refined_overlay.png` | Backgrounds for visual inspection. |

Stage 3 rejects a missing, incomplete, corrupt, stale, or wrong-version
refinement result. It does not silently fall back to `12_final`; the older
`10_final` SAM-only candidate product and legacy `src/landscape` output are
also not accepted as substitutes.

The refinement invariants are preserved:

- instance ID `0` is nodata;
- valid segments use contiguous instance IDs `1..N`;
- `segments[i]` and `masks[i]` correspond to instance ID `i + 1`; and
- every valid pixel belongs to exactly one final segment; and
- every final instance is either four-connected or an explicitly declared
  multipart object formed by consolidating topology-only micro-components; and
- every instance carries replayable lineage to its immutable Stage 2 source
  record or records.

## 3. Exact adjacency semantics

Stage 3 builds an undirected, simple graph. It has no self-loops and no
duplicate or parallel edges.

The configured definition is `shared_edge` with connectivity `4`. Two distinct
positive instance IDs are adjacent only when their pixels share a horizontal or
vertical grid side:

```text
shared side:  A B     -> edge

corner only:  A .
              . B     -> no edge
```

The right and bottom neighbor of every pixel are inspected. Different positive
IDs are placed in canonical order, so each unordered pair is counted once as a
graph edge even when it shares many sides. Instance ID `0` is ignored.

Corner-only or diagonal contact does not create an edge. Stage 3 does not use
mask dilation, a distance tolerance, bounding-box overlap, or polygon
intersection to invent adjacency.

For each adjacent pair, Stage 3 records:

- `shared_boundary_side_count`: total number of shared grid sides;
- `left_right_contact_count`: contacts found between left/right pixels;
- `top_bottom_contact_count`: contacts found between top/bottom pixels;
- `shared_boundary_length_pixels`: shared length in pixel-edge units; and
- `shared_boundary_length_projected`: length in source-CRS units when a usable
  affine transform and CRS are available, otherwise `null`.

`minimum_shared_boundary_pixels` is fixed at `1`. Therefore
every pair sharing even one pixel side is retained in the authoritative graph.
The separate analysis/display threshold marks one-side contacts as excluded
from the low-noise view, but the exact edge remains in `graph.json` and every
graph export.

The implementation scans the dense instance map in linear time with respect to
the number of tile pixels. It does not perform an all-pairs polygon comparison.

## 4. Graph identity and ordering

The graph is tile-qualified so it can be consumed safely by a later stage.

- A node ID has the form `<tile_id>::<segment_id>`.
- An edge ID is deterministic for the sorted pair of endpoint instance IDs.
- Node order follows refined instance order.
- Edge order is deterministic by its canonical endpoint pair.

An isolated refined segment still produces a node with degree zero. No segment
is dropped merely because it has no qualifying neighbor.

These IDs are stable for the same refinement artifact and Stage 3
configuration.
They are not source-wide object IDs: Stage 4 must assign those after reconciling
overlapping tiles.

## 5. Node metadata

Stage 3 selects these fields when creating each node, before writing any tile
artifact. The authoritative graph and its analysis subgraph both use the same
compact records. Tile/source identity and georeferencing are stored once in
`graph.tile_metadata`.

| Field | Meaning |
|---|---|
| `node_id`, `segment_id`, `instance_id` | Stable graph identity and lookup into the refined segmentation. |
| `segment_type` | Semantic class ID and name only. |
| `mean_semantic_confidence` | Confidence used for downstream overlap reconciliation. |
| `area_pixels`, `perimeter_pixels` | Exact pixel area and boundary length. |
| `geometry_type`, `geometry` | Exact unsimplified Polygon/MultiPolygon GeoJSON in tile pixels, including holes. |
| `centroid_xy`, `bbox_xyxy` | Tile-pixel centroid and half-open bounding box. |
| `representative_point_xy` | Interior segment pixel center nearest the raster centroid, with row-major tie breaking. |
| `touches_tile_boundary` | Whether the region reaches the tile boundary. |
| `is_topology_fragment` | Small derived flag needed by analysis/display filtering. |
| `degree`, `connected_component_id`, `is_isolated` | Graph topology summaries. |
| `analysis_view` | Inclusion flag and exclusion reasons for the analysis/display view. |

SAM predicted IoU/stability, raw model scores, label evidence, model provenance,
explanations, full refinement history, and copied Stage 2 ancestor records are
not stored on graph nodes. Repeated coordinate variants, metrics, and boundary
runs are also omitted. Audit records remain accessible through the graph-level
`source_refinement` path and checksums, plus each node's segment/instance IDs.
This reduces each tile independently; Stage 4 is not responsible for trimming.

## 6. Polygon and coordinate spaces

Pixel conventions remain the same as Stage 2 and Stage 2R:

- indices are zero-based;
- `(x, y)` means `(column, row)`;
- bounding boxes are half-open; and
- centroids and representative points use pixel-center coordinates.

`graph.json` stores each exact polygon only in tile-pixel coordinates. Source
pixel coordinates follow from the tile offset; projected/WGS84 coordinates can
be derived using the affine transform and CRS in `graph.tile_metadata`.
`nodes.geojson` still exports exact polygons in the best available coordinate
space (WGS84, projected, source pixels, then tile pixels). Missing
georeferencing never causes invented world coordinates.

The current geometry contract requires:

```yaml
geometry:
  simplification_tolerance: 0.0
```

Nonzero simplification is rejected. Exact pixel-edge
geometry is important for reproducible shared-boundary measurements and later
cross-tile reconciliation.

## 7. Edge metadata

Each edge retains IDs and representative relationship measurements:

| Field | Meaning |
|---|---|
| `edge_id` | Deterministic tile-qualified edge identifier. |
| `source`, `target`, `source_instance_id`, `target_instance_id` | Canonically ordered endpoints. |
| `same_semantic_class` | Whether endpoint classes match; class details are read from the nodes. |
| `shared_boundary_side_count`, `left_right_contact_count`, `top_bottom_contact_count` | Exact shared-side counts. |
| `shared_boundary_length_pixels`, `shared_boundary_length_projected` | Pixel and optional projected lengths. |
| `contact_fraction_of_source_pixel_perimeter`, `contact_fraction_of_target_pixel_perimeter` | Contact strength relative to each region's perimeter. |
| `contact_centroid_tile_pixel_xy` | Measured contact centroid used to route graph edges. |
| `centroid_distance_pixels` | Distance between region centroids. |
| `analysis_view` | Inclusion flag and exclusion reasons. |

Repeated relation/adjacency descriptions, endpoint class copies, nested contact
fractions, and contact coordinate variants are omitted. The adjacency rule is
stored once at graph level. Projected lengths share the unit and CRS in
`graph.projected_measurements`; numeric arrays store these once as
`projected_length_unit` and `projected_crs`. The adjacency CSV uses the same
graph-level reference.

Stage 3 creates no distance, containment, similarity, road-connectivity, or
semantic-association edges. Additional relation types require a future schema.

## 8. Graph-level metadata and validation

`graph.json` contains three top-level parts:

```text
schema_version
graph
nodes
edges
```

The `graph` mapping records the tile and graph identity, `directed=false`,
`multigraph=false`, the adjacency definition, node/edge counts, isolated-node
and connected-component counts, semantic class counts, coordinate reference,
configuration hash, and `source_refinement` provenance/checksums. Creation time
is recorded in the public manifest.

The top-level graph counts are always the authoritative counts. Each node and
edge in `graph.json` carries `analysis_view.included` and
`analysis_view.exclusion_reasons`, while `graph.analysis_view` records the
selected node/edge counts and thresholds. Stage 3 materializes that exact
selection as `graph_analysis.json` and `graph_analysis_arrays.npz`; it is a
checksum-bound subgraph of `graph.json`, not a second adjacency computation.
For rendered views,
`report.json.visualization_graph` separately records
`authoritative_node_count`, `authoritative_edge_count`, `displayed_node_count`,
`displayed_edge_count`, and the corresponding suppressed counts and thresholds.
A lower displayed count therefore means "hidden from the audit picture," not
"removed from the graph."

Before export and when reloading, Stage 3 validates that:

- node and edge counts match their arrays;
- node IDs and instance IDs are unique;
- instance IDs remain contiguous;
- compact nodes/edges contain only the permitted graph fields;
- each node's exact polygon agrees with its area, perimeter, and interior point;
- source audit references and artifact checksums remain consistent;
- every edge endpoint exists;
- no self-loop or duplicate edge exists; and
- every edge has at least one qualifying shared pixel side.

## 9. Configuration

Stage 3 uses the root `config.yaml`:

```yaml
stage3:
  input:
    source: refinement

  adjacency:
    method: shared_edge
    connectivity: 4
    minimum_shared_boundary_pixels: 1

  geometry:
    simplification_tolerance: 0.0

  visualization:
    random_seed: 42
    show_ids: false
    node_size_min: 12.0
    node_size_max: 72.0
    edge_width_min: 0.5
    edge_width_max: 3.0
    edge_alpha: 0.35
    minimum_display_node_area_pixels: 1
    minimum_display_unknown_area_pixels: 41
    minimum_display_topology_fragment_area_pixels: 41
    minimum_display_shared_boundary_pixels: 2
    route_edges_through_contacts: true
    show_contact_anchors: false

  output:
    root_directory: artifacts/stage3
```

The adjacency method, connectivity, and simplification tolerance are fixed by
the version 2 contract. Visualization ranges control area-scaled node sizes,
shared-boundary-scaled edge widths, edge opacity, and optional ID labels.
Semantic colors and all rendering order are deterministic. Labels are off by
default because real tiles can contain hundreds of nodes.

The four `minimum_display_*` settings define the preferred analysis/display
view without changing the authoritative graph:

- all classes below `minimum_display_node_area_pixels` are excluded;
- `unknown` nodes below `minimum_display_unknown_area_pixels` are excluded while
  equally small known objects remain visible; and
- the topology-fragment threshold remains a defensive visualization guard for
  older refinement products; current Stage 2R products already consolidate
  topology-only secondary pieces below the object-promotion floor; and
- edges below `minimum_display_shared_boundary_pixels` shared sides are
  excluded.

With the defaults, all known nodes remain included, unknown nodes through 40
pixels are excluded from the normal analysis/display view, native small known
objects are preserved, and a one-side adjacency is excluded from that view.
`route_edges_through_contacts: true`
draws each displayed edge as a polyline through the measured shared-boundary
contact centroid instead of as a misleading straight centroid-to-centroid
chord. Contact-anchor dots are off by default, but the routed edge still passes
through the actual contact. These options do not change graph IDs, exact
polygons, adjacency records, degrees, components, or full exports; the same
selection is stored explicitly in each record's `analysis_view` field.

Relative paths are resolved against `--project-root`. The refinement root is
derived from `stage2.output.root_directory` and the fixed phase directory
`13_refinement`.

## 10. Running Stage 3

When Stage 2 is complete, the direct command applies and validates configured
refinement, then builds the graph:

```powershell
python -m src stage3 run --tile-id IKONOS_2005-002__tile_00003 --refine-first
```

To run every step separately instead:

```powershell
python -m src stage2 run --tile-id IKONOS_2005-002__tile_00003
python -m src refine apply --tile-id IKONOS_2005-002__tile_00003
python -m src refine validate --tile-id IKONOS_2005-002__tile_00003
python -m src stage3 run --tile-id IKONOS_2005-002__tile_00003
```

Reviewed semantic decisions can be supplied to `refine apply` with
`--decisions`; omitting them still applies internal four-connected analysis,
micro-component identity consolidation, and all eligible replay-validated
`micro-unknown-v1` and `contextual-micro-v1` absorbs,
but no review-gated operation. In the one-command form, pass the same document
with `--refinement-decisions <path>`. `refine apply` regenerates proposals, so a
separate `refine propose` command is optional unless you want the review bundle.

Use an alternate project root or unified configuration when needed:

```powershell
python -m src stage3 run `
  --tile-id scene__tile_00000 `
  --project-root . `
  --config config.yaml `
  --refine-first
```

Add `--no-progress` to disable the progress bar.

The Python API is:

```python
from src.stage3 import load_stage3_result, run_stage3

result = run_stage3(
    "scene__tile_00000",
    project_root=".",
    config="config.yaml",
)
print(result.summary())

reloaded = load_stage3_result(
    "artifacts/stage3/scene__tile_00000/01_graph"
)
print(reloaded.summary())
```

## 11. Output artifacts

The default output directory is:

```text
artifacts/stage3/<tile_id>/01_graph/
```

It contains:

```text
01_graph/
  graph.json
  graph_arrays.npz
  graph_analysis.json
  graph_analysis_arrays.npz
  adjacency.csv
  nodes.geojson
  graph_overlay.png
  graph_spatial.png
  graph_spatial.svg
  stage3_overview.png
  refinement_graph_before_after.png
  report.json
  manifest.json
  phase_manifest.json
```

### `graph.json`

The authoritative, versioned graph document contains compact node/edge
features and one exact tile-pixel polygon per node. It excludes SAM, semantic
model, and refinement audit payloads.

### `graph_arrays.npz`

A compact, pickle-free representation for downstream numeric processing. Its
arrays include:

- `node_ids`, `instance_ids`, `class_ids`, and `area_pixels`;
- `representative_points_tile_xy`;
- `edge_index` with shape `(2, E)` using zero-based node-array positions;
- `endpoint_instance_ids`;
- `shared_boundary_side_count`;
- `shared_boundary_length_pixels`; and
- `shared_boundary_length_projected`, using `NaN` when unavailable.

`graph.json` remains authoritative for graph features and tile-pixel geometry.

### `graph_analysis.json` and `graph_analysis_arrays.npz`

The explicit low-noise graph used by the normal visualizations and recommended
for noise-sensitive downstream analysis. It contains only records whose
`analysis_view.included` flag is true, preserves their authoritative IDs and
metadata unchanged, and records the SHA-256 checksum and identity of its source
`graph.json`. Its graph metadata contains topology statistics recomputed for
the selected subgraph. The companion arrays use the same fields as
`graph_arrays.npz`; `edge_index` is indexed against the selected node array,
while `instance_ids` retain their original (possibly non-contiguous) refined
IDs. `graph_view="analysis"` and a separate analysis schema version prevent the
two array products from being confused.

### `adjacency.csv`

A human-inspectable edge table with endpoint IDs, same-class status,
orientation-specific contact counts, and pixel/projected boundary lengths.

### `nodes.geojson`

Exact node polygons for GIS use. The collection records its coordinate space
and CRS. Its properties are intentionally convenient and compact; use
the original Stage 2/2R artifacts for model diagnostics and refinement ancestry.

### Visualizations

- `graph_overlay.png` draws adjacency edges and nodes over the segmented tile.
- `graph_spatial.png` and `graph_spatial.svg` show the same spatial graph
  without the image background.
- `stage3_overview.png` places segmentation and graph views side by side for
  direct inspection.
- `refinement_graph_before_after.png` reconstructs the adjacency graph from
  immutable `12_final` and places it beside the refined Stage 3 graph. Both
  panels use their semantic maps as backgrounds, so purple consistently means
  the explicit `unknown` class. The same display thresholds and graph style are
  applied to both phases.

Nodes use tile-coordinate spatial positions, not a force-directed layout, so
their placement corresponds to image components. Node colors represent semantic
classes, node sizes represent area, and edge widths represent shared-boundary
strength. By default the renderer suppresses `unknown` nodes through 40 pixels
and edges supported by only one shared side. Topology-only secondary pieces
through 40 pixels are already part of their parent object rather than separate
nodes. A displayed edge passes through
the measured contact centroid even though the optional contact-anchor dot is
off by default. This makes the spatial reason for adjacency inspectable even
for concave or irregular segments.

These PNG/SVG files are readable audit products, not authoritative graph data.
The graph in the right-hand comparison panel is the authoritative refined
graph; the left-hand graph is a visual-only reconstruction recorded under
`report.json.refinement_graph_comparison` with before/after node, edge, unknown
node, and unknown-pixel counts.
Use `graph_analysis.json` for the exact graph shown by the normal view and use
the full node/edge counts in `graph.json` for audit. Inspect each record's
`analysis_view` or `report.json.analysis_graph` when you need to explain the
selection.

### `report.json` and `manifest.json`

`report.json` summarizes authoritative and analysis topology, class
composition, adjacency counts, isolates, connected components, validation
results, and displayed graph counts. `manifest.json`
records schema/status, configuration hash, output paths, creation metadata, and
`source_refinement`; it indexes and checksums both graph forms and records their
separate counts. `phase_manifest.json` records checksummed refinement inputs and
Stage 3 outputs.

## 12. Limitations and the Stage 4 boundary

Stage 3 deliberately remains tile-local:

- overlapping tiles can contain duplicate observations of the same real object;
- objects cut by a tile edge remain separate, truncated nodes;
- no edge crosses from one tile graph to another;
- node IDs are tile-qualified, not globally reconciled object IDs; and
- per-tile coverage does not prove source-wide topological correctness.

Stage 4 combines graphs only after reconciling overlapping segment geometry
and provenance. It matches candidates in exact source-pixel space, resolves
duplicate ownership, preserves all contributing tile/segment IDs,
assign stable global IDs, and then rebuild or merge adjacency without double
counting overlap seams.

Other current limitations are:

- graph quality inherits every boundary, semantic, and confidence error that
  remains after refinement;
- four-neighbor adjacency intentionally excludes corner-only contact;
- projected boundary lengths are unavailable without reliable georeferencing;
- polygons follow the raster pixel grid rather than an independently surveyed
  vector boundary; and
- analysis/display suppression keeps normal views readable but does not remove
  unresolved segments or weak contacts from the authoritative graph; and
- labels are disabled by default because real tiles can contain hundreds of
  authoritative nodes.

Stage 4 uses the authoritative `graph.json` and refinement raster because it
must preserve exact object identity and overlap geometry. Use
`graph_analysis.json` or `graph_analysis_arrays.npz` for low-noise downstream
analytics after combination, while retaining `graph.json`, `graph_arrays.npz`,
and the refinement manifest as audit inputs. Follow
`source_refinement` and the node segment/instance IDs when
an audit must continue back to immutable Stage 2. Do not infer cross-tile
identity from similarly colored preview pixels or from tile-local node IDs
alone.
