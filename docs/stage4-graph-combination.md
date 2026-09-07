# Stage 4 multi-tile graph combination

## 1. Purpose

Stage 4 combines any chosen set of completed Stage 3 tile graphs from one
source image into a single undirected graph. It does not concatenate JSON files
or trust centroid proximity. It uses exact Stage 1 source-pixel offsets and the
refinement rasters behind each graph to:

1. place every tile in one source-image coordinate grid;
2. identify duplicate segment predictions inside tile overlaps;
3. give every overlapping source pixel to exactly one tile prediction;
4. dissolve matched predictions into one combined object identity; and
5. rebuild adjacency from the resulting non-overlapping global geometries.

The output schema is `satellitegraph.stage4.combined-graph.v1`.

## 2. Run any selected number of tiles

Repeat `--tile-id` for exactly the Stage 3 graphs to combine:

```powershell
python -m src stage4 combine `
  --tile-id IKONOS_2005-002__tile_00000 `
  --tile-id IKONOS_2005-002__tile_00001 `
  --tile-id IKONOS_2005-002__tile_00002 `
  --name IKONOS_first_three_tiles
```

The default name is a deterministic hash of the sorted selection. The Python
API is equivalent:

```python
from src.stage4 import combine_tile_graphs

result = combine_tile_graphs(
    [
        "IKONOS_2005-002__tile_00000",
        "IKONOS_2005-002__tile_00001",
        "IKONOS_2005-002__tile_00002",
    ],
    output_name="IKONOS_first_three_tiles",
)
print(result.summary())
```

Every selected tile must already have a complete, checksum-valid
`artifacts/stage3/<tile_id>/01_graph`. Tiles must belong to the same source
image and share its affine transform and CRS. Non-adjacent selections are
allowed; the combined graph may then have multiple connected components.

## 3. Overlap reconciliation algorithm

### 3.1 Coordinate placement

`global_x0`, `global_y0`, width, and height place every local pixel in the
source-image pixel grid. The source affine transform is carried into the
combined graph so source pixels can be transformed to projected coordinates.
This follows GDAL's six-coefficient pixel-to-world affine model.

### 3.2 Duplicate candidates

Only tile pairs with a positive-area spatial overlap are compared. Within the
exact shared window, Stage 4 counts intersections between the two dense
instance maps. A candidate records:

- intersection pixels;
- intersection over union (IoU);
- intersection over the smaller visible region (IoS);
- coverage of each segment's overlap-visible pixels;
- semantic agreement; and
- a deterministic combined match score.

Default automatic matching requires at least four shared pixels, equal semantic
class, IoS `>= 0.60`, IoU `>= 0.20`, and coverage `>= 0.40` in both directions.
Unknown-to-unknown matches use the stricter IoS floor `0.80`. These are
overlap-local metrics: full-object IoU would unfairly penalize one tile for
seeing more of an object outside the shared window.

### 3.3 Mutual-best and transitive safety

A threshold-passing pair is merged only when each prediction is the other's
best candidate for that tile pair. Candidates are processed from strongest to
weakest with union-find, but a group may never acquire two predictions from the
same tile. This blocks an unsafe transitive chain from collapsing two distinct
same-tile objects through inconsistent predictions in neighboring tiles.

Cross-class intersections and weak or ambiguous matches are preserved in
`overlap_matches.json` with rejection reasons; Stage 4 never guesses a
cross-class identity.

### 3.4 Exact-once pixel ownership

Matching establishes identity, but it does not by itself remove duplicated
overlap pixels. For every shared source pixel, Stage 4 first prefers a tile
that actually has a valid segment prediction. When both tiles have
predictions, it chooses the tile where that pixel is farthest from the tile
boundary. This prefers the prediction with the greatest image context and
removes edge artifacts. Exact positional ties use semantic confidence,
followed by stable tile-ID order.

The owned masks are polygonized and dissolved. Consequently:

- every selected source pixel is represented at most once;
- final node geometries have zero area overlap;
- a matched object retains the complementary non-overlap portions from all its
  member tiles; and
- adjacency is recomputed from shared source-pixel boundaries instead of
  copying stale seam edges.

## 4. Why this algorithm

The design combines established ideas without claiming a learned guarantee:

- the remote-sensing tiling literature identifies overlap and stitching as a
  necessary part of large-image segmentation;
- SAHI uses IoU/IoS and non-maximum merging to reconcile sliced predictions;
- panoptic segmentation's IoU matching result motivates conservative unique
  correspondences; and
- OGC Simple Features supplies the polygon/multipolygon union model used by the
  exported geometries.

Primary references:

- [Tiling and Stitching Segmentation Output for Remote Sensing](https://arxiv.org/abs/1805.12219)
- [Slicing Aided Hyper Inference](https://arxiv.org/abs/2202.06934)
- [SAHI sliced-inference and overlap postprocessing](https://github.com/obss/sahi/blob/main/docs/guides/sliced-inference.md)
- [Panoptic Segmentation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.pdf)
- [GDAL Geotransform Tutorial](https://gdal.org/en/stable/tutorials/geotransforms_tut.html)
- [OGC Simple Feature Access](https://www.ogc.org/standards/sfa/)

The numerical thresholds are conservative engineering defaults and must be
calibrated on labeled overlap pairs before being presented as an error-rate
guarantee.

## 5. Artifacts

The output directory is `artifacts/stage4/<selection>/01_combined_graph/`:

| Artifact | Purpose |
|---|---|
| `graph.json` | Authoritative combined nodes, edges, membership, source-pixel geometries, and policy metadata. |
| `overlap_matches.json` | Accepted and rejected overlap candidates with exact evidence and reasons. |
| `graph_arrays.npz` | Pickle-free node arrays, source-pixel positions, edge index, and boundary lengths. |
| `nodes.geojson` | Combined exact polygon/multipolygon features in explicitly declared source-pixel coordinates. |
| `adjacency.csv` | Compact combined edge table. |
| `graph_analysis.json`, `graph_analysis_arrays.npz` | Selected combined graph, excluding unresolved unknowns below 41 pixels and edges supported by fewer than two sides. Small known objects remain. |
| `combined_graph.png` | Spatial source-pixel analysis graph overview. |
| `combined_graph_overlay.png` | Ownership-resolved source-tile mosaic with the analysis graph. Cyan dashed rectangles show tile extents. |
| `combined_graph_overlay_audit.png` | Every authoritative node and edge, including remaining unresolved micro-unknowns. |
| `report.json` | Input/output counts, duplicate reduction, conflicts, ownership, analysis exclusions, and validation. |
| `manifest.json` | Configuration, selected tile IDs, paths, and artifact checksums. |

In the overlay, each white adjacency line is routed through its measured
shared-boundary location, shown by a cyan contact marker. This makes irregular
and multipart-object relationships inspectable instead of implying that a
straight line between two representative points is the physical boundary.

Analysis selection happens after overlap reconciliation, using each final
combined object's area. Exclusion changes neither semantic pixels nor the
authoritative graph, and creates no shortcut edges across excluded regions.
Each authoritative node/edge declares `analysis_view.included` and reasons.

Each combined node lists every member tile/node ID and its original and owned
area. The source Stage 3 manifests and refinement maps are checksum-bound as
phase inputs, so a combined result can be audited back to every tile.

## 6. Current boundaries

Stage 4 currently combines tiles from one source image. Cross-date or
cross-sensor mosaics need registration and change detection before the same
identity rules are safe. RGB similarity is intentionally unnecessary for
ordinary overlaps because the compared pixels are already the same source
coordinates; semantic confidence is used only to break ownership ties. Road
continuity inferred across a real missing-data gap is also out of scope—Stage 4
combines observed geometry and never hallucinates a connection.
