# Experiment: oversized 2013 land polygons and localised land → building change

**Status:** investigation only (read-only on the frozen study). The legacy building ST-GNN baseline is unchanged.

**Run:**
```
python experiments/oversized_land_2013/investigate.py --study D:\new\DelAnyV2_Experiment\segmentation_v2
```
Outputs are written to `artifacts/experiments/oversized_land_2013/`.

## Findings (7 frozen tiles, 2013 → 2023)
"Converted" area means confirmed new-building evidence pixels from the rule layer. These are pseudo-labels, not ground
truth.

1. **The positive label is far larger than the change.**
   - **Overall:** the 71 2013 land nodes labelled `built_over` cover 362,563 m²; only 27,506 m² of that (7.6 %) is
     converted.
   - **Per node:** median converted fraction 20 %.
   - **Extreme case:** 0.2 % (tile_00006 #475, an 8.8 ha cropland node with 182 m² converted).
   - **Effect on the figure:** this is why the 2013 band of fig5c is dominated by large coloured areas. The GNN's
     positive 2013 nodes cover 330,456 m², while its positive 2023 buildings cover 35,300 m².
2. **Cause 1: the final merge undoes the size cap.**
   - **What the code does:** `segment_final.py` splits land regions above 8,000 m² into pieces of about 3,000 m²
     (`split_above_m2`, `target_region_m2`). `refine_merge.refine` then merges adjacent same-class pieces, pair by
     pair, through union-find chains, with **no size limit**.
   - **Effect:** land polygons over 1 ha go from 9 before the merge to 42 after; the maximum goes from 1.8 ha to
     8.8 ha. Of the >1 ha nodes, 41 of 42 are re-merged edge-split pieces.
   - **Caution:** the per-node `merged` flag under-reports this. Use `segment_ids_before_merge.tif`.
3. **Cause 2: whole-node labelling.**
   - **The rule:** a 2013 land node becomes `built_over` when at least 25 m² of conversion falls inside it, and then
     the entire node is the target.
   - **Even without the merge:** labelling at the pre-merge level would halve the label area (172,758 m²), but only
     15.9 % of that would be converted. Pre-merge pieces of about 3,000 m² are still about 10× a typical new building.
4. **Same problem in 2023:** 2023 land polygons are just as large (54 over 1 ha). The problem shows up in 2013 because
   the built-over target lives on 2013 land nodes.
5. **Sanity check:** conversion recomputed from pixels matches the frozen `converted_m2` within 21 m² for all 71 nodes.

`sheet_largest_built_over.png` shows the eight largest cases:
- **yellow:** the node that carries the label;
- **red:** the converted pixels;
- **white:** the pieces before the merge.

Visually the red areas sit on new structures, while most of each yellow polygon is unchanged land. That is a visual
check only, not ground truth.

## Implication
Fixing the segmentation alone (cap or skip the merge) helps, but it cannot localise the change: the land partition is
coarser than buildings by design. The change itself must become the object: a **conversion region**, the 2013-frame
footprint of a confirmed new 2023 building on 2013 land, as its own node and target.
