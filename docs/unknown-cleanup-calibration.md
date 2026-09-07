# Unknown-region cleanup and the combined graph

The September 7 four-tile baseline contained 1,082 unknown objects after
refinement; 941 were at most 40 pixels, and 423 were single pixels. Stage 4
previously drew the full audit graph, so it also exposed unresolved tiny
objects that the Stage 3 analysis view already excluded.

## Actual semantic cleanup

Refinement still examines four-connected unknown components up to 40 pixels.
At this image's 0.8 m pixel spacing, that ceiling corresponds to 25.6 square
metres. It is a candidate-selection limit, not proof that an object is false;
small buildings, trees, roads, and strong class boundaries need protection.
Other resolutions require recalibration of this pixel threshold.

The existing energy combines model probability, shared-boundary fraction,
boundary color similarity, and whole-region color similarity. Its ordinary
ceiling remains 0.40. A supported extension permits energy up to 0.50 only if:

- the target class is among the source's top two mean model probabilities;
- mean target probability on source pixels is at least 0.20;
- the target touches at least 25% of the source perimeter;
- boundary color similarity and region color similarity both reach 0.50;
- the best target beats the runner-up by at least 0.02 energy units; and
- border abstention, field/building uniqueness, road evidence, and standalone
  object-preservation checks also pass.

Strong semantic-consensus proposals retain their existing independent path.
Both proposal generation and replay enforce the extension. Each applied
correction records the threshold snapshot and original evidence, and assigns
the absorbed pixels the target's saved model probability rather than invented
high confidence. The node-count budget remains in force.

## Threshold evidence

The read-only sweep in `scripts/audit_unknown_cleanup.py` saves individual
evaluations and summaries in `artifacts/unknown_cleanup_audit.json`. On the
four-tile baseline, predicted eligible absorbs were:

| Tile | Original 0.40 | Supported 0.45 | Supported 0.50 | Supported 0.55 |
|---|---:|---:|---:|---:|
| 00000 | 192 | 332 | 383 | 386 |
| 00001 | 215 | 357 | 388 | 393 |
| 00002 | 185 | 248 | 270 | 270 |
| 00003 | 311 | 542 | 601 | 607 |

The 0.50 ceiling captures most of the increase before the observed plateau.
These are evidence-gated candidate counts, not measured accuracy. Labeled
validation tiles are needed to estimate false corrections or claim an optimal
threshold. Increasing the area ceiling above 40 was unnecessary for the
dominant single-pixel and small-fragment problem.

## Combined analysis view

The normal Stage 4 overlay and `graph_analysis.json` now omit unresolved
unknowns below 41 pixels and edges with fewer than two shared sides. The test
uses final combined area after overlap matching. Small known objects stay in
the graph, and no replacement links are invented across omitted nodes.

`graph.json` and `combined_graph_overlay_audit.png` retain all remaining
unknown regions and their original uncertainty. The report explicitly separates
this display/analysis selection from actual semantic absorption. Thus a clear
normal graph does not imply every ambiguous pixel has been classified.

## Verified rerun results

All four refinement outputs passed coverage and topology validation with no
gaps, semantic/mask mismatches, or disconnected instance IDs. Across the four
separate tiles, the previous refinement's 1,921 objects became 1,597; unknown
objects decreased from 1,082 to 744. An additional 2,621 unknown pixels were
actually assigned to evidence-supported neighboring regions. These totals
include overlapping tile pixels before Stage 4 ownership resolution.

| Combined selection | Previous full nodes | Updated full nodes | Analysis nodes | Analysis edges | Retained analysis unknowns |
|---|---:|---:|---:|---:|---:|
| Four tiles, 00000–00003 | 1,751 | 1,457 | 906 | 2,045 | 116 |
| Two tiles, 00002–00003 | 978 | 837 | 525 | 1,218 | 67 |

The four-tile analysis view omits 551 remaining small unknown nodes; this
selection changes no semantic pixels and must not be counted as successful
classification. The full graph retains these unresolved regions for audit.
The two-tile view similarly omits 312. Both combined outputs pass validation
and assign overlapping pixels once. The test suite passed all 155 tests.

Run `python -m scripts.report_unknown_cleanup` to regenerate the saved
comparison JSON and three-panel visual in `artifacts/unknown_cleanup_comparison`.
Its panels show the previous full graph, updated full graph, and updated
analysis graph, so actual cleanup and analysis filtering can be judged separately.

## Research sources

[GDAL raster sieving](https://gdal.org/en/stable/programs/gdal_sieve.html)
establishes small connected-region replacement by a neighboring polygon.
[Region adjacency graph merging](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_rag_merge.html)
adds appearance-based neighbor selection. Our implementation combines these
ideas with saved semantic evidence, an ambiguity margin, and class-specific
object checks.

[Kampffmeyer et al., CVPR Workshops 2016](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w19/papers/Kampffmeyer_Semantic_Segmentation_of_CVPR_2016_paper.pdf)
addresses small objects and uncertainty in remote-sensing segmentation. Its
motivation supports protecting meaningful small objects; it does not prescribe
our numerical thresholds. No source establishes a universal best pixel area
for different images and resolutions.
