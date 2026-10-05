# Discussion guide: how the temporal graph describes change

This guide explains the current SatelliteGraph implementation for discussion
with a supervisor. Each year's **combined tile graph** represents one mapped
image. Nodes are mapped regions; spatial edges connect regions that touch
within that year. A **temporal edge** points from a region in the earlier year
to an overlapping region in the later year. Full-folder and single-image
comparisons use the same temporal-graph builder.

## Read one change edge

Imagine a place labelled **bareland in 2013** and **building in 2023**. If both
observations are reliable, the new building occupies a substantial part of
that place, and the local comparison passes the checks below, the edge from
the old bareland node to the new building node is **building_constructed**.
Reversing the observations gives **building_demolished**. These labels describe
**candidate changes in the mapped footprint**. The images alone cannot prove
the date or physical cause of construction or demolition.

The change is measured at the **same ground location**, after alignment. It
is not decided by comparing two distant node centers. One large old polygon
may contain several new objects; each accepted edge must point to a small,
specific area where its stated change actually occurred.

## Rules for each mapped class

The examples below assume the place was visible in both years, both labels
were reliable, and the change passed the minimum-area and local-evidence checks.
“New” means the class was absent from that local comparison group in the
earlier year; “gone” means it is absent in the later year.

| Class | New class at this place | Class gone from this place | Class in both years |
|---|---|---|---|
| **Building** | Bareland or another known nonbuilding class becomes a building → **building_constructed**. | A mapped building becomes a known nonbuilding class → **building_demolished**. | A building gains footprint → **building_grew**; loses footprint → **building_shrank**; changes outline without clear net gain/loss → **building_footprint_changed**; sufficiently stable footprint → **building_unchanged**. |
| **Road** | Known nonroad ground becomes a road-like segment → **road_constructed**. | A mapped road becomes known nonroad ground → **road_removed**. | The road area grows, shrinks, changes outline, or remains stable → **road_grew**, **road_shrank**, **road_footprint_changed**, or **road_unchanged**. |
| **Water** | Known nonwater ground becomes mapped water → **water_gained**. | Mapped water becomes another known class → **water_lost**. | A water area grows, shrinks, changes outline, or remains stable → **water_grew**, **water_shrank**, **water_footprint_changed**, or **water_unchanged**. |
| **Tree cover** | Known nontree ground becomes tree cover → **tree_cover_gained**. | Tree cover becomes another known class → **tree_cover_lost**. | A canopy area grows, shrinks, changes outline, or remains stable → **tree_cover_grew**, **tree_cover_shrank**, **tree_cover_footprint_changed**, or **tree_cover_unchanged**. |
| **Agriculture** | Known nonagricultural ground becomes mapped agriculture → **agriculture_gained**. | Agriculture becomes another known class → **agriculture_lost**. | Its mapped area grows, shrinks, changes outline, or remains stable → **agriculture_grew**, **agriculture_shrank**, **agriculture_footprint_changed**, or **agriculture_unchanged**. |
| **Developed space** | Another known class becomes developed space → **developed_space_gained**. | Developed space becomes another known class → **developed_space_lost**. | Its mapped area grows, shrinks, changes outline, or remains stable → **developed_space_grew**, **developed_space_shrank**, **developed_space_footprint_changed**, or **developed_space_unchanged**. |
| **Rangeland** | Another known class becomes rangeland → **rangeland_gained**. | Rangeland becomes another known class → **rangeland_lost**. | Its mapped area grows, shrinks, changes outline, or remains stable → **rangeland_grew**, **rangeland_shrank**, **rangeland_footprint_changed**, or **rangeland_unchanged**. |
| **Bareland** | Another known class becomes bareland → **bareland_gained**. | Bareland becomes another known class → **bareland_lost**. | Its mapped area grows, shrinks, changes outline, or remains stable → **bareland_grew**, **bareland_shrank**, **bareland_footprint_changed**, or **bareland_unchanged**. |
| **Unknown** | No physical-change claim. | No physical-change claim. | Unknown is an observation-quality state, not a land-cover change class; transitions involving unknown are withheld from the readable change graph. |

“Water lost” means **not mapped as water on the later image**, not necessarily
permanently dried. “Tree cover gained” does not distinguish planting from
regrowth. “Agriculture gained” does not prove a new field was created. The
labels describe what the two maps support, not the cause.

## How two years are aligned and nodes are matched

1. Stage 4 combines the tiles of each year and converts tile-local pixel
   positions to whole-image positions. Stage 5 puts both years in one
   **metre-based geographic coordinate system**. A plain PNG needs reliable
   georeferencing to support this comparison.
2. If the satellite images are shifted, image comparison can estimate a
   global transform from image features. Folder comparison can use
   **--register-images** or supplied control points; without these it trusts
   the input georeferences.
3. A spatial search finds candidate old/new polygons near one another. Their
   overlap, centroid proximity, area, shape, and neighboring class patterns
   contribute to a similarity score. A normal one-to-one match needs polygon
   overlap (IoU) at least **0.20**, score at least **0.60**, and a score margin
   of **0.05** over competitors. Ambiguous nodes stay unmatched.
4. Already strong building matches act as local landmarks. When at least four
   stable buildings exist, at least three within **125 m** can establish a
   consistent local coordinate offset for another candidate. A second pass
   may recover a missed same-class building or other segment if it still has
   raw overlap, adjusted IoU at least **0.35**, plausible size, and a unique
   match. This offset is used **only to test correspondence**; it never moves
   the saved node or fabricates a change.

The score is an engineering ranking, not a probability of correctness. A very
large or merged building segment may remain unmatched if it lacks a unique
counterpart. Forcing such a pair would create misleading change edges.

## How a change is accepted or rejected

Matching and change detection answer different questions. Matching asks
whether two mapped regions correspond. Change detection compares the area
covered by **each class** at the same location, even if no object match is
accepted. For example, a new building can be detected against old bareland.

The current default checks are:

- At least **95%** of the area under comparison is observed in both years.
- Participating segment confidence is at least **0.60** and at least **95%**
  of the tested area has reliable known-class observations in both years.
- A gain or loss exceeds **max(4 m², 5% of the reference area)**. A **1 m**
  boundary band is discounted when testing this change so small alignment or
  segmentation offsets do not become change claims.
- An old/new edge must overlap the **actual changed area**. Its local support
  must exceed **max(1 m², 25% of the minimum event area)**, 5% of the smaller
  linked polygon, and 8% of their overlap. Narrow slivers are rejected.
- When several old regions compete to explain one new changed region, only
  the strongest local witness is kept. Same-class split/merge claims are
  rejected when their supposed sibling regions are far apart.
- For **road_constructed** or **road_removed**, the road footprint must be
  elongated (long/short side ratio at least **3**) or moderately elongated
  (at least **1.3**) and touch at least **two road nodes** in the same year's
  spatial graph. A compact building-to-road or road-to-building conversion is
  withheld even with road neighbors because roofs and yards can be confused
  with roads. This conservative rule can miss short isolated real roads.
- A same-class pair is called **class_unchanged** only if it has no accepted
  change event, IoU at least **0.50**, and at most **10%** total area difference.

If a class is absent in the earlier local group and present in the later one,
the edge gets that class's new/gained label. The reverse gives its
gone/lost label. If the class exists in both years, a clear net increase is
**class_grew**, a clear net decrease is **class_shrank**, and substantial gain
**and** loss with little net change is **class_footprint_changed**. Weak,
unknown, or coverage-limited evidence leads to **no readable change edge**.

One old/new pair can support more than one class event, such as bareland lost
and building constructed. The displayed edge uses one class-specific label,
prioritizing building, road, water, tree cover, agriculture, developed space,
rangeland, then bareland. All measured events remain in the saved data.

## What to inspect with the professor

The **all_edge_types.png** figure shows displayed labels with a legend; each
**change_<class>_<action>.png** figure shows one type. Small points and lines
mark **local evidence anchors**, which can differ from the centers of large
nodes. **nodes.geojson** stores complete node polygons, **changes.geojson**
stores mapped change areas, and **graph.json** and **report.json** store scores,
decisions, and rejection reasons. These outputs enable review, but precision
and recall require independently labelled reference data.

Implementation: [matching](../src/stage5/matching.py),
[change events](../src/stage5/events.py),
[edge filtering](../src/stage5/refinement.py),
[readable labels](../src/stage5/semantics.py), and
[shared folder/single-image builder](../src/stage5/pipeline.py).
