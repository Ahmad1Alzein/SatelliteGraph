# Class-specific temporal change events

Goal: add explicit class-specific change edges to Stage 5 for all nine classes,
including construction/demolition, appearance/disappearance, and extent change.

Design: preserve correspondence/split/merge edges and add a separate directed
`change_edges` collection referencing `change_events`. Each event has its own
class, geometry, area evidence, and source/target observation IDs. Endpoints
are always real observed regions: construction connects prior land cover to
the new built region; demolition connects the old built region to replacement
land cover. No fictional absence nodes or backward edges.

Compare same-class overlap components as unions to avoid mistaking segmentation
splits for area change. Compare exact polygons across all classes, independently
of identity-match thresholds, to detect changes within large background regions.
Keep gross gain/loss as well as net change, with configurable absolute/relative
thresholds and a registration boundary tolerance. Whole-object events require
sufficient common coverage and reliable known-class evidence. Unknown and low
confidence observations cannot establish physical presence/absence.

Physical labels are inferred candidate events, not ground-truth causes. With
two dates, construction/demolition refers to mapped footprint gain/loss; road
width/length, building height, tree planting/felling, flood/drought, or permanent
water disappearance cannot be inferred reliably from these graphs alone.

- [x] Add failing tests for all known classes, unknown transitions, expansion,
  contraction, simultaneous gain/loss, pure splits, missing coverage, low
  confidence, uncertainty bands, and changes without identity correspondences.
- [x] Implement isolated event policy and polygon-based event detection.
- [x] Integrate schema v2, preserve v1 loading, export event arrays and GIS
  geometries, summarize and visualize changes, configure via config.yaml.
- [x] Run Stage 5 suites and document the event vocabulary, rules, and evidence.

Validation: 46 tests pass, including all 81 class transitions. A focused review
identified uncertain pixels being included in supported physical-change edges;
regression tests now enforce known/reliable endpoints and clipped evidence areas.
The synthetic example exports and reloads 11 event groups successfully. Real
cross-year validation remains pending a second-year Stage 4 graph.
