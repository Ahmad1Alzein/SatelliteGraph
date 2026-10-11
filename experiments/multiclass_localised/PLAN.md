# Experimental pipeline: multi-class localised change regions (proposal, not implemented)

**Status:** diagnosis done, architecture proposed, awaiting approval.
- **Separate:** this is separate from the frozen legacy building ST-GNN (`pipelines/legacy_building_stgnn`) and from
  the current multi-class pipeline (`python -m src pipeline`), and changes neither.
- **Branch:** `experiment/oversized-land-2013` (local).

## 1. Diagnosis (7 frozen tiles, 2013 → 2023)
**Scripts:**
- `experiments/oversized_land_2013/investigate.py`
- `experiments/oversized_land_2013/diagnose_multiclass.py`
- `experiments/oversized_land_2013/render_disagreement.py`

**Outputs:** `artifacts/experiments/.../diagnosis/`. All read-only on the frozen study.

| # | Finding | Evidence |
|---|---|---|
| 1 | **Oversized objects in every vegetation class, both years**, caused by the final merge (union-find chains, no size cap) re-joining the ≤ 3,000 m² split pieces. | Objects > 1 ha before → after merge: grass/shrub 2 → 28 (2013), 8 → 37 (2023); cropland 5 → 8 and 6 → 12; tree 2 → 6 and 0 → 5. 68 % of cropland area and 36–41 % of grass/shrub area lie in objects > 1 ha. Buildings: 3 merged building blobs > 1 ha in 2013 (max 2.5 ha). |
| 2 | **Whole-object labels are far coarser than the change.** | Legacy 2013 `built_over`: 7.6 % of the labelled area changed. Improved v1 node states: 27 % (60,431 of 223,846 m²). Repo multi-class: candidate event regions are localised, but its GNN targets are whole nodes. Source nodes of the candidates are 4.1–5.1× the event area. Small changes inside large nodes can never be positive, because of the fraction rule. |
| 3 | **The two years' segmentations disagree on 49 % of the area** (1.86 of 3.81 km²). Most of this is not real change. | Of the disagreement: boundary band ≤ 1.5 m 55 %, vegetation-internal swaps (tree / grass / crop / bare) 29 %, building lean zone ≤ 6 m 7 % (an upper bound; it also contains some real new buildings), other built transitions 9 %. Same conclusion as the earlier shared-atoms PoC (NO-GO: about half the area flips class). |
| 4 | **Partial change is the norm, not the exception.** | Many objects have 5–75 % of their interior assigned another class in the other year: 2023 grass/shrub 306 of 699, 1.35 km². Labelling an object "changed" or "unchanged" as a whole is wrong for these. |
| 5 | **Localised, evidence-gated regions work, but cover too little.** | Improved v2 change regions: point precision 0.99, recall 0.022. Building evidence layer: precision 0.97. Both are assistant-annotated checks, not independent ground truth. |

**Conclusion:**
- **Change cannot be inferred from class labels.** Neither object labels nor pixel labels from two independently
  produced segmentations will do, because their classes are not consistent across dates (finding 3).
- **The change unit must be a localised region.** It should be supported by bi-temporal image evidence, attached to
  the objects it comes from and goes to, and given an explicit status that separates real change from season,
  segmentation, registration and uncertainty.

## 2. Proposed architecture: evidence-first change regions on a three-layer graph

```
2013 objects ──spatial──  (frozen, unchanged; identities kept)
   │  temporal links (identity / contained_in / overlap; split / merge / n:m entities)
2023 objects ──spatial──
   ▲ from            ▲ to
   └── change regions (NEW node type) ── adjacency ── change regions
```

**A. Object layer (unchanged).**
- Frozen per-year objects, spatial edges, temporal links and split / merge entities.
- Optional variant, evaluated separately: a size-capped merge (no merge above the split threshold), to report how
  much the merge artefact matters.

**B. Evidence layer (new; per pixel, per class family, at 0.75 m, optionally native 0.5 m).**
- Bi-temporal class probabilities (not labels) of both years.
- Image change: colour, greenness, texture and edge difference.
- Masks:
  - boundary band (registration ± 1.5 m);
  - roof-lean zone (from the existing roof-offset field);
  - shadow;
  - optional third date (2017 imagery, as evidence only) for seasonal persistence.

**C. Change regions (new node type).**
- **What they are:** connected components of transition evidence, at or above the 25 m² minimum mapping unit. They
  are found per transition family:
  - built gain / loss (vegetation → building, bare → paved, agriculture → road, …);
  - vegetation gain / loss (tree / grass / crop);
  - bare ↔ vegetated;
  - water.
- **Attributes of each region:**
  - geometry and area;
  - from-class and to-class, from mean probabilities inside the region;
  - evidence scores;
  - links: `from` edges to the overlapping 2013 objects and `to` edges to the 2023 objects, with overlap areas
    (full traceability);
  - **status**, assigned by a documented cascade:
    1. `registration_boundary` (inside the boundary band only);
    2. `roof_lean` (explained by the roof offset);
    3. `segmentation_inconsistency` (labels differ but probabilities / appearance do not);
    4. `season_sensitive` (vegetation-internal, phenology-only signal, not confirmed by structure or a third date);
    5. `confirmed_change` (strong two-sided evidence);
    6. `uncertain` (everything else).
- **Objects are never labelled wholesale.** Instead each object gets `changed_fraction`, its list of contained
  regions, and a derived state: unchanged / partially changed (fraction) / converted (≥ 80 %) / split / merged.

**D. Graph and GNN** (compatible with the repository's relation-typed RGNN).
- **Nodes:** 2013 objects, 2023 objects, change regions.
- **Relations:** existing spatial and temporal relations, plus `region_from`, `region_to` and `region_adjacent`
  (both directions).
- **Heads, on region nodes:**
  - real change vs not (status);
  - transition family;
- **Head, on object nodes:** changed fraction.
- **Training:** leave-one-tile-out (development tiles 0–3, held-out tiles 4–6). Comparisons: no edges, spatial only,
  full.

**E. Outputs.**
- Per-class and combined change maps for 2013 and 2023: regions coloured by transition, outlined by status.
- Traceability tables: region → source / target objects.
- Per-object partial-change tables.
- Per-tile overlays in the legacy figure style.

## 3. Evaluation design
| Measure | How | Evidence type |
|---|---|---|
| Class- and transition-specific precision / recall | Stratified sample (transition family × status × tile) annotated blind by 2 people (extending the existing 183-unit kit), with Cohen's κ, stratum-weighted estimates and 95 % CIs | **independent human** |
| Area-based | Flagged-area precision ("localisation ratio"), area estimates with CIs (good-practice stratified area estimation), boundary-band sensitivity | human sample + maps |
| Transition consistency | Region from / to classes agree with the linked 2013 / 2023 objects. Implausible transitions are flagged. Regions agree with split / merge entities. Results are stable to the band and minimum-unit settings | internal consistency |
| Status validity | Share of human-confirmed real change per status (should be high for `confirmed`, low for `registration_boundary` / `segmentation_inconsistency`) | independent human |
| Baselines | Legacy building ST-GNN (building families only, frozen), current repo multi-class, improved v1 / v2 | same sample |

Pseudo-label agreement is reported separately and never as accuracy. No tuning on held-out tiles 4–6.

## 4. Phases, with GO / NO-GO gates (criteria written before results)
| Phase | Content | Gate |
|---|---|---|
| P0 | Protocol: taxonomy, status cascade, MMU / band values, annotation design, dev / held-out split | your approval |
| P1 | Evidence layer and disagreement categorisation on dev tiles. Small human pilot (about 60 units) | statuses separate real from not-real better than class differencing (pre-set κ / precision thresholds) |
| P2 | Change regions, object links, graph schema, maps and traceability. Size-capped merge variant | localisation ratio and transition consistency improve vs v1 / legacy without recall loss on the pilot |
| P3 | Full human annotation (multi-class kit) and region pseudo-labels | κ ≥ pre-set level |
| P4 | Region-level RGNN, ablations, leave-one-tile-out | full graph > ablations on human labels |
| P5 | Held-out evaluation, figures, thesis tables, limitations | — |

## 5. Decisions needed from you
1. **Object layer base:** the frozen original segmentation (keeps the legacy comparison exact), or the improved
   native-0.5 m buildings variant? Recommended: the frozen base first, improved buildings as a variant.
2. **Third date:** may the 2017 imagery be used as evidence for seasonal / persistent change? Recommended: yes,
   evidence only.
3. **Human annotation:** who can annotate, and how much time (pilot about 1 h; full about 3–4 h per annotator)?
   Without it, accuracy stays unmeasured.
4. **Taxonomy:** keep 8 classes, with transitions grouped into the families above, and a 25 m² minimum mapping unit?
