# Refinement Stage: Teacher Discussion Document

The numerical examples below describe earlier runs. For the updated four-tile
cleanup, evidence-gated thresholds, and full versus analysis graph comparison,
see [Unknown-region cleanup](unknown-cleanup-calibration.md).

## 1. Purpose

The refinement stage sits between the final Stage 2 segmentation and the Stage
3 graph builder.

Its purpose is to convert a noisy segmentation into a safer set of spatial
entities before those entities become graph nodes. It performs deterministic
topology correction, contracts bounded micro-regions, preserves complete
lineage, and leaves uncertain large regions visible instead of inventing a
class.

> بالعربي: هذه المرحلة تنظّف نتيجة الـ segmentation قبل بناء الـ graph. الهدف
> ليس تغيير الصورة بشكل عشوائي، بل تقليل الأجزاء الصغيرة المزعجة مع المحافظة
> على كل pixel وتسجيل سبب كل تعديل.

The refinement stage never modifies the immutable Stage 2 result in
`12_final`. Its output is written separately to `13_refinement`.

## 2. Entity definition by class

The meaning of one segment depends on its semantic class. Therefore, the same
merge or split rule cannot be applied to every class.

| Class | Intended entity | Refinement policy |
|---|---|---|
| Building | One physical building | Do not merge adjacent buildings without object evidence. A connected split requires an explicit reviewed partition. |
| Tree | One continuous canopy | Preserve continuous canopy; avoid splitting because of shadows or crown texture alone. |
| Road | One junction-to-junction reach | Generic merging across a junction is forbidden. True road decomposition requires junction topology. |
| Agriculture | One field | Same-class fields remain separate unless there is strong evidence that the internal boundary is false. |
| Water, rangeland, bareland | One connected homogeneous region | Merging is possible when the shared boundary is weak and appearance is homogeneous. |
| Unknown | Uncertain valid land | A small unknown may be absorbed only when adjacent evidence supports a target; a large unknown remains unknown. |

> الفكرة الأساسية: `same class` لا تعني دائماً `same object`. مبنيان متجاوران
> أو حقلان متجاوران قد يكونان من نفس الفئة ولكنهما كيانان مختلفان.

### 2.1 Are these rules actually followed?

Yes, they are used as ontology and safety rules, but not every class-specific
segmentation algorithm is complete. The implementation status is:

| Class | Status | How the rule is enforced now | What is not solved automatically yet |
|---|---|---|---|
| Building | Partially implemented, safety enforced | Building-to-building merging is never auto-applied. A generic merge can only become a pending review proposal after strong contact/color gates. Unknown-to-building cleanup requires one unique building target and building probability `>= 0.35`. Strong standalone building evidence is preserved. Reviewed exact split partitions can be replayed. | The system does not yet detect two touching physical buildings and generate their dividing boundary automatically. |
| Tree | Partially implemented | Only disconnected/corner-connected topology is split automatically. No split is generated from canopy texture or shadow. Tree merges remain review-only. Small unknowns may join an adjacent tree, while strong standalone tree evidence is preserved. | It does not yet use a canopy-specific boundary model to prove whether two crowns form one continuous canopy. |
| Road | Protection implemented; decomposition pending | Generic road merge and same-class road absorb are hard-vetoed because they may cross a junction. Small unknown-to-road filling needs road probability `>= 0.30` or road in the top two classes. Output records are explicitly marked `road_reach_candidate`, not confirmed reaches. | Skeletonization, stable junction detection, and junction-to-junction reach partitioning are not automatic yet. |
| Agriculture | Protection implemented; field splitting pending | Generic agriculture merge/absorb is hard-vetoed so equal labels cannot erase a field boundary. Contextual unknown-to-agriculture cleanup requires exactly one agriculture target and source contact `>= 0.75`. | The system does not yet generate new field boundaries inside one connected Stage 2 agriculture segment. |
| Water, rangeland, bareland, developed space | Evidence, review, and selective automatic paths implemented | Direct neighbors are measured using boundary contact and RGB similarity. A reviewer can decide eligible proposals. In automatic mode, an interior same-class merge or small same-class absorb is selected only when the stricter score and every raw-evidence gate pass. Small supported unknowns can also be absorbed into these homogeneous classes. | The automatic thresholds are engineering defaults and still need operation-level calibration on held-out regions. |
| Unknown | Implemented for bounded micro-regions | Interior unknown components `<= 40` pixels are processed by strict or contextual cleanup. Accepted pixels are absorbed/relabelled into a directly adjacent known target. Border, protected-object, and abstained cases remain. Larger unknowns remain unknown. | Large unknown regions are not force-labelled. The contextual rule is heuristic and still requires ground-truth calibration. |

Therefore, “following the rule” currently means two things:

1. the pipeline prevents operations that would clearly violate the class
   definition; and
2. it only performs the class-specific corrections that have enough implemented
   evidence and validation.

It does **not** mean that road junctions, field boundaries, or touching-building
boundaries have already been discovered automatically.

> بالعربي: نحن نطبّق القواعد كـ **حماية وقيود** حتى لا نعمل merge أو split
> خاطئ. لكن بعض الحلول المتخصصة، مثل تقسيم الطريق من junction إلى junction أو
> اكتشاف حدود الحقول والمباني المتلامسة، ما زالت تحتاج algorithm خاص أو مراجعة
> بشرية. النظام لا يدّعي أنه حلّها تلقائياً.

## 3. Complete refinement workflow

### Step 1: Validate the Stage 2 input

The pipeline loads the authoritative Stage 2 `12_final` result and checks:

- manifest schema and completion status;
- masks, records, and raster dimensions;
- contiguous instance IDs;
- semantic consistency inside every instance;
- checksums and source freshness; and
- exact valid-pixel coverage.

### Step 2: Normalize topology

Stage 2 originally allowed 8-connected components. Stage 3 adjacency uses
4-connectivity: pixels must share a full side, not only a corner.

Refinement therefore divides every disconnected or corner-only-connected mask
into deterministic 4-connected components **internally**. Each child receives a
temporary local ID for evidence and replay but retains:

- its Stage 2 parent ID;
- component index and component count;
- source segment metadata;
- coordinates and polygon geometry; and
- an explicit `split_components` lineage record.

After proposal decisions and unknown cleanup, a secondary component must have
at least 41 pixels to become a published object. Smaller topology-only pieces
are consolidated back into the largest surviving sibling from the same Stage 2
object. Their pixels remain exact and appear as a multipart object, but they do
not create extra graph nodes. Crossing the 41-pixel floor makes a component
eligible, not automatically promoted: a hard budget consolidates the weakest
eligible pieces when needed so topology alone cannot publish more objects than
Stage 2. Only explicit approved semantic splits add to that budget. This
identity operation does not change semantic pixels.

### Step 3: Measure adjacency and similarity

Only directly adjacent segments are compared. Two segments are adjacent when
at least one pixel side is shared. Diagonal contact is excluded.

For every adjacent pair, the system measures:

1. shared boundary geometry;
2. color similarity across the exact shared interface;
3. whole-region mean-color similarity; and
4. saved semantic-model probabilities on the candidate pixels.

### Step 4: Refine micro-unknown regions

Only four-connected `unknown` regions with area at most 40 pixels are eligible
for automatic cleanup. This 40-pixel value is a hard safety ceiling in the
current version.

The evaluator produces one of four outcomes:

- `semantic_consensus`: semantic and spatial evidence strongly agree;
- `contextual`: the best adjacent target minimizes the contextual energy;
- `preserved_object`: evidence supports a standalone tree or building; or
- `abstain`: evidence is missing, contradictory, unsafe, or on the tile border.

An accepted region is **absorbed**, not deleted. Its pixels join the selected
adjacent entity, its semantic label and confidence are updated, and its source
lineage remains recorded.

### Step 5: Review or selectively automate pending operations

`human_review` leaves pending generic proposals for an explicit reviewer.
`automatic` may select only the conservative same-class homogeneous
merge/absorb subset after all raw gates pass; a failed gate records
`uncertain` and leaves the region unchanged. Risky operations remain protected:

- road and agriculture merges have hard ontology vetoes;
- building merges require object-level review;
- tree and unknown are protected from generic automatic merge/absorb;
- generic cross-class absorption is human-review-only;
- a connected semantic split requires an explicit, checksum-bound partition;
- the split children must be connected, disjoint, gap-free, and exactly cover
  the parent; and
- valid pixels cannot be changed to nodata merely to make the graph cleaner.

### Step 6: Rebuild and validate everything

After applying accepted operations, the pipeline rebuilds:

- instance and semantic maps;
- confidence and source maps;
- exact masks and polygons;
- segment records and coordinates;
- operation history and parent/child lineage; and
- visual and machine-readable reports.

The final invariant is:

> Every valid pixel belongs to exactly one refined segment, with no gap and no
> overlap.

## 4. How similarity is calculated

All similarity values are normalized to the interval `[0, 1]`:

- `1` means very similar or very strong support;
- `0` means very different or no support.

These values are deterministic evidence scores. They are **not** calibrated
probabilities that a refinement decision is correct.

### 4.1 Shared-boundary similarity

Let:

- `S` be the number of pixel sides shared by the source and target; and
- `P_source` be the raster perimeter of the source, measured in pixel sides.

The source contact fraction is:

```text
B = S / P_source
```

A large `B` means that much of the small source region is enclosed by or
attached to this target.

Example: if a source has perimeter 8 and shares 6 sides with a target:

```text
B = 6 / 8 = 0.75
```

### 4.2 Color similarity across the exact interface

RGB is first normalized to `[0, 1]`. For each shared pixel side, the color
difference between the opposing source and target pixels is measured with
normalized Euclidean distance:

```text
d_interface = ||RGB_source - RGB_target||₂ / sqrt(3)
```

The mean interface difference is converted to a similarity:

```text
C_boundary = 1 - mean(d_interface)
```

A high `C_boundary` means the boundary is visually weak: pixels immediately
across it look similar. This supports absorption or merge. A low value means
there is a visible color edge, which supports keeping the regions separate.

> بالعربي: نقارن لون الـ pixels على جانبي الحد مباشرة. إذا كان الفرق صغيراً،
> فالحد قد يكون ناتجاً عن segmentation noise وليس حداً حقيقياً.

### 4.3 Whole-region mean-color similarity

The average RGB vector is calculated for each region. Their normalized distance
is:

```text
d_region = ||meanRGB_source - meanRGB_target||₂ / sqrt(3)
```

The similarity is:

```text
C_region = 1 - d_region
```

Interface similarity answers, “Do both sides of the boundary look alike?”
Region similarity answers, “Do the two complete regions have similar average
appearance?”

Both are used because either measurement alone can be misleading.

### 4.4 Semantic similarity and confidence

For every candidate source, the saved semantic probability raster is sampled.
For class `k`:

```text
P(k) = mean probability of class k over all source pixels
```

The system also calculates:

```text
margin = P(top class) - P(second class)
purity = fraction of source pixels where the top class is the unique winner
```

- High probability indicates model support for the class.
- High margin indicates low ambiguity between the first and second choices.
- High purity indicates spatial agreement across the whole source region.

### 4.5 Strict similarity gate

The strict cleanup accepts a micro-unknown only when all required evidence
passes the configured thresholds:

| Evidence | Current threshold |
|---|---:|
| Maximum source area | 40 pixels |
| Mean target probability | `>= 0.60` |
| Probability margin | `>= 0.20` |
| Dominant-pixel purity | `>= 0.90` |
| Source boundary touching target | `>= 0.50` |
| Interface color similarity | `>= 0.70` |
| Region mean-color similarity | `>= 0.55` |
| Tile boundary | Must not touch |

The target must be directly adjacent and match the dominant predicted class.
The strict target classes are bareland, rangeland, developed space, tree, and
water. Unknown cannot be a target.

If several adjacent instances have the dominant class, target selection uses:

1. greatest number of shared pixel sides;
2. greatest interface color similarity; then
3. smallest deterministic instance ID.

Region similarity is a gate but is not a tie-breaker. The chosen target is
tested against all gates; the strict evaluator does not silently fall back to
the second target if the first one fails.

The stored strict proposal score is:

```text
score_strict = min(1,
    0.50 * target_probability
  + 0.25 * probability_margin
  + 0.25 * boundary_fraction)
```

This score is an audit/ranking value after the hard gates; it is not an
accuracy probability.

### 4.6 Contextual similarity energy

The remaining eligible micro-unknowns are evaluated against every directly
adjacent known target. For each target, the system calculates:

```text
E = 0.45 * (1 - P_target)
  + 0.25 * (1 - B)
  + 0.20 * (1 - C_boundary)
  + 0.10 * (1 - C_region)
```

where:

- `P_target` is the mean target-class probability on the unknown pixels;
- `B` is the source-perimeter contact fraction;
- `C_boundary` is exact interface RGB similarity; and
- `C_region` is whole-region mean RGB similarity.

The targets are ranked by **lowest energy**, with the smaller deterministic
target ID used only to keep the ranking reproducible. Ranking alone is not an
automatic approval: the absolute-energy and separation gates below must pass.

The weights sum to 1. Semantic evidence has the largest weight, followed by
physical contact, boundary appearance, and full-region appearance.

The stored contextual proposal score is:

```text
score_contextual = 1 - E
```

Again, this is a bounded ranking score, not a calibrated probability.

For a contextual (non-consensus) selection, the winning energy must be at most
`0.40`. When more than one target is eligible, its energy must also beat the
runner-up by at least `0.02`. A strong semantic-consensus selection instead
passes the separate probability, margin, and purity gates. These checks prevent
a merely relative winner from being applied when all targets are weak or when
the best two explanations are effectively tied. The values remain engineering
thresholds that need validation on geographically separate labeled data.

### 4.7 Worked example from the tile

For one 1-pixel unknown region, a road target had approximately:

```text
P_target    = 0.3875
B           = 0.5000
C_boundary  = 0.9821
C_region    = 0.8633
```

Therefore:

```text
E_road = 0.45(1 - 0.3875)
       + 0.25(1 - 0.5000)
       + 0.20(1 - 0.9821)
       + 0.10(1 - 0.8633)
       = approximately 0.4179
```

A neighboring tree target had slightly stronger semantic probability but a
weaker interface score, producing energy approximately `0.4181`. Under the
current policy this case **abstains**: the best energy exceeds `0.40`, and its
approximately `0.0002` lead is smaller than the required `0.02` margin.

This example demonstrates both the benefit and limitation of multi-signal
similarity: a decision does not depend on class probability alone, while weak
or nearly tied evidence is preserved as unknown instead of being forced. The
complete candidate table and evidence digest are retained for audit.

### 4.8 Class-specific safety gates

Similarity alone is not sufficient. Before a contextual target becomes
eligible, class policy is also applied:

- Agriculture requires exactly one adjacent agriculture target and at least
  `0.75` source-boundary contact.
- Building requires one unique building target and target probability at least
  `0.35`.
- Road requires target probability at least `0.30`, or road must be one of the
  top two predicted classes.
- Unknown is never a target.
- Tile-border micro-regions abstain.
- A likely standalone tree is preserved when area is at least 20 pixels,
  probability `>= 0.55`, margin `>= 0.15`, and purity `>= 0.80`.
- A likely standalone building is preserved when area is at least 20 pixels,
  probability `>= 0.65`, margin `>= 0.20`, and purity `>= 0.90`.

### 4.9 Similarity for generic merge proposals

Same-class merge proposals use a separate uncalibrated priority score:

```text
score_merge = 0.35 * semantic_match
            + 0.30 * C_boundary
            + 0.25 * C_region
            + 0.10 * shape_contact
```

For these candidates:

```text
semantic_match = 1
shape_contact = min(1, minimum_shared_perimeter_fraction / 0.25)
```

The configured proposal gates require:

- score `>= 0.98`;
- at least 8 shared pixel sides; and
- minimum shared-perimeter fraction `>= 0.15`.

In `human_review`, this score prioritizes review. In `automatic`, score alone
cannot approve a merge: the stricter default requires score `>= 0.99`, 12
shared sides, shared-perimeter fraction `>= 0.25`, interface and region
similarity each `>= 0.95`, an interior pair, no hard veto, and a class outside
agriculture, buildings, road, tree, and unknown. Because `semantic_match` is
fixed at 1 for same-class pairs, it is not independent evidence that two
segments are the same physical object. The `shape_contact` term is only a
shared-boundary proxy; it is not a full geometric shape descriptor.

### 4.10 Similarity for generic absorb proposals

The generic absorb path handles small known-class candidates and deliberately
skips unknown sources, because unknowns use the two specialized rules above.
It chooses a target by greatest source contact fraction, then greatest
interface similarity, then lowest target ID. Its review-priority score is:

```text
score_absorb = 0.50 * B
             + 0.25 * C_boundary
             + 0.25 * C_region
```

Current gates require source area `<= 40`, score `>= 0.98`, source contact
fraction `>= 0.80`, and a best-versus-runner-up **boundary-fraction** margin of
at least `0.25`. Cross-class cases additionally require target probability
`>= 0.90` and target-versus-best-other probability margin `>= 0.25`.
Cross-class road, agriculture, tree, and building sources are vetoed, and
unknown cannot be the target. Generic cross-class proposals remain
human-review operations. Automatic mode can select a same-class generic absorb
only with the stricter defaults: source area `<= 20`, score `>= 0.99`, at
least 8 shared sides, both RGB similarities `>= 0.95`, an interior source, and
a non-protected source/target class; all base contact and margin gates must
also pass.

## 5. Applying an accepted absorb operation

When a micro-unknown is absorbed:

1. The source and target are revalidated against the current raster.
2. Area, border contact, semantic probability, and exact raster adjacency are
   recomputed where required.
3. Strict RGB evidence is threshold-checked from the bound proposal. Contextual
   candidate evidence is protected by its SHA-256 digest, and the selected
   target is checked against the stored eligible-energy table. In a normal
   pipeline apply, proposals are regenerated from the immutable source first.
4. The source pixels receive the target semantic class.
5. Their confidence becomes the stored target-class probability.
6. Source pixels join the target instance.
7. The target's identity and semantic metadata remain primary.
8. Source and target lineage are both retained in the output record.

Multiple small sources may join the same target, but a source cannot be used by
conflicting operations.

The same proposal core supports two operating modes. `human_review` uses the
file workflow below. `automatic` generates versioned decisions, deterministically
resolves overlaps, rechecks raw evidence during replay, and records every weak,
protected, border, or conflicting candidate as an abstention. For many tiles,
`python -m src refine batch --mode automatic --workers 4` discovers completed
Stage 2 tile directories, resumes compatible validated outputs, isolates tile
failures, and writes `_batch/refinement_batch_report.json`.

### 5.1 How a human review is applied

The current review system is a file-based workflow rather than a graphical
application.

#### A. Generate the review bundle

```powershell
python -m src refine propose --tile-id IKONOS_2005-002__tile_00003
```

This writes:

```text
13_refinement/
  proposals.json
  decisions.template.json
  proposal_report.json
  proposal_manifest.json
  review/
    index.json
    segments.json
    segments.npz
    segmentation.npz
    candidates/*.png
```

`review/index.json` is the review catalogue. For every pending merge or absorb,
it records:

- proposal and entity IDs;
- class names and areas;
- full similarity evidence and vetoes;
- the expected pixel change;
- crop bounds; and
- the corresponding crop path and SHA-256 checksum.

Each crop contains three panels:

1. original satellite image;
2. normalized segmentation before the operation; and
3. proposed result.

Cyan marks the exact shared interface. Magenta marks pixels whose displayed
instance assignment would change.

#### B. Make one of three decisions

Open `decisions.template.json`. Every pending candidate starts as:

```json
{
  "proposal_id": "the exact proposal ID",
  "decision": "uncertain",
  "decision_source": "human_review",
  "reviewer": null,
  "reason": null
}
```

The reviewer changes only the intended review fields:

- `approved`: apply the proposed operation exactly as shown;
- `rejected`: record that the proposal is incorrect and do not apply it; or
- `uncertain`: defer the operation and leave the segmentation unchanged.

Example approval:

```json
{
  "proposal_id": "IKONOS_2005-002__tile_00003::proposal::merge::...",
  "decision": "approved",
  "decision_source": "human_review",
  "reviewer": "reviewer_name",
  "reason": "The interface is artificial and both regions form one continuous canopy."
}
```

Do not change the document's `tile_id` or `proposal_checksum`. They bind the
decisions to the exact tile, source pixels, configuration, and regenerated
proposal set.

#### C. Apply and validate reviewed decisions

```powershell
python -m src refine apply `
  --tile-id IKONOS_2005-002__tile_00003 `
  --decisions artifacts/stage2/IKONOS_2005-002__tile_00003/13_refinement/decisions.template.json

python -m src refine validate --tile-id IKONOS_2005-002__tile_00003
python -m src stage3 run --tile-id IKONOS_2005-002__tile_00003
```

During apply, proposals are regenerated from immutable Stage 2 data. The
pipeline rejects:

- a decision file for another tile;
- a stale or changed proposal checksum;
- unknown or duplicate proposal IDs;
- decisions for deterministic connectivity splits;
- hard class-policy vetoes;
- conflicting operations touching the same source; and
- any result with gaps, overlaps, an undeclared disconnected ID, or metadata
  mismatch.

Only `approved` semantic operations are replayed. `rejected` and `uncertain`
remain unapplied but preserve the review decision. Internal connectivity
analysis, micro-component identity consolidation, and bounded micro-unknown
cleanup still run.

#### D. Current tile example

The current tile has one pending human-review proposal: merging two small tree
segments. Its review crop is:

```text
review/candidates/merge_775204b5e692e58e2919.png
```

The evidence is strong as a review priority—12 shared sides, interface
similarity about `0.949`, region similarity about `0.988`, and proposal score
about `0.982`. However, both regions reach the tile boundary, so the image does
not show their full external context. The conservative current decision is
`uncertain`, not automatic approval.

> بالعربي: المراجع يرى الصورة الأصلية، الحدود الحالية، والنتيجة المقترحة، ثم
> يختار approved أو rejected أو uncertain. حتى لو اختار approved، النظام يعيد
> فحص الـ IDs والـ checksum والقواعد قبل تطبيق التعديل.

## 6. Graph construction after refinement

Stage 3 rebuilds the graph from the refined raster, not by patching the old
graph.

- One refined segment becomes one authoritative node.
- Every edge represents at least one real shared pixel side.
- Diagonal-only contact does not create an edge.
- Edge metadata records shared-side counts and exact contact coordinates.
- Visual edges are routed through the real contact location instead of drawing
  only a misleading centroid-to-centroid chord.

Two graph products are saved:

### Full audit graph

`graph.json` contains every refined segment and every valid adjacency. It is
the lossless source for auditing and changing cleanup policy later.

### Low-noise analysis graph

`graph_analysis.json` is the recommended graph for later combination in Stage
4. It excludes from the selected view:

- remaining unknown nodes through 40 pixels;
- edges supported by fewer than 2 shared pixel sides.

Topology-only secondary pieces through 40 pixels are already consolidated into
their parent identities before Stage 3, so they are not nodes that need visual
filtering. The remaining filtering is lineage-aware and preserves small native
road/building objects that a global area filter would incorrectly hide. Every
excluded node and edge remains in `graph.json` with an explicit exclusion
reason.

> بالعربي: لا نحذف المعلومات الأصلية. لدينا graph كامل للتدقيق وgraph نظيف
> للاستخدام العملي في المرحلة التالية.

## 7. Result on `IKONOS_2005-002__tile_00003`

### Segmentation result

| Measurement | Stage 2 | Refined |
|---|---:|---:|
| Total segments | 893 | 743 |
| Unknown segments | 601 | 424 |
| Unknown pixels | 10,074 | 8,985 |
| Unknown single-pixel segments | 228 | 165 |

Applied operations:

- 227 deterministic connectivity splits used for internal replay;
- 311 automatic micro-unknown absorbs;
- 538 total recorded operations; and
- 1,089 semantically relabeled pixels.

The 311 absorbs break down as follows:

- 43 strict-rule absorbs covering 125 pixels;
- 268 contextual-rule absorbs covering 964 pixels;
- 82 operations labelled `semantic_consensus`; and
- 229 operations labelled `contextual`.

Of the 960 normalized micro-unknown candidates evaluated by the contextual
pass, 311 were absorbed, 3 likely standalone objects were preserved, and 646
abstained. One same-class merge proposal remains pending and was not applied.

The important node accounting is:

```text
893 Stage 2 objects
   -> 1,433 internal four-connected components
   -> 1,122 internal components after 311 unknown absorbs
   -> 743 published objects after consolidating 379 micro-components
```

Six large secondary components met the promotion floor and fit within the hard
object-count budget. The final 743-object result is 150 objects smaller than
Stage 2, so connectivity analysis no longer causes the published node count to
grow. The result contains 579
single-part polygons and 164 explicitly recorded multipart objects.

Validation result:

- 262,144 valid pixels assigned;
- zero gaps;
- zero overlaps;
- zero undeclared disconnected final instance IDs; and
- all masks and semantic records agree.

### Graph result

| Product | Nodes | Edges | Intended use |
|---|---:|---:|---|
| Full audit graph | 743 | 1,706 | Stage 4 input and complete provenance |
| Low-noise analysis graph | 350 | 774 | Normal downstream analysis |

The analysis view suppresses 393 remaining micro-unknown nodes. It keeps 55
larger unknown regions because assigning them automatically would create false
semantic certainty.

## 8. Strengths of the method

- Deterministic and reproducible.
- Uses actual raster contact, not centroid distance.
- Combines semantic, geometric, and appearance evidence.
- Uses class-specific safety rules.
- Never leaves valid-pixel gaps.
- Preserves original metadata and complete lineage.
- Rechecks evidence during replay instead of trusting an editable proposal.
- Keeps an authoritative graph alongside the clean analysis graph.

## 9. Limitations and honest interpretation

1. RGB similarity is sensitive to shadows, illumination, seasonal state, and
   sensor differences.
2. Mean RGB can hide texture or multimodal appearance inside a region.
3. Semantic probabilities are model outputs; they are not claimed to be
   calibrated probabilities of correctness.
4. The contextual energy weights are engineering choices, not learned from
   labeled refinement decisions.
5. The contextual path now has a `0.40` maximum-energy cutoff and a `0.02`
   runner-up margin, but those engineering thresholds are not calibrated
   probabilities and still require validation.
6. The 40-pixel ceiling was selected as a conservative micro-noise boundary for
   this workflow and should be validated on geographically separate tiles.
7. A single image cannot reliably infer overpasses, every road junction, or all
   agricultural parcel boundaries.
8. Automatic connected semantic splitting is deliberately not generated in
   this version. Exact reviewed split replay is supported.
9. Large unknown regions remain visible because forced relabeling would reduce
   apparent noise while potentially increasing semantic error.

## 10. Suggested evaluation with the teacher

The most useful next experiment is to prepare a geographically separate,
manually reviewed validation set and evaluate operation-level precision.

Suggested questions:

1. What operation-level precision do the contextual `0.40` energy and `0.02`
   separation gates achieve on unseen geographies?
2. Should the four similarity weights be learned from reviewed decisions?
3. Should multispectral features replace or supplement RGB similarity?
4. What minimum physical object area should be used for each class?
5. How should divided roads, roundabouts, ramps, and overpasses be represented?
6. What image evidence defines a true agricultural field boundary?
7. Should tree segments represent one crown or one continuous canopy?
8. What target precision is required before enabling automatic building or
   agriculture splits?

Recommended metrics include:

- precision of accepted merge/absorb/split operations;
- false merges and false splits per 1,000 candidates;
- Boundary IoU;
- class-wise panoptic quality;
- split/merge Variation of Information; and
- graph node/edge correctness after refinement.

## 11. Reproduction command and outputs

Run refinement and Stage 3 together:

```powershell
python -m src stage3 run `
  --tile-id IKONOS_2005-002__tile_00003 `
  --refine-first `
  --refinement-mode automatic
```

Main outputs:

```text
artifacts/stage2/IKONOS_2005-002__tile_00003/13_refinement/
artifacts/stage3/IKONOS_2005-002__tile_00003/01_graph/graph.json
artifacts/stage3/IKONOS_2005-002__tile_00003/01_graph/graph_analysis.json
```

Stage 4 uses `graph.json` plus the exact refined raster to reconcile duplicate
geometry without losing provenance. Use `graph_analysis.json` for normal
low-noise downstream analysis of an individual tile. The combined Stage 4
graph records all source members so an audit can still inspect excluded or
uncertain objects.

## 12. Short presentation summary

> We refine the segmentation before constructing the graph. First, every mask
> is normalized into side-connected components. Next, unknown components no
> larger than 40 pixels are compared only with their real adjacent neighbors.
> The comparison combines semantic probability, shared-boundary fraction,
> exact boundary color similarity, and whole-region color similarity. Safe
> bounded cases are absorbed, while ambiguous, border, and protected-class
> cases are preserved or deferred. For scale, a separate selective automatic
> policy may also accept only same-class interior homogeneous merge/absorb
> proposals whose score and every raw gate pass; the human path retains visual
> review for sensitive edits. Every operation is replay-validated and retains
> full lineage. Finally, Stage 3 exports both a complete audit graph and a
> selected low-noise graph for downstream use.
