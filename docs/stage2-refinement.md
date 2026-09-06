# Stage 2R segmentation refinement

## 1. Purpose and position in the pipeline

Stage 2R refines completed Stage 2 tiles before Stage 3 builds their graphs. It
has two decision paths over the same proposal, replay, and validation core:

```text
Stage 2: 12_final (immutable segmentation)
                  |
                  v
          normalize and build proposals
                  / \
                 /   \
      human_review   automatic
      inspect/edit   select or abstain
                 \   /
                  \ /
             replay -> validate
                  |
                  v
          13_refinement (authoritative refined segmentation)
                  |
                  v
Stage 3: tile-local adjacency graph
```

Corrections are applied to the dense instance raster first. Masks, segment
records, previews, and the Stage 3 graph are derived again from that raster.
Stage 2R never edits graph nodes or edges independently of their pixels and
never overwrites `12_final`.

The two modes serve different scales:

| Mode | Intended use | Decision behavior |
|---|---|---|
| `human_review` | A manageable number of tiles or sensitive object/network edits | Writes a visual proposal bundle; only explicit `approved` decisions replay pending proposals. |
| `automatic` | Large unattended tile collections | Applies the versioned selective policy to pending generic proposals; candidates that fail any gate are recorded as `uncertain` and remain unchanged. |

The shared refinement policy is intentionally conservative:

- deterministic four-connected component normalization is applied
  automatically;
- the strict `micro-unknown-v1` rule and bounded `contextual-micro-v1` pass
  automatically contract eligible `unknown` regions of at most 40 pixels into
  directly adjacent known regions in either workflow mode;
- human-review mode leaves other eligible merge/absorb proposals for explicit
  approval;
- automatic mode can additionally approve narrowly gated, same-class interior
  merges and absorbs for non-protected classes;
- checksum-bound semantic split partitions can be reviewed and replayed through
  the Python core API, but the CLI does not yet guess or generate their
  boundaries and automatic mode never approves semantic splits; and
- an uncertain case is left unchanged rather than guessed.

This is a tile-local stage. Cross-tile identity and graph combination remain a
later-stage responsibility.

## 2. Segment ontology

The refinement rules use the following definition of one segment:

| Semantic class | One segment represents | Boundary policy |
|---|---|---|
| `buildings` | One physical building | Split distinct touching buildings; merge only fragments proven to belong to the same building. |
| `tree` | One continuous canopy | Preserve canopy area; split only across a genuine non-tree gap. |
| `road` | One junction-to-junction reach | Split at stable junctions; do not merge across a junction. |
| `agriculture land` | One agricultural field | Preserve field boundaries even when both sides have the same semantic class. |
| `water` | One connected homogeneous water region | Merge only across a weak false internal boundary. |
| `rangeland` | One connected homogeneous region | Require appearance and boundary agreement before merging. |
| `bareland` | One connected homogeneous region | Require appearance and boundary agreement before merging. |
| `developed space` | One connected homogeneous region | Provisional policy until a more specific object definition is supplied. |
| `unknown` | An unresolved region | Abstain unless a bounded automatic cleanup pass or explicit review establishes a safe label and adjacent destination. |

The ontology is stored under `refinement.ontology` in `config.yaml`. Changing
an ontology policy changes the meaning of downstream graph nodes and therefore
requires rerunning proposal, review, apply, and Stage 3.

## 3. Authoritative input and safety gate

The input directory is derived from `stage2.output.root_directory`:

```text
artifacts/stage2/<tile_id>/12_final/
```

Stage 2R requires a complete, fresh manifest with schema:

```text
satellitegraph.stage2.full-segmentation.v1
```

It consumes the lossless Stage 2 masks, dense maps, records, valid-data mask,
tile metadata, reports, and audit images. It rejects a missing, incomplete,
stale, wrong-version, or wrong-tile result. Input checksums recorded during
proposal must still match when decisions are applied; if Stage 2 changes, the
proposals and review decisions must be regenerated.

When `11_semantic/openearthmap.npz` contains a class-first probability tensor,
Stage 2R checksum-binds it and uses it as mandatory evidence for cross-class
absorption. Its absence causes the embedded micro-unknown rules to abstain and
prevents reviewed cross-class absorption from being applied safely. It does
not by itself disable the generic same-class automatic selector, which relies
on exact raster contact, RGB evidence, class policy, and proposal metadata.

RGB is loaded in this order: `02_preprocessing/sam_input.npy`, the original
`tile_metadata.image_path`, then `12_final/final_overlay.png`. The first two are
authoritative image evidence. The overlay is only a rendered fallback for a
human review bundle; automatic mode deliberately excludes it from proposal
evidence because segmentation colors would make an artificial boundary look
like source imagery. If no authoritative RGB is available, automatic mode
records that fact in `automatic_selection.source_evidence`, has no pair-color
evidence for generic operations, and lets cleanup rules that require that
evidence abstain. Connectivity normalization can still complete and publish a
valid conservative result.

The following invariants are non-negotiable:

- instance ID `0` is reserved for nodata;
- every valid pixel has exactly one positive instance ID;
- no operation creates an overlap or a valid-data gap;
- invalid pixels remain instance `0` and semantic nodata;
- masks and records agree exactly with the dense instance map; and
- final IDs are contiguous and deterministic.

Consequently, valid pixels are never simply deleted. A reviewed deletion must
be represented as absorption into a justified neighbor or as an explicit
relabeling to `unknown`. Pixels can become nodata only when the authoritative
valid-data mask already marks them invalid.

## 4. Two refinement workflows

`refinement.mode` in `config.yaml` defaults to `human_review`. The
`--mode`/`--refinement-mode` CLI options override it for one refinement or
combined Stage 3 run. Both modes regenerate proposals from immutable Stage 2
input; neither continues mutating an older refined raster.

### 4.1 Human review: generate proposals

Run:

```powershell
python -m src refine propose --tile-id scene__tile_00000
```

With explicit paths:

```powershell
python -m src refine propose `
  --tile-id scene__tile_00000 `
  --project-root . `
  --config config.yaml `
  --no-progress
```

Proposal mode reads and analyzes Stage 2 but does not modify it and does not
publish a completed refined segmentation. It emits deterministic proposal and
evidence artifacts for review. Re-running proposal against identical input and
configuration produces the same proposal identities and ordering.

### 4.2 Human review: inspect and decide

Every review-gated non-connectivity proposal has one of these decisions:

- approve the exact proposed operation;
- reject it;
- leave it uncertain.

Open `review/index.json` and inspect the linked crop for each candidate. Each
crop contains the original context, normalized segmentation before the
operation, and the proposed display result. Normalized instance IDs and class
names are printed below the panels, cyan marks the exact shared interface, and
magenta marks pixels whose displayed instance assignment changes. The index
retains the complete proposal, evidence, entity IDs, areas, Stage 2 lineage,
crop bounds, and crop checksum. Area by itself is never an approval criterion.
Merge and absorb scores in v1 are deterministic review-priority scores, not
calibrated correctness probabilities.

`decisions.template.json` already contains every pending proposal with the safe
default `uncertain`. Retain its `tile_id`, `proposal_checksum`, proposal IDs,
and `decision_count`, then edit only reviewed rows, for example:

```json
{
  "proposal_id": "scene__tile_00000::proposal::merge::<digest>",
  "decision": "approved",
  "decision_source": "human_review",
  "reviewer": "reviewer-id",
  "reason": "One building split by a weak internal model boundary"
}
```

Allowed reviewed decisions are `approved`, `rejected`, and `uncertain`.
Proposal IDs bind decisions to exact sources, targets, and parameters; do not
edit the proposal itself to express a review decision. If rows are added or
removed, update `decision_count` to match. Unknown proposal IDs are rejected.
Deterministic `split_components` proposals and eligible `micro-unknown-v1` or
`contextual-micro-v1` proposals are non-reviewable and therefore absent from
the template; their configured rules are replayed directly. The untouched
all-`uncertain` template authorizes no *review-gated* semantic operation, but it
does not disable either automatic micro-unknown rule.

### 4.3 Human review: apply decisions

Apply only deterministic connectivity normalization and both eligible embedded
micro-unknown passes, without approving any pending review proposal:

```powershell
python -m src refine apply `
  --tile-id scene__tile_00000 `
  --mode human_review
```

Apply an exported review decision document:

```powershell
python -m src refine apply `
  --tile-id scene__tile_00000 `
  --mode human_review `
  --decisions artifacts/stage2/scene__tile_00000/13_refinement/decisions.template.json
```

An omitted, empty, or all-`uncertain` decision document never authorizes a
review-gated semantic correction. It still applies enabled deterministic
operations, including `micro-unknown-v1` and `contextual-micro-v1` when their
configured gates pass.
Automatic and reviewed operations are replayed from the immutable Stage 2 input
in a deterministic order; Stage 2R does not incrementally mutate an earlier
refined output.

### 4.4 Automatic mode for one tile

Run the complete unattended path explicitly:

```powershell
python -m src refine auto --tile-id scene__tile_00000
```

This is equivalent to:

```powershell
python -m src refine apply `
  --tile-id scene__tile_00000 `
  --mode automatic
```

To make the same choice the project default, set:

```yaml
refinement:
  mode: automatic
```

Automatic mode regenerates the same proposals, produces its own
`automatic_selective_v1` decisions, replays selected and embedded operations,
validates the output, and publishes a completed `13_refinement`. It rejects a
supplied human decisions document rather than mixing decision authorities in
one run.

An unattended run is complete even when it abstains on some or all pending
proposals. `uncertain` means "insufficient evidence for an automatic edit," not
"rejected as false." The normalized source region remains in the authoritative
output and its reason is retained for audit. The automatic policy never
approves semantic splits, and the standard CLI does not generate their exact
partitions.

### 4.5 Generic automatic selection policy

The selector does not treat the weighted proposal score as a probability. A
pending proposal must first have passed proposal-generation class policy, have
no hard veto, and use exact four-neighbor shared-edge adjacency. It must then
pass the score **and every raw-evidence gate** below.

For a generic merge, the default automatic gates require:

- exactly two directly adjacent segments of the same class;
- a class outside `agriculture land`, `buildings`, `road`, `tree`, and
  `unknown`;
- neither segment touching the tile boundary;
- score at least `0.99`, at least 12 shared pixel sides, and minimum shared
  perimeter fraction at least `0.25`; and
- weak interface-color similarity and whole-region mean-color similarity both
  at least `0.95`.

For a generic absorb, the default automatic gates require:

- a directly adjacent source and target with the same non-protected class and
  an explicit declaration that no cross-class relabel will occur;
- an interior source of at most 20 pixels;
- score at least `0.99`, at least 8 shared pixel sides, source contact fraction
  at least `0.80`, and best-versus-runner-up contact margin at least `0.25`;
  and
- interface and region color similarities both at least `0.95`.

These gates are controlled by `refinement.automatic`; the report stores an
exact threshold snapshot. Missing, malformed, or non-finite evidence fails its
gate. The protected-class rule here applies to **generic** automatic
merge/absorb proposals. The separate contextual micro-unknown rule may target
a road, field, or building only through its specialized probability,
uniqueness, contact, border, and object-preservation checks described in
Section 6.2.

Eligible proposals are resolved in a deterministic order: absorbs before
merges, then decreasing shared-boundary fraction, interface similarity, region
similarity, score, shared-side count, and finally proposal ID. A source can
participate in only one selected operation. Multiple absorbs may reuse the
same receiver, but an absorb receiver conflicts with a merge touching that
entity. A conflict loser becomes `uncertain` with the selected proposal ID
recorded. Automatic proposal generation also bounds the pending semantic
candidate set using `maximum_selected_operations_per_tile` (score then stable
proposal ID determine the retained set), and the selector independently
enforces the same per-tile approval ceiling. The corresponding candidate cap
and selection limit are retained in the proposal and automatic reports.

### 4.6 Batch discovery, workers, resume, and failures

For all completed Stage 2 tiles under the configured artifact root:

```powershell
python -m src refine batch --mode automatic --workers 4
```

Discovery includes each immediate tile directory containing
`12_final/manifest.json`; tile IDs are deduplicated and sorted. To run an
explicit subset, repeat the option:

```powershell
python -m src refine batch `
  --mode automatic `
  --workers 4 `
  --tile-id scene__tile_00000 `
  --tile-id scene__tile_00001
```

`--workers` is a positive bound on independently processed tiles. Completion
order may vary, but report rows remain sorted by tile ID and each tile uses the
same deterministic policy.

Automatic batch mode resumes by default. It reports a tile as `skipped` only
after loading a completed, non-stale automatic result whose workflow mode and
configuration hash match and whose artifact validation still passes. A
missing, stale, incompatible, or invalid result is recomputed. `--force`
disables this compatible-result skip. Human-review batch mode regenerates
proposal bundles; it does not apply pending decisions and does not use the
completed-result resume shortcut.

Without `--fail-fast`, a tile exception is isolated and the remaining tiles
continue. The atomic per-run report
`artifacts/stage2/_batch/refinement_batch_report.json` acts as the failure
ledger: it records run mode/configuration, worker and resume settings,
aggregate counts, and an ordered row for every tile. Rows have status
`complete`, `proposed`, `skipped`, or `failed`; a failed row includes the
exception type and message. The overall status is `partial` when any row
failed. `--fail-fast` instead propagates the first error immediately, so use
the default continuation behavior when a complete failure ledger is required.

To prepare many review bundles without applying decisions:

```powershell
python -m src refine batch --mode human_review --workers 4
```

### 4.7 Validate a completed result

Run:

```powershell
python -m src refine validate --tile-id scene__tile_00000
```

Validation reloads the published artifacts and verifies coverage, masks,
semantics, connectivity, lineage, checksums, and replay metadata. Do not run
Stage 3 from a refinement directory that has not passed this command.

## 5. Deterministic connectivity normalization

Stage 2 used eight-neighbor connectivity while Stage 3 defines adjacency by
shared pixel sides. An eight-connected ID can therefore contain pieces that
meet only diagonally and become a disconnected `MultiPolygon` in the graph.

Stage 2R decomposes every positive Stage 2 instance into four-connected
components. The ordering rule is fixed:

1. visit Stage 2 instance IDs in ascending order;
2. sort that parent's components by top row, then left column, then decreasing
   area; and
3. assign refined IDs `1..N` in that order.

For example:

```text
Stage 2 instance map       Refined instance map

1 1 2 2                   1 1 3 3
1 1 2 2                   1 1 3 3
3 3 1 1                   4 4 2 2
3 3 1 1                   4 4 2 2
```

The two diagonal pieces belonging to old instance `1` become refined IDs `1`
and `2`. Their shared `entity_group_id` and Stage 2 lineage preserve the fact
that they came from the same original canopy candidate; they are nevertheless
separate graph nodes because they share no pixel side.

Connectivity normalization changes instance identity only. It does not alter
the semantic, confidence, source, or valid-data value of any pixel.

## 6. Operation policies and replay

### 6.1 Split

A true semantic split partitions one connected parent mask. The executable
Python contract accepts a full-size `int32` child-label raster: zero outside the
parent and contiguous labels `1..K` inside it. The proposal identity binds the
partition checksum, dimensions, child count, and minimum child area. Replay
requires a separate explicit approval and revalidates that every child is
four-connected, the children are disjoint, and their exact union equals the
parent. Semantic, confidence, source, and valid-data rasters remain unchanged.

Use `make_semantic_split_candidate(...)` to create the bound proposal and
`apply_reviewed_operations(..., split_partitions={proposal_id: partition})` to
replay it. The standard CLI intentionally does not synthesize these partitions
from RGB in v1: a class-specific candidate generator or a reviewed annotation
must supply the boundary. This prevents an uncalibrated watershed or texture
edge from silently becoming a real object boundary.

Class-specific rules include:

- buildings may split at a supported physical-building boundary;
- tree canopy splits require a real non-tree gap, not crown texture;
- road splits occur at reviewed junction regions;
- agriculture splits require a sustained field boundary; and
- homogeneous classes split only when a strong boundary separates internally
  coherent regions.

### 6.2 Absorb instead of delete

Absorption retires a false segment identity and assigns all of its valid pixels
to one directly adjacent receiver. No valid pixel is erased or changed to
nodata. For a cross-class absorption, replay relabels only the source pixels to
the receiver class and replaces their confidence with that class's saved
OpenEarthMap probabilities; `source_map` and `valid_mask` remain unchanged.

Automatic cleanup is bounded to normalized, four-connected `unknown`
components of at most 40 pixels. Forty is a hard contextual safety ceiling:
configuration may lower it but cannot raise it. The contextual pass always
abstains at the tile border, the strict defaults require the same abstention,
and neither path targets another `unknown`.

The strict `micro-unknown-v1` path auto-applies only when all configured
conditions pass:

- source area is at most `maximum_segment_area_pixels` and, by default, does
  not touch the tile boundary;
- the dominant mean OpenEarthMap class is unique, its mean probability and
  runner-up margin pass their thresholds, and that class wins on the required
  fraction of source pixels;
- a known region of exactly that dominant class shares pixel sides with the
  source;
- the selected target passes the source-perimeter contact fraction, weak color
  boundary, and region mean color-similarity thresholds; and
- the target class is present in `allowed_target_classes`.

Its automatic targets are `bareland`, `rangeland`, `developed space`, `tree`,
and `water`. The strict target list cannot include `road`, `agriculture land`,
or `buildings`.

The integrated `contextual-micro-v1` pass accounts for every bounded unknown.
It either absorbs into a directly adjacent known target, preserves strong
standalone tree/building evidence, or records an explicit abstention. Among
eligible targets it deterministically ranks this fixed energy, using the lower
target instance ID only to make equal-energy ordering reproducible:

```text
0.45 * (1 - target class probability)
+ 0.25 * (1 - source-boundary contact fraction)
+ 0.20 * (1 - weak RGB boundary score)
+ 0.10 * (1 - region RGB similarity)
```

Road, agricultural-field, and building targets receive additional protection:

- agriculture requires exactly one adjacent agriculture target and at least
  the configured source-boundary fraction;
- buildings require exactly one adjacent building target and the configured
  minimum building probability; and
- roads require either their configured minimum probability or membership in
  the source's two highest-probability classes.

Before choosing a contextual target, the pass preserves an unknown as a
standalone tree or building when it has no matching known neighbor and its
area, probability, probability margin, and pixel-purity values all pass the
corresponding configured evidence floors. Border-touching sources, missing
model/contact evidence, and cases with no eligible target abstain. A
non-consensus contextual winner must additionally have energy at most `0.40`
and, when a runner-up exists, lead it by at least `0.02`. Strong semantic
consensus uses its separate probability, probability-margin, and purity gates.

The complete default threshold blocks are:

```yaml
refinement:
  operations:
    unknown_cleanup:
      enabled: true
      auto_apply: true
      maximum_segment_area_pixels: 40
      allowed_target_classes:
        - bareland
        - rangeland
        - developed space
        - tree
        - water
      minimum_mean_target_probability: 0.60
      minimum_probability_margin: 0.20
      minimum_dominant_pixel_fraction: 0.90
      minimum_shared_boundary_fraction: 0.50
      minimum_weak_color_boundary_score: 0.70
      minimum_region_mean_color_similarity: 0.55
      require_not_tile_boundary: true
    contextual_cleanup:
      enabled: true
      auto_apply: true
      maximum_segment_area_pixels: 40
      semantic_consensus_minimum_probability: 0.55
      semantic_consensus_minimum_margin: 0.15
      semantic_consensus_minimum_purity: 0.80
      maximum_selected_energy: 0.40
      minimum_energy_margin: 0.02
      standalone_tree_minimum_area_pixels: 20
      standalone_tree_minimum_probability: 0.55
      standalone_tree_minimum_margin: 0.15
      standalone_tree_minimum_purity: 0.80
      standalone_building_minimum_area_pixels: 20
      standalone_building_minimum_probability: 0.65
      standalone_building_minimum_margin: 0.20
      standalone_building_minimum_purity: 0.90
      agriculture_minimum_boundary_fraction: 0.75
      building_minimum_probability: 0.35
      road_minimum_probability: 0.30
```

The exact threshold snapshot is stored in each automatic proposal. Immediately
before replay, Stage 2R checks that the snapshot still matches configuration.
For contextual operations it also verifies the evidence digest, recomputes
area, border, direct raster adjacency, probabilities, object-preservation and
protected-class gates, and confirms that the selected target is still the
deterministic minimum-energy candidate. Stale or altered proposals fail rather
than bypassing the gates.

Generic cross-class absorbs remain human-review-only. A generic same-class
absorb may be human-approved or selected by automatic mode when every stricter
gate in Section 4.5 passes. Both require a directly adjacent receiver plus
appearance and shared-boundary evidence. A reviewed cross-class absorb
additionally requires the saved probability channel for its known target
class. Selecting the largest neighbor, using area alone, or passing only the
combined score is never sufficient.

### 6.3 Merge

A merge can involve only directly adjacent segments. Compatible semantic
policy, weak interface evidence, sufficient shared-boundary support, appearance
agreement, and plausible union geometry are considered independently. Road and
agriculture merges, unknown merges, and other ontology violations are hard
vetoes: they are removed from the review queue and checked again during replay,
so a decision file cannot override them.

Human-approved merges are applied one at a time. Automatic mode is narrower:
it protects object/network classes, requires the stricter raw gates in Section
4.5, and resolves overlapping choices before replay. Proposals touching a
changed region must be recomputed so transitive merges never use stale
evidence.

### 6.4 Replay validation

Approval is an instruction to recheck an operation, not permission to bypass
the raster contract. Every apply run regenerates proposals from immutable
Stage 2, validates proposal and decision identities, rejects duplicate or
unknown IDs, and checks that no incompatible operations share an entity.
Class hard vetoes, direct adjacency, exact disjoint union, four-connectivity,
valid-pixel coverage, and output metadata are checked again.

For an `automatic_selective_v1` generic merge or absorb, replay recomputes pair
evidence from the normalized instance raster and source RGB, reconstructs the
candidate score, checks source/target classes and border state, and calls the
same versioned automatic evaluator again. The decision source must match the
configured policy version. For an embedded unknown absorb, replay instead
recomputes probabilities, area, adjacency, border and specialized object gates;
contextual operations also verify their evidence digest and minimum-energy
target, maximum energy, and runner-up separation. A stale threshold snapshot or
changed evidence fails the run rather than silently applying the edit.

After replay, result validation proves dense-map coverage, non-overlap,
four-connected positive IDs, mask/record agreement, semantic consistency,
contiguous IDs, and lineage integrity. The public manifest is marked complete
and non-stale only after the artifacts and their checksums have been written.

## 7. Segment lineage

Every refined segment has a new deterministic identity plus explicit links to
the immutable Stage 2 record. Core fields include:

| Field | Meaning |
|---|---|
| `instance_id` | Contiguous ID in the refined dense map. |
| `segment_id` | Deterministic refined segment identifier. |
| `entity_id` | Tile-qualified refined entity identifier. |
| `stage2_instance_id` | Direct Stage 2 parent for connectivity children. |
| `stage2_segment_id` | Direct Stage 2 segment identifier. |
| `component_index`, `component_count` | Position and count in four-connected normalization. |
| `source_stage2_instance_ids` | All original Stage 2 IDs contributing pixels. |
| `source_stage2_segment_ids` | Matching original segment identifiers. |
| `entity_group_id` | Common group for children derived from one original entity candidate. |
| `refinement_history` | Ordered operation IDs that produced the current region. |
| `stage2_metadata` | Deep copy of the original Stage 2 record for audit and downstream use. |

Operation records retain operation type, source and output IDs, evidence,
decision, decision source, vetoes, source mask checksums, configuration hash,
and exact before/after pixel counts. Retired IDs remain in the operation log;
they are never silently reused.

## 8. Output directory and artifacts

The completed result is written beside `12_final`:

```text
artifacts/stage2/<tile_id>/13_refinement/
```

Its public schema is:

```text
satellitegraph.stage2-refinement.v1
```

Proposal mode writes `proposals.json`, a checksum-bound
`decisions.template.json`, `proposal_report.json`, and
`proposal_manifest.json`. It also writes this self-contained review bundle:

```text
review/
  index.json
  segments.json
  segments.npz
  segmentation.npz
  candidates/
    <operation>_<proposal-digest>.png
```

The review segmentation, records, and masks are the normalized geometry
referenced by proposal entity IDs. `review/index.json` binds itself to the exact
`proposals.json` checksum and maps every pending merge/absorb proposal to its
entities and crop. `proposal_manifest.json` checksums all review files. The
public `manifest.json` has status `proposed` and is deliberately not consumable
by Stage 3.

A completed apply writes:

```text
13_refinement/
  segments.json
  segments.npz
  segmentation.npz
  refinement_operation_map.npz
  proposals.json
  decisions.json
  operations.json
  lineage.json
  tile_metadata.json
  class_registry.json
  refined_segmentation.png
  refined_overlay.png
  refinement_before_after.png
  report.json
  phase_manifest.json
  manifest.json
```

Use JSON artifacts for nested lineage and NPZ artifacts for exact numeric
arrays; preview PNGs are never an authoritative segmentation source. The
public manifest binds every completed artifact and every required Stage 2
input to a SHA-256 checksum.

`refinement_operation_map.npz` stores `0` for pixels whose current segment has
no refinement history and a positive operation index for pixels affected by
the most recent recorded operation. `report.json` and `operations.json` map
that index back to its stable operation ID. Keep this raster with future
tree-canopy area measurements so pipeline corrections are not mistaken for
real landscape change.

`report.json.comparison` records raw and refined segment/unknown counts, the
unknown-pixel delta, and the number of semantic pixels changed. Its
`unknown_cleanup` block records both rule versions, strict eligibility,
contextual evaluations, absorbs, preserved standalone objects, abstentions,
auto-applied operations, absorbed pixels, and both exact threshold blocks.
Every applied absorb also remains independently visible in `operations.json`
and lineage.

For either completed mode, `report.json.workflow_mode` identifies the decision
path. In automatic mode, `report.json.automatic_selection` stores the policy
version, selected IDs, approval/abstention counts by operation, exact threshold
snapshots, and a per-proposal evaluation containing observed raw evidence and
final reasons. Its `source_evidence` block also says whether RGB was available,
whether it was authoritative enough for automation, and whether the saved
semantic probability tensor was available. `decisions.json` preserves the
generated `approved` or `uncertain` records and their decision source;
`operations.json` contains only edits that survived replay. `manifest.json`
records the workflow mode, configuration hash, automatic policy version,
source checksums, and completed artifact checksums.

The separate batch report at
`artifacts/stage2/_batch/refinement_batch_report.json` is an aggregate run
ledger, not a segmentation artifact. It records every selected tile and lets
operators retry only failures or allow compatible automatic results to resume
as `skipped` on the next run.

The Python API is:

```python
from src.refinement import (
    apply_refinement,
    load_refinement_result,
    propose_refinement,
    run_refinement_batch,
    validate_refinement_result,
)

proposal = propose_refinement(
    "scene__tile_00000",
    project_root=".",
    config="config.yaml",
)

result = apply_refinement(
    "scene__tile_00000",
    project_root=".",
    config="config.yaml",
    workflow_mode="human_review",
    decisions=None,  # no pending human proposal is approved
)

validate_refinement_result(result)
reloaded = load_refinement_result(result.output_directory)

automatic_result = apply_refinement(
    "scene__tile_00000",
    project_root=".",
    config="config.yaml",
    workflow_mode="automatic",
)

batch = run_refinement_batch(
    None,  # discover every complete Stage 2 tile
    project_root=".",
    config="config.yaml",
    workflow_mode="automatic",
    workers=4,
    resume=True,
)
```

## 9. Stage 3 handoff

The default Stage 3 configuration uses:

```yaml
stage3:
  input:
    source: refinement
```

For one automatic tile, apply refinement and build Stage 3 in one command:

```powershell
python -m src stage3 run `
  --tile-id IKONOS_2005-002__tile_00003 `
  --refine-first `
  --refinement-mode automatic
```

The equivalent human-review sequence is:

```powershell
python -m src refine propose --tile-id IKONOS_2005-002__tile_00003
python -m src refine apply `
  --tile-id IKONOS_2005-002__tile_00003 `
  --mode human_review `
  --decisions artifacts/stage2/IKONOS_2005-002__tile_00003/13_refinement/decisions.template.json
python -m src refine validate --tile-id IKONOS_2005-002__tile_00003
python -m src stage3 run --tile-id IKONOS_2005-002__tile_00003
```

`refine propose` is useful for inspection but is not required before
`refine apply`; apply regenerates and checksum-binds the proposals itself. To
include manually reviewed operations in the one-command form, add
`--refinement-mode human_review --refinement-decisions <path>`; in the explicit
form, add the checksum-bound `--decisions <path>` to `refine apply`. Do not pass
a reviewed document with automatic mode. Both embedded micro-unknown passes
run either way, while generic automatic selection runs only in `automatic`.

Stage 3 must consume the completed `13_refinement` result explicitly. It must
not silently fall back to `12_final` when refinement is configured, because
that would display or export a graph that disagrees with reviewed pixels.
Every refined segment becomes one node and adjacency is recalculated from the
refined instance map. Node metadata distinguishes the three lineage layers:

- `refinement_metadata` is the exact refined record that produced the node;
- `stage2_metadata` is its primary immutable Stage 2 ancestor; and
- `source_stage2_metadata` contains every immutable Stage 2 ancestor, which is
  necessary after reviewed merge or absorption.

## 10. Implemented scope, research basis, and roadmap

This release establishes the reversible artifact contract, deterministic
connectivity correction, two refinement modes, narrowly gated automatic
micro-unknown cleanup, selective generic same-class automation, batch
orchestration, reviewed-operation replay, and strict validation. The generic
automatic thresholds are conservative engineering defaults; they have not yet
been learned or calibrated as correctness probabilities.

The papers below provide a research basis and specialist roadmap. A citation
does not mean its architecture or weights are integrated into this repository.

| Work | Relevance | Repository status |
|---|---|---|
| [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500) | Formalizes the reject/coverage trade-off: a system can abstain to reduce risk on the cases it accepts. | **Concept implemented, guarantee not implemented.** `automatic_selective_v1` abstains gate-by-gate, but it is a deterministic policy and does not provide the paper's calibrated risk guarantee. |
| [OpenEarthMap](https://openaccess.thecvf.com/content/WACV2023/html/Xia_OpenEarthMap_A_Benchmark_Dataset_for_Global_High-Resolution_Land_Cover_Mapping_WACV_2023_paper.html) | Supplies the eight-class high-resolution land-cover basis used for semantic evidence. | **Implemented upstream.** Stage 2 uses the OpenEarthMap FasterSeg model with configured horizontal-flip test-time augmentation and saves the class-probability tensor consumed by refinement. This is not a learned refinement model. |
| [Improved Road Connectivity by Joint Learning of Orientation and Segmentation](https://openaccess.thecvf.com/content_CVPR_2019/html/Batra_Improved_Road_Connectivity_by_Joint_Learning_of_Orientation_and_Segmentation_CVPR_2019_paper.html) | Adds orientation supervision and a learned connectivity-refinement model for fragmented road masks. | **Future specialist.** No orientation head or connectivity-refinement network is loaded; road junction edits remain protected/reviewed. |
| [DeepRoadMapper](https://openaccess.thecvf.com/content_iccv_2017/html/Mattyus_DeepRoadMapper_Extracting_Road_ICCV_2017_paper.html) | Converts road segmentation to topology and reasons about missing connections as shortest-path hypotheses. | **Future specialist.** Stage 3 currently graphs observed shared edges only and never hallucinates a missing road connection. |
| [PolyWorld](https://openaccess.thecvf.com/content/CVPR2022/html/Zorzi_PolyWorld_Polygonal_Building_Extraction_With_Graph_Neural_Networks_in_Satellite_CVPR_2022_paper.html) | Predicts and connects building vertices to produce precise polygons. | **Future specialist.** Building split/polygon proposals are not generated automatically; exact reviewed partitions can already enter the replay contract. |
| [Remote SAMsing: From Segment Anything to Segment Everything](https://arxiv.org/abs/2605.00256) | Proposes multi-pass SAM2 coverage and contextual cross-tile merging for very large remote-sensing scenes. | **Future research (preprint).** Current refinement is tile-local; multi-pass mask generation and cross-tile identity reconciliation are not implemented in Stage 2R. |

A full specialist combination should preserve one decision boundary rather than
letting each model mutate the raster independently:

```text
Stage 2 masks + OpenEarthMap probabilities
                 |
      current deterministic evidence
                 |                         future specialist evidence
                 |                 (road connectivity, building polygons,
                 |                  multi-pass/cross-tile SAM proposals)
                 +-------------------------+
                                           |
                              common proposal schema
                                  /             \
                         human_review       automatic selector
                                  \             /
                               replay + validation
```

The next implementation steps are to collect reviewed decisions as labeled
operation data, measure risk versus coverage on geographically held-out tiles,
calibrate class/sensor/resolution-specific automatic gates, and add specialist
adapters that emit checksum-bound proposals and raw evidence. Roads, fields,
buildings, canopy splits, and cross-tile joins should stay protected until each
specialist is validated at the required operation-level precision. A future
specialist should therefore plug into the existing proposal/abstain/replay
contract instead of bypassing it.

Recommended rollout:

1. run proposal-only mode on representative tiles;
2. review and label merge and absorb candidates, plus externally supplied
   checksum-bound split partitions where ground truth supports them;
3. measure precision separately by operation, class, sensor, and resolution;
4. calibrate thresholds on tiles not used for tuning; and
5. adjust automatic-cleanup thresholds only after measuring the resulting error
   rate on held-out tiles.

Automatic behavior currently includes connectivity normalization, bounded
replay-validated micro-unknown cleanup, and the stricter same-class generic
merge/absorb subset in Section 4.5. It does **not** include road-junction
decomposition, agricultural-field boundary generation, connected
building/canopy split generation, generic cross-class relabeling, or cross-tile
reconciliation. Larger, boundary-touching, protected, conflicting, or
otherwise ineligible regions remain visible in the authoritative output.
Leaving such a segment unchanged is an expected abstention, not a pipeline
failure.
