# SatelliteGraph: Complete Project Context and Thesis Discussion Pack

**Purpose:** upload this file to ChatGPT so it can help explain, discuss, plan, and write the thesis using the actual project implementation.

**Snapshot date:** 16 October 2026, Asia/Beirut.  
**Repository:** [Ahmad1Alzein/SatelliteGraph](https://github.com/Ahmad1Alzein/SatelliteGraph).  
**Audited code:** commit `db1bbfdc7e3cd7ffe6946e3d801cc7d162b4d86a` on `main`.  
**Language:** simple English, technical detail where needed, and Arabic explanations/glossary in Sections 32 and 48.

This is a context and methodology document, not a report of a new experiment. Preparing it did not run segmentation, a comparison pipeline, or GNN training. Code paths below are relative to the repository root. Defaults describe this snapshot; an individual run's saved configuration is authoritative for that run.

**Reading guide:** Sections 1–6 introduce the project; 7–17 explain image preparation, segmentation, graphs, and change evidence; 18–24 explain the retained GNN, its adapter, results, and scores; 25–34 cover the dashboard, implementation, limitations, thesis planning, and Arabic glossary. Sections 35–40 contain the expanded research audit and writing guidance: 96 paper/article entries, 51 grouped software/model/standards entries, 30 dataset/resource entries, and 21 unresolved or quarantined leads. Sections 41–48 explain the work phases, detailed old/new comparison, 35-issue problem ledger, retired experiments, saved-run identity, evaluation/diagnostic protocols, handoff gaps and Arabic discussion. Section 49 preserves additional partial OSM and historical point-reference tables, including weak results and protocol limitations. The resource audit uses the repository and 26 available local project session logs, with explicit limits on access and reading depth.

## Contents

- [1. Starter message to give ChatGPT](#1-starter-message-to-give-chatgpt)
- [2. The project in plain English](#2-the-project-in-plain-english)
- [3. Research scope and a suitable thesis description](#3-research-scope-and-a-suitable-thesis-description)
- [4. Four layers that must not be mixed](#4-four-layers-that-must-not-be-mixed)
- [5. The three selectable stages and all eight combinations](#5-the-three-selectable-stages-and-all-eight-combinations)
- [6. End-to-end data flow](#6-end-to-end-data-flow)
- [7. Inputs, image validity, and years](#7-inputs-image-validity-and-years)
- [8. Coordinates, resolution, and Stage 1 tiling](#8-coordinates-resolution-and-stage-1-tiling)
- [9. Registration and brightness/sharpness harmonization](#9-registration-and-brightnesssharpness-harmonization)
- [10. Semantic classes and the three different ID systems](#10-semantic-classes-and-the-three-different-id-systems)
- [11. Old segmentation: SAM 2.1 + OpenEarthMap FasterSeg](#11-old-segmentation-sam-21--openearthmap-fasterseg)
- [12. New segmentation: SegFormer + Delineate Anything v2 + region rules](#12-new-segmentation-segformer--delineate-anything-v2--region-rules)
- [13. Canonical segmentation artifacts and adapters](#13-canonical-segmentation-artifacts-and-adapters)
- [14. Spatial graphs: old adjacency and the improved graph](#14-spatial-graphs-old-adjacency-and-the-improved-graph)
- [15. Combining overlapping tiles into annual graphs](#15-combining-overlapping-tiles-into-annual-graphs)
- [16. Temporal graph and correspondence across dates](#16-temporal-graph-and-correspondence-across-dates)
- [17. Change evidence, candidate catalogue, and the classical method](#17-change-evidence-candidate-catalogue-and-the-classical-method)
- [18. The retained GNN: location, provenance, and exact task](#18-the-retained-gnn-location-provenance-and-exact-task)
- [19. Exact GNN input contract](#19-exact-gnn-input-contract)
- [20. Exact GNN architecture and message passing](#20-exact-gnn-architecture-and-message-passing)
- [21. Original labels, training, and seven-tile evaluation](#21-original-labels-training-and-seven-tile-evaluation)
- [22. Current new-input GNN adapter and why scores can be unavailable](#22-current-new-input-gnn-adapter-and-why-scores-can-be-unavailable)
- [23. Verified preserved results and what they support](#23-verified-preserved-results-and-what-they-support)
- [24. Scores, statuses, filtering, and pagination](#24-scores-statuses-filtering-and-pagination)
- [25. Dashboard pages and their purpose](#25-dashboard-pages-and-their-purpose)
- [26. Repository architecture and artifact layout](#26-repository-architecture-and-artifact-layout)
- [27. Configuration, execution, and reproducibility](#27-configuration-execution-and-reproducibility)
- [28. Practical failure cases and known limitations](#28-practical-failure-cases-and-known-limitations)
- [29. Suggested thesis structure and research questions](#29-suggested-thesis-structure-and-research-questions)
- [30. Information still needed from the thesis author](#30-information-still-needed-from-the-thesis-author)
- [31. Short discussion answers for thesis preparation](#31-short-discussion-answers-for-thesis-preparation)
- [32. Arabic explanation and glossary](#32-arabic-explanation-and-glossary)
- [33. Source map and literature starting points](#33-source-map-and-literature-starting-points)
- [34. Final handoff rules for ChatGPT](#34-final-handoff-rules-for-chatgpt)
- [35. Expanded resource audit: coverage, reading depth, and source status](#35-expanded-resource-audit-coverage-reading-depth-and-source-status)
- [36. Comprehensive recovered paper and article bibliography](#36-comprehensive-recovered-paper-and-article-bibliography)
- [37. Software, standards, model sources, datasets, and historical resources](#37-software-standards-model-sources-datasets-and-historical-resources)
- [38. Citation corrections, unresolved leads, and excluded retrieval noise](#38-citation-corrections-unresolved-leads-and-excluded-retrieval-noise)
- [39. Literature-to-thesis discussion map](#39-literature-to-thesis-discussion-map)
- [40. Instructions for a full master's-thesis writing session](#40-instructions-for-a-full-masters-thesis-writing-session)
- [41. Project work phases and the reason for each phase](#41-project-work-phases-and-the-reason-for-each-phase)
- [42. Old and new pipelines: detailed comparison and improvement claims](#42-old-and-new-pipelines-detailed-comparison-and-improvement-claims)
- [43. Consolidated problem ledger: symptoms, causes, actions and open limits](#43-consolidated-problem-ledger-symptoms-causes-actions-and-open-limits)
- [44. Retired GNN experiments, retained model protection and adapter changes](#44-retired-gnn-experiments-retained-model-protection-and-adapter-changes)
- [45. Saved-run identity and why results cannot be mixed](#45-saved-run-identity-and-why-results-cannot-be-mixed)
- [46. How to investigate weak results and compare improvements fairly](#46-how-to-investigate-weak-results-and-compare-improvements-fairly)
- [47. Handoff completeness, remaining source files and thesis-writing priorities](#47-handoff-completeness-remaining-source-files-and-thesis-writing-priorities)
- [48. Arabic explanation of the work phases, improvements and remaining problems](#48-arabic-explanation-of-the-work-phases-improvements-and-remaining-problems)
- [49. Additional partial-reference evaluation records, including weak results](#49-additional-partial-reference-evaluation-records-including-weak-results)

## 1. Starter message to give ChatGPT

Copy the following message into a new chat and attach this file:

> I am writing a thesis about SatelliteGraph, a satellite-image segmentation, spatial-graph, and temporal-change project. Read the complete attached project context before helping me. Explain in simple English because my first language is Arabic. Use Arabic explanations when I request them.
>
> Keep implemented methods, frozen experimental results, new-input training fits, and proposed future work separate. The current retained GNN is the original binary building ST-GNN. It is not a sixteen-class or sixteen-head change model. Do not invent experiments, independent ground truth, sensor metadata, accuracy, citations, architecture layers, or model weights. Attribute pretrained models and contributor code correctly.
>
> First summarize your understanding, identify the main research problem and contribution, and propose a thesis outline consistent with the implementation. Mark missing facts clearly. When drafting a chapter, explain the method, its inputs and outputs, its assumptions, its limitations, and the evidence supporting each result. Use the source map in this document to request specific code or artifacts if you need more detail. Do not recommend modifying or retraining the preserved GNN as though that were part of the completed experiment.
>
> Read the work-history and problem ledger in Sections 41–48 before describing improvements. Explain what changed between the old and new stage choices, which improvements are demonstrated, which are engineering capabilities, and which problems remain open. Do not attribute an image-evidence candidate to a GNN that returned insufficient training data, or describe a retired experiment as the retained model.

## 2. The project in plain English

SatelliteGraph compares satellite imagery from two dates. It first divides each image into labelled regions, such as buildings, trees, roads, or bare ground. It then represents these regions as a graph. Regions are nodes; their spatial relationships are edges. It combines overlapping tiles into one graph for each date and links regions between dates. Finally, it produces change candidates and, when sufficient targets exist, fits the retained GNN for a specific building-change task.

A simple example is a land region in the earlier image that contains a building in the later image. Segmentation proposes the land and building regions. Graph construction records their shape, appearance, and neighbours. Temporal matching records their overlap. Image-evidence rules evaluate the difference. The GNN uses region and relationship features to predict whether an applicable region belongs to its binary change category.

The graph is useful because a region can be interpreted with its neighbours and its corresponding observations at the other date. A roof-like patch surrounded by other buildings may provide different context from an isolated patch surrounded by vegetation. This explains why graph reasoning is relevant; it does not guarantee that every predicted change is correct.

The application includes a thesis dashboard for inspecting images, segmentation, graphs, matching, GNN outputs, changes, and evaluation evidence. It also includes a selectable pipeline workspace for new images.

## 3. Research scope and a suitable thesis description

A careful working description is:

> An object/region-based framework for multi-class satellite-image change inspection using semantic segmentation, spatial and temporal region graphs, deterministic image-evidence rules, and a building-focused relational GNN.

Possible working title:

> **SatelliteGraph: Region-Based Satellite Image Change Analysis with Spatio-Temporal Graphs and a Building-Focused GNN**

The land-cover segmentation and candidate catalogue are multi-class. The retained learned change detector is binary and building-focused. The thesis should make this distinction early.

The historic demonstration concerns Al Shouf, Lebanon, with 2013 and 2023 imagery and seven tiles, `tile_00000` through `tile_00006`. Historical documentation describes 512 × 512 source tiles, a 1.5 m grid, EPSG:32636, and an approximately 4.9 km × 0.77 km strip. These are recorded study settings, not a substitute for the original image-provider metadata. The general pipeline also accepts other images, supported years, dimensions, and coverage.

A defensible contribution can discuss the integrated workflow, interoperability of old and new stages, exact coordinate and overlap handling, explicit uncertainty/masking, reproducible artifact contracts, and the retained GNN experiment. Claims of a new foundation model, a new SegFormer architecture, or a newly invented R-GCN require evidence that this repository does not provide. The final novelty statement and individual contribution statement should be agreed with the thesis supervisor.

## 4. Four layers that must not be mixed

| Layer | What it contains | How to describe its evidence |
|---|---|---|
| Original Stage 1–5 system in `src/` | Tiling, SAM/OEM segmentation, refinement, adjacency graphs, tile combination, and polygon-based temporal change | Implemented software and deterministic engineering rules |
| General selectable runtime in `pipelines/semantic_temporal/runtime/` | Adapters and common contracts supporting old/new stage choices on new inputs | New-input processing; results require assessment for the chosen data |
| Preserved seven-tile study | Frozen segmentation/graphs/labels, original GNN code, saved predictions, reference figures and metrics | Historical pseudo-label agreement and visual-audit evidence under the recorded protocol |
| Retired experiments and old design notes | Former sixteen-head RGNN, external-training proposals, earlier configurations | Historical work, not the current model or current result |

The GNN implementation has two valid uses: reproducing its preserved study and fitting the same architecture on a new comparison through an adapter. These uses have different labels and evaluation protocols.

Some older Markdown files describe removed experiments or earlier restrictions. Current code, the current run ledger, and the current model reference files take precedence. In particular, statements about a current sixteen-head model, an external experimental checkpoint, mandatory placement metadata for every PNG, or universally unavailable frozen reproduction may be stale.

## 5. The three selectable stages and all eight combinations

The Streamlit workspace offers these choices independently:

| Stage | Old choice | New/current choice |
|---|---|---|
| Segmentation | `old`: SAM 2.1 + OpenEarthMap FasterSeg, followed by old automatic refinement | `new`: SegFormer + Delineate Anything v2 + deterministic region rules/refinement |
| Spatial graph | `old`: region adjacency only | `new`: adjacency + nearby-building and nearby-region context |
| Change detection | `classical`: old Stage 5 polygon change rules | `gnn`: original binary thesis building ST-GNN, with the current comparison catalogue |

The supported selections are:

| Segmentation | Graph | Change |
|---|---|---|
| old | old | classical |
| old | old | gnn |
| old | new | classical |
| old | new | gnn |
| new | old | classical |
| new | old | gnn |
| new | new | classical |
| new | new | gnn |

Interoperability means common labels, coordinates, masks, and artifact contracts allow these choices to connect. It does not mean all combinations have the same accuracy or the same scientific assumptions.

Two dependencies matter for a fair thesis experiment:

1. Old fused segmentation exposes confidence for its assigned label rather than a full class-probability tensor. The comparison code can therefore use the new SegFormer to recover full image evidence even when old segmentation is selected.
2. The graph method also selects the temporal-correspondence procedure: old graphs use the old Stage 5 matcher, whereas new graphs use the new overlap/roof-aware procedure. Changing the graph option is therefore not a pure edge-only ablation of the complete general pipeline.

These are real implementation details, not reasons to mislabel an experiment. Record them when comparing methods.

## 6. End-to-end data flow

```text
Earlier image(s) + later image(s) + years + optional map/placement information
    |
    v
Validate image types, masks, placement, model assets and coordinate information
    |
    v
Prepare one common coordinate frame and resolution
Estimate registration or use supplied controls where requested/required
Optionally harmonize brightness and sharpness for model inputs
    |
    v
Segment each tile with the selected old/new method
    -> instance IDs + classes + confidence/probabilities + valid mask + lineage
    |
    v
Build a graph for every tile using the selected spatial method
    |
    v
Resolve duplicate predictions and give overlap pixels exact-once ownership
Combine tile graphs into one annual graph per date
    |
    v
Build temporal correspondences, overlap relations and match components
Create common-grid image evidence and a change catalogue
    |
    +--> classical: publish adapted old Stage 5 extent-change candidates
    |
    +--> gnn: construct applicable building/built-over targets
              adapt features to the original GNN contract
              fit only if reliable positive AND negative targets exist
    |
    v
Validate scientific artifacts, render reports, bind manifests, inspect in Streamlit
```

The high-level orchestrator is `src/pipeline_select.py`. The native stage implementation is `runtime/runner.py::execute_stage`. Registration is an upstream input that is saved and reused. Downstream stages must not independently refit registration and move geometry away from its segmentation.

## 7. Inputs, image validity, and years

Supported general inputs include:

- One RGB/RGBA PNG or RGB GeoTIFF for each date.
- Tile folders with metadata and, where available, an original georeferenced raster.
- Metadata-free folders interpreted in natural filename order as a left-to-right row. This is an explicit assumed layout, not recovered geographic placement.
- A single year for segmentation and graph construction. The change stage requires two chronological years.

PNG alpha determines observation validity. GeoTIFF dataset masks determine observed pixels. Unknown land-cover class and unobserved/nodata are different states. An unknown observed pixel remains part of the image; nodata has no observation.

The general raster reader accepts uint8 RGB and converts uint16 RGB with a fixed `255/65535` scaling. It does not estimate a visual contrast stretch from each image. This matters when interpreting radiometric differences.

Tile filename, tile index, and local segment ID do not establish identity between dates. Two files both called `tile_00002.png` can contain different ground coverage. Conversely, two different file paths can contain identical imagery. Identity between dates must be inferred from valid coordinates and image/region evidence.

Record actual acquisition dates separately from user-entered year labels. Renaming a run as 2018/2024 does not turn 2017/2023 imagery into new observations.

**Source:** `runtime/inputs.py`, `runtime/contracts.py`, `src/folder_pipeline.py`.

## 8. Coordinates, resolution, and Stage 1 tiling

### 8.1 Coordinate frames

The project distinguishes local tile pixels, source-image pixels, and map coordinates. In Rasterio/Affine ordering, `[a,b,c,d,e,f]` means:

```text
X = a*x + b*y + c
Y = d*x + e*y + f
```

This array ordering is not GDAL's serialized geotransform ordering. Tile offsets are applied once. Polygon vertices describe pixel edges; adding an extra half-pixel offset would shift boundaries incorrectly. A node's inside-polygon representative point and its geometric centroid serve different purposes.

Mapped inputs are prepared in a common projected metre CRS. The general runtime can select an appropriate metre CRS, including a local UTM CRS when needed. It rejects unrelated mapped and pixel-only frames. In the general path, the later image is the reference for earlier-to-later registration. The older multi-year Stage 5 API instead registers later snapshots to its earliest reference; do not copy one API's direction into the other's description.

An unmapped PNG can be processed in image-relative coordinates. If physical metres per original pixel are known and supplied, physical scale can be used without inventing an absolute map location. Without that scale, distances and areas remain pixels and pixels squared. Variable names ending in `_m` or `_m2` do not make such measurements physical.

### 8.2 Original tiling stage

`src/io/tile_image.py` and `src/io/raster_tiling.py` prepare tiles from a source GeoTIFF. They preserve the original input. The default recorded tile size is 512, overlap 50, and nominal stride 462 pixels.

With edge mode `shift`, the last tile row/column moves inward to cover the source boundary without padding. Final overlap can exceed 50 pixels. With edge mode `pad`, a regular grid and transparent nodata padding are explicit. Metadata records offsets, valid extents, affine information, and observation counts.

Resolution mode `preserve` resizes the full footprint in its source CRS. A requested metre size for a geographic CRS is approximate at the image centre. Mode `projected` explicitly reprojects to a metre grid. It can create masked margins. Resizing, reprojection, registration, and tile stitching are four different operations.

### 8.3 Upsampling in new segmentation

The new segmentation works on a grid with twice the prepared height and width. Its affine pixel size is halved, so area per processing pixel becomes one quarter. This is interpolation for processing; it does not create new optical information or improve the sensor's actual ground sampling distance.

## 9. Registration and brightness/sharpness harmonization

Registration estimates how the same stable locations align between dates. Without adequate alignment, roof edges, roads, and vegetation boundaries can appear to move and produce false changes.

Mapped inputs use current-image landmark procedures involving phase guidance, SIFT descriptors and robust affine fitting, with fallbacks where appropriate. Pixel-only pairs use `pixel_registration.py`, combining phase-guided and mutual-descriptor candidates. A RANSAC affine fit rejects inconsistent landmark pairs, implausible reflection/scale, or insufficient distributed support.

The current general path automatically requests registration for two unmapped inputs unless supplied controls are available. Therefore an unchecked registration checkbox does not mean an unmapped PNG pair will bypass alignment. Supplied control points are expressed in the relevant common coordinate units.

Pixel affine fitting requires at least six distinct, two-dimensionally distributed inliers, at least 50% inlier support in the final fit, positive orientation, and supported scale bounds. Pixel scale bounds differ from the stricter mapped-world fit. Default residual tolerance is 2 comparison units, metres or pixels according to the frame. Fitted RMSE and p95 residuals are in-sample diagnostics, not independent registration accuracy.

Roof-offset compensation is separate. In metric new graphs, current building masks estimate roof displacement for building-to-building comparisons. This compensates a specific apparent roof shift; it is not a universal physical movement model. Other land-cover classes remain in the ground comparison frame.

Optional `--harmonize-images` matches the earlier image's per-channel histogram to the later image over common observed non-shadow pixels. It also blurs the sharper image toward the other image's measured sharpness using bounded Gaussian smoothing. Defaults include a 2% sharpness tolerance, maximum sigma 3, and 14 bisection steps. It records lookup tables and statistics. Original RGB is retained separately for review.

Harmonization can reduce appearance differences caused by imaging conditions. It cannot prove that a remaining difference is a real change or recover missing detail. Historical study offsets and audit labels are not reused as registration evidence for arbitrary new inputs.

**Source:** `runtime/registration.py`, `runtime/pixel_registration.py`, `runtime/preprocessing.py`, `src/stage5/registration.py`, `runtime/roof_offsets.py`.

## 10. Semantic classes and the three different ID systems

These mappings are essential when explaining adapters or reading artifacts:

| Meaning | Old fused Stage 2 ID | Downloaded new SegFormer channel | Canonical runtime ID |
|---|---:|---:|---:|
| Unknown/background observation | 8, project fusion unknown | 0, Background | 0 |
| Building | 7 | 8 | 1 |
| Road | 3 | 4 | 2 |
| Tree | 4 | 5 | 3 |
| Grass/shrub or rangeland | 1 | 2 | 4 |
| Cropland/agriculture | 6 | 7 | 5 |
| Bare ground/bareland | 0 | 1 | 6 |
| Paved/developed space | 2 | 3 | 7 |
| Water | 5 | 6 | 8 |
| Unobserved/nodata | 255 | masked outside observations | 255 |

The old FasterSeg has eight semantic output channels numbered 0–7. The old fused segmentation adds unknown ID 8. The new SegFormer has nine channels including background. Canonical runtime ID 8 means water, whereas old fused ID 8 means unknown. A direct ID copy between systems would be wrong.

Instance ID is a different concept: each segment has its own positive integer instance ID. Instance-map value 0 means no segment. Class-map value 0 means an observed unknown class in the canonical runtime. Do not interpret one raster using the other raster's meaning.

**Source:** `src/labels.py`, `src/stage2/classes.py`, `runtime/adapters.py::OLD_TO_CANONICAL`, `runtime/segmentation/semantic.py::OEM_TO_CLASS`.

## 11. Old segmentation: SAM 2.1 + OpenEarthMap FasterSeg

### 11.1 What each model does

SAM 2.1 Hiera-Large supplies class-agnostic masks. It proposes where region boundaries might be; it does not name the region building, tree, or road. The project uses its automatic-mask generator on still images, not a learned temporal-video change detector.

OpenEarthMap FasterSeg supplies the eight land-cover class scores. Its externally supplied architecture and weights are loaded from local files through the vendored lightweight implementation. SAM geometry and OEM semantics are fused deterministically.

Key local assets:

```text
checkpoints/sam2/sam2.1_hiera_large.pt
checkpoints/openearthmap/fasterseg/arch_1.pt
checkpoints/openearthmap/fasterseg/weights1.pt
third_party/sam2/
third_party/oem-lightweight/
```

The SAM paper describes the foundation-model family; the selected implementation/configuration is SAM 2.1 Hiera-Large. Architecture attribution belongs to the external model authors. The project does not train SAM or FasterSeg from scratch. See [SAM 2](https://arxiv.org/abs/2408.00714) and [OpenEarthMap](https://arxiv.org/abs/2210.10732) for the corresponding primary research context.

### 11.2 Processing steps

1. Validate the tile, metadata, RGB image, and valid mask.
2. Prepare model RGB according to the selected preprocessing configuration.
3. Run SAM automatic mask proposal generation.
4. Normalize masks to the tile frame and observed footprint.
5. Filter unsupported, tiny, oversized, or insufficiently observed candidates.
6. Remove near-duplicate masks deterministically.
7. Run FasterSeg semantics.
8. Fuse ordered SAM masks with semantic labels, subtracting pixels already claimed.
9. Add confident semantic components where SAM has left gaps.
10. Assign explicit unknown to remaining observed pixels.
11. Export immutable Stage 2 results and then apply Stage 2R refinement.

Recorded SAM defaults include points per side 32, batch size 8, predicted-IoU gate 0.82, stability gate 0.90, one crop layer, NMS 0.7, and minimum mask-region area 40. Project mask filtering includes minimum area 40, maximum tile-area ratio 0.95, and minimum valid fraction 0.5. Deduplication uses IoU 0.90 and containment 0.95. These defaults are engineering settings; use the saved configuration for any reported experiment.

FasterSeg inference uses its configured 1024 × 1024 input and recorded OEM normalization. Its optional TTA combines original and horizontal-flip logits before softmax. This differs from the new SegFormer's four-flip probability averaging.

Fusion orders masks using SAM predicted IoU, stability, area, and deterministic identifiers. Labels are selected by an unweighted majority among confident semantic pixels, with default confidence threshold 0.45. A mask with no confident votes becomes unknown. Its surviving connected pieces become unique instances.

The old confidence map is final-label support: where an OEM prediction agrees with the selected fused label it contributes confidence; disagreeing pixels contribute zero. Fusion unknown has zero confidence. This is not a complete probability distribution over all classes and is not a physical-change confidence.

### 11.3 Stage 2R refinement

Stage 2R preserves `12_final` and writes `13_refinement`. It works on the instance raster, then rebuilds masks, records, geometry, and lineage. It offers `human_review` and `automatic` modes; the selectable old pipeline uses automatic mode.

It normalizes four-connected components internally, preserves original segment ancestry, and controls topology-only proliferation. Secondary small components can be consolidated back into the original object identity. A topology budget prevents automatic splitting from arbitrarily increasing published object counts; approved semantic splits are treated separately.

Generic automatic merging/absorption is selective and abstains when evidence is inadequate. Protected classes include buildings, roads, trees, agriculture, and unknown. Automatic proposal gates include a minimum score of 0.99 plus independent contact, boundary, appearance, semantic, and interior-support conditions. A high proposal score alone is not sufficient.

Micro-unknown cleanup and its bounded contextual pass handle eligible unknown areas of at most 40 pixels. The final omission/contraction operation does not delete observed pixels: it absorbs a directly adjacent micro-unknown into a known neighbour according to longest shared boundary, then target probability, then deterministic lowest instance ID. An isolated unknown can remain unknown. The observation mask is preserved.

Human-review proposals are checksum-bound. Explicitly supplied semantic partitions can be replayed through the core API; automatic mode does not invent and approve semantic split boundaries.

**Source:** `src/stage2/`, `src/refinement/`, `runtime/adapters.py::run_old_segmentation`; details in `docs/stage2-segmentation.md` and `docs/stage2-refinement.md`.

## 12. New segmentation: SegFormer + Delineate Anything v2 + region rules

### 12.1 SegFormer supplies land-cover probabilities

The local checkpoint is `boehnen/satlens-segformer`, loaded with Transformers from local files. Its recorded configuration declares:

| Setting | Value |
|---|---|
| Architecture class | `SegformerForSemanticSegmentation` |
| Encoder stage widths | 32, 64, 160, 256 |
| Blocks per stage | 2, 2, 2, 2 |
| Attention heads per stage | 1, 2, 5, 8 |
| Spatial-reduction ratios | 8, 4, 2, 1 |
| Decoder hidden width | 256 |
| Output channels | 9, including background |

The SegFormer family uses a hierarchical transformer encoder and a lightweight MLP decoder. The local checkpoint's settings, rather than a guessed model name such as B5, should be reported. See the [SegFormer paper](https://arxiv.org/abs/2105.15203) for architecture attribution. Its benchmark scores are not SatelliteGraph results.

The runtime normalizes RGB using ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`. Default inference uses 512-pixel windows with 128 overlap. Overlapping window probabilities are blended with separable Hann weights clipped to a floor of 0.05. TTA averages original, horizontal, vertical, and both-axis flip probability maps.

Semantic inference is evaluated at 1×, 1.5×, and 2× image scale. The outputs are aligned to the 2× grid:

| Product | Meaning | Important caution |
|---|---|---|
| `probabilities.npy` | Mean canonical class probabilities across the three scales | Used for region semantic confidence |
| `shape_probabilities.npy` | Class probabilities from the 2× SegFormer evaluation | The name does not mean geometric/parcel confidence |
| `max_probabilities.npy` | Channelwise maximum over the scales | Channels need not sum to one; useful as conservative presence evidence |

### 12.2 Delineate Anything supplies parcel masks

`DelineateAnythingv2.pt` is an externally supplied field/parcel instance model loaded through `ultralytics.YOLO`. It supplies candidate region geometry and instance confidence, not the final eight land-cover class labels. Its publisher source is [MykolaL/DelineateAnything](https://huggingface.co/MykolaL/DelineateAnything).

The runtime uses native and enlarged image crops, default detection confidence 0.20, mask resizing, rejection of masks truncated by internal crop boundaries, and cross-window duplicate suppression above IoU 0.5. Final semantic interpretation still comes from the land-cover probabilities and deterministic rules. The exact internal layer-by-layer parcel network should not be invented from the filename; the verified integration interface is the Ultralytics segmentation/mask API and the pinned asset.

### 12.3 Region construction sequence

The implemented order is buildings, roads, water/pools, smoothed land classes, accepted parcel regions, residual class regions, coverage filling, large-region splitting, connectivity/small-region cleanup, and same-class refinement.

Important current defaults, expressed in the prepared grid's physical units when available and otherwise its declared image-relative units:

| Step | Implemented behaviour/defaults |
|---|---|
| Buildings | 2× semantic building argmax and probability ≥ 0.4; four-connected opening; minimum area 27 |
| Building separation | Narrow-neck splitting with erosion radius about `1.5/pixel_size`, minimum core area 20, followed by watershed/connectivity |
| Roads | Mean-probability road argmax outside buildings; closing width about `3.75/pixel_size`; minimum area 100 |
| Road instances | Skeleton branches between junction zones; short-spur absorption using length default 15 |
| Water | Semantic water probability ≥ 0.4 or pool RGB evidence; minimum water area 10 |
| Pools | OpenCV hue 80–100, saturation ≥ 110, value ≥ 100; opening; minimum area 6 |
| Land smoothing | Approximate width `5.25/pixel_size` for trees, grass/shrub, cropland, bare soil, paved/developed |
| Parcel acceptance | Area ≥ 150 and smoothed land-class purity ≥ 0.65; smaller accepted parcel masks claim pixels first |
| Residual land | Minimum component area 100 before completion/cleanup |
| Large land regions | Split above area 8000, or above 3000 with compactness below 0.06; image-gradient watershed targets pieces around area 3000 |

The building code actually calls narrow-neck splitting. An older docstring saying “no instance splitting” does not describe that executed call.

Bare-ground colour rules use excess green, local vegetation fraction, redness, and brightness:

```text
excess_green = (2*G - R - B) / (R + G + B + 1)
redness     = (R - B) / (R + B + 1)
```

Defaults include excess-green threshold 0.04, vegetation fraction below 0.05, redness at least 0.08, brightness at least 100, and smoothed bare-rule share at least 0.5. Selected grass/crop/paved labels can be overridden by this evidence. Such rules are useful engineering heuristics, but roofs, soil, shadows, and dry vegetation can have similar RGB appearance.

### 12.4 Refinement and uncertainty

New same-class refinement does not merge buildings. Road merges avoid junctions with a third road segment within the configured radius. Other same-class merges need sufficiently broad contact, weak boundary evidence, and similar appearance, texture, and semantic profile.

Recorded general limits include edge ratio ≤ 1.25, appearance distance ≤ 0.9, texture ratio ≤ 1.6, semantic L1 distance ≤ 0.45, and contact length ≥ `max(4, 0.35*sqrt(smaller_area))`. Parcel pairs use stricter edge/appearance limits 1.05 and 0.5.

The final exporter restores explicit unknown components where averaged semantic argmax is background, preserves unobserved pixels, normalizes connected regions as required, and completes observed-pixel coverage. The final segmentation is a partition of observed pixels, not a guarantee that every region is one real-world object.

### 12.5 Why some semantic scores are very low

A bare-ground or water region can be selected by a colour rule even if SegFormer assigns little probability to that class. The exported semantic score remains the raw mean probability of the assigned class over the region. It is not replaced with a large rule-support number.

Separate `rule_support.tif`, `rgb_rule`, and mean rule-support fields preserve the rule provenance. New generated bare/water nodes follow the original feature contract's rule-based flag, even when measured rule support is zero. Historical artifacts without this information remain unknown; the runtime does not reconstruct it from the class name alone.

This separation is scientifically useful: semantic model confidence, rule support, and physical-change confidence describe different things.

**Source:** `runtime/models.py`, `runtime/segmentation/semantic.py`, `regions.py`, `parcels.py`, `refinement.py`, and `segmentation/__init__.py`.

## 13. Canonical segmentation artifacts and adapters

Each canonical segmentation has a year, tile key, output directory, grid, shape, paths, and lineage. Typical artifacts are:

```text
artifact.json             typed description of the product
segment_ids.tif           positive instance IDs; 0 outside instances
segment_classes.tif       canonical semantic class IDs; 255 nodata
valid_mask.tif            observed-data mask
probabilities.npy         full probabilities, or explicitly marked old assigned-label support
shape_probabilities.npy   new 2× semantic evidence, where available
max_probabilities.npy     new multi-scale maximum evidence, where available
rgb.tif                   model-input RGB on the processing grid
original_rgb.tif          separately retained original appearance, where available
rule_support.tif          separate new colour-rule support, where available
nodes.csv                 region summary table
lineage.json              ancestry of instances
validation.json           checks and evidence-quality flags
merge_log.csv             applicable new refinement operations
manifest.json             hashes for the artifact files
```

Not every method provides every optional raster. The artifact path dictionary and evidence-quality flag determine what is available.

`runtime/adapters.py` converts the old refined segmentation into canonical IDs. Its probability tensor contains only confidence for the assigned label; other channels are zero. It explicitly records `assigned_label_confidence_only`. The comparison code can request full SegFormer evidence rather than pretend this partial tensor is a full softmax distribution.

`connected_region_view` splits disconnected components when the new graph method requires connected regions. It preserves class pixels, probabilities, rule support, and original lineage. This change in graph-region identity does not retrain segmentation or silently invent new class labels.

The adapters are interfaces. They are not new foundation models. Their correctness is important because class-ID or feature-order mistakes can change results even when the neural network itself is untouched.

## 14. Spatial graphs: old adjacency and the improved graph

### 14.1 Nodes

A node represents one segmented region observation at one date. It stores exact geometry, area, perimeter, semantic class, semantic confidence, probability summaries, representative point, centroid, morphology, RGB/HSV statistics, evidence flags, and lineage.

Examples of morphology:

```text
compactness = 4*pi*area / perimeter^2
solidity    = area / convex_hull_area
elongation  = long_side / short_side of the minimum rotated rectangle
```

Orientation is axial, so `sin(2*theta)` and `cos(2*theta)` avoid treating 0° and 180° as different orientations. Hue is also encoded circularly. The inside-polygon representative point is useful for displaying an edge without placing the node in a polygon hole.

### 14.2 Old graph

The old graph contains undirected region adjacency. Two regions are adjacent when they share at least one pixel side. Corner contact alone does not create an adjacency edge. Graph construction does not change segmentation geometry or perform another neural inference.

The original Stage 3 graph also has a separate analysis/display view that can suppress noisy micro-unknowns and minimal contacts. Exact graph artifacts remain authoritative.

### 14.3 New graph

The new graph retains adjacency and adds:

- `near`: nearby building-to-building pairs that do not already have adjacency.
- `context_near`: nearby known regions providing broader land-cover context, excluding duplicate building-only `near` pairs.

Default polygon-gap radius is 25 coordinate units, metres or pixels according to the grid. Candidate selection uses up to six neighbours per node before undirected union; another node can select a node too, so final degree can exceed six. Unknown regions do not receive these proximity-context edges. Corner-touching polygons can have zero polygon gap and still be non-adjacent under the pixel-side definition.

This provides context across streets, gardens, and small gaps that adjacency alone misses. It is a richer representation, not an independently established accuracy improvement for every image.

### 14.4 Spatial edge attributes

The canonical nine attributes are:

| Order | Attribute | Meaning |
|---:|---|---|
| 1 | `shared_boundary` | Length of shared border |
| 2 | `gap` | Minimum polygon-to-polygon separation |
| 3 | `distance` | Distance between representative node positions |
| 4 | `boundary_frac_source` | Shared border divided by source perimeter |
| 5 | `boundary_frac_target` | Shared border divided by target perimeter |
| 6 | `color_contrast` | Euclidean distance of mean RGB vectors divided by 255 |
| 7 | `same_class` | Whether endpoint class IDs agree |
| 8 | `dir_sin` | Sine of edge direction |
| 9 | `dir_cos` | Cosine of edge direction |

Canonical directed tensor copies swap source/target border fractions and negate direction components on reversal. The original GNN's relation converter has its own preserved reverse-edge convention, described later.

### 14.5 Canonical graph features are not the GNN feature contract

The general spatial tensor has **40 node features**: nine class indicators including unknown, geometry/orientation, RGB/HSV summaries, nine class probabilities, confidence, evidence-quality flags, and relative position. The temporal tensor appends `is_later`, making **41**.

These general graph tensors are not simply fed into the retained GNN. Its adapter reconstructs the original **38-feature** contract. Confusing 40/41 with 38 would misdescribe the current implementation.

**Source:** `src/stage3/`, `runtime/graphs.py::_row`, `_edges`, `_persist`, `NODE_FEATURES`, `EDGE_FEATURES`.

## 15. Combining overlapping tiles into annual graphs

Overlapping tiles can predict the same object twice. Concatenating their graphs would duplicate area, nodes, and edges. Stage 4 instead performs overlap-local identity matching followed by exact-once pixel ownership.

1. Place all tiles in the same source/prepared grid using their actual offsets and affine metadata.
2. Compare instance masks only in their shared tile overlap.
3. Require same-class overlap support. Default gates include intersection ≥ 4 pixels, intersection-over-smaller ≥ 0.60, IoU ≥ 0.20, and overlap-visible coverage ≥ 0.40 at both endpoints. Unknown-to-unknown uses the stricter 0.80 intersection-over-smaller gate.
4. Use mutual-best matching and safe union-find. A merged group may not contain two predictions from the same tile.
5. Give each represented overlap pixel to one tile: prefer an actual segment, then greater normalized distance from the tile edge, then semantic confidence, then stable tile order.
6. Polygonize/dissolve owned pixels and rebuild geometry and adjacency. For new graphs, recompute proximity edges on the annual objects.

The overlap match score is:

```text
0.45*intersection_over_smaller + 0.35*IoU + 0.20*minimum_bidirectional_coverage
```

It is an engineering identity score, not a calibrated probability.

Annual class probabilities and appearance are recomputed using owned pixels. Rule support is separately ownership-weighted. If contributing owned pixels lack provenance, the aggregate provenance remains unknown. Matched complementary pieces can form a multipolygon; graph-region identity should not automatically be called one individual physical building or tree.

Unequal tile counts and different tile sizes can be valid. Non-overlapping selections can produce multiple graph components. Equal tile counts do not establish equal ground coverage.

**Source:** `src/stage4/pipeline.py`, `runtime/graphs.py::combine_year_graphs`; see `docs/stage4-graph-combination.md`.

## 16. Temporal graph and correspondence across dates

The temporal graph retains each date's region observations and spatial edges, then adds directed relations across years. A node in 2013 and a node in 2023 are separate observations, even when they likely represent the same area. The elapsed year gap is retained as `delta_years`.

### 16.1 Coverage and observability

Yearly coverage is the union of observed polygons. Common coverage is their intersection. Missing coverage cannot safely become demolition, construction, or an unchanged training negative.

Nodes receive `observed_fraction` and `known_counterpart_fraction`. The latter measures overlapping counterpart area supported by a known class and adequate confidence. These gates can prevent GNN training when the counterpart segmentation is uncertain, even when a region looks plausible to a human.

### 16.2 Old graph temporal matching

The old graph choice uses `src/stage5/matching.py`. It compares overlap, distance, area ratio, compactness similarity, and neighbourhood-class context. Default geometric score:

```text
I = intersection_area / union_area
D = exp(-centroid_distance / max(search_radius, sqrt(min(area_A, area_B))))
A_sim = min(area_A, area_B) / max(area_A, area_B)
H = min(compactness_A, compactness_B) / max(compactness_A, compactness_B)
N = average of neighbourhood-distribution agreement and class-count cosine
S = 0.55*I + 0.15*D + 0.10*A_sim + 0.10*H + 0.10*N
```

Default one-to-one gates include IoU ≥ 0.20, score ≥ 0.60, and ambiguity margin ≥ 0.05 at both endpoints. Candidate assignment includes unmatched choices. Accepted local coordinate rescue and lineage split/merge conditions are conservative. Same semantic class is not mandatory for every temporal correspondence, because a real class transition can happen.

The old Stage 5 API retains candidate evidence and rejection reasons. Its `track_id` is an inferred correspondence chain, not proof that an individual real object persisted.

### 16.3 New graph temporal matching

For the new graph choice, an STRtree searches overlapping candidates. Building-to-building comparisons can use the current roof-offset compensation; other comparisons stay in the ground frame.

Let `shared` be compared intersection area, `A` and `B` be full region areas:

```text
fraction_source = shared / A
fraction_target = shared / B
IoU             = shared / (A + B - shared)
overlap_score   = sqrt(fraction_source * fraction_target)
```

An edge needs at least six processing-pixel areas of intersection and at least 25% of the smaller region area. Confidence is high for same-class IoU ≥ 0.5; medium for supported same-class overlap score ≥ 0.3 or cross-class overlap score ≥ 0.5; otherwise low. A supported contained-in case can rescue a lower-overlap same-class relation. Non-low edges are primary; low overlap edges remain secondary.

Primary components are classified as one-to-one, split, merge, many-to-many, or unmatched. A split or merge can arise from a segmentation difference rather than a physical event. It must not automatically become a positive change label.

The canonical temporal tensor has forward/backward channels. Its reverse attribute construction swaps endpoint fractions and negates roof-shift components. The retained GNN uses the original converter's convention rather than silently changing that frozen implementation.

**Source:** `runtime/temporal.py::build_temporal_graph`, `temporal_edge_features`, `build_primary_components`, `_roof_polygons`; `src/stage5/matching.py` and `refinement.py`.

## 17. Change evidence, candidate catalogue, and the classical method

### 17.1 A label difference is not automatically a physical change

Segmentation can differ because of season, shadow, illumination, registration, sensor viewing angle, interpolation, or model error. The project therefore separates raw candidates, stronger image evidence, training targets, structural observations, and GNN predictions.

`runtime/all_class_evidence.py::detect_evidence` compares the class maps and class-probability products in the owned common grid. It excludes unobserved data, unknown semantics, and dark/shadow pixels from confident evidence. Building evidence can use roof-aligned masks. Other classes remain in the ground frame; stable roof disagreement is excluded from unrelated land-change evidence.

For class gain, the later class should be present and the earlier class should be absent in a local neighbourhood. Loss reverses those roles. The maximum-over-scales tensor makes absence conservative: evidence of the class at any evaluated scale can prevent an absence conclusion. It is therefore not enough that the earlier argmax label differs.

The default policy is:

| Parameter | Default |
|---|---:|
| General present-class probability | 0.60 |
| Building presence probability | 0.50 |
| Maximum probability permitted for absence | 0.20 |
| Absence neighbourhood width | 3.75 comparison units, converted to pixels |
| Shadow grayscale cutoff | 20 |
| Reliable semantic confidence | 0.60 |
| Minimum observed fraction | 0.95 |
| Minimum known counterpart fraction | 0.95 |
| General positive changed fraction | 0.30 |
| Supported stable fraction | 0.95 |
| Building reliable component area | 100 area units |

Class-specific minimum connected areas are building 40, road 100, tree 150, grass/shrub 150, cropland 150, bare soil 100, paved/developed 100, and water 100. These policy values are interpreted in metres/metres squared for metric comparisons and in the declared image-relative units otherwise. They should not be silently presented as square metres for an unscaled PNG.

`boundary_margin` uses an explicitly supplied tolerance or a fitted registration p95 residual. It prevents small boundary movements within that tolerance from becoming confident evidence. It cannot measure unobserved external registration error.

### 17.2 Raw candidates and reliable targets are different products

`candidate_masks` retain eligible class differences above minimum area, even if the present/absent probability gates fail. This allows broad review. Strong masks additionally require those semantic, absence, observation, and alignment conditions.

`change_catalog.py` creates:

- Canonical region events such as gain, loss, conversion, expansion/contraction descriptors, and compound shape changes.
- Source and target node IDs, class information, exact geometry, area, related tasks, and evidence status.
- Structural observations such as graph neighbourhood/connectivity or matching changes, kept separate from physical events.
- A class-transition matrix and encoded transition raster.
- Node/task training targets with eligibility and explicit rejection reasons.

The sixteen gain/loss task names come from eight classes × two directions. They are evidence/target bookkeeping, not sixteen GNN heads. A target row can be present but ineligible or unlabeled.

The transition encoding is `16*source_class + target_class` for known pairs, 0 for unknown transitions, and 65535 for unobserved pixels. A transition matrix includes unchanged diagonal pixels. It is not a matrix of verified real-world conversions.

For a node, confident training eligibility checks observed coverage, known counterpart support, clear/aligned evidence, and semantic confidence. An eligible building positive needs a sufficiently large reliable component; general non-building positives require both minimum component area and changed-area share. Supported unchanged labels require same-class correspondence, a stable fraction of at least 0.95, and no reliable changed area.

Grass/shrub, cropland, and bare-soil class evidence receives conservative seasonal treatment in the catalogue. Conversion events can inherit seasonal sensitivity when one participating class is season-sensitive. This is a heuristic flag, not a seasonal model or a measured probability that season caused the difference. It can be too broad or incomplete.

Reviewed labels can be supplied by CSV, with columns `gid,task,label,confidence,source`. They are validated, and review cannot override missing coverage, unknown semantics, or a wrong class/year role. The source and review protocol must be recorded before calling these independent reference labels.

### 17.3 What the old classical selection actually publishes

`detect_classical` preserves the previous engine's result in `legacy_classical_changes.json` and its targets separately. It also builds the common image-evidence catalogue and transition artifacts.

When `--change classical` is selected, the displayed canonical events are adapted from the old Stage 5 polygon extent events: appeared/constructed, disappeared/demolished/removed, expanded, contracted, and reconfigured. Their old class IDs and geometry fields are converted once. They are published as review candidates; selecting classical does not train a GNN.

When `--change gnn` is selected, the general all-class image-evidence catalogue is used for review and supervision, followed by the retained building GNN if eligible labels permit fitting. Thus two runs with different change choices need not display an identical event list even when their segmentation and spatial graphs are the same.

Old Stage 5 change defaults in `config.yaml` include minimum area 4, minimum relative change 0.05, boundary tolerance 1, observed fraction 0.95, confidence 0.60, and known fraction 0.95. These are distinct from the common catalogue policy above; do not combine their threshold tables as though they were one rule set.

**Source:** `runtime/all_class_evidence.py`, `change_catalog.py`, `change_contracts.py`, `temporal.py::detect_classical`, `src/stage5/events.py`.

## 18. The retained GNN: location, provenance, and exact task

**Implementation:** `pipelines/legacy_building_stgnn/original/gnn_change.py`, class `RGNN`.  
**Current integration:** `pipelines/legacy_building_stgnn/adapter.py`.  
**Runtime entry:** `pipelines/semantic_temporal/runtime/gnn.py`, which delegates to the adapter.  
**Preservation checks:** `pipelines/legacy_building_stgnn/runner.py` and `reference/code_manifest.json`.

Git records Roaa Elmir's integration commit `a9f9df733fb944b2cdcf647dcdd5e0b41236e019`, dated 11 October 2026. This establishes integration credit; it does not independently prove authorship of every original algorithm. The original source snapshot's manifest records release creation on 4 October 2026.

This is project-specific PyTorch model code, not an online pretrained GNN checkpoint. The original implementation initializes a model for training and does not save trained weights. A `.pt` file named `stgraph_tensors.pt` contains input graph tensors; it must not be described as a trained GNN weight file.

The binary question depends on the node's role:

| Applicable node | Meaning of a positive GNN label |
|---|---|
| Later-year building | This building is new or extended |
| Earlier-year non-building land | This land region is tagged as built over |

Later non-buildings and earlier buildings are not these target roles. Demolition, tree loss, water loss, and every possible land-cover conversion are not separate outputs of this retained network. The UI's `building_gain` and `built_over` status entries describe two roles of one shared binary model; they are not two separately trained heads.

The removed experimental sixteen-head RGNN, its training controls, and experimental checkpoint must not be used to describe the current network. The identical `pipeline/gnn_change.py` inside the frozen experiment package is a preserved study copy, not a second experimental enhancement.

## 19. Exact GNN input contract

### 19.1 Input is a graph, not a pair of raw images

The model receives:

1. `x`: an `N × 38` node-feature matrix.
2. A dictionary of six relation channels. Each channel has an edge-index array and an edge-attribute matrix with ten columns.
3. Binary labels and masks during training.

The images influence these inputs through segmentation, morphology, appearance summaries, spatial relationships, temporal overlap, and label/evidence generation. The GNN itself does not contain a CNN that reads the original RGB images.

### 19.2 All 38 node features, in order

```text
 1 class_building          2 class_road
 3 class_tree             4 class_grass_shrub
 5 class_cropland          6 class_bare_soil
 7 class_paved_developed   8 class_water
 9 log_area              10 log_perimeter
11 compactness           12 solidity
13 log_elongation        14 orient_sin2
15 orient_cos2           16 mean_r
17 mean_g                18 mean_b
19 std_r                 20 std_g
21 std_b                 22 mean_s
23 mean_v                24 hue_sin
25 hue_cos               26 p_building
27 p_road                28 p_tree
29 p_grass_shrub         30 p_cropland
31 p_bare_soil           32 p_paved_developed
33 p_water               34 confidence
35 rule_based_class      36 pos_x
37 pos_y                 38 year_2023 / is_later
```

The original study calls the last indicator `year_2023`. The adapter supplies the equivalent “is the later input year” indicator, so new comparisons are not hard-coded to the literal year 2023.

Area and perimeter use natural logarithms; elongation is clipped to at least one before its logarithm. Orientation uses doubled-angle sine/cosine. RGB means, standard deviations, saturation, and value are divided by 255; hue uses circular sine/cosine. Positions are relative to graph bounds.

Continuous features are standardized with population standard deviation plus `1e-6`. Class indicators and `rule_based_class` are not standardized. The year indicator is appended afterward. Original study normalization uses the complete strip, which matters for evaluation interpretation.

For original rule-based bare-ground/water nodes, feature 34 represents rule support. It is distinct from feature 31 or 33, the corresponding semantic probability. The current adapter restores this convention only for known valid provenance, while retaining raw semantic confidence in the source graph, review score, and target gates.

### 19.3 Six relation channels

```text
spatial_touches
spatial_near
t_primary_fwd
t_primary_bwd
t_overlap_fwd
t_overlap_bwd
```

Primary temporal channels combine identity and contained-in relationships. Spatial channels allow reciprocal messages. Forward and backward temporal channels allow both dates to contribute context. `context_near` is retained in the common graph but excluded from this original GNN's inputs.

The nine spatial attributes are:

```text
shared_boundary_m, gap_m, distance_m,
boundary_frac_source, boundary_frac_target,
color_contrast, same_class, dir_sin, dir_cos
```

They are padded with one zero column to ten dimensions. The ten temporal attributes are:

```text
iou, frac_2013, frac_2023, overlap_score, shared_m2,
class_agreement, confidence_ord, frame_roof,
roof_shift_dx_m, roof_shift_dy_m
```

Confidence ordinal is high/medium/low → 2/1/0. Each relation's attribute columns undergo `nan_to_num` and standardization before use.

An important exact convention: the retained `original.relations()` duplicates attribute rows for reverse spatial messages and backward temporal channels. It does not swap endpoint fractions or negate direction/roof-shift attributes for the reversed messages. Canonical tensors use a different explicit reverse convention. The adapter calls the original function and preserves it; the thesis must describe the implementation actually used.

### 19.4 Verified frozen graph size

The frozen graph has 7,020 nodes: 3,744 earlier observations and 3,276 later observations. Its raw edge records are:

| Relation | Count |
|---|---:|
| Spatial touches | 16,873 |
| Spatial near | 2,266 |
| Temporal identity | 1,739 |
| Temporal contained-in | 411 |
| Temporal overlap | 5,767 |
| **Total raw records** | **27,056** |

The original relation converter creates 54,112 directed message edges from these records. This count belongs to the frozen strip, not an arbitrary new upload.

**Source:** `original/build_graph.py::node_features`, `original/build_mosaic.py`, `original/gnn_change.py::relations`; local frozen `strip/stgraph_tensors.pt` and `stgraph_validation.json`.

## 20. Exact GNN architecture and message passing

The network implements relational message passing directly in plain PyTorch. It does not require PyTorch Geometric, and it is not a GAT, LSTM, Transformer change detector, Siamese pixel network, or sixteen-head classifier.

### 20.1 Layer layout

| Part | Dimensions and operations |
|---|---|
| Input | 38 features per node |
| Relational layer 1 | Self projection 38 → 64; per-relation neighbour projection 38 → 64; per-relation edge projection 10 → 64; aggregate; ReLU; dropout 0.3 |
| Relational layer 2 | Self projection 64 → 64; per-relation neighbour projection 64 → 64; per-relation edge projection 10 → 64; aggregate; ReLU; dropout 0.3 |
| Binary head | Linear 64 → 64; ReLU; dropout 0.3; Linear 64 → 1 |
| Output conversion | Sigmoid converts one logit per node into a model probability |

For node `i`, layer `l`, and relation type `r`, the implemented computation is:

```text
message(j -> i, r) = W_r[l] * h_j[l] + U_r[l] * edge_features(j,i)

z_i = W_self[l] * h_i[l] + b_self[l]
      + sum over r of mean incoming messages of relation r

h_i[l+1] = Dropout_0.3(ReLU(z_i))
```

Incoming messages are averaged separately for each relation, then relation contributions are summed. A zero incoming degree is clamped to one so the division remains defined. The self projection has a bias. Relation neighbour and edge projections have no bias.

The self-node term is a learned projection, not a separate identity residual/skip connection. There is no explicit self-loop relation channel. Two layers allow information to propagate through two message-passing steps, including spatial and temporal context where those edges exist.

A plain explanation is: each region first describes itself, then receives separate summaries from touching regions, nearby buildings, and its cross-date overlaps. The second layer repeats this using the first layer's learned descriptions. The final head makes one binary decision score for the region.

The model is R-GCN-style rather than an exact claim to reproduce every design choice of the original R-GCN paper. The project adds learned edge-attribute contributions in its messages. The conceptual primary reference is [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103).

### 20.2 Output meaning

Training uses logits with binary cross-entropy. Inference obtains `p = sigmoid(logit)`. A selected threshold converts `p` into a binary prediction. The probability is support learned from the supplied targets; calibration against independently verified physical change has not been established.

A node score does not identify exactly which pixels changed inside a large land region. Whole-node coloring can visually overstate the footprint of a partial conversion. Geometry-level image-evidence events and node-level GNN predictions should be interpreted separately.

## 21. Original labels, training, and seven-tile evaluation

### 21.1 Original pseudo-labels

The original GNN reads local rule-layer tables and transfers their labels to strip nodes by pixel lineage.

- Later-year positive: `new_building` or `extension` with confidence string `high`.
- Earlier-year positive: `built_over`.
- Negative categories: persistent, segmentation difference, unexplained/no evidence, or unchanged.
- Other outcomes: masked/unlabeled.
- Later non-buildings and earlier buildings are masked as inapplicable roles.

Pixel lineage votes determine strip labels. Positive votes covering at least half the pixels take precedence. Otherwise at least half masked votes yield masking; otherwise the node is negative. The tile contributing most pixels determines the node's evaluation tile.

The underlying frozen earlier-land conversion rule differs from the new adapter: it can tag a land node when converted area is at least 100 m² **or** at least 30% of the node, after evidence-supported conversion-pair filtering. This does not mean the whole tagged land polygon was converted.

### 21.2 Optimization

| Setting | Original implementation |
|---|---|
| Device | CPU, deterministic algorithms |
| Torch threads | 4 |
| Seed | 0, reset for each fold |
| Batch | Full graph; loss only on training-mask labels |
| Optimizer | Adam |
| Learning rate | 0.005 |
| Weight decay | 0.0005 |
| Loss | Binary cross-entropy with logits |
| Positive weight | Training negative count / training positive count |
| Training budget | 300 epochs |
| Validation frequency | Every 5 epochs |
| Best state | Highest validation AP, or negative training loss when validation has no positives |
| Threshold | F1-optimal validation threshold among unique probabilities rounded to 3 decimals; fallback 0.5 |

There is no patience-based early termination in this loop. It runs all 300 epochs, evaluates 60 times, and restores the best in-memory state. Existing prose calling this “early stopping” should be qualified as best-state selection.

Weights are not written to a reusable checkpoint. Reproduction retrains the models under the recorded environment or explicitly reuses preserved predictions for figure generation.

### 21.3 Seven folds and three ablations

For fold `k`:

- Tile `k` supplies test labels.
- Tile `(k+1) mod 7` supplies validation labels.
- The remaining five tiles supply training labels.

The study pools held-out predictions from all seven test tiles. It compares:

1. MLP: same self-projection/head architecture with no graph relations.
2. Spatial GNN: touching and nearby-building relations only.
3. Full ST-GNN: spatial and temporal relations.

Three models × seven folds means 21 fits. Fold-selected thresholds are not separately recorded in the frozen summary/history files; a thesis must not invent their exact values.

### 21.4 Important evaluation qualification: transductive, pseudo-labeled

This is a **transductive node-label evaluation**. The full strip graph, including held-out node features and edges, remains available for message passing. Continuous-node and relation normalization are based on the full graph before fold label separation. The training loss uses only training labels; the held-out labels are not optimized directly.

Consequently, it is not an inductive test on completely unseen isolated graphs or an independent new geographic area. Also, its reference labels come from the rule layer. The metrics measure held-out agreement with those pseudo-labels, not independently established real-world change accuracy.

The small visual audit is a separate evidence source. Audit verdicts are loaded after prediction; they are not a statistically representative independent ground-truth dataset for the full study.

## 22. Current new-input GNN adapter and why scores can be unavailable

`adapter.py` reconstructs the original node features and calls the original `relations()` and `train_fold()`. It does not replace the architecture.

Current mapping safeguards include:

- Canonical classes become the original class-feature order.
- The later-year indicator is based on actual input order.
- `context_near` is excluded from the retained model.
- If a temporal edge lacks `overlap_score`, it uses `sqrt(max(0, fraction_source*fraction_target))`. An explicitly stored zero stays zero.
- Known rule-based bare-ground/water nodes use valid rule support for the original confidence feature. Missing/invalid support falls back to existing semantic confidence. Source probabilities and displayed scores are not inflated.
- Old accepted correspondence/split/merge links reach the appropriate original primary relation channels.

The current target adapter is deliberately conservative:

1. Applicable roles are later buildings and earlier classes 2–8.
2. Observed fraction must be at least 0.95.
3. Known counterpart fraction must be at least 0.95.
4. Raw semantic confidence must be at least 0.60.
5. The current catalogue must provide eligible supervision.
6. Later buildings use eligible catalogue labels 0/1.
7. Earlier land is positive if at least half its polygon intersects the union of eligible later catalogue-positive building polygons. An eligible unchanged catalogue label can supply a negative. Ambiguous cases remain masked.

Both positive and negative targets are required. If either is absent, the result is `insufficient_data` and probabilities remain unavailable. Unknown does not mean zero probability and must not be filled with a made-up score.

For a new-input fit, training and validation masks are the same current labelled nodes. The status is explicitly `training_only`. This is a way to run the retained architecture on a new area, not an independent evaluation. Only eligible labelled nodes receive exported probabilities; unsupported nodes remain unscored.

Default epochs are 300, with a minimum of five because state selection occurs every five epochs. Hidden width remains 64. `--gnn-hidden` is a compatibility option rather than an active architecture selector. `--gnn-checkpoint` is rejected because the original model has no saved checkpoint and the former experimental checkpoints were retired.

An `insufficient_data` comparison can still contain hundreds of all-class image-evidence candidates. These candidates do not become GNN predictions just because the selected pipeline option was `gnn`.

## 23. Verified preserved results and what they support

The following numbers come from the saved original summary/predictions and match `reference/output_reference.json`. They were not newly produced while writing this document.

### 23.1 Cohort and imbalance

| Quantity | Count |
|---|---:|
| All frozen graph nodes | 7,020 |
| Labelled nodes | 3,851 |
| Positive nodes | 125 |
| Negative nodes | 3,726 |
| Masked nodes | 3,169 |
| Earlier positive land nodes | 66 |
| Later positive building nodes | 59 |

Positive labels are about 3.25% of labelled nodes. Class imbalance is substantial; an apparently high overall accuracy could be misleading. The project reports AP, ROC-AUC, precision, recall, F1, and confusion counts instead of treating accuracy alone as sufficient.

### 23.2 Pooled held-out pseudo-label agreement

| Model | AP | ROC-AUC | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP | 0.084 | 0.734 | 0.161 | 0.280 | 0.204 | 35 | 183 | 90 |
| Spatial GNN | 0.100 | 0.710 | 0.148 | 0.240 | 0.183 | 30 | 173 | 95 |
| Full ST-GNN | **0.530** | **0.906** | **0.507** | **0.600** | **0.549** | **75** | **73** | **50** |

The full graph model performs better than the two ablations under this particular frozen protocol. This supports discussing temporal relationships as useful for pseudo-label agreement in this study. It does not establish universal improvement, calibrated confidence, or independent geographic generalization.

### 23.3 Per-year agreement

| Model/year | AP | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| MLP, earlier 2013 | 0.060 | 0.048 | 0.030 | 0.037 |
| MLP, later 2023 | 0.114 | 0.188 | 0.559 | 0.281 |
| Spatial GNN, earlier | 0.101 | 0.125 | 0.197 | 0.153 |
| Spatial GNN, later | 0.117 | 0.172 | 0.288 | 0.215 |
| Full ST-GNN, earlier | 0.266 | 0.346 | 0.424 | 0.381 |
| Full ST-GNN, later | 0.787 | 0.701 | 0.797 | 0.746 |

Full-GNN yearly confusion counts are earlier 28 TP / 53 FP / 38 FN and later 47 TP / 20 FP / 12 FN. The earlier built-over task is harder under these labels and objects than the later-building task. A ten-year gap alone does not explain every failure; object extent, supervision, segmentation, and graph context can contribute.

### 23.4 Visual audit agreement

| Audit verdict | Items | Detected by full GNN | Median model probability |
|---|---:|---:|---:|
| Real | 43 | 33 | 0.998 |
| Questionable | 13 | 10 | 0.996 |
| Unclear | 2 | 0 | 0.080 |

A high score on “questionable” cases demonstrates why model probability is not a guarantee of physical truth. The real-category detection count 33/43 should be described as audit agreement/coverage, not a complete independent precision or recall estimate for all changes.

### 23.5 Metrics in simple terms

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
F1        = 2*precision*recall / (precision + recall)
```

AP summarizes positive ranking; the implementation averages precision at positive ranks. ROC-AUC uses a rank-based calculation. These metrics need a clearly defined reference label source. Here, that source is the rule pseudo-labels under held-out tile masks.

### 23.6 Whole-land-node visualization caveat

Original figures can color a complete earlier land node even if only part was converted. An audit of 71 local per-tile `built_over` rows found total converted area divided by total tagged object area of about 7.58%. The mean individual object fraction was about 28.08%, with median 20.42%.

These are different statistics. Existing dashboard wording calling 7.6% an “average object” fraction is imprecise. Use the aggregate area-ratio interpretation. Do not use full colored land-node area as the amount of new building footprint.

## 24. Scores, statuses, filtering, and pagination

| Value | Where it comes from | What it does not establish |
|---|---|---|
| SAM predicted IoU/stability | SAM mask diagnostics | Land-cover correctness or temporal-change confidence |
| Parcel instance confidence | Delineate Anything detector | Final class confidence or physical change |
| Region semantic confidence | Mean assigned-class support/probability inside a region | Whether a before/after difference is real |
| RGB-rule support | Mean deterministic rule-support raster | A calibrated neural probability |
| Temporal match score | Geometry/context overlap heuristics | Physical object identity with certainty |
| GNN probability | Sigmoid output of the retained binary model | Independent accuracy or calibrated physical-change probability |

For new segmentation, a node score is the mean assigned-class probability over its pixels. Old adapted confidence uses the old fused assigned-label support described earlier. For an event involving multiple before/after nodes, the review segmentation score is the **lowest available** participating-node score. Before and after minima are also displayed separately.

The minimum is conservative. A high-confidence new region linked to a weak earlier region can therefore have a low event segmentation score. This is one reason crop scores can look much weaker than individual building segmentation scores. There is no artificial rescaling to make them look stronger.

When a catalogue event links several eligible GNN-scored nodes/tasks, its displayed `gnn_score` is the **highest available valid probability** among those links. This differs from the minimum used for segmentation support. It does not create probabilities for unsupported nodes or turn an all-class event into a learned multiclass prediction.

`filter_rows` sorts the entire filtered list by segmentation score descending before splitting it into pages. Missing, non-finite, and out-of-range scores stay unavailable and sort last. Ties retain stable input order. The viewer uses 12 changes per page and resets/clamps pagination when selection changes. Downloaded page grids use the same order and score descriptions.

Main statuses include:

| Status | Interpretation |
|---|---|
| `candidate` | A reviewable difference, without sufficient proof of physical change |
| `season_sensitive` | A heuristic seasonal-ambiguity flag |
| `reliable` | Passed strong image-evidence/review gates; still not independent ground truth |
| `predicted` | A GNN-supported node/footprint prediction |
| `training_only` | New-input model fitted/scored on the same labelled area |
| `insufficient_data` | Required positive/negative training examples are unavailable |
| `unassessed` / ineligible | Missing coverage, unknown semantics, weak support, or another exclusion |

At this snapshot, **Focus on major changes is enabled by default**. It hides thin/small-interior footprints and vegetation-subclass swaps; building candidates require substantial whole-building change without strong same-class persistence. Non-building patches below 10% of participating parent-region area on either date are also hidden. This is a display policy, not deletion of scientific events.

The exact focus gates in `runtime/preview.py` are:

- Erode the event footprint by twice the native pixel size; exclude it if less than 20% of its area remains, or its area is zero.
- Exclude conversions entirely among tree, grass, and crop classes as vegetation-subclass ambiguity.
- After the footprint gate, retain an original-GNN `predicted` land-to-building conversion linked to `built_over`; its earlier-land prediction may have no later-building node ID.
- For other building-related events, require at least 60% of a participating building footprint to intersect the event and less than 15% same-class building persistence. Persistence sums the relevant overlap fractions of all building counterparts, capped at one.
- For non-building events with known parent geometry, require at least 10% changed overlap relative to the summed participating parent area on every available date side. This can suppress a small patch inside a large landscape region.

To inspect every candidate, include weak/seasonal differences, turn major focus off, select all statuses/classes, set area and segmentation minima to zero, and turn the GNN-score filter off. Score zero alone does not override another active filter.

The major filter is a heuristic focus. A small real road or pond change inside a large parent region can be hidden by it. It should not be used as ground-truth filtering or as an accuracy evaluation rule.

**Source:** `runtime/preview.py`, `dashboard/change_review.py`, `runtime/reports/changes.py`. The cached result loader explicitly binds the display profile so presentation changes do not silently reuse obsolete rows.

## 25. Dashboard pages and their purpose

The application entry point is `thesis_dashboard.py`. It sets `TF_ENABLE_ONEDNN_OPTS=0` before importing model libraries and mounts Streamlit pages through `st.navigation`.

| Page/module | Purpose |
|---|---|
| `dashboard/pages/overview.py` | Thesis overview, study scope, pipeline narrative |
| `imagery.py` | Source imagery and geographic/visual context |
| `segmentation.py` | Multi-class region segmentation products |
| `graphs.py` | Spatial region graphs and their interpretation |
| `temporal.py` | Cross-year correspondence and temporal graph discussion |
| `gnn.py` | Retained original building ST-GNN and preserved evidence |
| `changes.py` | Saved study change visualization and its interpretation |
| `evaluation.py` | Metrics with their evidence type and limitations |
| `pipelines.py` | Run pipelines & results: choose methods, upload/path inputs, view verified comparisons |
| `reproduce.py` | Reproduction/data information |

The selectable workspace has Changes, Segmentation, Graphs, Matching, and GNN training tabs. `dashboard/data.py` loads saved data and evidence; `dashboard/ui.py` supplies shared UI helpers. `dashboard/run_results.py` distinguishes verified completed reports from partial historical exports.

Paired crops use the same location and scale on both dates, native original RGB where available, limited enlargement, local context, and a yellow footprint outline. They are not artificially sharpened. A large colored parent-node prediction should not be confused with a localized reliable change polygon.

UI convenience and scientific computation are separate. Rendering or filter changes can alter what is visible without changing predictions or scientific arrays. Missing report manifests indicate incomplete/unverified rendering rather than necessarily missing scientific computation.

## 26. Repository architecture and artifact layout

### 26.1 Main code areas

```text
thesis_dashboard.py                  Streamlit entry point
run.bat                             Local Windows launch helper
Readme.md                           Project entry documentation
config.yaml                         Original-stage configuration
requirements.txt                    Dependency specification

src/
  cli.py                            Command parser and dispatch
  pipeline_select.py                Selectable-stage orchestration
  pipeline_state.py                 Fingerprints, stage dependency/reuse rules
  folder_pipeline.py                Original annual/folder/image workflow
  io/                               Raster reading, resampling and tiling
  stage2/                           SAM/OEM segmentation and fusion
  landscape/                        Legacy landscape phase/backend helpers
  refinement/                       Stage 2R topology, evidence, review and replay
  stage3/                           Exact tile adjacency graphs
  stage4/                           Overlap reconciliation and combined graphs
  stage5/                           Coordinate conversion, matching and change events

pipelines/semantic_temporal/
  runtime/                          General current-input processing
  pipeline/                         Preserved original experiment algorithms
  run_pipeline.py                   Native general runner entry
  run_experiment.py                 Frozen full-experiment reproduction entry
  reference/                        Historical full-experiment references
  model_assets.json                 Pinned segmentation-asset manifest

pipelines/legacy_building_stgnn/
  adapter.py                        Current-input mapping into the retained GNN
  runner.py                         Frozen-input staging and reference verification
  original/                         Hash-protected original code
  reference/                        Code/input/output reference records

dashboard/                          Pages, loaders, controls and crop review
scripts/                            Downloads, preparation and test utilities
tests/                              Unit, artifact, integration and historical regression checks
third_party/                        Vendored SAM 2 and OEM runtime code
docs/                               Methodology, historical records and thesis discussion
```

`src/landscape` and older APIs remain in the repository. Their presence should not be interpreted as an additional selectable learned change model. The authoritative selectable flow is the orchestrator plus `runtime/runner.py`.

### 26.2 Common contracts

`runtime/contracts.py` defines immutable dataclasses:

| Contract | Main information |
|---|---|
| `GridSpec` | CRS, six affine coefficients, units, optional physical pixel size |
| `TileInput` | Key/year/image, metadata, source ID, source offset, shape and valid extent |
| `YearInput` | Year, tile list, optional original source raster |
| `RunContext` | Output, input years, resolved options, model provenance, input fingerprint |
| `SegmentationArtifact` | Year/tile, grid/shape, paths, lineage |
| `GraphArtifact` | Year/method, grid, paths |
| `TemporalArtifact` | Two years, directory, paths |

Canonical runtime schema is `satellitegraph.semantic-temporal.v1`. The all-class catalogue schema is `satellitegraph.all-class-change.v1`. Schema names describe data contracts, not model versions or accuracy.

### 26.3 Typical new comparison folder

```text
artifacts/image_comparisons/<run_id>/
  inputs/                           Uploaded before/after files, if used
  run.json                          Selected methods, options, statuses, timestamps, results
  context.json                      Resolved inputs, coordinates and model provenance
  prepared/
    context/                        Sealed preparation snapshot and manifest
    registration_controls.json      Landmark/fit evidence when available
    harmonization/                  Optional model-input preprocessing and statistics
  segmentation/<year>/<tile_key>/   Canonical segmentation artifacts
  graphs/<method>/<year>/<tile>/    Tile graph artifacts
  annual/<method>/<year>/           Annual graph, ownership and provenance
  temporal/
    artifact.json
    graph.json                      Nodes, spatial/temporal edges and components
    graph_tensors.npz
    feature_spec.json
    training_targets.csv
    classical_changes.json          Current canonical published catalogue
    legacy_classical_changes.json   Preserved old-engine result
    legacy_training_targets.csv
    class_transition_map.tif
    transition_matrix.json
    change_policy.json
    change_graph.json
    gnn/                            Current-input predictions/history/report when selected
  results/
    index.json                      Main entry point for rendered products
    segmentation/                   Segmentation PNGs and index
    graphs/                         Graph PNGs and index
    matching/                       Registration/matching illustrations and tables
    changes/                        Paired crop pages and index
    rendering/<stage>/manifest.json Verified render dependency/output records
    change_report.png
    major_changes.png
    gnn_predictions.png
    gnn_probabilities.png
  previews/                         Additional run previews where produced
  validation.json                   Run checks
  console.log                       Streamlit-launched processing log
```

Actual optional filenames/paths depend on the method and stage completion. The run ledger and artifact dictionaries are authoritative. Generated artifacts, data, and model weights are generally Git-ignored and do not appear in a normal GitHub clone.

### 26.4 Original and preserved output locations

Original stages normally write under `artifacts/stage2`, `stage3`, `stage4`, and `stage5`, controlled by configuration. The preserved building study uses `data/legacy_building_stgnn/` for a verified input bundle and `artifacts/legacy_building_stgnn/<run>/` for separate reproduction outputs. General uploaded-image runs do not overwrite that study.

## 27. Configuration, execution, and reproducibility

### 27.1 Environment and local model assets

The dependency file includes PyTorch/torchvision, NumPy, Pandas, OpenCV, Rasterio, Affine, Shapely, SciPy, Pillow, PyYAML, Transformers, Ultralytics, Safetensors, NetworkX, Streamlit, Plotly, pytest, and vendored old-model runtime dependencies.

General processing uses Rasterio and does not require `osgeo` Python bindings. The original frozen reproduction code imports GDAL bindings in its original export/visualization stack, so its recorded environment is a separate concern.

Segmentation inference loads local assets; it does not implicitly download missing checkpoints. `scripts/download_pipeline_models.py` is an explicit acquisition tool using pinned files, hashes, temporary files and size checks. It refuses to silently overwrite a differing existing asset.

Seven required segmentation-asset files are recorded: one SAM checkpoint, FasterSeg architecture and weights, one Delineate Anything checkpoint, and three SegFormer files. This does not mean seven separate neural networks.

Pinned new asset revisions are:

```text
MykolaL/DelineateAnything
46f98c05ece1d57047e36952a8b440583de38f0d

boehnen/satlens-segformer
19cdd879e535d76e7fcbbd1e8b54dac9c1da90d6
```

The manifest records old local hashes and honestly notes the absence of publisher checksums for the reused old assets. A recorded hash proves file identity relative to that manifest, not independent accuracy, legal permission for every input, or provenance beyond the recorded source.

### 27.2 Commands for the thesis reader

These are usage examples, not a claim that they were executed while preparing this file. Replace image paths with the actual inputs.

```powershell
# Start the dashboard from the repository root.
& ..\.venv\Scripts\python.exe -m streamlit run thesis_dashboard.py

# New segmentation + new graph + retained building GNN.
python -m src pipeline run --segmentation new --graph new --change gnn --images before.png after.png --years 2013 2023 --output artifacts/runs/example --device cpu

# Old segmentation + adjacency graph + old polygon change rules.
python -m src pipeline run --segmentation old --graph old --change classical --images before.png after.png --years 2013 2023 --output artifacts/runs/old_example --device cpu

# Run stages separately; later stages inherit saved input options.
python -m src pipeline segment --method new --images before.png after.png --years 2013 2023 --output artifacts/runs/staged --device cpu
python -m src pipeline graph --method new --output artifacts/runs/staged
python -m src pipeline change --method classical --output artifacts/runs/staged

# Resume the same verified run, with its saved input paths.
python -m src pipeline run --segmentation new --graph new --change gnn --output artifacts/runs/example --resume

# Verify an existing preserved building-GNN reproduction.
python -m src legacy-building-gnn verify --output artifacts/legacy_building_stgnn/full_run_2026-10-11
```

Other original APIs include `stage1 tile`, Stage 2 phase commands, refinement review/apply/batch commands, `stage3 run`, `stage4 combine`, `stage5 build --series`, `image run/compare`, and `folder run/compare`. See `src/cli.py` and the stage documents for exact required inputs.

`run.bat` is a local Windows convenience launcher. Its hard-coded local project/venv path is machine-specific; it is not a portable experiment specification.

### 27.3 Fingerprints and stage dependencies

`src/pipeline_state.py` hashes inputs, settings, model files, relevant configuration, and stage profiles. `run.json` records stage selection, start/finish state, scientific and render status, artifacts, and source commit.

Changed input imagery, segmentation assets, placement, or preparation settings invalidate dependent science. Changing a method rebuilds dependent stages. Change-policy/review-label changes belong to change processing. Presentation/profile changes can require rerendering without retraining or changing scientific arrays.

Completed artifacts and prepared context are hash-verified before reuse. Interrupted new segmentation can reuse verified completed tiles. Moved run paths can be relocated with verification; external source/model files still need to be available. A copied data-only export is not automatically a complete rerunnable experiment.

Current profile constants include graph v2, report v7, and original-thesis/current-change adapter v2. A version string indicates a cache/dependency policy, not a performance result.

### 27.4 Scientific completion and rendering completion

A stage can finish scientific computation and fail during visualization or manifest binding. The ledger distinguishes these states and retains scientific outputs. A missing `results/rendering/change/manifest.json` means the dashboard cannot verify a completed rendered comparison. It should not be interpreted as a confirmed GNN failure or silently ignored.

Explicit memory budgets protect large images: prepared segmentation output defaults to 16 million pixels per tile, comparison mosaics to 4 million pixels, and complete report canvas to 128 million pixels through the CLI; the dashboard requests a 256-million report budget. These are resource controls, not universal scalability guarantees. Full probability tensors and model inference still consume substantial RAM and time.

### 27.5 Preservation and verification evidence

Original source files are checked against a code manifest. The retained `gnn_change.py` SHA-256 at this snapshot is:

```text
19cff8d01ab5dc2c5c5692f90fdb986784c9a0d26717ead72c6a4d821e59c5d7
```

During this documentation audit, read-only checks found no protected original-code differences, verified 53 frozen bundle files, and passed 44 checks against the existing preserved reproduction outputs. These are integrity/reproduction checks on saved material, not newly trained accuracy experiments.

The previous code change also passed focused synthetic tests covering scores, filters, pagination, rule provenance, adapter mappings, graph contracts, and source integrity. Other historical full-suite counts in older documents belong to their dates and code versions. Unit-test success does not prove segmentation or physical-change accuracy.

## 28. Practical failure cases and known limitations

### 28.1 Identical before/after inputs

A concrete local diagnostic at this snapshot is run `d18ed7bb290c`. Its saved before and after PNG files are byte-identical and pixel-identical, with SHA-256:

```text
dc1d581d61db6df7972c801f7f7277473aaa7e3ed70b96b6164a0929a812be5b
```

Its catalogue contains two seasonal candidates; major-change focus shows one. The earlier correctly differing input pair in `57102c7fbeca` and `8060742522ec` contains 369 candidates, of which 53 pass the current major focus. These counts are diagnostics of saved inputs and filtering, not benchmark accuracy or a new pipeline rerun.

Two identical observations cannot provide evidence of temporal physical change. Small remaining processing differences should be reviewed as artifacts/uncertainty. The precise cause of those two residual candidates was not further isolated during this diagnostic. No duplicate-upload blocking or default-major-focus change was implemented after that diagnosis before commit `db1bbfd`; do not describe the proposed UI safeguard as completed work.

### 28.2 Small landscape differences

Large homogeneous land regions can be split/relabelled differently, producing a small candidate patch with little physical meaning. Colour ambiguity, seasonal vegetation, shadows, upsampling, and local segmentation boundaries can contribute. The major-focus filter and semantic-score slider help review but do not repair every upstream classification error.

### 28.3 Missing GNN probabilities

Several local general comparisons report `insufficient_data`, including comparisons with hundreds of visible all-class candidates. A catalogue difference does not automatically satisfy the required changed/unchanged building-target gates. Coverage, counterpart confidence, and absent stable examples can all leave no training targets. Fabricating probabilities or relaxing unknown masks just to show scores would invalidate interpretation.

### 28.4 Registration and viewing geometry

Low texture, repetitive roofs, seasonal appearance, different resolutions, and weak landmark distribution can make registration fail or leave local residual errors. A global affine cannot fully model terrain-induced parallax or all sensor-view effects. Roof compensation is limited and current-evidence-based. Fitted landmark residuals are not independent error measurements.

### 28.5 Evaluation and generalization limits

- Only one preserved geographic strip and two study dates.
- Strongly imbalanced pseudo-labels and one recorded random seed.
- Transductive full-graph evaluation rather than isolated unseen-graph inference.
- Rule-target dependence and a small non-representative visual audit.
- Uncalibrated semantic and GNN probabilities.
- Segmentation regions that do not always correspond to real individual objects.
- Different graph selections also change temporal matching in the general pipeline.
- Old segmentation comparisons can depend on new semantic evidence recovery.
- Displayed counts depend on active filters; visible cards are not all detections.
- Whole-node built-over coloring can overstate localized converted area.
- Source sensor, licensing and full acquisition metadata still require authoritative documentation.

These limitations belong in the thesis discussion. They do not negate the software or the recorded experiment; they define what the evidence supports.

## 29. Suggested thesis structure and research questions

### 29.1 Possible research questions

1. How do alternative segmentation methods affect region geometry, semantic uncertainty, and downstream candidate changes?
2. What useful context is added by spatial proximity and temporal region relationships beyond adjacency alone?
3. Under the preserved protocol, how does the full ST-GNN compare with the MLP and spatial-only GNN in pseudo-label agreement?
4. How do registration, seasonal ambiguity, segmentation confidence, and region extent affect false positives and review usefulness?
5. How can explicit masks, lineage, ownership, manifests, and clear evidence labels make the workflow reproducible and interpretable?

The frozen ablation table directly addresses part of question 3. Other questions need a designed evaluation; the existence of code does not mean those experiments have already been performed.

### 29.2 Chapter outline

| Chapter | Recommended content |
|---|---|
| 1. Introduction | Study problem, motivation, region/graph perspective, research questions, careful contribution statement |
| 2. Background and related work | Remote-sensing change detection, object-based analysis, semantic/instance segmentation, SAM/OEM/SegFormer/parcel models, relational GNNs, registration and uncertainty |
| 3. Data and study area | Authoritative imagery metadata, dates, CRS/GSD, tile layout, observation masks, preprocessing and data permissions |
| 4. Methodology | Old/new segmentation, class mappings, spatial graphs, exact-once mosaics, temporal correspondence, evidence catalogue, retained GNN architecture and target definitions |
| 5. Implementation and reproducibility | Module architecture, assets, contracts, fingerprints, manifests, dashboard, resource/environment settings |
| 6. Experiments and results | Frozen cohort/protocol/ablations, clearly labelled saved metrics, any separately completed new experiments, representative successes/failures |
| 7. Discussion | Pseudo-label dependence, transductive evaluation, seasonal and registration ambiguity, partial-land visualization, practical usability and threats to validity |
| 8. Conclusion and future work | Supported conclusions, practical contribution, and proposed independent validation/calibration/scalability work |

### 29.3 Figures that would help

- Full pipeline diagram showing model, deterministic geometry, evidence, and visualization boundaries.
- Paired source images and aligned views with observation masks.
- Segmentation examples comparing old and new outputs at the same location.
- Old adjacency versus new proximity/context edges on unchanged region geometry, where an isolated comparison is intended.
- Annual overlap ownership and lineage illustration.
- Temporal one-to-one, split, merge, and unmatched examples.
- Exact 38 → 64 → 64 → 1 GNN diagram with six relation channels.
- Frozen ablation metrics with “held-out pseudo-label agreement” in the caption.
- Failure examples, including partial built-over land, shadow/season ambiguity, and input/coverage mistakes.

An AI-generated illustrative image cannot replace an actual result figure. Captions must identify whether a figure is a schematic, preserved experiment result, or new-input example.

### 29.4 Future evaluation design, not completed results

For independent physical-change evaluation, create a reference set with a documented annotation protocol and blinded or independent review. Include unchanged areas as well as changed candidates, uncertain cases, all relevant classes, and locations not selected only because the algorithm found them.

Separate training, validation, and test areas spatially enough to address overlap and neighbourhood leakage. If claiming inductive generalization, isolate held-out graphs and fit normalization only on training data. Keep the preserved original experiment unchanged and name any improved protocol as a new experiment.

Measure segmentation quality separately using class IoU/Dice or boundary/object metrics on independent labels. Measure temporal matches using verified correspondences. Measure change precision/recall with a stated unit: node, event polygon, object, or pixel. Do not mix those units in one confusion matrix.

For stage comparisons, hold inputs, preparation, scale, evaluation labels, and downstream settings fixed where possible. Explicitly report the graph/matcher and old-segmentation/evidence coupling. Compare errors and confidence distributions, not just the number of visible cards.

Probability calibration, uncertainty intervals, multiple seeds, additional areas/sensors/dates, and sensitivity to thresholds are useful future analyses. They are not already established by this repository's saved scores.

## 30. Information still needed from the thesis author

ChatGPT should ask for or mark these as missing rather than inventing them:

| Needed fact | Why it matters |
|---|---|
| Final approved thesis title and research questions | Keeps claims aligned with the assessed contribution |
| University/template/citation style and length requirements | Controls thesis formatting and chapter structure |
| Exact imagery provider, sensor, acquisition date/month and processing level | Required data provenance; filenames/year fields are insufficient |
| Original spectral bands, bit depth, GSD, CRS, orthorectification and resampling history | Determines available information and measurement validity |
| Data/model licenses and permitted use | Supports the data/software statement |
| Independent annotation protocol and available reference labels | Determines which accuracy claims can be made |
| Student, contributor and supervisor roles | Supports honest individual contribution attribution |
| Which saved run is the final reported experiment | Prevents mixing old/new/retired outputs |
| Exact final environment and runtime measurements | Historical environment records may differ from the current machine |
| Which additional experiments have actually finished | Separates completed results from proposed work |

## 31. Short discussion answers for thesis preparation

**Why combine semantic and instance methods?** Semantic segmentation names the class of pixels. Instance/parcel geometry proposes boundaries. Their combination produces labelled regions suitable for a graph, but reconciliation is imperfect.

**Why use a graph instead of only pixel subtraction?** The graph records shape, neighbours, and temporal relationships. It can represent meaningful context and reduce reliance on isolated pixel differences. This benefit must be demonstrated under a stated evaluation.

**What exactly improved in the new graph?** It adds nearby-building and general region-context relations, retains richer region/edge summaries, and uses explicit connected-region handling. The original GNN consumes touching/near-building relations and temporal channels; it excludes the extra context-near relation.

**Is the GNN downloaded?** Its code was integrated as a contributor-supplied project implementation. It loads no pretrained GNN weights and saves no reusable trained checkpoint. The segmentation checkpoints are downloaded/external assets.

**What is temporal about the GNN?** Its graph contains observations from both dates and directed cross-date relations. It uses relation-specific message passing. It has no LSTM or recurrent time-series block.

**Does it predict all eight class changes?** The catalogue supports eight classes, but the learned model is the binary later-building/earlier-built-over task. The sixteen gain/loss task names are catalogue bookkeeping.

**Why can a score be 0.99 on a questionable change?** The network learned agreement with its supervision, and its probabilities are not independently calibrated to physical truth. A region-level prediction may also highlight a larger footprint than the localized change.

**Why is the segmentation score low?** It is raw assigned-class support, and the event uses the lowest available linked-node value. Colour rules can assign a class despite low semantic probability. A weak earlier side can limit an otherwise strong later side.

**Why are GNN scores unavailable?** The comparison lacks eligible positive and negative examples or the node is masked. Unavailable is an honest result and is not a zero score.

**Why can the old pipeline need the new semantic model?** The old fused artifacts lack complete alternative-class probabilities, so the shared comparison can recover full evidence using SegFormer. This coupling must be reported.

**Do split and merge edges prove construction or demolition?** No. They describe region correspondence/segmentation structure and may reflect a segmentation difference.

**Does identical before/after imagery validate temporal detection?** It is a useful negative-control input, but it contains no real temporal observation difference. Residual candidates are not evidence of real construction.

**What is the strongest supported GNN result?** The preserved full ST-GNN has AP 0.530 and F1 0.549 under seven-fold transductive held-out pseudo-label agreement, outperforming its two recorded ablations. That is narrower than independent real-world accuracy.

## 32. Arabic explanation and glossary

### 32.1 الفكرة الأساسية

مشروع **SatelliteGraph** يقارن صور أقمار صناعية من تاريخين. أولاً يقسم كل صورة إلى مناطق ويعطي كل منطقة فئة، مثل مبنى أو طريق أو أشجار. بعد ذلك يحول المناطق إلى رسم بياني: كل منطقة تصبح عقدة، والعلاقات بينها تصبح حواف. ثم يربط مناطق التاريخ الأول بمناطق التاريخ الثاني ويعرض فروقاً تحتاج إلى المراجعة.

هناك فرق مهم بين **اختلاف التصنيف** و**التغير الحقيقي على الأرض**. قد يتغير لون النبات بسبب الموسم، أو يظهر ظل مختلف، أو تكون الصورتان غير متطابقتين مكانياً. لذلك ليست كل منطقة مختلفة مبنى جديداً أو تغيراً مؤكداً.

### 32.2 مرحلتا التقسيم

في الطريقة القديمة، **SAM 2.1** يقترح حدود المناطق دون أن يعرف اسم فئتها. نموذج **OpenEarthMap FasterSeg** يعطي الفئة الدلالية. البرنامج يدمج الحدود مع التصنيف ثم ينفذ تنقيحاً محافظاً مع الاحتفاظ بالمصدر والتاريخ.

في الطريقة الجديدة، **SegFormer** يعطي احتمالات الفئات لكل بكسل، و**Delineate Anything v2** يعطي حدود مناطق أو قطع أرض. توجد أيضاً قواعد للألوان والشكل، خصوصاً للماء والتربة المكشوفة، ثم تقسيم وتنظيف ودمج مشروط للمناطق. تكبير الصورة داخل المعالجة لا يضيف تفاصيل حقيقية لم تكن موجودة في الصورة الأصلية.

### 32.3 الرسم البياني القديم والجديد

الرسم القديم يربط المناطق التي تشترك في ضلع أو حد. الرسم الجديد يحتفظ بهذه العلاقات ويضيف علاقات قرب بين المباني وبين مناطق معروفة أخرى. الهدف هو توفير سياق أوسع، مثل ترتيب المباني حول طريق. لكن وجود علاقات أكثر لا يثبت وحده أن الدقة أصبحت أعلى في كل الصور.

عند وجود بلاطات متداخلة، يجب ألا نحسب نفس البكسل مرتين. البرنامج يحدد ملكية واحدة لكل بكسل متداخل، ثم يعيد بناء المناطق والعلاقات من البكسلات التي يملكها كل مصدر.

### 32.4 ماذا يفعل نموذج GNN الموجود حالياً؟

نموذج GNN الحالي **ثنائي** وليس نموذجاً مستقلاً لكل فئات الغطاء الأرضي. يقيّم المباني في التاريخ الأحدث: هل هي جديدة أو ممتدة؟ ويقيّم الأرض غير المبنية في التاريخ الأقدم: هل أصبحت أرضاً مبنياً عليها؟

مدخلاته ليست الصورتين مباشرة. مدخلاته صفات المناطق والعلاقات بينها: 38 صفة لكل عقدة، وستة أنواع من العلاقات، وصفات للحواف. يحتوي على طبقتين لتمرير الرسائل، كل منهما بعرض 64، ثم رأس يعطي قيمة واحدة لكل عقدة. تستخدم دالة sigmoid لتحويلها إلى احتمال نموذجي.

لا توجد أوزان GNN جاهزة تم تنزيلها من الإنترنت. يوجد كود النموذج الأصلي الذي تم دمجه في المشروع بواسطة المساهمة المذكورة في Git. أما نماذج التقسيم فلها أوزان خارجية. نموذج GNN الأصلي يتدرب ولا يحفظ ملف أوزان نهائياً قابلاً لإعادة الاستخدام.

### 32.5 كيف نفهم النتائج والدرجات؟

درجة التقسيم تعبّر عن دعم النموذج للفئة داخل المنطقة. درجة التغير المعروضة للتقسيم تأخذ أقل درجة متاحة بين المناطق المرتبطة قبل التغير وبعده. لذلك قد تكون منخفضة بسبب أحد الطرفين. قواعد اللون قد تختار تربة أو ماء حتى عندما يكون احتمال هذه الفئة من SegFormer ضعيفاً.

درجة GNN تعبّر عن دعم النموذج للهدف الذي تدرب عليه، وليست ضماناً أن التغير حقيقي. إذا لم توجد أمثلة مناسبة للتغير وعدم التغير، تظهر الدرجة غير متاحة. لا يجوز اختراع درجة بديلة.

التقييم المحفوظ لسبع بلاطات يقيس الاتفاق مع تسميات ناتجة من قواعد المشروع، مع حجب تسميات بلاطة الاختبار. لكنه يستخدم الرسم البياني الكامل أثناء تمرير الرسائل؛ لذلك هو تقييم **transductive** وليس اختباراً مستقلاً على منطقة جديدة تماماً. النتائج المحفوظة لا تثبت دقة كل تشغيل جديد.

### 32.6 مصطلحات أساسية

| English | العربية | Simple meaning |
|---|---|---|
| Semantic segmentation | التقسيم الدلالي | إعطاء فئة لكل بكسل |
| Instance/region segmentation | تقسيم الكائنات أو المناطق | إعطاء هوية مستقلة لكل منطقة |
| Land cover | الغطاء الأرضي | الفئة المرئية مثل أشجار أو طريق |
| Node | عقدة | منطقة ممثلة في الرسم البياني |
| Edge | حافة أو علاقة | اتصال بين عقدتين |
| Adjacency | التجاور | اشتراك منطقتين في حد |
| Proximity | القرب المكاني | مسافة صغيرة بين منطقتين |
| Temporal correspondence | المطابقة بين التاريخين | علاقة محتملة بين منطقتين عبر الزمن |
| Registration | المحاذاة المكانية للصور | جعل المواقع نفسها متطابقة بين الصورتين |
| CRS | نظام الإحداثيات المرجعي | كيف ترتبط الإحداثيات بموقع جغرافي |
| Affine transform | التحويل الأفيني | تحويل خطي مع إزاحة للإحداثيات |
| GSD / metres per pixel | الدقة الأرضية أو متر لكل بكسل | حجم البكسل على الأرض |
| Valid mask | قناع البيانات المرصودة | تحديد البكسلات التي تحتوي على بيانات فعلية |
| Unknown | فئة غير معروفة | يوجد رصد لكن التصنيف غير مؤكد |
| Nodata | لا توجد بيانات | لا يوجد رصد صالح في هذا الموقع |
| Lineage | نسب المنطقة أو تتبع أصلها | معرفة المنطقة الأصلية التي جاء منها الناتج |
| Ownership | ملكية البكسلات | حساب البكسل المتداخل مرة واحدة |
| Feature | صفة | رقم يصف الشكل أو اللون أو الفئة |
| Message passing | تمرير الرسائل | جمع معلومات من الجيران في الرسم |
| Hidden layer | طبقة مخفية | تمثيل يتعلمه النموذج داخل الشبكة |
| Logit | قيمة قبل sigmoid | ناتج عددي قبل التحويل إلى احتمال |
| Pseudo-label | تسمية آلية أو شبه مرجعية | هدف تدريب مولد من القواعد بدلاً من حقيقة مستقلة |
| Ground truth | مرجع مستقل موثوق | تسميات حقيقية وفق بروتوكول تحقق واضح |
| Training-only | تدريب وتقييم على نفس المدخل | لا يثبت الأداء المستقل |
| Held-out labels | تسميات محجوبة للاختبار | لا تستخدم مباشرة في خسارة التدريب |
| Transductive evaluation | تقييم مع معرفة الرسم الكامل | الصفات والعلاقات موجودة مع حجب بعض التسميات |
| Inductive evaluation | تقييم على رسومات أو مناطق جديدة | اختبار تعميم إلى بيانات لم تدخل الرسم التدريبي |
| Precision | دقة الإيجابيات المتوقعة | كم توقعاً إيجابياً كان صحيحاً وفق المرجع |
| Recall | الاسترجاع | كم إيجابية مرجعية تم اكتشافها |
| F1 | المتوسط التوافقي للدقة والاسترجاع | توازن بين precision وrecall |
| AP | متوسط الدقة عبر ترتيب الدرجات | جودة ترتيب الإيجابيات قبل السلبيات |
| False positive | إيجابية كاذبة | تغير متوقع لا يدعمه المرجع |
| False negative | سلبية كاذبة | تغير مرجعي لم يكتشفه النموذج |
| Calibration | معايرة الاحتمالات | مدى توافق الاحتمال مع نسبة الصحة الفعلية |
| Ablation | تجربة إزالة مكون | معرفة أثر حذف علاقات أو جزء من المنهج |
| Reproducibility | قابلية إعادة الإنتاج | إعادة النتائج تحت شروط موثقة |
| Manifest / SHA-256 | سجل تحقق أو بصمة ملف | التحقق من هوية الملفات وعدم تغيرها |

## 33. Source map and literature starting points

### 33.1 First-party implementation sources

| Topic | Authoritative sources |
|---|---|
| Stage choices and dependencies | `src/pipeline_select.py`, `src/pipeline_state.py`, `runtime/runner.py` |
| Inputs, coordinates and context | `runtime/inputs.py`, `contracts.py`, `context.py`, `registration.py`, `pixel_registration.py` |
| Original tiling | `src/io/tile_image.py`, `src/io/raster_tiling.py` |
| Old segmentation | `src/stage2/pipeline.py`, `phases.py`, `backend.py`, `semantic.py`, `fusion.py`, `classes.py` |
| Old refinement | `src/refinement/pipeline.py`, `topology.py`, `automatic.py`, `unknown_cleanup.py`, `contextual_cleanup.py`, `review.py`, `splits.py`, `validation.py` |
| New models/segmentation | `runtime/models.py`, `segmentation/semantic.py`, `regions.py`, `parcels.py`, `refinement.py`, `__init__.py` |
| Interoperability | `runtime/adapters.py` |
| Graph geometry and adjacency | `src/stage3/geometry.py`, `adjacency.py`, `pipeline.py`, `runtime/graphs.py` |
| Annual reconciliation | `src/stage4/pipeline.py`, `runtime/graphs.py::combine_year_graphs` |
| Old temporal matching/events | `src/stage5/matching.py`, `registration.py`, `coordinates.py`, `refinement.py`, `events.py`, `semantics.py` |
| Current temporal graph | `runtime/temporal.py`, `roof_offsets.py`, `comparison.py` |
| All-class evidence/catalogue | `runtime/all_class_evidence.py`, `change_catalog.py`, `change_contracts.py` |
| Retained GNN | `pipelines/legacy_building_stgnn/original/gnn_change.py`, `original/build_graph.py`, `original/build_mosaic.py` |
| Current GNN adapter | `pipelines/legacy_building_stgnn/adapter.py` |
| Frozen reference verification | `pipelines/legacy_building_stgnn/runner.py`, `reference/code_manifest.json`, `reference/output_reference.json` |
| Reports/review | `runtime/reports/`, `runtime/preview.py`, `dashboard/change_review.py`, `dashboard/pages/pipelines.py` |
| Integrity/validation | `runtime/storage.py`, `visualization.py`, `preflight.py`, `src/stage3/schema.py`, `src/stage4/schema.py`, `src/stage5/schema.py` |
| Asset acquisition | `scripts/download_pipeline_models.py`, `pipelines/semantic_temporal/model_assets.json` |

Where `runtime/...` is abbreviated in this table, it means `pipelines/semantic_temporal/runtime/...`.

Useful explanatory documents include `docs/stage2-segmentation.md`, `stage2-refinement.md`, `stage3-graph-building.md`, `stage4-graph-combination.md`, `stage5-spatiotemporal-graph.md`, `temporal-edge-methodology-for-review.md`, `refinement-teacher-discussion.md`, `pipeline-options.md`, and `legacy-building-stgnn.md`. They need the snapshot qualifications in this document when wording conflicts with current code.

### 33.2 Primary research/model sources checked for this context

| Source | Role and appropriate claim |
|---|---|
| [Ravi et al., SAM 2](https://arxiv.org/abs/2408.00714) | External segmentation foundation-model lineage; not a new SatelliteGraph network |
| [Xia et al., OpenEarthMap](https://arxiv.org/abs/2210.10732) | Eight-class land-cover benchmark/model context; not the thesis's independent reference labels |
| [Xie et al., SegFormer](https://arxiv.org/abs/2105.15203) | Transformer encoder/MLP decoder architecture lineage |
| [MykolaL/DelineateAnything publisher repository](https://huggingface.co/MykolaL/DelineateAnything) | Source of the pinned parcel checkpoint and its publisher model card |
| [Schlichtkrull et al., relational GCN](https://arxiv.org/abs/1703.06103) | Conceptual relational message-passing reference; the local edge-aware implementation must be described from code |
| [SAM 2 official source](https://github.com/facebookresearch/sam2) | Upstream runtime/configuration source |
| [OpenEarthMap lightweight source](https://github.com/cliffbb/OEM-Lightweight) | Upstream FasterSeg runtime/model context |

The broader citation ledger is `docs/research-sources-for-thesis.md`. It contains sources for SIFT/RANSAC, object-based change, segmentation sensitivity, registration error, tiling/stitching, and possible future alternatives. Verify final author/year/venue/DOI exports before submitting the thesis. Do not copy an external paper's benchmark numbers into SatelliteGraph results or claim that a mentioned future method was implemented.

### 33.3 Historical documents requiring special caution

- `docs/all-class-change-results.md` and former multiclass-GNN plans describe retired experiments.
- Older `docs/pipeline-validation.md` sections mix dated software checks and previous model generations; test counts and model claims must keep their original date/scope.
- `docs/thesis-dashboard.md` statements that segmentation confidence is absent from cards/filters are stale relative to the current crop reviewer.
- `docs/legacy-building-stgnn.md` uses “early stopping”; actual code performs best-state selection after a fixed epoch budget.
- Some preserved figure references are original frozen tile_00002 visuals dated 5 October; the other six tile visualization reference sets are first-reproduction references dated 11 October, explicitly recorded in `output_reference.json`.

## 34. Final handoff rules for ChatGPT

When using this pack to help write the thesis:

1. State the study/run being discussed before quoting counts or metrics.
2. Use current implementation facts for methodology and historical references only for their named experiment.
3. Keep canonical graph tensors, GNN tensors, segmentation scores, rule support, match scores, and GNN probabilities distinct.
4. Use “candidate”, “strong image evidence”, “model support”, “pseudo-label agreement”, and “visual-audit agreement” accurately.
5. Do not claim independent real-world accuracy without an independent reference protocol.
6. Do not call the current GNN a sixteen-head multiclass detector or a downloaded pretrained checkpoint.
7. Do not treat whole-node painted area as localized changed area.
8. Attribute external models, contributor integration, and the student's own work separately.
9. Mark proposed experiments and improvements as future work until actually executed and evaluated.
10. Request a specific missing source/artifact when the answer is not contained here; never fill the gap with plausible-sounding invention.

The best next output from ChatGPT is a faithful thesis outline and a table separating **verified implementation**, **recorded results**, **limitations**, and **missing evidence**. Chapter drafting can then proceed one section at a time using this shared context.

## 35. Expanded resource audit: coverage, reading depth, and source status

This expanded part records resources recoverable from the repository, the present discussion, and locally available earlier project sessions. It is intended to keep the next ChatGPT conversation from losing the research context or misattributing an external method to SatelliteGraph.

### 35.1 What was available to audit

- The project source, model manifests, reference records, and 217 scanned project text/code/configuration files.
- The existing `docs/research-sources-for-thesis.md` ledger, which records 31 papers and ten grouped software/standards resources from earlier work.
- Twenty-six local project-matching session log files, spanning available records from 8–16 October 2026. Some are related session/subtask records, not necessarily 26 independent complete conversations.
- Public primary publication records, arXiv metadata/abstract pages, conference pages, bibliographic DOI metadata, and publisher/model/dataset records checked during this expansion.

The local log scan initially recovered hundreds of URLs. These include repeated paper versions, GitHub API blob requests, asset downloads, package documentation, search-result noise, and browser/setup troubleshooting. A URL appearing in a log does not by itself mean a paper was read or a method was used.

**Coverage boundary:** this is an audit of available local project evidence. It does not establish access to every ChatGPT conversation, deleted chat, inaccessible cloud-only thread, or earlier unlogged discussion. The earlier repository ledger supplies some context predating the available session logs. Resources absent from all these records cannot honestly be reconstructed from memory alone.

### 35.2 Meaning of the reading and use labels

| Label | Meaning |
|---|---|
| Project ledger | Recorded as consulted/referenced in earlier project documentation; not proof of complete full-text reading |
| Metadata/abstract rechecked | Primary title/authors/identifier and available abstract were accessed during this audit; full-text reading is not claimed |
| Registry metadata rechecked | DOI bibliographic record was checked; no substantive method conclusion is inferred from the record alone |
| Implemented component/lineage | The named model family, algorithm, or software is traceable to current code; only the actual local implementation is claimed |
| Methodological context/analogy | Useful motivation or conceptual comparison, without reproducing the paper's method or guarantees |
| Alternative/benchmark lead | Considered or surfaced during prior exploration; not a completed benchmark for the current retained GNN |
| Historical/retired work | Relevant to an earlier experiment or deleted enhancement; its results do not transfer to the current model |
| Unresolved or mismatched | Kept for traceability, but unsuitable as an asserted thesis reference until corrected or verified |

No blanket claim is made that every full paper below was read end to end. The next thesis-writing session should read the relevant full texts before making detailed literature comparisons. Metadata-only records can establish citation identity, not an algorithm's experimental conclusions.

### 35.3 How to use the expanded bibliography

The first table preserves earlier `P01`–`P31` identifiers. Additional arXiv, journal, and conference records follow, with aliases grouped where the same work has both a preprint and a published DOI. The separate software/data ledger records resources that are not research papers. The audit/quarantine section retains incorrect, incomplete, or insufficiently verified leads without promoting them to trusted citations.

For the thesis bibliography, deduplicate by DOI/title and choose the final publication where appropriate. Keep preprint/version dates explicit for works without a verified final publication. Do not cite a search snippet, model README, Reddit discussion, or dataset list as independent validation of SatelliteGraph accuracy.

## 36. Comprehensive recovered paper and article bibliography

The bibliographic entries below are a resource ledger rather than a claim that every listed architecture is implemented. Model-specific benchmark numbers from these publications are intentionally not imported into the project's result tables.

### 36.1 Earlier project citation ledger: P01–P31

Reading depth: **Project ledger** for every row in this first table. The role text distinguishes direct component lineage from analogy and future comparisons. The entries are preserved here so the context pack is usable without uploading a second document.

| ID | Reference and primary link | Project role and limitation |
|---|---|---|
| P01 | Ravi, N., et al. (2024). [*SAM 2: Segment Anything in Images and Videos*](https://arxiv.org/abs/2408.00714). arXiv:2408.00714. | The SAM 2.1 model supplies class-agnostic segment proposals in Stage 2. This project uses pretrained inference, not the paper's video-training method. |
| P02 | Xia, J., Yokoya, N., Adriano, B., & Broni-Bediako, C. (2023). [*OpenEarthMap: A Benchmark Dataset for Global High-Resolution Land Cover Mapping*](https://openaccess.thecvf.com/content/WACV2023/html/Xia_OpenEarthMap_A_Benchmark_Dataset_for_Global_High-Resolution_Land_Cover_Mapping_WACV_2023_paper.html). WACV, 6254–6264. | Eight known land-cover classes and the model/data context for Stage 2 semantic probabilities. The ninth `unknown` class is added by this project. |
| P03 | Chen, W., Gong, X., Liu, X., Zhang, Q., Li, Y., & Wang, Z. (2020). [*FasterSeg: Searching for Faster Real-Time Semantic Segmentation*](https://openreview.net/pdf?id=BJgqQ6NYvB). ICLR. | Architecture lineage for the OpenEarthMap FasterSeg model bundled with the project; the model weights are separately supplied assets. |
| P04 | Huang, B., Reichman, D., Collins, L. M., Bradbury, K., & Malof, J. M. (2018). [*Tiling and Stitching Segmentation Output for Remote Sensing: Basic Challenges and Recommendations*](https://arxiv.org/abs/1805.12219). arXiv:1805.12219. | Motivates overlap reconciliation when Stage 4 combines tile-level predictions. The project's exact ownership rule is its own implementation. |
| P05 | Akyon, F. C., Altinuc, S. O., & Temizel, A. (2022). [*Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection*](https://arxiv.org/abs/2202.06934). arXiv:2202.06934. | Reference for sliced-image overlap handling. The project does not run the SAHI inference package. |
| P06 | Kirillov, A., He, K., Girshick, R., Rother, C., & Dollár, P. (2019). [*Panoptic Segmentation*](https://openaccess.thecvf.com/content_CVPR_2019/html/Kirillov_Panoptic_Segmentation_CVPR_2019_paper.html). CVPR, 9404–9413. | Conceptual reference for reconciling unique object/region identities; no panoptic network is trained here. |
| P07 | Geifman, Y., & El-Yaniv, R. (2017). [*Selective Classification for Deep Neural Networks*](https://arxiv.org/abs/1705.08500). arXiv:1705.08500. | Motivates abstention in uncertain refinement. The project's deterministic gates do not reproduce a calibrated selective-risk guarantee. |
| P08 | Kampffmeyer, M., Salberg, A.-B., & Jenssen, R. (2016). [*Semantic Segmentation of Small Objects and Modeling of Uncertainty in Urban Remote Sensing Images Using Deep Convolutional Neural Networks*](https://openaccess.thecvf.com/content_cvpr_2016_workshops/w19/html/Kampffmeyer_Semantic_Segmentation_of_CVPR_2016_paper.html). CVPR Workshops. | Supports caution around small objects and uncertainty. It does not prescribe the project's 40-pixel unknown threshold. |
| P09 | Lowe, D. G. (2004). [*Distinctive Image Features from Scale-Invariant Keypoints*](https://doi.org/10.1023/B:VISI.0000029664.99615.94). *International Journal of Computer Vision*, 60, 91–110. | Algorithm lineage for SIFT landmarks used through OpenCV during optional cross-year registration. |
| P10 | Fischler, M. A., & Bolles, R. C. (1981). [*Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography*](https://doi.org/10.1145/358669.358692). *Communications of the ACM*, 24(6), 381–395. | Algorithm lineage for robust affine fitting from registration control points. |
| P11 | Chen, G., Zhao, K., & Powers, R. (2014). [*Assessment of the Image Misregistration Effects on Object-Based Change Detection*](https://pages.charlotte.edu/gang-chen/wp-content/uploads/sites/184/2014/01/Chen_2014_ISPRS_Misregistration-OBCD.pdf). *ISPRS Journal of Photogrammetry and Remote Sensing*, 87, 19–27. [DOI:10.1016/j.isprsjprs.2013.10.007](https://doi.org/10.1016/j.isprsjprs.2013.10.007). | Explains why residual shifts can create false object-level changes; motivates registration audits and boundary tolerance. |
| P12 | Yu, W., Zhou, W., Qian, Y., & Yan, J. (2016). [*A New Approach for Land Cover Classification and Change Analysis: Integrating Backdating and an Object-Based Method*](https://doi.org/10.1016/j.rse.2016.02.030). *Remote Sensing of Environment*, 177, 37–47. | Object-based multi-date land-cover analysis; conceptual basis for comparing class polygons over time. The backdating method is not implemented. |
| P13 | Ma, L., Li, M., Blaschke, T., Ma, X., Tiede, D., Cheng, L., Chen, Z., & Chen, D. (2016). [*Object-Based Change Detection in Urban Areas: The Effects of Segmentation Strategy, Scale, and Feature Space on Unsupervised Methods*](https://doi.org/10.3390/rs8090761). *Remote Sensing*, 8(9), 761. | Documents sensitivity of object-level change to segmentation choices. The project does not reproduce its unsupervised method. |
| P14 | Jung, S., Lee, W. H., & Han, Y. (2021). [*Change Detection of Building Objects in High-Resolution Single-Sensor and Multi-Sensor Imagery Considering the Sun and Sensor's Elevation and Azimuth Angles*](https://doi.org/10.3390/rs13183660). *Remote Sensing*, 13(18), 3660. | Cautions that different viewing and illumination geometry can imitate building change. The project's labels remain candidate map changes. |
| P15 | Su, S., Nawata, T., & Fuse, T. (2020). [*Building Change Detection from Bitemporal Aerial Images Using Deep Learning*](https://doi.org/10.5194/isprs-annals-V-2-2020-565-2020). *ISPRS Annals*, V-2-2020, 565–571. | Context for new/demolished building outcomes. The learned detector and its accuracy are not reproduced. |
| P16 | Pekel, J.-F., Cottam, A., Gorelick, N., & Belward, A. S. (2016). [*High-Resolution Mapping of Global Surface Water and Its Long-Term Changes*](https://doi.org/10.1038/nature20584). *Nature*, 540, 418–422. | Shows why temporal water interpretation benefits from many observations. A two-date `water_lost` edge does not establish permanence or seasonality. |
| P17 | Holme, P., & Saramäki, J. (2012). [*Temporal Networks*](https://doi.org/10.1016/j.physrep.2012.03.001). *Physics Reports*, 519(3), 97–125. | General temporal-network context for timestamped nodes and directed forward-time relationships. |
| P18 | Zhou, W., Troy, A., & Grove, M. (2008). [*Object-Based Land Cover Classification and Change Analysis in the Baltimore Metropolitan Area Using Multitemporal High Resolution Remote Sensing Data*](https://doi.org/10.3390/s8031613). *Sensors*, 8(3), 1613–1636. | Historical object-based high-resolution land-cover change precedent. |
| P19 | Gao, S., Li, W., Sun, K., Wei, J., Chen, Y., & Wang, X. (2022). [*Built-Up Area Change Detection Using Multi-Task Network with Object-Level Refinement*](https://doi.org/10.3390/rs14040957). *Remote Sensing*, 14(4), 957. | Example of learned object-level refinement; not used in current deterministic refinement. |
| P20 | Tsai, Y. H., Stow, D., & Weeks, J. (2011). [*Comparison of Object-Based Image Analysis Approaches to Mapping New Buildings in Accra, Ghana Using Multi-Temporal QuickBird Satellite Imagery*](https://doi.org/10.3390/rs3122707). *Remote Sensing*, 3(12), 2707–2726. | New-building detection under multitemporal high-resolution imagery; candidate methodological comparison. |
| P21 | Vakalopoulou, M., Karantzalos, K., Komodakis, N., & Paragios, N. (2015). [*Simultaneous Registration and Change Detection in Multitemporal, Very High Resolution Remote Sensing Data*](https://openaccess.thecvf.com/content_cvpr_workshops_2015/W13/html/Vakalopoulou_Simultaneous_Registration_and_2015_CVPR_paper.html). CVPR Workshops. | Alternative joint registration/change approach; not implemented. |
| P22 | Scheffler, D., Hollstein, A., Diedrich, H., Segl, K., & Hostert, P. (2017). [*AROSICS: An Automated and Robust Open-Source Image Co-Registration Software for Multi-Sensor Satellite Data*](https://doi.org/10.3390/rs9070676). *Remote Sensing*, 9(7), 676. | Recommended comparison for spatially varying registration error; current built-in fit is global affine. |
| P23 | Li, J., Hu, Q., & Ai, M. (2020). [*RIFT: Multi-Modal Image Matching Based on Radiation-Variation Insensitive Feature Transform*](https://doi.org/10.1109/TIP.2019.2959244). *IEEE Transactions on Image Processing*, 29, 3296–3310. | Candidate cross-sensor feature matcher to benchmark against SIFT; not run here. |
| P24 | Sarlin, P.-E., DeTone, D., Malisiewicz, T., & Rabinovich, A. (2020). [*SuperGlue: Learning Feature Matching with Graph Neural Networks*](https://openaccess.thecvf.com/content_CVPR_2020/html/Sarlin_SuperGlue_Learning_Feature_Matching_With_Graph_Neural_Networks_CVPR_2020_paper.html). CVPR, 4938–4947. | Context-aware local feature matching and unmatched decisions; not object-graph matching in this project. |
| P25 | Sun, J., Shen, Z., Wang, Y., Bao, H., & Zhou, X. (2021). [*LoFTR: Detector-Free Local Feature Matching with Transformers*](https://openaccess.thecvf.com/content/CVPR2021/html/Sun_LoFTR_Detector-Free_Local_Feature_Matching_With_Transformers_CVPR_2021_paper.html). CVPR, 8922–8931. | Candidate control-point generator for low-texture image pairs; not run here. |
| P26 | Lindenberger, P., Sarlin, P.-E., & Pollefeys, M. (2023). [*LightGlue: Local Feature Matching at Light Speed*](https://openaccess.thecvf.com/content/ICCV2023/html/Lindenberger_LightGlue_Local_Feature_Matching_at_Light_Speed_ICCV_2023_paper.html). ICCV, 17627–17638. | Candidate learned matcher benchmark; no LightGlue weights are loaded. |
| P27 | Vayer, T., Courty, N., Tavenard, R., Chapel, L., & Flamary, R. (2019). [*Optimal Transport for Structured Data with Application on Graphs*](https://proceedings.mlr.press/v97/titouan19a.html). ICML/PMLR 97, 6275–6284. | Fused Gromov–Wasserstein is a future option for node-feature and graph-structure matching; not implemented. |
| P28 | Batra, A., Singh, S., Pang, G., Basu, S., Jawahar, C. V., & Paluri, M. (2019). [*Improved Road Connectivity by Joint Learning of Orientation and Segmentation*](https://openaccess.thecvf.com/content_CVPR_2019/html/Batra_Improved_Road_Connectivity_by_Joint_Learning_of_Orientation_and_Segmentation_CVPR_2019_paper.html). CVPR, 10385–10393. | Demonstrates why road topology and orientation require dedicated evidence; no orientation head/refinement network is used here. |
| P29 | Mattyus, G., Luo, W., & Urtasun, R. (2017). [*DeepRoadMapper: Extracting Road Topology from Aerial Images*](https://openaccess.thecvf.com/content_iccv_2017/html/Mattyus_DeepRoadMapper_Extracting_Road_ICCV_2017_paper.html). ICCV, 3438–3446. | Road-centerline and missing-connection reasoning; this project currently keeps observed region adjacency only. |
| P30 | Zorzi, S., Bazrafkan, S., Habenschuss, S., & Fraundorfer, F. (2022). [*PolyWorld: Polygonal Building Extraction with Graph Neural Networks in Satellite Images*](https://openaccess.thecvf.com/content/CVPR2022/html/Zorzi_PolyWorld_Polygonal_Building_Extraction_With_Graph_Neural_Networks_in_Satellite_CVPR_2022_paper.html). CVPR, 1848–1857. | Future building-boundary comparison; its polygon predictor is not used. |
| P31 | Carvalho, O. L. F. de, Carvalho Júnior, O. A. de, Albuquerque, A. O. de, & Silva, D. G. e. (2026). [*Remote SAMsing: From Segment Anything to Segment Everything*](https://arxiv.org/abs/2605.00256). arXiv:2605.00256, preprint. | Future multi-pass, cross-tile SAM coverage comparison; not part of current tile-local Stage 2R. |

### 36.2 Additional recovered and verified records

Preprint years below are first-posted years, not automatically the final publication year. Published aliases are included when verified. DOI-only rows have bibliographic identity verification; detailed architectural/experimental claims require reading the full paper.

| ID | Reference and primary link | Reading depth, project role, and status |
|---|---|---|
| P32 | Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, Max Welling (2017, preprint first-posted year). [*Modeling Relational Data with Graph Convolutional Networks*](https://arxiv.org/abs/1703.06103). arXiv:1703.06103. | Metadata/abstract rechecked. Relational-convolution lineage for typed-edge message passing. The local RGNN is its own edge-attribute implementation; do not claim an exact reproduction of the paper. |
| P33 | Seyed Sadegh Mohseni Salehi, Deniz Erdogmus, Ali Gholipour (2017, preprint first-posted year). [*Tversky loss function for image segmentation using 3D fully convolutional deep networks*](https://arxiv.org/abs/1706.05721). arXiv:1706.05721. | Metadata/abstract rechecked. Historical imbalanced-loss comparison in medical segmentation. Tversky loss is not used by the retained GNN. |
| P34 | Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár (2017, preprint first-posted year). [*Focal Loss for Dense Object Detection*](https://arxiv.org/abs/1708.02002). arXiv:1708.02002. | Metadata/abstract rechecked. Historical imbalance-loss alternative. Current GNN uses weighted BCE, not focal loss. |
| P35 | Seyed Raein Hashemi, Seyed Sadegh Mohseni Salehi, Deniz Erdogmus, Sanjay P. Prabhu, Simon K. Warfield, Ali Gholipour (2018, preprint first-posted year). [*Asymmetric Loss Functions and Deep Densely Connected Networks for Highly Imbalanced Medical Image Segmentation: Application to Multiple Sclerosis Lesion Detection*](https://arxiv.org/abs/1803.11078). arXiv:1803.11078. Published record: IEEE Access (2019), [DOI:10.1109/ACCESS.2018.2886371](https://doi.org/10.1109/ACCESS.2018.2886371). | Metadata/abstract rechecked. Medical-segmentation loss analogy encountered in earlier training exploration; not the current GNN objective. |
| P36 | Nabila Abraham, Naimul Mefraz Khan (2018, preprint first-posted year). [*A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation*](https://arxiv.org/abs/1810.07842). arXiv:1810.07842. | Metadata/abstract rechecked. Medical focal-Tversky/Attention U-Net comparison lead; neither architecture nor loss is part of the retained GNN. |
| P37 | Rodrigo Caye Daudt, Bertrand Le Saux, Alexandre Boulch, Yann Gousseau (2018, preprint first-posted year). [*Multitask Learning for Large-scale Semantic Change Detection*](https://arxiv.org/abs/1810.08452). arXiv:1810.08452. | Metadata/abstract rechecked. Semantic-change/multitask and HRSCD literature context; no HRSCD benchmark is recorded for the retained study. |
| P38 | Huikai Wu, Junge Zhang, Kaiqi Huang (2019, preprint first-posted year). [*SparseMask: Differentiable Connectivity Learning for Dense Image Prediction*](https://arxiv.org/abs/1904.07642). arXiv:1904.07642. | Metadata/abstract rechecked. Dense-prediction architecture background encountered with segmentation source exploration; not a selected replacement for FasterSeg. |
| P39 | Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, Alexander M. Rush (2019, preprint first-posted year). [*HuggingFace's Transformers: State-of-the-art Natural Language Processing*](https://arxiv.org/abs/1910.03771). arXiv:1910.03771. | Metadata/abstract rechecked. Software-family attribution for Transformers, used to load SegFormer. The NLP application in this paper is not the project task. |
| P40 | Kunping Yang, Gui-Song Xia, Zicheng Liu, Bo Du, Wen Yang, Marcello Pelillo, Liangpei Zhang (2020, preprint first-posted year). [*Semantic Change Detection with Asymmetric Siamese Networks*](https://arxiv.org/abs/2010.05687). arXiv:2010.05687. | Metadata/abstract rechecked. Asymmetric Siamese semantic-change comparison and benchmark context; not the retained relational model. |
| P41 | Shiqi Tian, Ailong Ma, Zhuo Zheng, Yanfei Zhong (2020, preprint first-posted year). [*Hi-UCD: A Large-scale Dataset for Urban Semantic Change Detection in Remote Sensing Imagery*](https://arxiv.org/abs/2011.03247). arXiv:2011.03247. | Metadata/abstract rechecked. Hi-UCD urban semantic-change dataset lead; not the retained GNN training dataset. |
| P42 | Michael Yeung, Evis Sala, Carola-Bibiane Schönlieb, Leonardo Rundo (2021, preprint first-posted year). [*Unified Focal loss: Generalising Dice and cross entropy-based losses to handle class imbalanced medical image segmentation*](https://arxiv.org/abs/2102.04525). arXiv:2102.04525. Published record: Computerized Medical Imaging and Graphics (2022), [DOI:10.1016/j.compmedimag.2021.102026](https://doi.org/10.1016/j.compmedimag.2021.102026). | Metadata/abstract rechecked. Medical class-imbalance loss comparison. Unified focal loss was not adopted for the retained GNN. |
| P43 | Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, Ping Luo (2021, preprint first-posted year). [*SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*](https://arxiv.org/abs/2105.15203). arXiv:2105.15203. | Metadata/abstract rechecked. Implemented new-segmentation model-family lineage. Report the exact pinned nine-class checkpoint configuration from Section 12. |
| P44 | Li Shen, Yao Lu, Hao Chen, Hao Wei, Donghai Xie, Jiabao Yue, Rui Chen, Shouye Lv, Bitao Jiang (2021, preprint first-posted year). [*S2Looking: A Satellite Side-Looking Dataset for Building Change Detection*](https://arxiv.org/abs/2107.09244). arXiv:2107.09244. Published record: Remote Sensing (2021), [DOI:10.3390/rs13245094](https://doi.org/10.3390/rs13245094). | Metadata/abstract rechecked. Off-nadir building-change benchmark lead. Dataset exploration is not evidence of a completed SatelliteGraph benchmark. |
| P45 | Lei Ding, Haitao Guo, Sicong Liu, Lichao Mou, Jing Zhang, Lorenzo Bruzzone (2021, preprint first-posted year). [*Bi-Temporal Semantic Reasoning for the Semantic Change Detection in HR Remote Sensing Images*](https://arxiv.org/abs/2108.06103). arXiv:2108.06103. Published record: IEEE Transactions on Geoscience and Remote Sensing (2022), [DOI:10.1109/TGRS.2022.3154390](https://doi.org/10.1109/TGRS.2022.3154390). | Metadata/abstract rechecked. Bi-SRNet semantic-change comparison; its bi-temporal network and benchmark results are not the retained GNN. |
| P46 | Lei Ding, Jing Zhang, Kai Zhang, Haitao Guo, Bing Liu, Lorenzo Bruzzone (2022, preprint first-posted year). [*Joint Spatio-Temporal Modeling for the Semantic Change Detection in Remote Sensing Images*](https://arxiv.org/abs/2212.05245). arXiv:2212.05245. Published record: IEEE Transactions on Geoscience and Remote Sensing (2024), [DOI:10.1109/TGRS.2024.3362795](https://doi.org/10.1109/TGRS.2024.3362795). | Metadata/abstract rechecked. SCanNet semantic-change comparison; do not describe its architecture as the current RGNN. |
| P47 | Chao Pang, Jiang Wu, Jian Ding, Can Song, Gui-Song Xia (2023, preprint first-posted year). [*Detecting Building Changes with Off-Nadir Aerial Images*](https://arxiv.org/abs/2301.10922). arXiv:2301.10922. | Metadata/abstract rechecked. Off-nadir building-change context for roof displacement and viewpoint uncertainty; comparison lead only. |
| P48 | Clifford Broni-Bediako, Junshi Xia, Naoto Yokoya (2023, preprint first-posted year). [*Real-Time Semantic Segmentation: A Brief Survey & Comparative Study in Remote Sensing*](https://arxiv.org/abs/2309.06047). arXiv:2309.06047. | Metadata/abstract rechecked. Remote-sensing real-time segmentation survey and architecture-comparison background. |
| P49 | Bipul Neupane, Jagannath Aryal, Abbas Rajabifard (2023, preprint first-posted year). [*Supervised domain adaptation for building extraction from off-nadir aerial images*](https://arxiv.org/abs/2311.03867). arXiv:2311.03867. | Metadata/abstract rechecked. Off-nadir building-extraction/domain-adaptation alternative; current project does not implement its supervised adaptation. |
| P50 | Naoto Yokoya, Junshi Xia, Clifford Broni-Bediako (2023, preprint first-posted year). [*Submeter-level Land Cover Mapping of Japan*](https://arxiv.org/abs/2311.11252). arXiv:2311.11252. | Metadata/abstract rechecked. High-resolution Japanese land-cover mapping context; not a claim that the project uses its mapped imagery. |
| P51 | Kaiyu Li, Xiangyong Cao, Deyu Meng (2023, preprint first-posted year). [*A New Learning Paradigm for Foundation Model-based Remote Sensing Change Detection*](https://arxiv.org/abs/2312.01163). arXiv:2312.01163. | Metadata/abstract rechecked. Foundation-model change-learning alternative; not the retained binary GNN architecture. |
| P52 | Libo Wang, Sijun Dong, Ying Chen, Xiaoliang Meng, Shenghui Fang, Songlin Fei (2023, preprint first-posted year). [*MetaSegNet: Metadata-collaborative Vision-Language Representation Learning for Semantic Segmentation of Remote Sensing Images*](https://arxiv.org/abs/2312.12735). arXiv:2312.12735. | Metadata/abstract rechecked. Metadata/vision-language segmentation alternative. Current pipeline does not perform MetaSegNet metadata fusion. |
| P53 | Keyan Chen, Chengyang Liu, Wenyuan Li, Zili Liu, Hao Chen, Haotian Zhang, Zhengxia Zou, Zhenwei Shi (2023, preprint first-posted year). [*Time Travelling Pixels: Bitemporal Features Integration with Foundation Model for Remote Sensing Image Change Detection*](https://arxiv.org/abs/2312.16202). arXiv:2312.16202. | Metadata/abstract rechecked. Foundation-model bi-temporal feature alternative encountered during change-detector research; not the current runtime. |
| P54 | Deyi Ji, Siqi Gao, Mingyuan Tao, Hongtao Lu, Feng Zhao (2023, preprint first-posted year). [*ChangeNet: Multi-Temporal Asymmetric Change Detection Dataset*](https://arxiv.org/abs/2312.17428). arXiv:2312.17428. | Metadata/abstract rechecked. Multi-temporal asymmetric ChangeNet dataset lead; distinct from the 2018 ChangeNet architecture. |
| P55 | Kai Tang, Jin Chen (2024, preprint first-posted year). [*ChangeAnywhere: Sample Generation for Remote Sensing Change Detection via Semantic Latent Diffusion Model*](https://arxiv.org/abs/2404.08892). arXiv:2404.08892. | Metadata/abstract rechecked. Semantic latent-diffusion sample-generation alternative and historical synthetic-data context; not current GNN training. |
| P56 | Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, Hengshuang Zhao (2024, preprint first-posted year). [*Depth Anything V2*](https://arxiv.org/abs/2406.09414). arXiv:2406.09414. | Metadata/abstract rechecked. Depth-estimation alternative encountered during research. This is not Delineate Anything v2 or the current parcel checkpoint. |
| P57 | Chenyao Zhou, Haotian Zhang, Han Guo, Zhengxia Zou, Zhenwei Shi (2024, preprint first-posted year). [*A Late-Stage Bitemporal Feature Fusion Network for Semantic Change Detection*](https://arxiv.org/abs/2406.10678). arXiv:2406.10678. | Metadata/abstract rechecked. Late-stage semantic-change fusion comparison; not an implemented replacement for the retained GNN. |
| P58 | Chun-Jung Lin, Sourav Garg, Tat-Jun Chin, Feras Dayoub (2024, preprint first-posted year). [*Robust Scene Change Detection Using Visual Foundation Models and Cross-Attention Mechanisms*](https://arxiv.org/abs/2409.16850). arXiv:2409.16850. | Metadata/abstract rechecked. Foundation-model/cross-attention change-detection alternative; no cross-attention layer is present in the retained RGNN. |
| P59 | Yongshuo Zhu, Lu Li, Keyan Chen, Chenyang Liu, Fugen Zhou, Zhenwei Shi (2025, preprint first-posted year). [*Semantic-CD: Remote Sensing Image Semantic Change Detection towards Open-vocabulary Setting*](https://arxiv.org/abs/2501.06808). arXiv:2501.06808. | Metadata/abstract rechecked. Open-vocabulary semantic-change alternative. Current class catalogue remains fixed and the GNN binary. |
| P60 | Junshi Xia, Hongruixuan Chen, Clifford Broni-Bediako, Yimin Wei, Jian Song, Naoto Yokoya (2025, preprint first-posted year). [*OpenEarthMap-SAR: A Benchmark Synthetic Aperture Radar Dataset for Global High-Resolution Land Cover Mapping*](https://arxiv.org/abs/2501.10891). arXiv:2501.10891. | Metadata/abstract rechecked. OpenEarthMap SAR benchmark context; the current implementation uses RGB inputs, not SAR inference. |
| P61 | Mykola Lavreniuk, Nataliia Kussul, Andrii Shelestov, Bohdan Yailymov, Yevhenii Salii, Volodymyr Kuzin, Zoltan Szantoi (2025, preprint first-posted year). [*Delineate Anything: Resolution-Agnostic Field Boundary Delineation on Satellite Imagery*](https://arxiv.org/abs/2504.02534). arXiv:2504.02534. | Metadata/abstract rechecked. Delineate Anything family background. The current asset is v2; v1 results cannot describe the pinned v2 checkpoint. |
| P62 | Mykola Lavreniuk, Nataliia Kussul, Andrii Shelestov, Yevhenii Salii, Volodymyr Kuzin, Sergii Skakun, Zoltan Szantoi (2025, preprint first-posted year). [*Delineate Anything Flow: Fast, Country-Level Field Boundary Detection from Any Source*](https://arxiv.org/abs/2511.13417). arXiv:2511.13417. | Metadata/abstract rechecked. Delineate Anything Flow family comparison. Flow is not the parcel checkpoint loaded by this runtime. |
| P63 | Abdelrrahman Moubane (2026, preprint first-posted year). [*FOTBCD: A Large-Scale Building Change Detection Benchmark from French Orthophotos and Topographic Data*](https://arxiv.org/abs/2601.22596). arXiv:2601.22596. | Metadata/abstract rechecked. French orthophoto/topographic building-change benchmark lead; not the retained seven-tile data. |
| P64 | Omkar Oak, Rukmini Nazre, Rujuta Budke, Suraj Sawant (2026, preprint first-posted year). [*TERRA-CD: Multi-Temporal Framework for Multi-class and Semantic Change Detection*](https://arxiv.org/abs/2605.14651). arXiv:2605.14651. | Metadata/abstract rechecked. Multi-temporal/multiclass benchmark context; no completed current-model TERRA-CD experiment is recorded. |
| P65 | Mykola Lavreniuk, Nataliia Kussul, Andrii Shelestov, Yevhenii Salii, Volodymyr Kuzin, Charlotte Julia Li-Xing Wang, Zoltan Szantoi (2026, preprint first-posted year). [*Delineate Anything v2: A Global Foundation Model for Field Delineation*](https://arxiv.org/abs/2607.19069). arXiv:2607.19069. | Metadata/abstract rechecked. Implemented parcel/field model-family lineage for the pinned DelineateAnythingv2 asset; no project retraining is claimed. |
| P66 | Ying Liu, Haoyu Jiang, Xinying Zhong, Cong Lin, Yuanyuan Wu (2026). [*EMDNet: edge-aware multi-scale difference network for remote sensing image change detection*](https://doi.org/10.1007/s44443-026-01166-8). Journal of King Saud University Computer and Information Sciences, 38(7), 733. DOI:10.1007/s44443-026-01166-8. | Registry metadata rechecked. Edge-aware multi-scale difference alternative; not the current region-graph method. |
| P67 | Xuanguang Liu, Chenguang Dai, Lei Ding, Zhenchao Zhang, Yujie Li, Xibing Zuo, Mengmeng Li, Hanyun Wang, Yuzhe Miao (2025). [*GSTM-SCD: Graph-enhanced spatio-temporal state space model for semantic change detection in multi-temporal remote sensing images*](https://doi.org/10.1016/j.isprsjprs.2025.09.003). ISPRS Journal of Photogrammetry and Remote Sensing, 230, 73-91. DOI:10.1016/j.isprsjprs.2025.09.003. | Registry metadata rechecked. Graph/state-space semantic-change alternative. The retained RGNN has no state-space layer. |
| P68 | Akram Eftekhari, Farhad Samadzadegan, Farzaneh Dadrass Javan (2023). [*Building change detection using the parallel spatial-channel attention block and edge-guided deep network*](https://doi.org/10.1016/j.jag.2023.103180). International Journal of Applied Earth Observation and Geoinformation, 117, 103180. DOI:10.1016/j.jag.2023.103180. | Registry metadata rechecked. Attention/edge-guided building-change comparison; not the current architecture. |
| P69 | Yujun Quan, Anzhu Yu, Xuanbei Lu, Xuefeng Cao, Linyang Li, Xiong You (2025). [*A change detection framework with relative depth information assistance*](https://doi.org/10.1016/j.jag.2025.104942). International Journal of Applied Earth Observation and Geoinformation, 144, 104942. DOI:10.1016/j.jag.2025.104942. | Registry metadata rechecked. Relative-depth-assisted change alternative; no relative-depth channel is used by the retained GNN. |
| P70 | Yinqing Wang, Liangjun Zhao, Yueming Hu, Hui Dai, Yuanyang Zhang (2025). [*Multitask semantic change detection guided by spatiotemporal semantic interaction*](https://doi.org/10.1038/s41598-025-00750-8). Scientific Reports, 15(1), 16003. DOI:10.1038/s41598-025-00750-8. | Registry metadata rechecked. Spatiotemporal semantic-interaction/multitask alternative; not the retained binary building model. |
| P71 | Farid Ali, Soha Safwat Labib, Ayat Mahmoud, Ibrahim Eldesouky Fattoh (2026). [*TransSiamUNet based transformer-augmented Siamese-U-Net for precise change detection in satellite imagery*](https://doi.org/10.1038/s41598-026-43164-w). Scientific Reports, 16(1), 11689. DOI:10.1038/s41598-026-43164-w. | Registry metadata rechecked. Transformer/Siamese U-Net comparison; not the current relational message-passing architecture. |
| P72 | Sai Bhargav Kasetty, K. Rajakumar (2026). [*Lightweight liquid siamese neural network for robust multimodal satellite image change detection*](https://doi.org/10.1038/s41598-026-45655-2). Scientific Reports, 16(1), 20365. DOI:10.1038/s41598-026-45655-2. | Registry metadata rechecked. Liquid Siamese change-model alternative; correct identity of a recovered satellite-change reference. |
| P73 | Yu Shangguan, Jinjiang Li, Zheng Chen (2023). [*Multi-scale graph reasoning network for remote sensing image change detection*](https://doi.org/10.1080/01431161.2023.2221797). International Journal of Remote Sensing, 44(10), 3306-3332. DOI:10.1080/01431161.2023.2221797. | Registry metadata rechecked. Multi-scale graph-reasoning change-detection comparison, not a direct implementation dependency. |
| P74 | Haiyan Huang, Kai Ding, Duowang Zhu, Qimin Cheng, Xiao Huang, Xiaohu Huang, Siyuan Wang, Zhenfeng Shao (2026). [*ChangeVFM: unleashing the power of vision foundation models for semantic change detection in remote sensing images*](https://doi.org/10.1080/10095020.2026.2646372). Geo-spatial Information Science, 1-21. DOI:10.1080/10095020.2026.2646372. | Registry metadata rechecked. Vision-foundation semantic-change alternative; no completed ChangeVFM experiment is recorded here. |
| P75 | Wen Lu, Lu Wei, Minh Nguyen (2024). [*Bitemporal Attention Transformer for Building Change Detection and Building Damage Assessment*](https://doi.org/10.1109/JSTARS.2024.3354310). IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 17, 4917-4935. DOI:10.1109/JSTARS.2024.3354310. | Registry metadata rechecked. BAN/building-change and damage-assessment alternative; not the retained building ST-GNN. |
| P76 | YeKai Cui, Peng Duan, Jinjiang Li (2025). [*PDSDC: Progressive Spatiotemporal Difference Capture Network for Remote Sensing Change Detection*](https://doi.org/10.1109/JSTARS.2025.3569128). IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 18, 16879-16895. DOI:10.1109/JSTARS.2025.3569128. | Registry metadata rechecked. Progressive spatiotemporal-difference alternative; not an implemented retained-model layer. |
| P77 | R.M.A.M.B. Ratnayake, W.M.B.S.K. Wijenayake, D.M.U.P. Sumanasekara, G.M.R.I. Godaliyadda, H.M.V.R. Herath, M.P.B. Ekanayake (2025). [*Enhanced SCanNet with CBAM and Dice Loss for Semantic Change Detection*](https://doi.org/10.1109/MERCon67903.2025.11217111). 2025 Moratuwa Engineering Research Conference (MERCon), 84-89. DOI:10.1109/MERCon67903.2025.11217111. | Registry metadata rechecked. SCanNet enhancement/CBAM/Dice alternative. Current GNN uses neither CBAM nor Dice loss. |
| P78 | Yizheng Chen, Lequan Yu, Jen-Yeu Wang, Neil Panjwani, Jean-Pierre Obeid, Wu Liu, Lianli Liu, Nataliya Kovalchuk, Michael Francis Gensheimer, Lucas Kas Vitzthum, Beth M. Beadle, Daniel T. Chang, Quynh-Thu Le, Bin Han, Lei Xing (2023). [*Adaptive Region-Specific Loss for Improved Medical Image Segmentation*](https://doi.org/10.1109/TPAMI.2023.3289667). IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(11), 13408-13421. DOI:10.1109/TPAMI.2023.3289667. | Registry metadata rechecked. Medical region-specific loss comparison from earlier exploration; not current loss or satellite evidence. |
| P79 | Wu, H. C., Wang, R. F., Qiu, H., Wang, F., Gao, G., & Wu, D. (2025). [*Semantic consistency assisted change detection in remote sensing images*](https://doi.org/10.11834/jrs.20243419). *National Remote Sensing Bulletin*, 29(4), 1031–1045. [Primary publisher record](https://www.ygxb.ac.cn/en/article/doi/10.11834/jrs.20243419). | Registry and publisher identity rechecked. Semantic-consistency comparison; not current runtime. The registry mixed affiliations into its author list; the publisher author list is used here. |
| P80 | Hao Chen, Zhenwei Shi (2020). [*A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection*](https://doi.org/10.3390/rs12101662). Remote Sensing, 12(10), 1662. DOI:10.3390/rs12101662. | Registry metadata rechecked. Spatial-temporal attention and LEVIR dataset context; no retained-model LEVIR benchmark is recorded. |
| P81 | Chengzhe Sun, Jiangjiang Wu, Hao Chen, Chun Du (2022). [*SemiSANet: A Semi-Supervised High-Resolution Remote Sensing Image Change Detection Model Using Siamese Networks with Graph Attention*](https://doi.org/10.3390/rs14122801). Remote Sensing, 14(12), 2801. DOI:10.3390/rs14122801. | Registry metadata rechecked. Semi-supervised Siamese/graph-attention alternative. The retained RGNN is not a graph-attention network. |
| P82 | Junzheng Wu, Ruigang Fu, Qiang Liu, Weiping Ni, Kenan Cheng, Biao Li, Yuli Sun (2023). [*A Dual Neighborhood Hypergraph Neural Network for Change Detection in VHR Remote Sensing Images*](https://doi.org/10.3390/rs15030694). Remote Sensing, 15(3), 694. DOI:10.3390/rs15030694. | Registry metadata rechecked. Dual-neighborhood hypergraph alternative; the retained graph has ordinary typed edges, not hyperedges. |
| P83 | Biao Wang, Ao He, Chunlin Wang, Xiao Xu, Hui Yang, Yanlan Wu (2023). [*A Heterogeneity-Enhancement and Homogeneity-Restraint Network (HEHRNet) for Change Detection from Very High-Resolution Remote Sensing Imagery*](https://doi.org/10.3390/rs15225425). Remote Sensing, 15(22), 5425. DOI:10.3390/rs15225425. | Registry metadata rechecked. Heterogeneity/homogeneity change-network comparison; not current runtime. |
| P84 | Huan Zhong, Chen Wu, Ziqi Xiao (2025). [*LRNet: Change Detection in High-Resolution Remote Sensing Imagery via a Localization-Then-Refinement Strategy*](https://doi.org/10.3390/rs17111849). Remote Sensing, 17(11), 1849. DOI:10.3390/rs17111849. | Registry metadata rechecked. Localization-then-refinement change alternative; corrected DOI from the primary PDF. |
| P85 | Zhuli Xie, Gang Wan, Yunxia Yin, Guangde Sun, Dongdong Bu (2025). [*SDDGRNets: Level–Level Semantically Decomposed Dynamic Graph Reasoning Network for Remote Sensing Semantic Change Detection*](https://doi.org/10.3390/rs17152641). Remote Sensing, 17(15), 2641. DOI:10.3390/rs17152641. | Registry metadata rechecked. Semantically decomposed dynamic graph alternative; not the current two-layer RGNN. |
| P86 | Xinrun Wang, Yuxi Sun, Hao Zheng, Chengjun Li (2026). [*FD-ProtoSCD: Semantic Change Detection in High-Resolution Remote Sensing Images via Frequency-Domain Disentanglement and Dynamic Prototype Learning*](https://doi.org/10.3390/rs18172957). Remote Sensing, 18(17), 2957. DOI:10.3390/rs18172957. | Registry metadata rechecked. Frequency/prototype semantic-change alternative; not the current retained model. |
| P87 | Yong Xie, Yixuan Wang, Xin Wang, Yin Tan, Qin Qin (2025). [*Remote Sensing Image Change Detection Based on Dynamic Adaptive Context Attention*](https://doi.org/10.3390/sym17050793). Symmetry, 17(5), 793. DOI:10.3390/sym17050793. | Registry metadata rechecked. Dynamic context-attention alternative; no corresponding attention module is used here. |
| P88 | Yanis Benidir, Nicolas Gonthier, Clément Mallet (2025). [*The Change You Want To Detect: Semantic Change Detection In Earth Observation With Hybrid Data Generation*](https://openaccess.thecvf.com/content/CVPR2025/html/Benidir_The_Change_You_Want_To_Detect_Semantic_Change_Detection_In_CVPR_2025_paper.html). CVPR, 2204–2214. | Primary conference record/title rechecked. HySCDG/FSC-180k hybrid-data comparison; not current GNN training data. |
| P89 | Duowang Zhu, Xiaohu Huang, Haiyan Huang, Hao Zhou, Zhenfeng Shao (2025). [*Change3D: Revisiting Change Detection and Captioning from A Video Modeling Perspective*](https://openaccess.thecvf.com/content/CVPR2025/html/Zhu_Change3D_Revisiting_Change_Detection_and_Captioning_from_A_Video_Modeling_CVPR_2025_paper.html). CVPR, 24011–24022. | Primary conference record rechecked. Video-modeling/captioning alternative; not an architecture or output of the retained RGNN. |
| P90 | Hung Huy Nguyen, Pooyan Rahmanzadehgervi, Long Mai, Anh Totti Nguyen (2025). [*Improving Zero-Shot Object-Level Change Detection by Incorporating Visual Correspondence*](https://openaccess.thecvf.com/content/WACV2025/html/Nguyen_Improving_Zero-Shot_Object-Level_Change_Detection_by_Incorporating_Visual_Correspondence_WACV_2025_paper.html). WACV; final page range needs checking. | Primary conference record rechecked. Object-level correspondence/no-change comparison; not the current detector. |
| P91 | Aysim Toker, Marvin Eisenberger, Daniel Cremers, Laura Leal-Taixé (2024). [*SatSynth: Augmenting Image-Mask Pairs through Diffusion Models for Aerial Semantic Segmentation*](https://openaccess.thecvf.com/content/CVPR2024/html/Toker_SatSynth_Augmenting_Image-Mask_Pairs_through_Diffusion_Models_for_Aerial_Semantic_CVPR_2024_paper.html). CVPR, 27695–27705. | Primary conference record rechecked. Synthetic semantic-segmentation data alternative; not current model training. |
| P92 | Kaiyu Li, Ruixun Liu, Xiangyong Cao, Xueru Bai, Feng Zhou, Deyu Meng, Zhi Wang (2025). [*SegEarth-OV: Towards Training-Free Open-Vocabulary Segmentation for Remote Sensing Images*](https://openaccess.thecvf.com/content/CVPR2025/html/Li_SegEarth-OV_Towards_Training-Free_Open-Vocabulary_Segmentation_for_Remote_Sensing_Images_CVPR_2025_paper.html). CVPR, 10545–10556. | Primary conference record rechecked. Open-vocabulary segmentation alternative; not the current fixed nine-label checkpoint. |
| P93 | Ashley Varghese, Jayavardhana Gubbi, Akshaya Ramaswamy, Balamuralidhar P (2018). [*ChangeNet: A Deep Learning Architecture for Visual Change Detection*](https://openaccess.thecvf.com/content_eccv_2018_workshops/w7/html/Varghese_ChangeNet_A_Deep_Learning_Architecture_for_Visual_Change_Detection_ECCVW_2018_paper.html). ECCV Workshops. | Primary record/PDF author identity rechecked. Historical visual-change architecture; distinct from the later ChangeNet dataset. |
| P94 | Haolin Li, Jiaojiao Tian, Shan Wang, Ping Song, Yi She (2024). [*An Ensemble Learning Framework for Anomaly Detection of Important Geographical Entities*](https://isprs-archives.copernicus.org/articles/XLVIII-1-2024/317/2024/isprs-archives-XLVIII-1-2024-317-2024.pdf). ISPRS Archives, XLVIII-1-2024, starting page 317. | Primary PDF title/authors checked. Geographical-entity anomaly-detection comparison; not the current retained model. |
| P95 | Utkarsh Mall, Bharath Hariharan, Kavita Bala (2022). [*Change Event Dataset for Discovery from Spatio-temporal Remote Sensing Imagery*](https://www.cs.cornell.edu/projects/satellite-change-events/). NeurIPS Datasets and Benchmarks; verify final proceedings citation export. | Official dataset/project record checked. Multi-temporal change-event dataset lead; not the current training dataset. |
| P96 | Hengtong Shen, Li Yan, Hong Xie, Yaxuan Wei, Xinhao Li, Wenfei Shen, Peixian Lv, Fei Tan (2026, preprint first-posted year). [*Foundation Model-Driven Semantic Change Detection in Remote Sensing Imagery*](https://arxiv.org/abs/2602.13780). arXiv:2602.13780. | Primary metadata/abstract rechecked from a recovered Hugging Face paper lead. PerASCD foundation-model/semantic-consistency alternative; not the retained GNN or its loss. |

### 36.3 Duplicate/version identifiers retained for traceability

| Paper ID | Additional identifier | Deduplication rule |
|---|---|---|
| P17 | [arXiv:1108.1780](https://arxiv.org/abs/1108.1780) | Same work/family entry; do not count as an additional paper. |
| P07 | [arXiv:1705.08500](https://arxiv.org/abs/1705.08500) | Same work/family entry; do not count as an additional paper. |
| P35 | [DOI:10.1109/ACCESS.2018.2886371](https://doi.org/10.1109/ACCESS.2018.2886371) | Published counterpart grouped with its preprint above. |
| P04 | [arXiv:1805.12219](https://arxiv.org/abs/1805.12219) | Same work/family entry; do not count as an additional paper. |
| P24 | [arXiv:1911.11763](https://arxiv.org/abs/1911.11763) | Same work/family entry; do not count as an additional paper. |
| P03 | [arXiv:1912.10917](https://arxiv.org/abs/1912.10917) | Same work/family entry; do not count as an additional paper. |
| P42 | [DOI:10.1016/j.compmedimag.2021.102026](https://doi.org/10.1016/j.compmedimag.2021.102026) | Published counterpart grouped with its preprint above. |
| P25 | [arXiv:2104.00680](https://arxiv.org/abs/2104.00680) | Same work/family entry; do not count as an additional paper. |
| P44 | [DOI:10.3390/rs13245094](https://doi.org/10.3390/rs13245094) | Published counterpart grouped with its preprint above. |
| P45 | [DOI:10.1109/TGRS.2022.3154390](https://doi.org/10.1109/TGRS.2022.3154390) | Published counterpart grouped with its preprint above. |
| P05 | [arXiv:2202.06934](https://arxiv.org/abs/2202.06934) | Same work/family entry; do not count as an additional paper. |
| P02 | [arXiv:2210.10732](https://arxiv.org/abs/2210.10732) | Same work/family entry; do not count as an additional paper. |
| P46 | [DOI:10.1109/TGRS.2024.3362795](https://doi.org/10.1109/TGRS.2024.3362795) | Published counterpart grouped with its preprint above. |
| P26 | [arXiv:2306.13643](https://arxiv.org/abs/2306.13643) | Same work/family entry; do not count as an additional paper. |
| P01 | [arXiv:2408.00714](https://arxiv.org/abs/2408.00714) | Same work/family entry; do not count as an additional paper. |
| P31 | [arXiv:2605.00256](https://arxiv.org/abs/2605.00256) | Same work/family entry; do not count as an additional paper. |

This section contains **96 distinct paper/article ledger entries**, after grouping the listed preprint/published aliases. It is not a count of implemented models, completed benchmarks, or full papers read end to end.

## 37. Software, standards, model sources, datasets, and historical resources

### 37.1 Earlier software/standards ledger: S01–S10

These are technical references, not independent research experiments. The earlier project ledger records their role; not every page was reread in full during this expansion.

| ID | Primary technical source | Project role |
|---|---|---|
| S01 | [GDAL Geotransform Tutorial](https://gdal.org/en/stable/tutorials/geotransforms_tut.html) | Pixel-to-world affine coordinate convention and tiling metadata. |
| S02 | [Rasterio reprojection documentation](https://rasterio.readthedocs.io/en/stable/topics/reproject.html) | Resolution conversion, CRS reprojection, and yearly image/geometry alignment. |
| S03 | [GDAL raster sieve](https://gdal.org/en/stable/programs/gdal_sieve.html) | Background analogy for connected small-region replacement; the project's unknown cleanup uses its own gated implementation. |
| S04 | [scikit-image region-adjacency-graph merging example](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_rag_merge.html) | Comparison for appearance-based neighboring-region merging. |
| S05 | [OGC Simple Feature Access, Part 1](https://www.ogc.org/standards/sfa/) | Polygon/multipolygon geometry vocabulary and geospatial interchange. |
| S06 | [SciPy `linear_sum_assignment`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html) | One-to-one candidate assignment with unmatched options added by project code. |
| S07 | [OpenCV geometric transformations and RANSAC API](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html) and [phase correlation API](https://docs.opencv.org/4.x/d7/df3/group__imgproc__motion.html) | Registration implementation; cite with P09/P10 for algorithm lineage. |
| S08 | [SAHI sliced-inference guide](https://github.com/obss/sahi/blob/main/docs/guides/sliced-inference.md) | Comparison for overlap postprocessing; SAHI itself is not executed. |
| S09 | [SAM 2 source repository](https://github.com/facebookresearch/sam2) | Bundled `third_party/sam2` inference code and model usage. |
| S10 | [OpenEarthMap lightweight model repository](https://github.com/cliffbb/OEM-Lightweight) | Bundled `third_party/oem-lightweight` FasterSeg inference code. |

### 37.2 Runtime, model assets, development sources, and alternatives

Official technical links provide implementation and provenance context. Model-card training claims belong to their publishers; they are not independent measurements made by SatelliteGraph. The pinned local manifest and saved run environment govern reproduction. Alternative links are recovered leads, not a guarantee of ongoing availability or suitability.

| ID | Source | Use and limitation |
|---|---|---|
| S11 | [PyTorch documentation](https://docs.pytorch.org/docs/stable/index.html) | Implemented dependency: tensors, linear layers, Adam, weighted BCE. Exact installed version/run environment governs reproduction. |
| S12 | [Hugging Face Transformers SegFormer](https://huggingface.co/docs/transformers/model_doc/segformer) | Implemented model loader/API. Cite the SegFormer paper separately for architecture lineage. |
| S13 | [Ultralytics segmentation documentation](https://docs.ultralytics.com/tasks/segment/) | Implemented parcel checkpoint runtime interface. Do not infer the custom checkpoint backbone solely from this generic API. |
| S14 | [Streamlit documentation](https://docs.streamlit.io/) | Implemented dashboard/UI dependency; not scientific validation of change accuracy. |
| S15 | [Shapely documentation](https://shapely.readthedocs.io/en/stable/) | Implemented polygon/GEOS operations and geometry repair context. |
| S16 | [Rasterio warp API](https://rasterio.readthedocs.io/en/stable/api/rasterio.warp.html) | Implemented reprojection/resampling coordinate operations. |
| S17 | [NumPy documentation](https://numpy.org/doc/stable/) | Implemented array/statistical operations; cite exact environment where required. |
| S18 | [SciPy documentation](https://docs.scipy.org/doc/scipy/) | Implemented assignment, morphology/numerical utilities as used in local source. |
| S19 | [scikit-image documentation](https://scikit-image.org/docs/stable/) | Implemented image/region processing as used in local source; examples alone do not prescribe project thresholds. |
| S20 | [NetworkX documentation](https://networkx.org/documentation/stable/) | Graph tooling/background; inspect the specific module/run before attributing an operation to this library. |
| S21 | [Python documentation](https://docs.python.org/3/) | Execution/environment support; version in environment lock/run ledger matters. |
| S22 | [pytest documentation](https://docs.pytest.org/en/stable/) | Software regression checks. Passing unit tests do not establish scientific accuracy. |
| S23 | [PyTorch CUDA 12.8 wheel index](https://download.pytorch.org/whl/cu128) | Recovered installation resource; installation URL is not proof that every run used GPU/CUDA. |
| S24 | [Hugging Face Hub documentation](https://huggingface.co/docs/huggingface_hub/) | Model retrieval and pinned-revision context. Current execution loads local verified assets. |
| S25 | [SegFormer checkpoint: boehnen/satlens-segformer](https://huggingface.co/boehnen/satlens-segformer) | Current externally trained semantic checkpoint. Pinned revision 19cdd879e535d76e7fcbbd1e8b54dac9c1da90d6; architecture/config/hashes in Sections 12/27 and model_assets.json. |
| S26 | [Delineate Anything checkpoint publisher](https://huggingface.co/MykolaL/DelineateAnything) | Current DelineateAnythingv2.pt source. Pinned revision 46f98c05ece1d57047e36952a8b440583de38f0d; distinct from Depth Anything. |
| S27 | [Pinned DelineateAnythingv2 checkpoint](https://huggingface.co/MykolaL/DelineateAnything/resolve/46f98c05ece1d57047e36952a8b440583de38f0d/DelineateAnythingv2.pt) | Current asset; local SHA256 46700b8a279b07922953a11adaeb5e658d9a2384b6334c8e0a3090886218915a. This is a segmentation checkpoint, not GNN weights. |
| S28 | [SAM 2.1 Hiera large checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) | Current old-segmentation asset, from Meta distribution. Local hash/size recorded in model_assets.json; upstream checksum publication is not claimed. |
| S29 | [AROSICS official source](https://github.com/GFZ/arosics) | Registration alternative discussed; built-in pipeline uses its own global affine path. |
| S30 | [Depth Anything V2 source](https://github.com/DepthAnything/Depth-Anything-V2) | Explored depth-model alternative; not current parcel inference. |
| S31 | [Alternative Mask2Former satellite model](https://huggingface.co/mfaytin/mask2former-satellite) | Recovered alternative model lead; not the current checkpoint. |
| S32 | [Alternative OpenEarthMap SegFormer](https://huggingface.co/odil111/segformer-fine-tuned-on-openearthmap) | Recovered alternative model lead; current checkpoint is boehnen/satlens-segformer. |
| S33 | [Original retained GNN integration commit](https://github.com/Ahmad1Alzein/SatelliteGraph/commit/a9f9df733fb944b2cdcf647dcdd5e0b41236e019) | Contributor integration/provenance reference, not a downloaded pretrained GNN checkpoint. Contributor attribution does not establish sole authorship of every algorithm. |
| S34 | [Audited project snapshot](https://github.com/Ahmad1Alzein/SatelliteGraph/tree/db1bbfdc7e3cd7ffe6946e3d801cc7d162b4d86a) | Code snapshot for this document. Repository hosting alone does not include every ignored local data/model artifact. |
| S35 | [FasterSeg architecture source](https://github.com/VITA-Group/FasterSeg) | Recovered architecture source; current old segmentation uses the vendored OEM lightweight inference integration. |
| S36 | [Delineate Anything source](https://github.com/Lavreniuk/Delineate-Anything) | Recovered parcel-model family source; the exact current v2 asset is pinned separately. |
| S37 | [Original Segment Anything source](https://github.com/facebookresearch/segment-anything) | Family background/earlier SAM resource; current old segmentation loads SAM 2.1, not original SAM weights. |
| S38 | [SparseMask source](https://github.com/wuhuikai/SparseMask) | Recovered dense-prediction source background; not the selected current semantic model. |
| S39 | [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) | Recovered alternative segmentation framework; not the current SegFormer loader. |
| S40 | [Bi-SRNet alternate recovered source](https://github.com/DingLei14/Bi-SRNet) | Method/dataset source lead; verify redirect/version before reproducing. No local benchmark claimed. |
| S41 | [SCanNet source](https://github.com/ggsDing/SCanNet) | Recovered semantic-change comparison source; not current runtime. |
| S42 | [Enhanced SCanNet source](https://github.com/Buddhi19/Enhanced-SCanNet) | Recovered CBAM/Dice comparison source associated with P77; not retained-model architecture/loss. |
| S43 | [BAN source](https://github.com/likyoo/BAN) | Recovered bitemporal-attention comparison source associated with P75; not the retained RGNN. |
| S44 | [Robust scene-change source](https://github.com/ChadLin9596/Robust-Scene-Change-Detection) | Recovered visual-foundation/cross-attention alternative; not current runtime. |
| S45 | [ChangeAnywhere source](https://github.com/tangkai-RS/ChangeAnywhere) | Recovered synthetic sample-generation alternative; not current retained-GNN training. |
| S46 | [ClearSCD source](https://github.com/tangkai-RS/ClearSCD) | Recovered semantic-change alternative lead; final paper identity not independently established in this audit. |
| S47 | [AdaptOVCD source](https://github.com/Dmygithub/AdaptOVCD) | Recovered open-vocabulary change-detection alternative lead; no current implementation or verified paper attribution claimed. |
| S48 | [EAD-Former source](https://github.com/cujoh777/EAD-Former) | Recovered change-model alternative lead; not current runtime and no unverified architecture details imported. |
| S49 | [Seg2Change source](https://github.com/yogurts-sy/Seg2Change) | Recovered segmentation/change alternative lead; not a current runtime dependency. |
| S50 | [Unified focal loss source](https://github.com/mlyg/unified-focal-loss) | Historical loss comparison source associated with P42; current retained model uses weighted BCE. |
| S51 | [Depth Anything TensorRT wrapper](https://github.com/zhujiajian98/Depth-Anythingv2-TensorRT-python) | Recovered deployment alternative; not current parcel-model runtime. |

### 37.3 Dataset, benchmark, and external-data resources

The retained GNN study uses the preserved local seven-tile graph/labels, and new-input fits use their own generated targets. Except for model-family/data lineage and the explicitly historical experiment, the following are benchmark or data leads. A dataset download discussion does not establish a completed experiment.

| ID | Dataset/resource source | Verified identity or recorded status |
|---|---|---|
| D01 | [OpenEarthMap official project](https://open-earth-map.org/) | Class/model/data lineage. The retained Shouf GNN is not claimed to be trained/evaluated on the OEM benchmark labels. |
| D02 | [S2Looking dataset source](https://github.com/S2Looking/Dataset) | Off-nadir building-change benchmark lead; no current retained-model benchmark completed. |
| D03 | [LEVIR-CD project/dataset source](https://github.com/justchenhao/LEVIR) | Building-change benchmark lead; not the seven-tile retained dataset. |
| D04 | [SYSU-CD dataset source](https://github.com/liumency/SYSU-CD) | Change-detection benchmark lead; not current GNN training data. |
| D05 | [Hi-UCD-S source](https://github.com/Daisy-7/Hi-UCD-S) | Urban semantic-change dataset/code lead. Verify version/classes/license before any future experiment. |
| D06 | [Bi-SRNet / semantic-change source](https://github.com/ggsDing/Bi-SRNet) | Recovered method/dataset navigation lead; not a current dependency or completed SECOND benchmark. |
| D07 | [Extended LsSCD-Ex dataset concept DOI](https://doi.org/10.5281/zenodo.17765754) | DataCite identity checked: Kai Tang, Zhuo Zheng, Hongruixuan Chen, Xuehong Chen, Jin Chen (2025), Extended Large-Scale Semantic Change Detection (LsSCD-Ex) dataset. Historical external-training exploration, excluded from current retained GNN. |
| D08 | [Extended LsSCD-Ex specific version DOI](https://doi.org/10.5281/zenodo.17765755) | Same dataset/version family, not an additional independent dataset or current-model validation. |
| D09 | [DreamCD official source](https://github.com/tangkai-RS/DreamCD) | Dataset provenance/historical synthetic-change exploration. Retired experiment results do not describe the retained GNN. |
| D10 | [Landsat-SCD dataset record](https://doi.org/10.6084/m9.figshare.19946135.v1) | DataCite identity checked: Landsat-SCD_dataset.zip (2022). Different resource from S2Looking; not current RGB tile training data. |
| D11 | [Cornell satellite change events](https://www.cs.cornell.edu/projects/satellite-change-events/) | Official event dataset lead; includes CaiRoad/CalFire/TinyCaiRoad navigation. No current-model experiment recorded. |
| D12 | [VALOR-CD dataset catalogue](https://github.com/WangLukang/VALOR-CD) | Recovered resource catalogue/alternative. Dataset lists do not show which data were actually used locally. |
| D13 | [PerASCD source](https://github.com/SathShen/PerASCD) | Recovered semantic-change comparison/data lead; not a current dependency. |
| D14 | [Functional Map of the World dataset](https://github.com/fMoW/dataset) | Recovered remote-sensing data lead; task/labels differ and current GNN usage is not claimed. |
| D15 | [Google RSRCC resource](https://huggingface.co/datasets/google/RSRCC) | Recovered counterpart/semantic-data exploration lead; not current GNN supervision. |
| D16 | [Fields of the World baselines](https://github.com/fieldsoftheworld/ftw-baselines) | Parcel/field data and baseline context; no current GNN training claim. |
| D17 | [Doodleverse seg2map](https://github.com/Doodleverse/seg2map) | Recovered remote-sensing segmentation resource; not current runtime. |
| D18 | [Copernicus Land Monitoring Service](https://land.copernicus.eu/) | Potential authoritative land-data resource. It is not established as the source of the uploaded Shouf imagery. |
| D19 | [IGN France](https://www.ign.fr/) | Potential orthophoto/data source associated with French benchmark exploration; not asserted as Shouf image provider. |
| D20 | [ChangeNet dataset source, repository spelling ChangNet](https://github.com/jankyee/ChangNet) | Primary README checked. The recovered spelling jankyee/ChangeNet returned 404; use this repository spelling. No current GNN benchmark claimed. |
| D21 | [JL1-CUP-2024 in SECOND-style format](https://huggingface.co/datasets/BiliSakura/JL1-CUP-2024-Second-Format) | Recovered reformatted semantic-change dataset lead; verify original provenance/classes/license before any future experiment. |
| D22 | [PerASCD dataset discussion/files](https://huggingface.co/datasets/SathShen/PerASCD-datasets/discussions/1/files) | Recovered download/discussion lead; not current local GNN training data. |
| D23 | [Remote-sensing dataset catalogue](https://github.com/satellite-image-deep-learning/datasets) | Discovery index used in exploration; cite the original dataset authors instead of treating this index as a benchmark. |
| D24 | [Remote-sensing change literature catalogue](https://github.com/wenhwu/awesome-remote-sensing-change-detection) | Discovery list, not independent validation or proof that all listed papers were read. |
| D25 | [Other recovered satellite dataset catalogues](https://github.com/Polkadoty/satellite-datasets) | Discovery lead; an additional recovered catalogue is [UnlastingR/satellite-datasets](https://github.com/UnlastingR/satellite-datasets). Dataset selection/use must be established from actual runs. |
| D26 | [DFC2021 land-cover baseline](https://github.com/calebrob6/dfc2021-msd-baseline.2) | Recovered land-cover comparison/data lead; not the retained Shouf GNN benchmark. |
| D27 | [MVEO challenge source](https://github.com/mveo/mveo-challenge) | Recovered remote-sensing challenge/data lead; no completed current-model benchmark claimed. |
| D28 | [ChangeNet-v2 alternative source](https://github.com/suvaansh/ChangeNet-v2) | Recovered source lead; no equivalence to the 2018 architecture or later dataset is inferred from the short name. |
| D29 | [Landsat mosaic tooling](https://github.com/kylebarron/landsat-mosaic-latest) | Recovered imagery/tooling lead; not established as the source of the local Shouf imagery. |
| D30 | [OpenStreetMap](https://www.openstreetmap.org/about) | Source family for the saved partial 2023 building-reference comparison on tiles 1, 3 and 6. Exact copied tables and reference/protocol limits are preserved in Section 49; this is not complete two-date change ground truth. |

### 37.4 Local resources that are essential but are not external papers

The exact model asset manifest, original-source reference, original-output reference, original results, run ledgers, input hashes, scientific artifact manifests, and rendering manifests are listed in Sections 18–28 and 33. The training graph is not a checkpoint; a saved probability map is not independent ground truth. The original contributor source is protected separately from the current adapter.

Earlier sessions also used installation/help resources for Anaconda/Conda, CUDA/GPU drivers, package resolution, Git/GitHub, Hugging Face downloads, LibreOffice/Word/OpenXML and document tooling. These explain development or document production; they are not evidence for satellite-change methodology or accuracy. Cite the actual environment/software versions and relevant primary APIs when required, rather than padding the thesis bibliography with every troubleshooting page.

This section records **51 grouped software/model/standards entries** and **30 dataset/data-resource entries**. Some entries contain multiple related links; these counts are ledger rows, not counts of distinct dependencies or experiments.

## 38. Citation corrections, unresolved leads, and excluded retrieval noise

### 38.1 Corrections that matter when writing the thesis

1. DOI `10.3390/rs8090761` is the segmentation-strategy paper by Lei Ma and coauthors. Do not attribute it to a different author from a search snippet.
2. DOI `10.3390/rs13183660` is the sun/sensor-angle building-change paper by Sejung Jung, Won Hee Lee, and Youkyung Han.
3. DOI `10.1038/s41598-026-54829-x` identifies **Fault detection in seismic data using a true 3D global attention convolutional network with self-supervised denoising pretext training**, by Matin Mahzad and Majid Bagheri. It is not the lightweight liquid Siamese satellite-change paper. The latter's verified DOI is `10.1038/s41598-026-45655-2`.
4. **Delineate Anything v2** is the field/parcel model used here. **Depth Anything V2** is a different depth-estimation model encountered during exploration; it is not the current parcel segmentation checkpoint.
5. The CVPR 2025 conference webpage for Benidir and coauthors contains a trailing “f” in one rendering of its title. The paper PDF gives **The Change You Want To Detect: Semantic Change Detection In Earth Observation With Hybrid Data Generation**. Do not propagate the webpage typo into the thesis.
6. The WACV 2025 Nguyen webpage exposes an implausible single-page range in its BibTeX. Check the final proceedings/PDF pagination rather than inventing the missing endpoint.
7. A recovered `...remotesensing-17-01849.pdf` is **LRNet: Change Detection in High-Resolution Remote Sensing Imagery via a Localization-Then-Refinement Strategy**, by Huan Zhong, Chen Wu, and Ziqi Xiao (2025). The PDF gives DOI `10.3390/rs17111849`. Do not derive a DOI by guessing the issue number from the article number.
8. `ChangeNet` can refer to different papers/datasets in different years. Keep the author, full title, year, and identifier attached; do not merge the 2018 visual-change architecture with the later multi-temporal asymmetric dataset because their short names match.
9. A `stgraph_tensors.pt` file is graph data, not pretrained GNN weights. External segmentation/model checkpoints, externally trained retired checkpoints, and saved original GNN predictions are different resources.
10. The publisher author list for DOI `10.11834/jrs.20243419` corrects a registry record that mixed institutional affiliations into the authors. Use the six named authors in P79.
11. The dataset repository is spelled `jankyee/ChangNet`; the recovered `jankyee/ChangeNet` spelling returned 404.
12. Papers about graph attention, hypergraphs, state-space models, diffusion, depth assistance, foundation-model change detection, or focal/Tversky losses do not describe the retained RGNN's architecture/loss. The retained model has the exact linear relational message passing and weighted BCE described in Sections 19–22.

### 38.2 Unresolved or less relevant article leads

These entries preserve recovered leads without asserting that they were read in full, implemented, or suitable for the thesis. Resolve identity and relevance before citing. An unresolved link may be a mirror/duplicate of another entry; do not inflate the bibliography count.

| ID | Recovered identifier/link | Reason it is not a trusted methodological citation |
|---|---|---|
| U01 | [Recovered source lead](https://openreview.net/pdf/385360797c74f4441f02a80de16ec3a16f0cb21a.pdf) | Opaque PDF identifier; bot/verification gate prevented verified title identity. |
| U02 | [Recovered source lead](https://openreview.net/pdf/71ad747c0eccbbf8c88175c8d852d7d070ccad71.pdf) | Opaque PDF identifier; title/authors not established in this audit. |
| U03 | [Recovered source lead](https://ieeexplore.ieee.org/document/10694806) | Publisher JavaScript/access gate; document identity remains unresolved here. |
| U04 | [Recovered source lead](https://assets-eu.researchsquare.com/files/rs-6486635/v1_covered_8ae57c12-6bd5-4d70-915b-9259e59b4846.pdf) | Recovered preprint PDF lead; verified title/version citation not established. |
| U05 | [Recovered source lead](https://combinatorialpress.com/article/jcmcc/Volume%20127/Volume%20127a/Automated%20Classification%20and%20Change%20Detection%20of%20Remote%20Sensing%20Images%20Based%20on%20Convolutional%20Neural%20Networks.pdf) | Filename suggests a CNN classification/change article; author/year/venue record not verified. Filename is not adequate citation evidence. |
| U06 | [Recovered source lead](https://d-nb.info/1239825412/34) | Repository document identity not verified. |
| U07 | [Recovered source lead](https://dirros.openscience.si/Dokument.php?id=41795&lang=eng) | Repository lead; title and final publication identity unresolved. |
| U08 | [Recovered source lead](https://iris.unimore.it/retrieve/handle/11380/1310226/934863/Spotting_Virus_From_Satellites_Modeling_the_Circulation_of_West_Nile_Virus_Through_Graph_Neural_Networks-min%202.pdf) | Filename concerns West Nile virus/GNN modeling; not evidence for the retained change detector. Final citation not verified. |
| U09 | [Recovered source lead](https://repository.msa.edu.eg/server/api/core/bitstreams/83a7499d-19b1-4f19-9cda-e3ff13737e36/content) | Opaque repository bitstream; identity unresolved. |
| U10 | [Recovered source lead](https://ris.utwente.nl/ws/portalfiles/portal/479250174/1-s2.0-S0924271624001709-main.pdf) | Recovered journal PDF mirror; final title/DOI not verified here. |
| U11 | [Recovered source lead](https://www.diva-portal.org/smash/get/diva2%3A1975033/FULLTEXT01.pdf) | Recovered repository full text; final identity unresolved. |
| U12 | [Recovered source lead](https://www.iieta.org/download/file/fid/178649) | Opaque download lead; title/DOI unresolved. |
| U13 | [Recovered source lead](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5544752) | SSRN working-paper lead; verified title/version not established. |
| U14 | [Recovered source lead](https://www.sciencedirect.com/science/article/abs/pii/S0031320325010556) | Publisher PII lead; no verified title/DOI pairing in this audit. |
| U15 | [Recovered source lead](https://www.sciencedirect.com/science/article/pii/S0924271621002835) | Publisher PII lead; no verified title/DOI pairing in this audit. |
| U16 | [Recovered source lead](https://www.sciencedirect.com/science/article/pii/S0924271626002893) | Publisher PII lead; no verified title/DOI pairing in this audit. |
| U17 | [Recovered source lead](https://www.sciencedirect.com/science/article/pii/S0957417425007328) | Publisher PII lead; no verified title/DOI pairing in this audit. |
| U18 | [*Fault detection in seismic data using a true 3D global attention convolutional network with self-supervised denoising pretext training*](https://doi.org/10.1038/s41598-026-54829-x) | Seismic fault detection, incorrectly associated with a satellite-change lead. Correct satellite paper: 10.1038/s41598-026-45655-2. |
| U19 | [*Meta transfer learning for brain tumor segmentation using nnUNet in meningioma and metastasis cases*](https://doi.org/10.1038/s41598-025-20957-z) | Brain-tumour nnUNet/transfer-learning article; unrelated medical search result, not current satellite methodology. |
| U20 | [*U-Net Optimization for Hyperreflective Foci Segmentation in Retinal OCT*](https://doi.org/10.3390/diagnostics16060853) | Retinal OCT U-Net optimisation; unrelated medical search result, not current satellite methodology. |
| U21 | `10.3390/rs17091849` | Rejected guessed DOI: registry returned 404. Correct LRNet DOI is 10.3390/rs17111849. |

Truncated addresses containing ellipses, template placeholders such as `{repo}`, malformed PDF identifiers, and incomplete repository/config prefixes were also recovered. They are not valid reproducible citations. The normalized audit inventory preserves these retrieval leads where useful; this document uses the verified canonical records above.

### 38.3 What was deliberately not promoted to a thesis citation

The available logs also contain search-engine result pages, author homepages, social/community threads, mirrors/proxies, download URLs, badges, API blob/tree URLs, package installation discussions, example images, local Streamlit addresses, and XML document-format namespaces. They may explain retrieval or development activity but are not methodological research evidence. Some surfaced results are unrelated to this project, including medical or language-processing material reached through broad searches.

A normalized URL inventory and public-metadata audit files were retained locally under `artifacts/documentation_audit/`. They contain source identifiers and URL provenance rather than copies of private conversations. Query secrets/signed URL credentials are excluded from the resource index. These audit files are supporting traceability records, not an experiment dataset or a replacement for the self-contained bibliography in this document.

## 39. Literature-to-thesis discussion map

| Thesis question/topic | Resources to discuss | Link to the implementation and limit |
|---|---|---|
| Why object/region-based change analysis? | Object-based land-cover/change papers, segmentation-strategy study, historical urban-building mapping | Explain why regions and their stability matter; no claim that the project reproduces every baseline |
| Why separate geometry and semantics? | SAM 2, OpenEarthMap, FasterSeg, SegFormer, Delineate Anything | Attribute pretrained components and explain the local fusion/rules |
| How are large scenes tiled and stitched? | Tiling/stitching recommendations, SAHI comparison, panoptic segmentation, GDAL/Rasterio geometry | Describe actual overlap-local matching and exact-once ownership as project engineering |
| Why abstain and preserve unknowns? | Selective classification, small-object uncertainty, imbalanced-loss literature | Explain conservative gates; no calibrated selective-risk or focal-loss claim |
| Why can image differences be false changes? | Misregistration, off-nadir building change, sun/sensor-angle studies, AROSICS/RIFT/SIFT/RANSAC | Discuss global affine, roof compensation, seasonal appearance, and independent residual validation |
| What does a temporal/relational graph add? | Temporal Networks, R-GCN, graph/hypergraph/state-space change alternatives | Contrast the exact six-relation RGNN with attention/hypergraph/SSM alternatives that were not adopted |
| How does the task compare with semantic-change networks? | HRSCD/multitask work, asymmetric Siamese networks, Bi-SRNet, SCanNet, ChangeVFM/Semantic-CD and related work | Catalogue multi-class changes versus the retained learned binary building task |
| What would independent benchmarking require? | S2Looking, LEVIR/WHU/SYSU/Hi-UCD/SECOND/ChangeNet/FOTBCD/TERRA-CD resources | Match labels/resolution/tasks and separate geographic train/test data; do not claim a benchmark was completed merely because its download was explored |
| Why consider synthetic or hybrid data? | ChangeAnywhere/DreamCD family, FSC-180k/HySCDG, SatSynth, historical LsSCD-Ex resource | Explain the annotation challenge and the retired exploration; no claim that current GNN training uses these external data |
| How to assess practical usefulness? | Object-level correspondence work, road topology studies, PolyWorld, water temporal monitoring | Include no-change controls, localized footprints, error categories and field-specific limitations |

This map should guide a critical comparison, not a list of disconnected citations. For each related method, explain the task, data/reference labels, evaluation unit, assumptions, and how these differ from SatelliteGraph.

## 40. Instructions for a full master's-thesis writing session

The attached document is a technical context pack and research-resource ledger. It is not permission to manufacture missing experiments or bibliography facts. A useful thesis-writing workflow is:

1. Build an evidence matrix connecting each proposed chapter claim to code, a specific saved run, independent data, or a verified literature source.
2. Agree on the research questions, individual contribution, institution format, and final study/run selection.
3. Write the data chapter only after authoritative imagery metadata and label provenance are available.
4. Draft methodology from the current implementation, including dependencies and caveats that affect comparisons.
5. Present the frozen results with their pseudo-label/transductive scope. Keep new-input training-only examples and retired experiments separate.
6. Read and verify the full texts needed for the literature argument; export final BibTeX/RIS from primary sources and resolve duplicate identifiers.
7. Write an honest discussion of false positives, missing scores, seasonal/registration uncertainty, partial land-node extent, and evaluation limitations.
8. Propose additional experiments as future work until their data, protocols, runs, and results actually exist.

**Do not omit** the difference between multi-class catalogue and binary GNN, the 40/41 versus 38 feature contracts, the original versus new-input target rules, transductive evaluation, fixed-budget best-state selection, independent ground-truth gaps, old-segmentation/new-evidence coupling, graph/matcher coupling, score definitions, and historical resource status.

**Do not claim** that this audit had access to every unprovided cloud chat, that every recovered paper was read in full, that all referenced methods were implemented, or that a resource encountered during search was used to produce the retained thesis result.

## 41. Project work phases and the reason for each phase

This history explains how the present system developed. Dates and commit IDs below come from the retained local Git history; an implementation date is not an image acquisition date. The user discussions explain the goals and problems, but they do not establish that every proposed fix or requested rollback was completed. The history does not show every unlogged action or unavailable cloud conversation.

| Phase | Recorded period and anchor commits | Problem or goal | Work and resulting state |
|---|---|---|---|
| 1. Project and tiling | 21–24 July 2026; `58781be`, `6c6d3b3`, `786f85f` | Large raster imagery needs manageable tiles with reliable placement | Project structure and Stage 1 tiling were introduced. Tile metadata, offsets, masks and edge handling became inputs to later stages. |
| 2. Original segmentation | 4 August; `9676124` | Obtain labelled regions rather than only raw RGB pixels | Unified SAM 2.1/OpenEarthMap segmentation combined class-agnostic masks with semantic labels and deterministic complete-coverage fusion. |
| 3. Refinement and annual graphs | 6–7 September; `2358097`, `fe30070` | Segmentation fragments, uncertain small regions and overlapping tiles produced noisy or duplicate graph objects | Stage 2R refinement and Stage 4 combination added topology handling, evidence-gated cleanup, overlap matching and exact-once ownership. |
| 4. Graph size and temporal analysis | 18–30 September; `5a3de27`, `10357b3`, `15602e6`, `2a77d6c`, `f9dddac` | Graph storage, annual-image execution, contact definitions and cross-year reasoning needed improvement | Graph metadata was reduced; annual/single-PNG paths, spatiotemporal graphs, valid side-contact adjacency and clearer overlays were developed. |
| 5. Temporal matching and readable outcomes | 3–5 October; `fb2ed09`, `0063b72`, `bc89b6e` | Relocated outputs, ambiguous matches and class-specific change interpretation | Path handling, temporal previews, local cross-year matching and class-specific outcomes were refined and documented. |
| 6. Imported semantic-temporal study package | 7 October; `2415ec2`, `75184af`, `7028215` | Integrate the supplied 2013–2023 experiment without rewriting its scientific scripts | A byte-preserved experiment package, separate stage commands and initial old/new stage selection were added. Frozen reproduction remains separate from general input processing. |
| 7. General selectable runtime | 8 October; `b0dae64` through `02ea174` | Historic hardcoded assumptions did not support arbitrary files, years, folders and combinations | Native input preparation, old/new adapters, common graph contracts, temporal/evidence processing, reports, manifests, saved options, resume and model acquisition were added. Earlier GNN choices in this phase were later superseded. |
| 8. All-class evidence and visual inspection | 9 October; `cc60fb7`, `8f9279c`, `d345812`, `b693127`, `e230641`, `d763e99`, `482dde6` | Users needed more than building-only lists and needed to inspect each stage | Eight-class evidence, sixteen gain/loss task records, conversion events, stage PNGs, matching views and candidate grids were introduced. Prepared registration was reused across stages. An experimental sixteen-head learned model was also attempted; that model is retired. |
| 9. External-supervision exploration | 10 October; `c807f73` | Weak/unsupported pseudo-label training motivated externally annotated data and graph controls | LsSCD-Ex subset training, model/ablation/expert experiments and all-class review were explored. Support and transfer were inadequate for a claim of reliable all-class performance. The current runtime no longer uses this learned implementation. |
| 10. Retained building GNN and thesis dashboard | 11 October; `a9f9df7`, `a312168`, `d482a38` | Preserve the original building ST-GNN study and present its evidence clearly | Contributor integration added protected original scripts, input/output references and a reproducible study runner; the thesis dashboard presented saved metrics, maps and evidence-labelled pages. |
| 11. Restore the requested model and consolidate the app | 13 October; `e523bba`, `9806776`, `9fdeb08` | The user wanted the originally supplied thesis GNN, working stage choices and understandable documentation | Experimental runtime GNN/training code was removed; a new-input adapter calls the original binary model. The separate Streamlit entry point was incorporated into the dashboard. TensorFlow oneDNN was disabled at startup, and an English/Arabic guide was added. |
| 12. Restore useful score review and correct adapter inputs | 16 October; `db1bbfd` | Weak landscape candidates, low/absent scores and limited visible results caused confusion | Segmentation score display/filtering, global descending sorting and shared paginated review were added/restored; rule-support provenance and original-contract input mapping were corrected; the major-focus display gained the 10% parent-region rule. Protected original GNN code/results were preserved. |
| 13. Thesis context and research audit | This document, after the audited code snapshot | A new ChatGPT session needs accurate implementation, history, resources and limitations | The context pack and bibliography were assembled from source, saved artifacts, notes, Git history and available local session logs. This is documentation work; no new pipeline run or model training was performed. |

### 41.1 What the older cleanup experiment actually demonstrated

`docs/unknown-cleanup-calibration.md` records a four-tile refinement comparison using an approximately 0.8 m grid, not the later preserved seven-tile 1.5 m study. Its baseline had 1,082 unknown objects, including 941 through 40 pixels and 423 single pixels. The supported cleanup extension retained the 40-pixel candidate limit and allowed an energy ceiling of 0.50 only with independent semantic/contact/appearance/ambiguity gates.

That historical record reports 1,921 to 1,597 per-tile objects, 1,082 to 744 unknown objects, and 2,621 additionally assigned unknown pixels before overlap ownership. Its combined full graph had 1,751 to 1,457 nodes; the analysis selection then displayed 906 nodes and 2,045 edges while omitting 551 remaining small unknowns. The two-tile selection and threshold sweep are documented in that source.

These are historical geometry/coverage and graph-complexity observations, not independently measured semantic accuracy. Absorbing pixels with evidence differs from hiding unresolved nodes in an analysis view. The 0.8 m grid, four-tile selection, historical thresholds and recorded test count must stay attached to that experiment. They must not be merged with the retained GNN cohort or presented as a current new/old segmentation benchmark.

### 41.2 Important history distinctions

- The original Stage 1–5 workflow, imported frozen semantic study, general runtime and retained building GNN are related components with different contracts; they are not one unchanged executable from July onward.
- Adding the original thesis building GNN and retiring the experimental GNN were different events. The former is recorded on 11 October and the latter on 13 October.
- The retained history contains the earlier experimental commits and a later removal/restoration commit. A request to delete commits is not evidence that Git history was rewritten. This document records the history that actually remains.
- A request to return to a previous night's result does not prove that later execution produced identical arrays. The selected saved run and its inputs/configuration are the concrete evidence; see Section 45.
- Documentation and configuration assertions can become stale. Current source and the selected run's saved artifacts take precedence over a previous plan or success message.

## 42. Old and new pipelines: detailed comparison and improvement claims

There are three independent stage choices in the interface, not only two fixed whole pipelines. The names below compare stage behavior. Both stage selections share important infrastructure, and either segmentation can feed either graph and either change method through adapters.

| Aspect | Original/old choice | New/current choice or shared improvement | Benefit and remaining limitation |
|---|---|---|---|
| Segmentation geometry | SAM 2.1 proposes class-agnostic object masks | SegFormer evidence, Delineate Anything v2 field masks and class-specific deterministic region construction | Adds parcel-aware and class-specific construction. Neither method guarantees individual real-world objects. |
| Semantic model | OpenEarthMap FasterSeg supplies eight named classes; fusion adds unknown | Pinned nine-label SegFormer supplies background/unknown and eight named classes | Different semantic evidence family; no current independently labelled comparison establishes universal superiority. |
| Meaning of pretrained | External SAM/OEM assets are loaded locally | External SegFormer/parcel assets are loaded locally | Attribution and pinned hashes make model provenance explicit. These models were not trained from scratch as the student's segmentation contribution. |
| Processing grid | Old method operates on its prepared configured grid | New method applies a 2× processing grid with consistent affine scaling | Can aid processing/boundary operations; does not add sensor detail. Pixel-count comparisons require scale correction. |
| Multi-scale evidence | Old fusion exposes assigned-label support through the canonical adapter | New inference supplies probability products across scales/augmentations, including mean and conservative maximum evidence | Provides richer class evidence for absence tests. Maximum-over-scale channels are not one normalized probability distribution. |
| Region assembly | Ordered SAM mask fusion plus leftover coverage and refinement | Building/road/water/parcel/residual coverage sequence, component splitting and deterministic refinement | Makes class-specific geometry decisions explicit. Errors or priorities at one step can affect later regions. |
| Land parcels | Generic class-agnostic proposals interpreted by OEM | Field/parcel proposals from Delineate Anything plus semantic acceptance rules | Useful land-region geometry context; a field-delineation mask is not proof of agricultural land or a cadastral boundary. |
| Rule-assigned classes | Original fusion/cleanup use their own semantic and topology gates | Some bare/water regions can be assigned by RGB rules with separately recorded support | Preserves the source of the decision. Raw SegFormer support can remain weak even when a rule assigns a class. |
| Unknown/uncertainty | Unknown labels and conservative refinement abstention | Unknowns, evidence-quality flags, rule provenance and observability gates remain explicit | Avoids fabricated certainty. Unknowns and exclusions reduce model coverage. |
| Complete pixel coverage | Deterministic old fusion partitions valid tile pixels | New region assembly also partitions valid pixels | Both provide complete scientific coverage; this is a contract, not evidence that every label is correct. |
| Spatial relationships | Pixel-side adjacency only | Adjacency plus building `near` and broader `context_near` | Represents relationships across gaps; adds edges/cost and can connect irrelevant neighbours. |
| Connected regions | Original objects can retain multipart identity | New graph view can split disconnected components and preserve lineage | Better alignment between a graph node and a connected image region. Node counts and correspondence can change. |
| Node/edge summaries | Original graph schema is converted by the adapter | Both general graph choices expose the common canonical geometry/appearance/features | Do not claim that all 40 canonical features were absent from the old adapted graph. The key new distinction is relations/connectedness and associated matching behavior. |
| Annual tile reconciliation | Original Stage 4 overlap matching and ownership | Reused/adapted annual ownership, with annual statistics and proximity edges rebuilt | Avoids duplicate area and object identities. This is retained engineering, not an entirely new algorithm in the new graph choice. |
| Cross-date matching | Old graph selection uses the old Stage 5 matching behavior | New graph selection uses current overlap/primary correspondence and roof-aware handling | Supports a richer matching context. A graph-choice comparison also changes the matcher, so it is not an isolated edge ablation. |
| Input flexibility | Original study/stage commands depend on their own tile and metadata contracts | General runtime supports single images, mapped rasters, folders and explicit assumed layouts | More usable input paths. Metadata-free row placement is an assumption, not geographic recovery. |
| Alignment | Original APIs and frozen scripts have their recorded conventions | Current comparison saves one prepared alignment and reuses it across stages | Reduces inconsistent stage geometry. Global affine alignment can still fail or leave local parallax. |
| Classical change | Stage 5 polygon/extent rules with class-specific outcomes | Old classical events are adapted into the shared current reports | Makes the old method inspectable in the same viewer. Polygon differences still depend on segmentation/registration quality. |
| Learned change | The retained original building ST-GNN is a separate historical study | New-input adapter fits that same binary architecture when eligible labels exist | Enables the requested model on new inputs. It is a training-only fit, not deployment of pretrained weights or sixteen learned change heads. |
| Other-class changes | Old rules describe class-specific extent outcomes | Current catalogue lists eight-class gains/losses/conversions and structural differences | Broader review coverage. Catalogue candidates are not all GNN predictions or confirmed physical changes. |
| Review scores | Earlier views emphasized different diagnostics and sometimes hid semantic support | Current cards display raw segmentation support, optional GNN probabilities, score filter and global descending order | Makes weak candidates easier to inspect. Neither score is calibrated physical-change confidence. |
| Visualization | Original overlays/audit graphs and study figures | Shared segmentation, graph, matching, GNN and paginated crop reports for stage combinations | Easier inspection and traceability. A polished image cannot establish scientific accuracy. |
| Reuse and integrity | Original stage artifacts and historic references | Saved context, dependency hashes, scientific/render states and manifests | Detects stale/tampered/incomplete outputs. Missing external inputs or a partial export still prevents full reproduction. |
| Scientific evidence | Original rule behavior and recorded examples | Retained GNN ablation and saved audits plus software checks | Only the stated cohort/protocol supports measured findings; execution checks are not accuracy tests. |

### 42.1 Concrete examples of what the changes mean

**Buildings separated by a road:** adjacency alone may connect each building to the road but give no direct building-to-building edge. The new graph can add a `near` relation within its radius. The retained GNN can consume that relation. Additional `context_near` edges among other classes remain useful graph products but are excluded from the original GNN's six-relation contract.

**A small differently coloured patch in a large field:** both methods can segment it differently between dates. The current catalogue can retain that difference while major focus hides it if its changed footprint is below 10% of a participating parent area. This improves review focus; it does not prove the patch was a false change, and it can hide a small real change.

**A weak earlier land label and a strong later roof:** the event segmentation score is limited by the lower available participating-region support. A low value can therefore reflect uncertainty in the earlier land interpretation rather than a poor later roof mask. The score is not raised to make the change look stronger.

**One roof split into several later segments:** summing same-class counterpart persistence can prevent outline/split artifacts from passing major-building focus. Split/merge correspondence is still an observation relationship, not automatic evidence of physical construction or demolition.

### 42.2 Improvements that are demonstrated versus improvements that remain hypotheses

The following are implemented and inspectable engineering changes: interchangeable stages, explicit common contracts, richer spatial relations, preserved connected-region lineage, saved alignment reuse, evidence provenance, complete-candidate access, score sorting/filtering, and scientific/render integrity checks.

The preserved experiment provides one quantitative learned-model comparison: pooled AP rises from 0.084 for the MLP and 0.100 for the spatial-only GNN to 0.530 for the full ST-GNN; pooled F1 rises from 0.204 and 0.183 to 0.549. This is **held-out pseudo-label agreement under the frozen full-graph protocol**, not an old-versus-new segmentation benchmark or independent real-world accuracy.

Section 49 also preserves a restricted saved OSM building-instance comparison: the repository new segmentation has pooled F1 0.342 versus 0.221 for repository old segmentation on that partial 2023 reference. A different imported method called `improved` reaches 0.413; it must not be confused with the current runtime new segmentation. These saved proxy-reference observations are narrower than independent complete segmentation/change accuracy, and their experiment provenance still needs verification for final thesis claims.

Claims that the new segmentation is universally more accurate, that the new proximity graph improves arbitrary-image change precision, or that major focus increases physical-change accuracy require separately controlled independent evaluation. More changes, fewer changes, larger masks, more edges, stronger displayed scores or cleaner figures do not individually demonstrate those claims.

## 43. Consolidated problem ledger: symptoms, causes, actions and open limits

This ledger records problems encountered or exposed by the available project evidence. **Implemented correction** means source contains the stated change; it does not mean every user image was rerun successfully. **Mitigation** limits a symptom while the underlying uncertainty remains. **Expected limitation** is a deliberate contract/evidence boundary. No pipeline was rerun while preparing this ledger.

| ID | Problem or observed symptom | Diagnosis or evidence | Current action/status and remaining consequence |
|---|---|---|---|
| I01 | Registration failed with insufficient inlier support | Candidate landmarks can be false, repetitive, too local or inconsistent; image scales/viewpoints may differ | Current mapped/pixel paths have phase-guided and descriptor matching, robust fitting and support/scale/distribution gates. Legitimately weak pairs still stop. Original failing upload was not rerun in this documentation task. |
| I02 | Different stages could obtain different alignments | Earlier code repeated registration fitting instead of reusing the saved preparation | Implemented correction `482dde6`: sealed prepared context is reused downstream. Changing registration is an upstream change requiring consistent outputs. |
| I03 | Cross-year pixels appear moved even with a successful fit | Global affine cannot model all roof parallax, terrain and local deformation | Roof-aware handling and boundary margin mitigate some differences. Independent residual validation remains missing; fitted landmarks are not independent ground truth. |
| I04 | `GEOSException ... Unable to determine overlay result geometry dimension` | Empty/mixed geometry collections can enter area overlay operations | Current `src/stage5/events.py` normalizes polygonal area and returns an empty polygon for empty intersections; regression coverage exists. This fixes a specific case, not every possible invalid geometry. The originally failing full pair was not rerun here. |
| I05 | Missing `results/rendering/change/manifest.json` | A folder can contain an old ledger/scientific data without complete verified current renders | `dashboard/run_results.py` separates pending, verified and partial archived exports. Archived events can be viewed/downloaded with honest status. Complete crops require imagery/preparation/renders; a manifest must not be fabricated. |
| I06 | Pipeline computation finished but reports were not ready | Scientific and render completion are separate | Current ledger stores both statuses. Verified scientific artifacts can remain available after a rendering failure; that is not full visual completion. |
| I07 | Rasterio `NotGeoreferencedWarning` about identity affine | Image-relative rasters may have no real map geotransform; GDAL can omit identity metadata | Treat as a coordinate/provenance issue to inspect, not evidence that segmentation/GNN failed. Physical and pixel units remain explicit. Do not invent CRS or suppress meaningful provenance warnings to claim success. |
| I08 | TensorFlow oneDNN startup messages and small numerical variation concern | A dependency initializes TensorFlow even though the retained GNN is PyTorch | App startup sets `TF_ENABLE_ONEDNN_OPTS=0` in `9806776`. This changes that TensorFlow setting; it does not remove TensorFlow or guarantee all package logs disappear. It is not GNN retraining. |
| I09 | Log moved directly from `Segment 2013` to `Segment 2023` | Current native loop prints before each tile call and waits for that call to return; it does not print a separate per-tile completion line | Seeing the next tile indicates normal sequential progression, not a skipped first tile. A verified tile manifest is stronger completion evidence. An explicit timed per-tile completion log is not established as an implemented fix. |
| I10 | Many tiny unknowns/noisy graph fragments | Segmentation uncertainty and topology decomposition create small connected components | Evidence-gated cleanup, topology consolidation and separate analysis omission were implemented. Remaining unknowns stay in the full audit products; omission must not be counted as correct classification. |
| I11 | Duplicate objects/area in overlapping tiles | Independent tile predictions can represent the same footprint twice | Stage 4 overlap matching, safe grouping and exact-once ownership address this. Bad placement or imperfect matching can still change object grouping. |
| I12 | Graph contact dots/links looked misleading | Corner contacts, tiny boundaries, local contact display and graph identity were confused | Historical adjacency/contact corrections enforce side-sharing and clearer overlays. Visual simplification does not add links across omitted uncertain nodes. |
| I13 | Large numbers of weak changes in homogeneous landscapes | Boundary shifts, fragmented/relabelled regions, RGB ambiguity, shadow and seasonal appearance can make a small false difference | Candidate preservation plus score/area/status filters and major focus mitigate review load. No independent evaluation establishes that all weak differences are false or that all true changes remain visible. |
| I14 | Segmentation scores often below 0.65 or very low | Raw assigned-class semantic support; rule-assigned labels can disagree with the semantic network; event score takes the lowest available side | Scores restored and explained, without artificial inflation. The slider filters semantic support, not calibrated change truth. Scores from different segmentation methods are not automatically comparable. |
| I15 | GNN score repeatedly unavailable | Eligible positive and negative building/built-over targets are both required; many candidates are unrelated or masked | `insufficient_data` is intentional. Saved examples in Section 45 show the missing target class. Zero or invented probability would be misleading. |
| I16 | A one-change result despite score zero and weak results enabled | Other filters can remain active; one diagnostic run also had identical input images | All-candidate access requires every applicable filter to be permissive. For `d18ed7bb290c`, identical files are confirmed; there are two raw catalogue candidates and one major candidate, not an undisplayed set of forty valid detections. |
| I17 | `65 of 320 changes shown` caused concern that results were missing | Visible count is after all active class/status/area/score/major filters and before/with page selection; historical screenshot alone does not identify all active filters | Current review sorts the full filtered collection before pagination and exposes navigation. Do not infer 255 computation failures from this count. Exact historical filter state cannot be reconstructed from the count alone. |
| I18 | Only the first page was useful or ordering was confusing | Ordering after slicing would fail to rank the complete result collection; score availability also varies | Current shared reviewer sorts descending globally, leaves missing values last, and paginates 12 per page. Grid exports use the same score descriptions/order. |
| I19 | `Season may explain this` looked inconsistent with the displayed class | Current catalogue conversions consider both source and target; seasonal classes there are grass, crop and bare ground. Old classical event metadata uses a different vegetation set. | A building-target conversion can be flagged because of its source. The policies are not uniform and do not estimate actual season or physical truth. Manual review can resolve a supported case; acquisition-month metadata are still needed. |
| I20 | High GNN values, sometimes near 1, on questionable cases | Weighted-BCE fit and dependent pseudo-labels can produce confident agreement with wrong or ambiguous targets | Preserved outputs remain unchanged and their audit is disclosed. Calibration and independent annotation remain open work; model probability is not a correctness guarantee. |
| I21 | Earlier land node painted as broadly changed | Binary built-over prediction is a node decision; only part of a land region may be converted | Explain whole-node versus localized change. The aggregate 7.58% area ratio is not the mean individual 28.08% fraction. Existing loose dashboard wording is a documented presentation issue, not a corrected code change here. |
| I22 | Concern that the retained GNN had been enhanced/retrained/replaced | A separate experimental all-class model existed in earlier development | `e523bba` removed the experimental runtime model/training implementation and checkpoint route. Current adapter calls protected original code. Old scripts/records can still describe retired work; they do not determine current model choice. |
| I23 | GNN behavior changed despite unchanged original source | Adapter inputs, class mappings, overlap features, confidence provenance, labels and preprocessing can alter downstream results | Latest adapter corrections are documented in Sections 22/44. Original-source hash alone does not prove identical new-input results. Frozen study and its references remain separate. |
| I24 | Missing temporal `overlap_score` lost useful support | Some canonical overlap edges provide endpoint fractions without the original scalar field | Adapter now derives `sqrt(max(0, fraction_source*fraction_target))` only when the field is absent; explicit zero remains zero. Implemented correction with mapping regression; not an architecture change. |
| I25 | Rule-based class decisions lost their provenance or adapter support | Ownership/component conversion could omit rule metadata; raw semantic support alone misrepresented a rule-assigned feature | Current artifacts preserve rule provenance/support, and eligible bare/water rule support can feed the original confidence feature. Raw semantic scores and original target-confidence gate are not silently raised. |
| I26 | Saved outputs reused after model/input/option changes | Incomplete dependency fingerprints can make stale cache results look current | Input/model/metadata/context content and stage dependencies are fingerprinted; nested asset/provenance issues received regressions. Use verified saved state; presentation version changes do not establish scientific validity. |
| I27 | Arbitrary folders/year names failed or implied wrong placement | Original package assumptions differed from general runtime contracts | General inputs and saved stage options were added. Explicit metadata is preferable; automatic natural-order row layout is a documented assumption and can be wrong for another folder. |
| I28 | Old segmentation claimed full semantic probabilities when it supplied assigned-label support | An adapter tensor with zeros in other channels is not a complete semantic distribution | Evidence-quality flag identifies this state; comparison can recover full SegFormer evidence. Therefore a general old-segmentation comparison can still depend on the new semantic model. |
| I29 | New graph seemed like an isolated graph change | The graph method also selects temporal matching behavior and connected-region handling; extra context edges are not consumed by the retained GNN | Implemented coupling is explicit. Do not attribute any difference solely to spatial edges without an isolated experiment. |
| I30 | External/all-class GNN gave limited or poor useful output | Support imbalance, validation gates and cross-domain transfer limited supported directions in the historical experiment | Experimental code/checkpoint route is retired. Its conditional/small-subset results and failed transfer must not be merged with the original model's metrics. |
| I31 | Model could not recover omitted real changes | If segmentation omits a real object or merges it into a wrong region, the graph lacks the correct node/footprint | Expected upstream dependency. Graph message passing cannot reliably create a missing object geometry. Segmentation needs independent evaluation. |
| I32 | Large inputs/reports exhausted time or memory | Upsampling, multiple class tensors, full image reads, graphs and complete canvases scale with scene size | Explicit pixel/report budgets and spatial subsetting mitigate resource use. Unlimited arbitrary-scene scalability and GPU performance are not established. |
| I33 | Duplicate app entry points, README claims and checksum uncertainty | Standalone workspace and dashboard once coexisted; historical docs retained obsolete statements | One Streamlit entry point is current; `streamlit_app.py` is absent. `SHA256SUMS.txt` remains for the private artifact/data package. Older README prose saying there is no Streamlit app is stale; it is not current behavior. |
| I34 | Unknown exact imagery/license/acquisition metadata | Source filenames and entered years cannot establish provider/sensor/month/rights | Open documentation gap. Obtain authoritative metadata before final data chapter; do not treat guessed metadata as a fix. |
| I35 | Evaluation appeared stronger than supported | Pseudo-label dependence, full-graph transduction, one seed, small audit and training-only new fits; partial OSM and historical point tables have different evidence/protocol limits | Evidence is labelled precisely, including the weak tables in Section 49. Independent complete physical-change accuracy and broad generalization remain unestablished. |

### 43.1 Exact seasonal-label behavior

`runtime/change_catalog.py` defines `SEASONAL={4,5,6}`: grass/shrubs, crop and bare ground. For a conversion, it tests **either** endpoint. Thus bare-ground → building can receive a seasonal-sensitive status even though the target is a building. Tree and water are not included in this particular set, although they can also have real seasonal variation. A conversion between building, road and paved classes need not receive that status.

Two other display/evidence policies use different sets. The old classical adapter in `runtime/temporal.py` marks tree/grass/crop IDs `{3,4,5}` in an event's `season_sensitive` metadata, while keeping adapted event status `candidate`. Major focus in `runtime/preview.py` excludes conversions entirely among `{3,4,5}`. These are separate implementations, not one consistent learned seasonal classifier. This explains some inconsistent-looking flags and is a policy limitation to discuss or investigate in a future change; this documentation task does not unify the policies.

This flag is a conservative class-based ambiguity policy. It does not use independently verified acquisition seasons, a phenology model or measured weather to explain an individual event. A reviewed positive with sufficient recorded review confidence can produce `reliable` status and `season_resolved_by_review`, while preserving the underlying seasonal flag. This is resolution by review, not automatic proof that season was or was not the cause.

### 43.2 What remains open at the current snapshot

The principal open scientific problems are weak candidate precision on some new images, segmentation failure/uncertainty, seasonal/viewpoint ambiguity, limited training-target availability, uncalibrated probabilities, no representative independent reference set, and restricted geographic generalization evidence. The main open usability/provenance details include duplicate-input blocking, explicit per-tile completion timing, partial exports without crops, some stale documentation/UI wording, and the exact identity of unresolved bibliography leads.

Do not write “all problems solved.” A truthful statement is: the project includes targeted software corrections and clearer review/integrity controls, while important scientific-quality and evidence gaps remain.

## 44. Retired GNN experiments, retained model protection and adapter changes

### 44.1 What the retired work attempted

Earlier general-runtime work attempted a shared relational model with sixteen class/direction tasks, external annotation supervision, validation-based checkpoint selection, graph ablations and later specialist/expert routing. `docs/all-class-engineering-record.md` and `docs/all-class-change-results.md` preserve that history. Their uses of “current model” mean current **at that historical date**, before removal.

The historical native single-pair validation reported 284 image-evidence candidates, 25 scored grass-gain node/tasks and no accepted positives for that particular revised external model. These are historical reported observations from that earlier experiment. They are not the 369-candidate saved upload, the two-candidate identical-input diagnostic, or the retained GNN's seven-tile result. The record itself says professional target-image reliability was not achieved. Earlier conditional metrics, changed supervision populations and expert models are not directly comparable without their full cohort/protocol definitions.

The user subsequently selected the original thesis-dashboard building GNN and requested removal of the attempted enhancement/training model. The current tree has no `runtime/gnn_model.py`, `runtime/gnn_training.py`, `runtime/external_training.py` or `scripts/download_lsscd_subset.py` at their former paths. The current GNN adapter does not load that external experimental checkpoint. Historical data, notes, Git commits and frozen scripts may remain for traceability; remaining files must be interpreted by their role, not by the presence of `gnn` in a filename.

### 44.2 Which GNN files remain and why

| File family | Role | Interpretation |
|---|---|---|
| `pipelines/legacy_building_stgnn/original/gnn_change.py` | Protected original binary building model and training functions | Retained thesis GNN; source manifest and hash identify it |
| `pipelines/legacy_building_stgnn/adapter.py` | General new-input conversion and fitting | Reconstructs original feature/edge/target interface; separate from original-source protection |
| `pipelines/semantic_temporal/runtime/gnn.py` | Current runtime dispatch | Calls the retained adapter; not the retired sixteen-head network |
| `pipelines/semantic_temporal/pipeline/gnn_change.py` | Byte-preserved imported experiment script | Frozen package component with its own reproduction context; not automatically another active trained runtime model |
| Frozen graph tensors, predictions and histories under the data/reference package | Preserved evidence and reproducibility inputs/outputs | Tensors are data; predictions/history JSON are outputs; none should be relabelled as pretrained GNN weights |

### 44.3 What changed in the current adapter and what did not

The protected original architecture, original training functions and preserved study arrays/outputs are not replaced by the latest review work. However, adapters and upstream preparation are scientific inputs to new comparisons and can affect their fitted results. The current corrections include original-label class mapping, overlap-feature fallback and rule-supported confidence mapping with preserved provenance.

In particular, a valid separately recorded bare/water rule support can supply the original confidence **feature**, while node eligibility still checks raw confidence/observation/known-counterpart requirements. This is not a global rescaling of the displayed segmentation score. It can change a new-input feature vector, so it should be disclosed when comparing adapter versions.

To establish identical results, one must match **all** relevant source/model assets, original and adapted graph arrays, registration, semantics, feature order/values, relations, targets, options, random settings and environment, then compare saved outputs. Matching the original GNN Python hash alone is insufficient. This is why the frozen study's manifest/reference protection and a general upload's run ledger serve different purposes.

## 45. Saved-run identity and why results cannot be mixed

### 45.1 Concrete runs discussed in the available evidence

| Run or output family | What it represents | Observed status/evidence | Safe interpretation |
|---|---|---|---|
| Preserved seven-tile building study and `artifacts/legacy_building_stgnn/full_run_2026-10-11/` | Original building ST-GNN experiment/reproduction with saved predictions/figures | Protected cohort, seven-fold ablations and reference verification described in Sections 19–23/27 | Source for the recorded GNN metrics; not a prediction model fitted on each new uploaded pair |
| `artifacts/image_comparisons/8060742522ec/` | User-selected last uploaded-image result from the night of 12 October | Saved new/new/GNN selection; segmentation/graph/science/render complete; change `insufficient_data`, one positive target and zero negatives; no GNN probability fit | Rich candidate results here come from image evidence/catalogue, not from a successful new-input GNN probability output |
| `artifacts/image_comparisons/57102c7fbeca/` | Another saved comparison of the differing pair | Saved new/new/GNN selection; change `insufficient_data`, one positive target and zero negatives | Keep its run identity/options; do not treat it as a separate independent accuracy experiment merely because it has another ID |
| `artifacts/image_comparisons/d18ed7bb290c/` | Diagnostic associated with the reported one-change concern | Saved before/after byte/pixel identity; change `insufficient_data`, zero positive targets and six negatives; two raw seasonal candidates, one passing major focus | Does not show that a successful GNN removed forty real changes. The input pair and label support differ fundamentally. |
| `artifacts/all_class_validation/pair_final/` and related October 9–10 records | Earlier external/all-class experimental development | Historical reported single-tile validation and retired model limitations | Historical engineering evidence, not retained-model performance |
| `artifacts/shouf3/tile_00001_geo/` and other data-only exports | Older exports involved in missing-manifest messages | Availability of event JSON does not establish complete source/context/render package | May support archived event inspection; cannot automatically provide verified current crops or rerun capability |

The earlier documentation audit found 369 raw catalogue candidates and 53 passing the current major-focus policy for each of `8060742522ec` and `57102c7fbeca`. These are current-policy read-only diagnostic counts on saved data. They do not reconstruct every filter state or card count displayed on the original night, and they are not counts of independently confirmed changes.

### 45.2 Why “same tile number” or “same model” can give a different result

Possible differences include different source pixels, incorrect duplicate uploads, geographic coverage, entered years, metre scale, registration, harmonization, segmentation choice/weights/options, graph connectivity/matcher, change policy, target availability, adapter version, display profile and filters. Several can change the result before the GNN is called. Some saved runs selected GNN but never obtained both label classes, so no fitted GNN probabilities were produced.

The user's observation that yesterday's result looked useful is valuable usability evidence. It does not by itself identify which internal stage caused a later difference. The correct comparison uses saved input hashes, run configurations, scientific artifact hashes, target counts, model status and filter state. A saved screenshot or a short terminal log alone is insufficient to attribute the difference to changed GNN weights.

### 45.3 Practical evidence chain for discussing a result

Record the run folder/ID, input file hashes and observation dates; the method triple; prepared coordinate units/grid/registration; segmentation model revisions/options; graph/matching policy; candidate/target counts; original/adapter source version; GNN status and evaluation scope; scientific/render manifests; then active display filters/page selection. Keep the original run immutable when comparing a changed configuration. These are instructions for a future controlled review, not a claim that a new rerun was performed for this document.

## 46. How to investigate weak results and compare improvements fairly

### 46.1 Diagnose from the earliest uncertain stage

1. **Check inputs.** Confirm that the before and after contain different intended observations of the same area. Compare actual pixels/hashes, coverage, entered years and source metadata. Identical inputs are a useful no-change control, not a physical-change pair.
2. **Check alignment.** Inspect stable landmarks across the image, the saved affine and residual distribution. Test points not used to fit the transform when available. Do not lower support gates simply to force a run through.
3. **Check segmentation at the candidate location.** Inspect both class maps, boundaries, assigned-class probabilities, rule provenance, observed masks and any missing/merged real object.
4. **Check annual ownership.** Determine whether the region is from one tile or reconciled overlap, whether offsets were applied once and whether each represented pixel has one owner.
5. **Check correspondence.** Inspect primary/overlap links, split/merge components, roof/ground frame, same-class persistence and counterpart coverage.
6. **Check image evidence.** Separate a raw class difference from strong presence/absence evidence. Inspect local neighbourhoods, maximum-over-scale absence, shadows, seasonal endpoints, component area and boundary margin.
7. **Check target eligibility.** Read exact exclusion reasons and positive/negative counts. A large candidate catalogue does not guarantee valid training labels.
8. **Check the adapter and GNN status.** Confirm the 38-feature contract, six relation types, original code identity and current run status. If status is `insufficient_data`, do not interpret the case as an evaluated GNN probability failure.
9. **Check the viewer last.** Inspect all filters, sorting and pagination; compare raw catalogue count with filtered count. Review low scores as semantic support, not a change-confidence threshold.
10. **Assign an error category with evidence.** Examples are wrong input, registration/parallax, missed object, wrong class, boundary fragment, ownership mismatch, wrong temporal identity, weak target support, ambiguous physical truth or display exclusion. Preserve uncertain cases instead of forcing a binary explanation.

This sequence is a diagnostic protocol. It is not an instruction to modify the protected GNN or to run pipelines during the documentation task.

### 46.2 Suggested controlled old/new comparison

| Question | Hold fixed | Compare | Evidence needed before calling it an improvement |
|---|---|---|---|
| Does new segmentation label objects better? | Same source observations, preparation/coverage, physical scale and independent labels | Old versus new semantic/instance geometry, including unknowns and missed objects | Per-class/object/boundary metrics, error examples and comparable evaluation population |
| Does new spatial context help? | Identical region nodes/features, temporal relations, labels, seed and training/evaluation protocol | Adjacency versus added proximity relations in an isolated **new** experiment | Matched test performance and support, uncertainty across seeds/areas; current whole-stage selection also changes matcher |
| Does temporal context help the retained model? | The exact preserved cohort and protocol | Existing MLP/spatial/full frozen ablation | Current saved pseudo-label-agreement evidence supports this restricted comparison; independent physical truth needs additional labels |
| Does the current adapter faithfully map inputs? | Hand-checkable canonical nodes/edges and declared original contract | Expected versus reconstructed features, relations, confidence and masks | Mapping/provenance regressions and explicit artifact comparison; architecture unchanged is insufficient |
| Does major focus improve review usefulness? | Same immutable candidate catalogue and independently reviewed examples | All candidates versus focused view | Precision/coverage/review-time trade-off, including real small changes hidden by the filter |
| Are semantic/GNN probabilities trustworthy? | Independent reference and fixed fitted model/evaluation population | Reliability across probability bins, support/coverage and thresholds | Calibration analysis and independent outcomes; a high histogram peak or training score is insufficient |
| Can the system handle larger or other scenes? | Declared hardware/environment, model options and memory budgets | Different scene sizes, areas, sensors/views and class mixtures | Runtime/RAM/failure rates and independent accuracy across scenes; one working pair is not universal validation |

These additional comparisons are proposed future experiments unless a named completed run and protocol provide the evidence. Preserve the original frozen study unchanged; label improved protocols or modified adapters as new experiments.

### 46.3 Reporting failure cases in the thesis

For each case, show paired original crops, aligned class maps, relevant graph/match view and localized event footprint. State the true/uncertain interpretation, error category, method/configuration, applicable scores/status, and whether the case entered training or only review. Include unchanged controls and missed changes, not only successful algorithm-selected events. Keep candidate precision, GNN pseudo-label agreement, visual-audit agreement, segmentation quality and software integrity as separate results.

## 47. Handoff completeness, remaining source files and thesis-writing priorities

### 47.1 What this upload does and does not contain

| Material | Included in this document? | What a deeper thesis-writing session may still need |
|---|---|---|
| System purpose, stage choices, work history and main design decisions | Yes, explained in text/tables | Supervisor-approved framing of novelty and student contribution |
| Segmentation/graph/temporal/GNN methods and input-output contracts | Yes, with exact important parameters and module paths | Specific source files for line-level derivations or unlisted implementation details |
| Recorded frozen cohort, training protocol, ablations and audit results | Yes, summarized with scope and limitations; partial external-reference CSVs are preserved in Section 49 | Original JSON/tables/figures for independent recomputation, plotting or final thesis figure export |
| New-input weak-result examples and exact known diagnostic statuses | Yes, run IDs and observations are recorded | The immutable run folder, images, arrays, logs and saved filter context to reproduce a specific case |
| Research papers and technical/dataset references | Citation identities, links, roles and reading-depth status | Relevant full texts, final citation exports, unresolved-link identity and any publisher access requirements |
| Source imagery, trained segmentation checkpoints and graph tensors | Paths/provenance/contracts, not the binary files themselves | Authorized local files or permitted excerpts; private imagery must not be assumed redistributable |
| GNN pretrained deployment checkpoint | None; no such retained pretrained checkpoint is claimed | A future training/deployment experiment would be new work, not a missing download from this context pack |
| Independent physical-change ground truth and calibration | Gaps and future protocol are explained | Actual independent annotations, sampling protocol, agreements and completed evaluation |
| Complete history of every cloud chat | No; available local evidence is audited | Any unavailable earlier material that changes a factual claim; no invented reconstruction |
| University formatting, approved title and final authorship statement | Missing items identified | Author/supervisor/institution decisions |

### 47.2 Checksum, export and document-role details

`SHA256SUMS.txt` remains in the repository to verify the private data/artifact package. Its rows include frozen study data and results; it is not a model, configuration file or substitute for the per-run manifests. Possessing that text file does not supply the referenced imagery or checkpoint bytes. The README explains the private package layout and verification command. Original imagery is documented there as private; verify permitted use before sharing source files.

The project has one Streamlit entry point, `thesis_dashboard.py`; the selectable workspace is `dashboard/pages/pipelines.py`. Removing the former standalone `streamlit_app.py` did not remove the pipeline workspace. Having one top-level README does not require deleting methodology, validation or historical design documents: those serve different purposes. This context pack consolidates their essential facts and flags stale statements rather than pretending every older file is authoritative.

`artifacts/documentation_audit/` contains local research metadata, recovered public-link provenance, document assembly helpers and structural verification records. The app does not use it and the final context pack does not depend on it at runtime. It is not a change-detection dataset, model, training output or copy of all private conversations. It is ignored by Git. The self-contained document can still be read if that helper folder is removed.

### 47.3 Priority order for the next ChatGPT thesis session

First establish the final research questions, individual contribution and selected experiment. Then verify authoritative data provenance and draw the implementation diagram. Draft the old/new methodology using Sections 7–22/42, present only supported results from the correct run/cohort, and use the problem ledger to write threats to validity and discussion. Read the relevant full papers for a critical literature argument. Finish with proposed experiments clearly separated from completed work.

ChatGPT should not promise a complete scientifically validated thesis merely from a detailed software description. When a required fact or experiment is missing, it should identify the gap and draft the surrounding supported material without inventing evidence.

## 48. Arabic explanation of the work phases, improvements and remaining problems

### 48.1 كيف تطور المشروع؟

بدأ العمل بتقسيم الصورة الكبيرة إلى صور صغيرة مع حفظ مكان كل صورة. بعد ذلك استخدمنا SAM 2.1 لرسم حدود المناطق، وOpenEarthMap لتحديد نوعها، مثل مبنى أو شجرة أو طريق. ثم أضفنا تحسين التقسيم وبناء الرسم البياني ودمج الصور المتداخلة دون حساب البكسل نفسه مرتين. بعد ذلك أصبحنا نربط مناطق السنتين ونفحص الاختلافات بينها.

في المسار الجديد نستخدم SegFormer لإعطاء احتمالات أصناف الأرض، وDelineate Anything v2 للمساعدة في حدود الحقول والمناطق الأرضية، مع قواعد إضافية لبناء المناطق وتحسينها. الرسم البياني الجديد يضيف علاقات قرب بين المناطق، بدلاً من الاعتماد على التلامس فقط. هذه إضافات فعلية في البرنامج، لكن إثبات أنها أدق في كل الصور يحتاج إلى تقييم مستقل.

جُرّب سابقاً نموذج GNN موسع لفئات واتجاهات تغير متعددة مع بيانات تدريب خارجية. هذا النموذج التجريبي أزيل من مسار التشغيل الحالي بناءً على طلب المستخدم. النموذج المعتمد الآن هو نموذج المباني الأصلي الموجود في لوحة الرسالة، وليس النموذج التجريبي الموسع.

### 48.2 لماذا قد تكون النتائج ضعيفة رغم سلامة التشغيل؟

نجاح البرنامج يعني أنه أنتج ملفات صحيحة وفق العقود المطلوبة؛ لا يعني أن كل تغير حقيقي. إذا أخطأ التقسيم في نوع منطقة، أو لم تتطابق الصورتان جيداً، فقد يظهر اختلاف صغير كأنه تغير. الظل والموسم وزاوية التصوير وتقسيم حقل واحد إلى أجزاء مختلفة قد تسبب النتيجة نفسها. الرسم البياني لا يستطيع استعادة مبنى حقيقي لم يظهر أصلاً كموقع أو منطقة مناسبة في التقسيم.

درجة التقسيم تعبّر عن دعم النموذج للصنف، وليست احتمال صحة التغير. إذا كان أحد طرفي التغير ضعيفاً، فإن الدرجة المعروضة تأخذ أقل درجة متاحة بين المناطق المشاركة. بعض أصناف الأرض تحددها قواعد الألوان رغم أن احتمال SegFormer لها ضعيف؛ لذلك تبقى الدرجة منخفضة ولا نرفعها اصطناعياً.

عندما لا توجد أمثلة تدريب مناسبة للتغير وعدم التغير معاً، تظهر حالة `insufficient_data` ولا ينتج GNN احتمالات. وجود مئات مرشحات التغير لا يعني وجود أمثلة مناسبة لتدريب نموذج المباني. التشغيل المحفوظ `8060742522ec` كان يحتوي على مثال إيجابي واحد وصفر أمثلة سلبية، لذلك لا يصح وصف مرشحاته بأنها نتائج احتمالات GNN.

### 48.3 لماذا لا تظهر كل التغيرات؟

العدد المعروض يتأثر بمرشحات الصنف والحالة والمساحة والدرجة وتركيز التغيرات الكبيرة، ثم بالصفحات. جعل الحد الأدنى للدرجة صفراً لا يلغي بقية المرشحات. مرشح التغيرات الكبيرة قد يخفي اختلافاً صغيراً داخل منطقة أرضية واسعة؛ هذا تنظيم للعرض وليس حذفاً من سجل المرشحات العلمي أو إثباتاً أن الاختلاف غير حقيقي.

في حالة التشغيل `d18ed7bb290c` كانت صورتا قبل وبعد متطابقتين فعلياً. وُجد مرشحان صغيران وبقي واحد في عرض التغيرات الكبيرة. هذه الحالة لا تثبت أن GNN أضاع أربعين تغيراً؛ يجب أولاً التأكد من أن الصورتين هما الملاحظتان المختلفتان المطلوبتان.

عبارة «قد يفسره الموسم» هي قاعدة حذر تعتمد على صنف المصدر أو الهدف، وليست قياساً للموسم. في سجل التحولات الحالي تشمل القاعدة العشب والزراعة والتربة المكشوفة؛ لذلك قد تظهر مع مبنى إذا كان صنف الأرض السابقة واحداً منها. الأشجار والماء غير داخلين في هذه المجموعة رغم احتمال تأثرهما بالموسم. مسار التغير الكلاسيكي ومرشح العرض يستخدمان مجموعة نباتية مختلفة، ولهذا لا تكون العلامات موحدة تماماً. التحقق يحتاج إلى تاريخ الالتقاط والمراجعة، وليس إلى اسم الصنف الظاهر فقط.

### 48.4 ما الذي تحسن وما الذي ما زال يحتاج إلى عمل؟

تحسن تنظيم المسارات، وتوافق المراحل، وحفظ الإحداثيات، وإعادة استخدام المحاذاة، وتوثيق مصادر الدرجات، والتحقق من الملفات، وإمكانية استعراض المرشحات بالترتيب التنازلي وفي صفحات. بقيت مشكلات علمية مهمة: تغيرات ضعيفة أو خاطئة، عدم اليقين في التقسيم والموسم، نقص أمثلة التدريب المناسبة، وعدم وجود تقييم مستقل شامل على تغيرات حقيقية.

النتائج الرقمية الأصلية محفوظة ومحددة ببروتوكولها. ثبات ملف نموذج GNN لا يعني وحده ثبات كل النتائج الجديدة، لأن المدخلات والمحول والمحاذاة والتقسيم والأهداف قد تتغير. لذلك تفرق الرسالة بين المحافظة على النموذج الأصلي، وتصحيح المحول، وتحسين العرض، وإثبات الدقة علمياً. لا نكتب أن كل المشكلات حُلّت، ولا ننسب دقة مستقلة إلى نتائج مبنية على تسميات مولدة آلياً.

## 49. Additional partial-reference evaluation records, including weak results

The local `data/external_eval/` folder contains four CSV summaries that must not be omitted from the project history. `dashboard/data.py` loads the aggregate building, point and count summaries; `dashboard/pages/evaluation.py` presents the partial OSM building comparison and separately states that complete independent human change accuracy is unavailable. Merely loading a CSV does not verify its annotation protocol.

These tables are copied below exactly as saved, rather than recomputed. Their method names refer to the originating comparisons and are not automatic evidence that the latest commit/adapter was evaluated. No original reference point labels, complete point sampling protocol, confidence-interval construction, or exact producer environment/commit is established by these CSVs alone. They belong to a separate evidence family from the preserved seven-fold GNN pseudo-label metrics.

### 49.1 Partial 2023 building-instance comparison against OSM

The dashboard identifies this as 2023-only instance matching at IoU ≥ 0.5 on tiles 1, 3 and 6. The aggregate table contains 645 OSM reference instances. It is an external but incomplete reference for building geometry, not complete two-date change truth. Missing OSM buildings and temporal/positional mismatch can distort apparent false positives and negatives.

Keep four names distinct: `baseline (frozen v1)`, `improved`, `repo new segmentation (SegFormer + DA v2 runtime)`, and `repo old segmentation (SAM 2.1 + FasterSeg)`. The dashboard describes `improved` as a native 0.5 m building method from `OriginalPipeline_Improved`. It is not another name for the repository's new segmentation.

The aggregate recorded precision/recall/F1 are 0.443/0.316/0.369 for baseline, 0.487/0.358/0.413 for imported improved, 0.420/0.288/0.342 for repository new, and 0.192/0.259/0.221 for repository old. These are modest proxy-reference results, with substantial missed matches and variation across tiles. They must be reported with that limitation, not hidden behind a claim that all segmentation problems were solved.

Source: `data/external_eval/buildings_osm_all.csv`.

```csv
method,osm,pred,tp_iou50,osm_split,pred_merging_2plus,precision,recall,f1,f1_tile_00001,f1_tile_00003,f1_tile_00006
baseline (frozen v1),645,460,204,4,45,0.443,0.316,0.369,0.432,0.381,0.168
improved,645,474,231,7,47,0.487,0.358,0.413,0.472,0.404,0.274
repo new segmentation (SegFormer + DA v2 runtime),645,443,186,2,33,0.42,0.288,0.342,0.396,0.353,0.169
repo old segmentation (SAM 2.1 + FasterSeg),645,868,167,36,43,0.192,0.259,0.221,0.275,0.172,0.193
```

The per-tile file has nine rows for baseline, improved and repository new. It does not contain repository-old per-tile rows; do not fabricate them from the aggregate F1 columns. Split/merge counts, pixel metrics and instance metrics describe different error properties and should not be treated as interchangeable.

Source: `data/external_eval/buildings_osm_per_tile.csv`.

```csv
tile,method,osm,pred,tp_iou50,precision,recall,f1,mean_best_iou,osm_split,pred_merging_2plus,osm_inside_merges,pixel_iou,pixel_f1
tile_00001,baseline (frozen v1),252,192,96,0.5,0.381,0.4324,0.3625,3,22,62,0.4897,0.6574
tile_00001,improved,252,214,110,0.514,0.4365,0.4721,0.3934,6,17,42,0.5049,0.671
tile_00001,repo new segmentation (SegFormer + DA v2 runtime),252,188,87,0.4628,0.3452,0.3955,0.3514,1,16,45,0.4816,0.6501
tile_00003,baseline (frozen v1),286,208,94,0.4519,0.3287,0.3806,0.3379,1,17,42,0.4518,0.6224
tile_00003,improved,286,199,98,0.4925,0.3427,0.4041,0.3452,0,24,59,0.4592,0.6294
tile_00003,repo new segmentation (SegFormer + DA v2 runtime),286,196,85,0.4337,0.2972,0.3527,0.3118,1,11,27,0.4194,0.591
tile_00006,baseline (frozen v1),107,60,14,0.2333,0.1308,0.1677,0.2074,0,6,16,0.3461,0.5143
tile_00006,improved,107,61,23,0.377,0.215,0.2738,0.246,1,6,12,0.3629,0.5325
tile_00006,repo new segmentation (SegFormer + DA v2 runtime),107,59,14,0.2373,0.1308,0.1687,0.206,0,6,16,0.3435,0.5113
```

### 49.2 Historical point-reference summaries requiring protocol verification

`reference_points_all.csv` contains fifteen summaries with a nominal `n` field of 44 for building change and 47 for any change. Its method labels include baseline/improved original-task ST-GNN, improved multiclass methods and repository candidate selections. These are not automatically the retained study's prediction cohort or independently validated outputs of the current adapter.

The file includes very low original-task GNN F1 values, 0.042 and 0.024, alongside high-precision/low-recall rule selections and broad uncertainty intervals. Preserve these observations and investigate their definitions; do not silently replace them with the frozen F1 0.549, or claim that the difference alone proves the original GNN weights were modified. Different references, sampling, units, methods and alignment can produce very different metrics.

The headers `p_ci`, `r_ci` and `f1_ci` give interval strings but do not state the confidence level or estimation procedure. `n` is not the number of true positives. A recorded precision 1.0 with low recall and an interval `[1.00, 1.00]` is not sufficient evidence of perfect real-world accuracy. Until source labels, prediction mapping, thresholds, denominators, sampling/weighting and interval construction are verified, describe these as **historical summary values with unresolved protocol**, not final independent thesis accuracy.

Source: `data/external_eval/reference_points_all.csv`.

```csv
task,method,precision,p_ci,recall,r_ci,f1,f1_ci,n
building change,baseline rule layer,0.925,"[0.77, 1.00]",0.067,"[0.02, 0.93]",0.125,"[0.05, 0.96]",44
building change,improved rule layer,0.991,"[0.97, 1.00]",0.061,"[0.02, 0.92]",0.116,"[0.04, 0.95]",44
building change,improved multi-class (building nodes),1.0,"[1.00, 1.00]",0.01,"[0.00, 0.27]",0.021,"[0.00, 0.43]",44
building change,baseline ST-GNN (original task),0.042,"[0.01, 0.71]",0.043,"[0.01, 0.72]",0.042,"[0.01, 0.52]",44
building change,improved ST-GNN (original task),0.025,"[0.00, 1.00]",0.023,"[0.00, 0.42]",0.024,"[0.01, 0.42]",44
building change,improved multi-class ST-GNN (building),1.0,"[1.00, 1.00]",0.007,"[0.00, 0.16]",0.013,"[0.00, 0.28]",44
building change,repo candidates involving buildings,0.839,"[0.47, 1.00]",0.029,"[0.00, 0.55]",0.055,"[0.01, 0.66]",44
any change,improved v1 node states,0.627,"[0.40, 0.84]",0.06,"[0.03, 0.19]",0.109,"[0.06, 0.29]",47
any change,improved v2 regions + building states,1.0,"[1.00, 1.00]",0.015,"[0.01, 0.05]",0.03,"[0.02, 0.10]",47
any change,improved multi-class ST-GNN,0.927,"[0.32, 0.99]",0.26,"[0.01, 0.79]",0.406,"[0.03, 0.87]",47
any change,repo candidates (non-seasonal),0.898,"[0.63, 1.00]",0.012,"[0.00, 0.05]",0.024,"[0.01, 0.09]",47
any change,repo all events (incl. season-sensitive),0.441,"[0.06, 0.82]",0.756,"[0.23, 1.00]",0.557,"[0.08, 0.85]",47
building change,repo old: candidates involving buildings,1.0,"[1.00, 1.00]",0.01,"[0.00, 0.27]",0.02,"[0.00, 0.43]",44
any change,repo old: candidates (non-seasonal),0.052,"[0.00, 1.00]",0.013,"[0.00, 0.05]",0.02,"[0.00, 0.07]",47
any change,repo old: all events (incl. season-sensitive),0.61,"[0.16, 0.99]",0.752,"[0.22, 1.00]",0.674,"[0.14, 0.98]",47
```

### 49.3 Historical event-count comparison

The count table contrasts repository all/candidate/season-sensitive events with imported v2 regions/building states and older repository outputs on the same named tiles. Counts use different units and criteria; 730 catalogue events and 28 v2 regions do not mean one detector found 702 more true changes. Some fields express candidate area rather than confirmed changed area. No precision/recall can be derived from these counts alone.

Source: `data/external_eval/change_counts_all.csv`.

```csv
tile,repo_events_all,repo_candidates,repo_season_sensitive,repo_candidate_m2,repo_building_candidates,v2_change_regions,v2_region_m2,improved_new_building_high,old_events_all,old_candidates,old_season_sensitive,old_candidate_m2,old_building_candidates
tile_00001,730,313,417,64598,203,28,5702,5,502,249,253,72628,182
tile_00003,708,148,560,21741,118,31,4768,5,497,250,247,83610,170
tile_00006,494,95,399,16327,69,13,1262,3,300,89,211,33934,52
```

### 49.4 Provenance needed before final thesis use

For the OSM tables, obtain the exact export date/version, coverage/completeness assumptions, rasterization/alignment, prediction configurations and instance-assignment rule. An imported frozen OSM-reference script, `pipelines/semantic_temporal/pipeline/validation/osm_reference.py`, shifts OSM masks using correlation with SegFormer building probabilities on a 2× grid; this shows that model-dependent alignment exists in the repository, but it does not establish that this exact procedure produced every CSV above. Verify the producing pipeline before describing alignment as independent.

For the point tables, obtain original point locations/labels, uncertain/unchanged inclusion, sampling, reviewer instructions, source run identities, prediction-to-point mapping and interval code. This is a missing provenance task, not authorization to relabel the tables or invent a human reference.

Read-only source-file identities at this audit snapshot:

| File | Data rows | SHA-256 |
|---|---:|---|
| `buildings_osm_all.csv` | 4 | `6c5def36f2698510298e16f29fe2b3a6054018343fb8261da14ce409c2ead25b` |
| `buildings_osm_per_tile.csv` | 9 | `bb4b9cf8a9555db3ee509f4574aea5fca1eb3f4c47ddc018e5141f2be7ef5d8e` |
| `reference_points_all.csv` | 15 | `5f940593125dce441e56a25abcacee49f309d3a494fcf1886a8c37e50a84620f` |
| `change_counts_all.csv` | 3 | `e6d124659c3109316dd23467e923921283675606795f1ea4fd9d95785b59531c` |

**Arabic note:** توجد مقارنة جزئية مع مباني OpenStreetMap لسنة 2023 فقط، وتظهر أن أداء التقسيم ما زال محدوداً. كما توجد جداول قديمة لنقاط مرجعية فيها نتائج ضعيفة جداً لبعض طرق GNN، لكن بروتوكول إعدادها وتقييمها غير موثق بما يكفي في ملفات CSV وحدها. يجب حفظ هذه النتائج وشرح حدودها والتحقق من مصدرها؛ لا نحذفها ولا نستخدمها كدقة مستقلة نهائية، ولا نخلطها بنتائج النموذج الأصلي على التسميات المولدة آلياً.

