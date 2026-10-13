# All-class change results

> Historical experimental documentation. The enhanced all-class RGNN and its
> checkpoint/training commands below were retired on 2026-10-12. They are retained
> here to document earlier saved results, and are no longer runnable. Current
> choices and commands are in [pipeline options](pipeline-options.md); the
> dashboard now uses the [original thesis building ST-GNN](legacy-building-stgnn.md).

The general pipeline is `pipelines/semantic_temporal/runtime/`. It extends the existing relational GNN encoder with separate gain and loss heads for Buildings, Roads, Trees, Grass/shrubs, Cropland, Bare ground, Paved areas and Water. The original `src` image/folder stages and the frozen `run_experiment.py` remain separate and available.

## What to open

For the current upload-and-review page, run `python -m streamlit run thesis_dashboard.py` from this checkout and open **Run pipelines & results**. Upload a before PNG and an after PNG, then click **Run full comparison**. The page runs the native pipeline, shows progress, and retains segmentation, each graph, matching, scores and downloadable paired-crop grids. **Focus on major changes** is enabled by default; weak/seasonal differences are opt-in.

To run the entire method on one pair with no additional data:

```powershell
$python = 'C:/Users/ahmad/OneDrive/Desktop/Projects/.venv/Scripts/python.exe'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
& $python -m src pipeline run --segmentation new --graph new --change gnn `
  --images 'path/to/before.png' 'path/to/after.png' `
  --output artifacts/my_pair --device cpu --max-report-pixels 256000000 `
  --gnn-checkpoint checkpoints/semantic_temporal/gnn/lsscd_major_pixels/model.pt
& $python -m streamlit run thesis_dashboard.py
```

Automatic registration finds distributed image landmarks and the common visible region. Images can differ in dimensions and resolution; the pixel affine gate supports scale factors 0.25 to 4. Failed/weak registration stops comparison. PNG-only measurements are pixels and pixels squared, never invented metres. This model path expects optical RGB; actual radar/SAR needs a different validated feature/segmentation model.

All paths below are relative to the run's output folder. Start with `results/index.json`; it identifies current scientific stages, report status, task support, and image paths.

| File | Purpose |
|---|---|
| `results/segmentation/<year>/<image>/segmentation.png` | Original RGB next to colored segments, boundaries and class legend; produced by `pipeline segment` |
| `results/graphs/<method>/<year>/<image>/graph.png` | Per-image graph with class and edge legends; produced by `pipeline graph` |
| `results/graphs/<method>/<year>/annual_graph.png` | Owned annual graph across the selected images |
| `results/registration_matches.png` | Actual image landmarks used/rejected in alignment; georeference-only runs explicitly show no feature registration |
| `results/temporal_matches.png` | Main and partial object matches across years |
| `results/temporal_match_types.png` | One-to-one, split, merge, complex and unmatched correspondence |
| `results/class_change_matches.png` | Class conversions, displayed separately from object matches |
| `results/matching/` | Coverage-pair PNGs, component examples, complete match/control/component tables and index |
| `results/segmentation_overview.png`, `results/spatial_graph.png` | Combined stage overviews; independent stage PNGs are retained |
| `results/gnn_predictions.png`, `results/gnn_probabilities.png` | Gain/loss predictions and scores for all eight classes, including unavailable tasks |
| `results/class_transitions.png` | Complete 8-by-8 class transition matrix, including zero counts |
| `results/change_report.png` | One combined image containing paired original-RGB crops for every canonical candidate and additional GNN-only prediction |
| `results/major_changes.png` | Focused grid of substantial changes with strong image evidence or a GNN prediction; explicitly says when none pass |
| `results/changes/change_grid_001.png`, etc. | Readable 12-cell pages retaining every candidate; each cell has local before/after crops, footprint, type, status and available GNN score |
| `results/changes/index.json` | Event IDs, crop bounds, scores/statuses, compound groups, and separate structural observations |

Yellow footprints in the combined report use the same map window and scale in both years. A conversion is counted once, linking its earlier-class loss to its later-class gain. A shape-change group references its existing crops rather than adding duplicate physical events. Empty image areas are excluded from evidence and appear as a checker pattern; they are not a land-cover class in legends.

## Plain-language labels

| Edge label | Meaning |
|---|---|
| Share a border | Two areas touch in the same year |
| Nearby buildings | Nearby building regions, using the existing metric neighborhood rule |
| Nearby areas | Local context among other known-class areas |
| Main match across years | Accepted primary correspondence |
| Partial overlap across years | Weaker overlap retained separately |
| Class change | Derived image-supported conversion; never an input edge carrying the change label into the GNN |

Change labels include **New building**, **Building grew**, **Building shrank**, **Road added**, **Road got wider**, **Road removed**, **Less tree cover**, **Water expanded**, and **Trees → Buildings**. Other classes use Appeared, Disappeared, Expanded, Shrank or Changed shape. These subtypes come from image evidence and correspondence structure; the GNN itself scores class gains/losses. A GNN-only positive is labeled Added/increased or Removed/decreased without inventing a specific subtype.

**Strong image evidence** means a conservative image rule supported the candidate. **Candidate** needs review. **Season may explain this** marks seasonal land-cover conversions. **GNN prediction** identifies a model output. **Unavailable** means the task lacks supported training/inference, not a zero score. Split/merge, connection and neighborhood observations are separate from physical-change counts.

## Registration with images and metadata only

Registration aligns earlier imagery to later imagery using stable landmarks so offsets do not masquerade as change. Metadata places tiles within their original image; it does not by itself establish alignment between years. The mapped-image path reuses the project's phase-correlation-guided SIFT/RANSAC phase, followed by the robust world-coordinate fit. Fitted residuals are diagnostics, not independent alignment accuracy.

For irregular or overlapping tiles, provide a `metadata.csv` with unique image names and original-image offsets:

```csv
tile_name,x0,y0,width,height
tile_00000.png,0,0,512,512
tile_00001.png,462,0,512,512
```

The CSV alone is sufficient for image-relative placement and automatic registration. Without CSV placement, files are naturally sorted (`tile_2` before `tile_10`) and treated as a contiguous left-to-right row; use this fallback only when that describes the tiles. Malformed CSVs and explicitly missing CSV paths fail instead of silently switching placement. A true `--pixel-size` or complete georeferencing is optional, needed only for physical measurements. The affine describes each tile's own pixel-to-map transform, including its offset. Never invent coordinates or resolution. Alpha masks define observed image coverage.

```powershell
$python = '..\.venv\Scripts\python.exe'
& $python -m src pipeline run --segmentation new --graph new --change gnn `
  --folders 'inputs/earlier' 'inputs/later' --years 2013 2023 `
  --pixel-size 1.5 1.5 --register-images --harmonize-images `
  --output artifacts/my_comparison --device cpu
```

Replace `1.5` with the actual metres per pixel. Omit `--pixel-size` when complete metric georeferencing is supplied. With no CRS, landmark matching operates in the physical original-image frame. If automatic matching fails, supply measured `--registration-json`: `moving_xy` and `reference_xy` arrays in the comparison frame, with optional `max_residual_m`. The default automatic fit tolerance is 2 comparison units; `--registration-max-residual` explicitly changes it. A larger tolerance admits less precise landmarks and must be checked in `registration_matches.png` and saved residuals.

## Supplied 2013/2023 samples, native execution

Run from the project root. Preparation verifies all 14 PNG hashes, checks saved 2013 affine/offset records, and verifies 2023 source-window pixels. It writes derived metadata without changing original images/CSVs or fabricating the missing 2013 source raster.

```powershell
$python = '..\.venv\Scripts\python.exe'
$project = (Get-Location).Path
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
& $python scripts/prepare_2013_2023_samples.py --project-root $project `
  --output artifacts/all_class_validation/inputs

$common = @('--years','2013','2023','--metadata',
  'artifacts/all_class_validation/inputs/2013_metadata.csv',
  'artifacts/all_class_validation/inputs/2023_metadata.csv',
  '--device','cpu','--register-images',
  '--harmonize-images','--max-comparison-pixels','8000000')

# One image versus one image; inspect segmentation before continuing.
& $python -m src pipeline segment --method new `
  --images tiles17_23/2013/tile_00002.png tiles17_23/2023/tile_00002.png `
  --output artifacts/all_class_validation/pair_final --registration-max-residual 3 `
  --max-report-pixels 256000000 @common
& $python -m src pipeline graph --method new --output artifacts/all_class_validation/pair_final
& $python -m src pipeline change --method gnn --output artifacts/all_class_validation/pair_final
```

The sample-specific 3-metre tolerance was chosen after measuring landmark support at the original 1.5-metre scale; it is not a universal recommendation. Normal segmentation defaults remain 512-pixel windows, 128-pixel overlap and TTA enabled. No additional models or Python packages are required for this extension in the verified project environment. Actual execution was restricted to this single tile pair at the user's request; the partial full-folder attempt is not a validated result. Folder inputs remain available through `--folders` as shown above.

## Resume, review labels, and saved GNNs

```powershell
& $python -m src pipeline run --segmentation new --graph new --change gnn `
  --output artifacts/all_class_validation/pair_final --resume

# Retry only rendering after a report-budget failure.
& $python -m src pipeline run --segmentation new --graph new --change gnn `
  --output artifacts/all_class_validation/pair_final --resume --max-report-pixels 256000000

# Compatible sixteen-task checkpoint inference on another completed graph.
& $python -m src pipeline change --method gnn --output artifacts/another_run `
  --gnn-checkpoint path/to/a_supported_run/temporal/gnn/model.pt
```

Check that the checkpoint exists: tasks lacking both reliable changed and unchanged labels do not train. Unsupported heads never emit random scores. A legacy scalar building-only checkpoint is not a sixteen-task checkpoint; the legacy runtime profile retains its original scalar behavior. Shared-encoder training is marked `training_only` without enough independent spatial groups. Held-out agreement with image-derived pseudo labels is not human-audited accuracy.

Reviewed CSV rows use `gid,task,label,confidence,source`, for example:

```csv
gid,task,label,confidence,source
COPY_THE_NODE_ID_FROM_training_targets.csv,building_gain,1,1.0,human_review
```

Use `--reviewed-targets reviewed.csv` on a fresh run without `--gnn-checkpoint` to train supported heads; the fixed-checkpoint path performs inference and does not fine-tune external weights. The page disables its train button on fixed-checkpoint runs. Labels are 0 or 1 and tasks are the 16 `<class>_gain`/`<class>_loss` names in `temporal/training_targets.csv`. Reviews cannot override missing coverage or the wrong node class/year. `--change-policy policy.json` accepts validated partial threshold overrides; the resolved policy is retained with the scientific results.

Separate commands inherit saved options. Input/model/registration/harmonization changes require a fresh output folder. A graph-profile change rebuilds graph and change; policy/review/checkpoint changes rebuild change only. Report-budget/style changes rerender verified science. `run.json` distinguishes `scientific_status` and `render_status`; a failed PNG report is `render_failed` while scientific outputs remain available. The full combined report defaults to 128 million pixels and fails with the required size rather than silently dropping events.

Image matching overviews display at most 300 spatially distributed links and disclose omitted counts; full tables remain complete. Component galleries contain up to 12 examples per type. Large/gapped mosaics still need explicit memory budgets. All supported classes and conversions are represented, but noisy segmentation, season, occlusion and uncertain registration limit which real changes can be established.

## External annotation-supervised GNN training

The optional external training method uses [LsSCD-Ex](https://doi.org/10.5281/zenodo.17765755), whose eight semantic classes match the project's class system. Its 2013/2015 optical imagery has 0.6-metre source resolution and professionally annotated semantic masks. It is **CC BY-NC-SA 4.0, academic use only**, with Google Earth terms applying. Data and derived weights retain that provenance; they are not unrestricted commercial assets.

```powershell
& $python scripts/download_lsscd_subset.py --output data/external_training/lsscd_subset.zip
& $python -u -m pipelines.semantic_temporal.runtime.external_training `
  --archive data/external_training/lsscd_subset.zip `
  --selection-manifest data/external_training/lsscd_subset.json `
  --output artifacts/external_training/lsscd_major_pixels --epochs 200

# Use the generated sixteen-task checkpoint on a fresh optical PNG pair.
& $python -m src pipeline run --segmentation new --graph new --change gnn `
  --images before.png after.png --output artifacts/supervised_pair --device cpu `
  --gnn-checkpoint artifacts/external_training/lsscd_major_pixels/model.pt
```

The downloader samples 24 training, 10 validation and 10 test parent scenes with a recorded seed, one 256-pixel chip per parent. It preserves the official train/test split, verifies each downloaded member against the official ZIP CRC, and records local SHA-256 hashes. A bounded download does not claim to verify the full archive's MD5. For a complete `256.zip` download, omit `--selection-manifest`; its official MD5 is checked and selection balances annotation-class coverage.

Training runs the same full segmentation and graph methods used for inference. A positive/unchanged annotated target is accepted when its predicted region agrees with the reference class over at least 80% of its observed footprint. The optional `--include-false-class-negatives` policy also teaches rejection of false objects; see the current model section below. At least 70% class replacement is a positive; at most 5% is unchanged; intermediate size variations are masked. Normalization uses training nodes only. Validation chooses the model and per-task thresholds; the official test parents stay outside fitting. A deployment head needs training positives/negatives and validation precision of at least 0.85 with at least three correctly predicted validation positives. Missing support yields no score.

Read `training_report.json` for actual per-head support and held-out node metrics. These are selected-subset results on annotation-agreeing predicted nodes, not the full dataset benchmark, segmentation accuracy, calibrated probabilities or accuracy on your 2013/2023 imagery. External 0.6-metre imagery differs from your imagery; transfer still needs inspection. This checkpoint uses pixel-unit graphs and will correctly reject a metre-unit graph. No extra Python packages are needed beyond the project's installed segmentation/graph environment; the review page adds Streamlit from `requirements.txt`.

Externally supervised inference can score a known-class node with at least 95% shared coverage and sufficient current-image segmentation confidence even when the other image's segmentation is uncertain. Such scores remain model predictions. This separate inference policy does not relax pseudo-label training, review-label eligibility, class/year restrictions or missing-image coverage checks.

Small registration errors are screened using the measured 95th-percentile landmark residual by default; `--boundary-tolerance` overrides this. The major-change view rejects thin boundaries, building size/outline changes and vegetation subclass switches. It preserves all raw candidates in the scientific output and full grid pages. Native crop RGB follows exactly the same overlapping-tile ownership as scientific evidence, with a maximum 2x display enlargement. An intrinsically blurred source image remains blurred.


## Graph reasoning and quality checks

The decision model remains the project's relational GNN. Segmentation supplies region nodes; spatial edges supply neighboring context; directed temporal edges supply primary and partial correspondences, including observed overlaps between different classes. Derived change labels are excluded from model input edges. A checkpoint with `message_mode: temporal_difference.v1` also compares signed and absolute endpoint features, learns temporal edge gates, and preserves match presence/counts. The `legacy` mode retains the previous architecture and checkpoint behavior. The original frozen study is unchanged.

The stronger annotation-selected training set contains **104 image pairs: 60 training, 26 validation, 18 test**, with disjoint parent scenes and the original test selection held fixed. Added examples require substantial changed interiors and at least 80% whole annotated component replacement, reducing the emphasis on changed outlines. The complete official `256.zip` archive was downloaded and its official MD5 verified before this selection; each selected member also has recorded integrity checks. This is a selected challenge subset, not the full dataset benchmark.

The initial 72-pair run failed important quality checks: its building/road heads were unsupported, and grass heads failed held-out detection. It was not published as a reliable checkpoint. Increasing training epochs does not solve missing or inaccurate segmentations. Its held-out segmentation mean IoU was 42.3%, measured separately from change scores.

The graph comparison fits the same splits with full legacy RGNN, full temporal-difference RGNN, no edges, and spatial edges only. Model selection uses validation macro average precision, with validation loss as a tie-breaker; test metrics never choose the winner. Only full graph variants can become deployment checkpoints. Diagnostic edge-ablated models retain metrics rather than deployable weights, preventing ordinary inference from silently restoring their missing edges. Per-task thresholds still require validation precision>=0.85 and at least 3 true positives.

For a reproducible full segmentation/graph training run using the recorded stronger selection in this checkout:

```powershell
& $python -u -m pipelines.semantic_temporal.runtime.external_training `
  --archive data/external_training/lsscd_strong.zip `
  --selection-manifest data/external_training/lsscd_strong.json `
  --output artifacts/external_training/my_strong_run --epochs 200 `
  --message-mode temporal_difference.v1 --selection-metric macro_ap `
  --include-false-class-negatives
```

This command trains the explicitly requested architecture; the four-model comparison's selected architecture is recorded separately in `training_report.json`. External scores are uncalibrated model scores, not confidence that construction actually occurred. Thresholds differ by task. Strong validation performance can still fail on a different satellite/view or season; unsupported heads remain unavailable. Full-node GNN proposals highlight the scored region, while thin raw segmentation differences remain separate candidates.


## Earlier conditional baseline ? retained for comparison

The earlier `lsscd_strong_graph` package trained/evaluated only annotation-agreeing predicted regions. Its validation macro AP was 0.807 for temporal-difference RGNN, 0.764 for legacy RGNN, 0.643 for spatial-only and 0.572 for no edges. Its three-RGNN package enabled 11 directions. Those conditional scores excluded false-class segmentation regions. Inspection of the supplied pair found an obvious false building-loss prediction, so this package is not the current default and must not be described as dependable. Its artifacts remain available for reproducibility.

The earlier base's fixed-test macro AP was 0.8669 on eight adequately supported tasks, compared with 0.5290 spatial-only and 0.4109 no-edges. These ranking metrics are neither detection accuracy nor target-image accuracy. They must not be compared directly with the broader evaluation population below. The original conditional report is `artifacts/external_training/lsscd_strong_graph/graph_contribution.json`.

## Current model: training includes segmentation mistakes

The revised policy retains every previously accepted target, and adds a negative for a predicted known-class region when at most 5% of its current footprint has that reference class. It requires at least 95% observed coverage and 95% known-class reference coverage. Unknown/mixed reference regions stay masked. The region's own gain/loss task is negative; absent graph nodes are never invented. This applies to **all eight classes and both directions**.

On the same 104 verified image pairs and unchanged parent splits/features, this added **3,456 negative targets**. The four-model comparison and optional building diagnostic were retrained from scratch. The GNN still decides through the full spatial/temporal graph. Better rejection training does not repair segmentation masks, recover missed regions, or establish robustness to every satellite/view.

Current artifacts are in `artifacts/external_training/lsscd_robust_graph/`: `training_report.json`, `hard_negative_counts.json`, `graph_contribution.json` and `validated_task_experts/training_report.json`. Scores describe annotation-assessed predicted regions including false-class negatives, with ambiguous regions still excluded. This is a different evaluation population from the earlier conditional baseline. Validation support is separate from held-out performance; accuracy on the supplied imagery remains unmeasured.

The revised all-class comparison gives:

| Model | Validation macro AP | Fixed-test macro AP |
|---|---:|---:|
| Full temporal-difference RGNN | 0.415 | 0.559 |
| Full legacy RGNN | 0.400 | 0.439 |
| Spatial edges only | 0.270 | 0.424 |
| No edges | 0.338 | 0.304 |

Validation averages 14 adequately supported tasks; the fixed test averages 10 with at least three positives and three negatives. Test scores cover 1,264 graph nodes from 18 chips and 15 parent scenes, with task-specific masks. A 500-resample paired-parent bootstrap gives full-graph minus no-edges AP difference 0.256, interval [0.136, 0.340], and minus spatial-only 0.135, interval [0.024, 0.206]. One seed, small selected subsets, imperfect labels/segmentation and domain shift remain outside these intervals. The table compares complete base models, not the two-model task-expert package.

The packaged model contains two complete RGNNs: the enhanced base and a legacy grass-gain expert selected on class-level validation AP. No building-only diagnostic model qualified for routing. All 16 tasks have training targets; only these directions pass the validation gate:

| Class | Validation-supported directions | Independent test limitation |
|---|---|---|
| Buildings | None | No deployable threshold; retain review candidates |
| Roads | None | No deployable threshold; retain review candidates |
| Trees | None | No deployable threshold; retain review candidates |
| Grass/shrubs | Gain | Precision/recall 0/0 on 86 assessed nodes: **failed** |
| Cropland | Gain | No positive test examples: unverified |
| Bare ground | Gain, loss | Gain precision 0.333, recall 0.077; loss precision 1.0, recall 0.667 on only 11 nodes |
| Paved areas | None | No deployable threshold; retain review candidates |
| Water | None | No deployable threshold; retain review candidates |

**Validation support is not dependable detection.** This package is experimental; grass gain fails held-out detection, bare-ground gain is weak, and cropland gain has insufficient test support. The graph helps rank some changes, but these results do not meet the requested professional reliability across all classes. Training longer on the same small subset cannot establish that reliability. The page retains task statuses, test metrics, raw candidates and filters rather than claiming confirmed construction, demolition or land change.

The current installed checkpoint is `checkpoints/semantic_temporal/gnn/lsscd_major_pixels/model.pt` with its required sibling `model.json`. Its metadata records all 16 task statuses, exact model settings, training provenance, state hash, license and test limitations. Unavailable directions produce no GNN score; the page still shows their segmentation, graph and image-evidence candidates.

```powershell
& $python -m src pipeline run --segmentation new --graph new --change gnn `
  --images 'path/to/before.png' 'path/to/after.png' `
  --output artifacts/my_graph_comparison --device cpu --max-report-pixels 256000000 `
  --gnn-checkpoint checkpoints/semantic_temporal/gnn/lsscd_major_pixels/model.pt

& $python -m streamlit run thesis_dashboard.py
```

No additional checkpoint is loaded from source experiment directories during inference. The training command above reproduces the enhanced full-graph variant with the new negative policy; it does not itself execute the four-model comparison or select task experts. All raw candidates remain available separately from focused model-prediction grids. The source imagery's blur, occlusion, seasons and failed segmentations remain material limits.

Changing nested GNN weights now invalidates only the change stage, not segmentation/graphs. Runs created under the older broad model-directory fingerprint can require one fresh output; do not edit sealed contexts or bypass integrity checks. The latest supplied bare-image run is `artifacts/all_class_validation/image_pair_graph_review/`.


Latest native sample verification completed on 2026-10-10: 392/364 before/after region nodes, including 139/168 building regions; these are segmentation regions, not verified building counts. The revised checkpoint scored 25 eligible grass-gain node/tasks with zero positives. No strong major changes passed. All 284 raw candidates (63 candidates, 221 season-sensitive) remain in 24 paired-grid pages and the complete combined PNG. This does not establish absence of physical change. Full regression: 604 tests plus 114 subtests passed. Streamlit AppTest verified all five tabs and class/score filters. See [engineering decisions and verification](all-class-engineering-record.md).
