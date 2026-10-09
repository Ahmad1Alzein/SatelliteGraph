# All-class change results

The general pipeline is `pipelines/semantic_temporal/runtime/`. It extends the existing relational GNN encoder with separate gain and loss heads for Buildings, Roads, Trees, Grass/shrubs, Cropland, Bare ground, Paved areas and Water. The original `src` image/folder stages and the frozen `run_experiment.py` remain separate and available.

## What to open

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

Each folder needs a `metadata.csv` with unique image names and original-image offsets:

```csv
tile_name,x0,y0,width,height
tile_00000.png,0,0,512,512
tile_00001.png,462,0,512,512
```

Without an original GeoTIFF, either supply the true physical resolution with `--pixel-size` (one value per year), or add complete `tile_affine_a` through `tile_affine_f` and `target_crs` columns recovered from trustworthy georeferencing. The affine describes each tile's own pixel-to-map transform, including its offset. Never invent coordinates or resolution. Alpha masks define observed image coverage.

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

Use `--reviewed-targets reviewed.csv`; labels are 0 or 1 and tasks are the 16 `<class>_gain`/`<class>_loss` names in `temporal/training_targets.csv`. Reviews cannot override missing coverage or the wrong node class/year. `--change-policy policy.json` accepts validated partial threshold overrides; the resolved policy is retained with the scientific results.

Separate commands inherit saved options. Input/model/registration/harmonization changes require a fresh output folder. A graph-profile change rebuilds graph and change; policy/review/checkpoint changes rebuild change only. Report-budget/style changes rerender verified science. `run.json` distinguishes `scientific_status` and `render_status`; a failed PNG report is `render_failed` while scientific outputs remain available. The full combined report defaults to 128 million pixels and fails with the required size rather than silently dropping events.

Image matching overviews display at most 300 spatially distributed links and disclose omitted counts; full tables remain complete. Component galleries contain up to 12 examples per type. Large/gapped mosaics still need explicit memory budgets. All supported classes and conversions are represented, but noisy segmentation, season, occlusion and uncertain registration limit which real changes can be established.
