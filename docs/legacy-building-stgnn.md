# Legacy building spatio-temporal GNN (optional pipeline)

The original building-change GNN of the 2013 → 2023 Shouf study (7 tiles, frozen 4 October 2026), integrated as an
**optional, exactly reproducible** pipeline. It is kept separate from the multi-class pipeline and from any future
improved model: its own command, package (`pipelines/legacy_building_stgnn/`), input bundle
(`data/legacy_building_stgnn/`) and output folders (`artifacts/legacy_building_stgnn/<run>/`).

## Method (unchanged)
- **Graph:** one strip spatio-temporal graph of the 7 tiles: 7,020 region nodes (2013 and 2023), 27,056 edges.
  Spatial relations: `spatial_touches`, `spatial_near` (both directions). Temporal relations: primary
  (`temporal_identity` + `temporal_contained_in`) and `temporal_overlap`, forward and backward. Edge attributes
  (10 dims) enter every message.
- **Model:** relational GNN (R-GCN style), 2 layers, 64 hidden, dropout 0.3:
  `h_i' = ReLU(W0 h_i + Σ_r mean_{j∈N_r(i)} (W_r h_j + U_r e_ij))`, binary head.
- **Targets:**
  - **positive:** 2023 building nodes that are new / extended with high confidence, and 2013 land nodes built over;
  - **negative:** persistent / segmentation difference / no evidence;
  - **masked:** everything else.

  These are pseudo-labels from the audited rule-based change layer. 3,851 labelled nodes, 125 positives.
- **Training:**
  - **Evaluation scheme:** leave-one-tile-out (7 folds). The validation tile is the next tile, used for early
    stopping (validation AP every 5 epochs) and the threshold (F1-optimal on validation).
  - **Optimisation:** Adam (lr 5e-3, weight decay 5e-4), 300 epochs, positive class weighting, seed 0.
  - **Determinism:** CPU, deterministic algorithms.
- **Ablations:** MLP (no edges), GNN with spatial edges only, GNN with spatial + temporal edges.
- **No checkpoint:** a model is retrained for each held-out tile, and weights are not stored. Reproduction means
  retraining, which is bit-exact on CPU.

## Results (held-out, pooled over the 7 test tiles)
| Model | AP | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| MLP (node features only) | 0.084 | 0.734 | 0.161 | 0.28 | 0.204 |
| GNN, spatial edges only | 0.100 | 0.710 | 0.148 | 0.24 | 0.183 |
| **GNN, spatial + temporal** | **0.530** | **0.906** | **0.507** | **0.60** | **0.549** |

The earlier visual audit of high-confidence building changes: the GNN detects 33 of 43 "real", 10 of 13
"questionable" and 0 of 2 "unclear".

**Evidence type:**
- **What the scores measure:** agreement with rule-based pseudo-labels, not accuracy against independent ground
  truth.
- **The audit:** the audit verdicts are visual judgements, not an independent human reference.

These numbers must not be mixed with the multi-class pipeline's metrics, or with future improved models.

## Figures
- `figures/fig5c_gnn_change_map.png`: **primary thesis figure**. The complete 7-tile strip.
  - **Top band:** 2013 land built over.
  - **Bottom band:** 2023 new / extended buildings.
  - **Colours:** blue = GNN and rules agree, orange = GNN only, green = rules only.
- `figures/<tile>_predictions_overlay.png`: per-object view of one held-out tile, 2013 | 2023.
  - **Fills:** predicted nodes with P(change).
  - **Thin outlines:** the other year's predictions.
  - **Dashed:** rule changes the GNN missed.

  The full set per tile is in `work/gnn_viz_<tile>/` (probability map, galleries, `predicted_nodes.csv`).
- `figures/fig5a_gnn_architecture.png`, `figures/fig5b_gnn_results.png`: architecture and ablation bar chart.

## Commands
Run from the repository root. The interpreter needs GDAL (`osgeo`) and PyTorch. The reference reproduction used
Python 3.11 and torch 2.11.0+cu128 (CPU), numpy 2.3.5, pandas 3.0.6.

```powershell
# 1. once: copy and hash-verify the frozen inputs (77 MB, git-ignored)
python -m src legacy-building-gnn bundle --source D:\new\DelAnyV2_Experiment\segmentation_v2 `
    --metadata D:\new\DelAnyV2_Experiment\dataset\2023\metadata.csv

# 2. full reproduction: retrain leave-one-tile-out (about 12 min on CPU), fig5a/b/c, overlays for all 7 tiles, verification
python -m src legacy-building-gnn run --output artifacts/legacy_building_stgnn/<run-name>

#    figures only, from the frozen predictions (about 2 min); choose tiles
python -m src legacy-building-gnn run --output artifacts/legacy_building_stgnn/<run-name> --reuse-frozen-predictions --tiles tile_00002 tile_00005

# 3. re-check any run folder against the frozen originals
python -m src legacy-building-gnn verify --output artifacts/legacy_building_stgnn/<run-name>

# tests (the end-to-end figure test runs when the bundle exists)
python scripts/run_project_tests.py tests/test_legacy_building_stgnn.py
```

Each run writes `report.json`:
- **Recorded:** mode, evidence type, metrics, environment.
- **Verification:**
  - **GNN data files:** byte-identical to the frozen originals;
  - **`summary.json`:** identical except run timings;
  - **figures:** decoded pixels identical.

  The frozen originals exist for fig5a/b/c and tile_00002. For the other six tiles no overlays were ever produced
  originally, so their reference is the first reproduction (2026-10-11), and the report labels it that way.

## Limitations
- **Pseudo-labels:** labels come from the rule layer. High scores can partly reflect learning those rules, and recall
  of real changes the rules missed is unknown.
- **Scope:** buildings only (new / extended in 2023, built-over land in 2013). Demolition is not a target.
- **Reproduction:** needs the frozen strip graph. New imagery requires the full original pipeline
  (`pipelines/semantic_temporal/run_experiment.py`).
- **Resolution and environment:** 1.5 m source imagery. Bit-exact reproduction is only guaranteed on CPU in an
  equivalent PyTorch / numpy environment; `verify` reports any difference.
