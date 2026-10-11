# Legacy building spatio-temporal GNN (frozen 2013 → 2023 experiment)

An **optional**, exactly reproducible pipeline for the original building-change GNN of 4 October 2026.
It does not import, replace or modify the multi-class pipeline (`python -m src pipeline …`).

| Folder | Content |
|---|---|
| `original/` | byte-identical copies of the frozen release code (`FROZEN_strip_tile_00000-00006_v1/code`) and the per-tile overlay script (`run_dataset_2013_2023/validation/gnn_predictions_tile_viz.py`). **Never edit**: `reference/code_manifest.json` and the tests check every byte. |
| `reference/` | SHA-256 of the code, of the 53 frozen input files, and of the frozen outputs (GNN data files, summary without timings, decoded pixels of every figure). |
| `runner.py` | stages the verified inputs in the original folder layout, runs the original scripts unchanged in a new run folder, and verifies the outputs. |

See [docs/legacy-building-stgnn.md](../../docs/legacy-building-stgnn.md) for the method, results, commands and limitations.
