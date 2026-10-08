# Selectable pipeline: segmentation → graph → change detection

`python -m src pipeline …` runs either of the two pipelines in this repository, one stage at a time or end to
end:

| Stage | `old` (SatelliteGraph stages 2–5) | `new` (`pipelines/semantic_temporal`) |
|---|---|---|
| Segmentation | SAM 2.1 Hiera-L + OpenEarthMap FasterSeg, Stage 2R automatic refinement (1.5 m) | multi-scale SegFormer + Delineate Anything v2 + RGB rules, frozen releases with checks C1–C15 (0.75 m) |
| Graph | Stage 3 tile graphs + Stage 4 combined year graph (4-connected adjacency) | region graphs (`touches` + building `near` edges, 37 node / 9 edge features), temporal layer, spatio-temporal graph, 7-tile strip |
| Change: `classical` | Stage 5 temporal graph and change events | rule-based change labels on the spatio-temporal graph (the frozen 2013–2023 baseline) |
| Change: `gnn` | not available | relational GNN, leave-one-tile-out, deterministic CPU training |

## Supported combinations

Only native combinations run. Everything else is refused before any work starts.

| Segmentation | Graph | Change | |
|---|---|---|---|
| old | old | classical | SatelliteGraph end to end |
| new | new | classical | thesis pipeline up to the rule-based change labels |
| new | new | gnn | full thesis pipeline (the GNN's pseudo-labels are the classical change labels, so `gnn` runs `classical` first) |

The following combinations are **refused**:
- **old ↔ new mixes:** each graph builder reads its own segmentation format. Old maps are 1.5 m, with `unknown` and multi-part objects. New releases are validated 0.75 m single-region segments. A mix would need a format adapter, which doesn't exist yet.
- **old graph + gnn:** the old graph has no node or edge feature tensors, and the GNN labels come from the new temporal layer.

## Inputs

**new**
- Default folders, relative to the repository and git-ignored: `data/experiment_2013_2023/` (`dataset/2013`, `dataset/2023`, `aligned/`) and `checkpoints/experiment_2013_2023/` (`DelineateAnythingv2.pt`, `satlens-segformer/`). Use `--data` and `--checkpoints` to point elsewhere.
- Every file is checked against `pipelines/semantic_temporal/reference/input_manifest.json` before the run (24 files). Use `--skip-input-check` for other data. The final verification then reports differences from the thesis results, which is expected for other data.

**old**
- `--folders EARLIER LATER`: two tile folders, each with PNG tiles, `metadata.csv` and the source GeoTIFF, as for `folder run`.
- For the classical stage, also `--registration-json` or `--register-images`.

## Environment

```
conda env create -f pipelines\experiment_2013_2023\environment\environment_windows_exact.yml
conda activate repro-2013-2023
pip install -r requirements.txt          # old pipeline: SAM 2 (vendored), hydra, pycocotools, easydict, pytest
```

The exact environment reproduces the thesis results byte for byte on the original hardware (see
`pipelines/semantic_temporal/README.md`, section 5).

## Commands

End to end:

```
python -m src pipeline run --segmentation new --graph new --change gnn       --output artifacts\runs\new_new_gnn
python -m src pipeline run --segmentation new --graph new --change classical --output artifacts\runs\new_new_classical
python -m src pipeline run --segmentation old --graph old --change classical --output artifacts\runs\old_old_classical ^
       --folders tiles17_23\2013 tiles17_23\2023 --register-images
```

Stage by stage (the same `--output` folder each time; every stage checks that the previous one completed):

```
python -m src pipeline segment --method new       --output artifacts\runs\exp1
python -m src pipeline graph   --method new       --output artifacts\runs\exp1
python -m src pipeline change  --method classical --output artifacts\runs\exp1
python -m src pipeline change  --method gnn       --output artifacts\runs\exp1
```

Check a new → new → gnn run against the thesis results at any time:

```
python pipelines\experiment_2013_2023\verify_outputs.py artifacts\runs\new_new_gnn\work
```

## Outputs (one folder per run, under the git-ignored `artifacts/`)

| Path | Content |
|---|---|
| `<run>/run.json` | selected methods, per-stage status (`running` / `complete` / `failed` + error), options, git commit |
| **old** `<run>/annual_<year>/` | `stage2/` (segmentation, Stage 2R), `stage3/` + `stage4/` (graphs), overlays, `run.json` |
| **old** `<run>/classical_old/` | Stage 5 temporal graph, `changes.geojson`, `report.json` |
| **new** `<run>/work/final_segmentation/` | frozen segmentation releases `FROZEN_<year>_<tile>_<v>` |
| **new** `<run>/work/graphs/` | tile graphs, frozen temporal layers and spatio-temporal graphs, `change_stgraph_2013_2023_<tile>/` (classical change labels), `FROZEN_strip_tile_00000-00006_v1/` (strip mosaic, strip spatio-temporal graph, `gnn/`) |
| **new** `<run>/work/thesis_figures/`, `gnn_viz_tile_00002/` | figures (after the gnn stage) |
| **new** `<run>/work/REPRODUCTION_REPORT.md`, `logs/` | verification against the thesis results (partial after an early stage), run log, stage ledger |

The new pipeline's stage groups run all tiles per group: segmentation, then graph, then classical, then gnn.
Every step reads only the frozen outputs of the previous groups, so a staged run gives byte-identical results to
the original one-pass run.
