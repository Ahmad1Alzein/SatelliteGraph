# SatelliteGraph

SatelliteGraph turns an existing Stage 1 image tile into one complete semantic
instance segmentation. Stage 2 runs two models in one pipeline:

- **SAM 2.1 Hiera-Large** proposes detailed, class-agnostic segment geometry.
- **OpenEarthMap FasterSeg** supplies the named land-cover labels.
- Deterministic fusion resolves overlaps, fills uncovered valid pixels, and uses
  `unknown` when no OpenEarthMap class is confident enough.

The result is complete for one tile: every valid pixel belongs to exactly one
instance and has exactly one semantic value. Neighboring tiles are not stitched
or reconciled in Stage 2; cross-tile merging belongs to a later project stage.

See [Stage 2 full segmentation](docs/stage2-segmentation.md) for the complete
input, provenance, coordinate, and artifact contract.

## Setup

The repository has one dependency file and one project configuration file:

```powershell
python -m pip install -r requirements.txt
```

Edit `config.yaml` to configure project paths, both models, preprocessing,
filtering, fusion, visualization, and output. There is no separate Stage 2 or
landscape configuration and no Streamlit application.

The configured model assets are local and are not downloaded while the pipeline
runs:

```text
checkpoints/sam2/sam2.1_hiera_large.pt
checkpoints/openearthmap/fasterseg/arch_1.pt
checkpoints/openearthmap/fasterseg/weights1.pt
```

## Where to put the image

Put the source GeoTIFF at `data/raw/<source>.tif`. This location is controlled
by `project.paths.raw_directory` in `config.yaml`. Stage 1 prepares the two
artifacts that Stage 2 actually consumes:

```text
data/tiles/<source>/tile_XXXXX.png
data/metadata/<source>.csv
```

Run Stage 2 with the canonical tile ID `<source>__tile_XXXXX`; `run_stage2` and
the CLI take a `tile_id`, not an image path.

## Run one tile

From Python, use the single public runner:

```python
from src.stage2 import run_stage2

result = run_stage2("scene__tile_00000")
print(result.output_directory)
print(result.segments)
```

Or use the single full-pipeline CLI command:

```powershell
python -m src stage2 run --tile-id scene__tile_00000
```

The runner shows a `tqdm` phase progress bar. Add `--no-progress` to the CLI
command when progress output is not wanted.

Both forms use `config.yaml` by default and write the final product to
`artifacts/stage2/<tile_id>/12_final/`. The returned `Stage2Result` includes the
individual masks and records, instance and semantic maps, semantic confidence,
per-pixel provenance, the valid-data mask, normalized tile metadata, and the
artifact paths.

The final records preserve tile-local pixels, source/global pixels, projected
coordinates, and WGS84 coordinates when the source raster contains usable
georeferencing. GeoTIFF and GeoJSON exports are included alongside the NumPy and
JSON artifacts.
