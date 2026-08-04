from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from src.config import DEFAULT_CONFIG, OVERLAP, STRIDE, TILE_SIZE, load_config
from src.io.load_tif import load_tif
from src.io.tile_image import tile_image


def test_stage1_loads_and_tiles_a_tiny_geotiff_in_tmp_path(tmp_path: Path) -> None:
    """Exercise Stage 1 without reading or writing the repository data tree."""

    source = tmp_path / "tiny_scene.tif"
    height, width = 5, 7
    bands = np.arange(3 * height * width, dtype=np.uint8).reshape(3, height, width)
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs="EPSG:32633",
        transform=from_origin(500_000, 4_000_000, 10, 10),
    ) as dataset:
        dataset.write(bands)

    image, profile = load_tif(source)
    assert image.shape == (height, width, 3)
    assert np.array_equal(image, bands.transpose(1, 2, 0))
    assert profile["crs"].to_string() == "EPSG:32633"

    tiles_dir = tmp_path / "tiles"
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    tile_image(
        image=image,
        image_name=source.stem,
        tile_size=4,
        stride=3,
        tiles_dir=tiles_dir,
        metadata_dir=metadata_dir,
    )

    rows = pd.read_csv(metadata_dir / "tiny_scene.csv")
    tile_paths = sorted((tiles_dir / "tiny_scene").glob("*.png"))
    assert len(rows) == len(tile_paths) == 6
    assert list(rows.columns) == ["tile_id", "tile_name", "x0", "y0", "x1", "y1", "width", "height"]
    assert set(map(tuple, rows[["x0", "y0"]].to_numpy())) == {
        (0, 0),
        (3, 0),
        (6, 0),
        (0, 3),
        (3, 3),
        (6, 3),
    }
    for row in rows.itertuples(index=False):
        tile = cv2.imread(str(tiles_dir / "tiny_scene" / row.tile_name), cv2.IMREAD_COLOR)
        assert tile is not None
        assert tile.shape[:2] == (row.height, row.width)


def test_stage1_constants_come_from_the_unified_root_config() -> None:
    config = load_config(DEFAULT_CONFIG)
    tiling = config["project"]["tiling"]
    assert DEFAULT_CONFIG.name == "config.yaml"
    assert TILE_SIZE == tiling["tile_size"]
    assert OVERLAP == tiling["overlap"]
    assert STRIDE == tiling["tile_size"] - tiling["overlap"]
