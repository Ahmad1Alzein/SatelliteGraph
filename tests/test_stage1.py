from pathlib import Path
import sys

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import *
from src.io.load_tif import load_tif
from src.io.tile_image import tile_image


def test_stage1():

    tif_files = list(RAW_DIR.glob("*.tif"))

    assert len(tif_files) > 0, \
        "No TIFF images found in data/raw"

    tif = tif_files[0]

    print(f"\nLoading {tif.name}")

    image, profile = load_tif(tif)

    print("Image shape:", image.shape)

    tile_image(
        image=image,
        image_name=tif.stem,
        tile_size=TILE_SIZE,
        stride=STRIDE,
        tiles_dir=TILES_DIR,
        metadata_dir=METADATA_DIR,
    )

    metadata = METADATA_DIR / f"{tif.stem}.csv"

    assert metadata.exists(), \
        "Metadata CSV was not generated."

    tiles_folder = TILES_DIR / tif.stem

    assert tiles_folder.exists(), \
        "Tiles folder was not created."

    tiles = list(tiles_folder.glob("*.png"))

    assert len(tiles) > 0, \
        "No image tiles were generated."

    print(f"\n✓ Generated {len(tiles)} tiles")
    print("✓ Metadata file exists")
    print("✓ Stage 1 completed successfully")


if __name__ == "__main__":
    test_stage1()