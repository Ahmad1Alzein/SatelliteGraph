from pathlib import Path

from config import *

from io.load_tif import load_tif

from io.tile_image import tile_image


def main():

    tif_files = list(RAW_DIR.glob("*.tif"))

    if len(tif_files) == 0:

        print("No TIFF images found.")

        return

    for tif in tif_files:

        print()

        print("Loading:", tif.name)

        image, profile = load_tif(tif)

        print("Image shape:", image.shape)

        tile_image(

            image=image,

            image_name=tif.stem,

            tile_size=TILE_SIZE,

            stride=STRIDE,

            tiles_dir=TILES_DIR,

            metadata_dir=METADATA_DIR

        )


if __name__ == "__main__":

    main()