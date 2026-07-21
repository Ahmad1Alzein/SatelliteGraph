import cv2
import pandas as pd
from pathlib import Path


def tile_image(
        image,
        image_name,
        tile_size,
        stride,
        tiles_dir,
        metadata_dir
):

    tiles_path = Path(tiles_dir) / image_name
    tiles_path.mkdir(parents=True, exist_ok=True)

    H, W = image.shape[:2]

    metadata = []

    tile_id = 0

    for y in range(0, H, stride):

        for x in range(0, W, stride):

            x_end = min(x + tile_size, W)
            y_end = min(y + tile_size, H)

            tile = image[y:y_end, x:x_end]

            tile_name = f"tile_{tile_id:05d}.png"

            cv2.imwrite(
                str(tiles_path / tile_name),
                cv2.cvtColor(tile, cv2.COLOR_RGB2BGR)
            )

            metadata.append({

                "tile_id": tile_id,

                "tile_name": tile_name,

                "x0": x,

                "y0": y,

                "x1": x_end,

                "y1": y_end,

                "width": x_end - x,

                "height": y_end - y

            })

            tile_id += 1

    df = pd.DataFrame(metadata)

    df.to_csv(

        Path(metadata_dir) / f"{image_name}.csv",

        index=False

    )

    print(f"Generated {tile_id} tiles.")