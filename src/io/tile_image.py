import cv2
import pandas as pd
from pathlib import Path


def tile_image(
        image=None,
        image_name=None,
        tile_size=512,
        stride=None,
        tiles_dir=None,
        metadata_dir=None,
        *, source_path=None, data_directory=None, target_resolution=None,
        target_crs=None, overlap=None, resampling="average", progress=False,
        resolution_mode="preserve", edge_mode="shift",
):
    """Optionally resize an entire GeoTIFF, save it, then tile the saved raster.

    Use source_path/data_directory for georeferenced processing. Omitting
    target_resolution keeps the original pixels. Legacy array calls remain
    supported, but cannot request metre resolution without georeferencing.
    """
    if source_path is not None or isinstance(image,(str,Path)):
        from src.io.raster_tiling import prepare_raster_tiles
        if source_path is not None and image is not None:
            raise ValueError("Provide either image or source_path")
        if data_directory is None:
            raise ValueError("GeoTIFF tiling requires data_directory")
        if tiles_dir is not None or metadata_dir is not None:
            raise ValueError("Use data_directory for GeoTIFF outputs")
        if overlap is None:
            overlap = tile_size-stride if stride is not None else 50
        elif stride is not None and stride!=tile_size-overlap:
            raise ValueError("stride and overlap disagree")
        return prepare_raster_tiles(source_path if source_path is not None else image,data_directory,
            image_name=image_name,target_resolution=target_resolution,target_crs=target_crs,
            tile_size=tile_size,overlap=overlap,resampling=resampling,progress=progress,
            resolution_mode=resolution_mode,edge_mode=edge_mode)
    if target_resolution is not None or target_crs is not None:
        raise ValueError("Metre resolution conversion requires a georeferenced source_path")
    if image is None or image_name is None or tiles_dir is None or metadata_dir is None:
        raise ValueError("Array tiling requires image, image_name, tiles_dir and metadata_dir")
    if stride is None:
        stride = tile_size-(50 if overlap is None else overlap)
    if stride<=0 or tile_size<=0:
        raise ValueError("tile_size and stride must be positive")

    tiles_path = Path(tiles_dir) / image_name
    tiles_path.mkdir(parents=True, exist_ok=True)
    Path(metadata_dir).mkdir(parents=True,exist_ok=True)

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
