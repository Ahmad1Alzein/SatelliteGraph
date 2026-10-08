from pathlib import Path
import csv
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import from_origin


def make_year(root, year=2017, *, names=('a.png',), offsets=((3, 2),), shape=(12, 18), alpha=False):
    folder = Path(root) / str(year); folder.mkdir(parents=True)
    h, w = shape
    full = np.zeros((64, 96, 3), np.uint8)
    full[:, :, 0] = np.arange(96, dtype=np.uint8)
    full[:, :, 1] = np.arange(64, dtype=np.uint8)[:, None]
    full[:, :, 2] = 80
    with rasterio.open(folder/'source.tif', 'w', driver='GTiff', width=96, height=64,
                       count=3, dtype='uint8', crs='EPSG:32636', transform=from_origin(500000, 3700000, 1.5, 1.5)) as dst:
        dst.write(full.transpose(2, 0, 1))
    rows = []
    for name, (x, y) in zip(names, offsets):
        rgb = full[y:y+h, x:x+w].copy()
        if alpha:
            a = np.full((h, w, 1), 255, np.uint8); a[:2] = 0
            rgb = np.concatenate([rgb, a], axis=2)
        Image.fromarray(rgb).save(folder/name)
        rows.append(dict(tile_name=name, x0=x, y0=y, width=w, height=h))
    with (folder/'metadata.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return folder
