"""Read-only validation: python -m scripts.verify_stage1_tiles --manifest <path>."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window


def verify(manifest_path, *, all_pixels=False):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest["status"]!="complete":
        raise ValueError("Stage 1 conversion has not completed")
    with open(manifest["metadata_path"],newline="",encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    directory = Path(manifest["tiles_directory"])
    if len(rows)!=manifest["tile_count"] or len(list(directory.glob("tile_*.png")))!=len(rows):
        raise ValueError("Tile counts disagree")
    size,stride = manifest["tile_size"],manifest["stride"]
    samples = set(range(len(rows))) if all_pixels else {0,len(rows)//4,len(rows)//2,3*len(rows)//4,len(rows)-1}
    invalid_count = partial_count = 0
    with rasterio.open(manifest["source_raster"]) as raster:
        xs = manifest.get("grid_x_offsets",[j*stride for j in range(manifest["tile_columns"])])
        ys = manifest.get("grid_y_offsets",[j*stride for j in range(manifest["tile_rows"])])
        if len(rows)!=len(xs)*len(ys):
            raise ValueError("Tiles do not cover the complete Cartesian grid")
        for starts,length in [(xs,raster.width),(ys,raster.height)]:
            if not starts or starts[0]!=0 or starts[-1]+size<length:
                raise ValueError("Image boundary is not covered")
            if any(not 0<b-a<=stride for a,b in zip(starts,starts[1:])):
                raise ValueError("Grid contains gaps or inadequate overlap")
            if manifest.get("edge_mode")=="shift" and starts[-1]+size!=length:
                raise ValueError("Final tile does not meet the image boundary")
        if manifest.get("resolution_mode")=="preserve":
            with rasterio.open(manifest["original_source"]["path"]) as original:
                if raster.crs!=original.crs:
                    raise ValueError("Preserve mode changed the CRS")
                for u,v in [(0,0),(0,1),(1,0),(1,1)]:
                    np.testing.assert_allclose(raster.transform*(u*raster.width,v*raster.height),
                                               original.transform*(u*original.width,v*original.height),rtol=0,atol=1e-8)
        if manifest["target_resolution_m"] is not None and manifest.get("resolution_mode","projected")=="projected":
            if not raster.crs.is_projected or not np.isclose(raster.crs.linear_units_factor[1],1):
                raise ValueError("Output CRS is not in metres")
            np.testing.assert_allclose(raster.res,[manifest["target_resolution_m"]]*2)
        for i,row in enumerate(rows):
            image = cv2.imread(str(directory/row["tile_name"]),cv2.IMREAD_UNCHANGED)
            if image is None or image.shape!=(size,size,4):
                raise ValueError(f"Invalid tile dimensions: {row['tile_name']}")
            x,y,w,h = [int(row[k]) for k in ("x0","y0","valid_width","valid_height")]
            if (x,y)!=(xs[i%len(xs)],ys[i//len(xs)]):
                raise ValueError("Grid overlap/offsets disagree")
            if (w,h)!=(min(size,raster.width-x),min(size,raster.height-y)):
                raise ValueError("Invalid unpadded dimensions")
            valid = image[...,3]>0
            if valid[h:,:].any() or valid[:,w:].any() or valid.sum()!=int(row["valid_pixel_count"]):
                raise ValueError("Padding mask/valid pixel count mismatch")
            invalid_count += int(not valid.any())
            partial_count += int(valid.any() and not valid.all())
            if i in samples:
                window = Window(x,y,w,h)
                expected = raster.read([1,2,3],window=window).transpose(1,2,0)
                np.testing.assert_array_equal(image[:h,:w,:3][...,[2,1,0]],expected)
                np.testing.assert_array_equal(valid[:h,:w],np.all(raster.read_masks([1,2,3],window=window)>0,axis=0))
        return {"status":"verified","tile_count":len(rows),"tile_size":[size,size],
                "overlap_pixels":size-stride,"columns":manifest["tile_columns"],"rows":manifest["tile_rows"],
                "resolution":list(raster.res),"crs":str(raster.crs),"raster_size":[raster.width,raster.height],
                "fully_invalid_tile_count":invalid_count,"pixel_comparison_samples":len(samples),
                "partially_invalid_tile_count":partial_count,"full_raster_covered":True,
                "pixel_size_metres_at_centre":manifest.get("pixel_size_metres_at_centre"),
                "source_raster":manifest["source_raster"],"tiles_directory":str(directory)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",required=True)
    parser.add_argument("--all-pixels",action="store_true",help="Compare every tile's RGB pixels and mask with the saved raster")
    args = parser.parse_args()
    print(json.dumps(verify(args.manifest,all_pixels=args.all_pixels),indent=2))
