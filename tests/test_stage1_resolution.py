import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.io.raster_tiling import prepare_raster_tiles
from src.stage2.tiles import validate_tile
from src.stage2.preprocessing import prepare_image


def source_raster(root, *, crs="EPSG:32636", resolution=0.5, width=18, height=15):
    path = root/"data/raw/scene.tif"
    path.parent.mkdir(parents=True,exist_ok=True)
    transform = from_origin(500000,3700000,resolution,resolution) if crs!="EPSG:4326" else from_origin(35.6,33.8,resolution,resolution)
    with rasterio.open(path,"w",driver="GTiff",count=3,width=width,height=height,dtype="uint8",
                       crs=crs,transform=transform) as dataset:
        dataset.write(np.full((3,height,width),90,dtype=np.uint8))
    return path


class ResolutionTilingTests(unittest.TestCase):
    def test_default_covers_every_pixel_without_padding(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = source_raster(root,width=11,height=5)
            result = prepare_raster_tiles(source,root/"data",tile_size=4,overlap=1)
            with open(result["metadata_path"],newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({int(r["x0"]) for r in rows},{0,3,6,7})
            self.assertEqual({int(r["y0"]) for r in rows},{0,1})
            coverage = np.zeros((5,11),dtype=bool)
            for row in rows:
                image = cv2.imread(str(Path(result["tiles_directory"])/row["tile_name"]),cv2.IMREAD_UNCHANGED)
                self.assertTrue(np.all(image[...,3]==255))
                x,y = int(row["x0"]),int(row["y0"])
                coverage[y:y+4,x:x+4] = True
            self.assertTrue(coverage.all())
            _,metadata,report = validate_tile(root,"scene__tile_00007")
            self.assertTrue(report["valid"])
            self.assertEqual((metadata["global_x0"],metadata["global_y0"]),(7,1))
            self.assertEqual(metadata["padding"],{"top":0,"left":0,"right":0,"bottom":0})
            from scripts.verify_stage1_tiles import verify
            self.assertEqual(verify(root/"data/metadata/scene.stage1.json")["tile_count"],8)

    def test_main_tiling_method_converts_full_raster_before_tiles(self):
        from src.io.tile_image import tile_image
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = source_raster(root,width=27,height=27)
            with rasterio.open(source,"r+") as raster:
                ramp = np.indices((27,27)).sum(axis=0).astype(np.uint8)
                raster.write(np.stack([ramp]*3))
            result = tile_image(source_path=source,data_directory=root/"data",
                                target_resolution=1.5,tile_size=4,overlap=1)
            with rasterio.open(result["source_raster"]) as raster:
                self.assertEqual((raster.width,raster.height),(9,9))
                self.assertEqual(raster.res,(1.5,1.5))
                pixels = raster.read(1)
                self.assertEqual([int(pixels[y,x]) for x,y in [(0,0),(8,0),(0,8),(8,8)]],[2,26,26,50])
            self.assertEqual(result["tile_count"],9)
            from scripts.verify_stage1_tiles import verify
            self.assertEqual(verify(root/"data/metadata/scene_1p5m.stage1.json",all_pixels=True)["pixel_comparison_samples"],9)

    def test_geographic_resize_preserves_full_footprint_without_empty_margins(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = source_raster(root,crs="EPSG:4326",resolution=0.000005,width=60,height=60)
            result = prepare_raster_tiles(source,root/"data",target_resolution=1.5,tile_size=4,overlap=1)
            with rasterio.open(source) as original, rasterio.open(result["source_raster"]) as output:
                self.assertEqual(output.crs,original.crs)
                np.testing.assert_allclose(output.bounds,original.bounds,rtol=0,atol=1e-10)
                self.assertTrue(np.all(output.dataset_mask()==255))
                self.assertTrue(np.all(output.read()==90))

    def test_resample_and_exact_sized_padded_tiles(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = source_raster(root)
            result = prepare_raster_tiles(source,root/"data",target_resolution=1.5,tile_size=4,overlap=1,edge_mode="pad")
            with rasterio.open(result["source_raster"]) as dataset:
                self.assertEqual(dataset.res,(1.5,1.5))
                self.assertEqual((dataset.width,dataset.height),(6,5))
                self.assertEqual(dataset.crs.to_epsg(),32636)
            with open(result["metadata_path"],newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows),4)
            self.assertEqual({(int(r["x0"]),int(r["y0"])) for r in rows},{(0,0),(3,0),(0,3),(3,3)})
            for row in rows:
                image = cv2.imread(str(Path(result["tiles_directory"])/row["tile_name"]),cv2.IMREAD_UNCHANGED)
                self.assertEqual(image.shape,(4,4,4))
            with rasterio.open(source) as original:
                self.assertEqual(original.res,(0.5,0.5))

    def test_padding_mask_survives_stage2_preprocessing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = source_raster(root,width=5,height=5)
            result = prepare_raster_tiles(source,root/"data",tile_size=4,overlap=1,edge_mode="pad")
            image,metadata,_ = validate_tile(root,result["image_name"]+"__tile_00003")
            config = {"rgb_bands":[1,2,3],"normalization":"none","nodata_fill":0,
                      "lower_percentile":2,"upper_percentile":98}
            _,valid,_ = prepare_image(image,metadata,config)
            self.assertEqual(valid.sum(),4)
            self.assertEqual(metadata["original_unpadded_width"],2)
            self.assertEqual(metadata["padding"]["right"],2)

    def test_geographic_source_auto_selects_metric_utm(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = source_raster(root,crs="EPSG:4326",resolution=0.000005)
            result = prepare_raster_tiles(source,root/"data",target_resolution=1.5,tile_size=8,overlap=2,
                                          resolution_mode="projected",edge_mode="pad")
            with rasterio.open(result["source_raster"]) as dataset:
                self.assertEqual(dataset.crs.to_epsg(),32636)
                self.assertEqual(dataset.res,(1.5,1.5))

    def test_preserve_resolution_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = source_raster(root)
            result = prepare_raster_tiles(source,root/"data",tile_size=8,overlap=2)
            self.assertEqual(Path(result["source_raster"]),source)
            with self.assertRaises(FileExistsError):
                prepare_raster_tiles(source,root/"data",tile_size=8,overlap=2)

    def test_invalid_resolution_or_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = source_raster(root)
            for kwargs in [{"target_resolution":-1},{"target_resolution":float("nan")},
                           {"overlap":512},{"target_resolution":1.5,"target_crs":"EPSG:4326"},
                           {"target_resolution":1.5,"target_crs":"EPSG:4326","resolution_mode":"projected"}]:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    prepare_raster_tiles(source,root/"data",**kwargs)

    def test_source_mask_and_valid_black_pixels_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = source_raster(root,width=6,height=6)
            with rasterio.open(source,"r+") as raster:
                pixels = raster.read()
                pixels[:,0:3,0:3] = 0  # valid black is not a nodata sentinel
                raster.write(pixels)
                mask = np.full((6,6),255,dtype=np.uint8)
                mask[3:6,3:6] = 0
                raster.write_mask(mask)
            result = prepare_raster_tiles(source,root/"data",target_resolution=1.5,tile_size=4,overlap=1,edge_mode="pad")
            image = cv2.imread(str(Path(result["tiles_directory"])/"tile_00000.png"),cv2.IMREAD_UNCHANGED)
            self.assertTrue(np.all(image[0,0,:3]==0))
            self.assertEqual(image[0,0,3],255)
            self.assertEqual(image[1,1,3],0)

    def test_old_array_tiling_api_still_works(self):
        import tests.test_stage1 as test_stage1
        with tempfile.TemporaryDirectory() as temp:
            test_stage1.test_stage1_loads_and_tiles_a_tiny_geotiff_in_tmp_path(Path(temp))
        test_stage1.test_stage1_constants_come_from_the_unified_root_config()

    def test_cli_allows_explicit_keep_resolution(self):
        from src.cli import parser
        args = parser().parse_args(["stage1","tile","--source","scene.tif","--keep-resolution"])
        self.assertTrue(args.keep_resolution)
