"""The folder commands must retain their geospatial and completion contracts."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin
import yaml

from src.cli import parser
from src.config import DEFAULT_CONFIG, load_config


class FolderPipelineTests(unittest.TestCase):
    def test_cli_accepts_folder_and_two_completed_results(self):
        annual = parser().parse_args(["folder", "run", "--folder", "tiles17_23/2017",
                                      "--year", "2017", "--output", "artifacts/annual_2017"])
        self.assertEqual(annual.year, 2017)
        self.assertEqual(annual.folder, "tiles17_23/2017")
        temporal = parser().parse_args(["folder", "compare", "--first", "a", "--second", "b",
                                        "--output", "c", "--registration-json", "control.json"])
        self.assertEqual(temporal.registration_json, "control.json")

    def test_cli_accepts_one_png_as_the_image_input(self):
        args = parser().parse_args(["image", "run", "--image", "tiles17_23/2017/tile_00000.png",
                                    "--year", "2017", "--output", "artifacts/one_tile"])
        self.assertEqual(args.image, "tiles17_23/2017/tile_00000.png")
        self.assertEqual(args.year, 2017)

    def test_image_run_processes_only_named_png_from_multitile_folder(self):
        from src.folder_pipeline import run_image
        from src.stage4 import load_stage4_result
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root/"2017"
            folder.mkdir()
            cfg = load_config(DEFAULT_CONFIG)
            cfg["stage2"]["instance_model"]["backend"] = "fake"
            cfg["stage2"]["semantic_model"]["backend"] = "fake"
            yy, xx = np.indices((64, 128))
            image = np.empty((64, 128, 4), dtype=np.uint8)
            image[:, :, 0] = xx + 40
            image[:, :, 1] = yy + 40
            image[:, :, 2] = xx + yy + 40
            image[:, :, 3] = 255
            with rasterio.open(folder/"source.tif", "w", driver="GTiff", width=128, height=64,
                               count=3, dtype="uint8", crs="EPSG:32636",
                               transform=from_origin(500000, 3700000, 1.5, 1.5)) as output:
                output.write(np.moveaxis(cv2.cvtColor(image, cv2.COLOR_BGRA2RGB), -1, 0))
            for index, x0 in enumerate((0, 64)):
                cv2.imwrite(str(folder/f"tile_{index:05d}.png"), image[:, x0:x0+64])
            with (folder/"metadata.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["tile_name", "x0", "y0", "width", "height"])
                writer.writeheader()
                for index, x0 in enumerate((0, 64)):
                    writer.writerow(dict(tile_name=f"tile_{index:05d}.png", x0=x0, y0=0,
                                         width=64, height=64))
            result = run_image("2017/tile_00000.png", 2017, "single_run",
                               config=cfg, project_root=root, show_progress=False)
            state = json.loads((root/"single_run/run.json").read_text())
            self.assertEqual(result["tile_count"], 1)
            self.assertEqual(state["tile_ids"], ["source__tile_00000"])
            self.assertFalse((root/"single_run/stage2/source__tile_00001").exists())
            self.assertEqual(load_stage4_result(result["combined_graph_directory"]).graph["selected_tile_ids"],
                             ["source__tile_00000"])

    def test_folder_requires_georeference_and_exact_tile_metadata(self):
        from src.folder_pipeline import inspect_folder
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            cv2.imwrite(str(folder/"tile_00000.png"), np.full((8, 8, 3), 80, dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "georeferenced source raster"):
                inspect_folder(folder)
            with rasterio.open(folder/"source.tif", "w", driver="GTiff", width=8, height=8,
                               count=3, dtype="uint8", crs="EPSG:32636",
                               transform=from_origin(500000, 3700000, 1.5, 1.5)) as output:
                output.write(np.full((3, 8, 8), 80, dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, "metadata.csv"):
                inspect_folder(folder)
            with (folder/"metadata.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["tile_name", "x0", "y0", "width", "height"])
                writer.writeheader()
                writer.writerow(dict(tile_name="tile_00000.png", x0=0, y0=0, width=8, height=8))
            inputs = inspect_folder(folder)
            self.assertEqual(inputs["tiles"][0].name, "tile_00000.png")
            self.assertEqual(inputs["raster"].name, "source.tif")
            (folder/"source.tif").rename(folder/"source.tiff")
            self.assertEqual(inspect_folder(folder)["raster"].name, "source.tiff")

    def test_padded_edge_tile_keeps_valid_extent(self):
        from src.folder_pipeline import inspect_folder
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            rgb = np.arange(36, dtype=np.uint8).reshape(3, 4, 3)
            rgba = np.zeros((4, 4, 4), dtype=np.uint8)
            rgba[:3, :3, :3] = rgb[:, :3]
            rgba[:3, :3, 3] = 255
            cv2.imwrite(str(folder/"tile_00000.png"), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
            with rasterio.open(folder/"source.tif", "w", driver="GTiff", width=4, height=3,
                               count=3, dtype="uint8", crs="EPSG:32636",
                               transform=from_origin(500000, 3700000, 1.5, 1.5)) as output:
                output.write(np.moveaxis(rgb, -1, 0))
            csv_path = folder/"metadata.csv"
            with csv_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["tile_name", "x0", "y0", "width", "height",
                                                            "valid_width", "valid_height", "alpha_band", "valid_pixel_count"])
                writer.writeheader()
                writer.writerow(dict(tile_name="tile_00000.png", x0=0, y0=0, width=4, height=4,
                                     valid_width=3, valid_height=3, alpha_band=4, valid_pixel_count=9))
            row = inspect_folder(folder)["rows"][0]
            self.assertEqual((row["valid_width"], row["valid_height"], row["valid_pixel_count"]), (3, 3, 9))
            with csv_path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["tile_name", "x0", "y0", "width", "height",
                                                            "valid_width", "valid_height", "alpha_band", "valid_pixel_count"])
                writer.writeheader()
                writer.writerow(dict(tile_name="tile_00000.png", x0=1, y0=0, width=4, height=4,
                                     valid_width=3, valid_height=3, alpha_band=4, valid_pixel_count=9))
            with self.assertRaisesRegex(ValueError, "pixels.*source raster"):
                inspect_folder(folder)

    def test_annual_output_cannot_be_inside_tile_folder(self):
        from src.folder_pipeline import run_folder
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            image = np.full((8, 8, 3), 80, dtype=np.uint8)
            cv2.imwrite(str(folder/"tile_00000.png"), image)
            with rasterio.open(folder/"source.tif", "w", driver="GTiff", width=8, height=8,
                               count=3, dtype="uint8", crs="EPSG:32636",
                               transform=from_origin(500000, 3700000, 1.5, 1.5)) as output:
                output.write(np.moveaxis(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), -1, 0))
            with (folder/"metadata.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["tile_name", "x0", "y0"])
                writer.writeheader()
                writer.writerow(dict(tile_name="tile_00000.png", x0=0, y0=0))
            with self.assertRaisesRegex(ValueError, "separate from the input"):
                run_folder(folder, 2017, folder/"results")
            self.assertFalse((folder/"results").exists())

    def test_annual_staging_failure_is_recorded_for_retry(self):
        from src.folder_pipeline import run_folder
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root/"tiles"
            folder.mkdir()
            cv2.imwrite(str(folder/"tile_00000.png"), np.full((8, 8, 3), 80, dtype=np.uint8))
            with rasterio.open(folder/"source.tif", "w", driver="GTiff", width=8, height=8,
                               count=3, dtype="uint8", crs="EPSG:32636",
                               transform=from_origin(500000, 3700000, 1.5, 1.5)) as output:
                output.write(np.full((3, 8, 8), 80, dtype=np.uint8))
            with (folder/"metadata.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["tile_name", "x0", "y0"])
                writer.writeheader()
                writer.writerow(dict(tile_name="tile_00000.png", x0=0, y0=0))
            destination = root/"annual"
            with patch("src.folder_pipeline.shutil.copy2", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    run_folder(folder, 2017, destination)
            state = json.loads((destination/"run.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertIn("disk full", state["error"])

    def test_compare_rejects_incomplete_years(self):
        from src.folder_pipeline import compare_folders
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("a", "b"):
                (root/name).mkdir()
                (root/name/"run.json").write_text(json.dumps({"status": "running", "year": 2017}))
            with self.assertRaisesRegex(ValueError, "complete"):
                compare_folders(root/"a", root/"b", root/"comparison")

    def test_both_folder_commands_complete_with_test_models(self):
        from src.folder_pipeline import compare_folders, run_folder
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = load_config(DEFAULT_CONFIG)
            cfg["stage2"]["instance_model"]["backend"] = "fake"
            cfg["stage2"]["semantic_model"]["backend"] = "fake"
            (root/"fake-config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
            annual = []
            for year in (2017, 2023):
                folder = root/str(year)
                folder.mkdir()
                image = np.full((64, 64, 4), 80, dtype=np.uint8)
                yy, xx = np.indices((64, 64))
                image[:,:,0] = xx + 40
                image[:,:,1] = yy + 40
                image[:,:,2] = xx + yy + 40
                image[:,:,3] = 255
                cv2.imwrite(str(folder/"tile_00000.png"), image)
                with rasterio.open(folder/"source.tif", "w", driver="GTiff", width=64, height=64,
                                   count=3, dtype="uint8", crs="EPSG:32636",
                                   transform=from_origin(500000, 3700000, 1.5, 1.5)) as output:
                    output.write(np.moveaxis(cv2.cvtColor(image, cv2.COLOR_BGRA2RGB), -1, 0))
                with (folder/"metadata.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["tile_name", "x0", "y0", "width", "height"])
                    writer.writeheader()
                    writer.writerow(dict(tile_name="tile_00000.png", x0=0, y0=0, width=64, height=64))
                destination = root/f"annual_{year}"
                result = run_folder(folder, year, destination, config="fake-config.yaml", project_root=root,
                                    show_progress=False)
                self.assertEqual(result["status"], "complete")
                self.assertTrue(Path(result["full_image_with_graph"]).is_file())
                self.assertEqual(json.loads((destination/"run.json").read_text())["tiles"]["source__tile_00000"]["refinement_validation"]["status"], "valid")
                repeated = run_folder(folder, year, destination, config="fake-config.yaml", project_root=root,
                                      show_progress=False)
                self.assertEqual(repeated["combined_graph_directory"],result["combined_graph_directory"])
                if year == 2017:
                    stale = json.loads((destination/"run.json").read_text())
                    stale.update(status="failed", error="previous failure", current_step="old step")
                    (destination/"run.json").write_text(json.dumps(stale))
                    run_folder(folder, year, destination, config="fake-config.yaml", project_root=root,
                               show_progress=False)
                    recovered = json.loads((destination/"run.json").read_text())
                    self.assertEqual(recovered["status"], "complete")
                    self.assertNotIn("error", recovered)
                    self.assertNotIn("current_step", recovered)
                annual.append(destination)
            invalid_path = annual[1]/"run.json"
            invalid = json.loads(invalid_path.read_text())
            invalid["tiles"]["source__tile_00000"]["status"] = "failed"
            invalid_path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, "complete"):
                compare_folders(*annual, root/"comparison", config="fake-config.yaml", project_root=root)
            invalid["tiles"]["source__tile_00000"]["status"] = "complete"
            invalid_path.write_text(json.dumps(invalid))
            comparison = compare_folders(*annual, root/"comparison", config="fake-config.yaml", project_root=root)
            self.assertEqual(comparison["status"], "complete")
            self.assertTrue(Path(comparison["temporal_graph_image"]).is_file())
            self.assertTrue((root/"comparison/01_temporal_graph/graph.json").is_file())
            failed = json.loads((root/"comparison/run.json").read_text())
            failed["status"] = "failed"
            (root/"comparison/run.json").write_text(json.dumps(failed))
            retried = compare_folders(*annual, root/"comparison", config="fake-config.yaml", project_root=root)
            self.assertEqual(retried["status"], "complete")
            moving = [[500010, 3699990], [500080, 3699990], [500080, 3699920],
                      [500010, 3699920], [500045, 3699955], [500025, 3699940]]
            shifted = {"moving_xy": moving,
                       "reference_xy": [[x + 10000, y] for x, y in moving]}
            (root/"shift.json").write_text(json.dumps(shifted))
            with self.assertRaisesRegex(ValueError, "no common observed coverage"):
                compare_folders(*annual, root/"no_overlap", registration_json="shift.json",
                                config="fake-config.yaml", project_root=root)
            with rasterio.open(root/"2023/source.tif", "r+") as source:
                band = source.read(1)
                band[0, 0] ^= 1
                source.write(band, 1)
            with self.assertRaisesRegex(ValueError, "source raster changed"):
                compare_folders(*annual, root/"changed_image", register_images=True,
                                config="fake-config.yaml", project_root=root)
            self.assertFalse((root/"changed_image").exists())


if __name__ == "__main__":
    unittest.main()
