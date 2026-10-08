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


def test_annual_tile_registration_uses_geographic_overlap_not_filename(tmp_path):
    from src.folder_pipeline import _staged_year_registration
    rng = np.random.default_rng(9)
    base = cv2.GaussianBlur(rng.integers(0, 255, (320, 320), dtype=np.uint8), (5, 5), 0)
    for _ in range(100):
        x, y = rng.integers(15, 305, 2)
        cv2.circle(base, (int(x), int(y)), int(rng.integers(2, 8)),
                   int(rng.integers(30, 240)), -1)
    shifted = cv2.warpAffine(base, np.float32([[1, 0, -12], [0, 1, -15]]),
                             (320, 320), borderMode=cv2.BORDER_REFLECT)
    reference_path, moving_path = tmp_path/"earlier.png", tmp_path/"later.png"
    cv2.imwrite(str(reference_path), base[:256, :256])
    cv2.imwrite(str(moving_path), shifted[:256, 16:272])
    graph = {"source_affine_transform":list(from_origin(500000, 3700000, 1.5, 1.5))[:6],
             "crs":"EPSG:32636"}
    reference = {"preview_tiles":[{"path":str(reference_path), "tile_name":"a.png",
                                   "x0":0, "y0":0, "width":256, "height":256}]}
    moving = {"preview_tiles":[{"path":str(moving_path), "tile_name":"different.png",
                                "x0":16, "y0":0, "width":256, "height":256}]}
    controls, audit = _staged_year_registration(reference, moving, graph, graph, "EPSG:32636")
    assert audit["registered_tile_pairs"] == 1
    assert audit["tile_pairs"][0]["reference_tile"] == "a.png"
    assert audit["tile_pairs"][0]["moving_tile"] == "different.png"
    assert len(controls["moving_xy"]) >= 12


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

    def test_cli_accepts_two_completed_image_outputs_and_one_tile(self):
        args = parser().parse_args(["image", "compare", "--first", "annual_2013",
                                    "--second", "annual_2023", "--tile", "tile_00000.png",
                                    "--output", "tile_comparison"])
        self.assertEqual(args.tile, "tile_00000.png")
        self.assertEqual(args.first, "annual_2013")

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
            refinement_report = json.loads((root/"single_run/stage2/source__tile_00000/13_refinement/report.json").read_text())
            self.assertTrue(refinement_report["micro_unknown_omission"]["enabled"])
            self.assertEqual(refinement_report["micro_unknown_omission"]["maximum_area_pixels"], 40)
            self.assertTrue(state["pipeline_revision"])
            self.assertFalse((root/"single_run/stage2/source__tile_00001").exists())
            combined = load_stage4_result(result["combined_graph_directory"])
            self.assertEqual(combined.graph["selected_tile_ids"], ["source__tile_00000"])
            self.assertEqual(Path(result["full_image_with_graph"]).read_bytes(),
                             Path(combined.artifacts["graph_overlay_audit"]).read_bytes())
            state.pop("pipeline_revision")
            (root/"single_run/run.json").write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "predates"):
                run_image("2017/tile_00000.png", 2017, "single_run",
                          config=cfg, project_root=root, show_progress=False)

    def test_image_run_accepts_png_only_and_marks_local_pixel_coordinates(self):
        from src.folder_pipeline import run_image
        from src.stage4 import load_stage4_result
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root/"2017"
            folder.mkdir()
            pixels = np.full((64, 64, 3), 100, dtype=np.uint8)
            cv2.rectangle(pixels, (10, 10), (30, 30), (30, 160, 210), -1)
            cv2.imwrite(str(folder/"tile_00000.png"), pixels)
            cv2.imwrite(str(folder/"tile_00001.png"), pixels)
            cfg = load_config(DEFAULT_CONFIG)
            cfg["stage2"]["instance_model"]["backend"] = "fake"
            cfg["stage2"]["semantic_model"]["backend"] = "fake"

            result = run_image("2017/tile_00000.png", 2017, "single_run",
                               config=cfg, project_root=root, show_progress=False)

            state = json.loads((root/"single_run/run.json").read_text())
            combined = load_stage4_result(result["combined_graph_directory"])
            self.assertEqual(state["tile_count"], 1)
            self.assertEqual(state["coordinate_frame"], "local_pixel")
            self.assertIsNone(state["source_raster"])
            self.assertEqual(combined.graph["source_pixel_extent_xyxy"], [0, 0, 64, 64])
            self.assertIsNone(combined.graph["crs"])
            self.assertTrue(Path(result["full_image_with_graph"]).is_file())
            self.assertFalse((root/"single_run/stage2/tile_00000__tile_00001").exists())
            from src.folder_pipeline import compare_folders
            with self.assertRaisesRegex(ValueError, "georeference"):
                compare_folders("single_run", "single_run", "comparison", project_root=root)

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

    def test_segmentation_then_graph_equals_one_complete_run(self):
        """stop_after='segmentation' + a second call must give the same graph as one uninterrupted run."""
        from src.folder_pipeline import run_folder
        from src.stage4 import load_stage4_result
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = load_config(DEFAULT_CONFIG)
            cfg["stage2"]["instance_model"]["backend"] = "fake"
            cfg["stage2"]["semantic_model"]["backend"] = "fake"
            (root/"fake-config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
            folder = root/"2017"
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
            common = dict(config="fake-config.yaml", project_root=root, show_progress=False)
            staged = root/"staged"
            first = run_folder(folder, 2017, staged, stop_after="segmentation", **common)
            self.assertEqual(first["status"], "segmented")
            state = json.loads((staged/"run.json").read_text())
            self.assertEqual(state["status"], "segmented")
            self.assertEqual(state["tiles"]["source__tile_00000"]["status"], "segmented")
            self.assertTrue((staged/"stage2/source__tile_00000/13_refinement/segmentation.npz").is_file())
            self.assertFalse((staged/"stage3").exists())
            again = run_folder(folder, 2017, staged, stop_after="segmentation", **common)
            self.assertEqual(again["status"], "segmented")
            finished = run_folder(folder, 2017, staged, **common)
            self.assertEqual(finished["status"], "complete")
            whole = run_folder(folder, 2017, root/"whole", **common)
            a = load_stage4_result(finished["combined_graph_directory"])
            b = load_stage4_result(whole["combined_graph_directory"])
            self.assertEqual((len(a.nodes), len(a.edges)), (len(b.nodes), len(b.edges)))
            # same nodes and edges; provenance hashes differ only because they encode the output folder
            graphs = [json.loads((run/"stage3/source__tile_00000/01_graph/graph.json").read_text().replace(run.name, "RUN"))
                      for run in (staged, root/"whole")]
            self.assertEqual(graphs[0]["nodes"], graphs[1]["nodes"])
            self.assertEqual(graphs[0]["edges"], graphs[1]["edges"])
            with self.assertRaisesRegex(ValueError, "stop_after"):
                run_folder(folder, 2017, root/"bad", stop_after="graph", **common)

    def test_both_folder_commands_complete_with_test_models(self):
        from src.folder_pipeline import compare_folders, run_folder
        from src.stage4 import load_stage4_result
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
                state = json.loads((destination/"run.json").read_text())
                self.assertEqual(state["tiles"]["source__tile_00000"]["refinement_validation"]["status"], "valid")
                self.assertTrue(state["pipeline_revision"])
                report = json.loads((destination/"stage2/source__tile_00000/13_refinement/report.json").read_text())
                self.assertEqual(report["micro_unknown_omission"]["maximum_area_pixels"], 40)
                self.assertTrue(report["micro_unknown_omission"]["enabled"])
                combined = load_stage4_result(result["combined_graph_directory"])
                self.assertEqual(Path(result["full_image_with_graph"]).read_bytes(),
                                 Path(combined.artifacts["graph_overlay_audit"]).read_bytes())
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

            for destination in annual:
                run_path = destination/"run.json"
                run = json.loads(run_path.read_text(encoding="utf-8"))
                run["combined_graph_directory"] = str(
                    Path("Z:/old_location") / "stage4" / f"year_{run['year']}" / "01_combined_graph"
                )
                run_path.write_text(json.dumps(run), encoding="utf-8")
                manifest_path = destination/"stage4"/f"year_{run['year']}"/"01_combined_graph"/"manifest.json"
                saved = json.loads(manifest_path.read_text(encoding="utf-8"))
                saved["artifacts"] = {
                    key: str(Path("Z:/old_location") / Path(value).name)
                    for key, value in saved["artifacts"].items()
                }
                manifest_path.write_text(json.dumps(saved), encoding="utf-8")
            moved = compare_folders(*annual, root/"relocated_comparison",
                                    config="fake-config.yaml", project_root=root)
            self.assertEqual(moved["status"], "complete")

            from src.folder_pipeline import compare_images
            single = compare_images(*annual, root/"single_image_comparison",
                                    config="fake-config.yaml", project_root=root,
                                    show_progress=False)
            self.assertEqual(single["status"], "complete")
            self.assertEqual(single["tile_name"], "tile_00000.png")
            self.assertTrue(Path(single["temporal_graph_image"]).is_file())
            self.assertTrue(Path(single["all_edge_types_image"]).is_file())
            self.assertTrue(Path(single["matching_graph_image"]).is_file())
            self.assertTrue(Path(single["temporal_links_image"]).is_file())
            self.assertTrue(Path(single["change_graph_image"]).is_file())
            self.assertTrue(all(Path(value).is_file() for value in single["change_type_images"].values()))
            self.assertEqual(single["registration_method"], "identical_staged_png")
            self.assertTrue((root/"single_image_comparison/inputs/year_2017/graph.json").is_file())
            self.assertTrue((root/"single_image_comparison/inputs/year_2023/graph.json").is_file())


if __name__ == "__main__":
    unittest.main()
