from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config import DEFAULT_CONFIG
from src.stage2.artifacts import atomic_json, manifest, mark_downstream_stale
from src.stage2.backend import FakeSamBackend, Sam2Backend, create_backend
from src.stage2.config import load_config, validate_config
from src.stage2.masks import analyze_coverage, deduplicate_masks, filter_masks, normalize_masks
from src.stage2.phases import finalize_phase, run_all
from src.stage2.preprocessing import prepare_image
from src.stage2.storage import load_masks, save_masks
from src.stage2.tiles import discover_tiles, validate_tile
from src.stage2.visualization import overlay


def cfg(tmp_path: Path) -> dict:
    value = load_config(DEFAULT_CONFIG)
    value["backend"] = "fake"; value["output"]["root_directory"] = str(tmp_path / "artifacts")
    return value


def make_project(tmp_path: Path, image: np.ndarray | None = None) -> tuple[Path, str]:
    project = tmp_path / "project"; tiles = project / "data" / "tiles" / "scene"; metadata = project / "data" / "metadata"
    tiles.mkdir(parents=True); metadata.mkdir(parents=True)
    image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3) if image is None else image
    cv2.imwrite(str(tiles / "tile_00000.png"), image[..., ::-1])
    with (metadata / "scene.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["tile_id", "tile_name", "x0", "y0", "x1", "y1", "width", "height"]); writer.writeheader(); writer.writerow({"tile_id": 0, "tile_name": "tile_00000.png", "x0": 0, "y0": 0, "x1": 16, "y1": 16, "width": 16, "height": 16})
    return project, "scene__tile_00000"


def record(segment_id: str, mask: np.ndarray, score=.9, stability=.9, valid=1.0) -> dict:
    rows, cols = np.nonzero(mask); x, y = int(cols.min()), int(rows.min()); w, h = int(cols.max()+1-x), int(rows.max()+1-y)
    return {"segment_id": segment_id, "source_tile_id": "tile", "bbox_xywh": [x, y, w, h], "centroid_rc": [float(rows.mean()), float(cols.mean())], "area_pixels": int(mask.sum()), "predicted_iou": score, "stability_score": stability, "valid_pixel_fraction": valid, "touches_tile_boundary": False, "source_raw_mask_id": segment_id, "provenance": {}}


def test_tile_discovery_and_compatibility_validation(tmp_path):
    project, tile_id = make_project(tmp_path); assert discover_tiles(project)[0]["tile_id"] == tile_id
    image, metadata, report = validate_tile(project, tile_id); assert image.shape == (16, 16, 3) and report["valid"]
    assert metadata["crs"] is None and metadata["global_x0"] == 0


def test_tile_dimension_mismatch_is_error(tmp_path):
    project, tile_id = make_project(tmp_path); path = project / "data" / "metadata" / "scene.csv"
    path.write_text("tile_id,tile_name,x0,y0,x1,y1,width,height\n0,tile_00000.png,0,0,99,16,99,16\n")
    with pytest.raises(ValueError, match="dimensions"): validate_tile(project, tile_id)


@pytest.mark.parametrize("dtype,mode", [(np.uint8, "none"), (np.uint16, "per_band_percentile"), (np.float32, "min_max")])
def test_deterministic_preprocessing_dtypes(dtype, mode):
    image = np.arange(8*8*3).reshape(8, 8, 3).astype(dtype); config = {"rgb_bands": [3, 2, 1], "normalization": mode, "lower_percentile": 2, "upper_percentile": 98, "fixed_range": None, "nodata_fill": 0}
    first = prepare_image(image, {"nodata": None}, config); second = prepare_image(image, {"nodata": None}, config)
    assert first[0].dtype == np.uint8 and np.array_equal(first[0], second[0]) and first[2] == second[2]
    assert first[2]["operations"][0]["source_band"] == 3


def test_nodata_nan_and_infinity():
    image = np.ones((4, 4, 3), np.float32); image[0, 0] = np.nan; image[0, 1] = np.inf; image[0, 2] = -999
    config = {"rgb_bands": [1, 2, 3], "normalization": "fixed_range", "lower_percentile": 2, "upper_percentile": 98, "fixed_range": [0, 2], "nodata_fill": 7}
    rgb, valid, _ = prepare_image(image, {"nodata": -999}, config); assert not valid[0, :3].any() and np.all(rgb[0, :3] == 7)


def test_fake_raw_inference_and_serialization(tmp_path):
    image = np.zeros((12, 12, 3), np.uint8); backend = FakeSamBackend(); raw = backend.infer_raw(image)
    path, meta = tmp_path / "masks.npz", tmp_path / "masks.json"; save_masks(path, meta, [raw[0]["segmentation"]], [{"raw_mask_id": "raw_0"}])
    masks, records = load_masks(path, meta); assert np.array_equal(masks[0], raw[0]["segmentation"]) and records[0]["raw_mask_id"] == "raw_0"


def test_sam2_opencv_postprocessing_fills_holes_and_removes_islands():
    from sam2.utils.amg import remove_small_regions

    backend = Sam2Backend(None, None, None, "cpu", False, {"min_mask_region_area": 2})
    backend.cpu_postprocess_region_area = 2
    backend._remove_small_regions = remove_small_regions
    mask = np.zeros((7, 7), bool)
    mask[1:4, 1:4] = True
    mask[2, 2] = False
    mask[6, 6] = True
    records = backend._apply_cpu_postprocessing(
        [{"segmentation": mask, "area": int(mask.sum()), "bbox": [1, 1, 6, 6]}],
        mask.shape,
    )
    cleaned = records[0]["segmentation"]
    assert cleaned[2, 2] and not cleaned[6, 6]
    assert records[0]["area"] == 9
    assert records[0]["bbox"] == [1, 1, 3, 3]
    assert records[0]["postprocessing"]["backend"] == "opencv_cpu_fallback"


def test_normalization_rejects_empty_and_wrong_shape():
    good = np.zeros((8, 8), bool); good[2:4, 3:6] = True
    masks, records, report = normalize_masks([good, np.zeros((8, 8), bool), np.zeros((7, 8), bool)], [{"area": 99, "bbox": [-2, -2, 99, 99]}, {}, {}], (8, 8), np.ones((8, 8), bool), "tile")
    assert len(masks) == 1 and records[0]["area_pixels"] == 6 and records[0]["bbox_xywh"] == (3, 2, 3, 2) and report["rejected_count"] == 2


def test_area_confidence_and_nodata_filtering():
    mask = np.zeros((10, 10), bool); mask[1:4, 1:4] = True; rec = record("a", mask, score=.2, valid=.3)
    config = {"minimum_area": 10, "maximum_area_ratio": .95, "minimum_predicted_iou": .5, "minimum_stability_score": None, "minimum_valid_pixel_fraction": .5, "maximum_nodata_fraction": .5, "minimum_fill_ratio": .01, "border_policy": "keep"}
    _, accepted, rejected = filter_masks([mask], [rec], np.ones((10, 10), bool), config)
    assert not accepted and {"area_below_minimum", "predicted_iou_below_minimum", "valid_pixel_fraction_below_minimum", "nodata_fraction_above_maximum"} <= set(rejected[0]["rejection_reasons"])


def test_duplicate_and_nested_policies():
    parent = np.zeros((10, 10), bool); parent[1:8, 1:8] = True
    duplicate = parent.copy(); child = np.zeros_like(parent); child[2:5, 2:5] = True
    records = [record("parent", parent, .8), record("duplicate", duplicate, .9), record("child", child, .95)]
    config = {"mask_iou_threshold": .9, "containment_threshold": .95, "bbox_overlap_threshold": 0, "area_similarity_threshold": .5, "duplicate_policy": "keep_higher_predicted_iou", "nested_policy": "keep_parent"}
    masks, kept, report = deduplicate_masks([parent, duplicate, child], records, config)
    assert len(masks) == 1 and kept[0]["segment_id"] == "duplicate" and report["nested_relationships"]


@pytest.mark.parametrize("policy,expected", [("leave_unassigned", 0), ("single_background", 1), ("connected_components", 1)])
def test_coverage_and_background(policy, expected):
    valid = np.ones((8, 8), bool); mask = np.zeros_like(valid); mask[:4] = True
    config = {"background_policy": policy, "minimum_background_component_area": 2}
    masks, _, count, uncovered, report = analyze_coverage([mask], [record("a", mask)], valid, config)
    assert report["coverage_percentage"] == 50 and report["background_masks_added"] == expected and len(masks) == 1 + expected and uncovered.sum() == 32 and count.max() == 1


def test_visualization_is_deterministic(tmp_path):
    image = np.zeros((10, 10, 3), np.uint8); mask = np.zeros((10, 10), bool); mask[2:5, 2:5] = True; rec = record("a", mask)
    assert np.array_equal(overlay(image, [mask], [rec], seed=4), overlay(image, [mask], [rec], seed=4))


def test_manifest_invalidation_retains_outputs(tmp_path):
    tile = tmp_path / "tile"; first = tile / "02_preprocessing"; later = tile / "04_raw_inference"; output = later / "data.bin"
    output.parent.mkdir(parents=True); output.write_bytes(b"x"); atomic_json(later / "manifest.json", {"stale": False})
    changed = mark_downstream_stale(tile, "02_preprocessing")
    assert changed == ["04_raw_inference"] and output.exists() and json.loads((later / "manifest.json").read_text())["stale"]


def test_complete_mock_pipeline_and_final_loading(tmp_path):
    project, tile_id = make_project(tmp_path); config = cfg(tmp_path); config["filtering"]["minimum_area"] = 1
    results = run_all(config, project, tile_id, FakeSamBackend()); final = Path(results[-1]["output_directory"])
    masks, records = load_masks(final / "segments.npz", final / "segments.json")
    assert masks and records and (final / "manifest.json").is_file() and (final / "final_overlay.png").is_file()


@pytest.mark.real_sam
def test_optional_real_sam_smoke(tmp_path):
    del tmp_path
    config = load_config(DEFAULT_CONFIG)
    checkpoint = config.get("checkpoint")
    if not checkpoint or not Path(checkpoint).is_file():
        pytest.skip("No configured SAM checkpoint")
    config["automatic_mask_generator"].update(
        points_per_side=2,
        points_per_batch=1,
        crop_n_layers=0,
        min_mask_region_area=2,
    )
    backend = create_backend(config)
    with warnings.catch_warnings(record=True) as caught:
        masks = backend.infer_raw(np.zeros((32, 32, 3), np.uint8))
    assert isinstance(masks, list)
    assert backend.health_check()["postprocessing_backend"] in {
        "sam2_compiled_extension",
        "opencv_cpu_fallback",
    }
    assert not any("cannot import name '_C'" in str(item.message) for item in caught)
