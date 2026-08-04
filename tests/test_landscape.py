from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config import DEFAULT_CONFIG
from src.landscape.backend import FakeLandcoverBackend, OpenEarthMapBackend
from src.landscape.completion import complete_non_tree_landscape
from src.landscape.config import CLASS_NAMES, load_config, validate_config
from src.landscape.phases import run_all as run_landscape
from src.stage2.backend import FakeSamBackend
from src.stage2.config import load_config as load_stage2_config
from src.stage2.phases import run_all as run_stage2
from src.stage2.storage import load_masks


PROJECT_ROOT = Path(__file__).parents[1]


def _record(segment_id: str, mask: np.ndarray) -> dict:
    rows, cols = np.nonzero(mask)
    return {
        "segment_id": segment_id,
        "source_tile_id": "tile",
        "bbox_xywh": [int(cols.min()), int(rows.min()), int(cols.max() - cols.min() + 1), int(rows.max() - rows.min() + 1)],
        "centroid_rc": [float(rows.mean()), float(cols.mean())],
        "area_pixels": int(mask.sum()),
        "predicted_iou": 0.9,
        "stability_score": 0.9,
        "valid_pixel_fraction": 1.0,
        "touches_tile_boundary": False,
        "source_raw_mask_id": segment_id,
        "provenance": {},
    }


def _completion_config() -> dict:
    cfg = load_config(DEFAULT_CONFIG)["completion"]
    cfg["tree_dilation_pixels"] = 0
    cfg["minimum_sam_component_area"] = 1
    cfg["semantic_confidence_threshold"] = 0.4
    cfg["minimum_component_area"] = {name: 1 for name in [*CLASS_NAMES, "unknown"]}
    cfg["morphology_pixels"] = {}
    return cfg


def test_config_rejects_tree_completion_priority():
    cfg = load_config(DEFAULT_CONFIG)
    cfg["completion"]["priority"].append("tree")
    with pytest.raises(ValueError, match="Trees"):
        validate_config(cfg)


def test_fake_landcover_inference_is_deterministic():
    image = np.zeros((8, 8, 3), np.uint8)
    image[..., 0] = np.arange(8, dtype=np.uint8)
    backend = FakeLandcoverBackend()
    first, second = backend.infer(image), backend.infer(image)
    assert np.array_equal(first["label_map"], second["label_map"])
    assert first["probabilities"].shape == (8, 8, 8)
    assert backend.health_check()["model_loaded_successfully"]


def test_tree_exclusion_and_road_building_completion():
    shape = (16, 16)
    valid = np.ones(shape, bool)
    labels = np.full(shape, CLASS_NAMES.index("bareland"), np.uint8)
    labels[:, :3] = CLASS_NAMES.index("tree")
    labels[7:9, :] = CLASS_NAMES.index("road")
    labels[2:6, 10:14] = CLASS_NAMES.index("buildings")
    confidence = np.full(shape, 0.95, np.float32)
    sam = np.zeros(shape, bool)
    sam[2:8, 1:7] = True
    masks, records, tree, report = complete_non_tree_landscape(
        [sam], [_record("sam0", sam)], labels, confidence, valid, _completion_config()
    )
    assert np.array_equal(tree, labels == CLASS_NAMES.index("tree"))
    assert all(not np.logical_and(mask, tree).any() for mask in masks)
    assert {record["semantic_class"] for record in records} >= {"road", "buildings"}
    assert report["non_tree_coverage_percentage"] == 100.0


def test_openearthmap_missing_checkpoint_is_clear(tmp_path):
    backend = OpenEarthMapBackend(
        str(PROJECT_ROOT / "third_party" / "oem-lightweight"),
        str(tmp_path / "arch.pt"),
        str(tmp_path / "weights.pt"),
    )
    with pytest.raises(FileNotFoundError, match="architecture"):
        backend.load()


def test_complete_fake_landscape_pipeline_and_final_loading(tmp_path):
    project = tmp_path / "project"
    tiles = project / "data" / "tiles" / "scene"
    metadata = project / "data" / "metadata"
    tiles.mkdir(parents=True)
    metadata.mkdir(parents=True)
    image = np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
    cv2.imwrite(str(tiles / "tile_00000.png"), image[..., ::-1])
    with (metadata / "scene.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["tile_id", "tile_name", "x0", "y0", "x1", "y1", "width", "height"])
        writer.writeheader()
        writer.writerow({"tile_id": 0, "tile_name": "tile_00000.png", "x0": 0, "y0": 0, "x1": 16, "y1": 16, "width": 16, "height": 16})
    tile_id = "scene__tile_00000"

    stage2_cfg = load_stage2_config(DEFAULT_CONFIG)
    stage2_cfg["backend"] = "fake"
    stage2_cfg["automatic_mask_generator"] = {
        key: value for key, value in stage2_cfg["automatic_mask_generator"].items()
        if key not in {"mask_threshold", "use_m2m", "multimask_output"}
    }
    stage2_cfg["output"]["root_directory"] = str(project / "artifacts" / "stage2")
    stage2_cfg["filtering"]["minimum_area"] = 1
    run_stage2(stage2_cfg, project, tile_id, FakeSamBackend())

    landscape_cfg = load_config(DEFAULT_CONFIG)
    landscape_cfg["backend"] = "fake"
    landscape_cfg["stage2_root"] = str(project / "artifacts" / "stage2")
    landscape_cfg["output"]["root_directory"] = str(project / "artifacts" / "landscape")
    landscape_cfg["completion"] = _completion_config()
    results = run_landscape(landscape_cfg, project, tile_id, FakeLandcoverBackend())
    final = Path(results[-1]["output_directory"])
    masks, records = load_masks(final / "segments.npz", final / "segments.json")
    tree = np.load(final / "tree_exclusion_mask.npz")["tree_mask"].astype(bool)
    assert masks and records
    assert all(not np.logical_and(mask, tree).any() for mask in masks)
    assert (final / "landscape_manifest.json").is_file()
    assert (final / "final_overlay.png").is_file()


@pytest.mark.real_landscape
def test_optional_real_openearthmap_smoke():
    cfg = load_config(DEFAULT_CONFIG)
    if not Path(cfg["checkpoint"]).is_file() or not Path(cfg["architecture"]).is_file():
        pytest.skip("No configured OpenEarthMap assets")
    backend = OpenEarthMapBackend(
        cfg["repository"], cfg["architecture"], cfg["checkpoint"], input_size=64, use_tta=False
    )
    result = backend.infer(np.zeros((32, 32, 3), np.uint8))
    assert result["label_map"].shape == (32, 32)
