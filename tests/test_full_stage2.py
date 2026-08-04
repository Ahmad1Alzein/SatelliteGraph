from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine, from_origin
from rasterio.warp import transform as transform_coordinates

from src.config import DEFAULT_CONFIG, load_config, validate_config
from src.stage2 import load_stage2_result, run_stage2
from src.stage2.backend import FakeSamBackend
from src.stage2.classes import CLASS_NAMES, SOURCE_CODES, UNKNOWN_CLASS_ID
from src.stage2.fusion import fuse_segmentations
from src.stage2.semantic import FakeSemanticBackend


TILE_HEIGHT = 10
TILE_WIDTH = 12
TILE_X0 = 4
TILE_Y0 = 3
SOURCE_CRS = "EPSG:32633"
SOURCE_TRANSFORM = from_origin(500_000, 4_000_000, 10, 10)


def _make_georeferenced_project(tmp_path: Path) -> tuple[Path, str]:
    project = tmp_path / "project"
    tiles = project / "data" / "tiles" / "scene"
    metadata = project / "data" / "metadata"
    raw = project / "data" / "raw"
    tiles.mkdir(parents=True)
    metadata.mkdir(parents=True)
    raw.mkdir(parents=True)

    # validate_tile only reads metadata from this exact source raster. Its
    # dimensions deliberately cover the nonzero Stage 1 tile offset.
    with rasterio.open(
        raw / "scene.tif",
        "w",
        driver="GTiff",
        height=20,
        width=24,
        count=3,
        dtype="uint8",
        crs=SOURCE_CRS,
        transform=SOURCE_TRANSFORM,
    ) as source:
        source.write(np.zeros((3, 20, 24), dtype=np.uint8))

    image = np.zeros((TILE_HEIGHT, TILE_WIDTH, 3), dtype=np.uint8)
    # FakeSemanticBackend maps the first RGB channel modulo eight. Every OEM
    # class, including tree (id 4), therefore occurs at high confidence.
    image[..., 0] = np.arange(TILE_WIDTH, dtype=np.uint8)[None, :] % len(CLASS_NAMES)
    image[..., 1] = np.arange(TILE_HEIGHT, dtype=np.uint8)[:, None] * 10
    image[..., 2] = 127
    assert cv2.imwrite(str(tiles / "tile_00000.png"), image[..., ::-1])

    with (metadata / "scene.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["tile_id", "tile_name", "x0", "y0", "x1", "y1", "width", "height"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "tile_id": 0,
                "tile_name": "tile_00000.png",
                "x0": TILE_X0,
                "y0": TILE_Y0,
                "x1": TILE_X0 + TILE_WIDTH,
                "y1": TILE_Y0 + TILE_HEIGHT,
                "width": TILE_WIDTH,
                "height": TILE_HEIGHT,
            }
        )
    return project, "scene__tile_00000"


def _fake_config(project: Path) -> dict:
    config = load_config(DEFAULT_CONFIG)
    stage2 = config["stage2"]
    stage2["instance_model"]["backend"] = "fake"
    stage2["semantic_model"]["backend"] = "fake"
    stage2["output"]["root_directory"] = str(project / "artifacts" / "stage2")
    stage2["preprocessing"]["normalization"] = "none"
    stage2["filtering"]["minimum_area"] = 1
    fusion = stage2["fusion"]
    fusion["tree_confidence_threshold"] = 0.5
    fusion["tree_dilation_pixels"] = 0
    fusion["sam_tree_overlap_threshold"] = 0.95
    fusion["minimum_sam_component_area"] = 1
    fusion["semantic_confidence_threshold"] = 0.5
    fusion["minimum_component_area"] = {name: 1 for name in [*CLASS_NAMES, "unknown"]}
    fusion["morphology_pixels"] = {}
    validate_config(config)
    return config


def _assert_segment_coordinate_spaces(record: dict) -> None:
    coordinates = record["coordinates"]
    local = coordinates["tile_pixel"]
    source = coordinates["source_pixel"]
    assert coordinates["global_pixel"] == source
    np.testing.assert_allclose(
        source["bbox_xyxy"],
        np.asarray(local["bbox_xyxy"]) + [TILE_X0, TILE_Y0, TILE_X0, TILE_Y0],
    )
    np.testing.assert_allclose(
        source["centroid_xy"],
        np.asarray(local["centroid_xy"]) + [TILE_X0, TILE_Y0],
    )

    source_bbox = source["bbox_xyxy"]
    projected_corners = [
        SOURCE_TRANSFORM * (source_bbox[0], source_bbox[1]),
        SOURCE_TRANSFORM * (source_bbox[2], source_bbox[1]),
        SOURCE_TRANSFORM * (source_bbox[2], source_bbox[3]),
        SOURCE_TRANSFORM * (source_bbox[0], source_bbox[3]),
    ]
    xs, ys = zip(*projected_corners)
    expected_projected_bbox = [min(xs), min(ys), max(xs), max(ys)]
    projected = coordinates["projected"]
    assert projected["available"] and projected["crs"] == SOURCE_CRS
    np.testing.assert_allclose(projected["bbox_xyxy"], expected_projected_bbox)

    wgs84 = coordinates["wgs84"]
    assert wgs84["available"] and wgs84["crs"] == "EPSG:4326"
    expected_lon, expected_lat = transform_coordinates(
        SOURCE_CRS,
        "EPSG:4326",
        [projected["centroid_xy"][0]],
        [projected["centroid_xy"][1]],
    )
    np.testing.assert_allclose(wgs84["centroid_xy"], [expected_lon[0], expected_lat[0]], rtol=0, atol=1e-8)


def test_run_stage2_returns_and_persists_one_complete_georeferenced_segmentation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project, tile_id = _make_georeferenced_project(tmp_path)
    result = run_stage2(
        tile_id,
        project_root=project,
        config=_fake_config(project),
        sam_backend=FakeSamBackend(),
        semantic_backend=FakeSemanticBackend(),
    )
    progress_output = capsys.readouterr().err
    assert "Stage 2 | SAM 2 inference" in progress_output
    assert "Stage 2 | OpenEarthMap inference" in progress_output
    assert "Stage 2 | Complete" in progress_output

    assert result.tile_id == tile_id
    assert result.instance_map.shape == result.semantic_map.shape == (TILE_HEIGHT, TILE_WIDTH)
    assert result.valid_mask.all()
    assert np.all(result.instance_map[result.valid_mask] > 0)
    assert result.report["coverage"] == {
        "valid_pixel_count": TILE_HEIGHT * TILE_WIDTH,
        "assigned_valid_pixel_count": TILE_HEIGHT * TILE_WIDTH,
        "unassigned_valid_pixel_count": 0,
        "coverage_percentage": 100.0,
        "overlap_pixel_count": 0,
    }
    coverage_count = np.stack(result.masks, axis=0).sum(axis=0)
    assert np.all(coverage_count[result.valid_mask] == 1)
    assert np.array_equal(
        np.unique(result.instance_map[result.valid_mask]),
        np.arange(1, len(result.segments) + 1),
    )

    tree_id = CLASS_NAMES.index("tree")
    assert np.any(result.semantic_map == tree_id)
    assert any(record["label"]["name"] == "tree" for record in result.segments)

    for record in result.segments:
        label = record["label"]
        assert label["name"] in [*CLASS_NAMES, "unknown"]
        assert label["id"] == ([*CLASS_NAMES, "unknown"].index(label["name"]))
        assert label["source_model"] in {"openearthmap_fasterseg", "stage2_fusion_rule"}
        assert label["method"].strip()
        assert isinstance(label["evidence"], dict) and label["evidence"]
        assert label["explanation"].strip()
        geometry = record["geometry_source"]
        assert geometry["model"] in {"sam2", "openearthmap_fasterseg", "stage2_fusion_rule"}
        assert geometry["explanation"].strip()
        assert record["metrics"]["area_pixels"] == record["area_pixels"]
        assert record["metrics"]["touches_tile_boundary"] == record["touches_tile_boundary"]
        _assert_segment_coordinate_spaces(record)
    assert any(record["geometry_source"]["model"] == "sam2" for record in result.segments)
    assert any(record["geometry_source"]["model"] == "openearthmap_fasterseg" for record in result.segments)

    expected_tile_transform = SOURCE_TRANSFORM * Affine.translation(TILE_X0, TILE_Y0)
    np.testing.assert_allclose(
        result.tile_metadata["tile_affine_transform"],
        [expected_tile_transform.a, expected_tile_transform.b, expected_tile_transform.c,
         expected_tile_transform.d, expected_tile_transform.e, expected_tile_transform.f],
    )
    assert result.tile_metadata["global_x0"] == TILE_X0
    assert result.tile_metadata["global_y0"] == TILE_Y0
    assert result.report["coordinate_reference"]["projected"]["available"]
    assert result.report["coordinate_reference"]["geographic"]["available"]

    maps_path = result.output_directory / "segmentation.npz"
    with np.load(maps_path, allow_pickle=False) as persisted:
        assert np.array_equal(persisted["instance_map"], result.instance_map)
        assert np.array_equal(persisted["semantic_map"], result.semantic_map)
        assert np.array_equal(persisted["confidence_map"].astype(np.float32), result.confidence_map)
        assert np.array_equal(persisted["source_map"], result.source_map)
        assert np.array_equal(persisted["valid_mask"], result.valid_mask)

    with rasterio.open(result.output_directory / "instance_map.tif") as dataset:
        assert dataset.crs.to_string() == SOURCE_CRS
        assert dataset.transform == expected_tile_transform
        assert np.array_equal(dataset.read(1), result.instance_map)
    with rasterio.open(result.output_directory / "semantic_map.tif") as dataset:
        assert dataset.crs.to_string() == SOURCE_CRS
        assert dataset.transform == expected_tile_transform
        assert np.array_equal(dataset.read(1), result.semantic_map)

    assert all(Path(path).is_file() for path in result.artifacts.values())
    reloaded = load_stage2_result(result.output_directory)
    assert reloaded.segments == result.segments
    assert reloaded.report == result.report
    assert reloaded.tile_metadata == result.tile_metadata
    assert reloaded.phases == {}
    for returned, loaded in zip(result.masks, reloaded.masks):
        assert np.array_equal(returned, loaded)
    for field in ("instance_map", "semantic_map", "confidence_map", "source_map", "valid_mask"):
        assert np.array_equal(getattr(result, field), getattr(reloaded, field))


def test_fusion_uses_unknown_for_low_confidence_valid_pixels() -> None:
    shape = (4, 5)
    valid = np.ones(shape, dtype=bool)
    labels = np.zeros(shape, dtype=np.uint8)
    confidence = np.full(shape, 0.1, dtype=np.float32)
    metadata = {
        "tile_id": "scene__tile_00000",
        "global_x0": 7,
        "global_y0": 9,
        "source_affine_transform": [2, 0, 100, 0, -2, 200],
        "tile_affine_transform": [2, 0, 114, 0, -2, 182],
        "crs": "EPSG:4326",
    }
    fusion_config = load_config(DEFAULT_CONFIG)["stage2"]["fusion"]
    fusion_config["tree_dilation_pixels"] = 0
    fusion_config["minimum_sam_component_area"] = 1
    fusion_config["semantic_confidence_threshold"] = 0.5
    fusion_config["minimum_component_area"] = {name: 1 for name in [*CLASS_NAMES, "unknown"]}
    fusion_config["morphology_pixels"] = {}

    masks, records, instance_map, semantic_map, _, source_map, report = fuse_segmentations(
        [], [], labels, confidence, valid, metadata, fusion_config
    )

    assert masks and records
    assert np.all(instance_map > 0)
    assert np.all(semantic_map == UNKNOWN_CLASS_ID)
    assert np.all(source_map == SOURCE_CODES["fusion_unknown"])
    assert all(record["label"]["name"] == "unknown" for record in records)
    assert all(record["source_tile_id"] == metadata["tile_id"] for record in records)
    assert all(record["label"]["source_model"] == "stage2_fusion_rule" for record in records)
    assert report["class_pixel_counts"]["unknown"] == valid.size
    assert report["every_valid_pixel_exactly_once"]


class _WrongShapeSemanticBackend(FakeSemanticBackend):
    def infer(self, image: np.ndarray) -> dict[str, np.ndarray]:
        result = super().infer(image)
        return {
            "label_map": result["label_map"][:-1],
            "confidence": result["confidence"][:-1],
            "probabilities": result["probabilities"][:, :-1],
        }


def test_run_stage2_rejects_a_mismatched_semantic_shape(tmp_path: Path) -> None:
    project, tile_id = _make_georeferenced_project(tmp_path)
    with pytest.raises(ValueError, match=r"OpenEarthMap output shape .* does not match tile"):
        run_stage2(
            tile_id,
            project_root=project,
            config=_fake_config(project),
            sam_backend=FakeSamBackend(),
            semantic_backend=_WrongShapeSemanticBackend(),
            show_progress=False,
        )
