from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

from src.cli import main as cli_main
from src.config import DEFAULT_CONFIG, load_config, validate_config
from src.stage2 import run_stage2
from src.stage2.backend import FakeSamBackend
from src.stage2.classes import CLASS_NAMES
from src.stage2.semantic import FakeSemanticBackend
from src.refinement import apply_refinement
from src.stage3 import load_stage3_result, run_stage3
from src.stage3.adjacency import adjacent_instance_pairs, build_adjacency_edges
from src.stage3.geometry import build_node_geometries, polygonize_instance_map
from src.stage3.pipeline import _annotate_analysis_view
from src.stage3.schema import (
    ADJACENCY_DEFINITION,
    ANALYSIS_SCHEMA_VERSION,
    REFINEMENT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    make_edge_id,
    make_node_id,
    validate_analysis_graph_document,
    validate_graph_document,
)
from src.stage3.storage import save_adjacency_csv, save_graph_arrays, save_nodes_geojson


TILE_HEIGHT = 10
TILE_WIDTH = 12
TILE_X0 = 4
TILE_Y0 = 3
SOURCE_CRS = "EPSG:32633"
SOURCE_TRANSFORM = from_origin(500_000, 4_000_000, 10, 10)


def _polygon_test_map() -> np.ndarray:
    instance_map = np.zeros((8, 10), dtype=np.int32)
    instance_map[0, 0] = 2
    instance_map[1:6, 1:6] = 1
    instance_map[2:5, 2:5] = 0
    instance_map[6, 8] = 1
    return instance_map


def _visual_graph() -> tuple[list[dict], list[dict]]:
    nodes = [
        {
            "node_id": "tile::segment_a",
            "instance_id": 1,
            "segment_type": {"id": 4, "name": "tree"},
            "area_pixels": 9,
            "representative_point": {"tile_pixel": [8.5, 8.5]},
            "touches_tile_boundary": False,
        },
        {
            "node_id": "tile::segment_b",
            "instance_id": 2,
            "segment_type": {"id": 7, "name": "buildings"},
            "area_pixels": 4,
            "representative_point": {"tile_pixel": [24.5, 20.5]},
            "touches_tile_boundary": True,
        },
    ]
    edges = [
        {
            "edge_id": "tile::adj::000001--000002",
            "source": nodes[0]["node_id"],
            "target": nodes[1]["node_id"],
            "source_instance_id": 1,
            "target_instance_id": 2,
            "source_class_id": 4,
            "source_class": "tree",
            "target_class_id": 7,
            "target_class": "buildings",
            "same_semantic_class": False,
            "shared_boundary_side_count": 3,
            "left_right_contact_count": 2,
            "top_bottom_contact_count": 1,
            "shared_boundary_length_pixels": 3.0,
            "shared_boundary_length_projected": None,
        }
    ]
    return nodes, edges


def _make_fake_project(tmp_path: Path) -> tuple[Path, str]:
    project = tmp_path / "project"
    tiles = project / "data" / "tiles" / "scene"
    metadata = project / "data" / "metadata"
    raw = project / "data" / "raw"
    tiles.mkdir(parents=True)
    metadata.mkdir(parents=True)
    raw.mkdir(parents=True)

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
    fusion["minimum_sam_component_area"] = 1
    fusion["semantic_confidence_threshold"] = 0.5
    fusion["minimum_component_area"] = {name: 1 for name in [*CLASS_NAMES, "unknown"]}
    config["stage3"]["output"]["root_directory"] = str(project / "artifacts" / "stage3")
    validate_config(config)
    return config


def test_topology_fragment_analysis_threshold_must_be_nonnegative() -> None:
    config = load_config(DEFAULT_CONFIG)
    config["stage3"]["visualization"][
        "minimum_display_topology_fragment_area_pixels"
    ] = -1

    with pytest.raises(ValueError, match="minimum_display_topology_fragment_area_pixels"):
        validate_config(config)


def test_exact_polygon_multipolygon_hole_and_coordinate_fallbacks() -> None:
    instance_map = _polygon_test_map()
    polygons = polygonize_instance_map(instance_map)

    assert polygons == polygonize_instance_map(instance_map)
    assert polygons[2] == {
        "type": "Polygon",
        "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]],
    }
    assert polygons[1]["type"] == "MultiPolygon"
    assert polygons[1]["coordinates"][0] == [
        [[1, 1], [6, 1], [6, 6], [1, 6], [1, 1]],
        [[2, 2], [2, 5], [5, 5], [5, 2], [2, 2]],
    ]
    assert polygons[1]["coordinates"][1] == [
        [[8, 6], [9, 6], [9, 7], [8, 7], [8, 6]]
    ]

    geometries = build_node_geometries(
        instance_map,
        {
            "global_x0": 10,
            "global_y0": 20,
            "georeferencing_unavailable_reason": "synthetic raster has no affine or CRS",
        },
    )
    geometry = geometries[1]
    assert geometry["geometry_type"] == "MultiPolygon"
    assert geometry["coordinate_spaces"]["tile_pixel"]["geometry"] == polygons[1]
    assert geometry["coordinate_spaces"]["tile_pixel"]["bbox_xyxy"] == [1, 1, 9, 7]
    assert geometry["coordinate_spaces"]["source_pixel"]["bbox_xyxy"] == [11, 21, 19, 27]
    assert not geometry["coordinate_spaces"]["projected"]["available"]
    assert geometry["coordinate_spaces"]["projected"]["geometry"] is None
    assert geometry["coordinate_spaces"]["projected"]["unavailable_reason"]
    assert not geometry["coordinate_spaces"]["wgs84"]["available"]
    assert geometry["metrics"]["area_pixels"] == 17
    assert geometry["metrics"]["perimeter_pixel_units"] == 36
    assert geometry["metrics"]["component_count"] == 2
    assert geometry["metrics"]["ring_count"] == 3
    assert geometry["metrics"]["hole_count"] == 1
    assert not geometry["tile_boundary_contacts"]["touches"]

    boundary_polygon = geometries[2]
    assert boundary_polygon["geometry_type"] == "Polygon"
    assert boundary_polygon["tile_boundary_contacts"]["touches"]
    assert boundary_polygon["tile_boundary_contacts"]["top"]["side_count"] == 1
    assert boundary_polygon["tile_boundary_contacts"]["left"]["side_count"] == 1


def test_four_neighbour_adjacency_counts_projection_diagonals_and_nodata() -> None:
    instance_map = np.asarray(
        [
            [1, 1, 2, 0, 4],
            [1, 1, 2, 5, 0],
            [3, 3, 2, 0, 0],
            [3, 3, 2, 0, 6],
        ],
        dtype=np.int32,
    )
    metadata = {
        "global_x0": 10,
        "global_y0": 20,
        "tile_affine_transform": [2, 0, 100, 0, -3, 200],
        "crs": SOURCE_CRS,
    }
    edges = build_adjacency_edges(instance_map, tile_metadata=metadata)
    by_pair = {(edge["source"], edge["target"]): edge for edge in edges}

    assert adjacent_instance_pairs(instance_map) == [(1, 2), (1, 3), (2, 3), (2, 5)]
    assert list(by_pair) == [(1, 2), (1, 3), (2, 3), (2, 5)]
    assert (4, 5) not in by_pair  # These instances meet only at one corner.
    assert all(0 not in pair for pair in by_pair)

    vertical = by_pair[(1, 2)]
    assert vertical["shared_boundary_side_count"] == 2
    assert vertical["left_right_contact_count"] == 2
    assert vertical["top_bottom_contact_count"] == 0
    assert vertical["shared_boundary_length_projected"] == pytest.approx(6.0)
    assert vertical["contact_coordinates"]["tile_pixel"]["bbox_xyxy"] == [2, 0, 2, 2]
    assert vertical["contact_coordinates"]["source_pixel"]["bbox_xyxy"] == [12, 20, 12, 22]
    assert vertical["contact_coordinates"]["projected"]["available"]
    assert vertical["contact_coordinates"]["wgs84"]["available"]

    horizontal = by_pair[(1, 3)]
    assert horizontal["shared_boundary_side_count"] == 2
    assert horizontal["left_right_contact_count"] == 0
    assert horizontal["top_bottom_contact_count"] == 2
    assert horizontal["shared_boundary_length_projected"] == pytest.approx(4.0)

    filtered = build_adjacency_edges(
        instance_map,
        tile_metadata=metadata,
        minimum_shared_boundary_pixels=2,
    )
    assert [(edge["source"], edge["target"]) for edge in filtered] == [
        (1, 2),
        (1, 3),
        (2, 3),
    ]


def test_graph_schema_validates_structure_and_preserves_stage2_metadata() -> None:
    tile_id = "scene__tile_00000"
    first_id = make_node_id(tile_id, "segment_a")
    second_id = make_node_id(tile_id, "segment_b")
    first_metadata = {
        "instance_id": 1,
        "segment_id": "segment_a",
        "area_pixels": 1,
        "label": {"id": 4, "name": "tree", "evidence": {"votes": 8}},
        "provenance": {"source_raw_mask_id": "raw_1"},
    }
    geometries = build_node_geometries(
        np.asarray([[1, 2]], dtype=np.int32),
        {"global_x0": 0, "global_y0": 0},
    )
    document = {
        "schema_version": SCHEMA_VERSION,
        "graph": {
            "tile_id": tile_id,
            "directed": False,
            "multigraph": False,
            "adjacency_definition": ADJACENCY_DEFINITION,
            "minimum_shared_boundary_pixels": 1,
            "node_count": 2,
            "edge_count": 1,
        },
        "nodes": [
            {
                "node_id": first_id,
                "instance_id": 1,
                "area_pixels": 1,
                "segment_type": {"id": 4, "name": "tree"},
                "geometry_type": geometries[1]["geometry_type"],
                "geometry": geometries[1],
                "stage2_metadata": copy.deepcopy(first_metadata),
            },
            {
                "node_id": second_id,
                "instance_id": 2,
                "area_pixels": 1,
                "segment_type": {"id": 7, "name": "buildings"},
                "geometry_type": geometries[2]["geometry_type"],
                "geometry": geometries[2],
                "stage2_metadata": {
                    "instance_id": 2,
                    "segment_id": "segment_b",
                    "area_pixels": 1,
                    "label": {"id": 7, "name": "buildings"},
                },
            },
        ],
        "edges": [
            {
                "edge_id": make_edge_id(tile_id, 1, 2),
                "source": first_id,
                "target": second_id,
                "relation": "adjacent",
                "adjacency_definition": ADJACENCY_DEFINITION,
                "shared_boundary_side_count": 3,
                "left_right_contact_count": 2,
                "top_bottom_contact_count": 1,
                "shared_boundary_length_pixels": 3,
                "shared_boundary_length_projected": None,
                "shared_boundary_projected_unit": None,
                "shared_boundary_projected_crs": None,
                "source_instance_id": 1,
                "target_instance_id": 2,
                "source_class_id": 4,
                "target_class_id": 7,
            }
        ],
    }

    validate_graph_document(document)
    assert document["nodes"][0]["stage2_metadata"] == first_metadata

    mismatched_metadata = copy.deepcopy(document)
    mismatched_metadata["nodes"][0]["stage2_metadata"]["instance_id"] = 2
    with pytest.raises(ValueError, match="does not preserve matching segmentation metadata"):
        validate_graph_document(mismatched_metadata)

    duplicate_edge = copy.deepcopy(document)
    duplicate_edge["graph"]["edge_count"] = 2
    duplicate_edge["edges"].append(copy.deepcopy(duplicate_edge["edges"][0]))
    with pytest.raises(ValueError, match="duplicate edge"):
        validate_graph_document(duplicate_edge)


def test_storage_and_visualizations_are_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    nodes, edges = _visual_graph()

    first_arrays = tmp_path / "first.npz"
    second_arrays = tmp_path / "second.npz"
    save_graph_arrays(first_arrays, nodes, edges)
    save_graph_arrays(second_arrays, nodes, edges)
    with np.load(first_arrays, allow_pickle=False) as first, np.load(
        second_arrays, allow_pickle=False
    ) as second:
        assert first.files == second.files
        for name in first.files:
            np.testing.assert_equal(first[name], second[name])
        assert first["edge_index"].tolist() == [[0], [1]]
        assert first["representative_points_tile_xy"].tolist() == [[8.5, 8.5], [24.5, 20.5]]

    first_csv = tmp_path / "first.csv"
    second_csv = tmp_path / "second.csv"
    save_adjacency_csv(first_csv, edges)
    save_adjacency_csv(second_csv, edges)
    assert first_csv.read_bytes() == second_csv.read_bytes()

    collection = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "id": nodes[0]["node_id"], "geometry": None}],
    }
    first_geojson = tmp_path / "first.geojson"
    second_geojson = tmp_path / "second.geojson"
    save_nodes_geojson(first_geojson, collection)
    save_nodes_geojson(second_geojson, collection)
    assert first_geojson.read_bytes() == second_geojson.read_bytes()

    from src.stage3.visualization import (
        render_graph_only,
        render_graph_overlay,
        save_graph_png,
        save_graph_svg,
    )

    background = np.full((32, 40, 3), 96, dtype=np.uint8)
    first_overlay = render_graph_overlay(background, nodes, edges)
    second_overlay = render_graph_overlay(background, nodes, edges)
    assert np.array_equal(first_overlay, second_overlay)
    assert first_overlay.shape == background.shape

    first_graph = render_graph_only(nodes, edges, image_shape=(32, 40))
    second_graph = render_graph_only(nodes, edges, image_shape=(32, 40))
    assert np.array_equal(first_graph, second_graph)
    assert first_graph.shape == background.shape

    first_png = save_graph_png(nodes, edges, tmp_path / "first.png", image_shape=(32, 40))
    second_png = save_graph_png(nodes, edges, tmp_path / "second.png", image_shape=(32, 40))
    assert first_png.read_bytes() == second_png.read_bytes()

    first_svg = save_graph_svg(nodes, edges, tmp_path / "first.svg", image_shape=(32, 40))
    second_svg = save_graph_svg(nodes, edges, tmp_path / "second.svg", image_shape=(32, 40))
    assert first_svg.read_bytes() == second_svg.read_bytes()


@pytest.mark.parametrize(
    ("case", "manifest", "error", "message"),
    [
        (
            "wrong_schema",
            {
                "schema_version": "satellitegraph.refinement.unsupported.v999",
                "status": "complete",
                "stale": False,
            },
            ValueError,
            "requires refinement schema",
        ),
        (
            "stale",
            {
                "schema_version": REFINEMENT_SCHEMA_VERSION,
                "status": "complete",
                "stale": True,
            },
            RuntimeError,
            "stale",
        ),
    ],
)
def test_stage3_rejects_wrong_or_stale_stage2_input(
    tmp_path: Path,
    case: str,
    manifest: dict,
    error: type[Exception],
    message: str,
) -> None:
    tile_id = f"scene__tile_{case}"
    stage2_root = tmp_path / "stage2"
    final = stage2_root / tile_id / "13_refinement"
    final.mkdir(parents=True)
    manifest = {**manifest, "tile_id": tile_id}
    (final / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    config = load_config(DEFAULT_CONFIG)
    config["stage2"]["output"]["root_directory"] = str(stage2_root)
    config["stage3"]["output"]["root_directory"] = str(tmp_path / "stage3")
    with pytest.raises(error, match=message):
        run_stage3(tile_id, project_root=tmp_path, config=config, show_progress=False)


def test_analysis_view_excludes_only_lineage_marked_micro_fragment() -> None:
    native = {
        "node_id": "tile::native",
        "area_pixels": 3,
        "segment_type": {"id": 7, "name": "buildings"},
        "refinement_metadata": {
            "component_count": 1,
            "parent_instance_id": 2,
            "refinement_history": [],
        },
    }
    fragment = {
        "node_id": "tile::fragment",
        "area_pixels": 3,
        "segment_type": {"id": 7, "name": "buildings"},
        "refinement_metadata": {
            "component_count": 2,
            "parent_instance_id": 8,
            "refinement_history": ["tile::operation::split_components::abc"],
            "lineage": {
                "refinement_operation_ids": [
                    "tile::operation::split_components::abc"
                ]
            },
        },
    }
    edges = [
        {
            "source": native["node_id"],
            "target": fragment["node_id"],
            "shared_boundary_side_count": 4,
        }
    ]

    selection = _annotate_analysis_view(
        [native, fragment],
        edges,
        {
            "minimum_display_node_area_pixels": 1,
            "minimum_display_unknown_area_pixels": 41,
            "minimum_display_topology_fragment_area_pixels": 41,
            "minimum_display_shared_boundary_pixels": 2,
        },
    )

    assert native["analysis_view"] == {"included": True, "exclusion_reasons": []}
    assert fragment["analysis_view"] == {
        "included": False,
        "exclusion_reasons": ["micro_topology_fragment_below_analysis_area"],
    }
    assert edges[0]["analysis_view"]["included"] is False
    assert selection["node_count"] == 1
    assert selection["edge_count"] == 0


def test_fake_stage2_to_stage3_public_loader_and_cli(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    project, tile_id = _make_fake_project(tmp_path)
    config = _fake_config(project)
    stage2_result = run_stage2(
        tile_id,
        project_root=project,
        config=config,
        sam_backend=FakeSamBackend(),
        semantic_backend=FakeSemanticBackend(),
        show_progress=False,
    )
    refined_result = apply_refinement(
        tile_id,
        project_root=project,
        config=config,
        decisions=None,
        show_progress=False,
    )

    result = run_stage3(
        tile_id,
        project_root=project,
        config=config,
        show_progress=False,
    )
    assert result.output_directory == project / "artifacts" / "stage3" / tile_id / "01_graph"
    assert result.graph["directed"] is False
    assert result.graph["multigraph"] is False
    assert result.graph["adjacency_definition"] == ADJACENCY_DEFINITION
    assert len(result.nodes) == len(refined_result.segments)
    assert result.edges
    assert result.analysis_graph["view"] == "analysis"
    assert len(result.analysis_nodes) <= len(result.nodes)
    assert len(result.analysis_edges) <= len(result.edges)
    assert all(Path(path).is_file() for path in result.artifacts.values())
    assert {path.name for path in result.output_directory.iterdir()} >= {
        "graph.json",
        "graph_arrays.npz",
        "graph_analysis.json",
        "graph_analysis_arrays.npz",
        "adjacency.csv",
        "nodes.geojson",
        "graph_overlay.png",
        "graph_spatial.png",
        "graph_spatial.svg",
        "stage3_overview.png",
        "refinement_graph_before_after.png",
        "report.json",
        "manifest.json",
        "phase_manifest.json",
    }
    for node, refined_record in zip(result.nodes, refined_result.segments):
        assert node["refinement_metadata"] == refined_record
        assert node["source_stage2_metadata"]
        assert node["node_id"] == make_node_id(tile_id, refined_record["segment_id"])
        assert node["geometry"]["coordinate_spaces"]["tile_pixel"]["available"]
        assert node["geometry"]["coordinate_spaces"]["projected"]["available"]
        assert node["geometry"]["coordinate_spaces"]["wgs84"]["available"]

    with np.load(result.output_directory / "graph_arrays.npz", allow_pickle=False) as arrays:
        assert arrays["edge_index"].shape == (2, len(result.edges))
        assert arrays["instance_ids"].tolist() == list(range(1, len(result.nodes) + 1))
        assert arrays["graph_view"].item() == "authoritative"
    analysis_document = json.loads(
        (result.output_directory / "graph_analysis.json").read_text(encoding="utf-8")
    )
    authoritative_document = json.loads(
        (result.output_directory / "graph.json").read_text(encoding="utf-8")
    )
    validate_analysis_graph_document(analysis_document, authoritative_document)
    assert result.analysis_nodes == [
        node
        for node in result.nodes
        if node["analysis_view"]["included"] is True
    ]
    assert result.analysis_edges == [
        edge
        for edge in result.edges
        if edge["analysis_view"]["included"] is True
    ]
    with np.load(
        result.output_directory / "graph_analysis_arrays.npz", allow_pickle=False
    ) as arrays:
        assert arrays["schema_version"].item() == ANALYSIS_SCHEMA_VERSION
        assert arrays["graph_view"].item() == "analysis"
        assert arrays["edge_index"].shape == (2, len(result.analysis_edges))
        assert arrays["instance_ids"].tolist() == [
            node["instance_id"] for node in result.analysis_nodes
        ]
    node_collection = json.loads((result.output_directory / "nodes.geojson").read_text(encoding="utf-8"))
    assert len(node_collection["features"]) == len(result.nodes)
    assert node_collection["coordinate_space"] == "wgs84"
    comparison = result.report["refinement_graph_comparison"]
    assert comparison["visual_diagnostic_only"] is True
    assert comparison["purple_semantic_class"] == "unknown"
    assert comparison["before"]["node_count"] == len(stage2_result.segments)
    assert comparison["after"]["node_count"] == len(refined_result.segments)

    reloaded = load_stage3_result(result.output_directory)
    assert reloaded.graph == result.graph
    assert reloaded.nodes == result.nodes
    assert reloaded.edges == result.edges
    assert reloaded.analysis_graph == result.analysis_graph
    assert reloaded.analysis_nodes == result.analysis_nodes
    assert reloaded.analysis_edges == result.analysis_edges
    assert reloaded.report == result.report
    assert reloaded.artifacts == result.artifacts

    graph_bytes = (result.output_directory / "graph.json").read_bytes()
    analysis_graph_bytes = (result.output_directory / "graph_analysis.json").read_bytes()
    visualization_bytes = {
        name: (result.output_directory / name).read_bytes()
        for name in (
            "graph_overlay.png",
            "graph_spatial.png",
            "graph_spatial.svg",
            "stage3_overview.png",
            "refinement_graph_before_after.png",
        )
    }
    config_document = copy.deepcopy(config)
    config_document.pop("_config_path", None)
    config_path = project / "test-config.yaml"
    config_path.write_text(yaml.safe_dump(config_document, sort_keys=False), encoding="utf-8")

    capsys.readouterr()
    cli_main(
        [
            "stage3",
            "run",
            "--tile-id",
            tile_id,
            "--project-root",
            str(project),
            "--config",
            str(config_path),
            "--no-progress",
        ]
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "complete"
    assert summary["schema_version"] == SCHEMA_VERSION
    assert summary["tile_id"] == tile_id
    assert summary["node_count"] == len(result.nodes)
    assert summary["edge_count"] == len(result.edges)
    assert summary["analysis_node_count"] == len(result.analysis_nodes)
    assert summary["analysis_edge_count"] == len(result.analysis_edges)
    assert (result.output_directory / "graph.json").read_bytes() == graph_bytes
    assert (
        result.output_directory / "graph_analysis.json"
    ).read_bytes() == analysis_graph_bytes
    for name, expected in visualization_bytes.items():
        assert (result.output_directory / name).read_bytes() == expected
