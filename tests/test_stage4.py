from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest
from shapely.geometry import MultiPolygon, Point, box, mapping, shape

from src.cli import parser
from src.config import stage4_config_hash
from src.refinement import apply_refinement
from src.stage2.artifacts import atomic_json
from src.stage3 import run_stage3
from src.stage4 import combine_tile_graphs, load_stage4_result
from src.stage4.pipeline import _TileInput, _assign_ownership
from src.stage4.pipeline import _analysis_view
from tests.test_refinement import _write_stage2_fixture


def test_stage4_rejects_adjacency_between_regions_that_do_not_touch() -> None:
    from src.stage4.schema import SCHEMA_VERSION, make_combined_edge_id, validate_combined_graph

    nodes = [
        {"node_id": name, "instance_id": index, "member_node_ids": [name],
         "geometry_source_pixel": mapping(geometry), "segment_type": {"id": 1},
         "area_pixels": 100}
        for index, (name, geometry) in enumerate(
            (("left", box(0, 0, 10, 10)), ("right", box(20, 0, 30, 10))), 1
        )
    ]
    edge = {"edge_id": make_combined_edge_id("left", "right"),
            "source": "left", "target": "right", "same_semantic_class": True,
            "shared_boundary_length_pixels": 10,
            "contact_centroid_source_pixel_xy": [15, 5]}
    document = {"schema_version": SCHEMA_VERSION,
                "graph": {"directed": False, "multigraph": False,
                          "node_count": 2, "edge_count": 1,
                          "selected_tile_ids": ["tile_00000"]},
                "nodes": nodes, "edges": [edge]}
    with pytest.raises(ValueError, match="do not share a boundary"):
        validate_combined_graph(document)


def test_disjoint_contacts_use_a_real_boundary_anchor() -> None:
    from src.stage4.pipeline import _contact_anchor

    first = box(0, 0, 10, 10)
    second = MultiPolygon([box(-1, 2, 0, 3), box(10, 7, 11, 8)])
    contact = first.boundary.intersection(second.boundary)
    assert Point(contact.centroid).distance(first.boundary) > 1
    anchor = Point(_contact_anchor(contact))
    assert anchor.distance(first.boundary) < 1e-9
    assert anchor.distance(second.boundary) < 1e-9


def test_overlay_connects_the_two_adjacent_node_markers_via_their_contact() -> None:
    from src.stage4.pipeline import _node_to_node_edge_line

    positions = {"large": [0.0, 0.0], "small": [100.0, 0.0]}
    edge = {"source": "large", "target": "small",
            "contact_anchor_source_pixel_xy": [50.0, 0.0]}
    assert _node_to_node_edge_line(edge, positions) == [
        [0.0, 0.0], [50.0, 0.0], [100.0, 0.0]
    ]


def _build_tile(tmp_path: Path, tile_id: str, *, x0: int, config: dict | None = None) -> dict:
    final, generated_config = _write_stage2_fixture(tmp_path, tile_id=tile_id)
    selected = generated_config if config is None else config
    metadata_path = final / "tile_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "source_image_id": "shared-scene",
            "global_x0": x0,
            "global_y0": 0,
            "width": 4,
            "height": 4,
            "source_affine_transform": None,
            "tile_affine_transform": None,
            "crs": None,
            "crs_wkt": None,
        }
    )
    atomic_json(metadata_path, metadata)
    apply_refinement(
        tile_id,
        project_root=tmp_path,
        config=selected,
        workflow_mode="automatic",
        show_progress=False,
    )
    run_stage3(tile_id, project_root=tmp_path, config=selected, show_progress=False)
    return selected


def test_stage4_merges_duplicate_overlap_nodes_and_owns_pixels_once(tmp_path: Path) -> None:
    first = "shared-scene__tile_00000"
    second = "shared-scene__tile_00001"
    config = _build_tile(tmp_path, first, x0=0)
    _build_tile(tmp_path, second, x0=0, config=config)

    result = combine_tile_graphs(
        [second, first],
        project_root=tmp_path,
        config=config,
        output_name="two-identical-tiles",
    )

    assert result.graph["selected_tile_ids"] == [first, second]
    assert result.report["input_node_count"] == 6
    assert result.report["combined_node_count"] == 3
    assert result.report["accepted_overlap_match_count"] == 3
    assert result.report["removed_duplicate_node_count"] == 3
    assert result.report["overlap_pixels_owned_once"] is True
    assert all(node["duplicate_member_count"] == 2 for node in result.nodes)
    assert all(len(node["source_tile_ids"]) == 2 for node in result.nodes)
    assert sum(node["area_pixels"] for node in result.nodes) == 16
    assert all(len(edge["contact_centroid_source_pixel_xy"]) == 2 for edge in result.edges)
    by_id = {node["node_id"]: shape(node["geometry_source_pixel"]) for node in result.nodes}
    for edge in result.edges:
        anchor = Point(edge["contact_anchor_source_pixel_xy"])
        assert anchor.distance(by_id[edge["source"]].boundary) < 1e-9
        assert anchor.distance(by_id[edge["target"]].boundary) < 1e-9
    overlay_path = Path(result.artifacts["graph_overlay"])
    assert overlay_path.is_file()
    with Image.open(overlay_path) as overlay:
        assert overlay.width > overlay.height > 0
    assert set(result.report["visualization_backgrounds"]) == {first, second}
    geometries = [shape(node["geometry_source_pixel"]) for node in result.nodes]
    assert all(
        geometries[first_index].intersection(geometries[second_index]).area == 0
        for first_index in range(len(geometries))
        for second_index in range(first_index + 1, len(geometries))
    )

    loaded = load_stage4_result(result.output_directory)
    assert loaded.graph == result.graph
    assert loaded.nodes == result.nodes
    assert loaded.edges == result.edges


def test_stage4_combines_nonoverlapping_selected_tiles_without_false_matches(
    tmp_path: Path,
) -> None:
    first = "shared-scene__tile_00000"
    second = "shared-scene__tile_00001"
    config = _build_tile(tmp_path, first, x0=0)
    _build_tile(tmp_path, second, x0=10, config=config)

    result = combine_tile_graphs(
        [first, second],
        project_root=tmp_path,
        config=config,
    )

    assert result.report["accepted_overlap_match_count"] == 0
    assert result.report["combined_node_count"] == 6
    assert result.graph["source_pixel_extent_xyxy"] == [0, 0, 14, 4]
    assert all(node["duplicate_member_count"] == 1 for node in result.nodes)


def test_stage4_reconciles_three_overlapping_tiles_into_one_identity_per_object(
    tmp_path: Path,
) -> None:
    tile_ids = [f"shared-scene__tile_{index:05d}" for index in range(3)]
    config = _build_tile(tmp_path, tile_ids[0], x0=0)
    for tile_id in tile_ids[1:]:
        _build_tile(tmp_path, tile_id, x0=0, config=config)

    result = combine_tile_graphs(tile_ids[::-1], project_root=tmp_path, config=config)

    assert result.graph["selected_tile_ids"] == tile_ids
    assert result.report["input_node_count"] == 9
    assert result.report["combined_node_count"] == 3
    assert result.report["removed_duplicate_node_count"] == 6
    assert all(node["duplicate_member_count"] == 3 for node in result.nodes)
    assert sum(node["area_pixels"] for node in result.nodes) == 16


def test_stage4_partial_overlap_is_an_exact_nonoverlapping_pixel_partition(
    tmp_path: Path,
) -> None:
    first = "shared-scene__tile_00000"
    second = "shared-scene__tile_00001"
    config = _build_tile(tmp_path, first, x0=0)
    _build_tile(tmp_path, second, x0=2, config=config)

    result = combine_tile_graphs([first, second], project_root=tmp_path, config=config)

    geometries = [shape(node["geometry_source_pixel"]) for node in result.nodes]
    assert sum(node["area_pixels"] for node in result.nodes) == 24
    assert result.report["non_owner_overlap_pixel_count"] == 8
    assert all(
        geometries[first_index].intersection(geometries[second_index]).area == 0
        for first_index in range(len(geometries))
        for second_index in range(first_index + 1, len(geometries))
    )


def test_stage4_overlap_ownership_never_discards_the_only_valid_prediction(
    tmp_path: Path,
) -> None:
    import numpy as np

    def tile(tile_id: str, values: list[list[int]]) -> _TileInput:
        instance_map = np.asarray(values, dtype=np.int32)
        height, width = instance_map.shape
        return _TileInput(
            tile_id=tile_id,
            result=None,
            directory=tmp_path,
            instance_map=instance_map,
            confidence_map=np.ones(instance_map.shape, dtype=np.float32),
            nodes_by_instance={},
            x0=0,
            y0=0,
            width=width,
            height=height,
            ownership=np.ones(instance_map.shape, dtype=bool),
        )

    first = tile("first", [[0, 1], [1, 1]])
    second = tile("second", [[1, 1], [1, 1]])
    _assign_ownership([first, second])

    assert not first.ownership[0, 0]
    assert second.ownership[0, 0]
    represented_owner_count = (
        first.ownership & (first.instance_map > 0)
    ).astype(np.uint8) + (second.ownership & (second.instance_map > 0)).astype(np.uint8)
    assert np.all(represented_owner_count == 1)


def test_stage4_cli_accepts_any_repeated_tile_selection() -> None:
    arguments = parser().parse_args(
        [
            "stage4",
            "combine",
            "--tile-id",
            "scene__tile_00000",
            "--tile-id",
            "scene__tile_00001",
            "--name",
            "my-selection",
        ]
    )
    assert arguments.tile_ids == ["scene__tile_00000", "scene__tile_00001"]
    assert arguments.name == "my-selection"


def test_analysis_preserves_small_known_nodes_and_large_unknowns_without_bridging():
    nodes = [
        {"node_id": "known", "area_pixels": 1, "segment_type": {"id": 7}},
        {"node_id": "micro", "area_pixels": 40, "segment_type": {"id": 8}},
        {"node_id": "large", "area_pixels": 41, "segment_type": {"id": 8}},
    ]
    edges = [
        {"source": "known", "target": "micro", "shared_boundary_side_count": 4},
        {"source": "micro", "target": "large", "shared_boundary_side_count": 4},
    ]
    document = {"graph": {}, "nodes": nodes, "edges": edges}
    view, report = _analysis_view(document, {})
    assert [node["node_id"] for node in view["nodes"]] == ["known", "large"]
    assert view["edges"] == []
    assert report["excluded_unknown_node_count"] == 1
    assert report["semantic_pixels_changed"] == 0
    assert len(document["nodes"]) == 3


def test_stage4_configuration_hash_rejects_an_unsafe_match_threshold(
    tmp_path: Path,
) -> None:
    config = _build_tile(tmp_path, "shared-scene__tile_00000", x0=0)
    config["stage4"]["overlap"]["minimum_intersection_over_union"] = 1.1
    try:
        stage4_config_hash(config)
    except ValueError as exc:
        assert "between zero and one" in str(exc)
    else:  # pragma: no cover - protects against validation being removed.
        raise AssertionError("unsafe Stage 4 threshold was accepted")
