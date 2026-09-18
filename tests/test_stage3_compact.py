"""Tile graphs retain useful features without carrying model/audit payloads."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import shape

from src.stage3.adjacency import build_adjacency_edges
from src.stage3.geometry import build_node_geometries
from src.stage3.pipeline import _annotate_analysis_view, _build_edges, _build_nodes
from src.stage3.pipeline import Stage3Result
from src.stage3.visualization import _node_is_topology_fragment


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_summary_reports_the_loaded_schema_version(tmp_path, version):
    schema = f"satellitegraph.stage3.tile-adjacency-graph.{version}"
    result = Stage3Result(
        tile_id="scene__tile_00000", output_directory=tmp_path,
        graph={}, nodes=[], edges=[], analysis_graph={}, analysis_nodes=[],
        analysis_edges=[], report={"schema_version": schema}, artifacts={},
    )
    assert result.summary()["schema_version"] == schema


@pytest.mark.parametrize("georeferenced", [False, True])
def test_tile_builders_discard_model_payloads_before_export(georeferenced):
    instance_map = np.array([[1, 1, 2], [1, 1, 2]], dtype=np.int32)
    metadata = {"source_image_id": "scene", "global_x0": 10, "global_y0": 20}
    if georeferenced:
        metadata.update(tile_affine_transform=[2, 0, 500000, 0, -2, 4000000], crs="EPSG:32633")
    records = [
        {"instance_id": 1, "segment_id": "a", "area_pixels": 4,
         "label": {"id": 4, "name": "tree"}, "mean_semantic_confidence": 0.9},
        {"instance_id": 2, "segment_id": "b", "area_pixels": 2,
         "label": {"id": 7, "name": "buildings"}, "mean_semantic_confidence": 0.8},
    ]
    records[0]["lineage"] = {"refinement_operation_ids": ["tile::split_components::1"]}
    geometries = build_node_geometries(instance_map, metadata)
    result = SimpleNamespace(tile_id="scene__tile_00000", tile_metadata=metadata, segments=records)
    nodes = _build_nodes(result, geometries)

    # Extra SAM, semantic-model and refinement diagnostics must never enlarge
    # the tile graph, even when upstream records acquire new debug fields.
    noisy_records = copy.deepcopy(records)
    for record in noisy_records:
        record.update(predicted_iou=0.99, stability_score=0.98,
                      quality={"raw_scores": [0.5] * 1000},
                      provenance={"sam": "raw-mask" * 1000},
                      stage2_metadata={"model_trace": "x" * 10000},
                      stage2_metadata_records=[{"debug": "y" * 10000}],
                      future_model_diagnostics={"logits": [0.2] * 1000})
        record["label"]["evidence"] = {"votes": [100] * 1000}
    result.segments = noisy_records
    assert _build_nodes(result, geometries) == nodes
    assert len(json.dumps(nodes)) < 2500
    assert shape(nodes[0]["geometry"]).area == 4
    assert shape(nodes[1]["geometry"]).area == 2
    assert nodes[0]["mean_semantic_confidence"] == 0.9
    assert nodes[0]["representative_point_xy"] == [0.5, 0.5]
    assert _node_is_topology_fragment(nodes[0]) is True
    assert _node_is_topology_fragment(nodes[1]) is False

    raw_edges = build_adjacency_edges(instance_map, node_geometries=geometries, tile_metadata=metadata)
    edges = _build_edges(result.tile_id, nodes, raw_edges, geometries)
    assert len(edges) == 1
    assert edges[0]["shared_boundary_side_count"] == 2
    assert edges[0]["contact_centroid_tile_pixel_xy"] == [2, 1]
    assert edges[0]["shared_boundary_length_projected"] == (4 if georeferenced else None)
    assert "contact_coordinates" not in edges[0]
    assert "contact_fractions" not in edges[0]
    assert "source_class" not in edges[0]
    assert "shared_boundary_projected_crs" not in edges[0]
    assert "shared_boundary_projected_unit" not in edges[0]
    _annotate_analysis_view(nodes, edges, {"minimum_display_topology_fragment_area_pixels": 5})
    assert nodes[0]["analysis_view"]["included"] is False
    assert nodes[1]["analysis_view"]["included"] is True
    assert edges[0]["analysis_view"]["included"] is False
