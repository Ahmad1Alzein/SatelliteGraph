from __future__ import annotations

import copy

import numpy as np
import pytest

from src.stage3.visualization import (
    GraphStyle,
    _contact_deviation,
    _edge_polyline,
    _prepare_graph,
    _select_display_graph,
    render_graph_only,
)


def _node(
    node_id: str,
    x: float,
    y: float,
    *,
    area: float = 10,
    class_id: int = 4,
) -> dict:
    return {
        "id": node_id,
        "x": x,
        "y": y,
        "area_pixels": area,
        "class_id": class_id,
    }


def test_edge_uses_measured_contact_as_visual_proof_anchor() -> None:
    nodes = [_node("left", 2, 8), _node("right", 18, 8)]
    edges = [
        {
            "source": "left",
            "target": "right",
            "shared_boundary_side_count": 6,
            "contact_coordinates": {
                "tile_pixel": {
                    "available": True,
                    "centroid_xy": [7.5, 3.0],
                }
            },
        }
    ]

    visual_nodes, visual_edges = _prepare_graph(nodes, edges)

    assert len(visual_edges) == 1
    edge = visual_edges[0]
    assert (edge.contact_x, edge.contact_y) == pytest.approx((7.5, 3.0))
    positions = {node.token: (node.x, node.y) for node in visual_nodes}
    assert _edge_polyline(edge, positions, GraphStyle()) == [
        (2.0, 8.0),
        (7.5, 3.0),
        (18.0, 8.0),
    ]
    assert _contact_deviation(edge, positions) == pytest.approx(5.0)


def test_contact_routing_can_be_disabled_without_losing_contact_metadata() -> None:
    nodes = [_node("a", 1, 2), _node("b", 9, 2)]
    edges = [
        {
            "source": "a",
            "target": "b",
            "shared_boundary_length_pixels": 2,
            "contact_centroid_tile_pixel_xy": [4.0, 7.0],
        }
    ]
    visual_nodes, visual_edges = _prepare_graph(nodes, edges)
    positions = {node.token: (node.x, node.y) for node in visual_nodes}

    assert _edge_polyline(
        visual_edges[0],
        positions,
        GraphStyle(route_edges_through_contacts=False),
    ) == [(1.0, 2.0), (9.0, 2.0)]
    assert visual_edges[0].contact_y == 7.0


def test_display_suppression_hides_only_requested_noise() -> None:
    nodes = [
        _node("unknown-speck", 1, 1, area=2, class_id=8),
        _node("small-building", 4, 1, area=1, class_id=7),
        _node("tree", 8, 1, area=20, class_id=4),
    ]
    edges = [
        {"source": "unknown-speck", "target": "tree", "shared_boundary_side_count": 8},
        {"source": "small-building", "target": "tree", "shared_boundary_side_count": 3},
    ]
    original_nodes, original_edges = copy.deepcopy(nodes), copy.deepcopy(edges)
    visual_nodes, visual_edges = _prepare_graph(nodes, edges)

    selected_nodes, selected_edges = _select_display_graph(
        visual_nodes,
        visual_edges,
        GraphStyle(minimum_display_unknown_area_pixels=3),
    )

    assert [node.token for node in selected_nodes] == ["small-building", "tree"]
    assert [(edge.source, edge.target) for edge in selected_edges] == [
        ("small-building", "tree")
    ]
    assert len(visual_nodes) == 3
    assert len(visual_edges) == 2
    assert nodes == original_nodes
    assert edges == original_edges


def test_topology_fragment_filter_preserves_native_small_known_object() -> None:
    native = _node("native-building", 2, 2, area=3, class_id=7)
    fragment = _node("split-building-fragment", 6, 2, area=3, class_id=7)
    fragment["refinement_metadata"] = {
        "component_count": 2,
        "component_index": 1,
        "parent_instance_id": 19,
        "refinement_history": ["tile::operation::split_components::abc"],
        "lineage": {
            "refinement_operation_ids": [
                "tile::operation::split_components::abc"
            ]
        },
    }
    visual_nodes, visual_edges = _prepare_graph(
        [native, fragment],
        [
            {
                "source": "native-building",
                "target": "split-building-fragment",
                "shared_boundary_side_count": 4,
            }
        ],
    )

    selected_nodes, selected_edges = _select_display_graph(
        visual_nodes,
        visual_edges,
        GraphStyle(minimum_display_topology_fragment_area_pixels=41),
    )

    assert [node.token for node in selected_nodes] == ["native-building"]
    assert selected_edges == []


def test_weak_contact_filter_is_visual_only_and_deterministic() -> None:
    nodes = [_node("a", 2, 2), _node("b", 6, 2), _node("c", 10, 2)]
    edges = [
        {"source": "a", "target": "b", "shared_boundary_side_count": 1},
        {"source": "b", "target": "c", "shared_boundary_side_count": 4},
    ]
    visual_nodes, visual_edges = _prepare_graph(nodes, edges)
    style = GraphStyle(minimum_display_shared_boundary_pixels=2)

    first = _select_display_graph(visual_nodes, visual_edges, style)
    second = _select_display_graph(visual_nodes, visual_edges, style)

    assert [node.token for node in first[0]] == ["a", "b", "c"]
    assert [(edge.source, edge.target) for edge in first[1]] == [("b", "c")]
    assert first == second
    assert len(visual_edges) == 2


def test_contact_aware_filtered_render_is_repeatable() -> None:
    nodes = [
        _node("noise", 2, 8, area=1, class_id=8),
        _node("a", 4, 4, area=12, class_id=4),
        _node("b", 16, 14, area=16, class_id=7),
    ]
    edges = [
        {"source": "noise", "target": "a", "shared_boundary_side_count": 1},
        {
            "source": "a",
            "target": "b",
            "shared_boundary_side_count": 5,
            "contact_coordinates": {
                "tile_pixel": {"available": True, "centroid_xy": [10.0, 5.0]}
            },
        },
    ]
    style = GraphStyle(
        minimum_display_unknown_area_pixels=2,
        minimum_display_shared_boundary_pixels=2,
    )

    first = render_graph_only(nodes, edges, image_shape=(20, 20), style=style)
    second = render_graph_only(nodes, edges, image_shape=(20, 20), style=style)

    assert first.shape == (20, 20, 3)
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"minimum_display_unknown_area_pixels": -1}, "non-negative"),
        ({"minimum_display_topology_fragment_area_pixels": -1}, "non-negative"),
        ({"minimum_display_shared_boundary_pixels": float("nan")}, "finite"),
        ({"contact_anchor_alpha": 1.1}, "between 0 and 1"),
        ({"contact_anchor_minimum_deviation_pixels": -0.1}, "non-negative"),
        ({"show_contact_anchors": 1}, "must be a boolean"),
    ],
)
def test_noise_style_controls_are_validated(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        GraphStyle(**kwargs)
