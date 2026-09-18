"""Versioned schema helpers for tile-local Stage 3 graphs."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any


SCHEMA_VERSION = "satellitegraph.stage3.tile-adjacency-graph.v3"
ANALYSIS_SCHEMA_VERSION = "satellitegraph.stage3.analysis-view.v2"
SUPPORTED_SCHEMA_VERSIONS = {SCHEMA_VERSION, "satellitegraph.stage3.tile-adjacency-graph.v2"}
SUPPORTED_ANALYSIS_SCHEMA_VERSIONS = {ANALYSIS_SCHEMA_VERSION, "satellitegraph.stage3.analysis-view.v1"}
STAGE2_SCHEMA_VERSION = "satellitegraph.stage2.full-segmentation.v1"
REFINEMENT_SCHEMA_VERSION = "satellitegraph.stage2-refinement.v1"
ADJACENCY_DEFINITION = "shared_pixel_edge_4_connected"

COMPACT_NODE_FIELDS = {
    "node_id", "segment_id", "instance_id", "segment_type", "area_pixels",
    "mean_semantic_confidence", "geometry_type", "geometry", "centroid_xy",
    "bbox_xyxy", "representative_point_xy", "perimeter_pixels",
    "touches_tile_boundary", "is_topology_fragment", "degree",
    "connected_component_id", "is_isolated", "analysis_view",
}
COMPACT_EDGE_FIELDS = {
    "edge_id", "source", "target", "source_instance_id", "target_instance_id",
    "shared_boundary_side_count", "left_right_contact_count", "top_bottom_contact_count",
    "shared_boundary_length_pixels", "shared_boundary_length_projected",
    "contact_fraction_of_source_pixel_perimeter", "contact_fraction_of_target_pixel_perimeter",
    "contact_centroid_tile_pixel_xy", "same_semantic_class", "centroid_distance_pixels",
    "analysis_view",
}


def _validate_xy(value: Any, name: str, size: int = 2) -> None:
    if (not isinstance(value, (list, tuple)) or len(value) != size
            or not all(isinstance(item, (int, float)) and math.isfinite(item) for item in value)):
        raise ValueError(f"Invalid {name}")


def _validate_compact_node(node: Mapping[str, Any], tile_id: str) -> None:
    from shapely.geometry import Point, shape

    if set(node) != COMPACT_NODE_FIELDS:
        raise ValueError(f"Node {node['node_id']!r} has invalid compact fields")
    if not node["segment_id"] or node["node_id"] != make_node_id(tile_id, node["segment_id"]):
        raise ValueError("Stage 3 node identity disagrees with segment ID")
    segment_type = node["segment_type"]
    if (not isinstance(segment_type, Mapping) or set(segment_type) != {"id", "name"}
            or segment_type["id"] is None or not segment_type["name"]):
        raise ValueError("Stage 3 node has no semantic segment type")
    geometry = node["geometry"]
    if (not isinstance(geometry, Mapping) or set(geometry) != {"type", "coordinates"}
            or geometry["type"] not in {"Polygon", "MultiPolygon"}
            or geometry["type"] != node["geometry_type"]):
        raise ValueError("Stage 3 node has invalid tile-pixel geometry")
    polygon = shape(geometry)
    if polygon.is_empty or not polygon.is_valid:
        raise ValueError("Stage 3 node has invalid tile-pixel geometry")
    if int(node["area_pixels"]) <= 0 or not math.isclose(polygon.area, node["area_pixels"], abs_tol=1e-7):
        raise ValueError("Stage 3 node has inconsistent pixel area")
    if not math.isclose(polygon.length, node["perimeter_pixels"], abs_tol=1e-7):
        raise ValueError("Stage 3 node has inconsistent pixel perimeter")
    _validate_xy(node["representative_point_xy"], "node representative point")
    _validate_xy(node["centroid_xy"], "node centroid")
    _validate_xy(node["bbox_xyxy"], "node bounding box", 4)
    if not polygon.covers(Point(node["representative_point_xy"])):
        raise ValueError("Stage 3 node representative point is outside its geometry")


def make_node_id(tile_id: str, segment_id: str) -> str:
    """Return the stable, tile-qualified identifier for one Stage 2 segment."""

    return f"{tile_id}::{segment_id}"


def make_edge_id(tile_id: str, first_instance_id: int, second_instance_id: int) -> str:
    """Return a deterministic identifier for one unordered adjacency edge."""

    first, second = sorted((int(first_instance_id), int(second_instance_id)))
    return f"{tile_id}::adj::{first:06d}--{second:06d}"


def validate_graph_document(document: Mapping[str, Any]) -> None:
    """Validate the structural invariants of a persisted Stage 3 graph."""

    if document.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported Stage 3 graph schema: {document.get('schema_version')!r}")
    metadata = document.get("graph")
    nodes = document.get("nodes")
    edges = document.get("edges")
    if not isinstance(metadata, Mapping):
        raise ValueError("Stage 3 graph metadata must be a mapping")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("Stage 3 graph nodes and edges must be lists")
    if metadata.get("directed") is not False or metadata.get("multigraph") is not False:
        raise ValueError("Stage 3 graph must be undirected and non-multigraph")
    if metadata.get("adjacency_definition") != ADJACENCY_DEFINITION:
        raise ValueError("Stage 3 graph has an unsupported adjacency definition")
    if int(metadata.get("minimum_shared_boundary_pixels", -1)) != 1:
        raise ValueError("Stage 3 exact adjacency requires minimum_shared_boundary_pixels=1")
    if int(metadata.get("node_count", -1)) != len(nodes):
        raise ValueError("Stage 3 graph node_count does not match nodes")
    if int(metadata.get("edge_count", -1)) != len(edges):
        raise ValueError("Stage 3 graph edge_count does not match edges")

    compact = document["schema_version"] == SCHEMA_VERSION
    node_ids: set[str] = set()
    instance_ids: set[int] = set()
    node_id_by_instance: dict[int, str] = {}
    node_by_id: dict[str, Mapping[str, Any]] = {}
    for node in nodes:
        if not isinstance(node, Mapping):
            raise ValueError("Every Stage 3 node must be a mapping")
        node_id = str(node.get("node_id", ""))
        instance_id = int(node.get("instance_id", 0))
        if not node_id or node_id in node_ids:
            raise ValueError(f"Duplicate or empty Stage 3 node ID: {node_id!r}")
        if instance_id <= 0 or instance_id in instance_ids:
            raise ValueError(f"Duplicate or invalid Stage 3 instance ID: {instance_id}")
        if compact:
            _validate_compact_node(node, str(metadata.get("tile_id", "")))
        else:
            segmentation_metadata = node.get("refinement_metadata", node.get("stage2_metadata"))
            if (
                not isinstance(segmentation_metadata, Mapping)
                or int(segmentation_metadata.get("instance_id", 0)) != instance_id
            ):
                raise ValueError(
                    f"Node {node_id!r} does not preserve matching segmentation metadata"
                )
            area_pixels = int(node.get("area_pixels", 0))
            if area_pixels <= 0 or int(segmentation_metadata.get("area_pixels", 0)) != area_pixels:
                raise ValueError(f"Node {node_id!r} has inconsistent pixel area")
            segment_type = node.get("segment_type")
            if not isinstance(segment_type, Mapping) or segment_type.get("id") is None or not segment_type.get("name"):
                raise ValueError(f"Node {node_id!r} has no semantic segment type")
            geometry = node.get("geometry")
            if not isinstance(geometry, Mapping):
                raise ValueError(f"Node {node_id!r} has no exact geometry record")
            geometry_type = geometry.get("geometry_type")
            if geometry_type not in {"Polygon", "MultiPolygon"} or node.get("geometry_type") != geometry_type:
                raise ValueError(f"Node {node_id!r} has an invalid geometry type")
            spaces = geometry.get("coordinate_spaces")
            if not isinstance(spaces, Mapping) or set(spaces) != {
                "tile_pixel",
                "source_pixel",
                "projected",
                "wgs84",
            }:
                raise ValueError(f"Node {node_id!r} has incomplete coordinate spaces")
            tile_space = spaces["tile_pixel"]
            if (
                not isinstance(tile_space, Mapping)
                or tile_space.get("available") is not True
                or not isinstance(tile_space.get("geometry"), Mapping)
                or tile_space.get("geometry_type") != geometry_type
                or tile_space.get("representative_point_xy") is None
            ):
                raise ValueError(f"Node {node_id!r} has invalid tile-pixel geometry")
            metrics = geometry.get("metrics")
            if not isinstance(metrics, Mapping) or int(metrics.get("area_pixels", 0)) != area_pixels:
                raise ValueError(f"Node {node_id!r} has inconsistent geometry metrics")
        node_ids.add(node_id)
        instance_ids.add(instance_id)
        node_id_by_instance[instance_id] = node_id
        node_by_id[node_id] = node
    expected_instances = set(range(1, len(nodes) + 1))
    if instance_ids != expected_instances:
        raise ValueError("Stage 3 instance IDs must be contiguous from 1 through node_count")

    seen_pairs: set[tuple[str, str]] = set()
    edge_ids: set[str] = set()
    tile_id = str(metadata.get("tile_id", ""))
    for edge in edges:
        if not isinstance(edge, Mapping):
            raise ValueError("Every Stage 3 edge must be a mapping")
        source, target = str(edge.get("source", "")), str(edge.get("target", ""))
        if source not in node_ids or target not in node_ids:
            raise ValueError(f"Stage 3 edge has a missing endpoint: {(source, target)!r}")
        if source == target:
            raise ValueError(f"Stage 3 graph contains a self-loop: {source!r}")
        edge_id = str(edge.get("edge_id", ""))
        if not edge_id or edge_id in edge_ids:
            raise ValueError(f"Stage 3 graph contains a duplicate edge ID: {edge_id!r}")
        source_instance = int(edge.get("source_instance_id", 0))
        target_instance = int(edge.get("target_instance_id", 0))
        if source_instance >= target_instance:
            raise ValueError(f"Stage 3 edge instance endpoints are not canonically ordered: {edge_id!r}")
        if (
            node_id_by_instance.get(source_instance) != source
            or node_id_by_instance.get(target_instance) != target
        ):
            raise ValueError(f"Stage 3 edge endpoint IDs disagree with instance IDs: {edge_id!r}")
        if edge_id != make_edge_id(tile_id, source_instance, target_instance):
            raise ValueError(f"Stage 3 edge ID is not deterministic: {edge_id!r}")
        pair = tuple(sorted((source, target)))
        if pair in seen_pairs:
            raise ValueError(f"Stage 3 graph contains a duplicate edge: {pair!r}")
        if not compact and edge.get("relation") != "adjacent":
            raise ValueError(f"Stage 3 edge has an unsupported relation: {edge.get('relation')!r}")
        if not compact and edge.get("adjacency_definition") != ADJACENCY_DEFINITION:
            raise ValueError("Stage 3 edge has an unsupported adjacency definition")
        shared_count = int(edge.get("shared_boundary_side_count", 0))
        left_right = int(edge.get("left_right_contact_count", -1))
        top_bottom = int(edge.get("top_bottom_contact_count", -1))
        if shared_count <= 0:
            raise ValueError(f"Stage 3 edge has no shared pixel side: {pair!r}")
        if left_right < 0 or top_bottom < 0 or left_right + top_bottom != shared_count:
            raise ValueError(f"Stage 3 edge orientation counts disagree: {edge_id!r}")
        if float(edge.get("shared_boundary_length_pixels", 0)) != float(shared_count):
            raise ValueError(f"Stage 3 edge pixel boundary length disagrees: {edge_id!r}")
        projected_length = edge.get("shared_boundary_length_projected")
        projected_metadata = metadata.get("projected_measurements", {}) if compact else {}
        projected_unit = projected_metadata.get("unit") if compact else edge.get("shared_boundary_projected_unit")
        projected_crs = projected_metadata.get("crs") if compact else edge.get("shared_boundary_projected_crs")
        if projected_length is None:
            if projected_unit is not None:
                raise ValueError(f"Stage 3 edge has a projected unit without a length: {edge_id!r}")
        elif (
            not isinstance(projected_length, (int, float))
            or float(projected_length) <= 0
            or not projected_unit
            or not projected_crs
        ):
            raise ValueError(f"Stage 3 edge has incomplete projected length metadata: {edge_id!r}")
        source_node = node_by_id[source]
        target_node = node_by_id[target]
        if not compact and (
            edge.get("source_class_id") != source_node["segment_type"]["id"]
            or edge.get("target_class_id") != target_node["segment_type"]["id"]
        ):
            raise ValueError(f"Stage 3 edge endpoint classes disagree: {edge_id!r}")
        if compact:
            if set(edge) != COMPACT_EDGE_FIELDS:
                raise ValueError(f"Stage 3 edge has invalid compact fields: {edge_id!r}")
            _validate_xy(edge["contact_centroid_tile_pixel_xy"], "edge contact centroid")
            if edge["same_semantic_class"] != (source_node["segment_type"]["id"] == target_node["segment_type"]["id"]):
                raise ValueError(f"Stage 3 edge endpoint classes disagree: {edge_id!r}")
        seen_pairs.add(pair)
        edge_ids.add(edge_id)


def validate_analysis_graph_document(
    document: Mapping[str, Any],
    authoritative_document: Mapping[str, Any] | None = None,
) -> None:
    """Validate the deterministic low-noise view of an authoritative graph.

    The analysis artifact is a selected subgraph, not a replacement for the
    exact graph. Original node/edge records and instance IDs are retained, so
    the selected instance IDs are intentionally allowed to be non-contiguous.
    When the authoritative document is supplied, record equality and order are
    checked against its ``analysis_view.included`` flags.
    """

    if document.get("schema_version") not in SUPPORTED_ANALYSIS_SCHEMA_VERSIONS:
        raise ValueError(
            "Unsupported Stage 3 analysis graph schema: "
            f"{document.get('schema_version')!r}"
        )
    metadata = document.get("graph")
    nodes = document.get("nodes")
    edges = document.get("edges")
    if not isinstance(metadata, Mapping):
        raise ValueError("Stage 3 analysis graph metadata must be a mapping")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("Stage 3 analysis graph nodes and edges must be lists")
    if metadata.get("view") != "analysis" or metadata.get("selected_subgraph") is not True:
        raise ValueError("Stage 3 analysis graph must identify the selected analysis view")
    if metadata.get("directed") is not False or metadata.get("multigraph") is not False:
        raise ValueError("Stage 3 analysis graph must be undirected and non-multigraph")
    if metadata.get("adjacency_definition") != ADJACENCY_DEFINITION:
        raise ValueError("Stage 3 analysis graph has an unsupported adjacency definition")
    if int(metadata.get("connectivity", -1)) != 4:
        raise ValueError("Stage 3 analysis graph must use four-neighbor connectivity")
    if int(metadata.get("node_count", -1)) != len(nodes):
        raise ValueError("Stage 3 analysis graph node_count does not match nodes")
    if int(metadata.get("edge_count", -1)) != len(edges):
        raise ValueError("Stage 3 analysis graph edge_count does not match edges")

    node_ids: set[str] = set()
    for node in nodes:
        if not isinstance(node, Mapping):
            raise ValueError("Every Stage 3 analysis node must be a mapping")
        node_id = str(node.get("node_id", ""))
        if not node_id or node_id in node_ids:
            raise ValueError(f"Duplicate or empty Stage 3 analysis node ID: {node_id!r}")
        selection = node.get("analysis_view")
        if not isinstance(selection, Mapping) or selection.get("included") is not True:
            raise ValueError(f"Analysis node is not selected by analysis_view: {node_id!r}")
        node_ids.add(node_id)

    edge_ids: set[str] = set()
    for edge in edges:
        if not isinstance(edge, Mapping):
            raise ValueError("Every Stage 3 analysis edge must be a mapping")
        edge_id = str(edge.get("edge_id", ""))
        if not edge_id or edge_id in edge_ids:
            raise ValueError(f"Duplicate or empty Stage 3 analysis edge ID: {edge_id!r}")
        if str(edge.get("source", "")) not in node_ids or str(edge.get("target", "")) not in node_ids:
            raise ValueError(f"Stage 3 analysis edge has an excluded endpoint: {edge_id!r}")
        selection = edge.get("analysis_view")
        if not isinstance(selection, Mapping) or selection.get("included") is not True:
            raise ValueError(f"Analysis edge is not selected by analysis_view: {edge_id!r}")
        edge_ids.add(edge_id)

    source_graph = metadata.get("source_graph")
    if not isinstance(source_graph, Mapping) or source_graph.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError("Stage 3 analysis graph has invalid authoritative source metadata")
    if source_graph.get("artifact") != "graph.json" or not source_graph.get("graph_id"):
        raise ValueError("Stage 3 analysis graph has no authoritative graph identity")
    source_checksum = str(source_graph.get("sha256", ""))
    if len(source_checksum) != 64 or any(character not in "0123456789abcdef" for character in source_checksum):
        raise ValueError("Stage 3 analysis graph has an invalid authoritative source checksum")

    if authoritative_document is None:
        return
    validate_graph_document(authoritative_document)
    if source_graph["schema_version"] != authoritative_document["schema_version"]:
        raise ValueError("Stage 3 analysis source schema disagrees with graph.json")
    expected_schema = (ANALYSIS_SCHEMA_VERSION if authoritative_document["schema_version"] == SCHEMA_VERSION
                       else "satellitegraph.stage3.analysis-view.v1")
    if document["schema_version"] != expected_schema:
        raise ValueError("Stage 3 analysis schema disagrees with graph.json")
    authoritative_metadata = authoritative_document["graph"]
    expected_nodes = [
        node
        for node in authoritative_document["nodes"]
        if node.get("analysis_view", {}).get("included") is True
    ]
    expected_edges = [
        edge
        for edge in authoritative_document["edges"]
        if edge.get("analysis_view", {}).get("included") is True
    ]
    if nodes != expected_nodes:
        raise ValueError("Stage 3 analysis nodes disagree with authoritative analysis_view flags")
    if edges != expected_edges:
        raise ValueError("Stage 3 analysis edges disagree with authoritative analysis_view flags")
    if metadata.get("tile_id") != authoritative_metadata.get("tile_id"):
        raise ValueError("Stage 3 analysis and authoritative graph tile IDs disagree")
    if source_graph.get("graph_id") != authoritative_metadata.get("id"):
        raise ValueError("Stage 3 analysis authoritative graph ID disagrees")
    if int(metadata.get("authoritative_node_count", -1)) != len(authoritative_document["nodes"]):
        raise ValueError("Stage 3 analysis authoritative node count disagrees")
    if int(metadata.get("authoritative_edge_count", -1)) != len(authoritative_document["edges"]):
        raise ValueError("Stage 3 analysis authoritative edge count disagrees")
    if metadata.get("selection") != authoritative_metadata.get("analysis_view"):
        raise ValueError("Stage 3 analysis selection metadata disagrees with graph.json")
    if metadata.get("projected_measurements") != authoritative_metadata.get("projected_measurements"):
        raise ValueError("Stage 3 analysis projected measurements disagree with graph.json")
    if (
        int(source_graph.get("node_count", -1)) != len(authoritative_document["nodes"])
        or int(source_graph.get("edge_count", -1)) != len(authoritative_document["edges"])
    ):
        raise ValueError("Stage 3 analysis source graph counts disagree")
    statistics = metadata.get("graph_statistics")
    if not isinstance(statistics, Mapping) or (
        int(statistics.get("node_count", -1)) != len(nodes)
        or int(statistics.get("edge_count", -1)) != len(edges)
    ):
        raise ValueError("Stage 3 analysis graph statistics disagree with its records")


__all__ = [
    "ADJACENCY_DEFINITION",
    "ANALYSIS_SCHEMA_VERSION",
    "REFINEMENT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "STAGE2_SCHEMA_VERSION",
    "make_edge_id",
    "make_node_id",
    "validate_graph_document",
    "validate_analysis_graph_document",
]
