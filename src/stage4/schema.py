"""Schema and deterministic identifiers for combined multi-tile graphs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from shapely.geometry import Point, shape


SCHEMA_VERSION = "satellitegraph.stage4.combined-graph.v1"
MATCH_SCHEMA_VERSION = "satellitegraph.stage4.overlap-matches.v1"


def stable_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_combined_node_id(source_image_id: str, member_node_ids: list[str]) -> str:
    identity = {"source_image_id": source_image_id, "members": sorted(member_node_ids)}
    return f"{source_image_id}::combined-node::{stable_digest(identity)[:20]}"


def make_combined_edge_id(source: str, target: str) -> str:
    first, second = sorted((str(source), str(target)))
    return f"combined-edge::{stable_digest([first, second])[:20]}"


def validate_combined_graph(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported Stage 4 combined graph schema")
    graph = document.get("graph")
    nodes = document.get("nodes")
    edges = document.get("edges")
    if not isinstance(graph, Mapping) or not isinstance(nodes, list) or not isinstance(edges, list):
        raise ValueError("Stage 4 graph, nodes, and edges have invalid types")
    if graph.get("directed") is not False or graph.get("multigraph") is not False:
        raise ValueError("Stage 4 graph must be undirected and simple")
    if int(graph.get("node_count", -1)) != len(nodes) or int(graph.get("edge_count", -1)) != len(edges):
        raise ValueError("Stage 4 graph counts disagree with its records")
    selected_tiles = graph.get("selected_tile_ids")
    if not isinstance(selected_tiles, list) or not selected_tiles or selected_tiles != sorted(set(selected_tiles)):
        raise ValueError("Stage 4 selected tile IDs must be non-empty, unique, and sorted")

    node_ids: set[str] = set()
    instance_ids: set[int] = set()
    class_by_node: dict[str, int] = {}
    boundary_by_node: dict[str, Any] = {}
    for node in nodes:
        if not isinstance(node, Mapping):
            raise ValueError("Every Stage 4 node must be a mapping")
        node_id = str(node.get("node_id", ""))
        instance_id = int(node.get("instance_id", 0))
        members = node.get("member_node_ids")
        geometry = node.get("geometry_source_pixel")
        segment_type = node.get("segment_type")
        if not node_id or node_id in node_ids or instance_id <= 0 or instance_id in instance_ids:
            raise ValueError("Stage 4 node identifiers must be unique and positive")
        if not isinstance(members, list) or not members or members != sorted(set(members)):
            raise ValueError(f"Stage 4 node {node_id!r} has invalid members")
        if not isinstance(geometry, Mapping) or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            raise ValueError(f"Stage 4 node {node_id!r} has invalid source-pixel geometry")
        if not isinstance(segment_type, Mapping) or segment_type.get("id") is None:
            raise ValueError(f"Stage 4 node {node_id!r} has no semantic type")
        if int(node.get("area_pixels", 0)) <= 0:
            raise ValueError(f"Stage 4 node {node_id!r} has no owned pixels")
        node_ids.add(node_id)
        instance_ids.add(instance_id)
        class_by_node[node_id] = int(segment_type["id"])
        boundary_by_node[node_id] = shape(geometry).boundary
    if instance_ids != set(range(1, len(nodes) + 1)):
        raise ValueError("Stage 4 instance IDs must be contiguous")

    seen_pairs: set[tuple[str, str]] = set()
    edge_ids: set[str] = set()
    for edge in edges:
        if not isinstance(edge, Mapping):
            raise ValueError("Every Stage 4 edge must be a mapping")
        source, target = str(edge.get("source", "")), str(edge.get("target", ""))
        pair = tuple(sorted((source, target)))
        if source not in node_ids or target not in node_ids or source == target or pair in seen_pairs:
            raise ValueError("Stage 4 edge endpoints are missing, repeated, or self-linked")
        edge_id = str(edge.get("edge_id", ""))
        if edge_id != make_combined_edge_id(source, target) or edge_id in edge_ids:
            raise ValueError("Stage 4 edge ID is invalid or duplicated")
        if float(edge.get("shared_boundary_length_pixels", 0)) <= 0:
            raise ValueError("Stage 4 adjacency must have a positive shared boundary")
        actual_contact = boundary_by_node[source].intersection(boundary_by_node[target])
        if actual_contact.length < 0.5:
            raise ValueError("Stage 4 adjacency endpoints do not share a boundary")
        if abs(actual_contact.length - float(edge["shared_boundary_length_pixels"])) > 0.51:
            raise ValueError("Stage 4 adjacency length disagrees with the shared boundary")
        contact = edge.get("contact_centroid_source_pixel_xy")
        if (
            not isinstance(contact, list)
            or len(contact) != 2
            or any(not isinstance(value, (int, float)) for value in contact)
        ):
            raise ValueError("Stage 4 adjacency must record its source-pixel contact")
        anchor = edge.get("contact_anchor_source_pixel_xy")
        if anchor is not None and (
            not isinstance(anchor, list)
            or len(anchor) != 2
            or any(not isinstance(value, (int, float)) for value in anchor)
        ):
            raise ValueError("Stage 4 adjacency contact anchor must be a source-pixel point")
        if anchor is not None and Point(anchor).distance(actual_contact) > 1e-6:
            raise ValueError("Stage 4 adjacency contact anchor must lie on the shared boundary")
        if bool(edge.get("same_semantic_class")) != (class_by_node[source] == class_by_node[target]):
            raise ValueError("Stage 4 edge semantic flag disagrees with its endpoints")
        seen_pairs.add(pair)
        edge_ids.add(edge_id)


__all__ = [
    "MATCH_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "make_combined_edge_id",
    "make_combined_node_id",
    "stable_digest",
    "validate_combined_graph",
]
