"""Persistence helpers for Stage 3 graph artifacts."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.stage2.artifacts import atomic_npz

from .schema import ADJACENCY_DEFINITION, SCHEMA_VERSION


def save_graph_json(path: Path, document: dict[str, Any]) -> None:
    """Write the large canonical graph atomically without pretty-print bloat."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                document,
                stream,
                ensure_ascii=False,
                separators=(",", ":"),
                default=_json_default,
            )
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def save_graph_arrays(
    path: Path,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    *,
    schema_version: str = SCHEMA_VERSION,
    tile_id: str | None = None,
    graph_view: str = "authoritative",
    projected_measurements: Mapping[str, Any] | None = None,
) -> None:
    """Write compact, pickle-free arrays for efficient downstream graph loading."""

    node_index = {str(node["node_id"]): index for index, node in enumerate(nodes)}
    node_ids = np.asarray([str(node["node_id"]) for node in nodes], dtype=np.str_)
    instance_ids = np.asarray([int(node["instance_id"]) for node in nodes], dtype=np.int32)
    class_ids = np.asarray([int(node["segment_type"]["id"]) for node in nodes], dtype=np.int16)
    area_pixels = np.asarray([int(node["area_pixels"]) for node in nodes], dtype=np.int64)
    representative_points = np.asarray(
        [node["representative_point_xy"] if "representative_point_xy" in node
         else node["representative_point"]["tile_pixel"] for node in nodes], dtype=np.float64
    ).reshape(len(nodes), 2)
    if edges:
        edge_index = np.asarray(
            [[node_index[str(edge["source"])], node_index[str(edge["target"])]] for edge in edges],
            dtype=np.int32,
        ).T
        endpoint_instance_ids = np.asarray(
            [[int(edge["source_instance_id"]), int(edge["target_instance_id"])] for edge in edges],
            dtype=np.int32,
        ).T
        shared_sides = np.asarray(
            [int(edge["shared_boundary_side_count"]) for edge in edges], dtype=np.int64
        )
        shared_pixel_lengths = np.asarray(
            [float(edge["shared_boundary_length_pixels"]) for edge in edges], dtype=np.float64
        )
        shared_projected_lengths = np.asarray(
            [
                np.nan
                if edge.get("shared_boundary_length_projected") is None
                else float(edge["shared_boundary_length_projected"])
                for edge in edges
            ],
            dtype=np.float64,
        )
        projected_available = np.isfinite(shared_projected_lengths)
    else:
        edge_index = np.empty((2, 0), dtype=np.int32)
        endpoint_instance_ids = np.empty((2, 0), dtype=np.int32)
        shared_sides = np.empty(0, dtype=np.int64)
        shared_pixel_lengths = np.empty(0, dtype=np.float64)
        shared_projected_lengths = np.empty(0, dtype=np.float64)
        projected_available = np.empty(0, dtype=bool)
    projected_unit = next(
        (str(edge["shared_boundary_projected_unit"]) for edge in edges if edge.get("shared_boundary_projected_unit")),
        "",
    )
    projected_crs = next(
        (str(edge["shared_boundary_projected_crs"]) for edge in edges if edge.get("shared_boundary_projected_crs")),
        "",
    )
    if projected_measurements is not None:
        projected_unit = str(projected_measurements.get("unit") or "")
        projected_crs = str(projected_measurements.get("crs") or "")
    graph_tile_id = (
        str(tile_id)
        if tile_id is not None
        else (str(nodes[0].get("tile_id", "")) if nodes else "")
    )
    atomic_npz(
        path,
        schema_version=np.asarray(schema_version, dtype=np.str_),
        tile_id=np.asarray(graph_tile_id, dtype=np.str_),
        graph_view=np.asarray(graph_view, dtype=np.str_),
        adjacency_definition=np.asarray(ADJACENCY_DEFINITION, dtype=np.str_),
        projected_length_unit=np.asarray(projected_unit, dtype=np.str_),
        projected_crs=np.asarray(projected_crs, dtype=np.str_),
        node_ids=node_ids,
        instance_ids=instance_ids,
        class_ids=class_ids,
        area_pixels=area_pixels,
        representative_points_tile_xy=representative_points,
        edge_index=edge_index,
        endpoint_instance_ids=endpoint_instance_ids,
        shared_boundary_side_count=shared_sides,
        shared_boundary_length_pixels=shared_pixel_lengths,
        shared_boundary_length_projected=shared_projected_lengths,
        shared_boundary_projected_available=projected_available,
    )


def save_adjacency_csv(path: Path, edges: list[dict[str, Any]]) -> None:
    """Write a compact, human-inspectable adjacency edge table atomically."""

    fieldnames = [
        "edge_id",
        "source",
        "target",
        "source_instance_id",
        "target_instance_id",
        "same_semantic_class",
        "shared_boundary_side_count",
        "left_right_contact_count",
        "top_bottom_contact_count",
        "shared_boundary_length_pixels",
        "shared_boundary_length_projected",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for edge in edges:
                writer.writerow({key: edge.get(key) for key in fieldnames})
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def save_nodes_geojson(path: Path, collection: dict[str, Any]) -> None:
    """Persist the GIS node collection without pretty-printing polygon arrays."""

    save_graph_json(path, collection)


__all__ = ["save_adjacency_csv", "save_graph_arrays", "save_graph_json", "save_nodes_geojson"]
