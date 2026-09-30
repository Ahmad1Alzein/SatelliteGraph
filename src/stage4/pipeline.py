"""Combine exact tile graphs and reconcile duplicate overlap predictions."""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

_MATPLOTLIB_CACHE = Path(tempfile.gettempdir()) / "satellitegraph-matplotlib"
_MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from PIL import Image
from shapely import affinity
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

from src.config import DEFAULT_CONFIG, load_config, select_stage4_config, stable_hash
from src.stage2.artifacts import atomic_json, atomic_npz, checksum, manifest, now
from src.stage3 import load_stage3_result
from src.stage3.geometry import polygonize_instance_map
from src.stage3.storage import save_graph_json

from .schema import (
    MATCH_SCHEMA_VERSION,
    SCHEMA_VERSION,
    make_combined_edge_id,
    make_combined_node_id,
    stable_digest,
    validate_combined_graph,
)


@dataclass(slots=True)
class Stage4Result:
    output_directory: Path
    graph: dict[str, Any]
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    matches: list[dict[str, Any]]
    report: dict[str, Any]
    artifacts: dict[str, str]

    def summary(self) -> dict[str, Any]:
        return {
            "status": "complete",
            "schema_version": SCHEMA_VERSION,
            "output_directory": str(self.output_directory),
            "source_image_id": self.graph["source_image_id"],
            "tile_count": len(self.graph["selected_tile_ids"]),
            "input_node_count": self.report["input_node_count"],
            "combined_node_count": len(self.nodes),
            "combined_edge_count": len(self.edges),
            "analysis_graph": copy.deepcopy(self.report.get("analysis_graph", {})),
            "accepted_overlap_match_count": self.report["accepted_overlap_match_count"],
            "removed_duplicate_node_count": self.report["removed_duplicate_node_count"],
            "overlap_pixels_owned_once": self.report["overlap_pixels_owned_once"],
            "artifacts": copy.deepcopy(self.artifacts),
        }


@dataclass(slots=True)
class _TileInput:
    tile_id: str
    result: Any
    directory: Path
    instance_map: np.ndarray
    confidence_map: np.ndarray
    nodes_by_instance: dict[int, dict[str, Any]]
    x0: int
    y0: int
    width: int
    height: int
    ownership: np.ndarray

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return self.x0, self.y0, self.x0 + self.width, self.y0 + self.height


class _UnionFind:
    def __init__(self, values: Sequence[str], tile_by_value: Mapping[str, str]) -> None:
        self.parent = {value: value for value in values}
        self.tiles = {value: {str(tile_by_value[value])} for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union_if_tile_unique(self, first: str, second: str) -> bool:
        a, b = self.find(first), self.find(second)
        if a == b:
            return True
        if self.tiles[a] & self.tiles[b]:
            return False
        keep, drop = sorted((a, b))
        self.parent[drop] = keep
        self.tiles[keep] |= self.tiles[drop]
        return True


def combine_tile_graphs(
    tile_ids: Sequence[str],
    *,
    project_root: str | Path = ".",
    config: Mapping[str, Any] | str | Path | None = None,
    output_name: str | None = None,
) -> Stage4Result:
    """Combine any selected Stage 3 tile graphs into one source-coordinate graph."""

    selected = sorted({str(value) for value in tile_ids})
    if not selected or any(not value for value in selected):
        raise ValueError("Stage 4 requires at least one non-empty tile ID")
    if len(selected) != len(tile_ids):
        raise ValueError("Stage 4 tile IDs must not contain duplicates")
    root = Path(project_root).expanduser().resolve()
    project_config = _load_project_config(config)
    cfg = select_stage4_config(project_config)
    started = now()
    tiles = [_load_tile(root, tile_id, cfg) for tile_id in selected]
    source_image_id = _validate_common_frame(tiles)

    matches = _overlap_candidates(tiles, cfg["overlap"])
    union, accepted = _resolve_matches(tiles, matches)
    _assign_ownership(tiles)
    nodes, geometries, root_to_node, dropped = _build_combined_nodes(
        tiles, union, source_image_id
    )
    edges = _build_combined_edges(nodes, geometries, tiles, union, root_to_node)
    graph = _graph_metadata(
        tiles,
        nodes,
        edges,
        source_image_id,
        cfg,
        accepted_count=accepted,
    )
    document = {"schema_version": SCHEMA_VERSION, "graph": graph, "nodes": nodes, "edges": edges}
    validate_combined_graph(document)

    directory = _output_directory(root, source_image_id, selected, output_name, cfg)
    artifacts, report = _persist(
        directory,
        document,
        matches,
        tiles,
        cfg,
        accepted,
        dropped,
        started,
    )
    return load_stage4_result(directory)


def load_stage4_result(output_directory: str | Path) -> Stage4Result:
    directory = Path(output_directory).expanduser().resolve()
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Stage 4 manifest not found: {manifest_path}")
    public = json.loads(manifest_path.read_text(encoding="utf-8"))
    if public.get("schema_version") != SCHEMA_VERSION or public.get("status") != "complete":
        raise ValueError("Stage 4 manifest is incomplete or unsupported")
    graph_path, matches_path, report_path = (
        directory / "graph.json",
        directory / "overlap_matches.json",
        directory / "report.json",
    )
    document = json.loads(graph_path.read_text(encoding="utf-8"))
    validate_combined_graph(document)
    match_document = json.loads(matches_path.read_text(encoding="utf-8"))
    if match_document.get("schema_version") != MATCH_SCHEMA_VERSION:
        raise ValueError("Unsupported Stage 4 overlap match schema")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    artifacts = {str(key): str(value) for key, value in public.get("artifacts", {}).items()}
    for key, path_text in artifacts.items():
        path = Path(path_text)
        if not path.is_file() or checksum(path) != public["artifact_checksums"].get(key):
            raise RuntimeError(f"Stage 4 artifact is missing or changed: {path}")
    return Stage4Result(
        output_directory=directory,
        graph=document["graph"],
        nodes=document["nodes"],
        edges=document["edges"],
        matches=match_document["matches"],
        report=report,
        artifacts=artifacts,
    )


def _load_project_config(config: Mapping[str, Any] | str | Path | None) -> dict[str, Any]:
    if config is None:
        return load_config(DEFAULT_CONFIG)
    if isinstance(config, (str, Path)):
        return load_config(config)
    return copy.deepcopy(dict(config))


def _load_tile(root: Path, tile_id: str, cfg: Mapping[str, Any]) -> _TileInput:
    graph_root = _resolve(root, str(cfg["input"]["root_directory"]))
    directory = graph_root / tile_id / str(cfg["input"]["phase_directory"])
    result = load_stage3_result(directory)
    metadata = result.graph.get("tile_metadata", {})
    x0, y0 = metadata.get("global_x0"), metadata.get("global_y0")
    width, height = metadata.get("width"), metadata.get("height")
    if any(value is None for value in (x0, y0, width, height)):
        raise ValueError(f"Tile {tile_id!r} has no exact source-pixel extent")
    refinement_dir = Path(str(result.graph["source_refinement"]["output_directory"]))
    maps_path = refinement_dir / "segmentation.npz"
    if not maps_path.is_file():
        raise FileNotFoundError(f"Refinement map for tile {tile_id!r} is missing")
    with np.load(maps_path, allow_pickle=False) as archive:
        instance_map = np.asarray(archive["instance_map"], dtype=np.int32).copy()
        confidence_map = np.asarray(archive["confidence_map"], dtype=np.float32).copy()
    if instance_map.shape != (int(height), int(width)):
        raise ValueError(f"Tile {tile_id!r} map shape disagrees with metadata")
    nodes = {int(node["instance_id"]): copy.deepcopy(node) for node in result.nodes}
    if set(nodes) != set(range(1, len(nodes) + 1)):
        raise ValueError(f"Tile {tile_id!r} graph nodes are not contiguous")
    return _TileInput(
        tile_id=tile_id,
        result=result,
        directory=directory,
        instance_map=instance_map,
        confidence_map=confidence_map,
        nodes_by_instance=nodes,
        x0=int(x0),
        y0=int(y0),
        width=int(width),
        height=int(height),
        ownership=np.ones(instance_map.shape, dtype=bool),
    )


def _validate_common_frame(tiles: Sequence[_TileInput]) -> str:
    source_ids = {str(tile.result.graph.get("source_image_id", "")) for tile in tiles}
    if len(source_ids) != 1 or not next(iter(source_ids)):
        raise ValueError("All Stage 4 tiles must belong to one source image")
    transforms = {
        tuple(tile.result.graph.get("tile_metadata", {}).get("source_affine_transform") or [])
        for tile in tiles
    }
    crs_values = {str(tile.result.graph.get("tile_metadata", {}).get("crs") or "") for tile in tiles}
    if len(transforms) != 1 or len(crs_values) != 1:
        raise ValueError("All Stage 4 tiles must share one source affine transform and CRS")
    return next(iter(source_ids))


def _intersection_bounds(first: _TileInput, second: _TileInput) -> tuple[int, int, int, int] | None:
    ax0, ay0, ax1, ay1 = first.bounds
    bx0, by0, bx1, by1 = second.bounds
    bounds = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    return bounds if bounds[2] > bounds[0] and bounds[3] > bounds[1] else None


def _overlap_slices(tile: _TileInput, bounds: tuple[int, int, int, int]) -> tuple[slice, slice]:
    x0, y0, x1, y1 = bounds
    return slice(y0 - tile.y0, y1 - tile.y0), slice(x0 - tile.x0, x1 - tile.x0)


def _overlap_candidates(tiles: Sequence[_TileInput], cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    minimum_pixels = int(cfg["minimum_intersection_pixels"])
    for first_index, first in enumerate(tiles):
        for second in tiles[first_index + 1 :]:
            bounds = _intersection_bounds(first, second)
            if bounds is None:
                continue
            first_view = first.instance_map[_overlap_slices(first, bounds)]
            second_view = second.instance_map[_overlap_slices(second, bounds)]
            max_second = len(second.nodes_by_instance) + 1
            encoded = first_view.astype(np.int64) * max_second + second_view.astype(np.int64)
            keys, counts = np.unique(encoded, return_counts=True)
            first_areas = np.bincount(first_view.ravel(), minlength=len(first.nodes_by_instance) + 1)
            second_areas = np.bincount(second_view.ravel(), minlength=len(second.nodes_by_instance) + 1)
            pair_rows: list[dict[str, Any]] = []
            for key, raw_count in zip(keys.tolist(), counts.tolist()):
                first_id, second_id = divmod(int(key), max_second)
                intersection = int(raw_count)
                if first_id <= 0 or second_id <= 0 or intersection < minimum_pixels:
                    continue
                first_area, second_area = int(first_areas[first_id]), int(second_areas[second_id])
                union_area = first_area + second_area - intersection
                coverage_first = intersection / first_area
                coverage_second = intersection / second_area
                ios = intersection / min(first_area, second_area)
                iou = intersection / union_area
                first_node, second_node = first.nodes_by_instance[first_id], second.nodes_by_instance[second_id]
                first_class = int(first_node["segment_type"]["id"])
                second_class = int(second_node["segment_type"]["id"])
                same_class = first_class == second_class
                ios_floor = float(
                    cfg["unknown_minimum_intersection_over_smaller"]
                    if first_class == 8 and second_class == 8
                    else cfg["minimum_intersection_over_smaller"]
                )
                reasons: list[str] = []
                if bool(cfg["require_same_class"]) and not same_class:
                    reasons.append("semantic_class_conflict")
                if ios < ios_floor:
                    reasons.append("intersection_over_smaller_below_minimum")
                if iou < float(cfg["minimum_intersection_over_union"]):
                    reasons.append("intersection_over_union_below_minimum")
                if min(coverage_first, coverage_second) < float(cfg["minimum_bidirectional_coverage"]):
                    reasons.append("bidirectional_coverage_below_minimum")
                score = 0.45 * ios + 0.35 * iou + 0.20 * min(coverage_first, coverage_second)
                pair_rows.append(
                    {
                        "match_id": f"match::{stable_digest([first.tile_id, first_id, second.tile_id, second_id])[:20]}",
                        "first_tile_id": first.tile_id,
                        "first_instance_id": first_id,
                        "first_node_id": str(first_node["node_id"]),
                        "second_tile_id": second.tile_id,
                        "second_instance_id": second_id,
                        "second_node_id": str(second_node["node_id"]),
                        "class_id": first_class if same_class else None,
                        "class_name": str(first_node["segment_type"]["name"]) if same_class else None,
                        "same_semantic_class": same_class,
                        "intersection_pixels": intersection,
                        "first_visible_overlap_pixels": first_area,
                        "second_visible_overlap_pixels": second_area,
                        "coverage_of_first": coverage_first,
                        "coverage_of_second": coverage_second,
                        "intersection_over_smaller": ios,
                        "intersection_over_union": iou,
                        "score": score,
                        "overlap_bounds_source_pixel": list(bounds),
                        "eligible_before_assignment": not reasons,
                        "selected": False,
                        "reasons": reasons,
                    }
                )
            eligible = [row for row in pair_rows if row["eligible_before_assignment"]]
            best_first: dict[str, str] = {}
            best_second: dict[str, str] = {}
            for row in sorted(eligible, key=lambda value: (-value["score"], value["match_id"])):
                best_first.setdefault(row["first_node_id"], row["match_id"])
                best_second.setdefault(row["second_node_id"], row["match_id"])
            for row in pair_rows:
                mutual = (
                    best_first.get(row["first_node_id"]) == row["match_id"]
                    and best_second.get(row["second_node_id"]) == row["match_id"]
                )
                row["mutual_best"] = mutual
                if row["eligible_before_assignment"] and bool(cfg["require_mutual_best"]) and not mutual:
                    row["reasons"].append("not_mutual_best_match")
            candidates.extend(pair_rows)
    candidates.sort(key=lambda row: row["match_id"])
    return candidates


def _resolve_matches(
    tiles: Sequence[_TileInput], matches: list[dict[str, Any]]
) -> tuple[_UnionFind, int]:
    tile_by_node = {
        str(node["node_id"]): tile.tile_id
        for tile in tiles
        for node in tile.nodes_by_instance.values()
    }
    union = _UnionFind(sorted(tile_by_node), tile_by_node)
    eligible = [row for row in matches if not row["reasons"]]
    accepted = 0
    for row in sorted(eligible, key=lambda value: (-float(value["score"]), value["match_id"])):
        if union.union_if_tile_unique(row["first_node_id"], row["second_node_id"]):
            row["selected"] = True
            accepted += 1
        else:
            row["reasons"].append("would_create_multiple_members_from_one_tile")
    return union, accepted


def _edge_centrality(height: int, width: int) -> np.ndarray:
    rows = np.arange(height, dtype=np.float32)[:, None]
    cols = np.arange(width, dtype=np.float32)[None, :]
    distance = np.minimum.reduce(
        [
            np.broadcast_to(rows + 0.5, (height, width)),
            np.broadcast_to(cols + 0.5, (height, width)),
            np.broadcast_to(height - rows - 0.5, (height, width)),
            np.broadcast_to(width - cols - 0.5, (height, width)),
        ]
    )
    return distance / max(1.0, 0.5 * min(height, width))


def _assign_ownership(tiles: Sequence[_TileInput]) -> None:
    scores = {tile.tile_id: _edge_centrality(tile.height, tile.width) for tile in tiles}
    for index, tile in enumerate(tiles):
        owned = np.ones(tile.instance_map.shape, dtype=bool)
        for other_index, other in enumerate(tiles):
            if tile is other:
                continue
            bounds = _intersection_bounds(tile, other)
            if bounds is None:
                continue
            tile_slice = _overlap_slices(tile, bounds)
            other_slice = _overlap_slices(other, bounds)
            own_score = scores[tile.tile_id][tile_slice]
            other_score = scores[other.tile_id][other_slice]
            own_has_segment = tile.instance_map[tile_slice] > 0
            other_has_segment = other.instance_map[other_slice] > 0
            # A prediction can only win a pixel that it actually represents.
            # Among two represented pixels, centrality/confidence gives one
            # deterministic owner. If neither represents it, no geometry is emitted.
            loses = other_has_segment & ~own_has_segment
            comparable = own_has_segment == other_has_segment
            loses |= comparable & (other_score > own_score)
            tied_position = np.isclose(other_score, own_score)
            own_confidence = tile.confidence_map[tile_slice]
            other_confidence = other.confidence_map[other_slice]
            loses |= comparable & tied_position & (other_confidence > own_confidence)
            tied_evidence = tied_position & np.isclose(other_confidence, own_confidence)
            loses |= comparable & tied_evidence & (other_index < index)
            owned[tile_slice] &= ~loses
        tile.ownership = owned
    # Exact-once assertion over every pairwise overlap. A triple overlap is also
    # safe because each tile compared against every competitor using one total order.
    for index, first in enumerate(tiles):
        for second in tiles[index + 1 :]:
            bounds = _intersection_bounds(first, second)
            if bounds is None:
                continue
            count = first.ownership[_overlap_slices(first, bounds)].astype(np.uint8)
            count += second.ownership[_overlap_slices(second, bounds)].astype(np.uint8)
            if np.any(count > 1):
                raise RuntimeError("Stage 4 overlap ownership assigned pixels more than once")


def _build_combined_nodes(
    tiles: Sequence[_TileInput], union: _UnionFind, source_image_id: str
) -> tuple[list[dict[str, Any]], list[Any], dict[str, str], int]:
    members_by_root: dict[str, list[tuple[_TileInput, dict[str, Any]]]] = {}
    for tile in tiles:
        for node in tile.nodes_by_instance.values():
            root = union.find(str(node["node_id"]))
            members_by_root.setdefault(root, []).append((tile, node))
    ordered_roots = sorted(members_by_root, key=lambda value: min(str(row[1]["node_id"]) for row in members_by_root[value]))
    temporary_id = {root: index + 1 for index, root in enumerate(ordered_roots)}
    pieces: dict[str, list[Any]] = {root: [] for root in ordered_roots}
    owned_counts: dict[tuple[str, str], int] = {}
    for tile in tiles:
        local = np.zeros(tile.instance_map.shape, dtype=np.int32)
        for instance_id, node in tile.nodes_by_instance.items():
            root = union.find(str(node["node_id"]))
            mask = tile.ownership & (tile.instance_map == instance_id)
            count = int(mask.sum())
            if count:
                local[mask] = temporary_id[root]
                owned_counts[(root, tile.tile_id)] = count
        for raw_id, geometry in polygonize_instance_map(local).items():
            root = ordered_roots[int(raw_id) - 1]
            pieces[root].append(affinity.translate(shape(geometry), xoff=tile.x0, yoff=tile.y0))

    nodes: list[dict[str, Any]] = []
    geometries: list[Any] = []
    root_to_node: dict[str, str] = {}
    dropped = 0
    for root in ordered_roots:
        if not pieces[root]:
            dropped += 1
            continue
        geometry = _polygonal_union(pieces[root])
        members = sorted(members_by_root[root], key=lambda row: row[1]["node_id"])
        class_ids = {int(node["segment_type"]["id"]) for _, node in members}
        if len(class_ids) != 1:
            raise RuntimeError("A Stage 4 identity group contains conflicting semantic classes")
        class_id = next(iter(class_ids))
        class_name = str(members[0][1]["segment_type"]["name"])
        member_ids = [str(node["node_id"]) for _, node in members]
        node_id = make_combined_node_id(source_image_id, member_ids)
        weights = [owned_counts.get((root, tile.tile_id), 0) for tile, _ in members]
        confidences = [node.get("mean_semantic_confidence") for _, node in members]
        valid_confidence = [
            (float(value), weight)
            for value, weight in zip(confidences, weights)
            if value is not None and weight > 0
        ]
        mean_confidence = (
            sum(value * weight for value, weight in valid_confidence)
            / sum(weight for _, weight in valid_confidence)
            if valid_confidence
            else None
        )
        representative = geometry.representative_point()
        minx, miny, maxx, maxy = geometry.bounds
        node = {
            "node_id": node_id,
            "instance_id": len(nodes) + 1,
            "source_image_id": source_image_id,
            "segment_type": {"id": class_id, "name": class_name},
            "area_pixels": int(round(geometry.area)),
            "mean_semantic_confidence": mean_confidence,
            "member_node_ids": member_ids,
            "source_tile_ids": sorted({tile.tile_id for tile, _ in members}),
            "duplicate_member_count": len(members),
            "members": [
                {
                    "tile_id": tile.tile_id,
                    "node_id": str(member["node_id"]),
                    "instance_id": int(member["instance_id"]),
                    "segment_id": str(member["segment_id"]),
                    "original_area_pixels": int(member["area_pixels"]),
                    "owned_area_pixels": owned_counts.get((root, tile.tile_id), 0),
                    "mean_semantic_confidence": member.get("mean_semantic_confidence"),
                }
                for tile, member in members
            ],
            "geometry_type": geometry.geom_type,
            "geometry_source_pixel": mapping(geometry),
            "bbox_source_pixel_xyxy": [float(minx), float(miny), float(maxx), float(maxy)],
            "centroid_source_pixel_xy": [float(geometry.centroid.x), float(geometry.centroid.y)],
            "representative_point_source_pixel_xy": [float(representative.x), float(representative.y)],
            "overlap_resolution": {
                "method": (
                    "mutual_best_mask_overlap_then_edge_distance_confidence_ownership"
                ),
                "pixels_owned_exactly_once": True,
            },
        }
        nodes.append(node)
        geometries.append(geometry)
        root_to_node[root] = node_id
    return nodes, geometries, root_to_node, dropped


def _polygonal_union(values: Sequence[Any]) -> Any:
    geometry = unary_union(values)
    if isinstance(geometry, Polygon):
        return geometry
    if isinstance(geometry, MultiPolygon):
        return geometry
    if isinstance(geometry, GeometryCollection):
        polygons = [part for part in geometry.geoms if isinstance(part, (Polygon, MultiPolygon))]
        geometry = unary_union(polygons)
    if not isinstance(geometry, (Polygon, MultiPolygon)) or geometry.is_empty:
        raise RuntimeError("Stage 4 produced a non-polygonal combined object")
    return geometry


def _build_combined_edges(
    nodes: Sequence[Mapping[str, Any]],
    geometries: Sequence[Any],
    tiles: Sequence[_TileInput],
    union: _UnionFind,
    root_to_node: Mapping[str, str],
) -> list[dict[str, Any]]:
    by_id = {str(node["node_id"]): node for node in nodes}
    original_edge_ids: dict[tuple[str, str], list[str]] = {}
    for tile in tiles:
        for edge in tile.result.edges:
            first_root = union.find(str(edge["source"]))
            second_root = union.find(str(edge["target"]))
            first = root_to_node.get(first_root)
            second = root_to_node.get(second_root)
            if first is None or second is None or first == second:
                continue
            pair = tuple(sorted((first, second)))
            original_edge_ids.setdefault(pair, []).append(str(edge["edge_id"]))

    tree = STRtree(list(geometries))
    edges: list[dict[str, Any]] = []
    for first_index, first_geometry in enumerate(geometries):
        for raw_second in tree.query(first_geometry):
            second_index = int(raw_second)
            if second_index <= first_index:
                continue
            contact_geometry = first_geometry.boundary.intersection(
                geometries[second_index].boundary
            )
            shared = contact_geometry.length
            if shared < 0.5:
                continue
            anchor = _contact_anchor(contact_geometry)
            count = int(round(shared))
            first_node, second_node = nodes[first_index], nodes[second_index]
            source, target = str(first_node["node_id"]), str(second_node["node_id"])
            if int(first_node["instance_id"]) > int(second_node["instance_id"]):
                source, target = target, source
                first_node, second_node = second_node, first_node
            pair = tuple(sorted((source, target)))
            edges.append(
                {
                    "edge_id": make_combined_edge_id(source, target),
                    "source": source,
                    "target": target,
                    "source_instance_id": int(first_node["instance_id"]),
                    "target_instance_id": int(second_node["instance_id"]),
                    "relation": "adjacent",
                    "adjacency_definition": "shared_source_pixel_edge_after_overlap_ownership",
                    "shared_boundary_side_count": count,
                    "shared_boundary_length_pixels": float(count),
                    "contact_centroid_source_pixel_xy": [
                        float(contact_geometry.centroid.x),
                        float(contact_geometry.centroid.y),
                    ],
                    "contact_anchor_source_pixel_xy": anchor,
                    "source_class_id": int(first_node["segment_type"]["id"]),
                    "source_class": str(first_node["segment_type"]["name"]),
                    "target_class_id": int(second_node["segment_type"]["id"]),
                    "target_class": str(second_node["segment_type"]["name"]),
                    "same_semantic_class": int(first_node["segment_type"]["id"])
                    == int(second_node["segment_type"]["id"]),
                    "source_tile_edge_ids": sorted(original_edge_ids.get(pair, [])),
                }
            )
    edges.sort(key=lambda row: (row["source_instance_id"], row["target_instance_id"]))
    return edges


def _contact_anchor(contact_geometry: Any) -> list[float]:
    """Choose a point on the longest actual shared-boundary component."""

    pending = [contact_geometry]
    lines = []
    while pending:
        part = pending.pop()
        if part.geom_type in {"LineString", "LinearRing"} and part.length > 0:
            lines.append(part)
        elif hasattr(part, "geoms"):
            pending.extend(part.geoms)
    if not lines:
        raise ValueError("Shared boundary has no positive-length line component")
    selected = min(lines, key=lambda part: (-part.length, part.bounds))
    point = selected.interpolate(selected.length / 2)
    return [float(point.x), float(point.y)]


def _node_to_node_edge_line(
    edge: Mapping[str, Any], positions: Mapping[str, Sequence[float]],
) -> list[list[float]]:
    """Connect both node markers through their verified shared boundary."""

    contact = np.asarray(edge.get("contact_anchor_source_pixel_xy")
                         or edge["contact_centroid_source_pixel_xy"], dtype=float)
    return [list(positions[str(edge["source"])]), contact.tolist(),
            list(positions[str(edge["target"])])]


def _graph_metadata(
    tiles: Sequence[_TileInput],
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    source_image_id: str,
    cfg: Mapping[str, Any],
    *,
    accepted_count: int,
) -> dict[str, Any]:
    x0 = min(tile.x0 for tile in tiles)
    y0 = min(tile.y0 for tile in tiles)
    x1 = max(tile.x0 + tile.width for tile in tiles)
    y1 = max(tile.y0 + tile.height for tile in tiles)
    metadata = tiles[0].result.graph.get("tile_metadata", {})
    return {
        "id": f"{source_image_id}::combined::{stable_digest([tile.tile_id for tile in tiles])[:20]}",
        "source_image_id": source_image_id,
        "selected_tile_ids": [tile.tile_id for tile in tiles],
        "directed": False,
        "multigraph": False,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "source_pixel_extent_xyxy": [x0, y0, x1, y1],
        "source_affine_transform": copy.deepcopy(metadata.get("source_affine_transform")),
        "crs": metadata.get("crs"),
        "adjacency_definition": "shared_source_pixel_edge_after_overlap_ownership",
        "overlap_reconciliation": {
            **copy.deepcopy(dict(cfg["overlap"])),
            "ownership_method": cfg["ownership"]["method"],
            "accepted_match_count": accepted_count,
            "pixel_partition_is_exact": True,
        },
    }


def _persist(
    directory: Path,
    document: dict[str, Any],
    matches: list[dict[str, Any]],
    tiles: Sequence[_TileInput],
    cfg: Mapping[str, Any],
    accepted: int,
    dropped: int,
    started: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    directory.mkdir(parents=True, exist_ok=True)
    graph_path = directory / "graph.json"
    matches_path = directory / "overlap_matches.json"
    arrays_path = directory / "graph_arrays.npz"
    geojson_path = directory / "nodes.geojson"
    csv_path = directory / "adjacency.csv"
    report_path = directory / "report.json"
    preview_path = directory / "combined_graph.png"
    overlay_path = directory / "combined_graph_overlay.png"
    audit_overlay_path = directory / "combined_graph_overlay_audit.png"
    analysis_path = directory / "graph_analysis.json"
    analysis_arrays_path = directory / "graph_analysis_arrays.npz"
    phase_path = directory / "phase_manifest.json"
    manifest_path = directory / "manifest.json"
    analysis_document, analysis_report = _analysis_view(document, cfg.get("analysis", {}))
    save_graph_json(graph_path, document)
    atomic_json(analysis_path, analysis_document)
    _save_arrays(analysis_arrays_path, analysis_document["nodes"], analysis_document["edges"], schema_version=analysis_document["schema_version"])
    atomic_json(
        matches_path,
        {
            "schema_version": MATCH_SCHEMA_VERSION,
            "tile_ids": [tile.tile_id for tile in tiles],
            "matches": matches,
        },
    )
    _save_arrays(arrays_path, document["nodes"], document["edges"])
    _save_geojson(geojson_path, document)
    _save_csv(csv_path, document["edges"])
    _save_visualization(preview_path, analysis_document["nodes"], analysis_document["edges"])
    background_paths = {tile.tile_id: _tile_background_path(tile) for tile in tiles}
    _save_combined_overlay(
        overlay_path,
        tiles,
        analysis_document["nodes"],
        analysis_document["edges"],
        background_paths,
    )
    _save_combined_overlay(
        audit_overlay_path,
        tiles,
        document["nodes"],
        document["edges"],
        background_paths,
    )
    input_nodes = sum(len(tile.result.nodes) for tile in tiles)
    overlap_pixels = sum(int((~tile.ownership).sum()) for tile in tiles)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "source_image_id": document["graph"]["source_image_id"],
        "selected_tile_ids": [tile.tile_id for tile in tiles],
        "input_node_count": input_nodes,
        "combined_node_count": len(document["nodes"]),
        "combined_edge_count": len(document["edges"]),
        "analysis_graph": analysis_report,
        "overlap_candidate_count": len(matches),
        "eligible_overlap_match_count": sum(not row["reasons"] or row["selected"] for row in matches),
        "accepted_overlap_match_count": accepted,
        "semantic_conflict_count": sum("semantic_class_conflict" in row["reasons"] for row in matches),
        "removed_duplicate_node_count": input_nodes - len(document["nodes"]),
        "fully_occluded_unmatched_node_count": dropped,
        "non_owner_overlap_pixel_count": overlap_pixels,
        "overlap_pixels_owned_once": True,
        "pixel_overlap_policy": (
            "maximum_normalized_distance_to_tile_edge_then_semantic_confidence_then_tile_id"
        ),
        "visualization_backgrounds": {
            tile_id: str(path.resolve()) for tile_id, path in background_paths.items()
        },
        "validation": {"valid": True, "errors": []},
    }
    atomic_json(report_path, report)
    inputs = [
        path
        for tile in tiles
        for path in (
            tile.directory / "manifest.json",
            tile.directory / "graph.json",
            Path(str(tile.result.graph["source_refinement"]["output_directory"])) / "segmentation.npz",
        )
    ]
    inputs.extend(background_paths.values())
    inputs = list(dict.fromkeys(path.resolve() for path in inputs))
    outputs = [
        graph_path,
        matches_path,
        arrays_path,
        geojson_path,
        csv_path,
        report_path,
        preview_path,
        overlay_path,
        audit_overlay_path,
        analysis_path,
        analysis_arrays_path,
    ]
    manifest(phase_path, "01_combined_graph", inputs, outputs, stable_hash(dict(cfg)), started)
    artifacts = {
        "graph": str(graph_path.resolve()),
        "overlap_matches": str(matches_path.resolve()),
        "graph_arrays": str(arrays_path.resolve()),
        "nodes_geojson": str(geojson_path.resolve()),
        "adjacency_csv": str(csv_path.resolve()),
        "visualization": str(preview_path.resolve()),
        "graph_overlay": str(overlay_path.resolve()),
        "graph_overlay_audit": str(audit_overlay_path.resolve()),
        "analysis_graph": str(analysis_path.resolve()),
        "analysis_graph_arrays": str(analysis_arrays_path.resolve()),
        "report": str(report_path.resolve()),
        "phase_manifest": str(phase_path.resolve()),
    }
    public = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "stale": False,
        "created_at": now(),
        "configuration_hash": stable_hash(dict(cfg)),
        "source_image_id": document["graph"]["source_image_id"],
        "selected_tile_ids": [tile.tile_id for tile in tiles],
        "artifacts": artifacts,
        "artifact_checksums": {key: checksum(Path(path)) for key, path in artifacts.items()},
    }
    atomic_json(manifest_path, public)
    return artifacts, report


def _analysis_view(document: dict[str, Any], cfg: Mapping[str, Any]):
    """Keep unresolved micro-regions in audit, omit them from the analysis graph.

    This is an induced subgraph: no links are invented across excluded nodes.
    Filtering uses the final combined area so overlap matches can rescue a
    region whose total supported extent exceeds the minimum mapping unit.
    """
    minimum_area = int(cfg.get("minimum_unknown_area_pixels", 41))
    minimum_boundary = int(cfg.get("minimum_shared_boundary_pixels", 2))
    included = set()
    for node in document["nodes"]:
        excluded = int(node["segment_type"]["id"]) == 8 and node["area_pixels"] < minimum_area
        node["analysis_view"] = {
            "included": not excluded,
            "exclusion_reasons": ["unresolved_unknown_below_minimum_mapping_unit"] if excluded else [],
        }
        if not excluded:
            included.add(node["node_id"])
    for edge in document["edges"]:
        reasons = []
        if edge["source"] not in included or edge["target"] not in included:
            reasons.append("endpoint_excluded")
        if edge["shared_boundary_side_count"] < minimum_boundary:
            reasons.append("shared_boundary_below_analysis_minimum")
        edge["analysis_view"] = {"included": not reasons, "exclusion_reasons": reasons}
    nodes = [node for node in document["nodes"] if node["analysis_view"]["included"]]
    edges = [edge for edge in document["edges"] if edge["analysis_view"]["included"]]
    report = {
        "node_count": len(nodes), "edge_count": len(edges),
        "excluded_unknown_node_count": len(document["nodes"]) - len(nodes),
        "retained_unknown_node_count": sum(node["segment_type"]["id"] == 8 for node in nodes),
        "minimum_unknown_area_pixels": minimum_area,
        "minimum_shared_boundary_pixels": minimum_boundary,
        "semantic_pixels_changed": 0,
    }
    return {
        "schema_version": "satellitegraph.stage4.analysis-view.v1",
        "graph": {**copy.deepcopy(document["graph"]), "node_count": len(nodes), "edge_count": len(edges), "analysis_view": report},
        "nodes": nodes, "edges": edges,
    }, report


def _save_arrays(path: Path, nodes: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]], *, schema_version: str = SCHEMA_VERSION) -> None:
    index = {str(node["node_id"]): offset for offset, node in enumerate(nodes)}
    edge_index = (
        np.asarray([[index[edge["source"]], index[edge["target"]]] for edge in edges], dtype=np.int32).T
        if edges
        else np.empty((2, 0), dtype=np.int32)
    )
    atomic_npz(
        path,
        schema_version=np.asarray(schema_version),
        node_ids=np.asarray([node["node_id"] for node in nodes], dtype=np.str_),
        instance_ids=np.asarray([node["instance_id"] for node in nodes], dtype=np.int32),
        class_ids=np.asarray([node["segment_type"]["id"] for node in nodes], dtype=np.int16),
        area_pixels=np.asarray([node["area_pixels"] for node in nodes], dtype=np.int64),
        representative_points_source_xy=np.asarray(
            [node["representative_point_source_pixel_xy"] for node in nodes], dtype=np.float64
        ).reshape(len(nodes), 2),
        edge_index=edge_index,
        shared_boundary_side_count=np.asarray(
            [edge["shared_boundary_side_count"] for edge in edges], dtype=np.int64
        ),
    )


def _save_geojson(path: Path, document: Mapping[str, Any]) -> None:
    crs = document["graph"].get("crs")
    collection = {
        "type": "FeatureCollection",
        "coordinate_space": "source_pixel",
        "source_crs": crs,
        "features": [
            {
                "type": "Feature",
                "id": node["node_id"],
                "geometry": node["geometry_source_pixel"],
                "properties": {
                    "node_id": node["node_id"],
                    "instance_id": node["instance_id"],
                    "class_id": node["segment_type"]["id"],
                    "class_name": node["segment_type"]["name"],
                    "area_pixels": node["area_pixels"],
                    "source_tile_count": len(node["source_tile_ids"]),
                },
            }
            for node in document["nodes"]
        ],
    }
    atomic_json(path, collection)


def _save_csv(path: Path, edges: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "edge_id", "source", "target", "source_instance_id", "target_instance_id",
        "source_class", "target_class", "shared_boundary_side_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for edge in edges:
                writer.writerow({field: edge.get(field) for field in fields})
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _save_visualization(path: Path, nodes: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]]) -> None:
    positions = {str(node["node_id"]): node["representative_point_source_pixel_xy"] for node in nodes}
    colors = ["#cdbd7b", "#9c7a56", "#b0b0b0", "#ffd81f", "#278c36", "#3a8fd4", "#d4c85a", "#df4747", "#8555a5"]
    figure, axis = plt.subplots(figsize=(12, 8), constrained_layout=True)
    if edges:
        lines = [_node_to_node_edge_line(edge, positions) for edge in edges]
        axis.add_collection(LineCollection(lines, colors="#66717e", linewidths=0.35, alpha=0.35))
    if nodes:
        areas = np.asarray([node["area_pixels"] for node in nodes], dtype=float)
        sizes = 6.0 + 32.0 * np.log1p(areas) / max(1.0, float(np.log1p(areas).max()))
        axis.scatter(
            [positions[node["node_id"]][0] for node in nodes],
            [positions[node["node_id"]][1] for node in nodes],
            s=sizes,
            c=[colors[int(node["segment_type"]["id"])] for node in nodes],
            edgecolors="#18212b",
            linewidths=0.35,
            zorder=2,
        )
    axis.set_title(f"Combined graph: {len(nodes)} nodes, {len(edges)} edges")
    axis.set_aspect("equal")
    axis.invert_yaxis()
    axis.set_xlabel("source pixel x")
    axis.set_ylabel("source pixel y")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _tile_background_path(tile: _TileInput) -> Path:
    metadata = tile.result.graph.get("tile_metadata", {})
    image_path = metadata.get("image_path")
    if image_path:
        candidate = Path(str(image_path)).expanduser()
        if candidate.is_file():
            return candidate.resolve()
    refinement_directory = Path(
        str(tile.result.graph["source_refinement"]["output_directory"])
    )
    for filename in ("refined_overlay.png", "refined_segmentation.png"):
        candidate = refinement_directory / filename
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"No source or refinement image is available for tile {tile.tile_id!r}"
    )


def _save_combined_overlay(
    path: Path,
    tiles: Sequence[_TileInput],
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    background_paths: Mapping[str, Path],
) -> None:
    """Draw the global node-link graph over its ownership-resolved tile mosaic."""

    x0 = min(tile.x0 for tile in tiles)
    y0 = min(tile.y0 for tile in tiles)
    x1 = max(tile.x0 + tile.width for tile in tiles)
    y1 = max(tile.y0 + tile.height for tile in tiles)
    mosaic_aspect = (x1 - x0) / max(1, y1 - y0)
    figure_width = 16.0
    figure_height = min(12.0, max(4.5, figure_width / mosaic_aspect + 1.4))
    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height), constrained_layout=True
    )
    axis.set_facecolor("#111820")
    for tile in tiles:
        with Image.open(background_paths[tile.tile_id]) as source:
            rgb = np.asarray(source.convert("RGB"))
        if rgb.shape[:2] != (tile.height, tile.width):
            raise ValueError(
                f"Visualization image shape disagrees for tile {tile.tile_id!r}"
            )
        rgba = np.empty((tile.height, tile.width, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = tile.ownership.astype(np.uint8) * 255
        axis.imshow(
            rgba,
            origin="upper",
            extent=(tile.x0, tile.x0 + tile.width, tile.y0 + tile.height, tile.y0),
            interpolation="nearest",
            zorder=0,
        )
        axis.add_patch(
            Rectangle(
                (tile.x0, tile.y0),
                tile.width,
                tile.height,
                fill=False,
                edgecolor="#00d7ff",
                linewidth=0.8,
                linestyle="--",
                alpha=0.75,
                zorder=1,
            )
        )

    positions = {
        str(node["node_id"]): node["representative_point_source_pixel_xy"]
        for node in nodes
    }
    if edges:
        lines = [_node_to_node_edge_line(edge, positions) for edge in edges]
        contacts = [edge.get("contact_anchor_source_pixel_xy") or
                    edge["contact_centroid_source_pixel_xy"] for edge in edges]
        shared = np.asarray(
            [edge["shared_boundary_side_count"] for edge in edges], dtype=float
        )
        scaled = np.log1p(shared) / max(1.0, float(np.log1p(shared).max()))
        axis.add_collection(
            LineCollection(
                lines,
                colors="#ffffff",
                linewidths=0.35 + 1.35 * scaled,
                alpha=0.48,
                zorder=2,
            )
        )
        axis.scatter(
            [contact[0] for contact in contacts],
            [contact[1] for contact in contacts],
            s=3.0,
            c="#00d7ff",
            edgecolors="none",
            alpha=0.65,
            zorder=2.5,
        )
    if nodes:
        areas = np.asarray([node["area_pixels"] for node in nodes], dtype=float)
        sizes = 7.0 + 42.0 * np.log1p(areas) / max(
            1.0, float(np.log1p(areas).max())
        )
        colors = [
            [
                "#cdbd7b",
                "#9c7a56",
                "#b0b0b0",
                "#ffd81f",
                "#278c36",
                "#3a8fd4",
                "#d4c85a",
                "#df4747",
                "#8555a5",
            ][int(node["segment_type"]["id"])]
            for node in nodes
        ]
        axis.scatter(
            [positions[node["node_id"]][0] for node in nodes],
            [positions[node["node_id"]][1] for node in nodes],
            s=sizes,
            c=colors,
            edgecolors="#111111",
            linewidths=0.45,
            alpha=0.9,
            zorder=3,
        )
    axis.set_xlim(x0, x1)
    axis.set_ylim(y1, y0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("source pixel x")
    axis.set_ylabel("source pixel y")
    axis.set_title(
        f"Combined tile graph overlay: {len(tiles)} tiles, "
        f"{len(nodes)} nodes, {len(edges)} edges"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _output_directory(
    root: Path,
    source_image_id: str,
    tile_ids: Sequence[str],
    output_name: str | None,
    cfg: Mapping[str, Any],
) -> Path:
    name = output_name or f"{source_image_id}__{stable_digest(list(tile_ids))[:12]}"
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Stage 4 output name must be one safe path component")
    return _resolve(root, str(cfg["output"]["root_directory"])) / name / "01_combined_graph"


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


__all__ = ["Stage4Result", "combine_tile_graphs", "load_stage4_result"]
