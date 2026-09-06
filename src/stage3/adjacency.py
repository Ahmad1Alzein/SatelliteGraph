"""Deterministic four-neighbour adjacency for Stage 3 tile graphs.

An edge exists when two positive instance IDs share at least one complete
pixel side.  Corner-only contact is intentionally not adjacency.  The raster
is scanned only to the right and down, which observes every undirected contact
once before contacts are aggregated into one edge per unordered pair.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np

from .geometry import SpatialContext, spatial_context


ADJACENCY_DEFINITION = "shared_pixel_edge_4_connected"


def build_adjacency_edges(
    instance_map: np.ndarray,
    *,
    node_geometries: Mapping[int | str, Mapping[str, Any]] | None = None,
    tile_metadata: Mapping[str, Any] | None = None,
    minimum_shared_boundary_pixels: int = 1,
) -> list[dict[str, Any]]:
    """Return one stable edge record per adjacent unordered instance pair.

    Parameters
    ----------
    instance_map:
        Two-dimensional integer map.  Zero is excluded; positive values are
        graph instances.
    node_geometries:
        Optional output from :func:`src.stage3.geometry.build_node_geometries`.
        When supplied, endpoint contact fractions are included and every edge
        endpoint must have a matching record.
    tile_metadata:
        Optional Stage 2 tile metadata used for source-pixel, projected, and
        WGS84 contact coordinates and projected shared-boundary length.
    minimum_shared_boundary_pixels:
        Minimum number of shared pixel sides.  The exact region-adjacency graph
        uses the default value ``1``.

    Notes
    -----
    ``right`` scan contacts are vertical shared sides.  ``down`` scan contacts
    are horizontal shared sides.  Both descriptions are persisted to avoid an
    orientation ambiguity for consumers.
    """

    values = _validate_instance_map(instance_map)
    if (
        isinstance(minimum_shared_boundary_pixels, (bool, np.bool_))
        or not isinstance(minimum_shared_boundary_pixels, (int, np.integer))
        or int(minimum_shared_boundary_pixels) < 1
    ):
        raise ValueError("minimum_shared_boundary_pixels must be a positive integer")
    minimum = int(minimum_shared_boundary_pixels)

    contacts = _scan_contacts(values)
    if contacts is None:
        return []

    pairs = contacts["pairs"]
    unique_pairs, group = np.unique(pairs, axis=0, return_inverse=True)
    edge_count = len(unique_pairs)
    side_count = np.bincount(group, minlength=edge_count).astype(np.int64)
    vertical_count = np.bincount(
        group,
        weights=contacts["vertical"].astype(np.int64),
        minlength=edge_count,
    ).astype(np.int64)
    horizontal_count = side_count - vertical_count

    contact_bounds = _aggregate_endpoint_bounds(
        contacts["endpoint_x0"],
        contacts["endpoint_y0"],
        contacts["endpoint_x1"],
        contacts["endpoint_y1"],
        group,
        edge_count,
    )
    centroid_x = np.bincount(group, weights=contacts["midpoint_x"], minlength=edge_count) / side_count
    centroid_y = np.bincount(group, weights=contacts["midpoint_y"], minlength=edge_count) / side_count

    context = spatial_context(tile_metadata)
    coordinate_data = _contact_coordinate_data(
        contacts,
        group,
        edge_count,
        contact_bounds,
        centroid_x,
        centroid_y,
        context,
    )
    projected_lengths = _projected_shared_lengths(horizontal_count, vertical_count, context)

    kept = np.flatnonzero(side_count >= minimum)
    output: list[dict[str, Any]] = []
    for edge_index in kept.tolist():
        source = int(unique_pairs[edge_index, 0])
        target = int(unique_pairs[edge_index, 1])
        contact_spaces = {
            name: _coordinate_record_at(space, edge_index)
            for name, space in coordinate_data.items()
        }
        edge = {
            "edge_id": f"{source}--{target}",
            "source": source,
            "target": target,
            "source_instance_id": source,
            "target_instance_id": target,
            "relation": "adjacent",
            "adjacency_definition": ADJACENCY_DEFINITION,
            # Scalar aliases are intentionally kept at the top level for the
            # pipeline's node-link serializer and tabular exports.
            "shared_boundary_side_count": int(side_count[edge_index]),
            "left_right_contact_count": int(vertical_count[edge_index]),
            "top_bottom_contact_count": int(horizontal_count[edge_index]),
            "shared_boundary_length_pixels": int(side_count[edge_index]),
            "shared_boundary_length_projected": (
                float(projected_lengths["values"][edge_index])
                if projected_lengths["available"]
                else None
            ),
            "shared_boundary": {
                "side_count": int(side_count[edge_index]),
                "length_pixel_units": int(side_count[edge_index]),
                "orientation_counts": {
                    "horizontal": int(horizontal_count[edge_index]),
                    "vertical": int(vertical_count[edge_index]),
                },
                "scan_direction_counts": {
                    "right": int(vertical_count[edge_index]),
                    "down": int(horizontal_count[edge_index]),
                },
                "projected_length": _metric_record_at(projected_lengths, edge_index),
            },
            "contact_bbox_tile_pixel_xyxy": contact_spaces["tile_pixel"]["bbox_xyxy"],
            "contact_centroid_tile_pixel_xy": contact_spaces["tile_pixel"]["centroid_xy"],
            "contact_coordinates": contact_spaces,
        }
        if node_geometries is not None:
            edge["contact_fractions"] = _contact_fractions(
                source,
                target,
                int(side_count[edge_index]),
                edge["shared_boundary"]["projected_length"],
                node_geometries,
            )
        output.append(edge)
    return output


def adjacent_instance_pairs(instance_map: np.ndarray) -> list[tuple[int, int]]:
    """Return sorted adjacent pairs using the exact Stage 3 edge definition."""

    values = _validate_instance_map(instance_map)
    contacts = _scan_contacts(values)
    if contacts is None:
        return []
    pairs = np.unique(contacts["pairs"], axis=0)
    return [(int(source), int(target)) for source, target in pairs]


def _scan_contacts(values: np.ndarray) -> dict[str, np.ndarray] | None:
    left = values[:, :-1]
    right = values[:, 1:]
    right_contact = (left > 0) & (right > 0) & (left != right)
    right_rows, right_columns = np.nonzero(right_contact)
    right_first = left[right_contact].astype(np.int64, copy=False)
    right_second = right[right_contact].astype(np.int64, copy=False)

    upper = values[:-1, :]
    lower = values[1:, :]
    down_contact = (upper > 0) & (lower > 0) & (upper != lower)
    down_rows, down_columns = np.nonzero(down_contact)
    down_first = upper[down_contact].astype(np.int64, copy=False)
    down_second = lower[down_contact].astype(np.int64, copy=False)

    total = len(right_first) + len(down_first)
    if total == 0:
        return None

    first = np.concatenate([right_first, down_first])
    second = np.concatenate([right_second, down_second])
    pairs = np.column_stack([np.minimum(first, second), np.maximum(first, second)])

    # Right-neighbour contacts lie on x=column+1 and span one y unit.
    right_x = right_columns.astype(np.float64) + 1.0
    # Down-neighbour contacts lie on y=row+1 and span one x unit.
    down_y = down_rows.astype(np.float64) + 1.0
    endpoint_x0 = np.concatenate([right_x, down_columns.astype(np.float64)])
    endpoint_y0 = np.concatenate([right_rows.astype(np.float64), down_y])
    endpoint_x1 = np.concatenate([right_x, down_columns.astype(np.float64) + 1.0])
    endpoint_y1 = np.concatenate([right_rows.astype(np.float64) + 1.0, down_y])
    midpoint_x = 0.5 * (endpoint_x0 + endpoint_x1)
    midpoint_y = 0.5 * (endpoint_y0 + endpoint_y1)
    vertical = np.concatenate(
        [np.ones(len(right_first), dtype=bool), np.zeros(len(down_first), dtype=bool)]
    )
    return {
        "pairs": pairs,
        "endpoint_x0": endpoint_x0,
        "endpoint_y0": endpoint_y0,
        "endpoint_x1": endpoint_x1,
        "endpoint_y1": endpoint_y1,
        "midpoint_x": midpoint_x,
        "midpoint_y": midpoint_y,
        "vertical": vertical,
    }


def _contact_coordinate_data(
    contacts: Mapping[str, np.ndarray],
    group: np.ndarray,
    edge_count: int,
    tile_bounds: np.ndarray,
    centroid_x: np.ndarray,
    centroid_y: np.ndarray,
    context: SpatialContext,
) -> dict[str, dict[str, Any]]:
    tile_centroids = np.column_stack([centroid_x, centroid_y])
    output: dict[str, dict[str, Any]] = {
        "tile_pixel": _available_coordinate_arrays(
            "tile_pixel", None, tile_bounds, tile_centroids
        )
    }

    if context.source_offset_xy is not None:
        dx, dy = context.source_offset_xy
        source_bounds = tile_bounds + np.asarray([dx, dy, dx, dy], dtype=np.float64)
        source_centroids = tile_centroids + np.asarray([dx, dy], dtype=np.float64)
        output["source_pixel"] = _available_coordinate_arrays(
            "source_pixel", None, source_bounds, source_centroids
        )
    else:
        output["source_pixel"] = _unavailable_coordinate_arrays(
            "source_pixel", None, context.source_pixel_unavailable_reason, edge_count
        )

    world_endpoints: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    world_centroids: np.ndarray | None = None
    if context.tile_affine is not None:
        world_endpoints = _affine_contact_endpoints(contacts, context.tile_affine)
        world_bounds = _aggregate_endpoint_bounds(*world_endpoints, group, edge_count)
        world_x, world_y = _apply_affine_arrays(context.tile_affine, centroid_x, centroid_y)
        world_centroids = np.column_stack([world_x, world_y])
    else:
        world_bounds = None

    if context.projected_available and world_bounds is not None and world_centroids is not None:
        output["projected"] = _available_coordinate_arrays(
            "projected", context.source_crs_text, world_bounds, world_centroids
        )
    else:
        output["projected"] = _unavailable_coordinate_arrays(
            "projected", context.source_crs_text, context.projected_unavailable_reason, edge_count
        )

    if context.wgs84_available and world_endpoints is not None and world_centroids is not None:
        try:
            from rasterio.warp import transform

            x0, y0, x1, y1 = world_endpoints
            combined_x = np.concatenate([x0, x1, world_centroids[:, 0]])
            combined_y = np.concatenate([y0, y1, world_centroids[:, 1]])
            transformed_x, transformed_y = transform(
                context.source_crs,
                "EPSG:4326",
                combined_x.tolist(),
                combined_y.tolist(),
            )
            transformed_x = np.asarray(transformed_x, dtype=np.float64)
            transformed_y = np.asarray(transformed_y, dtype=np.float64)
            contact_count = len(x0)
            wgs_x0 = transformed_x[:contact_count]
            wgs_x1 = transformed_x[contact_count : 2 * contact_count]
            wgs_centroid_x = transformed_x[2 * contact_count :]
            wgs_y0 = transformed_y[:contact_count]
            wgs_y1 = transformed_y[contact_count : 2 * contact_count]
            wgs_centroid_y = transformed_y[2 * contact_count :]
            wgs_bounds = _aggregate_endpoint_bounds(
                wgs_x0,
                wgs_y0,
                wgs_x1,
                wgs_y1,
                group,
                edge_count,
            )
            wgs_centroids = np.column_stack([wgs_centroid_x, wgs_centroid_y])
            output["wgs84"] = _available_coordinate_arrays(
                "wgs84", "EPSG:4326", wgs_bounds, wgs_centroids
            )
        except Exception as exc:
            output["wgs84"] = _unavailable_coordinate_arrays(
                "wgs84",
                "EPSG:4326",
                f"Contact transformation to WGS84 failed: {type(exc).__name__}: {exc}",
                edge_count,
            )
    else:
        output["wgs84"] = _unavailable_coordinate_arrays(
            "wgs84", "EPSG:4326", context.wgs84_unavailable_reason, edge_count
        )
    return output


def _projected_shared_lengths(
    horizontal_count: np.ndarray,
    vertical_count: np.ndarray,
    context: SpatialContext,
) -> dict[str, Any]:
    if not context.projected_available:
        return {
            "available": False,
            "unavailable_reason": context.projected_unavailable_reason,
            "values": np.full(len(horizontal_count), np.nan),
            "unit": None,
            "crs": context.source_crs_text,
        }
    a, b, _, d, e, _ = context.tile_affine
    horizontal_unit_length = math.hypot(a, d)
    vertical_unit_length = math.hypot(b, e)
    values = horizontal_count * horizontal_unit_length + vertical_count * vertical_unit_length
    return {
        "available": True,
        "unavailable_reason": None,
        "values": values.astype(np.float64),
        "unit": context.linear_unit or "source_crs_unit",
        "crs": context.source_crs_text,
    }


def _contact_fractions(
    source: int,
    target: int,
    shared_pixel_length: int,
    projected_shared_length: Mapping[str, Any],
    node_geometries: Mapping[int | str, Mapping[str, Any]],
) -> dict[str, Any]:
    source_node = _node_record(node_geometries, source)
    target_node = _node_record(node_geometries, target)
    source_metrics = _node_metrics(source_node)
    target_metrics = _node_metrics(target_node)
    source_perimeter = _positive_metric(source_metrics.get("perimeter_pixel_units"), source)
    target_perimeter = _positive_metric(target_metrics.get("perimeter_pixel_units"), target)
    pixel = {
        "source": float(shared_pixel_length / source_perimeter),
        "target": float(shared_pixel_length / target_perimeter),
    }

    source_projected = source_metrics.get("projected_perimeter")
    target_projected = target_metrics.get("projected_perimeter")
    if (
        projected_shared_length.get("available")
        and isinstance(source_projected, Mapping)
        and source_projected.get("available")
        and isinstance(target_projected, Mapping)
        and target_projected.get("available")
    ):
        source_value = _positive_metric(source_projected.get("value"), source)
        target_value = _positive_metric(target_projected.get("value"), target)
        shared_value = float(projected_shared_length["value"])
        projected = {
            "available": True,
            "unavailable_reason": None,
            "source": float(shared_value / source_value),
            "target": float(shared_value / target_value),
        }
    else:
        projected = {
            "available": False,
            "unavailable_reason": "Projected shared length or endpoint perimeter is unavailable.",
            "source": None,
            "target": None,
        }
    return {"pixel_perimeter": pixel, "projected_perimeter": projected}


def _node_record(
    records: Mapping[int | str, Mapping[str, Any]],
    instance_id: int,
) -> Mapping[str, Any]:
    record = records.get(instance_id)
    if record is None:
        record = records.get(str(instance_id))
    if not isinstance(record, Mapping):
        raise ValueError(f"node_geometries has no record for adjacent instance {instance_id}")
    return record


def _node_metrics(record: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = record.get("metrics")
    if isinstance(metrics, Mapping):
        return metrics
    nested = record.get("geometry")
    if isinstance(nested, Mapping) and isinstance(nested.get("metrics"), Mapping):
        return nested["metrics"]
    raise ValueError("A node geometry record has no metrics mapping")


def _positive_metric(value: Any, instance_id: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Node {instance_id} has a non-numeric perimeter") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"Node {instance_id} must have a positive finite perimeter")
    return number


def _aggregate_endpoint_bounds(
    x0: np.ndarray,
    y0: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    group: np.ndarray,
    count: int,
) -> np.ndarray:
    minimum_x = np.full(count, np.inf, dtype=np.float64)
    minimum_y = np.full(count, np.inf, dtype=np.float64)
    maximum_x = np.full(count, -np.inf, dtype=np.float64)
    maximum_y = np.full(count, -np.inf, dtype=np.float64)
    np.minimum.at(minimum_x, group, np.minimum(x0, x1))
    np.minimum.at(minimum_y, group, np.minimum(y0, y1))
    np.maximum.at(maximum_x, group, np.maximum(x0, x1))
    np.maximum.at(maximum_y, group, np.maximum(y0, y1))
    return np.column_stack([minimum_x, minimum_y, maximum_x, maximum_y])


def _affine_contact_endpoints(
    contacts: Mapping[str, np.ndarray],
    affine: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x0, y0 = _apply_affine_arrays(affine, contacts["endpoint_x0"], contacts["endpoint_y0"])
    x1, y1 = _apply_affine_arrays(affine, contacts["endpoint_x1"], contacts["endpoint_y1"])
    return x0, y0, x1, y1


def _apply_affine_arrays(
    affine: Sequence[float],
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    a, b, c, d, e, f = affine
    return a * x + b * y + c, d * x + e * y + f


def _available_coordinate_arrays(
    coordinate_space: str,
    crs: str | None,
    bounds: np.ndarray,
    centroids: np.ndarray,
) -> dict[str, Any]:
    return {
        "available": True,
        "unavailable_reason": None,
        "coordinate_space": coordinate_space,
        "crs": crs,
        "bounds": bounds,
        "centroids": centroids,
        "bbox_basis": "shared_interface_segment_endpoints",
        "centroid_basis": "mean_of_shared_pixel_side_midpoints",
    }


def _unavailable_coordinate_arrays(
    coordinate_space: str,
    crs: str | None,
    reason: str | None,
    count: int,
) -> dict[str, Any]:
    return {
        "available": False,
        "unavailable_reason": reason or f"{coordinate_space} contact coordinates are unavailable",
        "coordinate_space": coordinate_space,
        "crs": crs,
        "bounds": np.full((count, 4), np.nan),
        "centroids": np.full((count, 2), np.nan),
        "bbox_basis": "shared_interface_segment_endpoints",
        "centroid_basis": "mean_of_shared_pixel_side_midpoints",
    }


def _coordinate_record_at(space: Mapping[str, Any], index: int) -> dict[str, Any]:
    available = bool(space["available"])
    return {
        "available": available,
        "unavailable_reason": space["unavailable_reason"],
        "coordinate_space": space["coordinate_space"],
        "crs": space["crs"],
        "bbox_xyxy": _clean_values(space["bounds"][index]) if available else None,
        "centroid_xy": _clean_values(space["centroids"][index]) if available else None,
        "bbox_basis": space["bbox_basis"],
        "centroid_basis": space["centroid_basis"],
    }


def _metric_record_at(metric: Mapping[str, Any], index: int) -> dict[str, Any]:
    available = bool(metric["available"])
    return {
        "available": available,
        "unavailable_reason": metric["unavailable_reason"],
        "value": float(metric["values"][index]) if available else None,
        "unit": metric["unit"],
        "crs": metric["crs"],
    }


def _validate_instance_map(instance_map: np.ndarray) -> np.ndarray:
    values = np.asarray(instance_map)
    if values.ndim != 2:
        raise ValueError(f"instance_map must be two-dimensional, got shape {values.shape}")
    if values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("instance_map dimensions must be positive")
    if values.dtype == np.bool_ or not np.issubdtype(values.dtype, np.integer):
        raise TypeError(f"instance_map must have an integer dtype, got {values.dtype}")
    if np.any(values < 0):
        raise ValueError("instance_map may contain only zero and positive instance IDs")
    return values


def _clean_values(values: Sequence[float]) -> list[int | float]:
    result: list[int | float] = []
    for value in values:
        number = float(value)
        result.append(int(number) if number.is_integer() else number)
    return result


__all__ = [
    "ADJACENCY_DEFINITION",
    "adjacent_instance_pairs",
    "build_adjacency_edges",
]
