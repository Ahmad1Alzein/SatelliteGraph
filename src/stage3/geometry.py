"""Exact raster geometry and spatial metadata for Stage 3 graph nodes.

The authoritative geometry input is Stage 2's dense ``instance_map``.  This
module deliberately does not infer topology from floating-point polygons:
polygons are an export of the raster, while adjacency is derived separately
from pixel-side contacts.

Pixel conventions match Stage 2:

* indices are zero based;
* ``x`` is column and ``y`` is row;
* polygon vertices lie on pixel edges; and
* centroids and representative points use pixel centres.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np


_SPACE_NAMES = ("tile_pixel", "source_pixel", "projected", "wgs84")


@dataclass(frozen=True, slots=True)
class SpatialContext:
    """Validated coordinate information derived from Stage 2 tile metadata.

    ``tile_affine`` maps tile-pixel edge coordinates to the source CRS.  It is
    derived from the source affine and source-pixel offset when the explicit
    tile affine is absent.  Availability reasons are retained instead of
    inventing an affine or CRS.
    """

    source_offset_xy: tuple[float, float] | None
    source_pixel_unavailable_reason: str | None
    tile_affine: tuple[float, float, float, float, float, float] | None
    affine_unavailable_reason: str | None
    source_crs: Any | None
    source_crs_text: str | None
    crs_unavailable_reason: str | None
    is_projected: bool
    linear_unit: str | None

    @property
    def projected_available(self) -> bool:
        return self.tile_affine is not None and self.source_crs is not None and self.is_projected

    @property
    def wgs84_available(self) -> bool:
        return self.tile_affine is not None and self.source_crs is not None

    @property
    def projected_unavailable_reason(self) -> str | None:
        if self.tile_affine is None:
            return self.affine_unavailable_reason
        if self.source_crs is None:
            return self.crs_unavailable_reason
        if not self.is_projected:
            return "The source CRS is not projected, so metric projected geometry is unavailable."
        return None

    @property
    def wgs84_unavailable_reason(self) -> str | None:
        if self.tile_affine is None:
            return self.affine_unavailable_reason
        if self.source_crs is None:
            return self.crs_unavailable_reason
        return None

    def tile_to_source_point(self, point_xy: Sequence[float]) -> list[int | float]:
        if self.source_offset_xy is None:
            raise ValueError(self.source_pixel_unavailable_reason or "Source-pixel offset is unavailable")
        return _clean_point(
            [point_xy[0] + self.source_offset_xy[0], point_xy[1] + self.source_offset_xy[1]]
        )

    def tile_to_world_point(self, point_xy: Sequence[float]) -> list[int | float]:
        if self.tile_affine is None:
            raise ValueError(self.affine_unavailable_reason or "Tile affine transform is unavailable")
        return _clean_point(_apply_affine(self.tile_affine, float(point_xy[0]), float(point_xy[1])))

    def world_to_wgs84_point(self, point_xy: Sequence[float]) -> list[float]:
        if not self.wgs84_available:
            raise ValueError(self.wgs84_unavailable_reason or "WGS84 transformation is unavailable")
        from rasterio.warp import transform

        xs, ys = transform(self.source_crs, "EPSG:4326", [float(point_xy[0])], [float(point_xy[1])])
        return [float(xs[0]), float(ys[0])]


def spatial_context(tile_metadata: Mapping[str, Any] | None) -> SpatialContext:
    """Create a reusable, validated spatial context for a Stage 2 tile."""

    metadata: Mapping[str, Any] = tile_metadata or {}
    offset, offset_reason = _source_offset(metadata)

    explicit_tile_affine = metadata.get("tile_affine_transform")
    tile_affine = _affine_coefficients(explicit_tile_affine)
    affine_reason: str | None = None
    if explicit_tile_affine is not None and tile_affine is None:
        affine_reason = "tile_metadata.tile_affine_transform is present but is not a finite six-value affine."

    if tile_affine is None and explicit_tile_affine is None:
        source_value = metadata.get("source_affine_transform")
        if source_value is None:
            source_value = metadata.get("affine_transform")
        source_affine = _affine_coefficients(source_value)
        if source_affine is not None and offset is not None:
            a, b, c, d, e, f = source_affine
            x0, y0 = offset
            tile_affine = (a, b, a * x0 + b * y0 + c, d, e, d * x0 + e * y0 + f)
        elif source_value is not None and source_affine is None:
            affine_reason = "The source affine transform is present but is not a finite six-value affine."
        elif source_affine is not None:
            affine_reason = (
                "A source affine exists, but global_x0/global_y0 are unavailable, so it cannot be offset to the tile."
            )
        else:
            affine_reason = "No usable tile or source affine transform is available."

    if tile_affine is not None:
        a, b, _, d, e, _ = tile_affine
        if math.isclose(a * e - b * d, 0.0, rel_tol=0.0, abs_tol=1e-15):
            tile_affine = None
            affine_reason = "The tile affine transform has a singular linear component."

    crs_value = metadata.get("crs_wkt") or metadata.get("crs")
    source_crs = None
    source_crs_text = None
    crs_reason: str | None = None
    is_projected = False
    linear_unit = None
    if crs_value is None:
        crs_reason = "The source raster has no CRS."
    else:
        try:
            from rasterio.crs import CRS

            source_crs = CRS.from_user_input(crs_value)
            source_crs_text = source_crs.to_string()
            is_projected = bool(source_crs.is_projected)
            if is_projected:
                try:
                    linear_unit = str(source_crs.linear_units or "source_crs_unit")
                except Exception:
                    linear_unit = "source_crs_unit"
        except Exception as exc:
            crs_reason = f"The source CRS could not be parsed: {type(exc).__name__}: {exc}"

    return SpatialContext(
        source_offset_xy=offset,
        source_pixel_unavailable_reason=offset_reason,
        tile_affine=tile_affine,
        affine_unavailable_reason=affine_reason,
        source_crs=source_crs,
        source_crs_text=source_crs_text,
        crs_unavailable_reason=crs_reason,
        is_projected=is_projected,
        linear_unit=linear_unit,
    )


def polygonize_instance_map(instance_map: np.ndarray) -> dict[int, dict[str, Any]]:
    """Polygonize every positive instance ID without simplification.

    The result is keyed by ascending instance ID.  Four-connected
    polygonization means an instance connected only at pixel corners is
    represented truthfully as a ``MultiPolygon``.  Interior rings are retained.
    Ring orientation, starting vertices, holes, and polygon parts are
    canonicalized so repeated runs produce byte-stable coordinate ordering.
    """

    values = _validate_instance_map(instance_map)
    positive_ids = np.unique(values[values > 0])
    if positive_ids.size == 0:
        return {}

    from affine import Affine
    from rasterio.features import shapes

    grouped: dict[int, list[list[list[list[int | float]]]]] = defaultdict(list)
    for geometry, raw_id in shapes(
        values.astype(np.int32, copy=False),
        mask=values > 0,
        connectivity=4,
        transform=Affine.identity(),
    ):
        instance_id = int(raw_id)
        grouped[instance_id].extend(_polygon_parts(geometry))

    result: dict[int, dict[str, Any]] = {}
    for raw_id in positive_ids:
        instance_id = int(raw_id)
        parts = grouped.get(instance_id)
        if not parts:
            raise RuntimeError(f"Raster polygonization produced no geometry for instance {instance_id}")
        result[instance_id] = _canonical_geometry(parts)
    return result


def build_node_geometries(
    instance_map: np.ndarray,
    tile_metadata: Mapping[str, Any] | None,
) -> dict[int, dict[str, Any]]:
    """Build exact geometry, coordinate variants, and metrics for all nodes.

    Returns an insertion-ordered dictionary keyed by ascending positive
    instance ID.  Every value is JSON-safe and can be merged directly with the
    corresponding Stage 2 segment record by a pipeline.
    """

    values = _validate_instance_map(instance_map)
    polygons = polygonize_instance_map(values)
    if not polygons:
        return {}

    ids = np.asarray(list(polygons), dtype=np.int64)
    stats = _raster_statistics(values, ids)
    context = spatial_context(tile_metadata)
    boundary_contacts = _tile_boundary_contacts(values, ids, context)
    coordinate_spaces = _coordinate_geometry_variants_batch(polygons, stats, context)

    result: dict[int, dict[str, Any]] = {}
    for instance_id in ids.tolist():
        geometry = polygons[instance_id]
        node_stats = stats[instance_id]
        polygon_area = _geometry_area(geometry)
        polygon_perimeter = _geometry_perimeter(geometry)
        if not math.isclose(polygon_area, node_stats["area_pixels"], rel_tol=0.0, abs_tol=1e-7):
            raise RuntimeError(
                f"Polygon area {polygon_area} disagrees with raster area "
                f"{node_stats['area_pixels']} for instance {instance_id}"
            )
        if not math.isclose(
            polygon_perimeter,
            node_stats["perimeter_pixel_units"],
            rel_tol=0.0,
            abs_tol=1e-7,
        ):
            raise RuntimeError(
                f"Polygon perimeter {polygon_perimeter} disagrees with raster perimeter "
                f"{node_stats['perimeter_pixel_units']} for instance {instance_id}"
            )

        geometry_counts = _geometry_counts(geometry)
        projected_metrics = _projected_metrics(node_stats, context)
        metrics = {
            **node_stats,
            **geometry_counts,
            "projected_area": projected_metrics["area"],
            "projected_perimeter": projected_metrics["perimeter"],
        }
        result[instance_id] = {
            "instance_id": instance_id,
            "geometry_type": geometry["type"],
            "coordinate_spaces": coordinate_spaces[instance_id],
            "representative_point_method": "segment_pixel_centre_nearest_to_raster_centroid_row_major_tiebreak",
            "metrics": metrics,
            "tile_boundary_contacts": boundary_contacts[instance_id],
        }
    return result


def coordinate_point_variants(
    point_xy: Sequence[float],
    tile_metadata: Mapping[str, Any] | None = None,
    *,
    context: SpatialContext | None = None,
) -> dict[str, dict[str, Any]]:
    """Transform one tile-pixel point into every supported coordinate space."""

    if len(point_xy) != 2 or not all(_is_finite_number(value) for value in point_xy):
        raise ValueError("point_xy must contain two finite numeric values")
    spatial = context or spatial_context(tile_metadata)
    tile_point = _clean_point(point_xy)
    result = {
        "tile_pixel": _available_point_space("tile_pixel", None, tile_point),
    }

    if spatial.source_offset_xy is not None:
        result["source_pixel"] = _available_point_space(
            "source_pixel", None, spatial.tile_to_source_point(tile_point)
        )
    else:
        result["source_pixel"] = _unavailable_point_space(
            "source_pixel", None, spatial.source_pixel_unavailable_reason
        )

    world_point: list[int | float] | None = None
    if spatial.tile_affine is not None:
        world_point = spatial.tile_to_world_point(tile_point)
    if spatial.projected_available and world_point is not None:
        result["projected"] = _available_point_space(
            "projected", spatial.source_crs_text, world_point
        )
    else:
        result["projected"] = _unavailable_point_space(
            "projected", spatial.source_crs_text, spatial.projected_unavailable_reason
        )

    if spatial.wgs84_available and world_point is not None:
        try:
            wgs84_point = spatial.world_to_wgs84_point(world_point)
            result["wgs84"] = _available_point_space("wgs84", "EPSG:4326", wgs84_point)
        except Exception as exc:
            result["wgs84"] = _unavailable_point_space(
                "wgs84", "EPSG:4326", f"Point transformation to WGS84 failed: {type(exc).__name__}: {exc}"
            )
    else:
        result["wgs84"] = _unavailable_point_space(
            "wgs84", "EPSG:4326", spatial.wgs84_unavailable_reason
        )
    return result


def _coordinate_geometry_variants(
    tile_geometry: Mapping[str, Any],
    tile_centroid: Sequence[float],
    tile_representative_point: Sequence[float],
    context: SpatialContext,
) -> dict[str, dict[str, Any]]:
    tile_geometry = _canonicalize_geojson(tile_geometry)
    result = {
        "tile_pixel": _available_geometry_space(
            "tile_pixel", None, tile_geometry, tile_centroid, tile_representative_point
        )
    }

    if context.source_offset_xy is not None:
        dx, dy = context.source_offset_xy
        source_geometry = _map_geometry(tile_geometry, lambda x, y: (x + dx, y + dy))
        source_centroid = context.tile_to_source_point(tile_centroid)
        source_representative = context.tile_to_source_point(tile_representative_point)
        result["source_pixel"] = _available_geometry_space(
            "source_pixel", None, source_geometry, source_centroid, source_representative
        )
    else:
        result["source_pixel"] = _unavailable_geometry_space(
            "source_pixel", None, context.source_pixel_unavailable_reason
        )

    world_geometry: dict[str, Any] | None = None
    world_centroid: list[int | float] | None = None
    world_representative: list[int | float] | None = None
    if context.tile_affine is not None:
        affine = context.tile_affine
        world_geometry = _map_geometry(tile_geometry, lambda x, y: _apply_affine(affine, x, y))
        world_centroid = context.tile_to_world_point(tile_centroid)
        world_representative = context.tile_to_world_point(tile_representative_point)

    if context.projected_available and world_geometry is not None:
        result["projected"] = _available_geometry_space(
            "projected",
            context.source_crs_text,
            world_geometry,
            world_centroid,
            world_representative,
        )
    else:
        result["projected"] = _unavailable_geometry_space(
            "projected", context.source_crs_text, context.projected_unavailable_reason
        )

    if context.wgs84_available and world_geometry is not None:
        try:
            from rasterio.warp import transform_geom

            wgs_geometry = transform_geom(
                context.source_crs,
                "EPSG:4326",
                world_geometry,
                precision=12,
            )
            wgs_geometry = _canonicalize_geojson(wgs_geometry)
            wgs_centroid = context.world_to_wgs84_point(world_centroid)
            wgs_representative = context.world_to_wgs84_point(world_representative)
            result["wgs84"] = _available_geometry_space(
                "wgs84", "EPSG:4326", wgs_geometry, wgs_centroid, wgs_representative
            )
        except Exception as exc:
            result["wgs84"] = _unavailable_geometry_space(
                "wgs84",
                "EPSG:4326",
                f"Geometry transformation to WGS84 failed: {type(exc).__name__}: {exc}",
            )
    else:
        result["wgs84"] = _unavailable_geometry_space(
            "wgs84", "EPSG:4326", context.wgs84_unavailable_reason
        )
    return result


def _coordinate_geometry_variants_batch(
    geometries: Mapping[int, Mapping[str, Any]],
    statistics: Mapping[int, Mapping[str, Any]],
    context: SpatialContext,
) -> dict[int, dict[str, dict[str, Any]]]:
    """Build geometry spaces with one GDAL transformation call per tile.

    Calling ``transform_geom`` separately for hundreds of small regions has a
    large fixed GDAL/PROJ overhead.  Batching preserves the exact same geometry
    semantics while keeping Stage 3 graph construction tile-scale efficient.
    """

    output: dict[int, dict[str, dict[str, Any]]] = {}
    world_data: dict[int, tuple[dict[str, Any], list[int | float], list[int | float]]] = {}
    for instance_id, raw_geometry in geometries.items():
        tile_geometry = _canonicalize_geojson(raw_geometry)
        stats = statistics[instance_id]
        tile_centroid = stats["centroid_tile_pixel_xy"]
        tile_representative = stats["representative_point_tile_pixel_xy"]
        spaces: dict[str, dict[str, Any]] = {
            "tile_pixel": _available_geometry_space(
                "tile_pixel", None, tile_geometry, tile_centroid, tile_representative
            )
        }

        if context.source_offset_xy is not None:
            dx, dy = context.source_offset_xy
            source_geometry = _map_geometry(tile_geometry, lambda x, y: (x + dx, y + dy))
            spaces["source_pixel"] = _available_geometry_space(
                "source_pixel",
                None,
                source_geometry,
                context.tile_to_source_point(tile_centroid),
                context.tile_to_source_point(tile_representative),
            )
        else:
            spaces["source_pixel"] = _unavailable_geometry_space(
                "source_pixel", None, context.source_pixel_unavailable_reason
            )

        if context.tile_affine is not None:
            affine = context.tile_affine
            world_geometry = _map_geometry(tile_geometry, lambda x, y: _apply_affine(affine, x, y))
            world_centroid = context.tile_to_world_point(tile_centroid)
            world_representative = context.tile_to_world_point(tile_representative)
            world_data[instance_id] = (world_geometry, world_centroid, world_representative)
        if context.projected_available and instance_id in world_data:
            world_geometry, world_centroid, world_representative = world_data[instance_id]
            spaces["projected"] = _available_geometry_space(
                "projected",
                context.source_crs_text,
                world_geometry,
                world_centroid,
                world_representative,
            )
        else:
            spaces["projected"] = _unavailable_geometry_space(
                "projected", context.source_crs_text, context.projected_unavailable_reason
            )
        output[instance_id] = spaces

    if not context.wgs84_available:
        for spaces in output.values():
            spaces["wgs84"] = _unavailable_geometry_space(
                "wgs84", "EPSG:4326", context.wgs84_unavailable_reason
            )
        return output

    ordered_ids = list(geometries)
    try:
        from rasterio.warp import transform, transform_geom

        transformed_geometries = transform_geom(
            context.source_crs,
            "EPSG:4326",
            [world_data[instance_id][0] for instance_id in ordered_ids],
            precision=12,
        )
        point_x: list[float] = []
        point_y: list[float] = []
        for instance_id in ordered_ids:
            _, centroid, representative = world_data[instance_id]
            point_x.extend([float(centroid[0]), float(representative[0])])
            point_y.extend([float(centroid[1]), float(representative[1])])
        transformed_x, transformed_y = transform(
            context.source_crs,
            "EPSG:4326",
            point_x,
            point_y,
        )
        for index, instance_id in enumerate(ordered_ids):
            geometry = _canonicalize_geojson(transformed_geometries[index])
            centroid = [transformed_x[2 * index], transformed_y[2 * index]]
            representative = [transformed_x[2 * index + 1], transformed_y[2 * index + 1]]
            output[instance_id]["wgs84"] = _available_geometry_space(
                "wgs84", "EPSG:4326", geometry, centroid, representative
            )
    except Exception as exc:
        reason = f"Geometry transformation to WGS84 failed: {type(exc).__name__}: {exc}"
        for spaces in output.values():
            spaces["wgs84"] = _unavailable_geometry_space("wgs84", "EPSG:4326", reason)
    return output


def _raster_statistics(values: np.ndarray, ids: np.ndarray) -> dict[int, dict[str, Any]]:
    height, width = values.shape
    positive_positions = np.flatnonzero(values.ravel() > 0)
    positive_values = values.ravel()[positive_positions]
    unique_ids, inverse = np.unique(positive_values, return_inverse=True)
    if not np.array_equal(unique_ids.astype(np.int64), ids):
        raise RuntimeError("Internal instance-ID ordering mismatch")

    rows = positive_positions // width
    columns = positive_positions % width
    counts = np.bincount(inverse, minlength=len(ids)).astype(np.int64)
    centroid_x = np.bincount(inverse, weights=columns + 0.5, minlength=len(ids)) / counts
    centroid_y = np.bincount(inverse, weights=rows + 0.5, minlength=len(ids)) / counts

    distance_squared = (columns + 0.5 - centroid_x[inverse]) ** 2 + (
        rows + 0.5 - centroid_y[inverse]
    ) ** 2
    minimum_distance = np.full(len(ids), np.inf, dtype=np.float64)
    np.minimum.at(minimum_distance, inverse, distance_squared)
    candidate = distance_squared == minimum_distance[inverse]
    representative_position = np.full(len(ids), values.size, dtype=np.int64)
    np.minimum.at(representative_position, inverse[candidate], positive_positions[candidate])
    representative_row = representative_position // width
    representative_column = representative_position % width

    minimum_row = np.full(len(ids), height, dtype=np.int64)
    minimum_column = np.full(len(ids), width, dtype=np.int64)
    maximum_row = np.full(len(ids), -1, dtype=np.int64)
    maximum_column = np.full(len(ids), -1, dtype=np.int64)
    np.minimum.at(minimum_row, inverse, rows)
    np.minimum.at(minimum_column, inverse, columns)
    np.maximum.at(maximum_row, inverse, rows)
    np.maximum.at(maximum_column, inverse, columns)

    padded = np.pad(values, 1, mode="constant", constant_values=0)
    centre = padded[1:-1, 1:-1]
    exposed_top = (centre > 0) & (centre != padded[:-2, 1:-1])
    exposed_bottom = (centre > 0) & (centre != padded[2:, 1:-1])
    exposed_left = (centre > 0) & (centre != padded[1:-1, :-2])
    exposed_right = (centre > 0) & (centre != padded[1:-1, 2:])

    top_count = _counts_for_ids(centre[exposed_top], ids)
    bottom_count = _counts_for_ids(centre[exposed_bottom], ids)
    left_count = _counts_for_ids(centre[exposed_left], ids)
    right_count = _counts_for_ids(centre[exposed_right], ids)

    output: dict[int, dict[str, Any]] = {}
    for index, raw_id in enumerate(ids):
        x0, y0 = int(minimum_column[index]), int(minimum_row[index])
        x1, y1 = int(maximum_column[index] + 1), int(maximum_row[index] + 1)
        area = int(counts[index])
        horizontal_sides = int(top_count[index] + bottom_count[index])
        vertical_sides = int(left_count[index] + right_count[index])
        perimeter = horizontal_sides + vertical_sides
        fill_ratio = area / ((x1 - x0) * (y1 - y0))
        compactness = 4.0 * math.pi * area / (perimeter * perimeter) if perimeter else None
        output[int(raw_id)] = {
            "area_pixels": area,
            "bbox_tile_pixel_xyxy": [x0, y0, x1, y1],
            "centroid_tile_pixel_xy": [float(centroid_x[index]), float(centroid_y[index])],
            "representative_point_tile_pixel_xy": [
                float(representative_column[index]) + 0.5,
                float(representative_row[index]) + 0.5,
            ],
            "bbox_fill_ratio": float(fill_ratio),
            "perimeter_pixel_units": int(perimeter),
            "boundary_side_counts": {
                "horizontal": horizontal_sides,
                "vertical": vertical_sides,
                "top": int(top_count[index]),
                "bottom": int(bottom_count[index]),
                "left": int(left_count[index]),
                "right": int(right_count[index]),
            },
            "compactness_pixel_grid": float(compactness) if compactness is not None else None,
        }
    return output


def _tile_boundary_contacts(
    values: np.ndarray,
    ids: np.ndarray,
    context: SpatialContext,
) -> dict[int, dict[str, Any]]:
    height, width = values.shape
    side_lines = {
        "top": (values[0, :], "x", 0, context.source_offset_xy[1] if context.source_offset_xy else None),
        "bottom": (
            values[-1, :],
            "x",
            height,
            context.source_offset_xy[1] + height if context.source_offset_xy else None,
        ),
        "left": (values[:, 0], "y", 0, context.source_offset_xy[0] if context.source_offset_xy else None),
        "right": (
            values[:, -1],
            "y",
            width,
            context.source_offset_xy[0] + width if context.source_offset_xy else None,
        ),
    }
    runs_by_side = {name: _positive_runs(line) for name, (line, _, _, _) in side_lines.items()}
    output: dict[int, dict[str, Any]] = {}
    for raw_id in ids:
        instance_id = int(raw_id)
        sides: dict[str, Any] = {}
        total = 0
        for side_name, (_, axis, local_fixed, source_fixed) in side_lines.items():
            local_runs = runs_by_side[side_name].get(instance_id, [])
            side_count = sum(end - start for start, end in local_runs)
            total += side_count
            tile_record = {
                "axis": axis,
                "fixed_coordinate": int(local_fixed),
                "runs": [list(run) for run in local_runs],
                "run_convention": "half-open [start, end) along axis",
            }
            if context.source_offset_xy is not None:
                offset = context.source_offset_xy[0] if axis == "x" else context.source_offset_xy[1]
                source_record = {
                    "available": True,
                    "unavailable_reason": None,
                    "axis": axis,
                    "fixed_coordinate": _clean_number(source_fixed),
                    "runs": [
                        [_clean_number(start + offset), _clean_number(end + offset)]
                        for start, end in local_runs
                    ],
                    "run_convention": "half-open [start, end) along axis",
                }
            else:
                source_record = {
                    "available": False,
                    "unavailable_reason": context.source_pixel_unavailable_reason,
                    "axis": axis,
                    "fixed_coordinate": None,
                    "runs": None,
                    "run_convention": "half-open [start, end) along axis",
                }
            sides[side_name] = {
                "side_count": int(side_count),
                "tile_pixel": tile_record,
                "source_pixel": source_record,
            }
        output[instance_id] = {
            "touches": total > 0,
            "side_count": int(total),
            **sides,
        }
    return output


def _projected_metrics(stats: Mapping[str, Any], context: SpatialContext) -> dict[str, dict[str, Any]]:
    reason = context.projected_unavailable_reason
    if not context.projected_available:
        return {
            "area": _unavailable_metric(reason),
            "perimeter": _unavailable_metric(reason),
        }
    a, b, _, d, e, _ = context.tile_affine
    pixel_area = abs(a * e - b * d)
    horizontal_length = math.hypot(a, d)
    vertical_length = math.hypot(b, e)
    counts = stats["boundary_side_counts"]
    perimeter = counts["horizontal"] * horizontal_length + counts["vertical"] * vertical_length
    unit = context.linear_unit or "source_crs_unit"
    return {
        "area": {
            "available": True,
            "unavailable_reason": None,
            "value": float(stats["area_pixels"] * pixel_area),
            "unit": f"{unit}^2",
            "crs": context.source_crs_text,
        },
        "perimeter": {
            "available": True,
            "unavailable_reason": None,
            "value": float(perimeter),
            "unit": unit,
            "crs": context.source_crs_text,
        },
    }


def _available_geometry_space(
    coordinate_space: str,
    crs: str | None,
    geometry: Mapping[str, Any],
    centroid_xy: Sequence[float],
    representative_point_xy: Sequence[float],
) -> dict[str, Any]:
    canonical = _canonicalize_geojson(geometry)
    return {
        "available": True,
        "unavailable_reason": None,
        "coordinate_space": coordinate_space,
        "crs": crs,
        "geometry_type": canonical["type"],
        "geometry": canonical,
        "bbox_xyxy": _geometry_bbox(canonical),
        "centroid_xy": _clean_point(centroid_xy),
        "representative_point_xy": _clean_point(representative_point_xy),
    }


def _unavailable_geometry_space(
    coordinate_space: str,
    crs: str | None,
    reason: str | None,
) -> dict[str, Any]:
    return {
        "available": False,
        "unavailable_reason": reason or f"{coordinate_space} geometry is unavailable",
        "coordinate_space": coordinate_space,
        "crs": crs,
        "geometry_type": None,
        "geometry": None,
        "bbox_xyxy": None,
        "centroid_xy": None,
        "representative_point_xy": None,
    }


def _available_point_space(
    coordinate_space: str,
    crs: str | None,
    point_xy: Sequence[float],
) -> dict[str, Any]:
    return {
        "available": True,
        "unavailable_reason": None,
        "coordinate_space": coordinate_space,
        "crs": crs,
        "point_xy": _clean_point(point_xy),
    }


def _unavailable_point_space(
    coordinate_space: str,
    crs: str | None,
    reason: str | None,
) -> dict[str, Any]:
    return {
        "available": False,
        "unavailable_reason": reason or f"{coordinate_space} point is unavailable",
        "coordinate_space": coordinate_space,
        "crs": crs,
        "point_xy": None,
    }


def _unavailable_metric(reason: str | None) -> dict[str, Any]:
    return {
        "available": False,
        "unavailable_reason": reason or "Projected metric is unavailable",
        "value": None,
        "unit": None,
        "crs": None,
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
    if values.size and int(values.max()) > np.iinfo(np.int32).max:
        raise ValueError("instance IDs exceed rasterio's supported int32 polygonization range")
    return values


def _counts_for_ids(values: np.ndarray, ids: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros(len(ids), dtype=np.int64)
    positions = np.searchsorted(ids, values.astype(np.int64, copy=False))
    return np.bincount(positions, minlength=len(ids)).astype(np.int64)


def _positive_runs(line: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    runs: dict[int, list[tuple[int, int]]] = defaultdict(list)
    if len(line) == 0:
        return runs
    start = 0
    current = int(line[0])
    for index in range(1, len(line) + 1):
        value = int(line[index]) if index < len(line) else None
        if value == current:
            continue
        if current > 0:
            runs[current].append((start, index))
        start, current = index, value
    return runs


def _source_offset(metadata: Mapping[str, Any]) -> tuple[tuple[float, float] | None, str | None]:
    x0, y0 = metadata.get("global_x0"), metadata.get("global_y0")
    if not _is_finite_number(x0) or not _is_finite_number(y0):
        return None, "tile_metadata.global_x0/global_y0 are not both finite numeric values."
    return (float(x0), float(y0)), None


def _affine_coefficients(value: Any) -> tuple[float, float, float, float, float, float] | None:
    if value is None:
        return None
    if all(hasattr(value, name) for name in ("a", "b", "c", "d", "e", "f")):
        raw = [value.a, value.b, value.c, value.d, value.e, value.f]
    elif isinstance(value, Mapping):
        try:
            raw = [value[name] for name in ("a", "b", "c", "d", "e", "f")]
        except KeyError:
            return None
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 6:
        raw = list(value[:6])
    else:
        return None
    if not all(_is_finite_number(item) for item in raw):
        return None
    return tuple(float(item) for item in raw)  # type: ignore[return-value]


def _apply_affine(
    affine: Sequence[float],
    x: float,
    y: float,
) -> tuple[float, float]:
    a, b, c, d, e, f = affine
    return a * x + b * y + c, d * x + e * y + f


def _map_geometry(
    geometry: Mapping[str, Any],
    operation: Callable[[float, float], Sequence[float]],
) -> dict[str, Any]:
    parts = []
    for polygon in _polygon_parts(geometry):
        mapped_polygon = []
        for ring in polygon:
            mapped_polygon.append(
                [[float(value) for value in operation(float(point[0]), float(point[1]))] for point in ring]
            )
        parts.append(mapped_polygon)
    return _canonical_geometry(parts)


def _canonicalize_geojson(geometry: Mapping[str, Any]) -> dict[str, Any]:
    return _canonical_geometry(_polygon_parts(geometry))


def _polygon_parts(geometry: Mapping[str, Any]) -> list[list[list[list[int | float]]]]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Polygon":
        raw_parts = [coordinates]
    elif geometry_type == "MultiPolygon":
        raw_parts = coordinates
    else:
        raise ValueError(f"Expected Polygon or MultiPolygon geometry, got {geometry_type!r}")
    if not isinstance(raw_parts, Sequence) or isinstance(raw_parts, (str, bytes)):
        raise ValueError("Polygon geometry coordinates must be a sequence")
    parts: list[list[list[list[int | float]]]] = []
    for raw_polygon in raw_parts:
        if not isinstance(raw_polygon, Sequence) or not raw_polygon:
            raise ValueError("Every polygon must contain an exterior ring")
        polygon: list[list[list[int | float]]] = []
        for raw_ring in raw_polygon:
            if not isinstance(raw_ring, Sequence):
                raise ValueError("Polygon rings must be coordinate sequences")
            ring = []
            for point in raw_ring:
                if (
                    not isinstance(point, Sequence)
                    or isinstance(point, (str, bytes))
                    or len(point) < 2
                    or not _is_finite_number(point[0])
                    or not _is_finite_number(point[1])
                ):
                    raise ValueError("Polygon positions must contain two finite numeric values")
                ring.append([_clean_number(point[0]), _clean_number(point[1])])
            polygon.append(ring)
        parts.append(polygon)
    return parts


def _canonical_geometry(parts: Sequence[Sequence[Sequence[Sequence[float]]]]) -> dict[str, Any]:
    canonical_parts = [_canonical_polygon(polygon) for polygon in parts]
    canonical_parts.sort(key=_polygon_sort_key)
    if len(canonical_parts) == 1:
        return {"type": "Polygon", "coordinates": canonical_parts[0]}
    return {"type": "MultiPolygon", "coordinates": canonical_parts}


def _canonical_polygon(polygon: Sequence[Sequence[Sequence[float]]]) -> list[list[list[int | float]]]:
    if not polygon:
        raise ValueError("A polygon must contain an exterior ring")
    exterior = _canonical_ring(polygon[0], positive_area=True)
    holes = [_canonical_ring(ring, positive_area=False) for ring in polygon[1:]]
    holes.sort(key=_ring_sort_key)
    return [exterior, *holes]


def _canonical_ring(
    ring: Sequence[Sequence[float]],
    *,
    positive_area: bool,
) -> list[list[int | float]]:
    points = [[_clean_number(point[0]), _clean_number(point[1])] for point in ring]
    if points and points[0] == points[-1]:
        points = points[:-1]
    compact: list[list[int | float]] = []
    for point in points:
        if not compact or point != compact[-1]:
            compact.append(point)
    if len(compact) < 3:
        raise ValueError("Polygon rings must contain at least three distinct positions")
    area = _signed_ring_area([*compact, compact[0]])
    if math.isclose(area, 0.0, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("Polygon rings may not have zero area")
    if (area > 0) != positive_area:
        compact.reverse()

    minimum_point = min(tuple(point) for point in compact)
    candidate_indices = [index for index, point in enumerate(compact) if tuple(point) == minimum_point]
    index = min(
        candidate_indices,
        key=lambda start: tuple(tuple(point) for point in (compact[start:] + compact[:start])),
    )
    rotated = compact[index:] + compact[:index]
    return [*rotated, rotated[0].copy()]


def _polygon_sort_key(polygon: Sequence[Sequence[Sequence[float]]]) -> tuple[Any, ...]:
    bbox = _rings_bbox(polygon)
    return (*bbox, -_polygon_area(polygon), tuple(tuple(point) for point in polygon[0]))


def _ring_sort_key(ring: Sequence[Sequence[float]]) -> tuple[Any, ...]:
    bbox = _rings_bbox([ring])
    return (*bbox, -abs(_signed_ring_area(ring)), tuple(tuple(point) for point in ring))


def _geometry_bbox(geometry: Mapping[str, Any]) -> list[int | float]:
    parts = _polygon_parts(geometry)
    xs = [float(point[0]) for polygon in parts for ring in polygon for point in ring]
    ys = [float(point[1]) for polygon in parts for ring in polygon for point in ring]
    return [_clean_number(min(xs)), _clean_number(min(ys)), _clean_number(max(xs)), _clean_number(max(ys))]


def _rings_bbox(rings: Sequence[Sequence[Sequence[float]]]) -> tuple[float, float, float, float]:
    xs = [float(point[0]) for ring in rings for point in ring]
    ys = [float(point[1]) for ring in rings for point in ring]
    return min(xs), min(ys), max(xs), max(ys)


def _geometry_area(geometry: Mapping[str, Any]) -> float:
    return sum(_polygon_area(polygon) for polygon in _polygon_parts(geometry))


def _polygon_area(polygon: Sequence[Sequence[Sequence[float]]]) -> float:
    return abs(_signed_ring_area(polygon[0])) - sum(abs(_signed_ring_area(ring)) for ring in polygon[1:])


def _signed_ring_area(ring: Sequence[Sequence[float]]) -> float:
    return 0.5 * sum(
        float(first[0]) * float(second[1]) - float(second[0]) * float(first[1])
        for first, second in zip(ring, ring[1:])
    )


def _geometry_perimeter(geometry: Mapping[str, Any]) -> float:
    return sum(
        math.hypot(float(second[0]) - float(first[0]), float(second[1]) - float(first[1]))
        for polygon in _polygon_parts(geometry)
        for ring in polygon
        for first, second in zip(ring, ring[1:])
    )


def _geometry_counts(geometry: Mapping[str, Any]) -> dict[str, int]:
    parts = _polygon_parts(geometry)
    ring_count = sum(len(polygon) for polygon in parts)
    return {
        "component_count": len(parts),
        "ring_count": ring_count,
        "hole_count": ring_count - len(parts),
    }


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _clean_point(point: Sequence[float]) -> list[int | float]:
    return [_clean_number(point[0]), _clean_number(point[1])]


def _clean_number(value: Any) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else number


__all__ = [
    "SpatialContext",
    "build_node_geometries",
    "coordinate_point_variants",
    "polygonize_instance_map",
    "spatial_context",
]
