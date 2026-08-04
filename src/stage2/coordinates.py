"""Coordinate metadata for Stage 2 segment records.

Pixel conventions used by this module are deliberately explicit:

* pixel indices are zero-based and ordered ``(x, y) == (column, row)``;
* ``bbox_xyxy`` values are half-open: ``[x_min, y_min, x_max, y_max)``;
* integer pixel indices identify pixels, while spatial coordinates for their
  centres are offset by ``0.5`` on both axes.

Consequently, an affine transform is applied to source-pixel *edge*
coordinates for a bounding footprint and to source-pixel *centre* coordinates
for a segment centroid.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


PIXEL_COORDINATE_CONVENTIONS: dict[str, str] = {
    "index_origin": "zero-based",
    "axis_order": "x, y (column, row)",
    "bbox": "bbox_xyxy is [x_min, y_min, x_max, y_max) with exclusive maximum edges (half-open)",
    "centroid": "centroid_xy is measured at pixel centres; pixel index (x, y) has centre coordinate (x + 0.5, y + 0.5)",
    "source_pixel_space": "source/global pixel coordinates equal tile-local coordinates plus global_x0/global_y0",
}


def segment_coordinate_metadata(record: Mapping[str, Any], tile_metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Build local, source/global, world, and WGS84 metadata for one segment.

    ``record`` must contain ``bbox_xywh`` (or ``bbox_xyxy``). Its usual
    ``centroid_rc`` value is interpreted as the mean zero-based mask pixel
    index in ``(row, column)`` order and converted to a pixel-centre ``(x, y)``
    coordinate. ``tile_metadata`` is the normalized result from
    :func:`src.stage2.tiles.validate_tile`.
    """
    local_bbox = _pixel_bbox(record)
    centroid_index, centroid_source = _centroid_index_xy(record, local_bbox)
    local_centroid = [centroid_index[0] + 0.5, centroid_index[1] + 0.5]

    offset_x = _required_number(tile_metadata.get("global_x0"), "tile_metadata.global_x0")
    offset_y = _required_number(tile_metadata.get("global_y0"), "tile_metadata.global_y0")
    source_bbox = [
        local_bbox[0] + offset_x,
        local_bbox[1] + offset_y,
        local_bbox[2] + offset_x,
        local_bbox[3] + offset_y,
    ]
    source_centroid = [local_centroid[0] + offset_x, local_centroid[1] + offset_y]

    tile_pixel = {
        "bbox_xyxy": _clean_numbers(local_bbox),
        "centroid_xy": _clean_numbers(local_centroid),
        "centroid_index_xy": _clean_numbers(centroid_index),
        "centroid_source": centroid_source,
    }
    source_pixel = {
        "bbox_xyxy": _clean_numbers(source_bbox),
        "centroid_xy": _clean_numbers(source_centroid),
        "offset_from_tile_xy": _clean_numbers([offset_x, offset_y]),
    }

    tile_affine = _affine_coefficients(tile_metadata.get("tile_affine_transform"))
    source_affine_value = tile_metadata.get("source_affine_transform")
    if source_affine_value is None:
        source_affine_value = tile_metadata.get("affine_transform")
    source_affine = _affine_coefficients(source_affine_value)
    crs = tile_metadata.get("crs")
    if tile_affine is not None:
        local_ring = _bbox_ring(local_bbox)
        world_ring = [_apply_affine(tile_affine, x, y) for x, y in local_ring]
        world_centroid = _apply_affine(tile_affine, local_centroid[0], local_centroid[1])
        world = _available_space(world_centroid, world_ring, crs=crs)
        world["affine_pixel_space"] = "tile_pixel"
        world["affine_transform"] = list(tile_affine)
    elif source_affine is not None:
        source_ring = _bbox_ring(source_bbox)
        world_ring = [_apply_affine(source_affine, x, y) for x, y in source_ring]
        world_centroid = _apply_affine(source_affine, source_centroid[0], source_centroid[1])
        world = _available_space(world_centroid, world_ring, crs=crs)
        world["affine_pixel_space"] = "source_pixel"
        world["affine_transform"] = list(source_affine)
    else:
        reason = tile_metadata.get("georeferencing_unavailable_reason") or "No usable affine transform is available for the source raster."
        world = _unavailable_space(str(reason), crs=crs)

    if not world["available"]:
        wgs84 = _unavailable_space(str(world["unavailable_reason"]), crs="EPSG:4326")
    elif not crs:
        wgs84 = _unavailable_space("The affine world coordinates have no source CRS, so they cannot be transformed to WGS84.", crs="EPSG:4326")
    else:
        wgs84 = _to_wgs84(world, crs)

    # ``source_pixel`` and ``global_pixel`` are intentional aliases in the
    # serialized schema: both terms are used by the Stage 1/2 interfaces.
    return {
        "conventions": dict(PIXEL_COORDINATE_CONVENTIONS),
        "tile_pixel": tile_pixel,
        "source_pixel": source_pixel,
        "global_pixel": copy.deepcopy(source_pixel),
        "projected": world,
        "world": copy.deepcopy(world),
        "wgs84": wgs84,
    }


def attach_segment_coordinates(
    record: dict[str, Any],
    tile_metadata: Mapping[str, Any],
    *,
    in_place: bool = False,
) -> dict[str, Any]:
    """Return a segment record with a ``coordinates`` mapping attached."""
    result = record if in_place else copy.deepcopy(record)
    result["coordinates"] = segment_coordinate_metadata(result, tile_metadata)
    return result


def attach_coordinates_to_segments(
    records: Iterable[dict[str, Any]],
    tile_metadata: Mapping[str, Any],
    *,
    in_place: bool = False,
) -> list[dict[str, Any]]:
    """Attach coordinate metadata to every segment record."""
    return [attach_segment_coordinates(record, tile_metadata, in_place=in_place) for record in records]


def segment_bbox_geojson_feature(
    record: Mapping[str, Any],
    *,
    prefer_wgs84: bool = True,
    properties: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a GeoJSON Feature for the segment's transformed bbox footprint.

    The geometry is the affine-transformed pixel bounding box, not a polygonized
    mask boundary. WGS84 is preferred when available; otherwise the source CRS
    world footprint is returned and its CRS is stated in feature properties.
    """
    coordinates = record.get("coordinates")
    if not isinstance(coordinates, Mapping):
        raise ValueError("Segment record has no coordinates; call attach_segment_coordinates first")

    wgs84 = coordinates.get("wgs84")
    projected = coordinates.get("projected")
    if prefer_wgs84 and isinstance(wgs84, Mapping) and wgs84.get("available"):
        space_name, space = "wgs84", wgs84
    elif isinstance(projected, Mapping) and projected.get("available"):
        space_name, space = "projected", projected
    elif isinstance(wgs84, Mapping) and wgs84.get("available"):
        space_name, space = "wgs84", wgs84
    else:
        reason = projected.get("unavailable_reason") if isinstance(projected, Mapping) else "coordinate metadata is incomplete"
        raise ValueError(f"Segment has no GeoJSON-compatible world footprint: {reason}")

    feature_properties: dict[str, Any] = {
        "segment_id": record.get("segment_id"),
        "source_tile_id": record.get("source_tile_id"),
        "coordinate_space": space_name,
        "crs": space.get("crs"),
        "geometry_basis": "segment_pixel_bbox",
    }
    if properties:
        feature_properties.update(properties)
    return {
        "type": "Feature",
        "id": record.get("segment_id"),
        "bbox": copy.deepcopy(space["bbox_xyxy"]),
        "geometry": copy.deepcopy(space["footprint"]),
        "properties": feature_properties,
    }


def _pixel_bbox(record: Mapping[str, Any]) -> list[float]:
    if record.get("bbox_xyxy") is not None:
        values = _numeric_sequence(record["bbox_xyxy"], "record.bbox_xyxy", length=4)
        x0, y0, x1, y1 = values
    elif record.get("bbox_xywh") is not None:
        x0, y0, width, height = _numeric_sequence(record["bbox_xywh"], "record.bbox_xywh", length=4)
        if width <= 0 or height <= 0:
            raise ValueError("record.bbox_xywh width and height must be positive")
        x1, y1 = x0 + width, y0 + height
    else:
        raise ValueError("Segment record must contain bbox_xywh or bbox_xyxy")
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Segment pixel bbox must have positive width and height")
    return [x0, y0, x1, y1]


def _centroid_index_xy(record: Mapping[str, Any], bbox: Sequence[float]) -> tuple[list[float], str]:
    if record.get("centroid_rc") is not None:
        row, column = _numeric_sequence(record["centroid_rc"], "record.centroid_rc", length=2)
        return [column, row], "record.centroid_rc (mean mask pixel index)"
    if record.get("centroid_index_xy") is not None:
        x, y = _numeric_sequence(record["centroid_index_xy"], "record.centroid_index_xy", length=2)
        return [x, y], "record.centroid_index_xy"
    if record.get("centroid_xy") is not None:
        x, y = _numeric_sequence(record["centroid_xy"], "record.centroid_xy", length=2)
        return [x - 0.5, y - 0.5], "record.centroid_xy (pixel-centre coordinate)"
    return [(bbox[0] + bbox[2] - 1.0) / 2.0, (bbox[1] + bbox[3] - 1.0) / 2.0], "derived from bbox centre because no mask centroid was present"


def _affine_coefficients(value: Any) -> tuple[float, float, float, float, float, float] | None:
    if value is None:
        return None
    if all(hasattr(value, name) for name in ("a", "b", "c", "d", "e", "f")):
        values = [value.a, value.b, value.c, value.d, value.e, value.f]
    elif isinstance(value, Mapping):
        try:
            values = [value[name] for name in ("a", "b", "c", "d", "e", "f")]
        except KeyError:
            return None
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) >= 6:
        values = list(value[:6])
    else:
        return None
    try:
        coefficients = tuple(float(item) for item in values)
    except (TypeError, ValueError):
        return None
    return coefficients if all(math.isfinite(item) for item in coefficients) else None


def _apply_affine(affine: Sequence[float], x: float, y: float) -> list[float]:
    a, b, c, d, e, f = affine
    return [a * x + b * y + c, d * x + e * y + f]


def _bbox_ring(bbox: Sequence[float]) -> list[list[float]]:
    x0, y0, x1, y1 = bbox
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def _available_space(centroid: Sequence[float], ring: Sequence[Sequence[float]], *, crs: Any) -> dict[str, Any]:
    clean_ring = [_clean_numbers(point) for point in ring]
    xs, ys = zip(*clean_ring)
    return {
        "available": True,
        "unavailable_reason": None,
        "crs": str(crs) if crs is not None else None,
        "centroid_xy": _clean_numbers(centroid),
        "bbox_xyxy": [min(xs), min(ys), max(xs), max(ys)],
        "footprint": {"type": "Polygon", "coordinates": [clean_ring]},
    }


def _unavailable_space(reason: str, *, crs: Any) -> dict[str, Any]:
    return {
        "available": False,
        "unavailable_reason": reason,
        "crs": str(crs) if crs is not None else None,
        "centroid_xy": None,
        "bbox_xyxy": None,
        "footprint": None,
    }


def _to_wgs84(world: Mapping[str, Any], source_crs: Any) -> dict[str, Any]:
    try:
        from rasterio.crs import CRS
        from rasterio.warp import transform

        source = CRS.from_user_input(source_crs)
        destination = CRS.from_epsg(4326)
        ring = world["footprint"]["coordinates"][0]
        xs, ys = transform(source, destination, [point[0] for point in ring], [point[1] for point in ring])
        centroid_x, centroid_y = transform(
            source,
            destination,
            [world["centroid_xy"][0]],
            [world["centroid_xy"][1]],
        )
        return _available_space([centroid_x[0], centroid_y[0]], list(zip(xs, ys)), crs="EPSG:4326")
    except Exception as exc:
        reason = f"World coordinates could not be transformed from {source_crs!s} to WGS84: {type(exc).__name__}: {exc}"
        return _unavailable_space(reason, crs="EPSG:4326")


def _numeric_sequence(value: Any, name: str, *, length: int) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} numeric values")
    result = [_required_number(item, f"{name}[{index}]") for index, item in enumerate(value)]
    return result


def _required_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _clean_numbers(values: Sequence[float]) -> list[int | float]:
    return [_clean_number(value) for value in values]


def _clean_number(value: float) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else number


__all__ = [
    "PIXEL_COORDINATE_CONVENTIONS",
    "attach_coordinates_to_segments",
    "attach_segment_coordinates",
    "segment_bbox_geojson_feature",
    "segment_coordinate_metadata",
]
