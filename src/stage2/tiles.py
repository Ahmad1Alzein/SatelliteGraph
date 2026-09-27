from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


def discover_tiles(project_root: Path, paths: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    tiles_root = _configured_path(project_root, paths, "tiles_directory", "data/tiles")
    metadata_root = _configured_path(project_root, paths, "metadata_directory", "data/metadata")
    found = []
    if not tiles_root.is_dir():
        return found
    for source_dir in sorted(p for p in tiles_root.iterdir() if p.is_dir()):
        csv_path = metadata_root / f"{source_dir.name}.csv"
        rows = _csv_rows(csv_path) if csv_path.is_file() else {}
        for image_path in sorted(source_dir.glob("*.png")):
            stem = image_path.stem
            row = rows.get(stem, {})
            local_id = str(row.get("tile_id", stem.removeprefix("tile_")))
            found.append({"tile_id": f"{source_dir.name}__{stem}", "source_image_id": source_dir.name, "local_tile_id": local_id, "image_path": str(image_path), "metadata_path": str(csv_path), "metadata": row})
    return found


def resolve_tile(project_root: Path, tile_id: str, paths: Mapping[str, Any] | None = None) -> dict[str, Any]:
    discovered = discover_tiles(project_root, paths)
    exact = [item for item in discovered if item["tile_id"] == tile_id]
    if exact:
        return exact[0]
    candidates = [item for item in discovered if item["local_tile_id"] == str(tile_id) or Path(item["image_path"]).stem == tile_id]
    if not candidates:
        raise FileNotFoundError(f"Tile '{tile_id}' was not found under data/tiles")
    if len(candidates) > 1:
        names = [item["tile_id"] for item in candidates[:10]]
        raise ValueError(f"Tile id '{tile_id}' is ambiguous; use one of: {names}")
    return candidates[0]


def validate_tile(
    project_root: Path,
    tile_id: str,
    paths: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    item = resolve_tile(project_root, tile_id, paths)
    image_path = Path(item["image_path"])
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Tile image is corrupt or unreadable: {image_path}")
    if image.ndim == 2:
        image = image[..., None]
    elif image.ndim == 3 and image.shape[2] >= 3:
        image = image[..., [2, 1, 0] + list(range(3, image.shape[2]))]
    h, w = image.shape[:2]
    row = item["metadata"]
    errors, warnings = [], []
    expected_w, expected_h = _int(row.get("width")), _int(row.get("height"))
    if expected_w is None or expected_h is None:
        warnings.append("width/height unavailable in Stage 1 metadata")
    elif (expected_w, expected_h) != (w, h):
        errors.append(f"image dimensions {w}x{h} disagree with metadata {expected_w}x{expected_h}")
    x0, y0 = _int(row.get("x0")), _int(row.get("y0"))
    if x0 is None or y0 is None:
        errors.append("global offsets x0/y0 are missing")
    elif x0 < 0 or y0 < 0:
        errors.append("global offsets must be non-negative")
    raw_root = _configured_path(project_root, paths, "raw_directory", "data/raw")
    source_raster, source_notes = _source_raster_metadata(raw_root, item["source_image_id"])
    normalized = {
        "tile_id": item["tile_id"], "local_tile_id": item["local_tile_id"],
        "source_image_id": item["source_image_id"], "image_path": str(image_path.resolve()),
        "metadata_path": str(Path(item["metadata_path"]).resolve()),
        "row_index": None, "column_index": None, "global_x0": x0, "global_y0": y0,
        "width": w, "height": h, "original_unpadded_width": w,
        "original_unpadded_height": h, "padding": {"top": 0, "right": 0, "bottom": 0, "left": 0},
        "selected_bands": list(range(1, image.shape[2] + 1)), "rgb_band_mapping": [1, 2, 3] if image.shape[2] >= 3 else None,
        "valid_data_available": False,
        "compatibility_notes": source_notes,
    }
    normalized.update(source_raster)
    if row.get("alpha_band"):
        alpha_band = _int(row["alpha_band"])
        valid_w,valid_h = _int(row.get("valid_width")),_int(row.get("valid_height"))
        if alpha_band!=4 or image.shape[2]!=4 or valid_w is None or valid_h is None or not (0<valid_w<=w and 0<valid_h<=h):
            errors.append("Invalid Stage 1 alpha/padding metadata")
        else:
            normalized.update(alpha_band_index=3,valid_data_available=True,
                              original_unpadded_width=valid_w,original_unpadded_height=valid_h,
                              padding={"top":0,"left":0,"right":w-valid_w,"bottom":h-valid_h})
    normalized["tile_affine_transform"] = _offset_affine(normalized["source_affine_transform"], x0, y0)
    normalized.update(_tile_spatial_bounds(normalized["tile_affine_transform"], w, h, normalized.get("crs")))
    source_width, source_height = normalized.get("source_raster_width"), normalized.get("source_raster_height")
    if x0 is not None and y0 is not None and source_width is not None and source_height is not None:
        if x0 + normalized["original_unpadded_width"] > source_width or y0 + normalized["original_unpadded_height"] > source_height:
            errors.append(
                f"tile extent ({x0}, {y0}, {x0 + w}, {y0 + h}) exceeds source raster "
                f"dimensions {source_width}x{source_height}"
            )
    if errors:
        raise ValueError("Tile validation failed: " + "; ".join(errors))
    report = {"valid": True, "errors": errors, "warnings": warnings + normalized["compatibility_notes"], "image_dtype": str(image.dtype), "image_shape": list(image.shape), "finite": bool(np.isfinite(image).all())}
    return image, normalized, report


def _csv_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return {row["tile_name"].rsplit(".", 1)[0]: row for row in csv.DictReader(stream)}


def _int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _source_raster_metadata(raw_root: Path, source_image_id: str) -> tuple[dict[str, Any], list[str]]:
    """Read source-raster metadata only; the raster bands are never read."""
    empty = {
        "source_raster_path": None,
        "source_raster_metadata_available": False,
        "source_raster_metadata_unavailable_reason": None,
        "source_dtype": None,
        "source_dtypes": None,
        "source_raster_width": None,
        "source_raster_height": None,
        "source_band_count": None,
        "nodata": None,
        "source_affine_transform": None,
        "tile_affine_transform": None,
        "affine_transform": None,
        "affine_transform_gdal": None,
        "crs": None,
        "crs_wkt": None,
        "projected_bounds": None,
        "geographic_bounds": None,
        "source_projected_bounds": None,
        "source_geographic_bounds": None,
        "geographic_bounds_unavailable_reason": None,
        "georeferencing_available": False,
        "georeferencing_unavailable_reason": None,
    }
    candidates = [raw_root / f"{source_image_id}{suffix}" for suffix in (".tif", ".tiff")]
    matches = [path for path in candidates if path.is_file()]
    if not matches:
        expected = " or ".join(str(path.resolve()) for path in candidates)
        reason = f"No exact source raster match was found; expected {expected}"
        empty["source_raster_metadata_unavailable_reason"] = reason
        empty["georeferencing_unavailable_reason"] = reason
        return empty, [reason, "Stage 1 PNG/CSV artifacts do not preserve source dtype, padding, or original band identities."]
    if len(matches) > 1:
        paths = ", ".join(str(path.resolve()) for path in matches)
        reason = f"Source raster metadata is ambiguous because both exact matches exist: {paths}"
        empty["source_raster_metadata_unavailable_reason"] = reason
        empty["georeferencing_unavailable_reason"] = reason
        return empty, [reason, "Stage 1 PNG/CSV artifacts do not preserve source dtype, padding, or original band identities."]

    path = matches[0]
    empty["source_raster_path"] = str(path.resolve())
    try:
        import rasterio

        with rasterio.open(path) as source:
            transform = source.transform
            crs = source.crs
            bounds = source.bounds
            dtypes = [str(value) for value in source.dtypes]
            affine_coefficients = [float(getattr(transform, name)) for name in ("a", "b", "c", "d", "e", "f")]
            projected_bounds = [float(bounds.left), float(bounds.bottom), float(bounds.right), float(bounds.top)]
            geographic_bounds = None
            geographic_bounds_reason = None
            if crs is None:
                geographic_bounds_reason = "The exact source raster has no CRS, so its bounds cannot be transformed to WGS84."
            else:
                try:
                    from rasterio.warp import transform_bounds

                    geographic_bounds = [
                        float(value)
                        for value in transform_bounds(crs, "EPSG:4326", *projected_bounds, densify_pts=21)
                    ]
                except Exception as exc:
                    geographic_bounds_reason = f"Source bounds could not be transformed to WGS84: {type(exc).__name__}: {exc}"
            metadata = {
                **empty,
                "source_raster_metadata_available": True,
                "source_raster_metadata_unavailable_reason": None,
                "source_dtype": dtypes[0] if dtypes else None,
                "source_dtypes": dtypes,
                "source_raster_width": int(source.width),
                "source_raster_height": int(source.height),
                "source_band_count": int(source.count),
                "nodata": _scalar(source.nodata),
                # Source/global pixel -> source-CRS world coordinates, in
                # Affine constructor order [a, b, c, d, e, f].
                "source_affine_transform": affine_coefficients,
                "affine_transform": affine_coefficients,
                "affine_transform_gdal": [float(value) for value in transform.to_gdal()],
                "crs": crs.to_string() if crs is not None else None,
                "crs_wkt": crs.to_wkt() if crs is not None else None,
                "source_projected_bounds": projected_bounds,
                "source_geographic_bounds": geographic_bounds,
                "geographic_bounds_unavailable_reason": geographic_bounds_reason,
                "georeferencing_available": crs is not None,
                "georeferencing_unavailable_reason": None if crs is not None else "The exact source raster has no CRS.",
            }
    except Exception as exc:
        reason = f"Could not read metadata from exact source raster {path.resolve()}: {type(exc).__name__}: {exc}"
        empty["source_raster_metadata_unavailable_reason"] = reason
        empty["georeferencing_unavailable_reason"] = reason
        return empty, [reason, "Stage 1 PNG/CSV artifacts do not preserve source dtype, padding, or original band identities."]

    notes = ["Source raster metadata was recovered without reading raster pixels; tile pixels still come from the Stage 1 PNG artifact.", "Stage 1 PNG/CSV artifacts do not preserve padding or original band identities."]
    if metadata["georeferencing_unavailable_reason"]:
        notes.append(metadata["georeferencing_unavailable_reason"])
    if metadata["geographic_bounds_unavailable_reason"] and metadata["geographic_bounds_unavailable_reason"] not in notes:
        notes.append(metadata["geographic_bounds_unavailable_reason"])
    return metadata, notes


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _offset_affine(transform: Any, x0: int | None, y0: int | None) -> list[float] | None:
    """Convert a source-pixel affine to a tile-local-pixel affine."""
    if transform is None or x0 is None or y0 is None:
        return None
    a, b, c, d, e, f = (float(value) for value in transform)
    return [a, b, a * x0 + b * y0 + c, d, e, d * x0 + e * y0 + f]


def _tile_spatial_bounds(
    transform: Any,
    width: int,
    height: int,
    crs: str | None,
) -> dict[str, Any]:
    if transform is None:
        return {"projected_bounds": None, "geographic_bounds": None}
    a, b, c, d, e, f = (float(value) for value in transform)
    corners = [
        (c, f),
        (a * width + c, d * width + f),
        (b * height + c, e * height + f),
        (a * width + b * height + c, d * width + e * height + f),
    ]
    xs, ys = zip(*corners)
    projected = [min(xs), min(ys), max(xs), max(ys)]
    geographic = None
    if crs:
        try:
            from rasterio.warp import transform as transform_coordinates

            lon, lat = transform_coordinates(crs, "EPSG:4326", list(xs), list(ys))
            geographic = [min(lon), min(lat), max(lon), max(lat)]
        except Exception:
            geographic = None
    return {"projected_bounds": projected, "geographic_bounds": geographic}


def _configured_path(
    project_root: Path,
    paths: Mapping[str, Any] | None,
    key: str,
    default: str,
) -> Path:
    value = (paths or {}).get(key, default)
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else project_root / path
