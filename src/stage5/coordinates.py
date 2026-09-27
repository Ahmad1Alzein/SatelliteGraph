"""Source-pixel edges -> native world CRS -> shared projected metre CRS."""
from __future__ import annotations

import numpy as np
from rasterio.crs import CRS
from rasterio.warp import transform_geom, transform
from shapely import affinity
from shapely.geometry import mapping, shape


def metric_crs(value):
    crs = CRS.from_user_input(value)
    if not crs.is_projected or not np.isclose(crs.linear_units_factor[1], 1.0):
        raise ValueError("Stage 5 target CRS must be projected with metre units")
    return crs


def affine_geometry(geometry, coefficients):
    """Repository/Rasterio order [a,b,c,d,e,f], NOT GDAL geotransform order."""
    values = np.asarray(coefficients, dtype=float)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("Affine must contain six finite [a,b,c,d,e,f] coefficients")
    a, b, c, d, e, f = values
    if abs(a * e - b * d) < 1e-15:
        raise ValueError("Singular affine transform")
    return affinity.affine_transform(geometry, [a, b, d, e, c, f])


def project_snapshot(document, target_crs, *, georeference=None):
    target = metric_crs(target_crs)
    metadata = {**document["graph"], **(georeference or {})}
    if not metadata.get("crs"):
        raise ValueError("Yearly graph has no CRS; provide a verified georeference")
    if metadata.get("source_affine_transform") is None:
        raise ValueError("Yearly graph has no source affine transform")
    source = CRS.from_user_input(metadata["crs"])
    geometries = []
    for node in document["nodes"]:
        polygon = shape(node["geometry_source_pixel"])
        if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
            raise ValueError(f"Invalid polygon for node {node['node_id']}")
        # Densify long source-pixel segments before a nonlinear CRS conversion.
        if source != target:
            polygon = polygon.segmentize(8)
        world = affine_geometry(polygon, metadata["source_affine_transform"])
        if source != target:
            world = shape(transform_geom(source, target, mapping(world)))
        if not world.is_valid or world.is_empty or not np.isfinite(world.bounds).all():
            raise ValueError("Coordinate transformation produced invalid geometry")
        geometries.append(world)
    return geometries


def wgs84_geometry(geometry, source_crs):
    # WGS84 output is interchange only; degrees are never used for matching.
    return transform_geom(source_crs, "EPSG:4326", mapping(geometry.segmentize(100)))


def project_point(xy, metadata, target_crs):
    a, b, c, d, e, f = metadata["source_affine_transform"]
    x, y = xy
    xs, ys = transform(metadata["crs"], target_crs, [a*x+b*y+c], [d*x+e*y+f])
    return [xs[0], ys[0]]
