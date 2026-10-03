"""Extract one observed tile window from an annual combined graph."""
from __future__ import annotations

import copy

from shapely.geometry import MultiPolygon, Polygon, box, mapping, shape

from src.stage4.schema import SCHEMA_VERSION, stable_digest, validate_combined_graph


def _polygonal(geometry):
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    parts = []
    pending = [geometry]
    while pending:
        part = pending.pop()
        if isinstance(part, (Polygon, MultiPolygon)) and not part.is_empty:
            parts.append(part)
        elif hasattr(part, "geoms"):
            pending.extend(part.geoms)
    from shapely.ops import unary_union
    return unary_union(parts) if parts else Polygon()


def _contact_anchor(contact):
    lines = []
    pending = [contact]
    while pending:
        part = pending.pop()
        if part.geom_type in {"LineString", "LinearRing"} and part.length > 0:
            lines.append(part)
        elif hasattr(part, "geoms"):
            pending.extend(part.geoms)
    if not lines:
        raise ValueError("Clipped adjacency has no shared line")
    line = min(lines, key=lambda item: (-item.length, item.bounds))
    point = line.interpolate(line.length / 2)
    return [float(point.x), float(point.y)]


def clip_combined_graph_to_tile(document, tile_id, extent):
    """Keep only the mapped pixels inside one PNG's source-pixel window.

    Annual nodes may cross overlap seams. Clipping their geometry avoids
    treating an adjacent tile's pixels as observations of this one PNG.
    """
    if tile_id not in document["graph"]["selected_tile_ids"]:
        raise ValueError(f"Tile {tile_id!r} is absent from the combined graph")
    x0, y0, x1, y1 = map(float, extent)
    if not x0 < x1 or not y0 < y1:
        raise ValueError("Tile extent must have positive width and height")
    window = box(x0, y0, x1, y1)
    nodes, shapes = [], {}
    for original in document["nodes"]:
        geometry = _polygonal(shape(original["geometry_source_pixel"]).intersection(window))
        if geometry.is_empty or geometry.area < 0.5:
            continue
        node = copy.deepcopy(original)
        node["instance_id"] = len(nodes) + 1
        node["area_pixels"] = int(round(geometry.area))
        node["geometry_type"] = geometry.geom_type
        node["geometry_source_pixel"] = mapping(geometry)
        node["bbox_source_pixel_xyxy"] = list(map(float, geometry.bounds))
        node["centroid_source_pixel_xy"] = [float(geometry.centroid.x), float(geometry.centroid.y)]
        node["representative_point_source_pixel_xy"] = list(map(float, geometry.representative_point().coords[0]))
        node["tile_window_clip"] = {"tile_id": tile_id, "source_pixel_extent_xyxy": [x0, y0, x1, y1]}
        nodes.append(node)
        shapes[node["node_id"]] = geometry
    if not nodes:
        raise ValueError(f"Tile {tile_id!r} has no mapped graph nodes")
    edges = []
    for original in document["edges"]:
        source, target = original["source"], original["target"]
        if source not in shapes or target not in shapes:
            continue
        contact = shapes[source].boundary.intersection(shapes[target].boundary)
        if contact.length < 0.5:
            continue
        edge = copy.deepcopy(original)
        edge["shared_boundary_length_pixels"] = float(contact.length)
        edge["shared_boundary_side_count"] = int(round(contact.length))
        edge["contact_centroid_source_pixel_xy"] = [float(contact.centroid.x), float(contact.centroid.y)]
        edge["contact_anchor_source_pixel_xy"] = _contact_anchor(contact)
        edges.append(edge)
    graph = copy.deepcopy(document["graph"])
    graph.update(id=f"{graph['id']}::tile::{stable_digest([tile_id, [x0,y0,x1,y1]])[:16]}",
                 selected_tile_ids=[tile_id], source_pixel_extent_xyxy=[x0, y0, x1, y1],
                 node_count=len(nodes), edge_count=len(edges),
                 tile_selection={"source_graph_id":document["graph"]["id"],
                                 "tile_id":tile_id, "clipped_to_source_pixel_window":True})
    selected = {"schema_version": SCHEMA_VERSION, "graph": graph,
                "nodes": nodes, "edges": edges}
    validate_combined_graph(selected)
    return selected
