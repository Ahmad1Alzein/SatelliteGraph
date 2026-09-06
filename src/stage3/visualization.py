"""Deterministic visualizations for a spatial graph built from one Stage 2 tile.

The functions in this module intentionally do not depend on NetworkX.  Nodes
and edges may be dictionaries, dataclass-like objects, JSON arrays, or numpy
arrays.  Coordinates are interpreted in Stage 2 tile-pixel space: ``x`` is a
column, ``y`` is a row, and the origin is the upper-left corner.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Keep Matplotlib's font cache in a writable batch-safe location.  This must
# happen before importing Matplotlib itself.
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(tempfile.gettempdir(), "satellitegraph-matplotlib"),
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from matplotlib import patheffects
from PIL import Image, ImageDraw


# Keep these values byte-for-byte consistent with Stage 2's semantic preview.
STAGE2_CLASS_NAMES: tuple[str, ...] = (
    "bareland",
    "rangeland",
    "developed space",
    "road",
    "tree",
    "water",
    "agriculture land",
    "buildings",
    "unknown",
)

STAGE2_CLASS_COLORS: np.ndarray = np.asarray(
    [
        (170, 120, 80),
        (196, 185, 110),
        (190, 190, 190),
        (245, 215, 40),
        (45, 145, 55),
        (55, 125, 220),
        (215, 175, 65),
        (220, 70, 70),
        (115, 80, 145),
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True, slots=True)
class GraphStyle:
    """Rendering parameters shared by PNG and SVG outputs."""

    dpi: int = 100
    graph_background: str = "#111820"
    edge_color: str = "#f4f7fb"
    edge_width_min: float = 0.55
    edge_width_max: float = 3.25
    edge_alpha_min: float = 0.20
    edge_alpha_max: float = 0.86
    minimum_display_shared_boundary_pixels: float = 0.0
    minimum_display_node_area_pixels: float = 0.0
    minimum_display_unknown_area_pixels: float = 0.0
    minimum_display_topology_fragment_area_pixels: float = 0.0
    route_edges_through_contacts: bool = True
    show_contact_anchors: bool = True
    contact_anchor_color: str = "#00d7ff"
    contact_anchor_size: float = 5.0
    contact_anchor_alpha: float = 0.88
    contact_anchor_outline_color: str = "#111111"
    contact_anchor_outline_width: float = 0.25
    contact_anchor_minimum_deviation_pixels: float = 2.0
    node_size_min: float = 30.0
    node_size_max: float = 230.0
    node_alpha: float = 0.96
    node_outline_color: str = "#111111"
    node_outline_width: float = 0.75
    boundary_outline_color: str = "#ff9f1c"
    boundary_outline_width: float = 2.25
    label_color: str = "#ffffff"
    label_outline_color: str = "#111111"
    label_font_size: float = 6.5
    background_alpha: float = 0.90

    def __post_init__(self) -> None:
        if self.dpi <= 0:
            raise ValueError("GraphStyle.dpi must be positive")
        for name in (
            "edge_width_min",
            "edge_width_max",
            "minimum_display_shared_boundary_pixels",
            "minimum_display_node_area_pixels",
            "minimum_display_unknown_area_pixels",
            "minimum_display_topology_fragment_area_pixels",
            "contact_anchor_size",
            "contact_anchor_outline_width",
            "contact_anchor_minimum_deviation_pixels",
            "node_size_min",
            "node_size_max",
            "node_outline_width",
            "boundary_outline_width",
            "label_font_size",
        ):
            if not math.isfinite(float(getattr(self, name))) or float(getattr(self, name)) < 0:
                raise ValueError(f"GraphStyle.{name} must be finite and non-negative")
        if self.edge_width_max < self.edge_width_min:
            raise ValueError("GraphStyle.edge_width_max must be >= edge_width_min")
        if self.node_size_max < self.node_size_min:
            raise ValueError("GraphStyle.node_size_max must be >= node_size_min")
        for name in (
            "edge_alpha_min",
            "edge_alpha_max",
            "contact_anchor_alpha",
            "node_alpha",
            "background_alpha",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"GraphStyle.{name} must be between 0 and 1")
        if self.edge_alpha_max < self.edge_alpha_min:
            raise ValueError("GraphStyle.edge_alpha_max must be >= edge_alpha_min")
        for name in ("route_edges_through_contacts", "show_contact_anchors"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"GraphStyle.{name} must be a boolean")


@dataclass(frozen=True, slots=True)
class _VisualNode:
    token: str
    identifier: Any
    x: float
    y: float
    area_pixels: float
    class_id: int | None
    class_name: str | None
    touches_boundary: bool
    topology_fragment: bool


@dataclass(frozen=True, slots=True)
class _VisualEdge:
    source: str
    target: str
    shared_boundary: float
    contact_x: float | None
    contact_y: float | None


def render_graph_overlay(
    background: str | Path | Image.Image | np.ndarray,
    nodes: Any,
    edges: Any,
    *,
    style: GraphStyle | None = None,
    show_labels: bool = False,
    show_legend: bool = False,
) -> np.ndarray:
    """Return an RGB graph overlay aligned to a Stage 2 image.

    ``background`` may be a PNG path, a ``12_final`` directory (where
    ``final_overlay.png`` is preferred), a PIL image, or an RGB numpy array.
    """

    image = _load_rgb(background)
    visual_nodes, visual_edges = _prepare_graph(nodes, edges)
    return _render_overlay_prepared(
        image,
        visual_nodes,
        visual_edges,
        style or GraphStyle(),
        show_labels=show_labels,
        show_legend=show_legend,
    )


def save_graph_overlay(
    background: str | Path | Image.Image | np.ndarray,
    nodes: Any,
    edges: Any,
    output_path: str | Path,
    *,
    style: GraphStyle | None = None,
    show_labels: bool = False,
    show_legend: bool = False,
) -> Path:
    """Render and atomically save a graph-over-segmentation PNG."""

    output = render_graph_overlay(
        background,
        nodes,
        edges,
        style=style,
        show_labels=show_labels,
        show_legend=show_legend,
    )
    return _save_png(output_path, output)


def render_graph_only(
    nodes: Any,
    edges: Any,
    *,
    image_shape: Sequence[int] | np.ndarray | Mapping[str, Any] | None = None,
    style: GraphStyle | None = None,
    show_labels: bool = False,
    show_legend: bool = False,
) -> np.ndarray:
    """Return an RGB spatial node-link view with an upper-left origin.

    ``image_shape`` uses numpy order ``(height, width)``.  When omitted, a
    minimal canvas is inferred from node positions.
    """

    visual_nodes, visual_edges = _prepare_graph(nodes, edges)
    height, width = _resolve_shape(image_shape, visual_nodes)
    return _render_graph_prepared(
        visual_nodes,
        visual_edges,
        height,
        width,
        style or GraphStyle(),
        show_labels=show_labels,
        show_legend=show_legend,
    )


def save_graph_png(
    nodes: Any,
    edges: Any,
    output_path: str | Path,
    *,
    image_shape: Sequence[int] | np.ndarray | Mapping[str, Any] | None = None,
    style: GraphStyle | None = None,
    show_labels: bool = False,
    show_legend: bool = False,
) -> Path:
    """Render and atomically save the graph-only raster view."""

    output = render_graph_only(
        nodes,
        edges,
        image_shape=image_shape,
        style=style,
        show_labels=show_labels,
        show_legend=show_legend,
    )
    return _save_png(output_path, output)


def save_graph_svg(
    nodes: Any,
    edges: Any,
    output_path: str | Path,
    *,
    image_shape: Sequence[int] | np.ndarray | Mapping[str, Any] | None = None,
    style: GraphStyle | None = None,
    show_labels: bool = False,
    show_legend: bool = False,
) -> Path:
    """Atomically save a vector graph view in tile-pixel coordinates."""

    visual_nodes, visual_edges = _prepare_graph(nodes, edges)
    height, width = _resolve_shape(image_shape, visual_nodes)
    selected_style = style or GraphStyle()
    figure, axis = _new_figure(height, width, selected_style, selected_style.graph_background)
    _draw_graph(
        axis,
        visual_nodes,
        visual_edges,
        selected_style,
        show_labels=show_labels,
        show_legend=show_legend,
    )
    return _save_svg(output_path, figure)


def render_graph_overview(
    background: str | Path | Image.Image | np.ndarray,
    nodes: Any,
    edges: Any,
    *,
    style: GraphStyle | None = None,
    show_labels: bool = False,
    show_legend: bool = False,
    titles: bool = True,
) -> np.ndarray:
    """Return a side-by-side Stage 2 segmentation and Stage 3 graph image."""

    image = _load_rgb(background)
    visual_nodes, visual_edges = _prepare_graph(nodes, edges)
    graph = _render_graph_prepared(
        visual_nodes,
        visual_edges,
        image.shape[0],
        image.shape[1],
        style or GraphStyle(),
        show_labels=show_labels,
        show_legend=show_legend,
    )
    return _side_by_side(image, graph, titles=titles)


def save_graph_overview(
    background: str | Path | Image.Image | np.ndarray,
    nodes: Any,
    edges: Any,
    output_path: str | Path,
    *,
    style: GraphStyle | None = None,
    show_labels: bool = False,
    show_legend: bool = False,
    titles: bool = True,
) -> Path:
    """Render and atomically save the side-by-side overview PNG."""

    output = render_graph_overview(
        background,
        nodes,
        edges,
        style=style,
        show_labels=show_labels,
        show_legend=show_legend,
        titles=titles,
    )
    return _save_png(output_path, output)


def render_graph_refinement_comparison(
    before_background: str | Path | Image.Image | np.ndarray,
    before_nodes: Any,
    before_edges: Any,
    after_background: str | Path | Image.Image | np.ndarray,
    after_nodes: Any,
    after_edges: Any,
    *,
    style: GraphStyle | None = None,
    show_labels: bool = False,
    show_legend: bool = True,
) -> np.ndarray:
    """Render Stage 2 and refined adjacency graphs in aligned panels.

    Semantic segmentation images are the preferred backgrounds for this
    diagnostic because their class colors remain meaningful: in particular,
    purple is the explicit ``unknown`` class.  The same display thresholds and
    graph style are applied to both phases so visual density is comparable.
    """

    before_image = _load_rgb(before_background)
    after_image = _load_rgb(after_background)
    if before_image.shape != after_image.shape:
        raise ValueError(
            "Before- and after-refinement graph backgrounds must have one shape"
        )
    before_visual_nodes, before_visual_edges = _prepare_graph(
        before_nodes, before_edges
    )
    after_visual_nodes, after_visual_edges = _prepare_graph(after_nodes, after_edges)
    selected_style = style or GraphStyle()
    before_panel = _render_overlay_prepared(
        before_image,
        before_visual_nodes,
        before_visual_edges,
        selected_style,
        show_labels=show_labels,
        show_legend=show_legend,
    )
    after_panel = _render_overlay_prepared(
        after_image,
        after_visual_nodes,
        after_visual_edges,
        selected_style,
        show_labels=show_labels,
        show_legend=show_legend,
    )
    return _labeled_side_by_side(
        before_panel,
        after_panel,
        left_title="Before refinement: Stage 2 adjacency graph",
        right_title="After refinement: Stage 3 adjacency graph",
    )


def save_graph_refinement_comparison(
    before_background: str | Path | Image.Image | np.ndarray,
    before_nodes: Any,
    before_edges: Any,
    after_background: str | Path | Image.Image | np.ndarray,
    after_nodes: Any,
    after_edges: Any,
    output_path: str | Path,
    *,
    style: GraphStyle | None = None,
    show_labels: bool = False,
    show_legend: bool = True,
) -> Path:
    """Save the aligned before/after graph diagnostic as a PNG."""

    output = render_graph_refinement_comparison(
        before_background,
        before_nodes,
        before_edges,
        after_background,
        after_nodes,
        after_edges,
        style=style,
        show_labels=show_labels,
        show_legend=show_legend,
    )
    return _save_png(output_path, output)


def save_graph_visualizations(
    background: str | Path | Image.Image | np.ndarray,
    nodes: Any,
    edges: Any,
    output_directory: str | Path,
    *,
    style: GraphStyle | None = None,
    show_labels: bool = False,
    show_legend: bool = False,
    prefix: str = "tile_graph",
) -> dict[str, Path]:
    """Create the standard overlay, graph PNG/SVG, and overview bundle."""

    if not prefix or Path(prefix).name != prefix:
        raise ValueError("prefix must be a non-empty filename stem")
    image = _load_rgb(background)
    visual_nodes, visual_edges = _prepare_graph(nodes, edges)
    selected_style = style or GraphStyle()
    directory = Path(output_directory).expanduser()

    overlay = _render_overlay_prepared(
        image,
        visual_nodes,
        visual_edges,
        selected_style,
        show_labels=show_labels,
        show_legend=show_legend,
    )
    graph = _render_graph_prepared(
        visual_nodes,
        visual_edges,
        image.shape[0],
        image.shape[1],
        selected_style,
        show_labels=show_labels,
        show_legend=show_legend,
    )
    overview = _side_by_side(image, graph, titles=True)
    paths = {
        "overlay": _save_png(directory / f"{prefix}_overlay.png", overlay),
        "graph_png": _save_png(directory / f"{prefix}.png", graph),
        "overview": _save_png(directory / f"{prefix}_overview.png", overview),
    }

    figure, axis = _new_figure(image.shape[0], image.shape[1], selected_style, selected_style.graph_background)
    _draw_graph(
        axis,
        visual_nodes,
        visual_edges,
        selected_style,
        show_labels=show_labels,
        show_legend=show_legend,
    )
    paths["graph_svg"] = _save_svg(directory / f"{prefix}.svg", figure)
    return paths


def _prepare_graph(nodes: Any, edges: Any) -> tuple[list[_VisualNode], list[_VisualEdge]]:
    visual_nodes = _normalise_nodes(nodes)
    visual_edges = _normalise_edges(edges, visual_nodes)
    return visual_nodes, visual_edges


def _normalise_nodes(source: Any) -> list[_VisualNode]:
    records = _records(source, collection_key="nodes")
    nodes: list[_VisualNode] = []
    tokens: set[str] = set()
    for fallback_id, value in records:
        record = _as_mapping(value, "node")
        identifier = _node_identifier(record, fallback_id)
        token = _id_token(identifier)
        if token in tokens:
            raise ValueError(f"Duplicate graph node id after normalization: {identifier!r}")
        x, y = _node_position(record)
        area = _node_area(record)
        class_id, class_name = _node_class(record)
        boundary = _node_boundary(record)
        topology_fragment = _node_is_topology_fragment(record)
        nodes.append(
            _VisualNode(
                token,
                identifier,
                x,
                y,
                area,
                class_id,
                class_name,
                boundary,
                topology_fragment,
            )
        )
        tokens.add(token)
    nodes.sort(key=lambda node: _natural_key(node.token))
    return nodes


def _normalise_edges(source: Any, nodes: Sequence[_VisualNode]) -> list[_VisualEdge]:
    node_tokens = {node.token for node in nodes}
    combined: dict[tuple[str, str], dict[str, float]] = {}
    for _, value in _records(source, collection_key="edges", edge_records=True):
        start, end, attributes = _edge_parts(value)
        source_token, target_token = _id_token(start), _id_token(end)
        if source_token not in node_tokens or target_token not in node_tokens:
            missing = start if source_token not in node_tokens else end
            raise ValueError(f"Edge references unknown node id: {missing!r}")
        if source_token == target_token:
            raise ValueError(f"Self edge is not valid for tile adjacency: {start!r}")
        pair = tuple(sorted((source_token, target_token), key=_natural_key))
        boundary = _edge_boundary(attributes)
        contact = _edge_contact(attributes)
        aggregate = combined.setdefault(
            pair,
            {
                "shared_boundary": 0.0,
                "contact_weight": 0.0,
                "contact_x_weighted": 0.0,
                "contact_y_weighted": 0.0,
            },
        )
        aggregate["shared_boundary"] += boundary
        if contact is not None:
            # Duplicate edge records are collapsed deterministically.  Their
            # contact location is the boundary-length-weighted mean, matching
            # the meaning of Stage 3's canonical contact centroid.
            weight = boundary if boundary > 0 else 1.0
            aggregate["contact_weight"] += weight
            aggregate["contact_x_weighted"] += contact[0] * weight
            aggregate["contact_y_weighted"] += contact[1] * weight

    visual_edges: list[_VisualEdge] = []
    for pair in sorted(combined, key=lambda item: (_natural_key(item[0]), _natural_key(item[1]))):
        aggregate = combined[pair]
        contact_weight = aggregate["contact_weight"]
        contact_x = (
            aggregate["contact_x_weighted"] / contact_weight if contact_weight > 0 else None
        )
        contact_y = (
            aggregate["contact_y_weighted"] / contact_weight if contact_weight > 0 else None
        )
        visual_edges.append(
            _VisualEdge(
                source=pair[0],
                target=pair[1],
                shared_boundary=aggregate["shared_boundary"],
                contact_x=contact_x,
                contact_y=contact_y,
            )
        )
    return visual_edges


def _records(
    source: Any,
    *,
    collection_key: str,
    edge_records: bool = False,
) -> list[tuple[Any | None, Any]]:
    if source is None:
        return []
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping) and collection_key in payload:
            payload = payload[collection_key]
        return _records(payload, collection_key=collection_key, edge_records=edge_records)
    if isinstance(source, np.ndarray):
        source = source.tolist()
    if isinstance(source, Mapping):
        if collection_key in source and _looks_like_collection_wrapper(source, collection_key):
            return _records(source[collection_key], collection_key=collection_key, edge_records=edge_records)
        if _looks_like_record(source, edge=edge_records):
            return [(None, source)]
        return [(key, value) for key, value in source.items()]
    if isinstance(source, Iterable) and not isinstance(source, (str, bytes)):
        return [(None, value) for value in source]
    raise TypeError(f"{collection_key} must be an iterable, mapping, numpy array, or JSON path")


def _looks_like_collection_wrapper(value: Mapping[str, Any], key: str) -> bool:
    return key in value and key not in {"source", "target", "u", "v"}


def _looks_like_record(value: Mapping[str, Any], *, edge: bool) -> bool:
    if edge:
        return any(
            pair[0] in value and pair[1] in value
            for pair in (("source", "target"), ("source_id", "target_id"), ("u", "v"), ("node_u", "node_v"))
        ) or "endpoints" in value
    keys = {
        "id",
        "node_id",
        "instance_id",
        "segment_id",
        "x",
        "centroid_xy",
        "centroid_rc",
        "coordinates",
        "bbox_xyxy",
        "position",
    }
    return bool(keys.intersection(value))


def _as_mapping(value: Any, kind: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "_asdict"):
        mapped = value._asdict()
        if isinstance(mapped, Mapping):
            return mapped
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"Each {kind} must be a mapping or dataclass-like object, got {type(value).__name__}")


def _node_identifier(record: Mapping[str, Any], fallback: Any | None) -> Any:
    for view in _views(record):
        # ``node_id`` is the graph-wide/composite identity.  ``instance_id``
        # is only tile-local and must not shadow it when both are present.
        for name in ("node_id", "id", "instance_id", "segment_id"):
            if name in view and view[name] is not None:
                return _scalar(view[name])
    if fallback is not None:
        return _scalar(fallback)
    raise ValueError("Graph node is missing instance_id/node_id/id/segment_id")


def _node_position(record: Mapping[str, Any]) -> tuple[float, float]:
    views = _views(record)
    preferred = (
        "representative_point_xy",
        "representative_xy",
        "visualization_xy",
        "position_xy",
        "tile_centroid_xy",
        "centroid_xy",
    )
    for view in views:
        for name in preferred:
            if name in view and view[name] is not None:
                return _xy(view[name], f"node.{name}")
        position = view.get("position")
        if isinstance(position, Mapping) and "x" in position and "y" in position:
            return _finite(position["x"], "node.position.x"), _finite(position["y"], "node.position.y")
        if "x" in view and "y" in view:
            return _finite(view["x"], "node.x"), _finite(view["y"], "node.y")

    for view in views:
        representative = view.get("representative_point")
        if isinstance(representative, Mapping):
            tile_value = representative.get("tile_pixel")
            if tile_value is not None:
                if isinstance(tile_value, Mapping) and "x" in tile_value and "y" in tile_value:
                    return (
                        _finite(tile_value["x"], "node.representative_point.tile_pixel.x"),
                        _finite(tile_value["y"], "node.representative_point.tile_pixel.y"),
                    )
                return _xy(tile_value, "node.representative_point.tile_pixel")

        coordinates = view.get("coordinates")
        if not isinstance(coordinates, Mapping):
            continue
        tile = coordinates.get("tile_pixel")
        if isinstance(tile, Mapping):
            for name in preferred:
                if name in tile and tile[name] is not None:
                    return _xy(tile[name], f"node.coordinates.tile_pixel.{name}")

    for view in views:
        if view.get("centroid_rc") is not None:
            row, column = _xy(view["centroid_rc"], "node.centroid_rc")
            return column + 0.5, row + 0.5
        if view.get("bbox_xyxy") is not None:
            x0, y0, x1, y1 = _numbers(view["bbox_xyxy"], 4, "node.bbox_xyxy")
            return (x0 + x1) / 2.0, (y0 + y1) / 2.0
        if view.get("bbox_xywh") is not None:
            x0, y0, width, height = _numbers(view["bbox_xywh"], 4, "node.bbox_xywh")
            return x0 + width / 2.0, y0 + height / 2.0
    raise ValueError("Graph node has no usable tile-pixel position")


def _node_area(record: Mapping[str, Any]) -> float:
    for view in _views(record):
        for name in ("area_pixels", "pixel_area", "area"):
            if view.get(name) is not None:
                area = _finite(view[name], f"node.{name}")
                return max(area, 0.0)
    return 1.0


def _node_class(record: Mapping[str, Any]) -> tuple[int | None, str | None]:
    class_id: int | None = None
    class_name: str | None = None
    for view in _views(record):
        for nested_name in ("segment_type", "label"):
            label = view.get(nested_name)
            if not isinstance(label, Mapping):
                continue
            if class_id is None and label.get("id") is not None:
                class_id = _optional_int(label["id"])
            if class_name is None and label.get("name") is not None:
                class_name = str(label["name"])
        if class_id is None:
            for name in ("semantic_class_id", "class_id", "label_id", "type_id"):
                if view.get(name) is not None:
                    class_id = _optional_int(view[name])
                    break
        if class_name is None:
            for name in ("semantic_class", "class_name", "label_name", "segment_type", "type"):
                if view.get(name) is not None and not isinstance(view[name], Mapping):
                    class_name = str(view[name])
                    break
    if class_id is None and class_name is not None:
        folded = class_name.casefold()
        class_id = next((index for index, name in enumerate(STAGE2_CLASS_NAMES) if name.casefold() == folded), None)
    if class_name is None and class_id is not None and 0 <= class_id < len(STAGE2_CLASS_NAMES):
        class_name = STAGE2_CLASS_NAMES[class_id]
    return class_id, class_name


def _node_boundary(record: Mapping[str, Any]) -> bool:
    for view in _views(record):
        for name in ("touches_tile_boundary", "boundary_node", "is_boundary", "boundary_contact"):
            if view.get(name) is not None:
                return _boolean(view[name])
    return False


def _views(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    views = [record]
    for name in ("metadata", "attributes", "properties", "data", "segment"):
        nested = record.get(name)
        if isinstance(nested, Mapping):
            views.append(nested)
    for view in tuple(views):
        metrics = view.get("metrics")
        if isinstance(metrics, Mapping) and metrics not in views:
            views.append(metrics)
    return views


def _edge_parts(value: Any) -> tuple[Any, Any, Mapping[str, Any]]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Mapping)):
        if len(value) < 2:
            raise ValueError("An edge sequence must contain at least two node ids")
        attributes: Mapping[str, Any]
        if len(value) >= 3 and isinstance(value[2], Mapping):
            attributes = value[2]
        elif len(value) >= 3:
            attributes = {"shared_boundary_pixels": value[2]}
        else:
            attributes = {}
        return value[0], value[1], attributes

    record = _as_mapping(value, "edge")
    for start_name, end_name in (
        ("source", "target"),
        ("source_id", "target_id"),
        ("u", "v"),
        ("node_u", "node_v"),
        ("from", "to"),
    ):
        if start_name in record and end_name in record:
            return record[start_name], record[end_name], record
    endpoints = record.get("endpoints")
    if isinstance(endpoints, Sequence) and not isinstance(endpoints, (str, bytes)) and len(endpoints) == 2:
        return endpoints[0], endpoints[1], record
    raise ValueError("Graph edge is missing source/target endpoints")


def _edge_boundary(record: Mapping[str, Any]) -> float:
    views = [record]
    for key in ("metadata", "attributes", "properties", "data"):
        nested = record.get(key)
        if isinstance(nested, Mapping):
            views.append(nested)
    for view in views:
        for name in (
            "shared_boundary_side_count",
            "shared_boundary_pixels",
            "shared_boundary_length_pixels",
            "shared_boundary",
            "boundary_pixels",
            "contact_pixels",
            "weight",
        ):
            if view.get(name) is not None:
                value = _finite(view[name], f"edge.{name}")
                if value < 0:
                    raise ValueError(f"edge.{name} must be non-negative")
                return value
    return 1.0


def _edge_contact(record: Mapping[str, Any]) -> tuple[float, float] | None:
    """Return the measured shared-interface centroid in tile pixels, if present.

    Stage 3 v2 stores the value under ``contact_coordinates.tile_pixel``;
    accepting the flat aliases keeps the visualization helpers useful with
    raw adjacency records and small hand-authored graph fixtures.
    """

    views = [record]
    for key in ("metadata", "attributes", "properties", "data"):
        nested = record.get(key)
        if isinstance(nested, Mapping):
            views.append(nested)

    for view in views:
        for name in ("contact_centroid_tile_pixel_xy", "contact_centroid_xy"):
            if view.get(name) is not None:
                return _xy(view[name], f"edge.{name}")

        coordinates = view.get("contact_coordinates")
        if not isinstance(coordinates, Mapping):
            continue
        tile = coordinates.get("tile_pixel")
        if not isinstance(tile, Mapping):
            continue
        if tile.get("available") is False:
            continue
        if tile.get("centroid_xy") is not None:
            return _xy(tile["centroid_xy"], "edge.contact_coordinates.tile_pixel.centroid_xy")
        if tile.get("bbox_xyxy") is not None:
            x0, y0, x1, y1 = _numbers(
                tile["bbox_xyxy"], 4, "edge.contact_coordinates.tile_pixel.bbox_xyxy"
            )
            return (x0 + x1) / 2.0, (y0 + y1) / 2.0

    for view in views:
        if view.get("contact_bbox_tile_pixel_xyxy") is not None:
            x0, y0, x1, y1 = _numbers(
                view["contact_bbox_tile_pixel_xyxy"], 4, "edge.contact_bbox_tile_pixel_xyxy"
            )
            return (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return None


def _render_overlay_prepared(
    image: np.ndarray,
    nodes: Sequence[_VisualNode],
    edges: Sequence[_VisualEdge],
    style: GraphStyle,
    *,
    show_labels: bool,
    show_legend: bool,
) -> np.ndarray:
    height, width = image.shape[:2]
    figure, axis = _new_figure(height, width, style, "#000000")
    axis.imshow(
        image,
        origin="upper",
        extent=(0, width, height, 0),
        interpolation="nearest",
        alpha=style.background_alpha,
        zorder=0,
    )
    _draw_graph(axis, nodes, edges, style, show_labels=show_labels, show_legend=show_legend)
    return _figure_rgb(figure)


def _render_graph_prepared(
    nodes: Sequence[_VisualNode],
    edges: Sequence[_VisualEdge],
    height: int,
    width: int,
    style: GraphStyle,
    *,
    show_labels: bool,
    show_legend: bool,
) -> np.ndarray:
    figure, axis = _new_figure(height, width, style, style.graph_background)
    _draw_graph(axis, nodes, edges, style, show_labels=show_labels, show_legend=show_legend)
    return _figure_rgb(figure)


def _new_figure(height: int, width: int, style: GraphStyle, facecolor: str) -> tuple[Figure, Any]:
    figure = Figure(
        figsize=(width / style.dpi, height / style.dpi),
        dpi=style.dpi,
        facecolor=facecolor,
        frameon=False,
    )
    axis = figure.add_axes((0, 0, 1, 1), facecolor=facecolor)
    axis.set_xlim(0, width)
    # Reversing the limits preserves Stage 2's upper-left, y-down pixel space.
    axis.set_ylim(height, 0)
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()
    return figure, axis


def _draw_graph(
    axis: Any,
    nodes: Sequence[_VisualNode],
    edges: Sequence[_VisualEdge],
    style: GraphStyle,
    *,
    show_labels: bool,
    show_legend: bool,
) -> None:
    nodes, edges = _select_display_graph(nodes, edges, style)
    positions = {node.token: (node.x, node.y) for node in nodes}
    if edges:
        boundary_values = np.asarray([edge.shared_boundary for edge in edges], dtype=np.float64)
        scale = _unit_scale(boundary_values, logarithmic=True)
        widths = style.edge_width_min + scale * (style.edge_width_max - style.edge_width_min)
        alphas = style.edge_alpha_min + scale * (style.edge_alpha_max - style.edge_alpha_min)
        segments = [_edge_polyline(edge, positions, style) for edge in edges]
        rgba = np.tile(np.asarray(_rgba(style.edge_color), dtype=np.float64), (len(edges), 1))
        rgba[:, 3] = alphas
        collection = LineCollection(
            segments,
            colors=rgba,
            linewidths=widths,
            antialiased=True,
            zorder=1,
        )
        collection.set_capstyle("round")
        axis.add_collection(collection)
        contact_edges = [
            edge
            for edge in edges
            if style.show_contact_anchors and edge.contact_x is not None and edge.contact_y is not None
            and _contact_deviation(edge, positions)
            >= style.contact_anchor_minimum_deviation_pixels
        ]
        if contact_edges:
            axis.scatter(
                [float(edge.contact_x) for edge in contact_edges],
                [float(edge.contact_y) for edge in contact_edges],
                s=style.contact_anchor_size,
                c=style.contact_anchor_color,
                alpha=style.contact_anchor_alpha,
                edgecolors=style.contact_anchor_outline_color,
                linewidths=style.contact_anchor_outline_width,
                zorder=1.5,
            )

    if nodes:
        areas = np.asarray([node.area_pixels for node in nodes], dtype=np.float64)
        node_scale = _unit_scale(areas, logarithmic=True)
        sizes = style.node_size_min + node_scale * (style.node_size_max - style.node_size_min)
        colors = np.asarray([_node_color(node) for node in nodes], dtype=np.float64) / 255.0
        axis.scatter(
            [node.x for node in nodes],
            [node.y for node in nodes],
            s=sizes,
            c=colors,
            alpha=style.node_alpha,
            edgecolors=style.node_outline_color,
            linewidths=style.node_outline_width,
            zorder=2,
        )
        boundary_indices = [index for index, node in enumerate(nodes) if node.touches_boundary]
        if boundary_indices:
            axis.scatter(
                [nodes[index].x for index in boundary_indices],
                [nodes[index].y for index in boundary_indices],
                s=sizes[boundary_indices] + 12.0,
                facecolors="none",
                edgecolors=style.boundary_outline_color,
                linewidths=style.boundary_outline_width,
                zorder=3,
            )
        if show_labels:
            for node in nodes:
                label = axis.text(
                    node.x,
                    node.y,
                    str(node.identifier),
                    color=style.label_color,
                    fontsize=style.label_font_size,
                    ha="center",
                    va="center",
                    clip_on=True,
                    zorder=4,
                )
                label.set_path_effects(
                    [patheffects.withStroke(linewidth=1.7, foreground=style.label_outline_color)]
                )

    if show_legend and nodes:
        present = sorted(
            {node.class_id for node in nodes if node.class_id is not None and 0 <= node.class_id < 9}
        )
        handles = [
            Patch(
                facecolor=STAGE2_CLASS_COLORS[class_id] / 255.0,
                edgecolor=style.node_outline_color,
                label=STAGE2_CLASS_NAMES[class_id],
            )
            for class_id in present
        ]
        if any(node.class_id not in range(9) if node.class_id is not None else True for node in nodes):
            handles.append(Patch(facecolor="#9aa2ad", edgecolor=style.node_outline_color, label="unclassified"))
        legend = axis.legend(
            handles=handles,
            loc="upper right",
            borderaxespad=0.45,
            framealpha=0.82,
            fontsize=max(5.0, style.label_font_size),
            ncols=1,
        )
        legend.set_zorder(5)


def _select_display_graph(
    nodes: Sequence[_VisualNode],
    edges: Sequence[_VisualEdge],
    style: GraphStyle,
) -> tuple[list[_VisualNode], list[_VisualEdge]]:
    """Select a readable view without mutating or redefining the graph.

    The returned lists are visualization-only.  In particular, a hidden node
    or edge remains present in ``graph.json`` and every analytical export.
    Unknown-only suppression is separate from the general area threshold so a
    caller can remove uncertain speckle without hiding small known objects.
    """

    visible_nodes = [
        node
        for node in nodes
        if node.area_pixels >= style.minimum_display_node_area_pixels
        and (
            not _is_unknown_node(node)
            or node.area_pixels >= style.minimum_display_unknown_area_pixels
        )
        and (
            not node.topology_fragment
            or node.area_pixels
            >= style.minimum_display_topology_fragment_area_pixels
        )
    ]
    visible_tokens = {node.token for node in visible_nodes}
    visible_edges = [
        edge
        for edge in edges
        if edge.source in visible_tokens
        and edge.target in visible_tokens
        and edge.shared_boundary >= style.minimum_display_shared_boundary_pixels
    ]
    return visible_nodes, visible_edges


def _node_is_topology_fragment(record: Mapping[str, Any]) -> bool:
    """Recognize a child created by deterministic connectivity splitting."""

    refined = record.get("refinement_metadata")
    if not isinstance(refined, Mapping):
        return False
    lineage = refined.get("lineage")
    lineage_ids = (
        lineage.get("refinement_operation_ids", [])
        if isinstance(lineage, Mapping)
        else []
    )
    operation_ids = [
        str(value)
        for value in [*lineage_ids, *refined.get("refinement_history", [])]
    ]
    provenance = refined.get("provenance")
    refinement_steps = (
        provenance.get("refinement", [])
        if isinstance(provenance, Mapping)
        else []
    )
    return any("::split_components::" in value for value in operation_ids) or any(
        isinstance(step, Mapping) and step.get("operation_type") == "split_components"
        for step in refinement_steps
    )


def _edge_polyline(
    edge: _VisualEdge,
    positions: Mapping[str, tuple[float, float]],
    style: GraphStyle,
) -> list[tuple[float, float]]:
    endpoints = [positions[edge.source], positions[edge.target]]
    if (
        not style.route_edges_through_contacts
        or edge.contact_x is None
        or edge.contact_y is None
    ):
        return endpoints
    return [endpoints[0], (edge.contact_x, edge.contact_y), endpoints[1]]


def _contact_deviation(
    edge: _VisualEdge,
    positions: Mapping[str, tuple[float, float]],
) -> float:
    """Distance from the measured contact to the naive centroid chord."""

    if edge.contact_x is None or edge.contact_y is None:
        return 0.0
    start = np.asarray(positions[edge.source], dtype=np.float64)
    end = np.asarray(positions[edge.target], dtype=np.float64)
    contact = np.asarray((edge.contact_x, edge.contact_y), dtype=np.float64)
    direction = end - start
    squared_length = float(np.dot(direction, direction))
    if squared_length <= 1e-12:
        return float(np.linalg.norm(contact - start))
    fraction = float(np.clip(np.dot(contact - start, direction) / squared_length, 0.0, 1.0))
    nearest = start + fraction * direction
    return float(np.linalg.norm(contact - nearest))


def _is_unknown_node(node: _VisualNode) -> bool:
    if node.class_id == len(STAGE2_CLASS_NAMES) - 1:
        return True
    return node.class_name is not None and node.class_name.casefold() == "unknown"


def _node_color(node: _VisualNode) -> np.ndarray:
    if node.class_id is not None and 0 <= node.class_id < len(STAGE2_CLASS_COLORS):
        return STAGE2_CLASS_COLORS[node.class_id]
    if node.class_name is not None:
        folded = node.class_name.casefold()
        for index, name in enumerate(STAGE2_CLASS_NAMES):
            if name.casefold() == folded:
                return STAGE2_CLASS_COLORS[index]
    return np.asarray((154, 162, 173), dtype=np.uint8)


def _unit_scale(values: np.ndarray, *, logarithmic: bool) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.float64)
    clean = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    if logarithmic:
        clean = np.log1p(clean)
    low, high = float(clean.min()), float(clean.max())
    if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-12):
        return np.full(clean.shape, 0.5, dtype=np.float64)
    return (clean - low) / (high - low)


def _resolve_shape(
    value: Sequence[int] | np.ndarray | Mapping[str, Any] | None,
    nodes: Sequence[_VisualNode],
) -> tuple[int, int]:
    if isinstance(value, np.ndarray):
        if value.ndim < 2:
            raise ValueError("image_shape array must have at least two dimensions")
        height, width = value.shape[:2]
    elif isinstance(value, Mapping):
        height, width = value.get("height"), value.get("width")
    elif value is not None:
        if isinstance(value, (str, bytes)) or len(value) < 2:
            raise ValueError("image_shape must be (height, width)")
        height, width = value[0], value[1]
    elif nodes:
        width = max(1, int(math.ceil(max(node.x for node in nodes) + 0.5)))
        height = max(1, int(math.ceil(max(node.y for node in nodes) + 0.5)))
    else:
        raise ValueError("image_shape is required when rendering a graph with no nodes")
    try:
        height, width = int(height), int(width)
    except (TypeError, ValueError) as exc:
        raise ValueError("image_shape height and width must be integers") from exc
    if height <= 0 or width <= 0:
        raise ValueError("image_shape height and width must be positive")
    return height, width


def _load_rgb(source: str | Path | Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if path.is_dir():
            candidates = (path / "final_overlay.png", path / "final_segmentation.png")
            path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        if not path.is_file():
            raise FileNotFoundError(f"Visualization background not found: {path}")
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    if isinstance(source, Image.Image):
        return np.asarray(source.convert("RGB"), dtype=np.uint8)
    array = np.asarray(source)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise ValueError("Background must be HxW, HxWx1, HxWx3, or HxWx4")
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    array = _to_uint8(array)
    if array.shape[2] == 4:
        alpha = array[..., 3:4].astype(np.float32) / 255.0
        array = np.rint(array[..., :3] * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return np.ascontiguousarray(array[..., :3])


def _to_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array.copy()
    values = np.asarray(array, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=255.0, neginf=0.0)
    if values.size and 0 <= float(values.min()) and float(values.max()) <= 1.0:
        values = values * 255.0
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def _side_by_side(left: np.ndarray, right: np.ndarray, *, titles: bool) -> np.ndarray:
    if left.shape[:2] != right.shape[:2]:
        raise ValueError("Overview panels must have the same height and width")
    if not titles:
        return _labeled_side_by_side(left, right, left_title=None, right_title=None)
    return _labeled_side_by_side(
        left,
        right,
        left_title="Stage 2 segmentation",
        right_title="Stage 3 adjacency graph",
    )


def _labeled_side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_title: str | None,
    right_title: str | None,
) -> np.ndarray:
    if left.shape[:2] != right.shape[:2]:
        raise ValueError("Overview panels must have the same height and width")
    if (left_title is None) != (right_title is None):
        raise ValueError("Both comparison titles must be supplied or omitted")
    gap = 8
    title_height = 28 if left_title is not None else 0
    height, width = left.shape[:2]
    canvas = np.full(
        (height + title_height, width * 2 + gap, 3),
        (17, 24, 32),
        dtype=np.uint8,
    )
    canvas[title_height:, :width] = left
    canvas[title_height:, width + gap :] = right
    if left_title is not None and right_title is not None:
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        _center_text(draw, (0, 0, width, title_height), left_title)
        _center_text(
            draw,
            (width + gap, 0, width * 2 + gap, title_height),
            right_title,
        )
        canvas = np.asarray(image, dtype=np.uint8)
    return canvas


def _center_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], value: str) -> None:
    bounds = draw.textbbox((0, 0), value)
    text_width, text_height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    x = box[0] + (box[2] - box[0] - text_width) / 2.0
    y = box[1] + (box[3] - box[1] - text_height) / 2.0 - bounds[1]
    draw.text((x, y), value, fill=(244, 247, 251))


def _figure_rgb(figure: Figure) -> np.ndarray:
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    result = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    figure.clear()
    return result


def _save_png(path_value: str | Path, image: np.ndarray) -> Path:
    path = Path(path_value).expanduser()
    if path.suffix.lower() != ".png":
        raise ValueError(f"PNG output path must end in .png: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(
            temporary_name,
            format="PNG",
            compress_level=6,
        )
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return path.resolve()


def _save_svg(path_value: str | Path, figure: Figure) -> Path:
    path = Path(path_value).expanduser()
    if path.suffix.lower() != ".svg":
        raise ValueError(f"SVG output path must end in .svg: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        # Matplotlib otherwise seeds SVG element IDs randomly.  A fixed hash
        # salt plus omitted date metadata makes identical graph inputs produce
        # byte-identical SVG output on the same dependency versions.
        with mpl.rc_context({"svg.hashsalt": "satellitegraph-stage3"}):
            canvas = FigureCanvasSVG(figure)
            canvas.print_svg(
                temporary_name,
                metadata={"Date": None, "Creator": "SatelliteGraph Stage 3"},
            )
        os.replace(temporary_name, path)
    finally:
        figure.clear()
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return path.resolve()


def _id_token(value: Any) -> str:
    value = _scalar(value)
    if value is None:
        raise ValueError("Node ids and edge endpoints cannot be null")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Node ids and edge endpoints must be finite")
    return str(value)


def _natural_key(value: str) -> tuple[Any, ...]:
    parts: list[Any] = []
    current = ""
    digit_mode: bool | None = None
    for character in value:
        is_digit = character.isdigit()
        if digit_mode is None or is_digit == digit_mode:
            current += character
        else:
            parts.append(int(current) if digit_mode else current.casefold())
            current = character
        digit_mode = is_digit
    if current:
        parts.append(int(current) if digit_mode else current.casefold())
    # Tag each part so Python never compares int and str directly.
    return tuple((0, part) if isinstance(part, int) else (1, part) for part in parts)


def _xy(value: Any, name: str) -> tuple[float, float]:
    x, y = _numbers(value, 2, name)
    return x, y


def _numbers(value: Any, length: int, name: str) -> list[float]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"{name} must contain exactly {length} numeric values")
    return [_finite(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _finite(value: Any, name: str) -> float:
    try:
        result = float(_scalar(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_int(value: Any) -> int | None:
    try:
        result = int(_scalar(value))
    except (TypeError, ValueError, OverflowError):
        return None
    return result


def _boolean(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0", ""}:
            return False
    return bool(value)


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _rgba(color: Any) -> tuple[float, float, float, float]:
    from matplotlib.colors import to_rgba

    return to_rgba(color)


__all__ = [
    "GraphStyle",
    "STAGE2_CLASS_COLORS",
    "STAGE2_CLASS_NAMES",
    "render_graph_only",
    "render_graph_overlay",
    "render_graph_overview",
    "render_graph_refinement_comparison",
    "save_graph_overlay",
    "save_graph_overview",
    "save_graph_png",
    "save_graph_refinement_comparison",
    "save_graph_svg",
    "save_graph_visualizations",
]
