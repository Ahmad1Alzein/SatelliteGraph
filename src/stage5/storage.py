"""Portable Stage 5 artifacts with explicit spatial/temporal edge arrays."""
from collections import Counter, defaultdict
from shutil import copyfile

import numpy as np
from shapely.geometry import shape

from src.stage2.artifacts import atomic_json, atomic_npz, checksum, now
from .coordinates import wgs84_geometry
from .schema import SCHEMA_VERSION


EDGE_TYPE_COLORS = {
    "adjacent": "#cbd5e1", "correspondence": "#00d4ff",
    "split": "#3b82f6", "merge": "#6366f1",
    "appeared": "#22c55e", "constructed": "#84cc16",
    "disappeared": "#ef4444", "demolished": "#be123c", "removed": "#f97316",
    "expanded": "#a855f7", "contracted": "#d946ef", "reconfigured": "#7c3aed",
    "persisted": "#eab308", "identified": "#0ea5e9",
    "became_unknown": "#fb923c", "unknown_persisted": "#64748b",
}
EDGE_TYPE_LABELS = {"adjacent":"Touching regions", "correspondence":"Same mapped region",
                    "split":"Region split", "merge":"Region merge",
                    "appeared":"Class appeared", "disappeared":"Class disappeared",
                    "constructed":"Building/road constructed",
                    "demolished":"Building demolished", "removed":"Road removed",
                    "expanded":"Class area expanded", "contracted":"Class area contracted",
                    "reconfigured":"Class footprint reshaped"}


def _outcome_color(relation):
    """Stable, class-specific palette for the public cross-year edge layer."""
    import colorsys
    from .semantics import CLASS_PREFIX
    hues = {0:0.08,1:0.13,2:0.77,3:0.05,4:0.36,5:0.57,6:0.24,7:0.94}
    class_id = next((cid for cid,prefix in CLASS_PREFIX.items()
                     if relation.startswith(prefix+"_")),None)
    if class_id is None:
        raise ValueError(f"Unknown outcome relation: {relation}")
    action = relation[len(CLASS_PREFIX[class_id])+1:]
    if action == "unchanged":
        saturation,value = 0.35,1.0
    elif action in {"constructed","gained"}:
        saturation,value = 0.95,1.0
    elif action == "grew":
        saturation,value = 0.62,1.0
    elif action in {"demolished","removed","lost"}:
        saturation,value = 0.85,0.72
    elif action == "shrank":
        saturation,value = 0.95,0.50
    else:
        saturation,value = 0.63,0.95
    red,green,blue = colorsys.hsv_to_rgb(hues[class_id],saturation,value)
    return "#{:02x}{:02x}{:02x}".format(round(red*255),round(green*255),round(blue*255))


def _outcome_label(relation):
    return relation.replace("_"," ").title()


def _assign_local_colors(edges, positions, *, radius):
    """Color nearby links differently at either end of the year transition."""
    import colorsys
    from scipy.spatial import cKDTree

    if not edges:
        return []
    def palette_color(index):
        red, green, blue = colorsys.hsv_to_rgb((index * 0.61803398875) % 1,
                                              (0.98,0.78,0.9)[index % 3],
                                              (1.0,0.92)[(index//3) % 2])
        return "#{:02x}{:02x}{:02x}".format(round(red*255),round(green*255),round(blue*255))

    palette = [palette_color(index) for index in range(36)]
    neighbors = [set() for _ in edges]
    for side in ("source", "target"):
        points = np.asarray([positions[edge[side]] for edge in edges], float)
        tree = cKDTree(points)
        for index, nearby in enumerate(tree.query_ball_point(points, radius)):
            neighbors[index].update(j for j in nearby if j != index)
    assignments = {}
    order = sorted(range(len(edges)), key=lambda i: (-len(neighbors[i]), edges[i]["edge_id"]))
    for i in order:
        used = Counter(assignments[j] for j in neighbors[i] if j in assignments)
        available = [color for color in range(len(palette)) if color not in used]
        if available:
            assignments[i] = available[i % len(available)]
        else:
            assignments[i] = len(palette)
            palette.append(palette_color(len(palette)))
    return [palette[assignments[i]] for i in range(len(edges))]


def save_result(directory, document, *, preview_layers=None):
    directory.mkdir(parents=True, exist_ok=True)
    # Publish incomplete first so interrupted overwrites never look successful.
    atomic_json(directory/"manifest.json", {"schema_version": SCHEMA_VERSION, "status": "writing"})
    atomic_json(directory/"graph.json", document)
    nodes = document["nodes"]
    index = {n["node_id"]: i for i, n in enumerate(nodes)}
    spatial_pairs = [(index[e["source"]], index[e["target"]]) for e in document["spatial_edges"]]
    spatial_pairs += [(b, a) for a, b in spatial_pairs.copy()]
    temporal_pairs = [(index[e["source"]], index[e["target"]]) for e in document["temporal_edges"]]
    change_pairs = [(index[e["source"]],index[e["target"]]) for e in document["change_edges"]]
    outcome_edges = document.get("semantic_temporal_edges",[])
    outcome_pairs = [(index[e["source"]],index[e["target"]]) for e in outcome_edges]
    atomic_npz(directory/"graph_arrays.npz",
               node_ids=np.asarray([n["node_id"] for n in nodes]),
               track_ids=np.asarray([n["track_id"] for n in nodes]),
               years=np.asarray([n["year"] for n in nodes], dtype=np.int32),
               class_ids=np.asarray([n["segment_type"]["id"] for n in nodes], dtype=np.int32),
               world_xy=np.asarray([n["centroid_world_xy"] for n in nodes], dtype=np.float64),
               area_m2=np.asarray([n["area_m2"] for n in nodes]),
               spatial_edge_index=np.asarray(spatial_pairs, dtype=np.int64).reshape(-1,2).T,
               temporal_edge_index=np.asarray(temporal_pairs, dtype=np.int64).reshape(-1,2).T,
               temporal_relations=np.asarray([e["relation"] for e in document["temporal_edges"]], dtype="U32"),
               temporal_scores=np.asarray([e["evidence"]["score"] for e in document["temporal_edges"]]),
               change_edge_index=np.asarray(change_pairs,dtype=np.int64).reshape(-1,2).T,
               change_relations=np.asarray([e["relation"] for e in document["change_edges"]],dtype="U32"),
               change_class_ids=np.asarray([e["class_id"] for e in document["change_edges"]],dtype=np.int32),
               change_event_ids=np.asarray([e["event_id"] for e in document["change_edges"]],dtype="U64"),
               change_support_area_m2=np.asarray([e["change_support_area_m2"] for e in document["change_edges"]]),
               semantic_edge_index=np.asarray(outcome_pairs,dtype=np.int64).reshape(-1,2).T,
               semantic_relations=np.asarray([e["relation"] for e in outcome_edges],dtype="U64"),
               world_crs=np.asarray(document["graph"]["world_crs"]))
    atomic_json(directory/"nodes.geojson", {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": wgs84_geometry(shape(n["geometry_world"]), document["graph"]["world_crs"]),
         "properties": {k: n[k] for k in ("node_id", "year", "track_id", "segment_type", "area_m2")}}
        for n in nodes]})
    atomic_json(directory/"changes.geojson",{"type":"FeatureCollection","features":[
        {"type":"Feature","geometry":wgs84_geometry(shape(e["geometry_world"]),document["graph"]["world_crs"]),
         "properties":{k:e[k] for k in ("event_id","relation","class_id","class_name","source_year","target_year",
                                       "status","measurements","cause_verified")}} for e in document["change_events"]]})
    atomic_json(directory/"report.json", {
        "years": document["graph"]["years"], "node_count": len(nodes),
        "spatial_edge_count": len(document["spatial_edges"]),
        "temporal_edge_count": len(document["temporal_edges"]),
        "temporal_relations": dict(Counter(e["relation"] for e in document["temporal_edges"])),
        "change_event_count":len(document["change_events"]),"change_edge_count":len(document["change_edges"]),
        "semantic_temporal_edge_count":len(outcome_edges),
        "semantic_relations":dict(Counter(e["relation"] for e in outcome_edges)),
        "change_events_by_class_and_relation":dict(Counter(f"{e['class_name']}:{e['relation']}" for e in document["change_events"])),
        "change_assessment_states":dict(Counter(a["status"] for a in document["change_assessments"])),
        "registration": document["graph"]["registration"],
        "matching_refinement":document["graph"].get("matching_refinement",[]),
        "change_refinement":document["graph"].get("change_refinement",[]),
        "comparisons": [{"source_year": c["source_year"], "target_year": c["target_year"],
                         "common_coverage_m2": c["common_coverage_m2"],
                         "states": dict(Counter(s["status"] for s in c["node_states"])),
                         "candidate_count": len(c["candidates"])} for c in document["comparisons"]],
        "limitations": ["Scores are uncalibrated, not probabilities.",
                        "Correspondence follows geographic regions, not proven physical object identities.",
                        "Split/merge can reflect segmentation changes.",
                        "Change links are dominant local witnesses; other overlapping pairs may be valid changes.",
                        "Construction/demolition are candidate mapped-footprint events; physical causes are not verified.",
                        "Water changes describe observed dates; seasonality and permanence are undetermined.",
                        "Georeferencing and in-sample RANSAC residuals do not establish independent registration accuracy."]})
    prepared = _prepare_preview_tiles(document, preview_layers)
    _preview(directory/"temporal_graph.png", document, prepared=prepared, mode="overview")
    _preview(directory/"all_edge_types.png", document, prepared=prepared, mode="edge_types")
    _preview(directory/"outcome_graph.png", document, prepared=prepared, mode="matching")
    copyfile(directory/"outcome_graph.png", directory/"matching_graph.png")
    copyfile(directory/"outcome_graph.png", directory/"temporal_links.png")
    _preview(directory/"change_graph.png", document, prepared=prepared, mode="change_all")
    relations = sorted({edge["relation"] for edge in outcome_edges if not edge["relation"].endswith("_unchanged")})
    for relation in relations:
        _preview(directory/f"change_{relation}.png", document, prepared=prepared,
                 mode="relation", relation=relation)
    atomic_json(directory/"preview_index.json", {
        "whole_graph":"temporal_graph.png", "matching":"matching_graph.png",
        "outcomes":"outcome_graph.png",
        "edge_types":"all_edge_types.png",
        "all_changes":"change_graph.png",
        "change_by_relation":{relation:f"change_{relation}.png" for relation in relations}})
    names = ["graph.json", "graph_arrays.npz", "nodes.geojson", "changes.geojson", "report.json",
             "temporal_graph.png", "all_edge_types.png", "outcome_graph.png", "matching_graph.png",
             "temporal_links.png", "change_graph.png",
             "preview_index.json", *(f"change_{relation}.png" for relation in relations)]
    atomic_json(directory/"manifest.json", {"schema_version": SCHEMA_VERSION, "status": "complete",
                "created_at": now(), "artifact_checksums": {name: checksum(directory/name) for name in names}})


def _prepared_tile(layer, tile, target_crs):
    """Read a saved PNG and map its pixels into the graph's metric CRS."""
    from PIL import Image
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.warp import calculate_default_transform, reproject

    with Image.open(tile["path"]) as image:
        rgba = np.asarray(image.convert("RGBA"))
    height, width = rgba.shape[:2]
    if (width, height) != (int(tile["width"]), int(tile["height"])):
        raise ValueError(f"Preview PNG size disagrees with saved tile metadata: {tile['path']}")
    transform = Affine(*layer["source_affine_transform"]) * Affine.translation(
        int(tile["x0"]), int(tile["y0"]))
    source_crs = CRS.from_user_input(layer["crs"])
    target = CRS.from_user_input(target_crs)
    if source_crs != target:
        source_corners = [transform * point for point in
                          ((0,0),(width,0),(width,height),(0,height))]
        source_x, source_y = zip(*source_corners)
        projected, new_width, new_height = calculate_default_transform(
            source_crs, target, width, height,
            min(source_x),min(source_y),max(source_x),max(source_y))
        warped = np.zeros((new_height, new_width, 4), dtype=np.uint8)
        for band in range(4):
            reproject(rgba[:, :, band], warped[:, :, band], src_transform=transform,
                      src_crs=source_crs, dst_transform=projected, dst_crs=target,
                      resampling=Resampling.nearest if band == 3 else Resampling.bilinear)
        rgba, transform = warped, projected
        height, width = new_height, new_width
    if layer.get("registration_affine"):
        transform = Affine(*layer["registration_affine"]) * transform
    corners = [transform * point for point in ((0,0),(width,0),(width,height),(0,height))]
    xs, ys = zip(*corners)
    return rgba, transform, (min(xs), min(ys), max(xs), max(ys))


def _prepare_preview_tiles(document, preview_layers):
    years = document["graph"]["years"]
    layers = {layer["year"]:layer for layer in (preview_layers or [])}
    return {year:[_prepared_tile(layers[year], tile, document["graph"]["world_crs"])
                  for tile in layers[year].get("tiles", [])] if year in layers else []
            for year in years}


def _preview(path, document, *, prepared=None, mode="overview", relation=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.transforms import Affine2D

    nodes = document["nodes"]
    years = document["graph"]["years"]
    prepared = prepared if prepared is not None else _prepare_preview_tiles(document, None)
    points = np.asarray([node["representative_point_world_xy"] for node in nodes])
    bounds = [tile[2] for yearly in prepared.values() for tile in yearly]
    x_min = min([float(points[:,0].min()), *(item[0] for item in bounds)])
    x_max = max([float(points[:,0].max()), *(item[2] for item in bounds)])
    y_min = min([float(points[:,1].min()), *(item[1] for item in bounds)])
    y_max = max([float(points[:,1].max()), *(item[3] for item in bounds)])
    width = max(x_max-x_min, 1.0)
    panel_height = max(y_max-y_min, 1.0)
    gap = panel_height*0.18
    total_height = len(years)*panel_height+(len(years)-1)*gap
    offsets = {year:(len(years)-index-1)*(panel_height+gap)
               for index,year in enumerate(years)}
    positions = {node["node_id"]:(node["representative_point_world_xy"][0]-x_min,
                                  node["representative_point_world_xy"][1]-y_min+offsets[node["year"]])
                 for node in nodes}

    fig_width = 24 if width/panel_height > 3 else 16
    fig_height = max(12, min(34, fig_width*total_height/width))
    if mode == "edge_types":
        fig_height = max(11,min(34,0.78*fig_width*total_height/width+2))
    fig, ax = plt.subplots(figsize=(fig_width,fig_height))
    for year in years:
        for rgba,transform,_ in prepared[year]:
            display = Affine2D.from_values(transform.a,transform.d,transform.b,
                                           transform.e,transform.c-x_min,
                                           transform.f-y_min+offsets[year])
            display_rgba = rgba
            if mode == "edge_types":
                display_rgba = rgba.copy()
                display_rgba[:,:,:3] = (display_rgba[:,:,:3].astype(np.float32)*0.58).astype(np.uint8)
            ax.imshow(display_rgba,origin="upper",extent=(0,rgba.shape[1],rgba.shape[0],0),
                      transform=display+ax.transData,interpolation="nearest",zorder=0)

    def draw_lines(edges, colors, *, linewidth=1.2, alpha=0.85, halo=True, zorder=3):
        if not edges:
            return
        segments = [[positions[edge["source"]],positions[edge["target"]]] for edge in edges]
        if halo:
            ax.add_collection(LineCollection(segments,colors="#08111e",
                                             linewidths=linewidth+1.3,alpha=0.65,zorder=zorder))
        ax.add_collection(LineCollection(segments,colors=colors,
                                         linewidths=linewidth,alpha=alpha,zorder=zorder+1))

    outcome_edges = document.get("semantic_temporal_edges",[])
    change_edges = [edge for edge in outcome_edges if not edge["relation"].endswith("_unchanged")]
    if mode == "edge_types":
        selected = outcome_edges
        colors = [_outcome_color(edge["relation"]) for edge in selected]
        draw_lines(selected,colors,linewidth=1.7,alpha=0.88,halo=False,zorder=3)
        title = "Cross-year outcomes by land-cover class"
    elif mode == "overview":
        selected = outcome_edges
        colors = [_outcome_color(edge["relation"]) for edge in selected]
        draw_lines(selected,colors,linewidth=1.45,alpha=0.78,halo=False,zorder=3)
        title = "Cross-year segment outcomes"
    elif mode == "matching":
        selected = outcome_edges
        colors = _assign_local_colors(selected,positions,
                                      radius=max(8.0,0.06*min(width,panel_height)))
        draw_lines(selected,colors,linewidth=1.9,alpha=0.96)
        title = f"Cross-year segment outcomes: {len(selected)} links"
    elif mode in {"change_all","relation"}:
        selected = change_edges if mode == "change_all" else [
            edge for edge in change_edges if edge["relation"] == relation]
        if mode == "relation":
            colors = _assign_local_colors(selected,positions,
                                          radius=max(8.0,0.06*min(width,panel_height)))
            title = f"{relation.replace('_',' ').title()} edges: {len(selected)} full links"
        else:
            colors = [_outcome_color(edge["relation"]) for edge in selected]
            title = f"Class-specific changes: {len(selected)} full node-to-node links"
        draw_lines(selected,colors,linewidth=1.75 if mode == "relation" else 1.2,
                   alpha=0.94 if mode == "relation" else 0.72,halo=mode == "relation")
    else:
        raise ValueError(f"Unsupported temporal preview mode: {mode}")

    if mode == "overview":
        all_xy = np.asarray(list(positions.values()))
        ax.scatter(all_xy[:,0],all_xy[:,1],s=20,c="#ffffff",edgecolors="#08111e",
                   linewidths=0.5,alpha=0.90,zorder=5)

    # One colored marker per incident edge; shared split/merge endpoints show
    # concentric colors instead of silently inheriting only the last link.
    incident = defaultdict(list)
    for edge,color in zip(selected,colors):
        for side in ("source","target"):
            node_id = edge[side]
            if color not in incident[node_id]:
                incident[node_id].append(color)
    max_degree = max((len(values) for values in incident.values()),default=0)
    marker_size = 58 if len(selected) > 350 else 90
    for ring in range(max_degree):
        group = [(node_id,values) for node_id,values in incident.items() if ring < len(values)]
        if not group:
            continue
        xy = np.asarray([positions[node_id] for node_id,_ in group])
        sizes = [marker_size+24*(len(values)-ring-1) for _,values in group]
        ax.scatter(xy[:,0],xy[:,1],s=sizes,
                   c=[values[ring] for _,values in group],edgecolors="#08111e",
                   linewidths=1.0,alpha=0.98,zorder=6+ring)

    for year in years:
        ax.text(0.01*width,offsets[year]+0.96*panel_height,str(year),
                ha="left",va="top",fontsize=15,fontweight="bold",color="white",
                bbox={"facecolor":"black","alpha":0.75,"edgecolor":"none","pad":4},zorder=20)
    margin = 0.025*max(width,panel_height)
    ax.set_xlim(-margin,width+margin)
    ax.set_ylim(-margin,total_height+margin)
    ax.set_aspect("equal",adjustable="box")
    ax.set_axis_off()
    ax.set_title(title,fontsize=19,pad=15)
    if mode == "edge_types":
        present = sorted({edge["relation"] for edge in outcome_edges})
        handles = [Line2D([0],[0],color=_outcome_color(relation),linewidth=3,
                          label=_outcome_label(relation)) for relation in present]
        fig.legend(handles=handles,loc="center left",bbox_to_anchor=(0.79,0.5),
                   title="Class and outcome",fontsize=11,title_fontsize=13,framealpha=0.97)
    if mode in {"matching","relation"}:
        fig.text(0.5,0.006,"Each line and its endpoints share a color; nearby links use different colors.",
                 ha="center",fontsize=11)
    fig.tight_layout(rect=(0,0,0.78,0.98) if mode == "edge_types" else None,pad=0.4)
    fig.savefig(path,dpi=170 if mode == "relation" else 190)
    plt.close(fig)
