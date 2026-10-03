"""Portable Stage 5 artifacts with explicit spatial/temporal edge arrays."""
from collections import Counter

import numpy as np
from shapely.geometry import shape

from src.stage2.artifacts import atomic_json, atomic_npz, checksum, now
from .coordinates import wgs84_geometry
from .schema import SCHEMA_VERSION


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
        "change_events_by_class_and_relation":dict(Counter(f"{e['class_name']}:{e['relation']}" for e in document["change_events"])),
        "change_assessment_states":dict(Counter(a["status"] for a in document["change_assessments"])),
        "registration": document["graph"]["registration"],
        "comparisons": [{"source_year": c["source_year"], "target_year": c["target_year"],
                         "common_coverage_m2": c["common_coverage_m2"],
                         "states": dict(Counter(s["status"] for s in c["node_states"])),
                         "candidate_count": len(c["candidates"])} for c in document["comparisons"]],
        "limitations": ["Scores are uncalibrated, not probabilities.",
                        "Correspondence follows geographic regions, not proven physical object identities.",
                        "Split/merge can reflect segmentation changes.",
                        "Construction/demolition are candidate mapped-footprint events; physical causes are not verified.",
                        "Water changes describe observed dates; seasonality and permanence are undetermined.",
                        "Georeferencing and in-sample RANSAC residuals do not establish independent registration accuracy."]})
    _preview(directory/"temporal_graph.png", document, preview_layers=preview_layers)
    names = ["graph.json", "graph_arrays.npz", "nodes.geojson", "changes.geojson", "report.json", "temporal_graph.png"]
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


def _preview(path, document, *, preview_layers=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    from matplotlib.transforms import Affine2D
    nodes = document["nodes"]
    years = document["graph"]["years"]
    layers = {layer["year"]: layer for layer in (preview_layers or [])}
    prepared = {year: [_prepared_tile(layers[year], tile, document["graph"]["world_crs"])
                       for tile in layers[year].get("tiles", [])] if year in layers else []
                for year in years}
    points = np.asarray([n["representative_point_world_xy"] for n in nodes])
    bounds = [tile[2] for tiles in prepared.values() for tile in tiles]
    x_min = min([float(points[:,0].min()), *(item[0] for item in bounds)])
    x_max = max([float(points[:,0].max()), *(item[2] for item in bounds)])
    y_min = min([float(points[:,1].min()), *(item[1] for item in bounds)])
    y_max = max([float(points[:,1].max()), *(item[3] for item in bounds)])
    width = max(x_max-x_min, 1.0)
    panel_height = max(y_max-y_min, 1.0)
    gap = panel_height * 0.18
    total_height = len(years)*panel_height + (len(years)-1)*gap
    offsets = {year:(len(years)-index-1)*(panel_height+gap)
               for index, year in enumerate(years)}
    positions = {n["node_id"]: [n["representative_point_world_xy"][0]-x_min,
                               n["representative_point_world_xy"][1]-y_min+offsets[n["year"]]]
                 for n in nodes}
    fig_width = 24 if width/panel_height > 3 else 11
    fig_height = max(9, min(20, fig_width*total_height/width))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    for year in years:
        for rgba, transform, _ in prepared[year]:
            display = Affine2D.from_values(transform.a, transform.d, transform.b,
                                           transform.e, transform.c-x_min,
                                           transform.f-y_min+offsets[year])
            ax.imshow(rgba, origin="upper", extent=(0, rgba.shape[1], rgba.shape[0], 0),
                      transform=display+ax.transData, interpolation="nearest", zorder=0)
    if document["spatial_edges"]:
        ax.add_collection(LineCollection(
            [[positions[e["source"]],positions[e["target"]]] for e in document["spatial_edges"]],
            colors="white",linewidths=0.22,alpha=0.12,zorder=1))
    if document["temporal_edges"]:
        ax.add_collection(LineCollection(
            [[positions[e["source"]],positions[e["target"]]] for e in document["temporal_edges"]],
            colors="#00c8ff",linewidths=0.85,alpha=0.72,zorder=3))
    for year in years:
        subset = [n for n in nodes if n["year"] == year]
        xy = np.asarray([positions[n["node_id"]] for n in subset])
        ax.scatter(xy[:,0], xy[:,1], c=[n["segment_type"]["id"] for n in subset],
                   cmap="tab10", vmin=0, vmax=9, s=3 if len(nodes)<40 else 1.6,
                   edgecolors="black",linewidths=0.1,alpha=0.95,zorder=5)
        ax.text(0.01*width, offsets[year]+0.96*panel_height, str(year),
                ha="left",va="top",fontsize=15,fontweight="bold",color="white",
                bbox={"facecolor":"black","alpha":0.7,"edgecolor":"none","pad":4},zorder=7)
    change_styles = [("Gain / construction",{"appeared","constructed"},"#15803d"),
                     ("Loss / removal",{"disappeared","demolished","removed"},"#dc2626"),
                     ("Extent change",{"expanded","contracted","reconfigured"},"#9333ea")]
    legend = [Line2D([0],[0],color="#00c8ff",label="Matched region")]
    from matplotlib.patches import Rectangle
    node_year = {node["node_id"]:node["year"] for node in nodes}
    for label,relations,color in change_styles:
        selected = [e for e in document["change_edges"] if e["relation"] in relations]
        if selected:
            for earlier, later in zip(years[:-1], years[1:]):
                pair = [e for e in selected if node_year[e["source"]] == earlier
                        and node_year[e["target"]] == later]
                if pair:
                    collection = LineCollection(
                        [[positions[e["source"]],positions[e["target"]]] for e in pair],
                        colors=color,linewidths=0.4,alpha=0.24,zorder=2)
                    gap_clip = Rectangle((0,offsets[later]+panel_height),width,gap,
                                         transform=ax.transData)
                    collection.set_clip_path(gap_clip)
                    ax.add_collection(collection)
            legend.append(Line2D([0],[0],color=color,label=label))
    ax.legend(handles=legend,loc="upper right",fontsize=9,framealpha=0.88)
    margin = 0.025*max(width,panel_height)
    ax.set_xlim(-margin,width+margin)
    ax.set_ylim(-margin,total_height+margin)
    ax.set_aspect("equal",adjustable="box")
    ax.set_axis_off()
    ax.set_title("Temporal graph: earlier image above later image; change links in gap",fontsize=16,pad=12)
    fig.tight_layout(pad=0.4)
    fig.savefig(path,dpi=180)
    plt.close(fig)
