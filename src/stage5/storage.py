"""Portable Stage 5 artifacts with explicit spatial/temporal edge arrays."""
from collections import Counter

import numpy as np
from shapely.geometry import shape

from src.stage2.artifacts import atomic_json, atomic_npz, checksum, now
from .coordinates import wgs84_geometry
from .schema import SCHEMA_VERSION


def save_result(directory, document):
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
    _preview(directory/"temporal_graph.png", document)
    names = ["graph.json", "graph_arrays.npz", "nodes.geojson", "changes.geojson", "report.json", "temporal_graph.png"]
    atomic_json(directory/"manifest.json", {"schema_version": SCHEMA_VERSION, "status": "complete",
                "created_at": now(), "artifact_checksums": {name: checksum(directory/name) for name in names}})


def _preview(path, document):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    nodes = document["nodes"]
    years = document["graph"]["years"]
    # Year panels share world northing; eastings are translated ONLY in this view.
    points = np.asarray([n["representative_point_world_xy"] for n in nodes])
    minimum = points.min(axis=0)
    span = max(float(np.ptp(points[:,0])), 1)
    # Separate nearly collinear yearly layers so their temporal links remain
    # visible. This is a display offset only, never a graph coordinate change.
    year_y_offset = span*0.45 if float(np.ptp(points[:,1])) < span*0.05 else 0.0
    positions = {n["node_id"]: [n["representative_point_world_xy"][0]-minimum[0] + years.index(n["year"])*span*1.4,
                               n["representative_point_world_xy"][1]-minimum[1]+years.index(n["year"])*year_y_offset] for n in nodes}
    fig, ax = plt.subplots(figsize=(min(24, 6*len(years)), 7))
    for kind, edges, color, alpha in [("spatial", document["spatial_edges"], "#a0a0a0", 0.3),
                                       ("temporal", document["temporal_edges"], "#2563eb", 0.12)]:
        ax.add_collection(LineCollection([[positions[e["source"]], positions[e["target"]]] for e in edges],
                                         colors=color, linewidths=0.6, alpha=alpha))
    for year in years:
        subset = [n for n in nodes if n["year"] == year]
        xy = np.asarray([positions[n["node_id"]] for n in subset])
        ax.scatter(xy[:,0], xy[:,1], c=[n["segment_type"]["id"] for n in subset],
                   cmap="tab10", vmin=0, vmax=9, s=35 if len(nodes)<40 else 8, zorder=3)
        ax.text(float(xy[:,0].mean()), float(xy[:,1].max())+span*.07, str(year), ha="center")
        if len(nodes)<40:
            from src.labels import FUSED_CLASS_NAMES
            for node,position in zip(subset,xy):
                ax.annotate(FUSED_CLASS_NAMES[int(node["segment_type"]["id"])],position,
                            xytext=(0,-12),textcoords="offset points",ha="center",fontsize=7,rotation=35)
    change_styles = [("Gain / construction",{"appeared","constructed"},"#15803d"),
                     ("Loss / removal",{"disappeared","demolished","removed"},"#dc2626"),
                     ("Extent change",{"expanded","contracted","reconfigured"},"#9333ea"),
                     ("Classification change",{"identified","became_unknown"},"#d97706")]
    from matplotlib.lines import Line2D
    legend = []
    for label,relations,color in change_styles:
        selected = [e for e in document["change_edges"] if e["relation"] in relations]
        if selected:
            ax.add_collection(LineCollection([[positions[e["source"]],positions[e["target"]]] for e in selected],
                                             colors=color,linewidths=1,alpha=0.6))
            legend.append(Line2D([0],[0],color=color,label=label))
    if legend:
        ax.legend(handles=legend,loc="best",fontsize=8)
    ax.autoscale()
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Yearly combined graphs: gray spatial edges; blue temporal links (earlier to later)")
    ax.set_xlabel("World easting with display-only panel offsets (metres)")
    ax.set_ylabel("World northing with display-only year offsets (metres)" if year_y_offset
                  else "World northing relative to display origin (metres)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
