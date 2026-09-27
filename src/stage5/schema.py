"""Validation for a mixed-edge, time-expanded region graph."""
import math
from shapely.geometry import shape

LEGACY_SCHEMA_VERSION = "satellitegraph.stage5.spatiotemporal-graph.v1"
SCHEMA_VERSION = "satellitegraph.stage5.spatiotemporal-graph.v2"


def validate_temporal_graph(document):
    if document.get("schema_version") not in {SCHEMA_VERSION, LEGACY_SCHEMA_VERSION}:
        raise ValueError("Unsupported Stage 5 schema")
    graph = document["graph"]
    nodes, spatial, temporal = document["nodes"], document["spatial_edges"], document["temporal_edges"]
    if (graph["node_count"] != len(nodes) or graph["spatial_edge_count"] != len(spatial)
            or graph["temporal_edge_count"] != len(temporal)):
        raise ValueError("Stage 5 graph counts disagree")
    by_id = {node["node_id"]: node for node in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("Duplicate Stage 5 node IDs")
    tracks = set()
    for node in nodes:
        geometry = shape(node["geometry_world"])
        if geometry.is_empty or not geometry.is_valid or not math.isfinite(geometry.area) or geometry.area <= 0:
            raise ValueError("Invalid Stage 5 world geometry")
        track_at_time = (node["year"], node["track_id"])
        if track_at_time in tracks:
            raise ValueError("A track cannot contain two nodes from the same year")
        tracks.add(track_at_time)
    seen = set()
    incoming, outgoing = set(), set()
    for kind, edges in [("spatial", spatial), ("temporal", temporal)]:
        for edge in edges:
            a, b = by_id.get(edge["source"]), by_id.get(edge["target"])
            if a is None or b is None or a is b:
                raise ValueError("Invalid Stage 5 edge endpoints")
            key = (kind, edge["source"], edge["target"])
            if key in seen:
                raise ValueError("Duplicate Stage 5 edge")
            seen.add(key)
            if kind == "spatial" and (a["year"] != b["year"] or edge["directed"]):
                raise ValueError("Spatial edges must be undirected within one year")
            if kind == "temporal":
                if a["year"] >= b["year"] or not edge["directed"]:
                    raise ValueError("Temporal edges must point forward in time")
                if edge["relation"] not in {"correspondence", "split", "merge"}:
                    raise ValueError("Unknown temporal relation")
                if edge["relation"] == "correspondence":
                    out_key, in_key = (a["node_id"], b["year"]), (b["node_id"], a["year"])
                    if out_key in outgoing or in_key in incoming or a["track_id"] != b["track_id"]:
                        raise ValueError("Correspondence must be one-to-one with a shared track")
                    outgoing.add(out_key)
                    incoming.add(in_key)
    if document["schema_version"] == SCHEMA_VERSION:
        _validate_changes(document,by_id)


def _validate_changes(document,by_id):
    from .events import EVENT_RELATIONS
    events,edges = document["change_events"],document["change_edges"]
    graph = document["graph"]
    if (graph["change_event_count"]!=len(events) or graph["change_edge_count"]!=len(edges)
            or graph["all_temporal_edge_count"]!=len(edges)+len(document["temporal_edges"])):
        raise ValueError("Change event/edge counts disagree")
    event_by_id = {e["event_id"]:e for e in events}
    if len(event_by_id)!=len(events):
        raise ValueError("Duplicate change event IDs")
    for event in events:
        if event["relation"] not in EVENT_RELATIONS or event["class_id"] not in range(9):
            raise ValueError("Invalid change event relation/class")
        geometry = shape(event["geometry_world"])
        if geometry.is_empty or not geometry.is_valid or not math.isfinite(geometry.area) or geometry.area<=0:
            raise ValueError("Invalid change event geometry")
        for side in ("source","target"):
            if not event[f"{side}_node_ids"]:
                raise ValueError("Change event needs observed endpoints")
            for node_id in event[f"{side}_node_ids"]:
                if node_id not in by_id or by_id[node_id]["year"]!=event[f"{side}_year"]:
                    raise ValueError("Change event endpoint year mismatch")
    seen,supported = set(),set()
    for edge in edges:
        event = event_by_id.get(edge["event_id"])
        if event is None or edge["relation"]!=event["relation"] or edge["class_id"]!=event["class_id"]:
            raise ValueError("Change edge disagrees with its event")
        key = (edge["event_id"],edge["source"],edge["target"])
        if key in seen:
            raise ValueError("Duplicate change edge")
        seen.add(key)
        a,b = by_id.get(edge["source"]),by_id.get(edge["target"])
        if (a is None or b is None or not edge["directed"] or a["year"]>=b["year"]
                or edge["source"] not in event["source_node_ids"] or edge["target"] not in event["target_node_ids"]):
            raise ValueError("Change edge must connect observed nodes forward in time")
        if edge["delta_years"]!=b["year"]-a["year"]:
            raise ValueError("Change edge elapsed years disagree")
        supported.add(edge["event_id"])
    if supported!=set(event_by_id):
        raise ValueError("Every change event needs a supporting edge")
