"""One human-readable cross-year outcome per supported observation pair."""
from __future__ import annotations

from collections import defaultdict

from src.stage4.schema import stable_digest
from src.labels import FUSED_CLASS_NAMES


CLASS_PREFIX = {
    0: "bareland", 1: "rangeland", 2: "developed_space", 3: "road",
    4: "tree_cover", 5: "water", 6: "agriculture", 7: "building",
}
ACTION = {
    "appeared": "gained", "disappeared": "lost",
    "constructed": "constructed", "demolished": "demolished",
    "removed": "removed", "expanded": "grew",
    "contracted": "shrank", "reconfigured": "footprint_changed",
}
# Prefer specific built/water events when one old/new pair has both a loss
# and a gain event. Every source event remains available in change_events.
CLASS_PRIORITY = {7: 8, 3: 7, 5: 6, 4: 5, 6: 4, 2: 3, 1: 2, 0: 1}


def build_semantic_temporal_edges(nodes, matching_edges, change_edges):
    """Create one typed edge per pair, abstaining where change is unverified."""
    by_id = {node["node_id"]: node for node in nodes}
    matches = {(edge["source"], edge["target"]): edge for edge in matching_edges}
    grouped = defaultdict(list)
    for edge in change_edges:
        grouped[(edge["source"], edge["target"])].append(edge)
    result = []
    for pair in sorted(set(matches) | set(grouped)):
        old, new = by_id[pair[0]], by_id[pair[1]]
        source_class = int(old["segment_type"]["id"])
        target_class = int(new["segment_type"]["id"])
        if 8 in (source_class, target_class):
            continue
        candidates = grouped[pair]
        match = matches.get(pair)
        if candidates:
            chosen = max(candidates, key=lambda edge: (
                CLASS_PRIORITY[edge["class_id"]],
                edge["change_support_area_m2"],
                edge["edge_id"],
            ))
            relation = f"{CLASS_PREFIX[chosen['class_id']]}_{ACTION[chosen['relation']]}"
            basis = "mapped_class_change"
            status = "candidate"
        elif match and match["relation"] == "correspondence" and source_class == target_class:
            evidence = match["evidence"]
            if (evidence["iou"] < 0.5 or
                    abs(new["area_m2"] / old["area_m2"] - 1) > 0.10):
                continue
            relation = f"{CLASS_PREFIX[source_class]}_unchanged"
            chosen = None
            basis = "same_class_stable_footprint"
            status = "no_detected_change"
        else:
            continue
        result.append({
            "edge_id": "outcome::" + stable_digest([pair, relation])[:24],
            "source": pair[0], "target": pair[1], "directed": True,
            "delta_years": new["year"] - old["year"],
            "relation": relation, "class_id": chosen["class_id"] if chosen else source_class,
            "class_name": chosen["class_name"] if chosen else FUSED_CLASS_NAMES[source_class],
            "status": status, "basis": basis,
            "change_event_ids": sorted({edge["event_id"] for edge in candidates}),
            "matching_edge_id": match["edge_id"] if match else None,
            "local_change_support_m2": chosen["change_support_area_m2"] if chosen else 0.0,
            "interpretation": "candidate_mapped_outcome_not_verified_physical_cause",
        })
    return result
