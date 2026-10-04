"""Conservative, geographically local witnesses for mapped life-change events.

Change events are areas. A change edge is only a witness connecting one old
observation with one new observation at that area, not a claim that every
overlapping polygon pair represents a physical object transition.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import math

from shapely.geometry import shape


LIFE_CHANGE_RELATIONS = frozenset({
    "appeared", "disappeared", "constructed", "demolished", "removed",
    "expanded", "contracted", "reconfigured",
})
GAIN_RELATIONS = frozenset({"appeared", "constructed"})
LOSS_RELATIONS = frozenset({"disappeared", "demolished", "removed"})
COMPACT_CLASSES = frozenset({4, 7})  # tree, building


def refine_matching_links(result, first, second, polygons_a, polygons_b, policy):
    """Abstain from split/merge lineage when siblings are distant or change class.

    A very large polygon can overlap several separated objects. Coverage alone
    would call that a merge/split even though there is no local object lineage.
    Correspondence links are left to the one-to-one assignment in the matcher.
    """
    grouped = defaultdict(list)
    for link in result["links"]:
        if link["relation"] in {"split", "merge"}:
            parent = link["i"] if link["relation"] == "split" else link["j"]
            grouped[(link["relation"], parent)].append(link)
    rejected = {}
    for (relation, parent), links in grouped.items():
        if len(links) < 2:
            reason = "incomplete_lineage"
        else:
            sources = {link["i"] for link in links}
            targets = {link["j"] for link in links}
            classes = {int(first["nodes"][i]["segment_type"]["id"]) for i in sources}
            classes.update(int(second["nodes"][j]["segment_type"]["id"]) for j in targets)
            if len(classes) != 1 or 8 in classes:
                reason = "mixed_or_unknown_lineage_class"
            else:
                siblings = ([polygons_b[link["j"]] for link in links] if relation == "split"
                            else [polygons_a[link["i"]] for link in links])
                visited = {0}
                pending = [0]
                while pending:
                    i = pending.pop()
                    for j, sibling in enumerate(siblings):
                        if j not in visited and siblings[i].distance(sibling) <= policy.search_radius_m:
                            visited.add(j)
                            pending.append(j)
                reason = None if len(visited) == len(siblings) else "distant_lineage_siblings"
        if reason:
            for link in links:
                rejected[(link["i"], link["j"])] = reason
    if rejected:
        result["links"] = [link for link in result["links"]
                           if (link["i"], link["j"]) not in rejected]
        for candidate in result["candidates"]:
            reason = rejected.get((candidate["i"], candidate["j"]))
            if reason:
                candidate.update(accepted=False, reason=reason)
        used_a = {link["i"] for link in result["links"]}
        used_b = {link["j"] for link in result["links"]}
        for state in result["node_states"]:
            side, index = state["side"], state["index"]
            fraction = state["visible_fraction"]
            used = used_a if side == "source" else used_b
            state["status"] = ("matched" if index in used else
                               "outside_common_coverage" if fraction <= 1e-9 else
                               "partial_coverage" if fraction < policy.minimum_visible_fraction else
                               "unresolved")
    return {"method":"local_same_class_lineage_refinement_v1",
            "rejected_link_count":len(rejected),
            "rejected_by_reason":dict(Counter(rejected.values())),
            "retained_link_count":len(result["links"])}


def refine_change_events(changes, earlier, later, policy, spatial_edges=()):
    """Keep strong local support and at most one witness per changed class node.

    Unknown/unchanged observations remain in the yearly graph but do not make
    life-change events. For a new class node, competing old nodes are ranked by
    their overlap with the actual changed area; for a disappearing class node,
    competing new nodes are handled symmetrically. This removes the spurious
    many-to-one and one-to-many *change witnesses* produced by broad polygons.
    """
    nodes = {node["node_id"]:node for node in [*earlier,*later]}
    polygons = {node_id:shape(node["geometry_world"]) for node_id,node in nodes.items()}
    road_neighbors = Counter()
    for spatial in spatial_edges:
        source_id, target_id = spatial["source"], spatial["target"]
        if (source_id in nodes and target_id in nodes and
                int(nodes[source_id]["segment_type"]["id"]) == 3 and
                int(nodes[target_id]["segment_type"]["id"]) == 3):
            road_neighbors[source_id] += 1
            road_neighbors[target_id] += 1
    by_event = defaultdict(list)
    for edge in changes["edges"]:
        by_event[edge["event_id"]].append(edge)
    retained_events,retained_edges = [],[]
    removed_events,removed_edges = Counter(),Counter()
    minimum_area = max(1.0,0.25*policy.minimum_area_m2)

    for original in changes["events"]:
        event_id = original["event_id"]
        candidates = by_event[event_id]
        class_id = int(original["class_id"])
        relation = original["relation"]
        if class_id == 8 or relation not in LIFE_CHANGE_RELATIONS:
            removed_events["observability_or_no_life_change"] += 1
            removed_edges["observability_or_no_life_change"] += len(candidates)
            continue
        geometry = shape(original["geometry_world"])
        if geometry.area < policy.minimum_area_m2:
            removed_events["event_area_below_threshold"] += 1
            removed_edges["event_area_below_threshold"] += len(candidates)
            continue

        strongest = {}
        for edge in candidates:
            source_id,target_id = edge["source"],edge["target"]
            source,target = nodes[source_id],nodes[target_id]
            source_class = int(source["segment_type"]["id"])
            target_class = int(target["segment_type"]["id"])
            if (relation in GAIN_RELATIONS and target_class != class_id
                    or relation in LOSS_RELATIONS and source_class != class_id
                    or relation not in GAIN_RELATIONS|LOSS_RELATIONS
                    and source_class != class_id and target_class != class_id):
                removed_edges["wrong_class_endpoint"] += 1
                continue
            old_polygon,new_polygon = polygons[source_id],polygons[target_id]
            support = old_polygon.intersection(new_polygon).intersection(geometry)
            support_area = support.area
            overlap = min(old_polygon.intersection(new_polygon).area,
                          float(edge["overlap_area_m2"]))
            smaller_area = min(old_polygon.area,new_polygon.area)
            if (support_area < minimum_area or overlap <= 0 or smaller_area <= 0
                    or support_area/smaller_area < 0.05
                    or support_area/overlap < 0.08):
                removed_edges["weak_local_change_support"] += 1
                continue
            # A registration sliver has no area after removing the configured
            # boundary tolerance. The edge must contain a spatial core.
            if policy.boundary_tolerance_m > 0 and support.buffer(-policy.boundary_tolerance_m).area <= 0:
                removed_edges["boundary_sliver"] += 1
                continue
            if relation in GAIN_RELATIONS:
                changed_id = target_id
                key = ("target",target_id)
            elif relation in LOSS_RELATIONS:
                changed_id = source_id
                key = ("source",source_id)
            elif target_class == class_id:
                changed_id = target_id
                key = ("target",target_id)
            else:
                changed_id = source_id
                key = ("source",source_id)
            class_polygon = polygons[changed_id]
            if class_id == 3 and relation in LIFE_CHANGE_RELATIONS:
                rectangle = class_polygon.minimum_rotated_rectangle
                vertices = list(rectangle.exterior.coords)
                sides = [math.dist(vertices[index], vertices[index+1]) for index in range(4)]
                elongation = max(sides)/max(min(sides), 1e-9)
                if elongation < 3.0 and not (elongation >= 1.3 and
                                             road_neighbors[changed_id] >= 2):
                    removed_edges["compact_road_without_road_context"] += 1
                    continue
            anchor_distance = class_polygon.representative_point().distance(support.representative_point())
            if (class_id in COMPACT_CLASSES and anchor_distance >
                    max(20.0,2.0*math.sqrt(class_polygon.area))):
                removed_edges["remote_compact_object_support"] += 1
                continue
            rank = (support_area, -anchor_distance, edge["edge_id"])
            previous = strongest.get(key)
            if previous is None or rank > previous[0]:
                if previous is not None:
                    removed_edges["weaker_competing_witness"] += 1
                selected = copy.deepcopy(edge)
                selected["change_support_area_m2"] = float(support_area)
                anchor = list(support.representative_point().coords[0])
                selected["source_anchor_world_xy"] = anchor
                selected["target_anchor_world_xy"] = anchor.copy()
                selected["refinement"] = {
                    "local_support_area_m2":float(support_area),
                    "smaller_node_support_fraction":float(support_area/smaller_area),
                    "overlap_change_fraction":float(support_area/overlap),
                    "class_node_anchor_distance_m":float(anchor_distance),
                    "selection":"dominant_local_witness_per_changed_class_node"}
                strongest[key] = (rank,selected)
            else:
                removed_edges["weaker_competing_witness"] += 1
        if not strongest:
            removed_events["no_plausible_local_witness"] += 1
            continue
        chosen = [entry[1] for key,entry in sorted(strongest.items())]
        event = copy.deepcopy(original)
        event["source_node_ids"] = sorted({edge["source"] for edge in chosen})
        event["target_node_ids"] = sorted({edge["target"] for edge in chosen})
        event["evidence"]["edge_refinement"] = {
            "raw_witness_count":len(candidates), "retained_witness_count":len(chosen),
            "minimum_local_support_area_m2":minimum_area,
            "minimum_smaller_node_support_fraction":0.05,
            "minimum_overlap_change_fraction":0.08,
            "selection":"dominant_local_witness_per_changed_class_node"}
        retained_events.append(event)
        retained_edges.extend(chosen)

    audit = {"method":"local_object_change_witness_refinement_v1",
             "raw_event_count":len(changes["events"]),
             "raw_edge_count":len(changes["edges"]),
             "retained_event_count":len(retained_events),
             "retained_edge_count":len(retained_edges),
             "removed_events_by_reason":dict(removed_events),
             "removed_edges_by_reason":dict(removed_edges),
             "policy":{"minimum_local_support_area_m2":minimum_area,
                       "minimum_smaller_node_support_fraction":0.05,
                       "minimum_overlap_change_fraction":0.08,
                       "boundary_tolerance_m":policy.boundary_tolerance_m,
                       "compact_class_anchor_distance_m":"max(20, 2*sqrt(class_area_m2))",
                       "road_change":"elongation >= 3 or elongation >= 1.3 with >= 2 adjacent road nodes",
                       "maximum_witnesses_per_changed_class_node":1},
             "interpretation":"filtered candidate map changes; physical causes need external validation"}
    return {"events":retained_events,"edges":retained_edges,
            "assessments":changes["assessments"],"audit":audit}
