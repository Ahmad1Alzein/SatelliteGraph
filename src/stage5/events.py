"""Class-specific extent/presence events with observed before/after endpoints.

Event inference uses polygon overlays independently of identity matching. Event
areas belong to the event group, not to each supporting edge (avoid double counts).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from numbers import Real

import numpy as np
from shapely.geometry import MultiPolygon, Polygon, mapping
from shapely.ops import unary_union
from shapely.strtree import STRtree

from src.labels import FUSED_CLASS_NAMES
from src.stage4.schema import stable_digest


CLASS_EVENTS = {i: {"appeared":"appeared", "disappeared":"disappeared"} for i in range(8)}
CLASS_EVENTS[3] = {"appeared":"constructed", "disappeared":"removed"}
CLASS_EVENTS[7] = {"appeared":"constructed", "disappeared":"demolished"}
EVENT_RELATIONS = {"appeared", "disappeared", "constructed", "demolished", "removed",
                   "expanded", "contracted", "reconfigured", "persisted",
                   "identified", "became_unknown", "unknown_persisted"}


@dataclass(frozen=True)
class ChangePolicy:
    minimum_area_m2: float = 4.0
    minimum_relative_change: float = 0.05
    boundary_tolerance_m: float = 1.0
    minimum_observed_fraction: float = 0.95
    minimum_confidence: float = 0.60
    minimum_known_fraction: float = 0.95

    def __post_init__(self):
        for key,value in asdict(self).items():
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,Real) or not math.isfinite(value):
                raise ValueError(f"Invalid change policy {key}")
            if key == "boundary_tolerance_m":
                valid = value >= 0
            elif key == "minimum_area_m2":
                valid = value > 0
            else:
                valid = 0 < value <= 1
            if not valid:
                raise ValueError(f"Invalid change policy {key}")


def _reliable(node, policy):
    value = node.get("mean_semantic_confidence")
    return (int(node["segment_type"]["id"]) != 8 and isinstance(value,Real)
            and not isinstance(value,bool) and math.isfinite(value)
            and policy.minimum_confidence <= value <= 1)


def _groups(class_id, earlier, later, overlaps):
    """Bipartite same-class components, including isolated observations."""
    keys = [(0,i) for i,n in enumerate(earlier) if int(n["segment_type"]["id"])==class_id]
    keys += [(1,j) for j,n in enumerate(later) if int(n["segment_type"]["id"])==class_id]
    adjacency = {key:set() for key in keys}
    for i,j,_ in overlaps:
        a,b = (0,i),(1,j)
        if a in adjacency and b in adjacency:
            adjacency[a].add(b)
            adjacency[b].add(a)
    seen = set()
    for start in keys:
        if start in seen:
            continue
        stack,group = [start],set()
        while stack:
            node = stack.pop()
            if node in group:
                continue
            group.add(node)
            stack.extend(adjacency[node]-group)
        seen |= group
        yield sorted(i for side,i in group if side==0),sorted(i for side,i in group if side==1)


def _polygonal_area(geometry):
    """Discard zero-area contacts from a coverage or reliability overlay."""
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    pending = [geometry]
    polygons = []
    while pending:
        part = pending.pop()
        if isinstance(part, (Polygon, MultiPolygon)) and not part.is_empty:
            polygons.append(part)
        elif hasattr(part, "geoms"):
            pending.extend(part.geoms)
    return unary_union(polygons) if polygons else Polygon()


def _area_intersection(first, second):
    # GEOS cannot overlay an empty GeometryCollection with some mixed
    # collections; empty polygonal results are unambiguously zero area.
    if first.is_empty or second.is_empty:
        return Polygon()
    return _polygonal_area(first.intersection(second))


def _event_id(class_id, relation, source_year, target_year, source_nodes,
              target_nodes, class_source_nodes, class_target_nodes, geometry):
    """Identify the actual change component, not just its shared edge support."""
    return "change-event::" + stable_digest([
        source_year, target_year, class_id, relation,
        sorted(source_nodes), sorted(target_nodes),
        sorted(class_source_nodes), sorted(class_target_nodes),
        geometry.wkb_hex])[:24]


def detect_change_events(earlier, later, polygons_a, polygons_b, policy, *, show_progress=False):
    from tqdm import tqdm
    footprint_a,footprint_b = unary_union(polygons_a),unary_union(polygons_b)
    common = _polygonal_area(footprint_a.intersection(footprint_b))
    known_a = unary_union([p for n,p in zip(earlier,polygons_a) if _reliable(n,policy)])
    known_b = unary_union([p for n,p in zip(later,polygons_b) if _reliable(n,policy)])
    reliable_common = _polygonal_area(known_a.intersection(known_b))
    tree = STRtree(polygons_b)
    overlaps = []
    for i,a in enumerate(tqdm(polygons_a, desc="Find changed nodes", unit="node",
                              mininterval=1.0, disable=not show_progress)):
        for j in sorted(int(j) for j in tree.query(a)):
            intersection = a.intersection(polygons_b[j])
            if intersection.area > 1e-9:
                overlaps.append((i,j,intersection))
    events,edges,assessments = [],[],[]

    def emit(class_id,relation,aa,bb,geometry,measurements,evidence,status="candidate"):
        aset,bset = set(aa),set(bb)
        support = []
        for i,j,intersection in overlaps:
            if i not in aset and j not in bset:
                continue
            if class_id!=8 and (not _reliable(earlier[i],policy) or not _reliable(later[j],policy)):
                continue
            # Continue to show change on same-class links as well as cross-class
            # source/replacement links. Each edge references one shared event.
            same_group = i in aset and j in bset
            changed_area = intersection.intersection(geometry).area
            if same_group or changed_area > 1e-9:
                support.append((i,j,intersection.area,changed_area))
        if not support:
            return
        event_id = _event_id(class_id, relation, earlier[0]["year"], later[0]["year"],
            (earlier[i]["node_id"] for i,_,_,_ in support),
            (later[j]["node_id"] for _,j,_,_ in support),
            (earlier[i]["node_id"] for i in aa),
            (later[j]["node_id"] for j in bb), geometry)
        event = {"event_id":event_id,"relation":relation,"class_id":class_id,
            "class_name":FUSED_CLASS_NAMES[class_id],"source_year":earlier[0]["year"],
            "target_year":later[0]["year"],"status":status,
            "source_node_ids":sorted({earlier[i]["node_id"] for i,_,_,_ in support}),
            "target_node_ids":sorted({later[j]["node_id"] for _,j,_,_ in support}),
            "class_source_node_ids":[earlier[i]["node_id"] for i in aa],
            "class_target_node_ids":[later[j]["node_id"] for j in bb],
            "geometry_world":json.loads(json.dumps(mapping(geometry))),"measurements":measurements,"evidence":evidence,
            "interpretation":"classification_observability" if class_id==8 else "inferred_mapped_class_extent_change",
            "cause_verified":False}
        events.append(event)
        for i,j,overlap,changed in support:
            source,target = earlier[i]["node_id"],later[j]["node_id"]
            edges.append({"edge_id":"change-edge::"+stable_digest([event_id,source,target])[:24],
                "source":source,"target":target,"directed":True,"relation":relation,
                "class_id":class_id,"class_name":FUSED_CLASS_NAMES[class_id],"event_id":event_id,
                "delta_years":later[j]["year"]-earlier[i]["year"],"status":status,
                "overlap_area_m2":overlap,"change_support_area_m2":changed,
                "semantic_from":int(earlier[i]["segment_type"]["id"]),
                "semantic_to":int(later[j]["segment_type"]["id"])})

    groups = [(class_id, aa, bb) for class_id in range(8)
              for aa, bb in _groups(class_id, earlier, later, overlaps)]
    for class_id, aa, bb in tqdm(groups, desc="Assess change groups", unit="group",
                                 mininterval=1.0, disable=not show_progress):
        before = unary_union([polygons_a[i] for i in aa])
        after = unary_union([polygons_b[j] for j in bb])
        observed = min([p.intersection(common).area/p.area for p in [before,after] if p.area>0])
        assessment = {"class_id":class_id,"source_year":earlier[0]["year"],"target_year":later[0]["year"],
            "source_node_ids":[earlier[i]["node_id"] for i in aa],
            "target_node_ids":[later[j]["node_id"] for j in bb],"observed_fraction":observed}
        assessments.append(assessment)
        if observed < policy.minimum_observed_fraction:
            assessment["status"] = "insufficient_coverage"
            continue
        # All measurements refer to common ground; never interpret tile clips
        # as physical shrinkage. Full areas retained separately for auditing.
        a,b = before.intersection(common),after.intersection(common)
        gain,loss = b.difference(a),a.difference(b)
        change = gain.union(loss)
        support_geometry = change if change.area > 1e-9 else a.union(b)
        reliable_fraction = support_geometry.intersection(reliable_common).area/support_geometry.area
        assessment["reliable_known_fraction"] = reliable_fraction
        if (reliable_fraction < policy.minimum_known_fraction
                or any(not _reliable(earlier[i],policy) for i in aa)
                or any(not _reliable(later[j],policy) for j in bb)):
            assessment["status"] = "insufficient_semantic_evidence"
            continue
        tolerance = policy.boundary_tolerance_m
        robust_gain = b.difference(a.buffer(tolerance)) if not a.is_empty else b.buffer(-tolerance)
        robust_loss = a.difference(b.buffer(tolerance)) if not b.is_empty else a.buffer(-tolerance)
        robust_gain = _area_intersection(robust_gain,reliable_common)
        robust_loss = _area_intersection(robust_loss,reliable_common)
        assessed_gain,assessed_loss = (_area_intersection(gain,reliable_common),
                                       _area_intersection(loss,reliable_common))
        base = a.area if a.area else b.area
        threshold = max(policy.minimum_area_m2,policy.minimum_relative_change*base)
        gain_flag,loss_flag = robust_gain.area >= threshold,robust_loss.area >= threshold
        net = robust_gain.area-robust_loss.area
        if not aa:
            relation = CLASS_EVENTS[class_id]["appeared"] if gain_flag else None
        elif not bb:
            relation = CLASS_EVENTS[class_id]["disappeared"] if loss_flag else None
        elif gain_flag and loss_flag:
            relation = "expanded" if net>=threshold else "contracted" if net<=-threshold else "reconfigured"
        elif gain_flag:
            relation = "expanded"
        elif loss_flag:
            relation = "contracted"
        else:
            relation = "persisted"
        if relation is None:
            assessment["status"] = "below_change_threshold"
            continue
        assessment["status"] = "assessed"
        assessment["relation"] = relation
        measurements = {"area_before_m2":a.area,"area_after_m2":b.area,
            "full_area_before_m2":before.area,"full_area_after_m2":after.area,
            "area_delta_m2":b.area-a.area,
            "relative_area_change":(b.area-a.area)/a.area if a.area else None,
            "gain_area_m2":assessed_gain.area,"loss_area_m2":assessed_loss.area,
            "raw_gain_area_m2":gain.area,"raw_loss_area_m2":loss.area,
            "unassessed_change_area_m2":change.difference(reliable_common).area,
            "supported_gain_area_m2":robust_gain.area,"supported_loss_area_m2":robust_loss.area,
            "gain_detected":gain_flag,"loss_detected":loss_flag,
            "decision_threshold_m2":threshold}
        evidence = {"observed_fraction":observed,"reliable_known_fraction":reliable_fraction,
                    "boundary_tolerance_m":tolerance,"group_source_count":len(aa),"group_target_count":len(bb),
                    "registration_accuracy":"not_independently_established_by_event_detection",
                    "stability_meaning":"no_extent_change_above_configured_tolerance"}
        event_geometry = (change if relation!="persisted" else a.union(b)).intersection(reliable_common)
        emit(class_id,relation,aa,bb,event_geometry,measurements,evidence)

    # Unknown is an observation state, not a physical land-cover class. These
    # transitions never establish demolition/construction or actual disappearance.
    for i,j,intersection in overlaps:
        ca,cb = int(earlier[i]["segment_type"]["id"]),int(later[j]["segment_type"]["id"])
        if 8 not in (ca,cb) or intersection.area < policy.minimum_area_m2:
            continue
        if ca!=8 and not _reliable(earlier[i],policy) or cb!=8 and not _reliable(later[j],policy):
            continue
        relation = "unknown_persisted" if ca==cb else "identified" if ca==8 else "became_unknown"
        # Limit support to exactly this overlap, even if one unknown polygon spans
        # several different known regions.
        emit(8,relation,[i],[j],intersection,
             {"observed_transition_area_m2":intersection.area},
             {"semantic_from":ca,"semantic_to":cb,"physical_change_determined":False},
             status="observability_change")
    return {"events":events,"edges":edges,"assessments":assessments}
