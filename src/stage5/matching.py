"""Geographic region correspondence with abstention and segmentation lineage."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from shapely.affinity import translate
from shapely.ops import unary_union
from shapely.strtree import STRtree
from tqdm import tqdm


@dataclass(frozen=True)
class MatchPolicy:
    search_radius_m: float = 5.0
    minimum_iou: float = 0.2
    minimum_score: float = 0.6
    ambiguity_margin: float = 0.05
    minimum_visible_fraction: float = 0.9
    lineage_child_coverage: float = 0.8
    lineage_parent_coverage: float = 0.8
    lineage_minimum_share: float = 0.1
    max_assignment_cells: int = 4_000_000

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"Invalid matching policy {key}")
            if key == "search_radius_m":
                valid = value > 0
            elif key == "max_assignment_cells":
                valid = isinstance(value, int) and value > 0
            else:
                valid = 0 < value <= 1
            if not valid:
                raise ValueError(f"Invalid matching policy {key}")


def _neighborhoods(document):
    types = {node["node_id"]: int(node["segment_type"]["id"]) for node in document["nodes"]}
    result = {node_id: Counter() for node_id in types}
    for edge in document["edges"]:
        a, b = edge["source"], edge["target"]
        result[a][types[b]] += 1
        result[b][types[a]] += 1
    return [result[node["node_id"]] for node in document["nodes"]]


def _context_similarity(a, b):
    if not a or not b:
        return 0.5  # missing context is neutral, not evidence of identity
    keys = set(a) | set(b)
    cosine = sum(a[k]*b[k] for k in keys) / math.sqrt(
        sum(value*value for value in a.values()) * sum(value*value for value in b.values()))
    distribution = 1 - 0.5 * sum(abs(a[k]/sum(a.values()) - b[k]/sum(b.values())) for k in keys)
    return 0.5 * (cosine + distribution)


def _components(pairs):
    adjacency = defaultdict(set)
    for i, j in pairs:
        a, b = (0, i), (1, j)
        adjacency[a].add(b)
        adjacency[b].add(a)
    seen = set()
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack, group = [start], set()
        while stack:
            item = stack.pop()
            if item in group:
                continue
            group.add(item)
            stack.extend(adjacency[item] - group)
        seen |= group
        yield sorted(i for side, i in group if side == 0), sorted(i for side, i in group if side == 1)


def _rescue_with_local_coordinate_consensus(first, second, visible_a, visible_b,
                                            candidates, links, used_a, used_b):
    """Recover conservative same-class matches using residual shifts of stable buildings.

    The global image registration is already applied. This pass estimates only
    a small local *residual* from existing high-overlap correspondences and does
    not move stored node geometries or infer physical change from the offset.
    """
    anchors, shifts = [], []
    for link in links:
        if link["relation"] != "correspondence":
            continue
        i, j = link["i"], link["j"]
        evidence = link["evidence"]
        if (int(first["nodes"][i]["segment_type"]["id"]) != 7 or
                int(second["nodes"][j]["segment_type"]["id"]) != 7 or
                evidence["iou"] < 0.55 or evidence["score"] < 0.70 or
                evidence["area_similarity"] < 0.60):
            continue
        a, b = visible_a[i].centroid, visible_b[j].centroid
        displacement = np.asarray([b.x-a.x, b.y-a.y], dtype=float)
        if np.linalg.norm(displacement) <= 10.0:
            anchors.append([a.x, a.y])
            shifts.append(displacement)
    audit = {"method":"local_coordinate_consensus_v1", "stable_building_anchor_count":len(anchors),
             "eligible_candidate_count":0, "recovered_link_count":0}
    if len(anchors) < 4:
        return audit
    tree = cKDTree(np.asarray(anchors))
    shifts = np.asarray(shifts)
    eligible = []
    for candidate in candidates:
        i, j = candidate["i"], candidate["j"]
        if i in used_a or j in used_b or candidate["iou"] < 0.15:
            continue
        class_a = int(first["nodes"][i]["segment_type"]["id"])
        class_b = int(second["nodes"][j]["segment_type"]["id"])
        if class_a != class_b or class_a == 8:
            continue
        if candidate["area_similarity"] < (0.35 if class_a == 7 else 0.45):
            continue
        a, b = visible_a[i], visible_b[j]
        nearby = tree.query_ball_point([a.centroid.x, a.centroid.y], r=125.0)
        if len(nearby) < 3:
            continue
        local_shifts = shifts[nearby]
        predicted = np.median(local_shifts, axis=0)
        if (np.linalg.norm(predicted) > 8.0 or
                np.percentile(np.linalg.norm(local_shifts-predicted, axis=1), 75) > 5.0):
            continue
        shifted_b = translate(b, xoff=-predicted[0], yoff=-predicted[1])
        overlap = a.intersection(shifted_b).area
        adjusted_iou = overlap/(a.area+shifted_b.area-overlap)
        if adjusted_iou < 0.35:
            continue
        residual = np.linalg.norm(np.asarray([b.centroid.x-a.centroid.x,
                                              b.centroid.y-a.centroid.y])-predicted)
        tolerance = max(4.0,0.35*math.sqrt(min(a.area,b.area)))
        if residual > tolerance:
            continue
        rescue_score = (0.65*adjusted_iou + 0.20*candidate["area_similarity"] +
                        0.15*math.exp(-residual/tolerance))
        if rescue_score < 0.48:
            continue
        candidate.update(coordinate_rescue_score=float(rescue_score),
                         adjusted_iou=float(adjusted_iou),
                         predicted_shift_xy_m=predicted.tolist(),
                         centroid_residual_m=float(residual),
                         local_anchor_count=len(nearby))
        eligible.append(candidate)
    audit["eligible_candidate_count"] = len(eligible)
    by_source, by_target = defaultdict(list), defaultdict(list)
    for candidate in eligible:
        by_source[candidate["i"]].append(candidate)
        by_target[candidate["j"]].append(candidate)
    for candidate in sorted(eligible,key=lambda c:(-c["coordinate_rescue_score"],c["i"],c["j"])):
        i, j = candidate["i"], candidate["j"]
        if i in used_a or j in used_b:
            continue
        score = candidate["coordinate_rescue_score"]
        alternatives = [other["coordinate_rescue_score"] for other in
                        by_source[i]+by_target[j] if other is not candidate]
        if score-max(alternatives,default=0) < 0.05:
            candidate["reason"] = "coordinate_rescue_ambiguous"
            continue
        candidate.update(accepted=True,reason="coordinate_rescue",
                         rescue_method="local_coordinate_consensus")
        links.append({"i":i,"j":j,"relation":"correspondence","evidence":candidate.copy()})
        used_a.add(i)
        used_b.add(j)
        audit["recovered_link_count"] += 1
    return audit


def match_snapshots(first, second, polygons_a, polygons_b, policy, *, show_progress=False):
    """Return indexed links, all nearby candidates, coverage, and observation states."""
    footprint_a, footprint_b = unary_union(polygons_a), unary_union(polygons_b)
    common = footprint_a.intersection(footprint_b)
    visible_a = [p.intersection(common) for p in polygons_a]
    visible_b = [p.intersection(common) for p in polygons_b]
    fractions_a = [v.area/p.area for v, p in zip(visible_a, polygons_a)]
    fractions_b = [v.area/p.area for v, p in zip(visible_b, polygons_b)]
    context_a, context_b = _neighborhoods(first), _neighborhoods(second)
    candidates = []
    tree = STRtree(visible_b)
    for i, a in enumerate(tqdm(visible_a, desc="Match source nodes", unit="node",
                               mininterval=1.0, disable=not show_progress)):
        if a.area <= 1e-9:
            continue
        for j in sorted(int(j) for j in tree.query(a.buffer(policy.search_radius_m))):
            b = visible_b[j]
            if b.area <= 1e-9:
                continue
            intersection = a.intersection(b).area
            iou = intersection / (a.area + b.area - intersection)
            distance = a.centroid.distance(b.centroid)
            area_similarity = min(a.area, b.area)/max(a.area, b.area)
            ca, cb = 4*math.pi*a.area/a.length**2, 4*math.pi*b.area/b.length**2
            shape_similarity = min(ca, cb)/max(ca, cb)
            context = _context_similarity(context_a[i], context_b[j])
            proximity = math.exp(-distance / max(policy.search_radius_m, math.sqrt(min(a.area, b.area))))
            # Class agreement is deliberately NOT an identity gate: land cover changes.
            score = 0.55*iou + 0.15*proximity + 0.10*area_similarity + 0.10*shape_similarity + 0.10*context
            candidates.append({"i": i, "j": j, "iou": iou,
                "intersection_m2": intersection, "source_overlap_fraction": intersection/a.area,
                "target_overlap_fraction": intersection/b.area, "centroid_distance_m": distance,
                "area_similarity": area_similarity, "shape_similarity": shape_similarity,
                "neighborhood_similarity": context, "proximity_similarity": proximity,
                "score": score, "accepted": False, "reason": "below_threshold"})
    lookup = {(c["i"], c["j"]): c for c in candidates}
    links, used_a, used_b = [], set(), set()
    # Strong overlap components identify clear one-to-many / many-to-one lineage
    # BEFORE assignment, preventing one child from incorrectly inheriting identity.
    strong = [(c["i"], c["j"]) for c in candidates
              if min(c["source_overlap_fraction"], c["target_overlap_fraction"]) >= policy.lineage_minimum_share
              and max(c["source_overlap_fraction"], c["target_overlap_fraction"]) >= policy.lineage_child_coverage]
    strong_set = set(strong)
    for aa, bb in _components(strong):
        relation = "split" if len(aa) == 1 and len(bb) > 1 else "merge" if len(aa) > 1 and len(bb) == 1 else None
        if relation is None:
            continue
        pairs = [(i, j) for i in aa for j in bb if (i, j) in strong_set]
        if relation == "split":
            coverage = sum(lookup[p]["source_overlap_fraction"] for p in pairs)
            children_ok = all(lookup[p]["target_overlap_fraction"] >= policy.lineage_child_coverage for p in pairs)
        else:
            coverage = sum(lookup[p]["target_overlap_fraction"] for p in pairs)
            children_ok = all(lookup[p]["source_overlap_fraction"] >= policy.lineage_child_coverage for p in pairs)
        if coverage < policy.lineage_parent_coverage or not children_ok:
            continue
        for i, j in pairs:
            candidate = lookup[i, j]
            candidate.update(accepted=True, reason=relation)
            links.append({"i": i, "j": j, "relation": relation, "evidence": candidate.copy()})
        used_a.update(aa)
        used_b.update(bb)

    plausible = [c for c in candidates if c["i"] not in used_a and c["j"] not in used_b
                 and c["iou"] >= policy.minimum_iou]
    eligible = [c for c in plausible if c["score"] >= policy.minimum_score]
    rows, columns = defaultdict(list), defaultdict(list)
    for c in plausible:
        rows[c["i"]].append(c)
        columns[c["j"]].append(c)
    unambiguous = []
    for c in eligible:
        alternatives = [other["score"] for other in rows[c["i"]] + columns[c["j"]] if other is not c]
        margin = c["score"] - max(alternatives, default=0)
        c["alternative_margin"] = margin
        c["reason"] = "ambiguous" if margin < policy.ambiguity_margin else "not_assigned"
        if margin >= policy.ambiguity_margin:
            unambiguous.append((c["i"], c["j"]))
    # Solve independent sparse candidate components with dummy unmatched columns.
    unambiguous_set = set(unambiguous)
    for aa, bb in _components(unambiguous):
        if len(aa) * (len(bb) + len(aa)) > policy.max_assignment_cells:
            raise ValueError("Assignment component exceeds memory budget; tighten candidate gates")
        costs = np.full((len(aa), len(bb) + len(aa)), 1e6)
        costs[:, len(bb):] = 0
        b_index = {j: index for index, j in enumerate(bb)}
        for row, i in enumerate(aa):
            for c in rows[i]:
                if (i, c["j"]) in unambiguous_set and c["j"] in b_index:
                    costs[row, b_index[c["j"]]] = policy.minimum_score - c["score"] - 1e-12
        row_ids, col_ids = linear_sum_assignment(costs)
        for row, col in zip(row_ids, col_ids):
            if col >= len(bb) or costs[row, col] >= 0:
                continue
            i, j = aa[row], bb[col]
            c = lookup[i, j]
            c.update(accepted=True, reason="correspondence")
            links.append({"i": i, "j": j, "relation": "correspondence", "evidence": c.copy()})
            used_a.add(i)
            used_b.add(j)
    rescue_audit = _rescue_with_local_coordinate_consensus(
        first, second, visible_a, visible_b, candidates, links, used_a, used_b)
    states = []
    for side, fractions, used in [("source", fractions_a, used_a), ("target", fractions_b, used_b)]:
        for index, fraction in enumerate(fractions):
            status = ("matched" if index in used else "outside_common_coverage" if fraction <= 1e-9
                      else "partial_coverage" if fraction < policy.minimum_visible_fraction else "unresolved")
            states.append({"side": side, "index": index, "visible_fraction": fraction, "status": status})
    return {"links": sorted(links, key=lambda link: (link["i"], link["j"])),
            "candidates": candidates, "node_states": states,
            "common_coverage_m2": common.area,
            "fractions_a": fractions_a, "fractions_b": fractions_b,
            "coordinate_rescue": rescue_audit}
