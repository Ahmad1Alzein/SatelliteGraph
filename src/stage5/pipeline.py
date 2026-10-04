"""One Stage 4 combined graph per year -> an auditable spatiotemporal graph."""
from __future__ import annotations

import copy
from dataclasses import asdict
import json
from pathlib import Path
import re

import numpy as np
from shapely.geometry import mapping, Point, MultiPoint
from shapely.ops import unary_union

from src.stage2.artifacts import checksum
from src.stage4 import load_stage4_result
from src.stage4.schema import stable_digest, validate_combined_graph
from .coordinates import affine_geometry, metric_crs, project_snapshot, project_point
from .matching import MatchPolicy, match_snapshots
from .events import ChangePolicy, detect_change_events
from .refinement import refine_change_events, refine_matching_links
from .semantics import build_semantic_temporal_edges
from .registration import fit_registration, image_control_points
from .schema import SCHEMA_VERSION, validate_temporal_graph


def _identity(prefix, values):
    return f"{prefix}::{stable_digest(values)[:24]}"


def build_temporal_graph(snapshots, *, target_crs=None, policy=None, change_policy=None,
                         output_directory=None, show_progress=False):
    """Build from >=2 {year, document|graph_directory} snapshots.

    Optional per-snapshot fields: georeference (verified CRS/affine override),
    registration (moving_xy/reference_xy in target CRS), image_path and
    register_to_reference (SIFT registration to the earliest year's image).
    The document API trusts supplied documents after schema validation; the
    directory API additionally verifies upstream artifact checksums.
    """
    if len(snapshots) < 2:
        raise ValueError("Stage 5 requires at least two yearly combined graphs")
    years = [s.get("year") for s in snapshots]
    if any(isinstance(y, bool) or not isinstance(y, int) or not 1 <= y <= 9999 for y in years):
        raise ValueError("Snapshot years must be integer years in 1..9999")
    if len(set(years)) != len(years):
        raise ValueError("Snapshot years must be unique")
    ordered = sorted(snapshots, key=lambda s: s["year"])
    policy = policy if isinstance(policy, MatchPolicy) else MatchPolicy(**(policy or {}))
    change_policy = change_policy if isinstance(change_policy, ChangePolicy) else ChangePolicy(**(change_policy or {}))
    documents, provenance = [], []
    for snapshot in ordered:
        if ("document" in snapshot) == ("graph_directory" in snapshot):
            raise ValueError("Each snapshot requires exactly one document or graph_directory")
        inputs = []
        if "document" in snapshot:
            document = copy.deepcopy(snapshot["document"])
        else:
            directory = Path(snapshot["graph_directory"]).resolve()
            loaded = load_stage4_result(directory)
            document = {"schema_version": "satellitegraph.stage4.combined-graph.v1",
                        "graph": loaded.graph, "nodes": loaded.nodes, "edges": loaded.edges}
            public = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
            if public.get("stale"):
                raise ValueError(f"Stale Stage 4 input: {directory}")
            inputs = [{"path": str(path), "sha256": checksum(path)}
                      for path in (directory/"graph.json", directory/"manifest.json")]
        extra_paths = [*snapshot.get("input_paths", []),
                       *(tile["path"] for tile in snapshot.get("preview_tiles", []))]
        known_paths = {item["path"] for item in inputs}
        for value in extra_paths:
            path = Path(value).resolve()
            if str(path) not in known_paths:
                inputs.append({"path":str(path), "sha256":checksum(path)})
                known_paths.add(str(path))
        validate_combined_graph(document)
        if not document["nodes"]:
            raise ValueError("Yearly combined graph must contain observed nodes")
        documents.append(document)
        provenance.append({"year": snapshot["year"], "graph_id": document["graph"]["id"],
                           "source_image_id": document["graph"]["source_image_id"],
                           "tile_ids": document["graph"]["selected_tile_ids"],
                           "document_sha256": stable_digest(document), "inputs": inputs,
                           "georeference_override": copy.deepcopy(snapshot.get("georeference")),
                           "year_source": "user_supplied"})
    first_metadata = {**documents[0]["graph"], **ordered[0].get("georeference", {})}
    if target_crs is None:
        target_crs = first_metadata.get("crs")
        if target_crs is None:
            raise ValueError("Missing CRS: a verified world georeference is required")
    target = metric_crs(target_crs)
    all_polygons, registrations = [], []
    for index, (snapshot, document) in enumerate(zip(ordered, documents)):
        polygons = project_snapshot(document, target, georeference=snapshot.get("georeference"))
        registration = snapshot.get("registration")
        if registration and snapshot.get("register_to_reference"):
            raise ValueError("Choose control points or image registration, not both")
        if index == 0 and (registration or snapshot.get("register_to_reference")):
            raise ValueError("The first year defines the reference; register later years to it")
        audit = {"method": "georeference_only", "accuracy_status": "not_independently_verified"}
        if snapshot.get("register_to_reference"):
            if not snapshot.get("image_path") or not ordered[0].get("image_path"):
                raise ValueError("Image registration needs image_path for moving and reference years")
            moving, reference = image_control_points(snapshot["image_path"], ordered[0]["image_path"], target)
            registration = {"moving_xy": moving, "reference_xy": reference}
            for path in (snapshot["image_path"], ordered[0]["image_path"]):
                resolved = Path(path).resolve()
                provenance[index]["inputs"].append({"path": str(resolved), "sha256": checksum(resolved)})
        if registration:
            audit = fit_registration(**registration)
            # At least a meaningful spatial baseline relative to the selected graph.
            footprint = unary_union(polygons)
            inliers = np.asarray(registration["moving_xy"])[np.asarray(audit["inlier_mask"], dtype=bool)]
            support = MultiPoint(inliers).convex_hull.intersection(footprint).area
            audit["selected_footprint_support_fraction"] = support / footprint.area
            if support < 0.05 * footprint.area:
                raise ValueError("Registration control points cover too little of the yearly graph")
            polygons = [affine_geometry(p, audit["affine"]) for p in polygons]
            audit["control_points_sha256"] = stable_digest({key: np.asarray(value).tolist()
                for key, value in registration.items()})
            audit["point_source"] = ("phase_guided_sift_ransac" if snapshot.get("registration_diagnostics")
                                     else "sift_mutual_ratio" if snapshot.get("register_to_reference") else "supplied")
            if snapshot.get("registration_diagnostics"):
                audit["image_registration"] = copy.deepcopy(snapshot["registration_diagnostics"])
        registrations.append({"year": snapshot["year"], **audit})
        all_polygons.append(polygons)

    nodes, spatial_edges, temporal_edges, comparisons, matching_refinement = [], [], [], [], []
    change_events, change_edges, change_assessments, change_refinement = [], [], [], []
    nodes_by_year = []
    for snapshot, document, polygons, registration in zip(ordered, documents, all_polygons, registrations):
        yearly_nodes = []
        year = snapshot["year"]
        for original, polygon in zip(document["nodes"], polygons):
            node_id = _identity("observation", [year, document["graph"]["id"], original["node_id"]])
            node = {"node_id": node_id, "year": year, "source_node_id": original["node_id"],
                    "source_graph_id": document["graph"]["id"],
                    "track_id": _identity("track", [node_id]),
                    "segment_type": copy.deepcopy(original["segment_type"]),
                    "mean_semantic_confidence": original.get("mean_semantic_confidence"),
                    "geometry_source_pixel": copy.deepcopy(original["geometry_source_pixel"]),
                    "geometry_world": mapping(polygon), "area_m2": polygon.area,
                    "centroid_world_xy": [polygon.centroid.x, polygon.centroid.y],
                    "representative_point_world_xy": list(polygon.representative_point().coords[0]),
                    "analysis_view": copy.deepcopy(original.get("analysis_view", {"included": True}))}
            nodes.append(node)
            yearly_nodes.append(node)
        nodes_by_year.append(yearly_nodes)
        ids = {n["source_node_id"]: n["node_id"] for n in yearly_nodes}
        for original in document["edges"]:
            metadata = {**document["graph"], **snapshot.get("georeference", {})}
            source_contact = original.get("contact_anchor_source_pixel_xy")
            if source_contact is None:
                source_contact = original["contact_centroid_source_pixel_xy"]
            contact = project_point(source_contact, metadata, target)
            if "affine" in registration:
                contact = list(affine_geometry(Point(contact), registration["affine"]).coords[0])
            spatial_edges.append({"edge_id": _identity("spatial", [year, original["edge_id"]]),
                                  "source": ids[original["source"]], "target": ids[original["target"]],
                                  "year": year, "directed": False, "relation": "spatial_adjacency",
                                  "contact_world_xy": contact,
                                  "source_edge": copy.deepcopy(original)})
    for index in range(len(ordered) - 1):
        result = match_snapshots(documents[index], documents[index+1],
                                 all_polygons[index], all_polygons[index+1], policy,
                                 show_progress=show_progress)
        matching_refinement.append({"source_year":ordered[index]["year"],
                                    "target_year":ordered[index+1]["year"],
                                    **refine_matching_links(result, documents[index], documents[index+1],
                                                            all_polygons[index], all_polygons[index+1], policy)})
        earlier, later = nodes_by_year[index], nodes_by_year[index+1]
        changes = detect_change_events(earlier,later,all_polygons[index],all_polygons[index+1],
                                       change_policy, show_progress=show_progress)
        changes = refine_change_events(changes, earlier, later, change_policy, spatial_edges)
        change_refinement.append({"source_year":earlier[0]["year"],
                                  "target_year":later[0]["year"], **changes["audit"]})
        change_events.extend(changes["events"])
        change_edges.extend(changes["edges"])
        change_assessments.extend(changes["assessments"])
        for link in result["links"]:
            i, j = link["i"], link["j"]
            a, b = earlier[i], later[j]
            local_overlap = all_polygons[index][i].intersection(all_polygons[index+1][j])
            local_anchor = list(local_overlap.representative_point().coords[0])
            limited = min(result["fractions_a"][i], result["fractions_b"][j]) < policy.minimum_visible_fraction
            if link["relation"] == "correspondence":
                b["track_id"] = a["track_id"]
            classes = [int(n["segment_type"]["id"]) for n in (a, b)]
            semantic_changed = None if 8 in classes else classes[0] != classes[1]
            measurable = not limited and link["relation"] == "correspondence"
            evidence = {k: v for k, v in link["evidence"].items() if k not in {"i", "j"}}
            temporal_edges.append({"edge_id": _identity("temporal", [a["node_id"], b["node_id"]]),
                "source": a["node_id"], "target": b["node_id"], "directed": True,
                "relation": link["relation"], "delta_years": b["year"]-a["year"],
                "source_anchor_world_xy": local_anchor,
                "target_anchor_world_xy": local_anchor.copy(),
                "evidence": evidence,
                "interpretation": "region_correspondence_not_proven_physical_object_identity",
                "change": {"semantic_from": classes[0], "semantic_to": classes[1],
                           "semantic_changed": semantic_changed, "coverage_limited": limited,
                           "area_delta_m2": b["area_m2"]-a["area_m2"] if measurable else None,
                           "area_relative_change": b["area_m2"]/a["area_m2"]-1 if measurable else None,
                           "geometry_change": 1-evidence["iou"],
                           "status": "candidate_change_requires_validation"}})
        for state in result["node_states"]:
            yearly = earlier if state["side"] == "source" else later
            state["node_id"] = yearly[state.pop("index")]["node_id"]
        for candidate in result["candidates"]:
            candidate["source"] = earlier[candidate.pop("i")]["node_id"]
            candidate["target"] = later[candidate.pop("j")]["node_id"]
        comparisons.append({"source_year": ordered[index]["year"], "target_year": ordered[index+1]["year"],
                            "common_coverage_m2": result["common_coverage_m2"],
                            "node_states": result["node_states"], "candidates": result["candidates"]})
    semantic_edges = build_semantic_temporal_edges(nodes, temporal_edges, change_edges)
    document = {"schema_version": SCHEMA_VERSION, "graph": {
        "id": _identity("spatiotemporal", [provenance, asdict(policy), asdict(change_policy),
                                         target.to_wkt(), registrations,
                                         "local_object_change_witness_refinement_v1",
                                         "local_same_class_lineage_refinement_v1",
                                         "class_specific_outcome_edges_v1"]),
        "edge_model": "undirected_spatial_and_directed_forward_temporal",
        "primary_temporal_edge_collection":"semantic_temporal_edges",
        "raw_temporal_evidence_collections":["temporal_edges","change_edges"],
        "years": sorted(years), "world_crs": target.to_wkt(), "world_units": "metre",
        "node_count": len(nodes), "spatial_edge_count": len(spatial_edges),
        "temporal_edge_count": len(temporal_edges), "matching_policy": asdict(policy),
        "change_edge_count":len(change_edges), "change_event_count":len(change_events),
        "semantic_temporal_edge_count":len(semantic_edges),
        "all_temporal_edge_count":len(temporal_edges)+len(change_edges),"change_policy":asdict(change_policy),
        "registration": registrations, "snapshots": provenance,
        "change_refinement": change_refinement,
        "matching_refinement": matching_refinement,
        "matching_scope": "adjacent_observed_years",
        "coverage_definition": "intersection_of_unions_of_observed_region_polygons",
        "score_calibration": "engineering_heuristic_not_probability"},
        "nodes": nodes, "spatial_edges": spatial_edges, "temporal_edges": temporal_edges,
        "change_edges":change_edges,"change_events":change_events,"change_assessments":change_assessments,
        "semantic_temporal_edges":semantic_edges,
        "comparisons": comparisons}
    validate_temporal_graph(document)
    if output_directory is not None:
        from .storage import save_result
        preview_layers = [{"year":snapshot["year"], "tiles":snapshot.get("preview_tiles", []),
                           "source_affine_transform":source["graph"]["source_affine_transform"],
                           "crs":source["graph"]["crs"],
                           "registration_affine":registration.get("affine")}
                          for snapshot, source, registration in zip(ordered, documents, registrations)]
        save_result(Path(output_directory).resolve(), document, preview_layers=preview_layers)
    return document


def load_stage5_result(directory):
    directory = Path(directory).resolve()
    public = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    from .schema import LEGACY_SCHEMA_VERSION
    if public.get("schema_version") not in {SCHEMA_VERSION,LEGACY_SCHEMA_VERSION} or public.get("status") != "complete":
        raise ValueError("Incomplete or unsupported Stage 5 manifest")
    expected = {"graph.json", "graph_arrays.npz", "nodes.geojson", "report.json", "temporal_graph.png"}
    if public["schema_version"] == SCHEMA_VERSION:
        expected.add("changes.geojson")
    artifact_names = set(public["artifact_checksums"])
    new_previews = "preview_index.json" in artifact_names
    if new_previews:
        required = expected | {"matching_graph.png", "temporal_links.png",
                               "change_graph.png", "preview_index.json"}
        legend_preview = "all_edge_types.png" in artifact_names
        outcome_preview = "outcome_graph.png" in artifact_names
        if legend_preview:
            required.add("all_edge_types.png")
        if outcome_preview:
            required.add("outcome_graph.png")
        if not required.issubset(artifact_names):
            raise ValueError("Stage 5 manifest has an incomplete preview index")
    elif artifact_names != expected and artifact_names != expected | {"change_graph.png", "temporal_links.png"}:
        raise ValueError("Stage 5 manifest has an incomplete artifact index")
    for filename, digest in public["artifact_checksums"].items():
        path = directory/filename
        if not path.is_file() or checksum(path) != digest:
            raise RuntimeError(f"Stage 5 artifact missing or changed: {path}")
    document = json.loads((directory/"graph.json").read_text(encoding="utf-8"))
    if document.get("schema_version") != public["schema_version"]:
        raise ValueError("Stage 5 manifest and graph schema versions disagree")
    validate_temporal_graph(document)
    if new_previews:
        if "semantic_temporal_edges" in document:
            relations = sorted({edge["relation"] for edge in document["semantic_temporal_edges"]
                                if not edge["relation"].endswith("_unchanged")})
        else:
            relations = sorted({edge["relation"] for edge in document["change_edges"]})
        if artifact_names != required | {f"change_{relation}.png" for relation in relations}:
            raise ValueError("Stage 5 manifest does not index every observed change relation")
        index = json.loads((directory/"preview_index.json").read_text(encoding="utf-8"))
        preview_index = {"whole_graph":"temporal_graph.png", "matching":"matching_graph.png",
                         "all_changes":"change_graph.png",
                         "change_by_relation":{relation:f"change_{relation}.png" for relation in relations}}
        if legend_preview:
            preview_index["edge_types"] = "all_edge_types.png"
        if outcome_preview:
            preview_index["outcomes"] = "outcome_graph.png"
        if index != preview_index:
            raise ValueError("Stage 5 preview index disagrees with the graph")
    for snapshot in document["graph"]["snapshots"]:
        for item in snapshot["inputs"]:
            path = Path(item["path"])
            if not path.is_file() or checksum(path) != item["sha256"]:
                raise RuntimeError(f"Stage 5 upstream input missing or changed: {path}")
    return document


def run_stage5(series_path, *, project_root=".", config=None, output_name=None):
    from src.config import load_config, DEFAULT_CONFIG
    root = Path(project_root).resolve()
    series_path = Path(series_path)
    series_path = series_path if series_path.is_absolute() else root/series_path
    series = json.loads(series_path.read_text(encoding="utf-8"))
    cfg = load_config(config or DEFAULT_CONFIG).get("stage5", {})
    snapshots = copy.deepcopy(series["snapshots"])
    for snapshot in snapshots:
        if "document" in snapshot:
            raise ValueError("CLI series must reference completed Stage 4 graph directories")
        for key in ("graph_directory", "image_path"):
            if key in snapshot:
                path = Path(snapshot[key])
                snapshot[key] = str(path if path.is_absolute() else root/path)
        for tile in snapshot.get("preview_tiles", []):
            image = Path(tile["path"])
            tile["path"] = str(image if image.is_absolute() else root/image)
    name = output_name or series.get("name") or stable_digest(series)[:12]
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Stage 5 output name must be a safe collection name")
    directory = root/cfg.get("output_directory", "artifacts/stage5")/name/"01_temporal_graph"
    document = build_temporal_graph(snapshots, target_crs=series.get("target_crs", cfg.get("target_crs")),
                                    policy=cfg.get("matching", {}),change_policy=cfg.get("changes",{}), output_directory=directory)
    return {"status": "complete", "output_directory": str(directory),
            "years": document["graph"]["years"],
            **{key: document["graph"][key] for key in ("node_count", "spatial_edge_count", "temporal_edge_count",
                                                      "change_edge_count","change_event_count","all_temporal_edge_count")}}
