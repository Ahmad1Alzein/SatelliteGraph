"""Build and persist one tile-local graph from validated Stage 2R refinement."""

from __future__ import annotations

import copy
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from tqdm import tqdm

from src.config import DEFAULT_CONFIG, load_config, select_stage3_config, validate_config
from src.stage2 import load_stage2_result
from src.stage2.artifacts import atomic_json, checksum, manifest, now, require_fresh
from src.refinement import load_refinement_result

from .adjacency import build_adjacency_edges
from .config import config_hash
from .geometry import build_node_geometries
from .schema import (
    ADJACENCY_DEFINITION,
    ANALYSIS_SCHEMA_VERSION,
    REFINEMENT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    make_edge_id,
    make_node_id,
    validate_analysis_graph_document,
    validate_graph_document,
)
from .storage import save_adjacency_csv, save_graph_arrays, save_graph_json, save_nodes_geojson


@dataclass(slots=True)
class Stage3Result:
    """Loaded Stage 3 graph and the artifacts that describe it."""

    tile_id: str
    output_directory: Path
    graph: dict[str, Any]
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    analysis_graph: dict[str, Any]
    analysis_nodes: list[dict[str, Any]]
    analysis_edges: list[dict[str, Any]]
    report: dict[str, Any]
    artifacts: dict[str, str]

    def summary(self) -> dict[str, Any]:
        """Return the compact JSON-safe summary used by the CLI."""

        statistics = self.report.get("graph_statistics", {})
        display = self.report.get("visualization_graph", {})
        return {
            "status": "complete",
            "schema_version": SCHEMA_VERSION,
            "tile_id": self.tile_id,
            "output_directory": str(self.output_directory),
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "connected_component_count": statistics.get("connected_component_count"),
            "isolated_node_count": statistics.get("isolated_node_count"),
            "displayed_node_count": display.get("displayed_node_count"),
            "displayed_edge_count": display.get("displayed_edge_count"),
            "analysis_node_count": len(self.analysis_nodes),
            "analysis_edge_count": len(self.analysis_edges),
            "semantic_class_counts": statistics.get("semantic_class_counts", {}),
            "artifacts": self.artifacts,
        }


@dataclass(slots=True)
class _BeforeRefinementGraph:
    """Visual-only graph reconstructed from immutable ``12_final`` inputs."""

    directory: Path
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    image_shape: tuple[int, int]
    inputs: list[Path]


def run_stage3(
    tile_id: str,
    *,
    project_root: str | Path = ".",
    config: Mapping[str, Any] | str | Path | None = None,
    show_progress: bool = True,
) -> Stage3Result:
    """Build one exact tile graph from a validated Stage 2R refinement result."""

    root = Path(project_root).expanduser().resolve()
    project_config = _load_project_config(config)
    stage3_config = select_stage3_config(project_config)
    progress = _Stage3Progress(enabled=show_progress)
    started = now()
    try:
        progress.start("load")
        stage2_directory, stage2_manifest, stage2_result, inputs = _load_stage2_input(
            root, tile_id, stage3_config
        )
        canonical_tile_id = stage2_result.tile_id
        progress.finish()

        progress.start("geometry")
        node_geometries = build_node_geometries(
            stage2_result.instance_map,
            stage2_result.tile_metadata,
        )
        nodes = _build_nodes(stage2_result, node_geometries)
        progress.finish()

        progress.start("adjacency")
        raw_edges = build_adjacency_edges(
            stage2_result.instance_map,
            node_geometries=node_geometries,
            tile_metadata=stage2_result.tile_metadata,
            minimum_shared_boundary_pixels=int(
                stage3_config["adjacency"]["minimum_shared_boundary_pixels"]
            ),
        )
        edges = _build_edges(canonical_tile_id, nodes, raw_edges, node_geometries)
        analysis_view = _annotate_analysis_view(
            nodes, edges, stage3_config["visualization"]
        )
        graph_statistics = _annotate_graph(nodes, edges)
        before_graph = _build_before_refinement_graph(
            stage2_directory,
            expected_tile_id=canonical_tile_id,
            expected_shape=stage2_result.instance_map.shape,
            stage3_config=stage3_config,
        )
        inputs = _unique_paths(
            [
                *inputs,
                *before_graph.inputs,
                stage2_directory / "refined_overlay.png",
                stage2_directory / "refined_segmentation.png",
            ]
        )
        for input_path in inputs:
            require_fresh(input_path)
        progress.finish()

        graph_document = _graph_document(
            canonical_tile_id,
            nodes,
            edges,
            graph_statistics,
            stage2_result,
            stage2_directory,
            stage2_manifest,
            inputs,
            config_hash(stage3_config),
            analysis_view,
        )
        validate_graph_document(graph_document)

        progress.start("export")
        output_directory = _output_directory(root, canonical_tile_id, stage3_config)
        artifacts, report = _persist_result(
            output_directory=output_directory,
            graph_document=graph_document,
            graph_statistics=graph_statistics,
            stage2_result=stage2_result,
            stage2_directory=stage2_directory,
            stage2_manifest=stage2_manifest,
            stage3_config=stage3_config,
            inputs=inputs,
            before_graph=before_graph,
            started=started,
        )
        progress.finish()
        progress.complete()
    finally:
        progress.close()

    loaded = load_stage3_result(output_directory)
    if loaded.report != report or loaded.artifacts != artifacts:
        raise RuntimeError("Persisted Stage 3 result does not match the completed in-memory result")
    return loaded


def load_stage3_result(output_directory: str | Path) -> Stage3Result:
    """Load a previously completed Stage 3 graph without rebuilding it."""

    directory = Path(output_directory).expanduser().resolve()
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Stage 3 manifest not found: {manifest_path}")
    artifact_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if artifact_manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported Stage 3 result schema: {artifact_manifest.get('schema_version')!r}")
    if artifact_manifest.get("status") != "complete":
        raise ValueError(f"Stage 3 result is not complete: {artifact_manifest.get('status')!r}")
    if artifact_manifest.get("stale"):
        raise RuntimeError(f"Stage 3 result is stale: {directory}")

    graph_path = directory / "graph.json"
    analysis_graph_path = directory / "graph_analysis.json"
    report_path = directory / "report.json"
    if not graph_path.is_file() or not analysis_graph_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(f"Stage 3 graph, analysis graph, or report is missing from: {directory}")
    graph_document = json.loads(graph_path.read_text(encoding="utf-8"))
    validate_graph_document(graph_document)
    analysis_document = json.loads(analysis_graph_path.read_text(encoding="utf-8"))
    validate_analysis_graph_document(analysis_document, graph_document)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    tile_id = str(artifact_manifest.get("tile_id", ""))
    if not tile_id or graph_document["graph"].get("tile_id") != tile_id:
        raise ValueError("Stage 3 manifest and graph tile IDs disagree")
    artifacts = {str(key): str(value) for key, value in artifact_manifest.get("artifacts", {}).items()}
    _validate_persisted_artifacts(
        directory,
        artifact_manifest,
        graph_document,
        analysis_document,
        report,
        artifacts,
    )
    return Stage3Result(
        tile_id=tile_id,
        output_directory=directory,
        graph=graph_document["graph"],
        nodes=graph_document["nodes"],
        edges=graph_document["edges"],
        analysis_graph=analysis_document["graph"],
        analysis_nodes=analysis_document["nodes"],
        analysis_edges=analysis_document["edges"],
        report=report,
        artifacts=artifacts,
    )


def _validate_persisted_artifacts(
    directory: Path,
    artifact_manifest: Mapping[str, Any],
    graph_document: Mapping[str, Any],
    analysis_document: Mapping[str, Any],
    report: Mapping[str, Any],
    artifacts: Mapping[str, str],
) -> None:
    expected = {
        "graph": directory / "graph.json",
        "graph_arrays": directory / "graph_arrays.npz",
        "analysis_graph": directory / "graph_analysis.json",
        "analysis_graph_arrays": directory / "graph_analysis_arrays.npz",
        "nodes_geojson": directory / "nodes.geojson",
        "adjacency_csv": directory / "adjacency.csv",
        "graph_overlay": directory / "graph_overlay.png",
        "graph_spatial": directory / "graph_spatial.png",
        "graph_spatial_svg": directory / "graph_spatial.svg",
        "overview": directory / "stage3_overview.png",
        "report": directory / "report.json",
        "phase_manifest": directory / "phase_manifest.json",
        "manifest": directory / "manifest.json",
    }
    if isinstance(report.get("refinement_graph_comparison"), Mapping):
        expected["refinement_graph_before_after"] = (
            directory / "refinement_graph_before_after.png"
        )
    if set(artifacts) != set(expected):
        raise ValueError("Stage 3 manifest artifact index is incomplete")
    for key, path in expected.items():
        if Path(artifacts[key]).expanduser().resolve() != path.resolve():
            raise ValueError(f"Stage 3 manifest artifact path disagrees for {key!r}")
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"Stage 3 artifact is missing or empty: {path}")

    artifact_checksums = artifact_manifest.get("artifact_checksums")
    if not isinstance(artifact_checksums, Mapping):
        raise ValueError("Stage 3 manifest has no artifact checksums")
    if set(artifact_checksums) != set(expected) - {"manifest"}:
        raise ValueError("Stage 3 manifest artifact checksums are incomplete")
    for key, expected_checksum in artifact_checksums.items():
        if key not in expected or key == "manifest":
            raise ValueError(f"Stage 3 manifest has an unexpected artifact checksum: {key!r}")
        if checksum(expected[key]) != expected_checksum:
            raise RuntimeError(f"Stage 3 artifact checksum mismatch: {expected[key]}")

    phase = json.loads(expected["phase_manifest"].read_text(encoding="utf-8"))
    if phase.get("phase") != "01_graph" or phase.get("status") != "complete" or phase.get("stale"):
        raise RuntimeError("Stage 3 phase manifest is incomplete or stale")
    recorded_outputs = {Path(value).expanduser().resolve() for value in phase.get("outputs", [])}
    required_phase_outputs = {
        path.resolve() for key, path in expected.items() if key not in {"manifest", "phase_manifest"}
    }
    if not required_phase_outputs.issubset(recorded_outputs):
        raise ValueError("Stage 3 phase manifest does not list every output artifact")
    for record in phase.get("inputs", []):
        if not isinstance(record, Mapping) or not record.get("path") or not record.get("sha256"):
            raise ValueError("Stage 3 phase manifest contains an invalid input record")
        input_path = Path(str(record["path"])).expanduser()
        if not input_path.is_file() or checksum(input_path) != record["sha256"]:
            raise RuntimeError(f"Stage 3 source input changed or is missing: {input_path}")

    node_count = len(graph_document["nodes"])
    edge_count = len(graph_document["edges"])
    if (
        int(artifact_manifest.get("node_count", -1)) != node_count
        or int(artifact_manifest.get("edge_count", -1)) != edge_count
    ):
        raise ValueError("Stage 3 manifest counts disagree with graph.json")
    report_statistics = report.get("graph_statistics")
    if not isinstance(report_statistics, Mapping) or (
        int(report_statistics.get("node_count", -1)) != node_count
        or int(report_statistics.get("edge_count", -1)) != edge_count
    ):
        raise ValueError("Stage 3 report counts disagree with graph.json")

    analysis_node_count = len(analysis_document["nodes"])
    analysis_edge_count = len(analysis_document["edges"])
    if artifact_manifest.get("analysis_schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise ValueError("Stage 3 manifest has an unsupported analysis graph schema")
    if (
        int(artifact_manifest.get("analysis_node_count", -1)) != analysis_node_count
        or int(artifact_manifest.get("analysis_edge_count", -1)) != analysis_edge_count
    ):
        raise ValueError("Stage 3 manifest counts disagree with graph_analysis.json")
    report_analysis = report.get("analysis_graph")
    if not isinstance(report_analysis, Mapping) or (
        report_analysis.get("schema_version") != ANALYSIS_SCHEMA_VERSION
        or int(report_analysis.get("node_count", -1)) != analysis_node_count
        or int(report_analysis.get("edge_count", -1)) != analysis_edge_count
    ):
        raise ValueError("Stage 3 report counts disagree with graph_analysis.json")
    source_graph = analysis_document["graph"].get("source_graph", {})
    if (
        source_graph.get("artifact") != expected["graph"].name
        or source_graph.get("sha256") != checksum(expected["graph"])
    ):
        raise RuntimeError("Stage 3 analysis graph does not match the authoritative graph artifact")

    with np.load(expected["graph_arrays"], allow_pickle=False) as arrays:
        if str(arrays["schema_version"].item()) != SCHEMA_VERSION:
            raise ValueError("Stage 3 graph arrays have an unsupported schema")
        if str(arrays["graph_view"].item()) != "authoritative":
            raise ValueError("Stage 3 graph arrays have an invalid graph view")
        if str(arrays["tile_id"].item()) != graph_document["graph"]["tile_id"]:
            raise ValueError("Stage 3 graph arrays have a different tile ID")
        if str(arrays["adjacency_definition"].item()) != ADJACENCY_DEFINITION:
            raise ValueError("Stage 3 graph arrays have a different adjacency definition")
        if arrays["instance_ids"].shape != (node_count,) or arrays["edge_index"].shape != (
            2,
            edge_count,
        ):
            raise ValueError("Stage 3 graph array dimensions disagree with graph.json")

    with np.load(expected["analysis_graph_arrays"], allow_pickle=False) as arrays:
        if str(arrays["schema_version"].item()) != ANALYSIS_SCHEMA_VERSION:
            raise ValueError("Stage 3 analysis graph arrays have an unsupported schema")
        if str(arrays["graph_view"].item()) != "analysis":
            raise ValueError("Stage 3 analysis graph arrays have an invalid graph view")
        if str(arrays["tile_id"].item()) != analysis_document["graph"]["tile_id"]:
            raise ValueError("Stage 3 analysis graph arrays have a different tile ID")
        if str(arrays["adjacency_definition"].item()) != ADJACENCY_DEFINITION:
            raise ValueError("Stage 3 analysis graph arrays have a different adjacency definition")
        if arrays["instance_ids"].shape != (analysis_node_count,) or arrays[
            "edge_index"
        ].shape != (2, analysis_edge_count):
            raise ValueError(
                "Stage 3 analysis graph array dimensions disagree with graph_analysis.json"
            )
        if arrays["node_ids"].tolist() != [
            str(node["node_id"]) for node in analysis_document["nodes"]
        ]:
            raise ValueError("Stage 3 analysis graph array node order disagrees")
        if arrays["endpoint_instance_ids"].tolist() != [
            [int(edge[key]) for edge in analysis_document["edges"]]
            for key in ("source_instance_id", "target_instance_id")
        ]:
            raise ValueError("Stage 3 analysis graph array edge order disagrees")

    geojson = json.loads(expected["nodes_geojson"].read_text(encoding="utf-8"))
    if geojson.get("type") != "FeatureCollection" or len(geojson.get("features", [])) != node_count:
        raise ValueError("Stage 3 node GeoJSON count disagrees with graph.json")
    with expected["adjacency_csv"].open(newline="", encoding="utf-8") as stream:
        if sum(1 for _ in csv.DictReader(stream)) != edge_count:
            raise ValueError("Stage 3 adjacency CSV count disagrees with graph.json")

    from PIL import Image

    image_keys = ["graph_overlay", "graph_spatial", "overview"]
    if "refinement_graph_before_after" in expected:
        image_keys.append("refinement_graph_before_after")
    for key in image_keys:
        with Image.open(expected[key]) as image:
            image.verify()


class _Stage3Progress:
    LABELS = {
        "load": "Validate and load Stage 2",
        "geometry": "Build exact node geometry",
        "adjacency": "Build adjacency edges",
        "export": "Validate, visualize, and export",
    }

    def __init__(self, *, enabled: bool) -> None:
        self.bar = tqdm(
            total=len(self.LABELS),
            desc="Stage 3 | Starting",
            unit="phase",
            dynamic_ncols=True,
            disable=not enabled,
        )

    def start(self, phase: str) -> None:
        self.bar.set_description_str(f"Stage 3 | {self.LABELS[phase]}", refresh=True)

    def finish(self) -> None:
        self.bar.update(1)

    def complete(self) -> None:
        self.bar.set_description_str("Stage 3 | Complete", refresh=True)

    def close(self) -> None:
        self.bar.close()


def _load_project_config(config: Mapping[str, Any] | str | Path | None) -> dict[str, Any]:
    if config is None:
        return load_config(DEFAULT_CONFIG)
    if isinstance(config, (str, Path)):
        return load_config(config)
    project_config = copy.deepcopy(dict(config))
    validate_config(project_config)
    return project_config


def _load_stage2_input(
    project_root: Path,
    requested_tile_id: str,
    stage3_config: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], Any, list[Path]]:
    stage2_root = _resolved_root(project_root, str(stage3_config["stage2_root"]))
    if stage3_config.get("input", {}).get("source") != "refinement":
        raise ValueError("Stage 3 requires stage3.input.source='refinement'")
    directory = _safe_tile_directory(stage2_root, requested_tile_id) / "13_refinement"
    public_manifest_path = directory / "manifest.json"
    require_fresh(public_manifest_path)
    public_manifest = json.loads(public_manifest_path.read_text(encoding="utf-8"))
    if public_manifest.get("schema_version") != REFINEMENT_SCHEMA_VERSION:
        raise ValueError(
            f"Stage 3 requires refinement schema {REFINEMENT_SCHEMA_VERSION!r}; "
            f"found {public_manifest.get('schema_version')!r}"
        )
    if public_manifest.get("status") != "complete":
        raise ValueError(f"Refinement result is not complete: {public_manifest.get('status')!r}")
    if public_manifest.get("stale"):
        raise RuntimeError(
            f"Refinement result is stale; rerun refinement for {requested_tile_id!r}"
        )
    manifest_tile_id = str(public_manifest.get("tile_id", ""))
    if manifest_tile_id != requested_tile_id:
        raise ValueError(
            f"Requested tile {requested_tile_id!r} does not match refinement manifest tile {manifest_tile_id!r}"
        )

    phase_manifest_path = directory / "phase_manifest.json"
    if not phase_manifest_path.is_file():
        raise FileNotFoundError(f"Refinement phase manifest not found: {phase_manifest_path}")
    phase_manifest = json.loads(phase_manifest_path.read_text(encoding="utf-8"))
    if phase_manifest.get("status") != "complete" or phase_manifest.get("stale"):
        raise RuntimeError(f"Refinement phase is incomplete or stale: {phase_manifest_path}")

    inputs = [
        public_manifest_path,
        phase_manifest_path,
        directory / "segments.npz",
        directory / "segments.json",
        directory / "segmentation.npz",
        directory / "refinement_operation_map.npz",
        directory / "tile_metadata.json",
        directory / "class_registry.json",
        directory / "lineage.json",
        directory / "operations.json",
        directory / "report.json",
    ]
    for path in inputs:
        require_fresh(path)
    result = load_refinement_result(directory)
    if result.tile_id != requested_tile_id:
        raise ValueError("Loaded refinement tile ID does not match the requested tile")
    if not result.report.get("every_valid_pixel_exactly_once", False):
        raise ValueError("Refinement report does not guarantee every valid pixel exactly once")
    if result.instance_map.shape != result.valid_mask.shape:
        raise ValueError("Refinement instance and valid maps have different shapes")
    if len(result.segments) != int(result.instance_map.max(initial=0)):
        raise ValueError("Refinement segment count does not match the maximum instance ID")
    return directory, public_manifest, result, inputs


def _build_before_refinement_graph(
    refinement_directory: Path,
    *,
    expected_tile_id: str,
    expected_shape: tuple[int, int],
    stage3_config: Mapping[str, Any],
) -> _BeforeRefinementGraph:
    """Reconstruct the pre-refinement graph for a visual comparison only.

    The authoritative Stage 3 graph continues to contain only refined nodes
    and edges.  This second graph is derived from the sibling immutable
    ``12_final`` result and exists to make the effect of refinement inspectable.
    """

    directory = refinement_directory.parent / "12_final"
    inputs = [
        directory / "manifest.json",
        directory / "segments.npz",
        directory / "segments.json",
        directory / "segmentation.npz",
        directory / "tile_metadata.json",
        directory / "report.json",
        directory / "final_segmentation.png",
    ]
    for path in inputs:
        require_fresh(path)
    result = load_stage2_result(directory)
    if result.tile_id != expected_tile_id:
        raise ValueError("Stage 2 and refinement tile IDs disagree")
    if result.instance_map.shape != expected_shape:
        raise ValueError("Stage 2 and refinement raster shapes disagree")

    geometries = build_node_geometries(result.instance_map, result.tile_metadata)
    nodes = _build_nodes(result, geometries)
    raw_edges = build_adjacency_edges(
        result.instance_map,
        node_geometries=geometries,
        tile_metadata=result.tile_metadata,
        minimum_shared_boundary_pixels=int(
            stage3_config["adjacency"]["minimum_shared_boundary_pixels"]
        ),
    )
    edges = _build_edges(result.tile_id, nodes, raw_edges, geometries)
    _annotate_analysis_view(nodes, edges, stage3_config["visualization"])
    return _BeforeRefinementGraph(
        directory=directory.resolve(),
        nodes=nodes,
        edges=edges,
        image_shape=tuple(int(value) for value in result.instance_map.shape),
        inputs=[path.resolve() for path in inputs],
    )


def _unique_paths(paths: Sequence[Path]) -> list[Path]:
    unique: dict[str, Path] = {}
    for path in paths:
        resolved = path.expanduser().resolve()
        unique[str(resolved)] = resolved
    return list(unique.values())


def _build_nodes(stage2_result: Any, geometries: Mapping[int, Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected_ids = set(range(1, len(stage2_result.segments) + 1))
    if set(geometries) != expected_ids:
        raise ValueError("Polygonized Stage 3 node IDs do not match Stage 2 instances")
    tile_id = stage2_result.tile_id
    source_image_id = stage2_result.tile_metadata.get("source_image_id")
    nodes: list[dict[str, Any]] = []
    for instance_id, stage2_record in enumerate(stage2_result.segments, 1):
        if int(stage2_record.get("instance_id", 0)) != instance_id:
            raise ValueError(f"Stage 2 record order disagrees at instance {instance_id}")
        geometry = copy.deepcopy(dict(geometries[instance_id]))
        label = copy.deepcopy(stage2_record.get("label", {}))
        if "id" not in label or "name" not in label:
            raise ValueError(f"Stage 2 instance {instance_id} has no semantic label")
        spaces = geometry["coordinate_spaces"]
        representative = {
            name: copy.deepcopy(value.get("representative_point_xy")) if value.get("available") else None
            for name, value in spaces.items()
        }
        metrics = geometry["metrics"]
        boundary = geometry["tile_boundary_contacts"]
        segment_id = str(stage2_record["segment_id"])
        node = {
            "id": make_node_id(tile_id, segment_id),
            "node_id": make_node_id(tile_id, segment_id),
            "tile_id": tile_id,
            "source_image_id": source_image_id,
            "segment_id": segment_id,
            "instance_id": instance_id,
            "mask_index": int(stage2_record.get("mask_index", instance_id - 1)),
            "segment_type": {"id": int(label["id"]), "name": str(label["name"])},
            "label": label,
            "geometry_source": copy.deepcopy(stage2_record.get("geometry_source")),
            "source": stage2_record.get("source"),
            "source_code": stage2_record.get("source_code"),
            "area_pixels": int(stage2_record["area_pixels"]),
            "mean_semantic_confidence": stage2_record.get("mean_semantic_confidence"),
            "quality": {
                "predicted_iou": stage2_record.get("predicted_iou"),
                "stability_score": stage2_record.get("stability_score"),
                "metrics": copy.deepcopy(stage2_record.get("metrics", {})),
            },
            "provenance": copy.deepcopy(stage2_record.get("provenance", {})),
            "explanation": stage2_record.get("explanation"),
            "geometry_type": geometry["geometry_type"],
            "geometry": geometry,
            "coordinates": copy.deepcopy(stage2_record.get("coordinates", {})),
            "representative_point": representative,
            "representative_point_xy": representative["tile_pixel"],
            "tile_boundary_contacts": boundary,
            "touches_tile_boundary": bool(boundary["touches"]),
            "geometry_metrics": metrics,
            "refinement_metadata": copy.deepcopy(stage2_record),
            "source_stage2_metadata": copy.deepcopy(
                stage2_record.get(
                    "stage2_metadata_records",
                    [stage2_record.get("stage2_metadata", {})],
                )
            ),
            # Compatibility alias for consumers of the v1 graph schema.  It
            # now means the primary immutable Stage 2 ancestor, while the
            # exact raster-producing record lives in refinement_metadata.
            "stage2_metadata": copy.deepcopy(stage2_record.get("stage2_metadata", {})),
        }
        if int(metrics["area_pixels"]) != node["area_pixels"]:
            raise ValueError(f"Polygonized area disagrees with Stage 2 instance {instance_id}")
        nodes.append(node)
    return nodes


def _build_edges(
    tile_id: str,
    nodes: list[dict[str, Any]],
    raw_edges: list[dict[str, Any]],
    geometries: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_instance = {int(node["instance_id"]): node for node in nodes}
    edges: list[dict[str, Any]] = []
    for raw in raw_edges:
        first = int(raw.get("source_instance_id", raw.get("source", 0)))
        second = int(raw.get("target_instance_id", raw.get("target", 0)))
        first, second = sorted((first, second))
        if first not in by_instance or second not in by_instance or first == second:
            raise ValueError(f"Adjacency builder returned invalid endpoints: {(first, second)}")
        source_node, target_node = by_instance[first], by_instance[second]
        count = int(
            raw.get(
                "shared_boundary_side_count",
                raw.get("shared_boundary", {}).get("side_count", 0),
            )
        )
        left_right = int(
            raw.get(
                "left_right_contact_count",
                raw.get("shared_boundary", {}).get("left_right_side_count", 0),
            )
        )
        top_bottom = int(
            raw.get(
                "top_bottom_contact_count",
                raw.get("shared_boundary", {}).get("top_bottom_side_count", 0),
            )
        )
        pixel_length = float(raw.get("shared_boundary_length_pixels", count))
        projected_length = raw.get("shared_boundary_length_projected")
        if isinstance(projected_length, Mapping):
            projected_length = projected_length.get("value") if projected_length.get("available") else None
        projected_metric = raw.get("shared_boundary", {}).get("projected_length", {})
        source_perimeter = float(geometries[first]["metrics"]["perimeter_pixel_units"])
        target_perimeter = float(geometries[second]["metrics"]["perimeter_pixel_units"])
        source_centroid = geometries[first]["metrics"]["centroid_tile_pixel_xy"]
        target_centroid = geometries[second]["metrics"]["centroid_tile_pixel_xy"]
        edge = {
            "edge_id": make_edge_id(tile_id, first, second),
            "source": source_node["node_id"],
            "target": target_node["node_id"],
            "source_instance_id": first,
            "target_instance_id": second,
            "relation": "adjacent",
            "adjacency_definition": ADJACENCY_DEFINITION,
            "contact_type": "shared_boundary",
            "shared_boundary_side_count": count,
            "left_right_contact_count": left_right,
            "top_bottom_contact_count": top_bottom,
            "shared_boundary_length_pixels": pixel_length,
            "shared_boundary_length_projected": (
                None if projected_length is None else float(projected_length)
            ),
            "shared_boundary_projected_unit": raw.get("shared_boundary_projected_unit")
            or projected_metric.get("unit"),
            "shared_boundary_projected_crs": raw.get("shared_boundary_projected_crs")
            or projected_metric.get("crs"),
            "contact_fraction_of_source_pixel_perimeter": count / source_perimeter,
            "contact_fraction_of_target_pixel_perimeter": count / target_perimeter,
            "contact_fractions": copy.deepcopy(raw.get("contact_fractions", {})),
            "contact_coordinates": copy.deepcopy(raw.get("contact_coordinates", {})),
            "source_class_id": source_node["segment_type"]["id"],
            "source_class": source_node["segment_type"]["name"],
            "target_class_id": target_node["segment_type"]["id"],
            "target_class": target_node["segment_type"]["name"],
            "same_semantic_class": source_node["segment_type"]["id"]
            == target_node["segment_type"]["id"],
            "centroid_distance_pixels": float(
                math.hypot(
                    float(target_centroid[0]) - float(source_centroid[0]),
                    float(target_centroid[1]) - float(source_centroid[1]),
                )
            ),
            "centroid_distance_projected": _projected_centroid_distance(
                geometries[first], geometries[second]
            ),
        }
        if count <= 0 or left_right + top_bottom != count:
            raise ValueError(f"Adjacency contact counts are inconsistent for {(first, second)}")
        edges.append(edge)
    edges.sort(key=lambda edge: (edge["source_instance_id"], edge["target_instance_id"]))
    return edges


def _projected_centroid_distance(
    first_geometry: Mapping[str, Any], second_geometry: Mapping[str, Any]
) -> dict[str, Any]:
    first = first_geometry["coordinate_spaces"]["projected"]
    second = second_geometry["coordinate_spaces"]["projected"]
    if first.get("available") and second.get("available"):
        x1, y1 = first["centroid_xy"]
        x2, y2 = second["centroid_xy"]
        projected = first_geometry["metrics"]["projected_perimeter"]
        return {
            "available": True,
            "unavailable_reason": None,
            "value": float(math.hypot(float(x2) - float(x1), float(y2) - float(y1))),
            "unit": projected.get("unit"),
            "crs": first.get("crs"),
        }
    return {
        "available": False,
        "unavailable_reason": first.get("unavailable_reason") or second.get("unavailable_reason"),
        "value": None,
        "unit": None,
        "crs": first.get("crs") or second.get("crs"),
    }


def _annotate_graph(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    instance_ids = [int(node["instance_id"]) for node in nodes]
    parent = {instance_id: instance_id for instance_id in instance_ids}
    degree = {instance_id: 0 for instance_id in instance_ids}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root == second_root:
            return
        low, high = sorted((first_root, second_root))
        parent[high] = low

    for edge in edges:
        first, second = int(edge["source_instance_id"]), int(edge["target_instance_id"])
        degree[first] += 1
        degree[second] += 1
        union(first, second)

    groups: dict[int, list[int]] = {}
    for instance_id in instance_ids:
        groups.setdefault(find(instance_id), []).append(instance_id)
    ordered_groups = sorted((sorted(values) for values in groups.values()), key=lambda values: values[0])
    component_by_instance: dict[int, str] = {}
    for index, values in enumerate(ordered_groups, 1):
        component_id = f"component_{index:05d}"
        component_by_instance.update({instance_id: component_id for instance_id in values})

    for node in nodes:
        instance_id = int(node["instance_id"])
        node["degree"] = int(degree[instance_id])
        node["connected_component_id"] = component_by_instance[instance_id]
        node["is_isolated"] = degree[instance_id] == 0

    degree_values = np.asarray(list(degree.values()), dtype=np.int64)
    class_counts: dict[str, int] = {}
    geometry_counts: dict[str, int] = {}
    for node in nodes:
        class_name = str(node["segment_type"]["name"])
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        geometry_type = str(node["geometry_type"])
        geometry_counts[geometry_type] = geometry_counts.get(geometry_type, 0) + 1
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "connected_component_count": len(ordered_groups),
        "connected_component_sizes": [len(values) for values in ordered_groups],
        "isolated_node_count": int((degree_values == 0).sum()) if degree_values.size else 0,
        "degree": {
            "minimum": int(degree_values.min()) if degree_values.size else 0,
            "maximum": int(degree_values.max()) if degree_values.size else 0,
            "mean": float(degree_values.mean()) if degree_values.size else 0.0,
            "median": float(np.median(degree_values)) if degree_values.size else 0.0,
        },
        "semantic_class_counts": class_counts,
        "geometry_type_counts": geometry_counts,
        "tile_boundary_node_count": sum(bool(node["touches_tile_boundary"]) for node in nodes),
        "same_class_edge_count": sum(bool(edge["same_semantic_class"]) for edge in edges),
        "total_shared_boundary_side_count": sum(
            int(edge["shared_boundary_side_count"]) for edge in edges
        ),
    }


def _annotate_analysis_view(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Mark the preferred low-noise view while preserving the exact graph."""

    minimum_node_area = int(config.get("minimum_display_node_area_pixels", 0))
    minimum_unknown_area = int(
        config.get("minimum_display_unknown_area_pixels", 0)
    )
    minimum_topology_fragment_area = int(
        config.get("minimum_display_topology_fragment_area_pixels", 0)
    )
    minimum_boundary = int(
        config.get("minimum_display_shared_boundary_pixels", 0)
    )
    included_ids: set[str] = set()
    for node in nodes:
        reasons: list[str] = []
        area = int(node["area_pixels"])
        if area < minimum_node_area:
            reasons.append("below_minimum_display_node_area")
        if (
            str(node["segment_type"]["name"]) == "unknown"
            and area < minimum_unknown_area
        ):
            reasons.append("micro_unknown_below_analysis_area")
        if (
            _is_topology_fragment_node(node)
            and area < minimum_topology_fragment_area
        ):
            reasons.append("micro_topology_fragment_below_analysis_area")
        included = not reasons
        node["analysis_view"] = {
            "included": included,
            "exclusion_reasons": reasons,
        }
        if included:
            included_ids.add(str(node["node_id"]))

    included_edge_count = 0
    for edge in edges:
        reasons: list[str] = []
        if (
            str(edge["source"]) not in included_ids
            or str(edge["target"]) not in included_ids
        ):
            reasons.append("endpoint_excluded_from_analysis_view")
        if int(edge["shared_boundary_side_count"]) < minimum_boundary:
            reasons.append("weak_single_side_display_contact")
        included = not reasons
        edge["analysis_view"] = {
            "included": included,
            "exclusion_reasons": reasons,
        }
        included_edge_count += int(included)

    return {
        "authoritative_graph_is_unchanged": True,
        "filters_are_analysis_and_display_only": True,
        "minimum_node_area_pixels": minimum_node_area,
        "minimum_unknown_area_pixels": minimum_unknown_area,
        "minimum_topology_fragment_area_pixels": minimum_topology_fragment_area,
        "minimum_shared_boundary_pixels": minimum_boundary,
        "node_count": len(included_ids),
        "edge_count": included_edge_count,
        "suppressed_node_count": len(nodes) - len(included_ids),
        "suppressed_edge_count": len(edges) - included_edge_count,
    }


def _is_topology_fragment_node(node: Mapping[str, Any]) -> bool:
    """Return whether refinement created this node by connectivity splitting.

    Native Stage 2 objects also receive normalized component metadata, but they
    have no ``split_components`` operation in their lineage.  Absorption may
    reset component counters, so the durable operation lineage is authoritative.
    """

    record = node.get("refinement_metadata")
    if not isinstance(record, Mapping):
        return False
    lineage = record.get("lineage")
    lineage_ids = (
        lineage.get("refinement_operation_ids", [])
        if isinstance(lineage, Mapping)
        else []
    )
    history = record.get("refinement_history", [])
    operation_ids = [str(value) for value in [*lineage_ids, *history]]
    provenance = record.get("provenance")
    refinement_steps = (
        provenance.get("refinement", [])
        if isinstance(provenance, Mapping)
        else []
    )
    return any("::split_components::" in value for value in operation_ids) or any(
        isinstance(step, Mapping) and step.get("operation_type") == "split_components"
        for step in refinement_steps
    )


def _graph_document(
    tile_id: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    statistics: Mapping[str, Any],
    stage2_result: Any,
    stage2_directory: Path,
    stage2_manifest: Mapping[str, Any],
    inputs: list[Path],
    stage3_configuration_hash: str,
    analysis_view: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "graph": {
            "id": f"{tile_id}::tile_adjacency_graph",
            "tile_id": tile_id,
            "source_image_id": stage2_result.tile_metadata.get("source_image_id"),
            "directed": False,
            "multigraph": False,
            "adjacency_definition": ADJACENCY_DEFINITION,
            "connectivity": 4,
            "minimum_shared_boundary_pixels": 1,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "tile_shape_hw": [int(value) for value in stage2_result.instance_map.shape],
            "tile_metadata": copy.deepcopy(stage2_result.tile_metadata),
            "coordinate_reference": copy.deepcopy(
                stage2_result.report.get("coordinate_reference", {})
            ),
            "graph_statistics": copy.deepcopy(dict(statistics)),
            "analysis_view": copy.deepcopy(dict(analysis_view)),
            "configuration_hash": stage3_configuration_hash,
            "source_refinement": {
                "schema_version": stage2_manifest.get("schema_version"),
                "configuration_hash": stage2_manifest.get("configuration_hash"),
                "output_directory": str(stage2_directory.resolve()),
                "input_checksums": _input_checksum_index(
                    inputs, anchor=stage2_directory.parent
                ),
            },
        },
        "nodes": nodes,
        "edges": edges,
    }


def _persist_result(
    *,
    output_directory: Path,
    graph_document: dict[str, Any],
    graph_statistics: Mapping[str, Any],
    stage2_result: Any,
    stage2_directory: Path,
    stage2_manifest: Mapping[str, Any],
    stage3_config: Mapping[str, Any],
    inputs: list[Path],
    before_graph: _BeforeRefinementGraph,
    started: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    output_directory.mkdir(parents=True, exist_ok=True)
    graph_path = output_directory / "graph.json"
    arrays_path = output_directory / "graph_arrays.npz"
    analysis_graph_path = output_directory / "graph_analysis.json"
    analysis_arrays_path = output_directory / "graph_analysis_arrays.npz"
    nodes_geojson_path = output_directory / "nodes.geojson"
    adjacency_path = output_directory / "adjacency.csv"
    report_path = output_directory / "report.json"
    public_manifest_path = output_directory / "manifest.json"
    phase_manifest_path = output_directory / "phase_manifest.json"

    # Invalidate any previous completed generation before replacing its
    # artifacts.  Every artifact write below is atomic, and completion is
    # published only by the final manifest write.
    atomic_json(
        public_manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "in_progress",
            "stale": True,
            "tile_id": stage2_result.tile_id,
            "started_at": started,
        },
    )
    save_graph_json(graph_path, graph_document)
    save_graph_arrays(arrays_path, graph_document["nodes"], graph_document["edges"])
    analysis_document = _analysis_graph_document(graph_document, graph_path)
    validate_analysis_graph_document(analysis_document, graph_document)
    save_graph_json(analysis_graph_path, analysis_document)
    save_graph_arrays(
        analysis_arrays_path,
        analysis_document["nodes"],
        analysis_document["edges"],
        schema_version=ANALYSIS_SCHEMA_VERSION,
        tile_id=str(graph_document["graph"]["tile_id"]),
        graph_view="analysis",
    )
    save_nodes_geojson(nodes_geojson_path, _nodes_geojson(graph_document["nodes"], stage2_result))
    save_adjacency_csv(adjacency_path, graph_document["edges"])

    visualization_paths = _save_visualizations(
        stage2_directory,
        output_directory,
        graph_document["nodes"],
        graph_document["edges"],
        stage2_result.instance_map.shape,
        stage3_config["visualization"],
        before_graph=before_graph,
    )
    visualization_graph = _visualization_graph_statistics(
        graph_document["nodes"],
        graph_document["edges"],
        stage3_config["visualization"],
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "tile_id": stage2_result.tile_id,
        "adjacency_definition": ADJACENCY_DEFINITION,
        "corner_only_contacts_are_edges": False,
        "minimum_shared_boundary_pixels": int(
            stage3_config["adjacency"]["minimum_shared_boundary_pixels"]
        ),
        "graph_statistics": copy.deepcopy(dict(graph_statistics)),
        "analysis_graph": {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "selected_subgraph": True,
            "node_count": len(analysis_document["nodes"]),
            "edge_count": len(analysis_document["edges"]),
            "authoritative_node_count": len(graph_document["nodes"]),
            "authoritative_edge_count": len(graph_document["edges"]),
            "selection": copy.deepcopy(analysis_document["graph"]["selection"]),
            "graph_statistics": copy.deepcopy(
                analysis_document["graph"]["graph_statistics"]
            ),
        },
        "visualization_graph": visualization_graph,
        "refinement_graph_comparison": {
            "visual_diagnostic_only": True,
            "workflow_mode": stage2_result.report.get("workflow_mode"),
            "before_source": str(before_graph.directory),
            "before": _comparison_graph_counts(
                before_graph.nodes, before_graph.edges
            ),
            "after": _comparison_graph_counts(
                graph_document["nodes"], graph_document["edges"]
            ),
            "artifact": "refinement_graph_before_after.png",
            "purple_semantic_class": "unknown",
        },
        "input_validation": {
            "refinement_schema_version": stage2_manifest.get("schema_version"),
            "refinement_status": stage2_manifest.get("status"),
            "refinement_every_valid_pixel_exactly_once": bool(
                stage2_result.report.get("every_valid_pixel_exactly_once")
            ),
            "refinement_segment_count": len(stage2_result.segments),
            "node_count_matches_refinement": len(graph_document["nodes"])
            == len(stage2_result.segments),
        },
        "validation": {
            "one_node_per_positive_instance": True,
            "all_refinement_metadata_preserved": True,
            "source_stage2_lineage_preserved": True,
            "exact_unsimplified_polygon_geometry": True,
            "every_edge_has_shared_pixel_side": all(
                int(edge["shared_boundary_side_count"]) > 0
                for edge in graph_document["edges"]
            ),
            "no_self_loops": all(
                edge["source"] != edge["target"] for edge in graph_document["edges"]
            ),
            "deterministic_node_and_edge_order": True,
            "analysis_graph_is_selected_authoritative_subgraph": True,
            "analysis_graph_source_checksum_recorded": True,
        },
        "limitations": {
            "tile_local_only": True,
            "cross_tile_reconciliation_performed": False,
            "tile_boundary_contacts_are_not_cross_tile_edges": True,
        },
    }
    atomic_json(report_path, report)

    outputs = [
        graph_path,
        arrays_path,
        analysis_graph_path,
        analysis_arrays_path,
        nodes_geojson_path,
        adjacency_path,
        report_path,
        *visualization_paths.values(),
    ]
    phase = manifest(
        phase_manifest_path,
        "01_graph",
        inputs,
        outputs,
        config_hash(dict(stage3_config)),
        started,
    )
    if phase.get("status") != "complete":
        raise RuntimeError("Stage 3 phase manifest did not complete")

    artifacts = {
        "graph": str(graph_path.resolve()),
        "graph_arrays": str(arrays_path.resolve()),
        "analysis_graph": str(analysis_graph_path.resolve()),
        "analysis_graph_arrays": str(analysis_arrays_path.resolve()),
        "nodes_geojson": str(nodes_geojson_path.resolve()),
        "adjacency_csv": str(adjacency_path.resolve()),
        "graph_overlay": str(visualization_paths["graph_overlay"].resolve()),
        "graph_spatial": str(visualization_paths["graph_spatial"].resolve()),
        "graph_spatial_svg": str(visualization_paths["graph_spatial_svg"].resolve()),
        "overview": str(visualization_paths["overview"].resolve()),
        "refinement_graph_before_after": str(
            visualization_paths["refinement_graph_before_after"].resolve()
        ),
        "report": str(report_path.resolve()),
        "phase_manifest": str(phase_manifest_path.resolve()),
    }
    public_manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "stale": False,
        "tile_id": stage2_result.tile_id,
        "created_at": now(),
        "configuration_hash": config_hash(dict(stage3_config)),
        "adjacency_definition": ADJACENCY_DEFINITION,
        "source_refinement": {
            "schema_version": stage2_manifest.get("schema_version"),
            "output_directory": str(stage2_directory.resolve()),
            "input_checksums": copy.deepcopy(
                graph_document["graph"]["source_refinement"]["input_checksums"]
            ),
        },
        "node_count": len(graph_document["nodes"]),
        "edge_count": len(graph_document["edges"]),
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "analysis_node_count": len(analysis_document["nodes"]),
        "analysis_edge_count": len(analysis_document["edges"]),
        "artifacts": artifacts,
        "artifact_checksums": {
            key: checksum(Path(path)) for key, path in artifacts.items()
        },
    }
    atomic_json(public_manifest_path, public_manifest)
    artifacts["manifest"] = str(public_manifest_path.resolve())
    # Rewrite once so the public manifest's own artifact index is complete.
    public_manifest["artifacts"] = artifacts
    atomic_json(public_manifest_path, public_manifest)
    return artifacts, report


def _analysis_graph_document(
    authoritative_document: Mapping[str, Any],
    authoritative_path: Path,
) -> dict[str, Any]:
    """Build the explicit low-noise subgraph selected by analysis flags."""

    authoritative_graph = authoritative_document["graph"]
    selected_nodes = [
        copy.deepcopy(node)
        for node in authoritative_document["nodes"]
        if node.get("analysis_view", {}).get("included") is True
    ]
    selected_edges = [
        copy.deepcopy(edge)
        for edge in authoritative_document["edges"]
        if edge.get("analysis_view", {}).get("included") is True
    ]
    # Statistics are computed on copies because the topology helper annotates
    # nodes with degrees and connected-component IDs.  Records in this export
    # must remain byte-for-byte equivalent to their authoritative counterparts.
    analysis_statistics = _annotate_graph(
        copy.deepcopy(selected_nodes),
        copy.deepcopy(selected_edges),
    )
    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "graph": {
            "id": f"{authoritative_graph['id']}::analysis",
            "tile_id": authoritative_graph["tile_id"],
            "source_image_id": authoritative_graph.get("source_image_id"),
            "view": "analysis",
            "selected_subgraph": True,
            "directed": False,
            "multigraph": False,
            "adjacency_definition": ADJACENCY_DEFINITION,
            "connectivity": int(authoritative_graph.get("connectivity", 4)),
            "node_count": len(selected_nodes),
            "edge_count": len(selected_edges),
            "authoritative_node_count": len(authoritative_document["nodes"]),
            "authoritative_edge_count": len(authoritative_document["edges"]),
            "tile_shape_hw": copy.deepcopy(authoritative_graph.get("tile_shape_hw")),
            "coordinate_reference": copy.deepcopy(
                authoritative_graph.get("coordinate_reference", {})
            ),
            "selection": copy.deepcopy(authoritative_graph["analysis_view"]),
            "graph_statistics": analysis_statistics,
            "source_graph": {
                "artifact": authoritative_path.name,
                "schema_version": SCHEMA_VERSION,
                "graph_id": authoritative_graph["id"],
                "node_count": len(authoritative_document["nodes"]),
                "edge_count": len(authoritative_document["edges"]),
                "sha256": checksum(authoritative_path),
            },
        },
        "nodes": selected_nodes,
        "edges": selected_edges,
    }


def _visualization_graph_statistics(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Describe the readable display view without changing graph topology."""

    minimum_node_area = int(config.get("minimum_display_node_area_pixels", 0))
    minimum_unknown_area = int(
        config.get("minimum_display_unknown_area_pixels", 0)
    )
    minimum_topology_fragment_area = int(
        config.get("minimum_display_topology_fragment_area_pixels", 0)
    )
    minimum_boundary = int(
        config.get("minimum_display_shared_boundary_pixels", 0)
    )
    displayed_ids = {
        str(node["node_id"])
        for node in nodes
        if int(node["area_pixels"]) >= minimum_node_area
        and (
            str(node["segment_type"]["name"]) != "unknown"
            or int(node["area_pixels"]) >= minimum_unknown_area
        )
        and (
            not _is_topology_fragment_node(node)
            or int(node["area_pixels"]) >= minimum_topology_fragment_area
        )
    }
    displayed_edge_count = sum(
        str(edge["source"]) in displayed_ids
        and str(edge["target"]) in displayed_ids
        and int(edge["shared_boundary_side_count"]) >= minimum_boundary
        for edge in edges
    )
    return {
        "authoritative_graph_is_unchanged": True,
        "filters_are_display_only": True,
        "minimum_display_node_area_pixels": minimum_node_area,
        "minimum_display_unknown_area_pixels": minimum_unknown_area,
        "minimum_display_topology_fragment_area_pixels": minimum_topology_fragment_area,
        "minimum_display_shared_boundary_pixels": minimum_boundary,
        "route_edges_through_actual_contacts": bool(
            config.get("route_edges_through_contacts", True)
        ),
        "show_actual_contact_anchors": bool(config.get("show_contact_anchors", True)),
        "authoritative_node_count": len(nodes),
        "authoritative_edge_count": len(edges),
        "displayed_node_count": len(displayed_ids),
        "displayed_edge_count": int(displayed_edge_count),
        "suppressed_node_count": len(nodes) - len(displayed_ids),
        "suppressed_edge_count": len(edges) - int(displayed_edge_count),
    }


def _nodes_geojson(nodes: list[dict[str, Any]], stage2_result: Any) -> dict[str, Any]:
    preferred_space = "tile_pixel"
    for candidate in ("wgs84", "projected", "source_pixel", "tile_pixel"):
        if all(node["geometry"]["coordinate_spaces"][candidate]["available"] for node in nodes):
            preferred_space = candidate
            break
    features = []
    for node in nodes:
        spatial = node["geometry"]["coordinate_spaces"][preferred_space]
        features.append(
            {
                "type": "Feature",
                "id": node["node_id"],
                "geometry": copy.deepcopy(spatial["geometry"]),
                "properties": {
                    "node_id": node["node_id"],
                    "tile_id": node["tile_id"],
                    "source_image_id": node["source_image_id"],
                    "segment_id": node["segment_id"],
                    "instance_id": node["instance_id"],
                    "class_id": node["segment_type"]["id"],
                    "class_name": node["segment_type"]["name"],
                    "geometry_type": node["geometry_type"],
                    "area_pixels": node["area_pixels"],
                    "degree": node["degree"],
                    "touches_tile_boundary": node["touches_tile_boundary"],
                    "coordinate_space": preferred_space,
                    "crs": spatial.get("crs"),
                },
            }
        )
    crs = None
    if nodes:
        crs = nodes[0]["geometry"]["coordinate_spaces"][preferred_space].get("crs")
    return {
        "type": "FeatureCollection",
        "schema_version": SCHEMA_VERSION,
        "coordinate_space": preferred_space,
        "crs": crs,
        "source_crs": stage2_result.tile_metadata.get("crs"),
        "features": features,
    }


def _comparison_graph_counts(
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    unknown_nodes = [
        node
        for node in nodes
        if str(node.get("segment_type", {}).get("name", "")) == "unknown"
    ]
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "unknown_node_count": len(unknown_nodes),
        "unknown_pixel_count": sum(
            int(node.get("area_pixels", 0)) for node in unknown_nodes
        ),
    }


def _input_checksum_index(
    paths: Sequence[Path], *, anchor: Path
) -> dict[str, str]:
    """Index checksums without collisions between 12_final and refinement."""

    root = anchor.expanduser().resolve()
    result: dict[str, str] = {}
    for path in paths:
        resolved = path.expanduser().resolve()
        try:
            key = resolved.relative_to(root).as_posix()
        except ValueError:
            key = str(resolved)
        if key in result:
            raise ValueError(f"Duplicate Stage 3 input checksum key: {key!r}")
        result[key] = checksum(resolved)
    return result


def _save_visualizations(
    stage2_directory: Path,
    output_directory: Path,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    image_shape: tuple[int, int],
    config: Mapping[str, Any],
    *,
    before_graph: _BeforeRefinementGraph,
) -> dict[str, Path]:
    from .visualization import (
        GraphStyle,
        save_graph_overlay,
        save_graph_overview,
        save_graph_png,
        save_graph_refinement_comparison,
        save_graph_svg,
    )

    edge_alpha = float(config["edge_alpha"])
    style = GraphStyle(
        node_size_min=float(config["node_size_min"]),
        node_size_max=float(config["node_size_max"]),
        edge_width_min=float(config["edge_width_min"]),
        edge_width_max=float(config["edge_width_max"]),
        edge_alpha_min=edge_alpha * 0.5,
        edge_alpha_max=edge_alpha,
        minimum_display_node_area_pixels=float(
            config.get("minimum_display_node_area_pixels", 0)
        ),
        minimum_display_unknown_area_pixels=float(
            config.get("minimum_display_unknown_area_pixels", 0)
        ),
        minimum_display_topology_fragment_area_pixels=float(
            config.get("minimum_display_topology_fragment_area_pixels", 0)
        ),
        minimum_display_shared_boundary_pixels=float(
            config.get("minimum_display_shared_boundary_pixels", 0)
        ),
        route_edges_through_contacts=bool(
            config.get("route_edges_through_contacts", True)
        ),
        show_contact_anchors=bool(config.get("show_contact_anchors", True)),
    )
    show_ids = bool(config["show_ids"])
    background = stage2_directory / "refined_overlay.png"
    if not background.is_file():
        background = stage2_directory / "refined_segmentation.png"
    if not background.is_file():
        raise FileNotFoundError(
            f"Refinement visualization background is missing: {stage2_directory}"
        )
    before_semantic = before_graph.directory / "final_segmentation.png"
    after_semantic = stage2_directory / "refined_segmentation.png"
    if not before_semantic.is_file() or not after_semantic.is_file():
        raise FileNotFoundError(
            "Semantic backgrounds for the refinement graph comparison are missing"
        )
    return {
        "graph_overlay": save_graph_overlay(
            background,
            nodes,
            edges,
            output_directory / "graph_overlay.png",
            style=style,
            show_labels=show_ids,
            show_legend=False,
        ),
        "graph_spatial": save_graph_png(
            nodes,
            edges,
            output_directory / "graph_spatial.png",
            image_shape=image_shape,
            style=style,
            show_labels=show_ids,
            show_legend=True,
        ),
        "graph_spatial_svg": save_graph_svg(
            nodes,
            edges,
            output_directory / "graph_spatial.svg",
            image_shape=image_shape,
            style=style,
            show_labels=show_ids,
            show_legend=True,
        ),
        "overview": save_graph_overview(
            background,
            nodes,
            edges,
            output_directory / "stage3_overview.png",
            style=style,
            show_labels=show_ids,
            show_legend=True,
        ),
        "refinement_graph_before_after": save_graph_refinement_comparison(
            before_semantic,
            before_graph.nodes,
            before_graph.edges,
            after_semantic,
            nodes,
            edges,
            output_directory / "refinement_graph_before_after.png",
            style=style,
            show_labels=show_ids,
            show_legend=True,
        ),
    }


def _output_directory(
    project_root: Path, tile_id: str, stage3_config: Mapping[str, Any]
) -> Path:
    output_root = _resolved_root(project_root, str(stage3_config["output"]["root_directory"]))
    return _safe_tile_directory(output_root, tile_id) / "01_graph"


def _resolved_root(project_root: Path, configured_path: str) -> Path:
    path = Path(configured_path).expanduser()
    return (path if path.is_absolute() else project_root / path).resolve()


def _safe_tile_directory(root: Path, tile_id: str) -> Path:
    if not isinstance(tile_id, str) or not tile_id.strip():
        raise ValueError("tile_id must be a non-empty string")
    candidate = (root / tile_id).resolve()
    if root != candidate and root not in candidate.parents:
        raise ValueError(f"tile_id escapes the configured artifact root: {tile_id!r}")
    return candidate


__all__ = ["Stage3Result", "load_stage3_result", "run_stage3"]
