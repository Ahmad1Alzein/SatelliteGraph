"""One-call, two-model Stage 2 execution and final artifact persistence."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from tqdm import tqdm

from src.config import DEFAULT_CONFIG, instance_config, load_config, semantic_config, stable_hash, validate_config

from .artifacts import atomic_json, atomic_npz, manifest, mark_downstream_stale, now, require_fresh
from .backend import SegmentationBackend
from .classes import CLASS_NAMES, NODATA_CLASS_ID, class_registry
from .fusion import fuse_segmentations
from .phases import run_all as run_instance_phases
from .semantic import SemanticBackend, create_semantic_backend, timed_semantic_inference
from .storage import load_masks, save_masks
from .visualization import save_image


SCHEMA_VERSION = "satellitegraph.stage2.full-segmentation.v1"


@dataclass(slots=True)
class Stage2Result:
    """Loaded final Stage 2 result returned by :func:`run_stage2`."""

    tile_id: str
    output_directory: Path
    masks: list[np.ndarray]
    segments: list[dict[str, Any]]
    instance_map: np.ndarray
    semantic_map: np.ndarray
    confidence_map: np.ndarray
    source_map: np.ndarray
    valid_mask: np.ndarray
    tile_metadata: dict[str, Any]
    report: dict[str, Any]
    artifacts: dict[str, str]
    phases: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        """Return the JSON-safe portion used by the command-line interface."""

        return {
            "status": "complete",
            "schema_version": SCHEMA_VERSION,
            "tile_id": self.tile_id,
            "output_directory": str(self.output_directory),
            "segment_count": len(self.segments),
            "semantic_class_counts": self.report.get("semantic_class_counts", {}),
            "coverage": self.report.get("coverage", {}),
            "coordinate_reference": self.report.get("coordinate_reference", {}),
            "artifacts": self.artifacts,
        }


def run_stage2(
    tile_id: str,
    *,
    project_root: str | Path = ".",
    config: Mapping[str, Any] | str | Path | None = None,
    sam_backend: SegmentationBackend | None = None,
    semantic_backend: SemanticBackend | None = None,
    show_progress: bool = True,
) -> Stage2Result:
    """Run SAM 2 and OpenEarthMap, fuse them, and return one final segmentation.

    SAM 2 supplies class-agnostic geometry. OpenEarthMap supplies all named
    semantic labels. The fusion step resolves overlaps deterministically and
    assigns every valid pixel exactly once, including trees and an explicit
    ``unknown`` fallback for low-confidence pixels.
    """

    root = Path(project_root).expanduser().resolve()
    if config is None:
        project_cfg = load_config(DEFAULT_CONFIG)
    elif isinstance(config, (str, Path)):
        project_cfg = load_config(config)
    else:
        project_cfg = copy.deepcopy(dict(config))
        validate_config(project_cfg)

    sam_cfg = instance_config(project_cfg)
    semantic_cfg = semantic_config(project_cfg)
    progress = _Stage2Progress(enabled=show_progress)
    try:
        sam_phases = run_instance_phases(
            sam_cfg,
            root,
            tile_id,
            sam_backend,
            progress_callback=progress.event,
        )
        canonical_tile_id = str(sam_phases[0]["tile_id"])
        progress.event("infer-semantic", "started")
        try:
            semantic_phase = _run_semantic_phase(
                project_cfg,
                sam_cfg,
                semantic_cfg,
                root,
                canonical_tile_id,
                semantic_backend,
            )
        except Exception:
            progress.event("infer-semantic", "failed")
            raise
        progress.event("infer-semantic", "completed")
        progress.event("fuse-export", "started")
        try:
            fusion_phase = _run_fusion_phase(
                project_cfg,
                sam_cfg,
                semantic_cfg,
                root,
                canonical_tile_id,
            )
        except Exception:
            progress.event("fuse-export", "failed")
            raise
        progress.event("fuse-export", "completed")
        progress.complete()
    finally:
        progress.close()
    return load_stage2_result(
        fusion_phase["output_directory"],
        phases={"sam2": sam_phases, "openearthmap": semantic_phase, "fusion": fusion_phase},
    )


class _Stage2Progress:
    """Small tqdm adapter that shows the currently executing Stage 2 phase."""

    LABELS = {
        "validate-tile": "Validate tile",
        "prepare-input": "Prepare model input",
        "infer-raw": "SAM 2 inference",
        "normalize-masks": "Normalize SAM masks",
        "filter-masks": "Filter SAM masks",
        "deduplicate-masks": "Deduplicate SAM masks",
        "analyze-coverage": "Analyze SAM coverage",
        "visualize-instance": "Render SAM audit previews",
        "finalize-instance": "Finalize SAM candidates",
        "infer-semantic": "OpenEarthMap inference",
        "fuse-export": "Fuse models and export final result",
    }

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.current = "Starting"
        self.bar = tqdm(
            total=len(self.LABELS),
            desc="Stage 2 | Starting",
            unit="phase",
            dynamic_ncols=True,
            disable=not enabled,
        )

    def event(self, phase: str, status: str) -> None:
        label = self.LABELS.get(phase, phase)
        self.current = label
        if status == "started":
            self.bar.set_description_str(f"Stage 2 | {label}", refresh=True)
        elif status == "completed":
            self.bar.update(1)
        elif status == "failed":
            self.bar.set_description_str(f"Stage 2 | FAILED: {label}", refresh=True)

    def complete(self) -> None:
        self.bar.set_description_str("Stage 2 | Complete", refresh=True)

    def close(self) -> None:
        self.bar.close()


def load_stage2_result(
    output_directory: str | Path,
    *,
    phases: dict[str, Any] | None = None,
) -> Stage2Result:
    """Load a previously finalized result without rerunning either model."""

    directory = Path(output_directory).expanduser().resolve()
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Final Stage 2 manifest not found: {manifest_path}")
    artifact_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if artifact_manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported Stage 2 result schema: {artifact_manifest.get('schema_version')!r}")
    masks, segments = load_masks(directory / "segments.npz", directory / "segments.json")
    with np.load(directory / "segmentation.npz", allow_pickle=False) as data:
        instance_map = data["instance_map"].astype(np.int32)
        semantic_map = data["semantic_map"].astype(np.uint8)
        confidence_map = data["confidence_map"].astype(np.float32)
        source_map = data["source_map"].astype(np.uint8)
        valid_mask = data["valid_mask"].astype(bool)
    tile_metadata = json.loads((directory / "tile_metadata.json").read_text(encoding="utf-8"))
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    _validate_loaded_result(masks, segments, instance_map, semantic_map, valid_mask)
    return Stage2Result(
        tile_id=str(artifact_manifest["tile_id"]),
        output_directory=directory,
        masks=masks,
        segments=segments,
        instance_map=instance_map,
        semantic_map=semantic_map,
        confidence_map=confidence_map,
        source_map=source_map,
        valid_mask=valid_mask,
        tile_metadata=tile_metadata,
        report=report,
        artifacts={key: str(value) for key, value in artifact_manifest["artifacts"].items()},
        phases=phases or {},
    )


def _run_semantic_phase(
    project_cfg: dict[str, Any],
    sam_cfg: dict[str, Any],
    cfg: dict[str, Any],
    project_root: Path,
    tile_id: str,
    backend: SemanticBackend | None,
) -> dict[str, Any]:
    started = now()
    root = _output_root(sam_cfg, project_root)
    prep = root / tile_id / "02_preprocessing"
    image_path = prep / "sam_input.npy"
    valid_path = prep / "valid_pixels.npz"
    require_fresh(image_path)
    require_fresh(valid_path)
    image = np.load(image_path, allow_pickle=False)
    valid = np.load(valid_path, allow_pickle=False)["valid_pixels"].astype(bool)
    backend = backend or create_semantic_backend(cfg, project_root)
    result, inference_report = timed_semantic_inference(backend, image)
    label_map = np.asarray(result["label_map"], dtype=np.uint8)
    confidence = np.asarray(result["confidence"], dtype=np.float32)
    if label_map.shape != valid.shape or confidence.shape != valid.shape:
        raise ValueError(
            f"OpenEarthMap output shape label={label_map.shape}, confidence={confidence.shape} "
            f"does not match tile {valid.shape}"
        )
    if np.any(label_map >= len(CLASS_NAMES)):
        raise ValueError("OpenEarthMap emitted a class id outside its eight-class model contract")
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("OpenEarthMap confidence must contain finite values between 0 and 1")

    directory = root / tile_id / "11_semantic"
    data_path = directory / "openearthmap.npz"
    arrays: dict[str, np.ndarray] = {"label_map": label_map, "confidence": confidence.astype(np.float16)}
    probabilities = result.get("probabilities")
    if cfg["inference"].get("save_probabilities", True) and probabilities is not None:
        probabilities = np.asarray(probabilities, dtype=np.float32)
        if probabilities.shape != (len(CLASS_NAMES), *valid.shape):
            raise ValueError(
                f"OpenEarthMap probability shape {probabilities.shape} does not match "
                f"({len(CLASS_NAMES)}, {valid.shape[0]}, {valid.shape[1]})"
            )
        arrays["probabilities"] = probabilities.astype(np.float16)
    atomic_npz(data_path, **arrays)
    report = {
        **inference_report,
        "schema_version": SCHEMA_VERSION,
        "tile_id": tile_id,
        "class_names": list(CLASS_NAMES),
        "shape": list(valid.shape),
        "valid_pixel_count": int(valid.sum()),
        "class_pixel_counts": {
            name: int(((label_map == class_id) & valid).sum())
            for class_id, name in enumerate(CLASS_NAMES)
        },
    }
    report_path = directory / "report.json"
    atomic_json(report_path, report)
    preview_path = directory / "semantic_preview.png"
    save_image(preview_path, _semantic_preview(label_map, valid))
    outputs = [data_path, report_path, preview_path]
    phase_manifest = manifest(
        directory / "manifest.json",
        "11_semantic",
        [image_path, valid_path],
        outputs,
        stable_hash({"semantic_model": project_cfg["stage2"]["semantic_model"]}),
        started,
    )
    phase_manifest["invalidated_phases"] = mark_downstream_stale(directory.parent, "11_semantic")
    atomic_json(directory / "manifest.json", phase_manifest)
    return {**report, "output_directory": str(directory.resolve())}


def _run_fusion_phase(
    project_cfg: dict[str, Any],
    sam_cfg: dict[str, Any],
    semantic_cfg: dict[str, Any],
    project_root: Path,
    tile_id: str,
) -> dict[str, Any]:
    started = now()
    root = _output_root(sam_cfg, project_root)
    tile_root = root / tile_id
    inputs = [
        tile_root / "10_final" / "segments.npz",
        tile_root / "10_final" / "segments.json",
        tile_root / "11_semantic" / "openearthmap.npz",
        tile_root / "02_preprocessing" / "valid_pixels.npz",
        tile_root / "01_validation" / "tile_metadata.normalized.json",
    ]
    for path in inputs:
        require_fresh(path)
    sam_masks, sam_records = load_masks(inputs[0], inputs[1])
    with np.load(inputs[2], allow_pickle=False) as data:
        label_map = data["label_map"].astype(np.uint8)
        confidence = data["confidence"].astype(np.float32)
    valid = np.load(inputs[3], allow_pickle=False)["valid_pixels"].astype(bool)
    tile_metadata = json.loads(inputs[4].read_text(encoding="utf-8"))
    fusion_cfg = project_cfg["stage2"]["fusion"]
    masks, records, instance_map, semantic_map, confidence_map, source_map, fusion_report = fuse_segmentations(
        sam_masks,
        sam_records,
        label_map,
        confidence,
        valid,
        tile_metadata,
        fusion_cfg,
    )

    directory = tile_root / "12_final"
    segments_npz, segments_json = directory / "segments.npz", directory / "segments.json"
    save_masks(segments_npz, segments_json, masks, records)
    # Reload so the Python result and persisted records share the exact schema,
    # including mask_index values assigned by save_masks.
    masks, records = load_masks(segments_npz, segments_json)
    maps_path = directory / "segmentation.npz"
    atomic_npz(
        maps_path,
        instance_map=instance_map.astype(np.int32),
        semantic_map=semantic_map.astype(np.uint8),
        confidence_map=confidence_map.astype(np.float16),
        source_map=source_map.astype(np.uint8),
        valid_mask=valid.astype(bool),
    )
    metadata_path = directory / "tile_metadata.json"
    registry_path = directory / "class_registry.json"
    atomic_json(metadata_path, tile_metadata)
    atomic_json(registry_path, {"schema_version": SCHEMA_VERSION, "classes": class_registry()})

    raster_paths = _write_geotiffs(directory, instance_map, semantic_map, tile_metadata)
    geojson_path = _write_geojson(directory, instance_map, records, tile_metadata)
    preview_path = directory / "final_segmentation.png"
    overlay_path = directory / "final_overlay.png"
    preview = _semantic_preview(semantic_map, valid)
    save_image(preview_path, preview)
    image = np.load(tile_root / "02_preprocessing" / "sam_input.npy", allow_pickle=False)
    overlay = image.copy()
    overlay[valid] = np.rint(0.55 * image[valid] + 0.45 * preview[valid]).astype(np.uint8)
    boundaries = _instance_boundaries(instance_map)
    overlay[boundaries] = (255, 255, 255)
    save_image(overlay_path, overlay)

    class_counts: dict[str, int] = {}
    for record in records:
        name = str(record.get("label", {}).get("name", "unknown"))
        class_counts[name] = class_counts.get(name, 0) + 1
    report = {
        **fusion_report,
        "schema_version": SCHEMA_VERSION,
        "tile_id": tile_id,
        "segment_count": len(records),
        "semantic_class_counts": class_counts,
        "coverage": {
            "valid_pixel_count": int(valid.sum()),
            "assigned_valid_pixel_count": int(((instance_map > 0) & valid).sum()),
            "unassigned_valid_pixel_count": int(((instance_map == 0) & valid).sum()),
            "coverage_percentage": float(100 * ((instance_map > 0) & valid).sum() / max(1, valid.sum())),
            "overlap_pixel_count": 0,
        },
        "coordinate_reference": _coordinate_reference(tile_metadata),
        "model_roles": {
            "geometry": "SAM 2 supplies class-agnostic candidate boundaries; OpenEarthMap supplies geometry for uncovered pixels.",
            "labels": "All eight named land-cover labels come from OpenEarthMap FasterSeg. SAM 2 does not predict class names.",
            "fallback": "The Stage 2 fusion rule labels remaining low-confidence valid pixels as unknown.",
        },
    }
    report_path = directory / "report.json"
    atomic_json(report_path, report)
    config_hash = stable_hash(project_cfg["stage2"])
    artifacts = {
        "segments": str(segments_json.resolve()),
        "masks": str(segments_npz.resolve()),
        "segmentation_maps": str(maps_path.resolve()),
        "class_registry": str(registry_path.resolve()),
        "tile_metadata": str(metadata_path.resolve()),
        "report": str(report_path.resolve()),
        "geojson": str(geojson_path.resolve()),
        "instance_geotiff": str(raster_paths["instance"].resolve()),
        "semantic_geotiff": str(raster_paths["semantic"].resolve()),
        "preview": str(preview_path.resolve()),
        "overlay": str(overlay_path.resolve()),
    }
    artifact_manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "tile_id": tile_id,
        "created_at": now(),
        "configuration_hash": config_hash,
        "models": {
            "instance_geometry": {
                "name": "SAM 2.1",
                "backend": sam_cfg["backend"],
                "model_type": sam_cfg.get("model_type"),
                "checkpoint": sam_cfg.get("checkpoint"),
                "semantic_labels": False,
            },
            "semantic_labels": {
                "name": "OpenEarthMap FasterSeg",
                "backend": semantic_cfg["backend"],
                "checkpoint": semantic_cfg.get("checkpoint"),
                "class_names": list(CLASS_NAMES),
            },
        },
        "coordinate_reference": report["coordinate_reference"],
        "artifacts": artifacts,
    }
    artifact_manifest_path = directory / "manifest.json"
    atomic_json(artifact_manifest_path, artifact_manifest)
    outputs = [
        segments_npz,
        segments_json,
        maps_path,
        metadata_path,
        registry_path,
        report_path,
        geojson_path,
        *raster_paths.values(),
        preview_path,
        overlay_path,
        artifact_manifest_path,
    ]
    phase_manifest = manifest(
        directory / "phase_manifest.json",
        "12_final",
        inputs,
        outputs,
        config_hash,
        started,
    )
    atomic_json(directory / "phase_manifest.json", phase_manifest)
    _validate_loaded_result(masks, records, instance_map, semantic_map, valid)
    return {**report, "output_directory": str(directory.resolve()), "artifacts": artifacts}


def _output_root(cfg: dict[str, Any], project_root: Path) -> Path:
    path = Path(cfg["output"]["root_directory"])
    return path if path.is_absolute() else project_root / path


def _semantic_preview(labels: np.ndarray, valid: np.ndarray) -> np.ndarray:
    palette = np.asarray(
        [
            (170, 120, 80),
            (196, 185, 110),
            (190, 190, 190),
            (245, 215, 40),
            (45, 145, 55),
            (55, 125, 220),
            (215, 175, 65),
            (220, 70, 70),
            (115, 80, 145),
        ],
        dtype=np.uint8,
    )
    output = np.zeros((*labels.shape, 3), dtype=np.uint8)
    known = valid & (labels < len(palette))
    output[known] = palette[labels[known]]
    return output


def _instance_boundaries(instance_map: np.ndarray) -> np.ndarray:
    boundaries = np.zeros(instance_map.shape, bool)
    boundaries[1:, :] |= instance_map[1:, :] != instance_map[:-1, :]
    boundaries[:, 1:] |= instance_map[:, 1:] != instance_map[:, :-1]
    return boundaries & (instance_map > 0)


def _write_geotiffs(
    directory: Path,
    instance_map: np.ndarray,
    semantic_map: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Path]:
    import rasterio
    from affine import Affine

    directory.mkdir(parents=True, exist_ok=True)
    coefficients = metadata.get("tile_affine_transform")
    if coefficients:
        transform = Affine(*coefficients[:6])
        crs = metadata.get("crs_wkt") or metadata.get("crs")
    else:
        transform = Affine.translation(float(metadata.get("global_x0") or 0), float(metadata.get("global_y0") or 0))
        crs = None
    paths = {"instance": directory / "instance_map.tif", "semantic": directory / "semantic_map.tif"}
    for path, array, dtype, nodata in (
        (paths["instance"], instance_map.astype(np.int32), "int32", 0),
        (paths["semantic"], semantic_map.astype(np.uint8), "uint8", NODATA_CLASS_ID),
    ):
        temporary = path.with_suffix(".tmp.tif")
        with rasterio.open(
            temporary,
            "w",
            driver="GTiff",
            height=array.shape[0],
            width=array.shape[1],
            count=1,
            dtype=dtype,
            crs=crs,
            transform=transform,
            nodata=nodata,
            compress="deflate",
        ) as dataset:
            dataset.write(array, 1)
            dataset.update_tags(schema_version=SCHEMA_VERSION, tile_id=str(metadata["tile_id"]))
        temporary.replace(path)
    return paths


def _write_geojson(
    directory: Path,
    instance_map: np.ndarray,
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> Path:
    from affine import Affine
    from rasterio.features import shapes
    from rasterio.warp import transform_geom

    coefficients = metadata.get("tile_affine_transform")
    source_crs = metadata.get("crs_wkt") or metadata.get("crs")
    if coefficients:
        transform = Affine(*coefficients[:6])
        coordinate_space = str(metadata.get("crs") or "source_crs")
    else:
        transform = Affine.translation(float(metadata.get("global_x0") or 0), float(metadata.get("global_y0") or 0))
        coordinate_space = "source_image_pixel"
    grouped: dict[int, list[dict[str, Any]]] = {}
    for geometry, value in shapes(instance_map.astype(np.int32), mask=instance_map > 0, transform=transform):
        if source_crs:
            try:
                geometry = transform_geom(source_crs, "EPSG:4326", geometry, precision=9)
                coordinate_space = "EPSG:4326"
            except Exception:
                # Keep truthful source-CRS coordinates when the installed PROJ
                # database cannot transform a custom legacy CRS.
                pass
        grouped.setdefault(int(value), []).append(geometry)
    features = []
    for instance_id, record in enumerate(records, 1):
        geometries = grouped.get(instance_id, [])
        geometry = _combine_geometries(geometries)
        label = record.get("label", {})
        features.append(
            {
                "type": "Feature",
                "id": instance_id,
                "geometry": geometry,
                "properties": {
                    "instance_id": instance_id,
                    "segment_id": record["segment_id"],
                    "label_id": label.get("id"),
                    "label": label.get("name"),
                    "label_source": label.get("source_model"),
                    "geometry_source": record.get("geometry_source", {}).get("model"),
                    "area_pixels": record.get("area_pixels"),
                },
            }
        )
    collection = {
        "type": "FeatureCollection",
        "schema_version": SCHEMA_VERSION,
        "coordinate_space": coordinate_space,
        "source_crs": metadata.get("crs"),
        "features": features,
    }
    path = directory / "segments.geojson"
    atomic_json(path, collection)
    return path


def _combine_geometries(geometries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not geometries:
        return None
    if len(geometries) == 1:
        return geometries[0]
    polygons: list[Any] = []
    for geometry in geometries:
        if geometry["type"] == "Polygon":
            polygons.append(geometry["coordinates"])
        elif geometry["type"] == "MultiPolygon":
            polygons.extend(geometry["coordinates"])
    return {"type": "MultiPolygon", "coordinates": polygons}


def _coordinate_reference(metadata: dict[str, Any]) -> dict[str, Any]:
    available = bool(metadata.get("tile_affine_transform") and metadata.get("crs"))
    return {
        "tile_pixel": {
            "available": True,
            "origin": "upper-left",
            "axes": {"x": "column", "y": "row"},
            "indexing": "zero-based",
        },
        "source_pixel": {
            "available": metadata.get("global_x0") is not None and metadata.get("global_y0") is not None,
            "offset": {"x": metadata.get("global_x0"), "y": metadata.get("global_y0")},
        },
        "projected": {
            "available": available,
            "crs": metadata.get("crs"),
            "reason_unavailable": None if available else metadata.get("georeferencing_unavailable_reason"),
        },
        "geographic": {
            "available": bool(metadata.get("geographic_bounds")),
            "crs": "EPSG:4326" if metadata.get("geographic_bounds") else None,
        },
        "bbox_convention": "[x, y, width, height] with exclusive maximum edges",
        "centroid_convention": "pixel centers",
    }


def _validate_loaded_result(
    masks: list[np.ndarray],
    records: list[dict[str, Any]],
    instance_map: np.ndarray,
    semantic_map: np.ndarray,
    valid: np.ndarray,
) -> None:
    if len(masks) != len(records):
        raise ValueError("Final mask and segment counts differ")
    if instance_map.shape != valid.shape or semantic_map.shape != valid.shape:
        raise ValueError("Final maps and valid mask have different shapes")
    if np.any(instance_map[~valid] != 0) or np.any(semantic_map[~valid] != NODATA_CLASS_ID):
        raise ValueError("Invalid pixels must be instance 0 and semantic nodata")
    if np.any(instance_map[valid] <= 0):
        raise ValueError("Full Stage 2 result contains unassigned valid pixels")
    expected_ids = np.arange(1, len(records) + 1, dtype=np.int32)
    actual_ids = np.unique(instance_map[valid])
    if not np.array_equal(actual_ids, expected_ids):
        raise ValueError(f"Instance ids are not contiguous: expected {expected_ids.tolist()}, got {actual_ids.tolist()}")
    for instance_id, (mask, record) in enumerate(zip(masks, records), 1):
        if not np.array_equal(mask, instance_map == instance_id):
            raise ValueError(f"Mask data disagrees with instance map for {record.get('segment_id')}")
        label_id = int(record["label"]["id"])
        if np.any(semantic_map[mask] != label_id):
            raise ValueError(f"Semantic map disagrees with segment label for {record.get('segment_id')}")
