from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.stage2.artifacts import atomic_json, atomic_npz, manifest, mark_downstream_stale, now, require_fresh
from src.stage2.storage import load_masks, save_masks
from src.stage2.visualization import comparison, save_image

from .backend import LandcoverBackend, create_landcover_backend, timed_inference
from .completion import complete_non_tree_landscape
from .config import CLASS_NAMES, config_hash, relevant_config
from .visualization import binary_overlay, completed_overlay, landcover_map, landcover_overlay, semantic_label_map


def _resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _root(cfg: dict[str, Any], project_root: Path) -> Path:
    return _resolve(project_root, cfg["output"]["root_directory"])


def _stage2_root(cfg: dict[str, Any], project_root: Path) -> Path:
    return _resolve(project_root, cfg["stage2_root"])


def _finish(
    cfg: dict[str, Any],
    project_root: Path,
    tile_id: str,
    phase: str,
    inputs: list[Path],
    outputs: list[Path],
    started: str,
) -> dict[str, Any]:
    directory = _root(cfg, project_root) / tile_id / phase
    result = manifest(directory / "manifest.json", phase, inputs, outputs, config_hash(cfg, phase), started)
    result["invalidated_phases"] = mark_downstream_stale(directory.parent, phase)
    atomic_json(directory / "manifest.json", result)
    return result


def check_model_phase(
    cfg: dict[str, Any],
    project_root: Path,
    tile_id: str | None = None,
    backend: LandcoverBackend | None = None,
) -> dict[str, Any]:
    del tile_id
    started = now()
    backend = backend or create_landcover_backend(cfg)
    backend.load()
    report = backend.health_check()
    directory = _root(cfg, project_root) / "_model" / "01_model"
    path = directory / "health_check.json"
    atomic_json(path, report)
    inputs = [Path(cfg[key]) for key in ("architecture", "checkpoint") if cfg.get(key)]
    manifest(directory / "manifest.json", "01_model", inputs, [path], config_hash(cfg, "01_model"), started)
    return {**report, "output_directory": str(directory)}


def infer_landcover_phase(
    cfg: dict[str, Any],
    project_root: Path,
    tile_id: str,
    backend: LandcoverBackend | None = None,
) -> dict[str, Any]:
    started = now()
    prep = _stage2_root(cfg, project_root) / tile_id / "02_preprocessing"
    image_path, valid_path = prep / "sam_input.npy", prep / "valid_pixels.npz"
    require_fresh(image_path)
    require_fresh(valid_path)
    image = np.load(image_path, allow_pickle=False)
    backend = backend or create_landcover_backend(cfg)
    result, report = timed_inference(backend, image)
    valid = np.load(valid_path, allow_pickle=False)["valid_pixels"].astype(bool)
    if result["label_map"].shape != valid.shape:
        raise ValueError(f"OpenEarthMap output shape {result['label_map'].shape} does not match tile {valid.shape}")
    directory = _root(cfg, project_root) / tile_id / "02_landcover"
    data_path = directory / "landcover.npz"
    arrays = {
        "label_map": result["label_map"].astype(np.uint8),
        "confidence": result["confidence"].astype(np.float16),
    }
    if cfg["inference"].get("save_probabilities", True):
        arrays["probabilities"] = result["probabilities"].astype(np.float16)
    atomic_npz(data_path, **arrays)
    report.update(
        class_names=CLASS_NAMES,
        shape=list(valid.shape),
        valid_pixel_count=int(valid.sum()),
        class_pixel_counts={name: int(((result["label_map"] == index) & valid).sum()) for index, name in enumerate(CLASS_NAMES)},
    )
    report_path = directory / "landcover_report.json"
    atomic_json(report_path, report)
    preview = directory / "landcover_preview.png"
    overlay_path = directory / "landcover_overlay.png"
    save_image(preview, landcover_map(result["label_map"], valid))
    save_image(overlay_path, landcover_overlay(image, result["label_map"], valid, cfg["visualization"]["alpha"]))
    outputs = [data_path, report_path, preview, overlay_path]
    _finish(cfg, project_root, tile_id, "02_landcover", [image_path, valid_path], outputs, started)
    return {**report, "output_directory": str(directory)}


def complete_phase(cfg: dict[str, Any], project_root: Path, tile_id: str) -> dict[str, Any]:
    started = now()
    stage2 = _stage2_root(cfg, project_root) / tile_id
    landcover = _root(cfg, project_root) / tile_id / "02_landcover"
    inputs = [
        stage2 / "10_final" / "segments.npz",
        stage2 / "10_final" / "segments.json",
        stage2 / "02_preprocessing" / "valid_pixels.npz",
        landcover / "landcover.npz",
    ]
    for path in inputs:
        require_fresh(path)
    sam_masks, sam_records = load_masks(inputs[0], inputs[1])
    valid = np.load(inputs[2], allow_pickle=False)["valid_pixels"].astype(bool)
    with np.load(inputs[3], allow_pickle=False) as data:
        label_map = data["label_map"].astype(np.uint8)
        confidence = data["confidence"].astype(np.float32)
    masks, records, tree_mask, report = complete_non_tree_landscape(
        sam_masks, sam_records, label_map, confidence, valid, cfg["completion"]
    )
    directory = _root(cfg, project_root) / tile_id / "03_completion"
    mask_path, json_path = directory / "completed_masks.npz", directory / "completed_masks.json"
    save_masks(mask_path, json_path, masks, records)
    atomic_npz(directory / "tree_exclusion_mask.npz", tree_mask=tree_mask)
    atomic_json(directory / "completion_report.json", report)
    outputs = [mask_path, json_path, directory / "tree_exclusion_mask.npz", directory / "completion_report.json"]
    _finish(cfg, project_root, tile_id, "03_completion", inputs, outputs, started)
    return {**report, "output_directory": str(directory)}


def visualize_phase(cfg: dict[str, Any], project_root: Path, tile_id: str) -> dict[str, Any]:
    started = now()
    stage2_prep = _stage2_root(cfg, project_root) / tile_id / "02_preprocessing"
    source = _root(cfg, project_root) / tile_id
    inputs = [
        stage2_prep / "sam_input.npy",
        stage2_prep / "valid_pixels.npz",
        source / "02_landcover" / "landcover.npz",
        source / "03_completion" / "completed_masks.npz",
        source / "03_completion" / "completed_masks.json",
        source / "03_completion" / "tree_exclusion_mask.npz",
    ]
    for path in inputs:
        require_fresh(path)
    image = np.load(inputs[0], allow_pickle=False)
    valid = np.load(inputs[1], allow_pickle=False)["valid_pixels"].astype(bool)
    with np.load(inputs[2], allow_pickle=False) as data:
        labels = data["label_map"].astype(np.uint8)
    masks, records = load_masks(inputs[3], inputs[4])
    tree = np.load(inputs[5], allow_pickle=False)["tree_mask"].astype(bool)
    vis = cfg["visualization"]
    directory = source / "04_visualization"
    paths = {
        "landcover": directory / "landcover_overlay.png",
        "trees": directory / "tree_exclusion_overlay.png",
        "roads": directory / "road_completion_overlay.png",
        "buildings": directory / "building_completion_overlay.png",
        "completed": directory / "completed_overlay.png",
        "labels": directory / "completed_semantic_map.png",
        "comparison": directory / "completion_comparison.png",
    }
    landcover_image = landcover_overlay(image, labels, valid, vis["alpha"])
    tree_image = binary_overlay(image, tree, (255, 0, 255), vis["alpha"])
    road = np.logical_or.reduce([m for m, r in zip(masks, records) if r.get("semantic_class") == "road"]) if any(r.get("semantic_class") == "road" for r in records) else np.zeros(valid.shape, bool)
    buildings = np.logical_or.reduce([m for m, r in zip(masks, records) if r.get("semantic_class") == "buildings"]) if any(r.get("semantic_class") == "buildings" for r in records) else np.zeros(valid.shape, bool)
    road_image = binary_overlay(image, road, (255, 255, 0), vis["alpha"])
    building_image = binary_overlay(image, buildings, (255, 0, 0), vis["alpha"])
    completed = completed_overlay(image, masks, records, vis["alpha"], vis["random_seed"])
    semantic = semantic_label_map(masks, records, valid.shape)
    for path, value in [
        (paths["landcover"], landcover_image),
        (paths["trees"], tree_image),
        (paths["roads"], road_image),
        (paths["buildings"], building_image),
        (paths["completed"], completed),
        (paths["labels"], semantic),
        (paths["comparison"], comparison([image, landcover_image, tree_image, completed])),
    ]:
        save_image(path, value)
    inspection = [{key: record.get(key) for key in ("segment_id", "source", "semantic_class", "area_pixels", "bbox_xywh", "centroid_rc", "predicted_iou", "mean_semantic_confidence", "touches_tile_boundary")} for record in records]
    table = directory / "segment_inspection.json"
    atomic_json(table, {"records": inspection})
    outputs = [*paths.values(), table]
    _finish(cfg, project_root, tile_id, "04_visualization", inputs, outputs, started)
    return {"segment_count": len(records), "output_directory": str(directory), "outputs": [str(path) for path in outputs]}


def finalize_phase(cfg: dict[str, Any], project_root: Path, tile_id: str) -> dict[str, Any]:
    started = now()
    root = _root(cfg, project_root)
    source = root / tile_id
    stage2 = _stage2_root(cfg, project_root) / tile_id
    inputs = [
        source / "03_completion" / "completed_masks.npz",
        source / "03_completion" / "completed_masks.json",
        source / "03_completion" / "tree_exclusion_mask.npz",
        source / "03_completion" / "completion_report.json",
        source / "02_landcover" / "landcover.npz",
        stage2 / "10_final" / "manifest.json",
    ]
    for path in inputs:
        require_fresh(path)
    masks, records = load_masks(inputs[0], inputs[1])
    report = json.loads(inputs[3].read_text(encoding="utf-8"))
    directory = source / "05_final"
    segments_npz, segments_json = directory / "segments.npz", directory / "segments.json"
    save_masks(segments_npz, segments_json, masks, records)
    shutil.copy2(inputs[2], directory / "tree_exclusion_mask.npz")
    shutil.copy2(inputs[4], directory / "landcover.npz")
    visual = source / "04_visualization"
    if (visual / "completed_overlay.png").is_file():
        shutil.copy2(visual / "completed_overlay.png", directory / "final_overlay.png")
    if (visual / "completed_semantic_map.png").is_file():
        shutil.copy2(visual / "completed_semantic_map.png", directory / "final_semantic_map.png")
    final_report = {
        "tile_id": tile_id,
        "segment_count": len(records),
        "tree_pixels_excluded": report["tree_pixel_count"],
        "non_tree_coverage_percentage": report["non_tree_coverage_percentage"],
        "semantic_class_counts": _class_counts(records),
        "completion": report,
        "configuration_hash": config_hash(cfg, "05_final"),
    }
    report_path = directory / "landscape_report.json"
    atomic_json(report_path, final_report)
    final_manifest = {
        "tile_id": tile_id,
        "purpose": "non-tree semantic completion after Stage 2",
        "stage2_manifest": str(inputs[5].resolve()),
        "backend": cfg["backend"],
        "architecture": cfg.get("architecture"),
        "checkpoint": cfg.get("checkpoint"),
        "class_names": CLASS_NAMES,
        "segments": str(segments_json.resolve()),
        "mask_data": str(segments_npz.resolve()),
        "landcover_data": str((directory / "landcover.npz").resolve()),
        "tree_exclusion_mask": str((directory / "tree_exclusion_mask.npz").resolve()),
        "configuration": relevant_config(cfg, "05_final"),
        "configuration_hash": config_hash(cfg, "05_final"),
        "created_at": now(),
        "provenance": [{"segment_id": record["segment_id"], "source": record.get("source"), "source_raw_mask_id": record.get("source_raw_mask_id")} for record in records],
    }
    artifact_manifest = directory / "landscape_manifest.json"
    atomic_json(artifact_manifest, final_manifest)
    outputs = [segments_npz, segments_json, directory / "tree_exclusion_mask.npz", directory / "landcover.npz", report_path, artifact_manifest]
    for optional in (directory / "final_overlay.png", directory / "final_semantic_map.png"):
        if optional.is_file():
            outputs.append(optional)
    phase_manifest = _finish(cfg, project_root, tile_id, "05_final", inputs, outputs, started)
    atomic_json(directory / "phase_manifest.json", phase_manifest)
    return {**final_report, "output_directory": str(directory)}


PHASES: dict[str, Callable[..., dict[str, Any]]] = {
    "check-model": check_model_phase,
    "infer-landcover": infer_landcover_phase,
    "complete-non-tree": complete_phase,
    "visualize": visualize_phase,
    "finalize": finalize_phase,
}


def run_all(
    cfg: dict[str, Any],
    project_root: Path,
    tile_id: str,
    backend: LandcoverBackend | None = None,
) -> list[dict[str, Any]]:
    return [
        infer_landcover_phase(cfg, project_root, tile_id, backend),
        complete_phase(cfg, project_root, tile_id),
        visualize_phase(cfg, project_root, tile_id),
        finalize_phase(cfg, project_root, tile_id),
    ]


def _class_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for record in records:
        name = str(record.get("semantic_class", "unknown"))
        result[name] = result.get(name, 0) + 1
    return result
