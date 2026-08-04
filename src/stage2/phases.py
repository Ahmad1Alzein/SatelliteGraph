from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .artifacts import atomic_json, atomic_npz, manifest, mark_downstream_stale, now, phase_dir, require_fresh
from .backend import SegmentationBackend, create_backend, timed_inference
from .config import relevant_config, stable_hash
from .masks import analyze_coverage, deduplicate_masks, filter_masks, normalize_masks
from .preprocessing import prepare_image
from .storage import load_masks, save_masks
from .tiles import validate_tile
from .visualization import boundary_overlay, comparison, coverage_preview, inspection_rows, label_map, overlay, overlap_preview, save_image


def _root(cfg: dict[str, Any], project_root: Path) -> Path:
    root = Path(cfg["output"]["root_directory"])
    return root if root.is_absolute() else project_root / root


def _finish(cfg: dict[str, Any], project_root: Path, tile_id: str, phase: str, inputs: list[Path], outputs: list[Path], started: str) -> dict[str, Any]:
    directory = phase_dir(_root(cfg, project_root), tile_id, phase)
    result = manifest(directory / "manifest.json", phase, inputs, outputs, stable_hash(relevant_config(cfg, phase)), started)
    result["invalidated_phases"] = mark_downstream_stale(directory.parent, phase)
    atomic_json(directory / "manifest.json", result)
    return result


def validate_phase(cfg: dict[str, Any], project_root: Path, tile_id: str) -> dict[str, Any]:
    started = now(); image, metadata, report = validate_tile(project_root, tile_id, cfg.get("_project_paths"))
    tile_id = metadata["tile_id"]; directory = phase_dir(_root(cfg, project_root), tile_id, "01_validation"); directory.mkdir(parents=True, exist_ok=True)
    metadata_path, report_path, preview = directory / "tile_metadata.normalized.json", directory / "validation.json", directory / "tile_preview.png"
    atomic_json(metadata_path, metadata); atomic_json(report_path, report)
    preview_image = _uint8_preview(image); save_image(preview, preview_image)
    _finish(cfg, project_root, tile_id, "01_validation", [Path(metadata["image_path"]), Path(metadata["metadata_path"])], [metadata_path, report_path, preview], started)
    return {"tile_id": tile_id, **report, "output_directory": str(directory)}


def prepare_phase(cfg: dict[str, Any], project_root: Path, tile_id: str) -> dict[str, Any]:
    tile_id = _canonical(cfg, project_root, tile_id); started = now(); previous = phase_dir(_root(cfg, project_root), tile_id, "01_validation")
    metadata_path = previous / "tile_metadata.normalized.json"; require_fresh(metadata_path)
    metadata = _json(metadata_path); image = cv2.imread(metadata["image_path"], cv2.IMREAD_UNCHANGED)
    if image is None: raise ValueError(f"Validated tile is no longer readable: {metadata['image_path']}")
    if image.ndim == 2: image = image[..., None]
    elif image.shape[2] >= 3: image = image[..., [2, 1, 0] + list(range(3, image.shape[2]))]
    rgb, valid, report = prepare_image(image, metadata, cfg["preprocessing"])
    directory = phase_dir(_root(cfg, project_root), tile_id, "02_preprocessing")
    npy, valid_path, report_path = directory / "sam_input.npy", directory / "valid_pixels.npz", directory / "preprocessing.json"
    selected_path = directory / "source_selected_bands.npy"
    band_indices = [int(value) - 1 for value in cfg["preprocessing"]["rgb_bands"]]
    directory.mkdir(parents=True, exist_ok=True); _atomic_npy(npy, rgb); _atomic_npy(selected_path, image[..., band_indices]); atomic_npz(valid_path, valid_pixels=valid)
    atomic_json(report_path, report); save_image(directory / "sam_input.png", rgb); save_image(directory / "comparison_preview.png", comparison([_uint8_preview(image), rgb]))
    outputs = [npy, selected_path, valid_path, report_path, directory / "sam_input.png", directory / "comparison_preview.png"]
    _finish(cfg, project_root, tile_id, "02_preprocessing", [metadata_path, Path(metadata["image_path"])], outputs, started)
    return {**report, "output_directory": str(directory)}


def check_model_phase(cfg: dict[str, Any], project_root: Path, tile_id: str | None = None, backend: SegmentationBackend | None = None) -> dict[str, Any]:
    started = now(); backend = backend or create_backend(cfg, project_root); backend.load(); report = backend.health_check()
    directory = _root(cfg, project_root) / "_model" / "03_model"; path = directory / "health_check.json"; atomic_json(path, report)
    checkpoint = Path(cfg["checkpoint"]) if cfg.get("checkpoint") else None
    if checkpoint is not None and not checkpoint.is_absolute(): checkpoint = project_root / checkpoint
    manifest(directory / "manifest.json", "03_model", [checkpoint] if checkpoint else [], [path], stable_hash(relevant_config(cfg, "03_model")), started)
    return {**report, "output_directory": str(directory)}


def infer_phase(cfg: dict[str, Any], project_root: Path, tile_id: str, backend: SegmentationBackend | None = None) -> dict[str, Any]:
    tile_id = _canonical(cfg, project_root, tile_id); started = now(); prep = phase_dir(_root(cfg, project_root), tile_id, "02_preprocessing")
    image_path, valid_path = prep / "sam_input.npy", prep / "valid_pixels.npz"; require_fresh(image_path); require_fresh(valid_path)
    image = np.load(image_path, allow_pickle=False); backend = backend or create_backend(cfg, project_root)
    raw, report = timed_inference(backend, image, {"tile_id": tile_id})
    masks, records = [], []
    for index, item in enumerate(raw):
        record = {k: v for k, v in item.items() if k != "segmentation"}; record["raw_mask_id"] = f"raw_{index:05d}"; record["source_tile_id"] = tile_id
        segmentation = item["segmentation"]
        if not isinstance(segmentation, np.ndarray):
            from .masks import decode_raw_segmentation
            segmentation = decode_raw_segmentation(segmentation, image.shape[:2])
        masks.append(segmentation.astype(bool)); records.append(record)
    directory = phase_dir(_root(cfg, project_root), tile_id, "04_raw_inference"); npz, js = directory / "raw_masks.npz", directory / "raw_masks.json"
    save_masks(npz, js, masks, records); atomic_json(directory / "inference_config.json", cfg["automatic_mask_generator"]); atomic_json(directory / "inference_report.json", report)
    _finish(cfg, project_root, tile_id, "04_raw_inference", [image_path], [npz, js, directory / "inference_config.json", directory / "inference_report.json"], started)
    return {**report, "output_directory": str(directory)}


def normalize_phase(cfg: dict[str, Any], project_root: Path, tile_id: str) -> dict[str, Any]:
    tile_id = _canonical(cfg, project_root, tile_id); started = now(); rawdir = phase_dir(_root(cfg, project_root), tile_id, "04_raw_inference"); prep = phase_dir(_root(cfg, project_root), tile_id, "02_preprocessing")
    inputs = [rawdir / "raw_masks.npz", rawdir / "raw_masks.json", prep / "valid_pixels.npz"]
    for p in inputs: require_fresh(p)
    masks, records = load_masks(inputs[0], inputs[1]); valid = np.load(inputs[2])["valid_pixels"].astype(bool)
    out_masks, out_records, report = normalize_masks(masks, records, valid.shape, valid, tile_id)
    directory = phase_dir(_root(cfg, project_root), tile_id, "05_normalized"); npz, js = directory / "normalized_masks.npz", directory / "normalized_masks.json"
    save_masks(npz, js, out_masks, out_records); atomic_json(directory / "normalization_report.json", report)
    _finish(cfg, project_root, tile_id, "05_normalized", inputs, [npz, js, directory / "normalization_report.json"], started)
    return {**report, "output_directory": str(directory)}


def filter_phase(cfg: dict[str, Any], project_root: Path, tile_id: str) -> dict[str, Any]:
    tile_id = _canonical(cfg, project_root, tile_id); started = now(); norm = phase_dir(_root(cfg, project_root), tile_id, "05_normalized"); prep = phase_dir(_root(cfg, project_root), tile_id, "02_preprocessing")
    inputs = [norm / "normalized_masks.npz", norm / "normalized_masks.json", prep / "valid_pixels.npz"]
    for p in inputs: require_fresh(p)
    masks, records = load_masks(inputs[0], inputs[1]); valid = np.load(inputs[2])["valid_pixels"].astype(bool)
    accepted_masks, accepted, rejected = filter_masks(masks, records, valid, cfg["filtering"])
    directory = phase_dir(_root(cfg, project_root), tile_id, "06_filtering"); npz, js = directory / "accepted_masks.npz", directory / "accepted_masks.json"
    save_masks(npz, js, accepted_masks, accepted)
    rejected_ids = {record["segment_id"] for record in rejected}
    rejected_masks = [mask for mask, record in zip(masks, records) if record["segment_id"] in rejected_ids]
    rejected_npz, rejected_json = directory / "rejected_masks.npz", directory / "rejected_masks.json"
    save_masks(rejected_npz, rejected_json, rejected_masks, rejected)
    report = {"input_count": len(records), "accepted_count": len(accepted), "rejected_count": len(rejected), "rejection_reason_counts": _reason_counts(rejected)}; atomic_json(directory / "filtering_report.json", report)
    _finish(cfg, project_root, tile_id, "06_filtering", inputs, [npz, js, rejected_npz, rejected_json, directory / "filtering_report.json"], started)
    return {**report, "output_directory": str(directory)}


def deduplicate_phase(cfg: dict[str, Any], project_root: Path, tile_id: str) -> dict[str, Any]:
    tile_id = _canonical(cfg, project_root, tile_id); started = now(); source = phase_dir(_root(cfg, project_root), tile_id, "06_filtering"); inputs = [source / "accepted_masks.npz", source / "accepted_masks.json"]
    for p in inputs: require_fresh(p)
    masks, records = load_masks(*inputs); out_masks, out_records, report = deduplicate_masks(masks, records, cfg["deduplication"])
    directory = phase_dir(_root(cfg, project_root), tile_id, "07_deduplication"); npz, js = directory / "final_candidate_masks.npz", directory / "final_candidate_masks.json"
    save_masks(npz, js, out_masks, out_records); atomic_json(directory / "duplicate_groups.json", report["duplicate_groups"]); atomic_json(directory / "nested_relationships.json", report["nested_relationships"]); atomic_json(directory / "deduplication_report.json", report)
    _finish(cfg, project_root, tile_id, "07_deduplication", inputs, [npz, js, directory / "duplicate_groups.json", directory / "nested_relationships.json", directory / "deduplication_report.json"], started)
    return {**report, "output_directory": str(directory)}


def coverage_phase(cfg: dict[str, Any], project_root: Path, tile_id: str) -> dict[str, Any]:
    tile_id = _canonical(cfg, project_root, tile_id); started = now(); source = phase_dir(_root(cfg, project_root), tile_id, "07_deduplication"); prep = phase_dir(_root(cfg, project_root), tile_id, "02_preprocessing")
    inputs = [source / "final_candidate_masks.npz", source / "final_candidate_masks.json", prep / "valid_pixels.npz"]
    for p in inputs: require_fresh(p)
    masks, records = load_masks(inputs[0], inputs[1]); valid = np.load(inputs[2])["valid_pixels"].astype(bool)
    final_masks, final_records, count_map, uncovered, report = analyze_coverage(masks, records, valid, cfg["coverage"])
    directory = phase_dir(_root(cfg, project_root), tile_id, "08_coverage"); atomic_npz(directory / "coverage_map.npz", count_map=count_map); atomic_npz(directory / "uncovered_mask.npz", uncovered=uncovered)
    save_masks(directory / "coverage_masks.npz", directory / "coverage_masks.json", final_masks, final_records); atomic_json(directory / "coverage_report.json", report)
    save_image(directory / "coverage_preview.png", coverage_preview(count_map, valid)); save_image(directory / "overlap_preview.png", overlap_preview(count_map))
    outputs = [directory / name for name in ["coverage_map.npz", "uncovered_mask.npz", "coverage_masks.npz", "coverage_masks.json", "coverage_report.json", "coverage_preview.png", "overlap_preview.png"]]
    _finish(cfg, project_root, tile_id, "08_coverage", inputs, outputs, started); return {**report, "output_directory": str(directory)}


def visualize_phase(cfg: dict[str, Any], project_root: Path, tile_id: str) -> dict[str, Any]:
    tile_id = _canonical(cfg, project_root, tile_id); started = now(); root = _root(cfg, project_root); prep = phase_dir(root, tile_id, "02_preprocessing"); coverage = phase_dir(root, tile_id, "08_coverage")
    inputs = [prep / "sam_input.npy", coverage / "coverage_masks.npz", coverage / "coverage_masks.json"]
    for p in inputs: require_fresh(p)
    image = np.load(inputs[0]); final_masks, final_records = load_masks(inputs[1], inputs[2]); vis = cfg["visualization"]
    directory = phase_dir(root, tile_id, "09_visualization"); outputs = []
    sources = [("raw_mask_overlay.png", "04_raw_inference", "raw_masks"), ("accepted_mask_overlay.png", "06_filtering", "accepted_masks"), ("rejected_mask_overlay.png", "06_filtering", "rejected_masks"), ("deduplicated_mask_overlay.png", "07_deduplication", "final_candidate_masks")]
    for filename, phase, stem in sources:
        base = phase_dir(root, tile_id, phase)
        if (base / f"{stem}.npz").is_file():
            masks, records = load_masks(base / f"{stem}.npz", base / f"{stem}.json"); path = directory / filename
            save_image(path, overlay(image, masks, records, alpha=vis["alpha"], seed=vis["random_seed"], show_ids=vis["show_ids"], show_scores=vis["show_scores"])); outputs.append(path)
    final_overlay = overlay(image, final_masks, final_records, vis["alpha"], vis["random_seed"], vis["show_ids"], vis["show_scores"])
    for filename, value in [("mask_boundary_overlay.png", boundary_overlay(image, final_masks)), ("random_color_label_map.png", label_map(final_masks, image.shape[:2], vis["random_seed"])), ("phase_comparison.png", comparison([image, final_overlay, boundary_overlay(image, final_masks)]))]:
        path = directory / filename; save_image(path, value); outputs.append(path)
    if final_masks:
        selected = np.where(final_masks[0][..., None], image, 0)
        path = directory / "selected_mask_preview.png"; save_image(path, selected); outputs.append(path)
    rejected_path = phase_dir(root, tile_id, "06_filtering") / "rejected_masks.json"
    rejected = _json(rejected_path).get("records", []) if rejected_path.is_file() else []
    table = inspection_rows(final_records, _json(phase_dir(root, tile_id, "07_deduplication") / "duplicate_groups.json")); atomic_json(directory / "mask_inspection.json", {"records": table, "rejected": rejected}); outputs.append(directory / "mask_inspection.json")
    _finish(cfg, project_root, tile_id, "09_visualization", inputs, outputs, started); return {"mask_count": len(final_records), "output_directory": str(directory), "outputs": [str(p) for p in outputs]}


def finalize_phase(cfg: dict[str, Any], project_root: Path, tile_id: str) -> dict[str, Any]:
    tile_id = _canonical(cfg, project_root, tile_id); started = now(); root = _root(cfg, project_root); coverage = phase_dir(root, tile_id, "08_coverage"); validation = phase_dir(root, tile_id, "01_validation"); vis = phase_dir(root, tile_id, "09_visualization")
    inputs = [coverage / "coverage_masks.npz", coverage / "coverage_masks.json", coverage / "coverage_report.json", validation / "tile_metadata.normalized.json"]
    for p in inputs: require_fresh(p)
    masks, records = load_masks(inputs[0], inputs[1]); metadata, coverage_report = _json(inputs[3]), _json(inputs[2]); directory = phase_dir(root, tile_id, "10_final")
    segments_npz, segments_json = directory / "segments.npz", directory / "segments.json"; save_masks(segments_npz, segments_json, masks, records)
    config_hash = stable_hash(relevant_config(cfg, "10_final")); report = {"tile_identity": tile_id, "segment_count": len(records), "coverage_statistics": coverage_report, "filtering_summary": _optional_json(phase_dir(root, tile_id, "06_filtering") / "filtering_report.json"), "deduplication_summary": _optional_json(phase_dir(root, tile_id, "07_deduplication") / "deduplication_report.json"), "configuration_hash": config_hash}
    atomic_json(directory / "segmentation_report.json", report)
    final_manifest = {"tile": metadata, "preprocessing_reference": str((phase_dir(root, tile_id, "02_preprocessing") / "preprocessing.json").resolve()), "backend": cfg["backend"], "checkpoint": cfg.get("checkpoint"), "model_type": cfg.get("model_type"), "model_config": cfg.get("model_config"), "configuration": relevant_config(cfg, "10_final"), "configuration_hash": config_hash, "segments": str(segments_json.resolve()), "mask_data": str(segments_npz.resolve()), "created_at": now(), "provenance": [{"segment_id": r["segment_id"], "source_raw_mask_id": r.get("source_raw_mask_id"), "provenance": r.get("provenance", {})} for r in records]}; atomic_json(directory / "manifest.json", final_manifest)
    if (vis / "deduplicated_mask_overlay.png").is_file(): shutil.copy2(vis / "deduplicated_mask_overlay.png", directory / "final_overlay.png")
    if (vis / "random_color_label_map.png").is_file(): shutil.copy2(vis / "random_color_label_map.png", directory / "final_label_map.png")
    phase_manifest = _finish(cfg, project_root, tile_id, "10_final", inputs, [segments_npz, segments_json, directory / "segmentation_report.json", directory / "manifest.json"], started)
    atomic_json(directory / "phase_manifest.json", phase_manifest)
    # Restore the requested stable artifact manifest after the generic phase manifest write.
    atomic_json(directory / "manifest.json", final_manifest)
    return {**report, "output_directory": str(directory)}


PHASES: dict[str, Callable[..., dict[str, Any]]] = {"validate-tile": validate_phase, "prepare-input": prepare_phase, "check-model": check_model_phase, "infer-raw": infer_phase, "normalize-masks": normalize_phase, "filter-masks": filter_phase, "deduplicate-masks": deduplicate_phase, "analyze-coverage": coverage_phase, "visualize": visualize_phase, "finalize": finalize_phase}


def run_all(
    cfg: dict[str, Any],
    project_root: Path,
    tile_id: str,
    backend: SegmentationBackend | None = None,
    progress_callback: Callable[[str, str], None] | None = None,
) -> list[dict[str, Any]]:
    """Run the instance phases, optionally reporting phase start/completion."""

    def execute(name: str, function: Callable[..., dict[str, Any]], *args: Any) -> dict[str, Any]:
        if progress_callback:
            progress_callback(name, "started")
        try:
            result = function(*args)
        except Exception:
            if progress_callback:
                progress_callback(name, "failed")
            raise
        if progress_callback:
            progress_callback(name, "completed")
        return result

    results = [execute("validate-tile", validate_phase, cfg, project_root, tile_id)]
    tile_id = results[0]["tile_id"]
    results.append(execute("prepare-input", prepare_phase, cfg, project_root, tile_id))
    results.append(execute("infer-raw", infer_phase, cfg, project_root, tile_id, backend))
    results.append(execute("normalize-masks", normalize_phase, cfg, project_root, tile_id))
    results.append(execute("filter-masks", filter_phase, cfg, project_root, tile_id))
    results.append(execute("deduplicate-masks", deduplicate_phase, cfg, project_root, tile_id))
    results.append(execute("analyze-coverage", coverage_phase, cfg, project_root, tile_id))
    results.append(execute("visualize-instance", visualize_phase, cfg, project_root, tile_id))
    results.append(execute("finalize-instance", finalize_phase, cfg, project_root, tile_id))
    return results


def _canonical(cfg: dict[str, Any], project_root: Path, tile_id: str) -> str:
    candidate = _root(cfg, project_root) / tile_id / "01_validation" / "tile_metadata.normalized.json"
    if candidate.is_file(): return tile_id
    _, metadata, _ = validate_tile(project_root, tile_id, cfg.get("_project_paths")); return metadata["tile_id"]


def _json(path: Path) -> Any: return json.loads(path.read_text(encoding="utf-8"))
def _optional_json(path: Path) -> Any: return _json(path) if path.is_file() else None
def _uint8_preview(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2: image = image[..., None]
    image = image[..., :3] if image.shape[2] >= 3 else np.repeat(image[..., :1], 3, axis=2)
    if image.dtype == np.uint8: return image
    finite = np.isfinite(image); values = image[finite]
    if not values.size: return np.zeros(image.shape, np.uint8)
    lo, hi = np.percentile(values, [2, 98]); return np.clip((np.nan_to_num(image, nan=lo)-lo)*255/max(1e-12, hi-lo), 0, 255).astype(np.uint8)
def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(".npy.tmp")
    with temporary.open("wb") as stream: np.save(stream, array, allow_pickle=False)
    temporary.replace(path)
def _reason_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    output: dict[str, int] = {}
    for record in records:
        for reason in record["rejection_reasons"]: output[reason] = output.get(reason, 0) + 1
    return output
