from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class SegmentMask:
    segment_id: str
    source_tile_id: str
    bbox_xywh: tuple[int, int, int, int]
    centroid_rc: tuple[float, float]
    area_pixels: int
    predicted_iou: float | None
    stability_score: float | None
    valid_pixel_fraction: float
    touches_tile_boundary: bool
    source_raw_mask_id: str
    provenance: dict[str, Any]


def decode_raw_segmentation(value: Any, shape: tuple[int, int]) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(bool)
    if isinstance(value, dict) and isinstance(value.get("counts"), list):
        flat, bit = [], False
        for count in value["counts"]:
            flat.extend([bit] * int(count))
            bit = not bit
        return np.asarray(flat[: shape[0] * shape[1]], bool).reshape(shape, order="F")
    if isinstance(value, dict) and isinstance(value.get("counts"), (str, bytes)):
        try:
            from pycocotools import mask as mask_utils
        except ImportError as exc:
            raise RuntimeError("Compressed COCO RLE requires pycocotools") from exc
        return mask_utils.decode(value).astype(bool)
    raise ValueError("Unsupported raw segmentation representation")


def normalize_masks(raw_masks: list[np.ndarray], raw_records: list[dict[str, Any]], shape: tuple[int, int], valid: np.ndarray, tile_id: str) -> tuple[list[np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    accepted_masks, accepted, rejected = [], [], []
    h, w = shape
    for index, (mask, raw) in enumerate(zip(raw_masks, raw_records)):
        raw_id = str(raw.get("raw_mask_id", f"raw_{index:05d}"))
        reasons = []
        mask = np.asarray(mask)
        if mask.shape != shape:
            reasons.append(f"dimension mismatch: {mask.shape} != {shape}")
        else:
            mask = mask.astype(bool)
            if not mask.any():
                reasons.append("empty mask")
        if reasons:
            rejected.append({"raw_mask_id": raw_id, "reasons": reasons})
            continue
        rows, cols = np.nonzero(mask)
        x0, x1, y0, y1 = int(cols.min()), int(cols.max()) + 1, int(rows.min()), int(rows.max()) + 1
        area = int(mask.sum())
        valid_fraction = float(np.logical_and(mask, valid).sum() / area)
        segment = SegmentMask(
            segment_id=f"seg_{len(accepted):05d}", source_tile_id=tile_id,
            bbox_xywh=(x0, y0, x1 - x0, y1 - y0),
            centroid_rc=(float(rows.mean()), float(cols.mean())), area_pixels=area,
            predicted_iou=_float(raw.get("predicted_iou")), stability_score=_float(raw.get("stability_score")),
            valid_pixel_fraction=valid_fraction,
            touches_tile_boundary=bool(x0 == 0 or y0 == 0 or x1 == w or y1 == h),
            source_raw_mask_id=raw_id,
            provenance={"raw_mask_id": raw_id, "raw_reported_area": raw.get("area"), "raw_bbox": raw.get("bbox")},
        )
        accepted_masks.append(mask)
        accepted.append(asdict(segment))
    return accepted_masks, accepted, {"input_count": len(raw_records), "normalized_count": len(accepted), "rejected_count": len(rejected), "rejected": rejected}


def filter_masks(masks: list[np.ndarray], records: list[dict[str, Any]], valid: np.ndarray, config: dict[str, Any]) -> tuple[list[np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    accepted_masks, accepted, rejected = [], [], []
    total = valid.size
    for mask, record in zip(masks, records):
        reasons = []
        area = int(mask.sum())
        bbox = record["bbox_xywh"]
        fill_ratio = area / max(1, int(bbox[2]) * int(bbox[3]))
        metrics = {"area_pixels": area, "area_ratio": area / total, "predicted_iou": record.get("predicted_iou"), "stability_score": record.get("stability_score"), "valid_pixel_fraction": record["valid_pixel_fraction"], "nodata_fraction": 1 - record["valid_pixel_fraction"], "fill_ratio": fill_ratio, "touches_tile_boundary": record["touches_tile_boundary"]}
        if area < config["minimum_area"]: reasons.append("area_below_minimum")
        if metrics["area_ratio"] > config["maximum_area_ratio"]: reasons.append("area_ratio_above_maximum")
        if config.get("minimum_predicted_iou") is not None and (record.get("predicted_iou") is None or record["predicted_iou"] < config["minimum_predicted_iou"]): reasons.append("predicted_iou_below_minimum")
        if config.get("minimum_stability_score") is not None and (record.get("stability_score") is None or record["stability_score"] < config["minimum_stability_score"]): reasons.append("stability_below_minimum")
        if metrics["valid_pixel_fraction"] < config["minimum_valid_pixel_fraction"]: reasons.append("valid_pixel_fraction_below_minimum")
        if metrics["nodata_fraction"] > config["maximum_nodata_fraction"]: reasons.append("nodata_fraction_above_maximum")
        if fill_ratio < config.get("minimum_fill_ratio", 0): reasons.append("thin_or_degenerate")
        if config.get("border_policy") == "reject" and record["touches_tile_boundary"]: reasons.append("touches_tile_boundary")
        result = {**record, "metrics": metrics, "accepted": not reasons, "rejection_reasons": reasons}
        if reasons:
            rejected.append(result)
        else:
            accepted_masks.append(mask); accepted.append(result)
    return accepted_masks, accepted, rejected


def deduplicate_masks(masks: list[np.ndarray], records: list[dict[str, Any]], config: dict[str, Any]) -> tuple[list[np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    n = len(masks)
    pairs, nested = [], []
    parent = list(range(n))
    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a: int, b: int) -> None:
        a, b = find(a), find(b)
        if a != b: parent[b] = a
    boxes = [_xyxy(r["bbox_xywh"]) for r in records]
    for i in range(n):
        for j in range(i + 1, n):
            if _bbox_intersection(boxes[i], boxes[j]) <= 0: continue
            inter = int(np.logical_and(masks[i], masks[j]).sum())
            if not inter: continue
            ai, aj = int(masks[i].sum()), int(masks[j].sum())
            union_area = ai + aj - inter
            iou, containment = inter / union_area, inter / min(ai, aj)
            area_similarity = min(ai, aj) / max(ai, aj)
            metric = {"a": records[i]["segment_id"], "b": records[j]["segment_id"], "mask_iou": iou, "intersection_over_smaller": containment, "area_similarity": area_similarity}
            is_nested = containment >= config["containment_threshold"]
            duplicate = iou >= config["mask_iou_threshold"] and area_similarity >= config.get("area_similarity_threshold", 0)
            if is_nested and not duplicate: nested.append(metric)
            if duplicate:
                pairs.append(metric); union(i, j)
    groups: dict[int, list[int]] = {}
    for i in range(n): groups.setdefault(find(i), []).append(i)
    kept_masks, kept_records, group_reports = [], [], []
    removed: set[int] = set()
    nested_policy = config["nested_policy"]
    if nested_policy in {"keep_parent", "keep_child"}:
        for relationship in nested:
            i = next(k for k, r in enumerate(records) if r["segment_id"] == relationship["a"])
            j = next(k for k, r in enumerate(records) if r["segment_id"] == relationship["b"])
            larger, smaller = (i, j) if masks[i].sum() >= masks[j].sum() else (j, i)
            removed.add(smaller if nested_policy == "keep_parent" else larger)
    for indices in groups.values():
        active = [i for i in indices if i not in removed]
        if not active: continue
        policy = config["duplicate_policy"]
        if len(active) == 1 or policy == "keep_all": selected = active
        elif policy == "merge_union":
            merged = np.logical_or.reduce([masks[i] for i in active]); base = dict(records[_best(active, records, "predicted_iou")])
            base["segment_id"] += "_union"; base["provenance"] = {"merged_from": [records[i]["segment_id"] for i in active]}
            _refresh(base, merged); kept_masks.append(merged); kept_records.append(base); selected = []
        else:
            key = {"keep_higher_predicted_iou": "predicted_iou", "keep_higher_stability": "stability_score"}.get(policy)
            if key: selected = [_best(active, records, key)]
            elif policy == "keep_larger": selected = [max(active, key=lambda i: masks[i].sum())]
            else: selected = [min(active, key=lambda i: masks[i].sum())]
        for i in selected: kept_masks.append(masks[i]); kept_records.append(records[i])
        if len(indices) > 1: group_reports.append({"members": [records[i]["segment_id"] for i in indices], "selected": [records[i]["segment_id"] for i in selected], "policy": policy})
    report = {"input_count": n, "duplicate_candidate_pairs": len(pairs), "duplicate_pairs": pairs, "duplicate_groups": group_reports, "nested_relationships": nested, "removed_masks": sorted(records[i]["segment_id"] for i in removed), "output_count": len(kept_records)}
    return kept_masks, kept_records, report


def analyze_coverage(masks: list[np.ndarray], records: list[dict[str, Any]], valid: np.ndarray, config: dict[str, Any]) -> tuple[list[np.ndarray], list[dict[str, Any]], np.ndarray, np.ndarray, dict[str, Any]]:
    count_map = np.zeros(valid.shape, np.uint16)
    for mask in masks: count_map += mask.astype(np.uint16)
    covered = valid & (count_map > 0); uncovered = valid & (count_map == 0); overlap = valid & (count_map > 1)
    valid_count = int(valid.sum())
    report = {"valid_pixel_count": valid_count, "covered_valid_pixel_count": int(covered.sum()), "uncovered_valid_pixel_count": int(uncovered.sum()), "coverage_percentage": 100 * covered.sum() / max(1, valid_count), "overlap_pixel_count": int(overlap.sum()), "overlap_percentage": 100 * overlap.sum() / max(1, valid_count), "boundary_touching_mask_count": sum(bool(r["touches_tile_boundary"]) for r in records)}
    number, labels, stats, _ = cv2.connectedComponentsWithStats(uncovered.astype(np.uint8), 8)
    components = []
    out_masks, out_records = list(masks), list(records)
    min_area = config["minimum_background_component_area"]
    for label in range(1, number):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area: components.append({"component": label, "area": area})
    policy = config["background_policy"]
    backgrounds = []
    if policy == "single_background" and uncovered.any(): backgrounds = [uncovered]
    elif policy == "connected_components": backgrounds = [labels == c["component"] for c in components]
    for index, mask in enumerate(backgrounds):
        rec = _background_record(mask, records[0]["source_tile_id"] if records else "unknown", index)
        out_masks.append(mask); out_records.append(rec)
    report["uncovered_connected_components"] = components
    report["background_masks_added"] = len(backgrounds)
    return out_masks, out_records, count_map, uncovered, report


def _float(value: Any) -> float | None:
    try: return float(value) if value is not None else None
    except (TypeError, ValueError): return None


def _xyxy(box: list[int] | tuple[int, ...]) -> tuple[int, int, int, int]:
    x, y, w, h = map(int, box); return x, y, x + w, y + h


def _bbox_intersection(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def _best(indices: list[int], records: list[dict[str, Any]], key: str) -> int:
    return max(indices, key=lambda i: records[i].get(key) if records[i].get(key) is not None else -1)


def _refresh(record: dict[str, Any], mask: np.ndarray) -> None:
    rows, cols = np.nonzero(mask); x0, x1, y0, y1 = cols.min(), cols.max() + 1, rows.min(), rows.max() + 1
    record.update(area_pixels=int(mask.sum()), bbox_xywh=[int(x0), int(y0), int(x1-x0), int(y1-y0)], centroid_rc=[float(rows.mean()), float(cols.mean())])


def _background_record(mask: np.ndarray, tile_id: str, index: int) -> dict[str, Any]:
    record = {"segment_id": f"background_{index:05d}", "source_tile_id": tile_id, "predicted_iou": None, "stability_score": None, "valid_pixel_fraction": 1.0, "touches_tile_boundary": bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()), "source_raw_mask_id": "generated_background", "provenance": {"type": "segmentation_placeholder", "semantic_label": False}}
    _refresh(record, mask); return record
