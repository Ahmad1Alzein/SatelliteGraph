"""Persist human-reviewable refinement proposal artifacts.

The review bundle is deliberately self-contained: every entity referenced by
``proposals.json`` can be resolved to an exact normalized mask and record, and
every pending merge/absorb proposal has a deterministic visual crop.  PNGs are
review aids only; the NPZ/JSON geometry remains authoritative.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.stage2.storage import save_masks

from .schema import SCHEMA_VERSION
from .storage import atomic_json, save_segmentation_bundle, sha256_file
from .visualization import (
    proposal_review_crop,
    review_crop_bounds,
    save_png,
    semantic_preview,
)


REVIEW_SCHEMA_VERSION = f"{SCHEMA_VERSION}.review.v1"


@dataclass(slots=True)
class ReviewBundle:
    """Paths and safe default decisions produced for one proposal run."""

    index_path: Path
    segments_path: Path
    masks_path: Path
    segmentation_path: Path
    crop_paths: list[Path]
    decision_records: list[dict[str, Any]]

    @property
    def output_paths(self) -> list[Path]:
        return [
            self.segments_path,
            self.masks_path,
            self.segmentation_path,
            *self.crop_paths,
            self.index_path,
        ]


def save_review_bundle(
    output_directory: str | Path,
    *,
    tile_id: str,
    proposal_checksum: str,
    proposals: Sequence[Mapping[str, Any]],
    masks: Sequence[np.ndarray],
    records: Sequence[Mapping[str, Any]],
    instance_map: np.ndarray,
    semantic_map: np.ndarray,
    confidence_map: np.ndarray,
    source_map: np.ndarray,
    valid_mask: np.ndarray,
    image: np.ndarray | None,
    padding: int,
    alpha: float,
    random_seed: int,
) -> ReviewBundle:
    """Write normalized geometry, a review index, and candidate crops.

    The returned decisions contain every pending proposal with the conservative
    default ``uncertain``.  Passing the untouched template to apply therefore
    cannot authorize a semantic edit.
    """

    directory = Path(output_directory).expanduser() / "review"
    crop_directory = directory / "candidates"
    crop_directory.mkdir(parents=True, exist_ok=True)
    # Proposal regeneration replaces the review set. Remove only direct PNG
    # children of this exact, resolved candidate directory so stale crops can
    # never be mistaken for candidates in the new checksum-bound index.
    resolved_crop_directory = crop_directory.resolve()
    if resolved_crop_directory.parent != directory.resolve():
        raise ValueError("Review candidate directory escaped the refinement output")
    for stale_crop in sorted(crop_directory.glob("*.png")):
        if stale_crop.is_file() and stale_crop.resolve().parent == resolved_crop_directory:
            stale_crop.unlink()

    clean_records = [copy.deepcopy(dict(record)) for record in records]
    clean_masks = [np.asarray(mask, dtype=bool).copy() for mask in masks]
    if len(clean_records) != len(clean_masks):
        raise ValueError("Review records and masks must have equal lengths")

    segments_path = directory / "segments.json"
    masks_path = directory / "segments.npz"
    segmentation_path = directory / "segmentation.npz"
    save_masks(masks_path, segments_path, clean_masks, clean_records)
    save_segmentation_bundle(
        segmentation_path,
        instance_map=np.asarray(instance_map),
        semantic_map=np.asarray(semantic_map),
        confidence_map=np.asarray(confidence_map),
        source_map=np.asarray(source_map),
        valid_mask=np.asarray(valid_mask),
    )

    entity_records: dict[str, dict[str, Any]] = {}
    entity_masks: dict[str, np.ndarray] = {}
    for record, mask in zip(clean_records, clean_masks):
        entity_id = str(record.get("entity_id", ""))
        if not entity_id or entity_id in entity_records:
            raise ValueError("Normalized review records require unique entity_id values")
        entity_records[entity_id] = record
        entity_masks[entity_id] = mask

    display_image = (
        semantic_preview(np.asarray(semantic_map), np.asarray(valid_mask))
        if image is None
        else np.asarray(image)
    )
    pending = sorted(
        (
            copy.deepcopy(dict(proposal))
            for proposal in proposals
            if str(proposal.get("decision")) == "pending"
        ),
        key=lambda proposal: str(proposal["proposal_id"]),
    )
    unsupported = sorted(
        {
            str(proposal.get("operation_type"))
            for proposal in pending
            if str(proposal.get("operation_type")) not in {"merge", "absorb"}
        }
    )
    if unsupported:
        raise ValueError(
            "The v1 review bundle supports pending merge/absorb proposals only; "
            f"got {unsupported}"
        )

    crop_paths: list[Path] = []
    review_items: list[dict[str, Any]] = []
    decision_records: list[dict[str, Any]] = []
    for proposal in pending:
        proposal_id = str(proposal["proposal_id"])
        operation_type = str(proposal["operation_type"])
        source_ids = [str(value) for value in proposal["source_entity_ids"]]
        target_id = (
            None
            if proposal.get("target_entity_id") is None
            else str(proposal["target_entity_id"])
        )
        referenced_ids = [*source_ids, *([target_id] if target_id is not None else [])]
        missing = sorted(value for value in referenced_ids if value not in entity_records)
        if missing:
            raise ValueError(
                f"Review proposal {proposal_id!r} references missing normalized entities: {missing}"
            )

        proposed_map, focus_mask, interface_mask, effect = _proposed_display_map(
            np.asarray(instance_map),
            proposal,
            entity_records,
            entity_masks,
        )
        crop = _render_candidate_crop(
            display_image,
            np.asarray(instance_map),
            proposed_map,
            np.asarray(valid_mask),
            focus_mask,
            interface_mask,
            proposal,
            [entity_records[value] for value in source_ids],
            None if target_id is None else entity_records[target_id],
            padding=padding,
            alpha=alpha,
            random_seed=random_seed,
        )
        crop_path = crop_directory / _crop_name(proposal_id, operation_type)
        save_png(crop_path, crop)
        crop_paths.append(crop_path.resolve())

        y0, x0, y1, x1 = review_crop_bounds(focus_mask, padding=padding)
        review_items.append(
            {
                "proposal": proposal,
                "source_entities": [
                    _entity_summary(entity_records[value]) for value in source_ids
                ],
                "target_entity": (
                    None if target_id is None else _entity_summary(entity_records[target_id])
                ),
                "review_crop": {
                    "path": crop_path.relative_to(directory).as_posix(),
                    "sha256": sha256_file(crop_path),
                    "bounds_xyxy": [x0, y0, x1, y1],
                    "cyan_pixels": "exact shared interface",
                    "magenta_pixels": "pixels whose displayed instance assignment changes",
                },
                "proposed_effect": effect,
                "default_decision": "uncertain",
            }
        )
        decision_records.append(
            {
                "proposal_id": proposal_id,
                "decision": "uncertain",
                "decision_source": "human_review",
                "reviewer": None,
                "reason": None,
            }
        )

    index_path = directory / "index.json"
    geometry = {
        "segments": _artifact_record(segments_path, directory),
        "masks": _artifact_record(masks_path, directory),
        "segmentation": _artifact_record(segmentation_path, directory),
    }
    atomic_json(
        index_path,
        {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "tile_id": str(tile_id),
            "proposal_checksum": str(proposal_checksum),
            "normalized_geometry": geometry,
            "normalized_entity_count": len(clean_records),
            "candidate_count": len(review_items),
            "candidates": review_items,
            "instructions": [
                "Review the original, normalized-before, and proposed panels.",
                "Cyan marks the exact shared interface; magenta marks the displayed change.",
                "Use the full proposal evidence and vetoes before editing decisions.template.json.",
                "Leave uncertain when the imagery does not prove the operation.",
            ],
        },
    )
    return ReviewBundle(
        index_path=index_path.resolve(),
        segments_path=segments_path.resolve(),
        masks_path=masks_path.resolve(),
        segmentation_path=segmentation_path.resolve(),
        crop_paths=crop_paths,
        decision_records=decision_records,
    )


def _proposed_display_map(
    instance_map: np.ndarray,
    proposal: Mapping[str, Any],
    entity_records: Mapping[str, Mapping[str, Any]],
    entity_masks: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    proposed = np.asarray(instance_map).copy()
    operation_type = str(proposal["operation_type"])
    source_ids = [str(value) for value in proposal["source_entity_ids"]]
    target_id = (
        None
        if proposal.get("target_entity_id") is None
        else str(proposal["target_entity_id"])
    )
    focus_ids = [*source_ids, *([target_id] if target_id is not None else [])]
    focus = np.logical_or.reduce([entity_masks[value] for value in focus_ids])
    instance_ids = {
        value: int(entity_records[value]["instance_id"]) for value in focus_ids
    }
    if operation_type == "merge":
        output_instance_id = min(instance_ids[value] for value in source_ids)
        for source_id in source_ids:
            proposed[entity_masks[source_id]] = output_instance_id
        interface = _shared_interface(
            np.asarray(instance_map),
            {instance_ids[value] for value in source_ids},
            {instance_ids[value] for value in source_ids},
        )
    elif operation_type == "absorb" and target_id is not None:
        output_instance_id = instance_ids[target_id]
        proposed[entity_masks[source_ids[0]]] = output_instance_id
        interface = _shared_interface(
            np.asarray(instance_map),
            {instance_ids[source_ids[0]]},
            {instance_ids[target_id]},
        )
    else:  # guarded by save_review_bundle
        raise ValueError(f"Unsupported review operation: {operation_type!r}")
    changed = proposed != instance_map
    return proposed, focus, interface, {
        "display_only": True,
        "changed_pixel_count": int(changed.sum()),
        "source_instance_ids": [instance_ids[value] for value in source_ids],
        "target_instance_id": None if target_id is None else instance_ids[target_id],
        "display_output_instance_id": output_instance_id,
        "valid_pixel_coverage_unchanged": True,
    }


def _shared_interface(
    instance_map: np.ndarray,
    first_ids: set[int],
    second_ids: set[int],
) -> np.ndarray:
    values = np.asarray(instance_map)
    result = np.zeros(values.shape, dtype=bool)

    def mark(first: np.ndarray, second: np.ndarray, axis: str) -> None:
        different = first != second
        cross = different & (
            (np.isin(first, list(first_ids)) & np.isin(second, list(second_ids)))
            | (np.isin(first, list(second_ids)) & np.isin(second, list(first_ids)))
        )
        if axis == "horizontal":
            result[:, :-1] |= cross
            result[:, 1:] |= cross
        else:
            result[:-1, :] |= cross
            result[1:, :] |= cross

    mark(values[:, :-1], values[:, 1:], "horizontal")
    mark(values[:-1, :], values[1:, :], "vertical")
    return result


def _render_candidate_crop(
    image: np.ndarray,
    before: np.ndarray,
    proposed: np.ndarray,
    valid_mask: np.ndarray,
    focus_mask: np.ndarray,
    interface_mask: np.ndarray,
    proposal: Mapping[str, Any],
    source_records: Sequence[Mapping[str, Any]],
    target_record: Mapping[str, Any] | None,
    *,
    padding: int,
    alpha: float,
    random_seed: int,
) -> np.ndarray:
    changed = before != proposed
    y0, x0, y1, x1 = review_crop_bounds(focus_mask | changed, padding=padding)
    cropped_focus = np.ones((y1 - y0, x1 - x0), dtype=bool)
    panel = proposal_review_crop(
        image[y0:y1, x0:x1],
        before[y0:y1, x0:x1],
        proposed[y0:y1, x0:x1],
        focus_mask=cropped_focus,
        valid_mask=valid_mask[y0:y1, x0:x1],
        padding=0,
        alpha=alpha,
        random_seed=random_seed,
        # Drawing every nearby instance ID makes dense satellite crops
        # unreadable. The exact source/target IDs and classes are listed in
        # the footer and index instead.
        show_ids=False,
        titles=True,
    ).copy()
    crop_interface = interface_mask[y0:y1, x0:x1]
    height, width = crop_interface.shape
    title_height = 26
    gap = 6
    for start_x in (0, width + gap):
        view = panel[title_height : title_height + height, start_x : start_x + width]
        view[crop_interface] = (0, 255, 255)

    visual = Image.fromarray(panel, mode="RGB")
    scale = max(1, math.ceil(720 / max(1, visual.width)))
    if scale > 1:
        visual = visual.resize(
            (visual.width * scale, visual.height * scale),
            resample=Image.Resampling.NEAREST,
        )
    footer_lines = _footer_lines(proposal, source_records, target_record)
    footer_height = 18 + 15 * len(footer_lines)
    canvas = Image.new("RGB", (visual.width, visual.height + footer_height), (20, 26, 34))
    canvas.paste(visual, (0, 0))
    draw = ImageDraw.Draw(canvas)
    for index, line in enumerate(footer_lines):
        draw.text((8, visual.height + 7 + index * 15), line, fill=(245, 247, 250))
    return np.asarray(canvas, dtype=np.uint8)


def _footer_lines(
    proposal: Mapping[str, Any],
    source_records: Sequence[Mapping[str, Any]],
    target_record: Mapping[str, Any] | None,
) -> list[str]:
    proposal_id = str(proposal["proposal_id"])
    operation = str(proposal["operation_type"])
    evidence = proposal.get("evidence", {})
    sources = "; ".join(_entity_text(record) for record in source_records)
    target = "none" if target_record is None else _entity_text(target_record)
    vetoes = (
        evidence.get("hard_vetoes", evidence.get("vetoes", []))
        if isinstance(evidence, Mapping)
        else []
    )
    veto_text = "none" if not vetoes else ", ".join(str(value) for value in vetoes)
    return [
        "panels left-to-right: ORIGINAL | NORMALIZED BEFORE | PROPOSED DISPLAY",
        f"{operation.upper()}  proposal={proposal_id.rsplit('::', 1)[-1]}  score={_number(proposal.get('score'))}",
        f"sources: {sources}",
        f"target: {target}",
        (
            "evidence: shared_sides="
            f"{evidence.get('shared_boundary_side_count', 'n/a')}  "
            f"shared_fraction={_number(evidence.get('minimum_perimeter_shared_boundary_fraction'))}  "
            f"weak_boundary={_number(evidence.get('weak_color_boundary_score'))}  "
            f"appearance={_number(evidence.get('region_mean_color_similarity'))}"
        ),
        f"vetoes: {veto_text}  |  cyan=exact interface; magenta=displayed changed pixels",
    ]


def _entity_text(record: Mapping[str, Any]) -> str:
    label = record.get("label", {})
    class_name = label.get("name", "unknown") if isinstance(label, Mapping) else "unknown"
    return (
        f"I{int(record['instance_id'])} {class_name} "
        f"{int(record.get('area_pixels', 0))}px"
    )


def _entity_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    label = record.get("label", {})
    return {
        "entity_id": str(record["entity_id"]),
        "segment_id": str(record["segment_id"]),
        "instance_id": int(record["instance_id"]),
        "class_id": None if not isinstance(label, Mapping) else int(label["id"]),
        "class_name": None if not isinstance(label, Mapping) else str(label["name"]),
        "area_pixels": int(record.get("area_pixels", 0)),
        "bbox_xyxy": copy.deepcopy(record.get("bbox_xyxy")),
        "source_stage2_instance_ids": copy.deepcopy(
            record.get("source_stage2_instance_ids", [])
        ),
        "source_stage2_segment_ids": copy.deepcopy(
            record.get("source_stage2_segment_ids", [])
        ),
    }


def _artifact_record(path: Path, relative_to: Path) -> dict[str, str]:
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": sha256_file(path),
    }


def _crop_name(proposal_id: str, operation_type: str) -> str:
    digest = hashlib.sha256(proposal_id.encode("utf-8")).hexdigest()[:20]
    return f"{operation_type}_{digest}.png"


def _number(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.3f}"


__all__ = ["REVIEW_SCHEMA_VERSION", "ReviewBundle", "save_review_bundle"]
