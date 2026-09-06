"""Deterministic previews and review crops for refinement proposals."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.stage2.classes import NODATA_CLASS_ID
from src.stage2.visualization import save_image as _save_stage2_image


# Byte-for-byte equivalent to the Stage 2 final semantic preview palette.
SEMANTIC_COLORS = np.asarray(
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


def semantic_preview(semantic_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Render the Stage 2 semantic palette, including purple ``unknown``."""

    labels = np.asarray(semantic_map)
    valid = np.asarray(valid_mask)
    if labels.ndim != 2 or labels.shape != valid.shape:
        raise ValueError("semantic_map and valid_mask must be 2-D with one common shape")
    if labels.dtype == np.bool_ or not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("semantic_map must use an integer dtype")
    if valid.dtype != np.bool_:
        raise TypeError("valid_mask must use boolean dtype")
    output = np.zeros((*labels.shape, 3), dtype=np.uint8)
    known = valid & (labels >= 0) & (labels < len(SEMANTIC_COLORS))
    output[known] = SEMANTIC_COLORS[labels[known]]
    nodata = ~valid | (labels == NODATA_CLASS_ID)
    output[nodata] = (0, 0, 0)
    return output


def instance_preview(
    instance_map: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    random_seed: int = 42,
) -> np.ndarray:
    """Render stable per-instance colors independent of ID density/order."""

    instances = _instance_map(instance_map)
    valid = _valid_mask(valid_mask, instances.shape)
    seed = _nonnegative_integer(random_seed, "random_seed")
    output = np.zeros((*instances.shape, 3), dtype=np.uint8)
    assigned_flat = np.flatnonzero((valid & (instances > 0)).reshape(-1))
    if assigned_flat.size:
        identifiers, inverse = np.unique(
            instances.reshape(-1)[assigned_flat], return_inverse=True
        )
        palette = np.stack([_instance_color(int(value), seed) for value in identifiers])
        output.reshape(-1, 3)[assigned_flat] = palette[inverse]
    output[~valid] = (32, 32, 32)
    return output


def instance_overlay(
    image: np.ndarray,
    instance_map: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    alpha: float = 0.45,
    random_seed: int = 42,
    show_boundaries: bool = True,
    show_ids: bool = False,
    changed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Overlay deterministic instances, optional IDs, and changed pixels."""

    instances = _instance_map(instance_map)
    rgb = _rgb(image, instances.shape)
    valid = _valid_mask(valid_mask, instances.shape)
    opacity = _fraction(alpha, "alpha")
    preview = instance_preview(instances, valid, random_seed=random_seed)
    output = rgb.astype(np.float32)
    assigned = valid & (instances > 0)
    output[assigned] = (
        (1.0 - opacity) * output[assigned] + opacity * preview[assigned].astype(np.float32)
    )
    output = np.clip(np.rint(output), 0, 255).astype(np.uint8)
    if show_boundaries:
        output[_boundaries(instances, valid)] = (255, 255, 255)

    if changed_mask is not None:
        changed = np.asarray(changed_mask)
        if changed.shape != instances.shape or changed.dtype != np.bool_:
            raise ValueError("changed_mask must be boolean with the instance-map shape")
        changed_boundary = _binary_boundary(changed)
        output[changed & ~changed_boundary] = np.rint(
            0.55 * output[changed & ~changed_boundary] + 0.45 * np.asarray((255, 0, 255))
        ).astype(np.uint8)
        output[changed_boundary] = (255, 0, 255)

    if show_ids:
        output = _draw_instance_ids(output, instances, valid)
    return output


def proposal_review_crop(
    image: np.ndarray,
    before_instance_map: np.ndarray,
    proposed_instance_map: np.ndarray,
    *,
    changed_mask: np.ndarray | None = None,
    focus_mask: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    padding: int = 32,
    alpha: float = 0.45,
    random_seed: int = 42,
    show_ids: bool = True,
    titles: bool = True,
) -> np.ndarray:
    """Build ``original | before | proposed`` panels around one proposal.

    Changed pixels are highlighted in magenta in the proposed panel.  By
    default, a proposal's directly changed instance IDs define those pixels.
    Callers comparing complete pipeline products may supply a semantic/topology
    ``changed_mask`` so harmless global ID renumbering is not presented as a
    change.  When no focus is supplied and nothing changed, the full image is
    shown.
    """

    before = _instance_map(before_instance_map)
    proposed = _instance_map(proposed_instance_map)
    if proposed.shape != before.shape:
        raise ValueError("before and proposed instance maps must have the same shape")
    rgb = _rgb(image, before.shape)
    valid = _valid_mask(valid_mask, before.shape)
    if changed_mask is None:
        changed = before != proposed
    else:
        changed = np.asarray(changed_mask)
        if changed.shape != before.shape or changed.dtype != np.bool_:
            raise ValueError("changed_mask must be boolean with the instance-map shape")
    if focus_mask is None:
        focus = changed
    else:
        focus = np.asarray(focus_mask)
        if focus.shape != before.shape or focus.dtype != np.bool_:
            raise ValueError("focus_mask must be boolean with the instance-map shape")
        focus = focus | changed
    padding_value = _nonnegative_integer(padding, "padding")
    y0, x0, y1, x1 = review_crop_bounds(focus, padding=padding_value)

    original_panel = rgb[y0:y1, x0:x1].copy()
    if np.any(focus):
        focus_boundary = _binary_boundary(focus)[y0:y1, x0:x1]
        original_panel[focus_boundary] = (255, 215, 0)
    before_panel = instance_overlay(
        rgb,
        before,
        valid,
        alpha=alpha,
        random_seed=random_seed,
        show_ids=show_ids,
    )[y0:y1, x0:x1]
    proposed_panel = instance_overlay(
        rgb,
        proposed,
        valid,
        alpha=alpha,
        random_seed=random_seed,
        show_ids=show_ids,
        changed_mask=changed,
    )[y0:y1, x0:x1]
    return _panels(
        [original_panel, before_panel, proposed_panel],
        ["Original", "Before", "Proposed"],
        titles=titles,
    )


def review_crop_bounds(focus_mask: np.ndarray, *, padding: int = 32) -> tuple[int, int, int, int]:
    """Return deterministic ``(y0, x0, y1, x1)`` bounds for a focus mask."""

    focus = np.asarray(focus_mask)
    if focus.ndim != 2 or focus.dtype != np.bool_:
        raise ValueError("focus_mask must be a 2-D boolean array")
    if any(length <= 0 for length in focus.shape):
        raise ValueError("focus_mask dimensions must be positive")
    pad = _nonnegative_integer(padding, "padding")
    rows, columns = np.nonzero(focus)
    if not len(rows):
        return (0, 0, focus.shape[0], focus.shape[1])
    return (
        max(0, int(rows.min()) - pad),
        max(0, int(columns.min()) - pad),
        min(focus.shape[0], int(rows.max()) + 1 + pad),
        min(focus.shape[1], int(columns.max()) + 1 + pad),
    )


def save_png(path: str | Path, image: np.ndarray) -> Path:
    """Save an RGB image atomically through the existing Stage 2 helper."""

    target = Path(path).expanduser()
    if target.suffix.lower() != ".png":
        raise ValueError(f"PNG output path must end in .png: {target}")
    rgb = _rgb_any_shape(image)
    _save_stage2_image(target, rgb)
    return target.resolve()


def save_refinement_visualizations(
    output_directory: str | Path,
    *,
    image: np.ndarray,
    instance_map: np.ndarray,
    semantic_map: np.ndarray,
    valid_mask: np.ndarray,
    before_instance_map: np.ndarray | None = None,
    before_semantic_map: np.ndarray | None = None,
    alpha: float = 0.45,
    random_seed: int = 42,
    show_ids: bool = False,
) -> dict[str, Path]:
    """Save the standard refined segmentation, overlay, and optional review."""

    directory = Path(output_directory).expanduser()
    semantic_path = save_png(
        directory / "refined_segmentation.png", semantic_preview(semantic_map, valid_mask)
    )
    overlay_path = save_png(
        directory / "refined_overlay.png",
        instance_overlay(
            image,
            instance_map,
            valid_mask,
            alpha=alpha,
            random_seed=random_seed,
            show_ids=show_ids,
        ),
    )
    paths = {"segmentation": semantic_path, "overlay": overlay_path}
    if before_instance_map is not None:
        changed_mask = None
        if before_semantic_map is not None:
            changed_mask = refinement_change_mask(
                before_instance_map,
                instance_map,
                before_semantic_map=before_semantic_map,
                after_semantic_map=semantic_map,
                valid_mask=valid_mask,
            )
        review_path = save_png(
            directory / "refinement_before_after.png",
            proposal_review_crop(
                image,
                before_instance_map,
                instance_map,
                changed_mask=changed_mask,
                valid_mask=valid_mask,
                padding=0,
                alpha=alpha,
                random_seed=random_seed,
                show_ids=show_ids,
            ),
        )
        paths["before_after"] = review_path
    return paths


def refinement_change_mask(
    before_instance_map: np.ndarray,
    after_instance_map: np.ndarray,
    *,
    before_semantic_map: np.ndarray,
    after_semantic_map: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return pixels with a real semantic or segmentation-boundary change.

    Numeric instance IDs are storage identifiers, not stable identities across
    the normalization/refinement boundary.  Comparing them directly can mark
    nearly every pixel after a harmless global renumbering.  This comparison is
    invariant to such renumbering: it includes pixels whose semantic class was
    relabeled and pixels along an instance boundary that was added or removed.
    """

    before = _instance_map(before_instance_map)
    after = _instance_map(after_instance_map)
    if after.shape != before.shape:
        raise ValueError("before and after instance maps must have the same shape")
    valid = _valid_mask(valid_mask, before.shape)
    before_semantic = _semantic_map(
        before_semantic_map, before.shape, "before_semantic_map"
    )
    after_semantic = _semantic_map(
        after_semantic_map, before.shape, "after_semantic_map"
    )

    semantic_changed = valid & (before_semantic != after_semantic)
    boundary_changed = _boundaries(before, valid) ^ _boundaries(after, valid)
    return semantic_changed | boundary_changed


def _instance_color(instance_id: int, random_seed: int) -> np.ndarray:
    digest = hashlib.blake2b(
        f"{random_seed}:{instance_id}".encode("ascii"), digest_size=3, person=b"satgraph"
    ).digest()
    # Avoid nearly black colors while retaining a broad deterministic gamut.
    values = np.frombuffer(digest, dtype=np.uint8).astype(np.uint16)
    return ((values % 192) + 48).astype(np.uint8)


def _boundaries(instances: np.ndarray, valid: np.ndarray) -> np.ndarray:
    boundary = np.zeros(instances.shape, dtype=bool)
    horizontal = valid[:, 1:] & valid[:, :-1] & (instances[:, 1:] != instances[:, :-1])
    vertical = valid[1:, :] & valid[:-1, :] & (instances[1:, :] != instances[:-1, :])
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary & valid & (instances > 0)


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    boundary = np.zeros(mask.shape, dtype=bool)
    horizontal = mask[:, 1:] != mask[:, :-1]
    vertical = mask[1:, :] != mask[:-1, :]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    # A full-image mask otherwise has no internal transition; mark its frame.
    if np.all(mask):
        boundary[[0, -1], :] = True
        boundary[:, [0, -1]] = True
    return boundary & (mask | _dilate_one(mask))


def _dilate_one(mask: np.ndarray) -> np.ndarray:
    dilated = mask.copy()
    dilated[1:, :] |= mask[:-1, :]
    dilated[:-1, :] |= mask[1:, :]
    dilated[:, 1:] |= mask[:, :-1]
    dilated[:, :-1] |= mask[:, 1:]
    return dilated


def _draw_instance_ids(image: np.ndarray, instances: np.ndarray, valid: np.ndarray) -> np.ndarray:
    canvas = Image.fromarray(image, mode="RGB")
    draw = ImageDraw.Draw(canvas)
    rows, columns = np.nonzero(valid & (instances > 0))
    if not len(rows):
        return np.asarray(canvas, dtype=np.uint8)
    identifiers, inverse, counts = np.unique(
        instances[rows, columns], return_inverse=True, return_counts=True
    )
    x_sums = np.bincount(inverse, weights=columns, minlength=len(identifiers))
    y_sums = np.bincount(inverse, weights=rows, minlength=len(identifiers))
    for index, identifier in enumerate(identifiers):
        x = float(x_sums[index] / counts[index])
        y = float(y_sums[index] / counts[index])
        label = str(int(identifier))
        draw.text(
            (x, y),
            label,
            anchor="mm",
            fill=(255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    return np.asarray(canvas, dtype=np.uint8)


def _panels(images: list[np.ndarray], labels: list[str], *, titles: bool) -> np.ndarray:
    if len(images) != len(labels) or not images:
        raise ValueError("Each review panel requires one label")
    height = max(image.shape[0] for image in images)
    title_height = 26 if titles else 0
    gap = 6
    width = sum(image.shape[1] for image in images) + gap * (len(images) - 1)
    canvas = np.full((height + title_height, width, 3), (20, 26, 34), dtype=np.uint8)
    starts: list[tuple[int, int]] = []
    x = 0
    for panel in images:
        canvas[title_height : title_height + panel.shape[0], x : x + panel.shape[1]] = panel
        starts.append((x, panel.shape[1]))
        x += panel.shape[1] + gap
    if titles:
        pil = Image.fromarray(canvas, mode="RGB")
        draw = ImageDraw.Draw(pil)
        for (start, panel_width), label in zip(starts, labels):
            bounds = draw.textbbox((0, 0), label)
            text_width = bounds[2] - bounds[0]
            draw.text(
                (start + (panel_width - text_width) / 2.0, 6),
                label,
                fill=(245, 247, 250),
            )
        canvas = np.asarray(pil, dtype=np.uint8)
    return canvas


def _instance_map(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or any(length <= 0 for length in array.shape):
        raise ValueError(f"instance_map must have a non-empty 2-D shape, got {array.shape}")
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.integer):
        raise TypeError("instance_map must use an integer dtype")
    if np.any(array < 0):
        raise ValueError("instance_map may contain only zero and positive IDs")
    return array


def _valid_mask(value: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if value is None:
        return np.ones(shape, dtype=bool)
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.bool_:
        raise ValueError("valid_mask must be boolean with the instance-map shape")
    return array


def _semantic_map(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have the instance-map shape")
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must use an integer dtype")
    return array


def _rgb(value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = _rgb_any_shape(value)
    if array.shape[:2] != shape:
        raise ValueError(f"image shape {array.shape[:2]} does not match map shape {shape}")
    return array


def _rgb_any_shape(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in {1, 3, 4}:
        raise ValueError("image must be HxW, HxWx1, HxWx3, or HxWx4")
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.dtype == np.uint8:
        converted = array.copy()
    else:
        converted = array.astype(np.float64)
        if not np.isfinite(converted).all():
            raise ValueError("image values must be finite")
        if converted.size and 0.0 <= float(converted.min()) and float(converted.max()) <= 1.0:
            converted *= 255.0
        converted = np.clip(np.rint(converted), 0, 255).astype(np.uint8)
    if converted.shape[2] == 4:
        opacity = converted[..., 3:4].astype(np.float32) / 255.0
        converted = np.rint(
            converted[..., :3].astype(np.float32) * opacity + 255.0 * (1.0 - opacity)
        ).astype(np.uint8)
    return np.ascontiguousarray(converted[..., :3])


def _fraction(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


def _nonnegative_integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a non-negative integer")
    if int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


__all__ = [
    "SEMANTIC_COLORS",
    "instance_overlay",
    "instance_preview",
    "proposal_review_crop",
    "refinement_change_mask",
    "review_crop_bounds",
    "save_png",
    "save_refinement_visualizations",
    "semantic_preview",
]
