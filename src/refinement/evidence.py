"""Deterministic raster evidence used to rank refinement proposals.

All pair relationships use the same exact rule as Stage 3: two positive
instances are adjacent only when their pixels share at least one full side.
Diagonal contact is never included.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


ADJACENCY_DEFINITION = "shared_pixel_edge_4_connected"


def adjacency_contacts(instance_map: np.ndarray) -> list[dict[str, Any]]:
    """Return exact contact, area, perimeter, and contact-fraction evidence."""

    values = _instance_map(instance_map)
    contacts = _scan_contacts(values)
    if contacts is None:
        return []
    unique_pairs, groups = np.unique(contacts["pairs"], axis=0, return_inverse=True)
    count = len(unique_pairs)
    shared = np.bincount(groups, minlength=count).astype(np.int64)
    vertical = np.bincount(
        groups, weights=contacts["vertical"].astype(np.int64), minlength=count
    ).astype(np.int64)
    horizontal = shared - vertical
    areas = _areas(values)
    perimeters = _perimeters(values)

    output: list[dict[str, Any]] = []
    for index, pair in enumerate(unique_pairs):
        source, target = int(pair[0]), int(pair[1])
        source_perimeter = int(perimeters[source])
        target_perimeter = int(perimeters[target])
        shared_sides = int(shared[index])
        output.append(
            {
                "source_instance_id": source,
                "target_instance_id": target,
                "adjacency_definition": ADJACENCY_DEFINITION,
                "shared_boundary_side_count": shared_sides,
                "left_right_contact_count": int(vertical[index]),
                "top_bottom_contact_count": int(horizontal[index]),
                "source_area_pixels": int(areas[source]),
                "target_area_pixels": int(areas[target]),
                "source_perimeter_pixel_sides": source_perimeter,
                "target_perimeter_pixel_sides": target_perimeter,
                "source_shared_boundary_fraction": float(shared_sides / source_perimeter),
                "target_shared_boundary_fraction": float(shared_sides / target_perimeter),
                "minimum_perimeter_shared_boundary_fraction": float(
                    shared_sides / min(source_perimeter, target_perimeter)
                ),
            }
        )
    return output


def region_color_statistics(
    instance_map: np.ndarray, rgb: np.ndarray
) -> dict[int, dict[str, Any]]:
    """Return normalized RGB mean/std statistics for every positive instance."""

    values = _instance_map(instance_map)
    colors = _rgb(rgb, values.shape)
    positive = values > 0
    if not np.any(positive):
        return {}
    identifiers, inverse, counts = np.unique(
        values[positive].astype(np.int64), return_inverse=True, return_counts=True
    )
    pixels = colors[positive]
    sums = np.zeros((len(identifiers), 3), dtype=np.float64)
    squares = np.zeros_like(sums)
    np.add.at(sums, inverse, pixels)
    np.add.at(squares, inverse, pixels * pixels)
    means = sums / counts[:, None]
    variances = np.maximum(0.0, squares / counts[:, None] - means * means)
    stds = np.sqrt(variances)
    return {
        int(identifier): {
            "area_pixels": int(counts[index]),
            "mean_rgb_01": means[index].tolist(),
            "std_rgb_01": stds[index].tolist(),
        }
        for index, identifier in enumerate(identifiers)
    }


def boundary_color_evidence(
    instance_map: np.ndarray,
    rgb: np.ndarray,
    pairs: Iterable[tuple[int, int]] | None = None,
) -> list[dict[str, Any]]:
    """Measure color discontinuity across each requested shared interface.

    RGB values are normalized to ``[0, 1]``.  The L2 delta is normalized by
    ``sqrt(3)``, so all returned color-difference values also lie in ``[0, 1]``.
    Only directly opposing pixels across the actual shared sides contribute.
    """

    values = _instance_map(instance_map)
    colors = _rgb(rgb, values.shape)
    requested = _pair_filter(pairs)
    contacts = _scan_contacts(values)
    if contacts is None:
        return []

    unique_pairs, groups = np.unique(contacts["pairs"], axis=0, return_inverse=True)
    first_rgb = colors[contacts["first_rows"], contacts["first_columns"]]
    second_rgb = colors[contacts["second_rows"], contacts["second_columns"]]
    absolute = np.abs(first_rgb - second_rgb)
    l1 = absolute.mean(axis=1)
    l2 = np.linalg.norm(first_rgb - second_rgb, axis=1) / np.sqrt(3.0)
    order = np.argsort(groups, kind="stable")
    group_counts = np.bincount(groups, minlength=len(unique_pairs)).astype(np.int64)
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(group_counts)])
    sorted_l1 = l1[order]
    sorted_l2 = l2[order]

    output: list[dict[str, Any]] = []
    for index, pair_array in enumerate(unique_pairs):
        pair = (int(pair_array[0]), int(pair_array[1]))
        if requested is not None and pair not in requested:
            continue
        start, stop = int(offsets[index]), int(offsets[index + 1])
        pair_l1 = sorted_l1[start:stop]
        pair_l2 = sorted_l2[start:stop]
        output.append(
            {
                "source_instance_id": pair[0],
                "target_instance_id": pair[1],
                "shared_boundary_side_count": int(stop - start),
                "mean_interface_rgb_l1_delta": float(pair_l1.mean()),
                "mean_interface_rgb_l2_delta": float(pair_l2.mean()),
                "median_interface_rgb_l2_delta": float(np.median(pair_l2)),
                "p90_interface_rgb_l2_delta": float(np.percentile(pair_l2, 90)),
                "weak_color_boundary_score": float(1.0 - pair_l2.mean()),
            }
        )
    return output


def pair_evidence(instance_map: np.ndarray, rgb: np.ndarray) -> list[dict[str, Any]]:
    """Combine adjacency and appearance evidence for all merge/absorb pairs."""

    contacts = adjacency_contacts(instance_map)
    if not contacts:
        return []
    boundary = {
        (row["source_instance_id"], row["target_instance_id"]): row
        for row in boundary_color_evidence(instance_map, rgb)
    }
    regions = region_color_statistics(instance_map, rgb)
    output: list[dict[str, Any]] = []
    for contact in contacts:
        source = contact["source_instance_id"]
        target = contact["target_instance_id"]
        source_mean = np.asarray(regions[source]["mean_rgb_01"], dtype=np.float64)
        target_mean = np.asarray(regions[target]["mean_rgb_01"], dtype=np.float64)
        mean_distance = float(np.linalg.norm(source_mean - target_mean) / np.sqrt(3.0))
        row = dict(contact)
        row.update(boundary[(source, target)])
        row["region_mean_rgb_l2_delta"] = mean_distance
        row["region_mean_color_similarity"] = float(1.0 - mean_distance)
        output.append(row)
    return output


def _scan_contacts(values: np.ndarray) -> dict[str, np.ndarray] | None:
    left = values[:, :-1]
    right = values[:, 1:]
    right_mask = (left > 0) & (right > 0) & (left != right)
    right_rows, right_columns = np.nonzero(right_mask)
    right_first = left[right_mask].astype(np.int64, copy=False)
    right_second = right[right_mask].astype(np.int64, copy=False)

    upper = values[:-1, :]
    lower = values[1:, :]
    down_mask = (upper > 0) & (lower > 0) & (upper != lower)
    down_rows, down_columns = np.nonzero(down_mask)
    down_first = upper[down_mask].astype(np.int64, copy=False)
    down_second = lower[down_mask].astype(np.int64, copy=False)
    if len(right_first) + len(down_first) == 0:
        return None

    first = np.concatenate([right_first, down_first])
    second = np.concatenate([right_second, down_second])
    pairs = np.column_stack([np.minimum(first, second), np.maximum(first, second)])
    return {
        "pairs": pairs,
        "vertical": np.concatenate(
            [np.ones(len(right_first), dtype=bool), np.zeros(len(down_first), dtype=bool)]
        ),
        "first_rows": np.concatenate([right_rows, down_rows]),
        "first_columns": np.concatenate([right_columns, down_columns]),
        "second_rows": np.concatenate([right_rows, down_rows + 1]),
        "second_columns": np.concatenate([right_columns + 1, down_columns]),
    }


def _areas(values: np.ndarray) -> dict[int, int]:
    identifiers, counts = np.unique(values[values > 0], return_counts=True)
    return {int(identifier): int(count) for identifier, count in zip(identifiers, counts)}


def _perimeters(values: np.ndarray) -> dict[int, int]:
    padded = np.pad(values, 1, constant_values=0)
    center = padded[1:-1, 1:-1]
    positive = center > 0
    boundary_ids = np.concatenate(
        [
            center[positive & (center != padded[:-2, 1:-1])],
            center[positive & (center != padded[2:, 1:-1])],
            center[positive & (center != padded[1:-1, :-2])],
            center[positive & (center != padded[1:-1, 2:])],
        ]
    )
    identifiers, counts = np.unique(boundary_ids, return_counts=True)
    return {int(identifier): int(count) for identifier, count in zip(identifiers, counts)}


def _instance_map(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or any(length <= 0 for length in array.shape):
        raise ValueError(f"instance_map must have a non-empty 2-D shape, got {array.shape}")
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"instance_map must use an integer dtype, got {array.dtype}")
    if np.any(array < 0):
        raise ValueError("instance_map may contain only zero and positive IDs")
    return array


def _rgb(value: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (*shape, 3):
        raise ValueError(f"rgb must have shape {(*shape, 3)}, got {array.shape}")
    if array.dtype == np.bool_ or not np.issubdtype(array.dtype, np.number):
        raise TypeError("rgb must use a numeric dtype")
    colors = array.astype(np.float64)
    if not np.isfinite(colors).all() or colors.size == 0:
        raise ValueError("rgb values must be finite")
    minimum, maximum = float(colors.min()), float(colors.max())
    if minimum < 0:
        raise ValueError("rgb values cannot be negative")
    if np.issubdtype(array.dtype, np.integer):
        if maximum <= 255.0:
            return colors / 255.0
        raise ValueError("integer rgb values must be in [0, 255]")
    if maximum <= 1.0:
        return colors
    if maximum <= 255.0:
        return colors / 255.0
    raise ValueError("rgb values must be in [0, 1] or [0, 255]")


def _pair_filter(pairs: Iterable[tuple[int, int]] | None) -> set[tuple[int, int]] | None:
    if pairs is None:
        return None
    result: set[tuple[int, int]] = set()
    for value in pairs:
        if isinstance(value, (str, bytes)):
            raise ValueError("Each pair must contain exactly two positive instance IDs")
        try:
            source, target = value
        except (TypeError, ValueError) as exc:
            raise ValueError("Each pair must contain exactly two positive instance IDs") from exc
        if isinstance(source, (bool, np.bool_)) or isinstance(target, (bool, np.bool_)):
            raise ValueError("Pair instance IDs must be positive integers")
        if not isinstance(source, (int, np.integer)) or not isinstance(target, (int, np.integer)):
            raise ValueError("Pair instance IDs must be positive integers")
        source, target = int(source), int(target)
        if source <= 0 or target <= 0 or source == target:
            raise ValueError("Pair instance IDs must be distinct and positive")
        result.add((min(source, target), max(source, target)))
    return result


__all__ = [
    "ADJACENCY_DEFINITION",
    "adjacency_contacts",
    "boundary_color_evidence",
    "pair_evidence",
    "region_color_statistics",
]
