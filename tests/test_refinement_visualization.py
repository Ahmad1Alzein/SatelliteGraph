from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.refinement.visualization import (
    refinement_change_mask,
    save_refinement_visualizations,
)


def test_change_mask_ignores_global_instance_id_renumbering() -> None:
    before_instances = np.asarray(
        [
            [10, 10, 20, 20],
            [10, 10, 20, 20],
        ],
        dtype=np.int32,
    )
    after_instances = np.asarray(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ],
        dtype=np.int32,
    )
    semantics = np.asarray(
        [
            [4, 4, 7, 7],
            [4, 4, 7, 7],
        ],
        dtype=np.uint8,
    )

    changed = refinement_change_mask(
        before_instances,
        after_instances,
        before_semantic_map=semantics,
        after_semantic_map=semantics,
    )

    assert not changed.any()


def test_change_mask_marks_true_semantic_relabel_pixels() -> None:
    before_instances = np.asarray([[10, 10, 20], [10, 10, 20]], dtype=np.int32)
    after_instances = np.asarray([[1, 1, 2], [1, 1, 2]], dtype=np.int32)
    before_semantics = np.asarray([[8, 8, 4], [8, 8, 4]], dtype=np.uint8)
    after_semantics = np.full((2, 3), 4, dtype=np.uint8)

    changed = refinement_change_mask(
        before_instances,
        after_instances,
        before_semantic_map=before_semantics,
        after_semantic_map=after_semantics,
    )

    np.testing.assert_array_equal(
        changed,
        np.asarray([[True, True, False], [True, True, False]]),
    )


def test_change_mask_marks_only_added_or_removed_instance_boundaries() -> None:
    before_instances = np.full((3, 4), 10, dtype=np.int32)
    after_instances = np.asarray(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [1, 1, 2, 2],
        ],
        dtype=np.int32,
    )
    semantics = np.full((3, 4), 4, dtype=np.uint8)

    changed = refinement_change_mask(
        before_instances,
        after_instances,
        before_semantic_map=semantics,
        after_semantic_map=semantics,
    )

    np.testing.assert_array_equal(
        changed,
        np.asarray(
            [
                [False, True, True, False],
                [False, True, True, False],
                [False, True, True, False],
            ]
        ),
    )


def test_change_mask_validates_semantic_maps() -> None:
    instances = np.ones((2, 2), dtype=np.int32)

    with pytest.raises(ValueError, match="before_semantic_map"):
        refinement_change_mask(
            instances,
            instances,
            before_semantic_map=np.ones((1, 2), dtype=np.uint8),
            after_semantic_map=np.ones((2, 2), dtype=np.uint8),
        )

    with pytest.raises(TypeError, match="after_semantic_map"):
        refinement_change_mask(
            instances,
            instances,
            before_semantic_map=np.ones((2, 2), dtype=np.uint8),
            after_semantic_map=np.ones((2, 2), dtype=np.float32),
        )


def test_saved_before_after_does_not_highlight_renumbered_ids(tmp_path: Path) -> None:
    before_instances = np.asarray(
        [[10, 10, 20, 20], [10, 10, 20, 20]], dtype=np.int32
    )
    after_instances = np.asarray([[1, 1, 2, 2], [1, 1, 2, 2]], dtype=np.int32)
    semantics = np.asarray([[4, 4, 7, 7], [4, 4, 7, 7]], dtype=np.uint8)
    valid = np.ones((2, 4), dtype=bool)

    paths = save_refinement_visualizations(
        tmp_path,
        image=np.zeros((2, 4, 3), dtype=np.uint8),
        instance_map=after_instances,
        semantic_map=semantics,
        valid_mask=valid,
        before_instance_map=before_instances,
        before_semantic_map=semantics,
        show_ids=False,
    )

    panel = np.asarray(Image.open(paths["before_after"]).convert("RGB"))
    magenta = np.all(panel == np.asarray((255, 0, 255), dtype=np.uint8), axis=2)
    assert not magenta.any()
