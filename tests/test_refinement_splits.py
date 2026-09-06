from __future__ import annotations

import copy

import numpy as np
import pytest

from src.refinement.records import make_component_entity_id
from src.refinement.schema import make_proposal, validate_proposal
from src.refinement.splits import (
    canonical_semantic_split_partition,
    make_semantic_split_candidate,
    make_semantic_split_candidates,
    semantic_split_partition_checksum,
)
from src.refinement.topology import (
    apply_reviewed_operations,
    normalize_4_connected_components,
)


TILE_ID = "scene__tile_split"


def _normalized_fixture():
    instance_map = np.full((6, 6), 2, dtype=np.int32)
    instance_map[1:5, 1:5] = 1
    semantic_map = np.full((6, 6), 7, dtype=np.uint8)
    confidence_map = np.full((6, 6), 0.9, dtype=np.float32)
    source_map = np.full((6, 6), 2, dtype=np.uint8)
    valid_mask = np.ones((6, 6), dtype=bool)
    records = [
        {
            "instance_id": instance_id,
            "segment_id": f"stage2_segment_{instance_id:06d}",
            "source_tile_id": TILE_ID,
            "label": {"id": 7, "name": "buildings", "method": "fixture"},
            "semantic_class": "buildings",
            "class_name": "buildings",
            "provenance": {"fixture": True},
        }
        for instance_id in (1, 2)
    ]
    return normalize_4_connected_components(
        instance_map,
        semantic_map,
        confidence_map,
        source_map,
        valid_mask,
        records,
        tile_id=TILE_ID,
    )


def _valid_partition() -> np.ndarray:
    partition = np.zeros((6, 6), dtype=np.int32)
    partition[1:5, 1:3] = 1
    partition[1:5, 3:5] = 2
    return partition


def _parameters(
    partition: np.ndarray,
    *,
    child_count: int = 2,
    minimum_child_area_pixels: int = 1,
) -> dict:
    canonical = canonical_semantic_split_partition(partition)
    return {
        "partition_sha256": semantic_split_partition_checksum(canonical),
        "partition_shape": list(canonical.shape),
        "partition_dtype": "int32",
        "child_count": child_count,
        "child_label_order": "ascending_contiguous_from_1",
        "minimum_child_area_pixels": minimum_child_area_pixels,
        "review_only": True,
        "automatic": False,
    }


def _proposal_for(partition: np.ndarray, *, minimum_child_area_pixels: int = 1) -> dict:
    return make_proposal(
        TILE_ID,
        "semantic_split",
        [make_component_entity_id(TILE_ID, 1, 0)],
        parameters=_parameters(
            partition,
            minimum_child_area_pixels=minimum_child_area_pixels,
        ),
        evidence={"fixture": True},
        decision="pending",
    )


def test_reviewed_semantic_split_is_explicit_exact_and_replayable() -> None:
    normalized = _normalized_fixture()
    parent = normalized.masks[0]
    candidate = make_semantic_split_candidate(
        TILE_ID,
        normalized.records[0]["entity_id"],
        parent,
        _valid_partition().astype(np.uint8),
        minimum_child_area_pixels=4,
        evidence={"boundary_source": "review_fixture"},
        score=0.75,
    )
    proposal = candidate.proposal
    proposal_id = proposal["proposal_id"]
    assert proposal["operation_type"] == "semantic_split"
    assert proposal["decision"] == "pending"
    assert candidate.partition_map.dtype == np.int32

    unchanged = apply_reviewed_operations(
        normalized,
        [proposal],
        decisions=None,
        split_partitions={proposal_id: candidate.partition_map},
    )
    np.testing.assert_array_equal(unchanged.instance_map, normalized.instance_map)
    assert len(unchanged.records) == len(normalized.records)
    assert unchanged.operations == normalized.operations

    applied = apply_reviewed_operations(
        normalized,
        [proposal],
        decisions={proposal_id: "approved"},
        split_partitions={proposal_id: candidate.partition_map},
    )
    assert len(applied.records) == 3
    np.testing.assert_array_equal(applied.semantic_map, normalized.semantic_map)
    np.testing.assert_array_equal(applied.confidence_map, normalized.confidence_map)
    np.testing.assert_array_equal(applied.source_map, normalized.source_map)
    np.testing.assert_array_equal(applied.valid_mask, normalized.valid_mask)
    children = applied.records[:2]
    assert [record["semantic_split_child_label"] for record in children] == [1, 2]
    assert all(record["stage2_metadata"] == normalized.records[0]["stage2_metadata"] for record in children)
    assert all(record["source_stage2_instance_ids"] == [1] for record in children)
    assert all(record["refinement_history"] for record in children)
    assert np.array_equal(applied.masks[0] | applied.masks[1], parent)
    assert not np.any(applied.masks[0] & applied.masks[1])

    operation = applied.operations[-1]
    assert operation["operation_type"] == "semantic_split"
    assert operation["proposal_id"] == proposal_id
    assert operation["source_instance_ids"] == [1]
    assert operation["output_instance_ids"] == [1, 2]
    assert operation["decision_source"] == "review_decision"
    assert operation["invariants"]["children_union_equals_parent"] is True

    replayed = apply_reviewed_operations(
        normalized,
        [proposal],
        decisions={proposal_id: "approved"},
        split_partitions={proposal_id: candidate.partition_map.copy()},
    )
    np.testing.assert_array_equal(replayed.instance_map, applied.instance_map)
    assert replayed.records == applied.records
    assert replayed.operations == applied.operations


def test_semantic_split_never_accepts_embedded_approval_or_auto_apply() -> None:
    proposal = _proposal_for(_valid_partition())
    embedded = copy.deepcopy(proposal)
    embedded["decision"] = "approved"
    with pytest.raises(ValueError, match="separate explicit review"):
        validate_proposal(embedded)
    with pytest.raises(ValueError, match="never|separate explicit review"):
        make_proposal(
            TILE_ID,
            "semantic_split",
            proposal["source_entity_ids"],
            parameters=proposal["parameters"],
            evidence={},
            decision="auto_apply",
        )


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("checksum", "checksum"),
        ("outside", "outside"),
        ("gap", "cover"),
        ("labels", "contiguous"),
        ("disconnected", "4-connected"),
        ("minimum_area", "below minimum"),
    ],
)
def test_semantic_split_revalidates_partition_before_mutation(case: str, message: str) -> None:
    normalized = _normalized_fixture()
    supplied = _valid_partition()
    proposal_partition = supplied.copy()
    minimum_area = 1
    if case == "checksum":
        supplied = np.where(supplied == 1, 2, np.where(supplied == 2, 1, 0)).astype(np.int32)
    elif case == "outside":
        supplied[0, 0] = 1
        proposal_partition = supplied
    elif case == "gap":
        supplied[2, 2] = 0
        proposal_partition = supplied
    elif case == "labels":
        supplied[supplied == 2] = 3
        proposal_partition = supplied
    elif case == "disconnected":
        for row in range(1, 5):
            for column in range(1, 5):
                supplied[row, column] = 1 + ((row + column) % 2)
        proposal_partition = supplied
    else:
        supplied[1:5, 1:5] = 1
        supplied[1, 1] = 2
        proposal_partition = supplied
        minimum_area = 2

    proposal = _proposal_for(
        proposal_partition,
        minimum_child_area_pixels=minimum_area,
    )
    with pytest.raises(ValueError, match=message):
        apply_reviewed_operations(
            normalized,
            [proposal],
            decisions={proposal["proposal_id"]: "approved"},
            split_partitions={proposal["proposal_id"]: supplied},
        )


def test_semantic_split_batch_helper_and_conflicts_are_deterministic() -> None:
    normalized = _normalized_fixture()
    source = normalized.records[0]["entity_id"]
    proposals, partitions = make_semantic_split_candidates(
        TILE_ID,
        [
            {
                "source_entity_id": source,
                "parent_mask": normalized.masks[0],
                "partition_map": _valid_partition(),
                "minimum_child_area_pixels": 2,
            }
        ],
    )
    assert list(partitions) == [proposals[0]["proposal_id"]]
    split = proposals[0]
    merge = make_proposal(
        TILE_ID,
        "merge",
        [normalized.records[0]["entity_id"], normalized.records[1]["entity_id"]],
        parameters={"review_only": True},
        evidence={},
        decision="pending",
    )
    decisions = {
        split["proposal_id"]: "approved",
        merge["proposal_id"]: "approved",
    }
    with pytest.raises(ValueError, match="already replaced|Conflicting"):
        apply_reviewed_operations(
            normalized,
            [split, merge],
            decisions=decisions,
            split_partitions=partitions,
        )
