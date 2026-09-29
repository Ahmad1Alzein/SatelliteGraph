from __future__ import annotations

import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from src.cli import main as cli_main
from src.config import DEFAULT_CONFIG, load_config
from src.refinement import (
    apply_refinement,
    load_refinement_result,
    propose_refinement,
    validate_refinement_result,
)
from src.refinement.topology import (
    consolidate_micro_components,
    normalize_4_connected_components,
)
from src.stage2.artifacts import atomic_json, atomic_npz, checksum, manifest, now
from src.stage2.pipeline import SCHEMA_VERSION as STAGE2_SCHEMA_VERSION
from src.stage2.storage import save_masks


TILE_ID = "scene__tile_00000"


def _stage2_map() -> np.ndarray:
    """Return a gap-free map whose first ID is connected only diagonally."""

    return np.asarray(
        [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 1, 1],
            [3, 3, 1, 1],
        ],
        dtype=np.int32,
    )


def _expected_refined_map() -> np.ndarray:
    """IDs follow parent-ID order, then top-left component order."""

    return np.asarray(
        [
            [1, 1, 3, 3],
            [1, 1, 3, 3],
            [4, 4, 2, 2],
            [4, 4, 2, 2],
        ],
        dtype=np.int32,
    )


def _record(tile_id: str, instance_id: int, name: str, mask: np.ndarray) -> dict:
    rows, cols = np.nonzero(mask)
    x0, y0 = int(cols.min()), int(rows.min())
    x1, y1 = int(cols.max()) + 1, int(rows.max()) + 1
    area = int(mask.sum())
    label_id = {"road": 3, "tree": 4, "buildings": 7, "unknown": 8}[name]
    centroid_xy = [float(cols.mean() + 0.5), float(rows.mean() + 0.5)]
    return {
        "instance_id": instance_id,
        "segment_id": f"stage2_segment_{instance_id:06d}",
        "source_tile_id": tile_id,
        "source": "openearthmap_geometry_and_label",
        "source_code": 2,
        "area_pixels": area,
        "bbox_xywh": [x0, y0, x1 - x0, y1 - y0],
        "bbox_xyxy": [x0, y0, x1, y1],
        "centroid_rc": [float(rows.mean()), float(cols.mean())],
        "centroid_xy": centroid_xy,
        "touches_tile_boundary": bool(x0 == 0 or y0 == 0 or x1 == 4 or y1 == 4),
        "mean_semantic_confidence": 0.95,
        "label": {
            "id": label_id,
            "name": name,
            "source_model": "openearthmap_fasterseg",
            "method": "fixture",
            "evidence": {"fixture": True},
            "explanation": "Synthetic, deterministic refinement fixture.",
        },
        "geometry_source": {
            "model": "openearthmap_fasterseg",
            "method": "fixture",
            "explanation": "Synthetic, deterministic refinement fixture.",
        },
        "metrics": {
            "area_pixels": area,
            "area_ratio": area / 16,
            "valid_pixel_fraction": 1.0,
            "nodata_fraction": 0.0,
            "fill_ratio": area / ((x1 - x0) * (y1 - y0)),
            "touches_tile_boundary": bool(x0 == 0 or y0 == 0 or x1 == 4 or y1 == 4),
        },
        "coordinates": {
            "tile_pixel": {
                "bbox_xyxy": [x0, y0, x1, y1],
                "centroid_xy": centroid_xy,
            },
            "source_pixel": {
                "bbox_xyxy": [x0, y0, x1, y1],
                "centroid_xy": centroid_xy,
            },
            "global_pixel": {
                "bbox_xyxy": [x0, y0, x1, y1],
                "centroid_xy": centroid_xy,
            },
            "projected": {"available": False},
            "wgs84": {"available": False},
        },
        "provenance": {"fixture": True},
        "explanation": "Synthetic Stage 2 segment used by refinement tests.",
    }


def _write_stage2_fixture(
    tmp_path: Path,
    *,
    tile_id: str = TILE_ID,
    public_manifest_overrides: dict | None = None,
    labels_override: dict[int, str] | None = None,
) -> tuple[Path, dict]:
    """Write the smallest authoritative Stage 2 artifact accepted downstream."""

    stage2_root = tmp_path / "artifacts" / "stage2"
    final = stage2_root / tile_id / "12_final"
    final.mkdir(parents=True)

    instance_map = _stage2_map()
    labels = labels_override or {1: "tree", 2: "buildings", 3: "road"}
    masks = [instance_map == instance_id for instance_id in range(1, 4)]
    records = [
        _record(tile_id, instance_id, labels[instance_id], masks[instance_id - 1])
        for instance_id in range(1, 4)
    ]
    semantic_map = np.zeros(instance_map.shape, dtype=np.uint8)
    for record, mask in zip(records, masks):
        semantic_map[mask] = int(record["label"]["id"])
    valid_mask = np.ones(instance_map.shape, dtype=bool)

    save_masks(final / "segments.npz", final / "segments.json", masks, records)
    atomic_npz(
        final / "segmentation.npz",
        instance_map=instance_map,
        semantic_map=semantic_map,
        confidence_map=np.full(instance_map.shape, 0.95, dtype=np.float16),
        source_map=np.full(instance_map.shape, 2, dtype=np.uint8),
        valid_mask=valid_mask,
    )
    atomic_json(
        final / "tile_metadata.json",
        {
            "tile_id": tile_id,
            "source_tile_id": tile_id,
            "source_image_id": "scene",
            "tile_name": "tile_00000.png",
            "width": 4,
            "height": 4,
            "global_x0": 0,
            "global_y0": 0,
            "global_x1": 4,
            "global_y1": 4,
            "tile_affine_transform": None,
            "crs": None,
            "crs_wkt": None,
            "georeferencing_unavailable_reason": "synthetic pixel-only fixture",
        },
    )
    atomic_json(
        final / "report.json",
        {
            "schema_version": STAGE2_SCHEMA_VERSION,
            "tile_id": tile_id,
            "segment_count": 3,
            "every_valid_pixel_exactly_once": True,
            "coverage": {
                "valid_pixel_count": 16,
                "assigned_valid_pixel_count": 16,
                "unassigned_valid_pixel_count": 0,
                "coverage_percentage": 100.0,
                "overlap_pixel_count": 0,
            },
            "coordinate_reference": {
                "pixel": {"available": True},
                "projected": {"available": False},
                "geographic": {"available": False},
            },
        },
    )
    atomic_json(
        final / "class_registry.json",
        {
            "schema_version": STAGE2_SCHEMA_VERSION,
            "classes": [
                {"id": 3, "name": "road"},
                {"id": 4, "name": "tree"},
                {"id": 7, "name": "buildings"},
                {"id": 8, "name": "unknown"},
            ],
        },
    )
    preview = np.zeros((4, 4, 3), dtype=np.uint8)
    preview[semantic_map == 3] = (190, 190, 190)
    preview[semantic_map == 4] = (45, 145, 55)
    preview[semantic_map == 7] = (220, 70, 70)
    assert cv2.imwrite(str(final / "final_segmentation.png"), preview[..., ::-1])
    assert cv2.imwrite(str(final / "final_overlay.png"), preview[..., ::-1])

    artifacts = {
        "segments": str((final / "segments.json").resolve()),
        "masks": str((final / "segments.npz").resolve()),
        "segmentation_maps": str((final / "segmentation.npz").resolve()),
        "class_registry": str((final / "class_registry.json").resolve()),
        "tile_metadata": str((final / "tile_metadata.json").resolve()),
        "report": str((final / "report.json").resolve()),
        "preview": str((final / "final_segmentation.png").resolve()),
        "overlay": str((final / "final_overlay.png").resolve()),
    }
    public_manifest = {
        "schema_version": STAGE2_SCHEMA_VERSION,
        "status": "complete",
        "tile_id": tile_id,
        "created_at": now(),
        "configuration_hash": "synthetic-stage2-fixture",
        "coordinate_reference": {"pixel": {"available": True}},
        "artifacts": artifacts,
        **(public_manifest_overrides or {}),
    }
    atomic_json(final / "manifest.json", public_manifest)
    outputs = [Path(value) for value in artifacts.values()] + [final / "manifest.json"]
    manifest(
        final / "phase_manifest.json",
        "12_final",
        [],
        outputs,
        "synthetic-stage2-fixture",
        now(),
    )

    config = load_config(DEFAULT_CONFIG)
    config["stage2"]["output"]["root_directory"] = str(stage2_root)
    config["stage3"]["output"]["root_directory"] = str(tmp_path / "artifacts" / "stage3")
    return final, config


def _four_connected_component_count(mask: np.ndarray) -> int:
    count, _ = cv2.connectedComponents(np.ascontiguousarray(mask, dtype=np.uint8), connectivity=4)
    return count - 1


def test_automatic_refinement_reassigns_unknown_regions_below_41_pixels(tmp_path: Path) -> None:
    stage2, config = _write_stage2_fixture(
        tmp_path, labels_override={1: "unknown", 2: "buildings", 3: "road"}
    )
    result = apply_refinement(
        TILE_ID, project_root=tmp_path, config=config,
        workflow_mode="automatic", show_progress=False,
    )
    validate_refinement_result(result)
    assert all(record["label"]["name"] != "unknown" for record in result.segments)
    assert not np.any(result.semantic_map == 8)
    assert result.valid_mask.all()
    assert np.all(result.instance_map > 0)
    assert any(
        operation.get("decision_source") == "automatic_micro_unknown_omission"
        for operation in result.operations
    )
    assert int(result.report["comparison"]["refined_unknown_pixel_count"]) == 0


def test_connectivity_normalization_is_internal_and_micro_fragments_do_not_add_objects(
    tmp_path: Path,
) -> None:
    stage2, config = _write_stage2_fixture(tmp_path)
    original_maps_checksum = checksum(stage2 / "segmentation.npz")

    proposal = propose_refinement(
        TILE_ID,
        project_root=tmp_path,
        config=config,
        show_progress=False,
    )
    assert checksum(stage2 / "segmentation.npz") == original_maps_checksum
    assert proposal.tile_id == TILE_ID
    assert [item["operation_type"] for item in proposal.proposals] == ["split_components"]
    assert proposal.proposals[0]["decision"] == "auto_apply"
    proposal_manifest = json.loads(
        (proposal.output_directory / "manifest.json").read_text(encoding="utf-8")
    )
    assert proposal_manifest["status"] == "proposed"
    assert proposal_manifest["stale"] is True
    with pytest.raises((ValueError, RuntimeError), match="complete|stale"):
        load_refinement_result(proposal.output_directory)

    result = apply_refinement(
        TILE_ID,
        project_root=tmp_path,
        config=config,
        decisions=None,
        show_progress=False,
    )
    validate_refinement_result(result)
    assert result.output_directory == stage2.parent / "13_refinement"
    # The two four-pixel pieces of Stage 2 object 1 are still separated while
    # proposals are evaluated, but neither is large enough to justify a new
    # published object.  Final object identity therefore matches Stage 2.
    np.testing.assert_array_equal(result.instance_map, _stage2_map())
    np.testing.assert_array_equal(result.semantic_map, np.load(stage2 / "segmentation.npz")["semantic_map"])
    assert result.valid_mask.all()
    assert np.all(result.instance_map[result.valid_mask] > 0)
    assert np.array_equal(np.unique(result.instance_map), np.arange(1, 4))
    assert _four_connected_component_count(result.instance_map == 1) == 2
    assert all(
        _four_connected_component_count(result.instance_map == instance_id) == 1
        for instance_id in (2, 3)
    )

    assert [record["stage2_instance_id"] for record in result.segments] == [1, 2, 3]
    assert [record["component_index"] for record in result.segments] == [0, 0, 0]
    assert [record["component_count"] for record in result.segments] == [2, 1, 1]
    assert result.segments[0]["connected_component_count"] == 2
    assert result.segments[0]["is_four_connected"] is False
    assert [record["source_stage2_instance_ids"] for record in result.segments] == [
        [1],
        [2],
        [3],
    ]
    for record in result.segments:
        parent = int(record["stage2_instance_id"])
        assert record["stage2_metadata"]["instance_id"] == parent
        assert record["source_stage2_segment_ids"] == [record["stage2_metadata"]["segment_id"]]
    assert result.segments[0]["refinement_history"]
    assert result.segments[1]["refinement_history"] == []
    assert result.segments[2]["refinement_history"] == []
    identity = result.report["object_identity"]
    assert identity["source_stage2_segment_count"] == 3
    assert identity["internal_connected_component_count"] == 4
    assert identity["output_object_count"] == 3
    assert identity["consolidated_micro_component_count"] == 1
    assert identity["promoted_large_secondary_component_count"] == 0
    assert identity["multipart_object_count"] == 1
    assert len(result.operations) == 1
    split = result.operations[0]
    assert split["operation_type"] == "split_components"
    assert split["source_instance_ids"] == [1]
    assert split["output_instance_ids"] == [1, 2]
    assert split["decision_source"] == "deterministic_topology_rule"
    assert split["invariants"]["children_union_equals_parent"] is True

    reloaded = load_refinement_result(result.output_directory)
    np.testing.assert_array_equal(reloaded.instance_map, result.instance_map)
    assert reloaded.segments == result.segments
    assert reloaded.operations == result.operations
    assert reloaded.lineage == result.lineage
    assert reloaded.report == result.report

    replayed = apply_refinement(
        TILE_ID,
        project_root=tmp_path,
        config=config,
        decisions=None,
        show_progress=False,
    )
    np.testing.assert_array_equal(replayed.instance_map, result.instance_map)
    assert replayed.segments == result.segments
    assert replayed.operations == result.operations
    assert replayed.lineage == result.lineage
    assert checksum(stage2 / "segmentation.npz") == original_maps_checksum


def test_topology_alone_never_increases_the_published_object_count() -> None:
    instance_map = np.full((10, 10), 2, dtype=np.int32)
    instance_map[1:4, 1:4] = 1  # nine-pixel anchor
    instance_map[7:9, 7:9] = 1  # four-pixel secondary component
    semantic_map = np.full(instance_map.shape, 7, dtype=np.uint8)
    semantic_map[instance_map == 1] = 4
    confidence_map = np.full(instance_map.shape, 0.95, dtype=np.float32)
    source_map = np.full(instance_map.shape, 2, dtype=np.uint8)
    valid_mask = np.ones(instance_map.shape, dtype=bool)
    records = [
        _record("area-floor", instance_id, class_name, instance_map == instance_id)
        for instance_id, class_name in ((1, "tree"), (2, "buildings"))
    ]
    normalized = normalize_4_connected_components(
        instance_map,
        semantic_map,
        confidence_map,
        source_map,
        valid_mask,
        records,
        tile_id="area-floor",
    )
    assert len(normalized.records) == 3

    consolidated = consolidate_micro_components(
        normalized,
        minimum_object_component_area_pixels=5,
        source_stage2_segment_count=2,
    )
    assert len(consolidated.records) == 2
    assert consolidated.records[0]["connected_component_count"] == 2
    assert consolidated.report["object_identity"][
        "consolidated_micro_component_count"
    ] == 1

    count_capped = consolidate_micro_components(
        normalized,
        minimum_object_component_area_pixels=4,
        source_stage2_segment_count=2,
    )
    assert len(count_capped.records) == 2
    assert count_capped.report["object_identity"][
        "promoted_large_secondary_component_count"
    ] == 0
    assert count_capped.report["object_identity"][
        "budget_consolidated_component_count"
    ] == 1
    assert count_capped.report["object_identity"][
        "topology_alone_cannot_increase_object_count"
    ] is True


@pytest.mark.parametrize(
    ("case", "error", "message"),
    [
        ("wrong_tile", ValueError, "different tile|tile"),
        ("wrong_checksum", RuntimeError, "bound|proposals|review"),
        ("unknown_proposal", ValueError, "unknown|proposal"),
        ("connectivity_decision", ValueError, "connectivity|deterministic|reviewable"),
    ],
)
def test_review_decisions_are_bound_and_never_silently_dropped(
    tmp_path: Path,
    case: str,
    error: type[Exception],
    message: str,
) -> None:
    _, config = _write_stage2_fixture(tmp_path)
    proposal = propose_refinement(
        TILE_ID,
        project_root=tmp_path,
        config=config,
        show_progress=False,
    )
    connectivity_id = proposal.proposals[0]["proposal_id"]

    if case in {"wrong_tile", "wrong_checksum"}:
        decision_document = json.loads(
            Path(proposal.artifacts["decisions_template"]).read_text(encoding="utf-8")
        )
        if case == "wrong_tile":
            decision_document["tile_id"] = "different__tile_00000"
        else:
            decision_document["proposal_checksum"] = "0" * 64
        decisions_path = tmp_path / f"{case}.json"
        decisions_path.write_text(json.dumps(decision_document), encoding="utf-8")
        decisions: object = decisions_path
    elif case == "unknown_proposal":
        decisions = {"missing::proposal::id": "approved"}
    else:
        decisions = {connectivity_id: "approved"}

    with pytest.raises(error, match=message):
        apply_refinement(
            TILE_ID,
            project_root=tmp_path,
            config=config,
            decisions=decisions,
            show_progress=False,
        )


def test_review_bundle_resolves_candidates_and_uncertain_template_is_noop(
    tmp_path: Path,
) -> None:
    stage2, config = _write_stage2_fixture(
        tmp_path,
        labels_override={1: "tree", 2: "tree", 3: "road"},
    )
    preprocessing = stage2.parent / "02_preprocessing"
    preprocessing.mkdir()
    np.save(
        preprocessing / "sam_input.npy",
        cv2.cvtColor(
            cv2.imread(str(stage2 / "final_segmentation.png"), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        ),
        allow_pickle=False,
    )
    # This test exercises the review bundle, not the production eligibility
    # thresholds, so make the tiny synthetic interface reviewable explicitly.
    config["refinement"]["operations"]["merge"].update(
        {
            "minimum_shared_boundary_pixels": 1,
            "minimum_shared_boundary_fraction": 0.01,
            "score_threshold": 0.0,
        }
    )
    proposal = propose_refinement(
        TILE_ID,
        project_root=tmp_path,
        config=config,
        show_progress=False,
    )
    pending = [item for item in proposal.proposals if item["decision"] == "pending"]
    assert pending
    assert {item["operation_type"] for item in pending} <= {"merge", "absorb"}

    proposals_path = Path(proposal.artifacts["proposals"])
    template_path = Path(proposal.artifacts["decisions_template"])
    review_index_path = Path(proposal.artifacts["review_index"])
    review_index = json.loads(review_index_path.read_text(encoding="utf-8"))
    template = json.loads(template_path.read_text(encoding="utf-8"))
    assert review_index["proposal_checksum"] == checksum(proposals_path)
    assert template["proposal_checksum"] == checksum(proposals_path)
    assert review_index["candidate_count"] == len(pending)
    assert template["decision_count"] == len(pending)
    assert {item["proposal_id"] for item in template["decisions"]} == {
        item["proposal_id"] for item in pending
    }
    assert {item["decision"] for item in template["decisions"]} == {"uncertain"}

    normalized_records = json.loads(
        Path(proposal.artifacts["review_segments"]).read_text(encoding="utf-8")
    )["records"]
    normalized_entity_ids = {str(record["entity_id"]) for record in normalized_records}
    for item in review_index["candidates"]:
        reviewed = item["proposal"]
        referenced = set(str(value) for value in reviewed["source_entity_ids"])
        if reviewed["target_entity_id"] is not None:
            referenced.add(str(reviewed["target_entity_id"]))
        assert referenced <= normalized_entity_ids
        assert item["source_entities"]
        assert isinstance(reviewed["evidence"], dict)
        crop_path = review_index_path.parent / item["review_crop"]["path"]
        assert crop_path.is_file()
        assert checksum(crop_path) == item["review_crop"]["sha256"]
        crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        assert crop is not None and crop.size > 0

    with np.load(proposal.artifacts["review_segmentation"], allow_pickle=False) as archive:
        normalized_instance_map = np.asarray(archive["instance_map"]).copy()
    result = apply_refinement(
        TILE_ID,
        project_root=tmp_path,
        config=config,
        decisions=template_path,
        show_progress=False,
    )
    assert not np.array_equal(normalized_instance_map, _stage2_map())
    np.testing.assert_array_equal(result.instance_map, _stage2_map())
    assert not any(
        operation["operation_type"] in {"merge", "absorb"}
        for operation in result.operations
    )
    assert result.decisions == template["decisions"]


def test_human_and_automatic_modes_generate_the_same_proposal_set(
    tmp_path: Path,
) -> None:
    stage2, config = _write_stage2_fixture(
        tmp_path,
        labels_override={1: "tree", 2: "tree", 3: "road"},
    )
    preprocessing = stage2.parent / "02_preprocessing"
    preprocessing.mkdir()
    np.save(
        preprocessing / "sam_input.npy",
        cv2.cvtColor(
            cv2.imread(str(stage2 / "final_segmentation.png"), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        ),
        allow_pickle=False,
    )
    config["refinement"]["operations"]["merge"].update(
        {
            "minimum_shared_boundary_pixels": 1,
            "minimum_shared_boundary_fraction": 0.01,
            "score_threshold": 0.0,
        }
    )

    human = propose_refinement(
        TILE_ID,
        project_root=tmp_path,
        config=config,
        show_progress=False,
    )
    automatic = apply_refinement(
        TILE_ID,
        project_root=tmp_path,
        config=config,
        workflow_mode="automatic",
        show_progress=False,
    )

    assert automatic.proposals == human.proposals
    pending_ids = {
        proposal["proposal_id"]
        for proposal in human.proposals
        if proposal["decision"] == "pending"
    }
    assert pending_ids
    assert {decision["proposal_id"] for decision in automatic.decisions} == pending_ids
    assert {decision["decision"] for decision in automatic.decisions} == {"uncertain"}
    assert automatic.report["workflow_mode"] == "automatic"


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        (
            {"schema_version": "satellitegraph.stage2.unsupported.v999"},
            ValueError,
            "Stage 2 schema|schema",
        ),
        ({"stale": True}, RuntimeError, "stale"),
        ({"status": "incomplete"}, ValueError, "complete|status"),
        ({"tile_id": "different__tile_00000"}, ValueError, "tile"),
    ],
)
def test_proposal_rejects_wrong_incomplete_or_stale_stage2(
    tmp_path: Path,
    overrides: dict,
    error: type[Exception],
    message: str,
) -> None:
    _, config = _write_stage2_fixture(tmp_path, public_manifest_overrides=overrides)
    with pytest.raises(error, match=message):
        propose_refinement(
            TILE_ID,
            project_root=tmp_path,
            config=config,
            show_progress=False,
        )


def test_validation_forbids_deleting_any_valid_pixel(tmp_path: Path) -> None:
    _, config = _write_stage2_fixture(tmp_path)
    result = apply_refinement(
        TILE_ID,
        project_root=tmp_path,
        config=config,
        decisions=None,
        show_progress=False,
    )
    corrupted = copy.deepcopy(result)
    corrupted.instance_map = corrupted.instance_map.copy()
    corrupted.instance_map[0, 0] = 0
    with pytest.raises(ValueError, match="valid|unassigned|coverage"):
        validate_refinement_result(corrupted)


def test_stage3_consumes_the_refined_segmentation_when_configured(tmp_path: Path) -> None:
    from src.stage3 import run_stage3

    _, config = _write_stage2_fixture(tmp_path)
    refined = apply_refinement(
        TILE_ID,
        project_root=tmp_path,
        config=config,
        decisions=None,
        show_progress=False,
    )
    graph = run_stage3(
        TILE_ID,
        project_root=tmp_path,
        config=config,
        show_progress=False,
    )

    assert len(graph.nodes) == len(refined.segments) == 3
    assert [node["instance_id"] for node in graph.nodes] == [1, 2, 3]
    assert graph.nodes[0]["geometry_type"] == "MultiPolygon"
    for node, refined_record in zip(graph.nodes, refined.segments):
        assert node["segment_id"] == refined_record["segment_id"]
        assert node["area_pixels"] == refined_record["area_pixels"]
        assert node["segment_type"]["id"] == refined_record["label"]["id"]
        assert "refinement_metadata" not in node
        assert "stage2_metadata" not in node
        assert "source_stage2_metadata" not in node
    from rasterio.features import rasterize

    restored = rasterize(
        [(node["geometry"], node["instance_id"]) for node in graph.nodes],
        out_shape=refined.instance_map.shape,
        dtype="int32",
    )
    np.testing.assert_array_equal(restored, refined.instance_map)
    source = graph.graph["source_refinement"]
    assert Path(source["output_directory"]).resolve() == refined.output_directory.resolve()


def test_refinement_cli_propose_apply_and_validate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, config = _write_stage2_fixture(tmp_path)
    config.pop("_config_path", None)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    common = [
        "--tile-id",
        TILE_ID,
        "--project-root",
        str(tmp_path),
        "--config",
        str(config_path),
        "--no-progress",
    ]

    cli_main(["refine", "propose", *common])
    proposed = json.loads(capsys.readouterr().out)
    assert proposed["status"] == "proposed"
    assert proposed["proposal_count"] == 1

    cli_main(["refine", "apply", *common])
    applied = json.loads(capsys.readouterr().out)
    assert applied["status"] == "complete"
    assert applied["segment_count"] == 3

    cli_main(["refine", "validate", *common])
    validated = json.loads(capsys.readouterr().out)
    assert validated["status"] == "valid"
    assert validated["validation"]["valid"] is True
