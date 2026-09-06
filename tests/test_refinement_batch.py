from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cli import main as cli_main
from src.refinement import apply_refinement
from src.refinement.batch import (
    discover_refinement_tiles,
    run_refinement_batch,
)
from tests.test_refinement import _write_stage2_fixture


def test_automatic_mode_is_audited_and_rejects_human_decisions(
    tmp_path: Path,
) -> None:
    _, config = _write_stage2_fixture(tmp_path, tile_id="scene__tile_00001")

    result = apply_refinement(
        "scene__tile_00001",
        project_root=tmp_path,
        config=config,
        workflow_mode="automatic",
        show_progress=False,
    )

    assert result.report["workflow_mode"] == "automatic"
    assert result.report["automatic_selection"]["policy_version"] == (
        "automatic_selective_v1"
    )
    manifest = json.loads(
        (result.output_directory / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["workflow_mode"] == "automatic"
    assert manifest["automatic_policy_version"] == "automatic_selective_v1"

    with pytest.raises(ValueError, match="generates its own decisions"):
        apply_refinement(
            "scene__tile_00001",
            project_root=tmp_path,
            config=config,
            workflow_mode="automatic",
            decisions={"unused": "approved"},
            show_progress=False,
        )


def test_batch_discovers_orders_processes_and_resumes_tiles(tmp_path: Path) -> None:
    _, config = _write_stage2_fixture(tmp_path, tile_id="scene__tile_00002")
    _write_stage2_fixture(tmp_path, tile_id="scene__tile_00001")

    assert discover_refinement_tiles(project_root=tmp_path, config=config) == [
        "scene__tile_00001",
        "scene__tile_00002",
    ]

    first = run_refinement_batch(
        project_root=tmp_path,
        config=config,
        workflow_mode="automatic",
        workers=2,
        show_progress=False,
    )
    first_summary = first.summary()
    assert first_summary["status"] == "complete"
    assert first_summary["status_counts"] == {"complete": 2}
    assert [row["tile_id"] for row in first_summary["tiles"]] == [
        "scene__tile_00001",
        "scene__tile_00002",
    ]
    assert first.report_path.is_file()

    resumed = run_refinement_batch(
        project_root=tmp_path,
        config=config,
        workflow_mode="automatic",
        workers=1,
        show_progress=False,
    ).summary()
    assert resumed["status"] == "complete"
    assert resumed["status_counts"] == {"skipped": 2}


def test_batch_records_a_bad_tile_without_losing_good_tiles(tmp_path: Path) -> None:
    _, config = _write_stage2_fixture(tmp_path, tile_id="good__tile_00000")
    _write_stage2_fixture(
        tmp_path,
        tile_id="bad__tile_00000",
        public_manifest_overrides={"stale": True},
    )

    summary = run_refinement_batch(
        project_root=tmp_path,
        config=config,
        workflow_mode="automatic",
        continue_on_error=True,
        show_progress=False,
    ).summary()

    assert summary["status"] == "partial"
    assert summary["success_count"] == 1
    assert summary["failure_count"] == 1
    by_tile = {row["tile_id"]: row for row in summary["tiles"]}
    assert by_tile["good__tile_00000"]["status"] == "complete"
    assert by_tile["bad__tile_00000"]["status"] == "failed"
    assert "stale" in by_tile["bad__tile_00000"]["error"]["message"].lower()


def test_refine_auto_and_batch_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, config = _write_stage2_fixture(tmp_path, tile_id="cli__tile_00000")
    config.pop("_config_path", None)
    config_path = tmp_path / "project.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    common = [
        "--project-root",
        str(tmp_path),
        "--config",
        str(config_path),
        "--no-progress",
    ]

    cli_main(
        [
            "refine",
            "auto",
            "--tile-id",
            "cli__tile_00000",
            *common,
        ]
    )
    automatic = json.loads(capsys.readouterr().out)
    assert automatic["workflow_mode"] == "automatic"

    cli_main(["refine", "batch", *common])
    batch = json.loads(capsys.readouterr().out)
    assert batch["status"] == "complete"
    assert batch["status_counts"] == {"skipped": 1}

