"""Selectable pipeline: option wiring, refusal of non-native combinations, and the run ledger.

The real stage runners are replaced by fakes; the end-to-end runs are covered by the experiment's own
verification (pipelines/experiment_2013_2023/verify_outputs.py) and the existing folder-pipeline tests.
"""
from __future__ import annotations

import json

import pytest

from src import cli
from src import pipeline_select as ps


@pytest.fixture
def fakes(monkeypatch):
    calls = []
    monkeypatch.setattr(ps, "_run_new", lambda out, until, opts: calls.append(("new", until)) or {"work": str(out / "work")})
    monkeypatch.setattr(ps, "_run_old_annual", lambda out, opts, stop: calls.append(("old_annual", stop)) or {"annual": {}})
    monkeypatch.setattr(ps, "_run_old_classical", lambda out, opts: calls.append(("old_classical", None)) or {"status": "complete"})
    return calls


@pytest.mark.parametrize("combo", sorted(ps.NATIVE))
def test_native_combinations_run_in_order(tmp_path, fakes, combo):
    seg, graph, change = combo
    result = ps.run(seg, graph, change, tmp_path / "run", folders=["a/2013", "b/2023"])
    assert set(result) == {"segmentation", "graph", "change"}
    expected = {("new", "new", "gnn"): [("new", "segmentation"), ("new", "graph"), ("new", "gnn")],
                ("new", "new", "classical"): [("new", "segmentation"), ("new", "graph"), ("new", "classical")],
                ("old", "old", "classical"): [("old_annual", "segmentation"), ("old_annual", None), ("old_classical", None)]}[combo]
    assert fakes == expected
    ledger = json.loads((tmp_path / "run" / "run.json").read_text())
    assert [ledger["stages"][s]["status"] for s in ("segmentation", "graph", "change")] == ["complete"] * 3
    assert [ledger["stages"][s]["method"] for s in ("segmentation", "graph", "change")] == list(combo)


@pytest.mark.parametrize("combo", [("old", "new", "classical"), ("new", "old", "classical"), ("old", "old", "gnn"),
                                   ("new", "old", "gnn"), ("old", "new", "gnn")])
def test_non_native_combinations_are_refused_before_running(tmp_path, fakes, combo):
    with pytest.raises(ValueError):
        ps.run(*combo, tmp_path / "run")
    assert fakes == [] and not (tmp_path / "run").exists()


def test_stages_require_their_predecessor(tmp_path, fakes):
    with pytest.raises(ValueError, match="segmentation"):
        ps.graph("new", tmp_path / "run")
    ps.segment("new", tmp_path / "run")
    with pytest.raises(ValueError, match="graph"):
        ps.change("gnn", tmp_path / "run")


def test_graph_method_must_match_segmentation(tmp_path, fakes):
    ps.segment("old", tmp_path / "run", folders=["a/2013", "b/2023"])
    with pytest.raises(ValueError, match="not supported"):
        ps.graph("new", tmp_path / "run")


def test_old_graph_refuses_gnn(tmp_path, fakes):
    ps.segment("old", tmp_path / "run", folders=["a/2013", "b/2023"])
    ps.graph("old", tmp_path / "run")
    with pytest.raises(ValueError, match="gnn change detection needs the new graph"):
        ps.change("gnn", tmp_path / "run")


def test_failed_stage_is_recorded(tmp_path, monkeypatch):
    def boom(out, until, opts):
        raise RuntimeError("stage did not complete")
    monkeypatch.setattr(ps, "_run_new", boom)
    with pytest.raises(RuntimeError):
        ps.segment("new", tmp_path / "run")
    ledger = json.loads((tmp_path / "run" / "run.json").read_text())
    assert ledger["stages"]["segmentation"]["status"] == "failed"


def test_old_pipeline_needs_two_year_folders():
    with pytest.raises(ValueError, match="--folders"):
        ps._years({})


def test_cli_pipeline_commands_parse():
    a = cli.parser().parse_args(["pipeline", "run", "--segmentation", "new", "--graph", "new", "--change", "gnn",
                                 "--output", "artifacts/runs/x"])
    assert (a.stage, a.phase, a.segmentation, a.graph, a.change) == ("pipeline", "run", "new", "new", "gnn")
    assert a.data == "data/experiment_2013_2023" and a.checkpoints == "checkpoints/experiment_2013_2023"
    a = cli.parser().parse_args(["pipeline", "segment", "--method", "old", "--output", "o", "--folders", "x/2013", "y/2023"])
    assert a.method == "old" and a.folders == ["x/2013", "y/2023"]
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["pipeline", "change", "--method", "deep", "--output", "o"])


def test_cli_dispatches_to_selector(tmp_path, fakes, capsys):
    cli.main(["pipeline", "run", "--segmentation", "new", "--graph", "new", "--change", "classical",
              "--output", str(tmp_path / "run"), "--project-root", str(tmp_path)])
    out = json.loads(capsys.readouterr().out)
    assert out["change"]["method"] == "classical" and fakes[-1] == ("new", "classical")


def test_experiment_package_is_complete():
    pkg = ps.PACKAGE
    for f in ("run_experiment.py", "verify_outputs.py", "README.md", "pipeline/run_tile_pipeline.py",
              "reference/input_manifest.json", "reference/expected_outputs.json", "reference/expected_results.json",
              "reference/audit_verdicts.csv", "reference/shift_samples.csv", "environment/environment_windows_exact.yml"):
        assert (pkg / f).is_file(), f
    assert len(json.loads((pkg / "reference" / "input_manifest.json").read_text())) == 24
    assert len(json.loads((pkg / "reference" / "expected_outputs.json").read_text())) == 551
