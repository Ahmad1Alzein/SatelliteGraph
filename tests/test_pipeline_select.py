"""Selectable pipeline: option wiring, refusal of non-native combinations, and the run ledger.

The real stage runners are replaced by fakes; the end-to-end runs are covered by the experiment's own
verification (pipelines/semantic_temporal/verify_outputs.py) and the existing folder-pipeline tests.
"""
from __future__ import annotations

import json

import pytest

from src import cli
from src import pipeline_select as ps


@pytest.fixture
def fakes(monkeypatch):
    calls=[]
    monkeypatch.setattr(ps,'_execute_stage',lambda out,stage,method,options,state:calls.append((stage,method)) or {'effective':options})
    return calls


@pytest.mark.parametrize('combo',sorted(ps.NATIVE))
def test_all_combinations_run_in_order(tmp_path,fakes,combo):
    result=ps.run(*combo,tmp_path/'run',folders=['a/2017','b/2023'])
    assert list(result)==['segmentation','graph','change']
    assert fakes==list(zip(('segmentation','graph','change'),combo))
    state=json.loads((tmp_path/'run/run.json').read_text())
    assert all(v['status']=='complete' for v in state['stages'].values())


def test_stages_require_their_predecessor(tmp_path,fakes):
    with pytest.raises(ValueError,match='segmentation'): ps.graph('new',tmp_path/'run')
    ps.segment('new',tmp_path/'run')
    with pytest.raises(ValueError,match='graph'): ps.change('gnn',tmp_path/'run')


def test_failed_stage_is_recorded(tmp_path,monkeypatch):
    def boom(*args): raise RuntimeError('stage did not complete')
    monkeypatch.setattr(ps,'_execute_stage',boom)
    with pytest.raises(RuntimeError): ps.segment('new',tmp_path/'run')
    assert json.loads((tmp_path/'run/run.json').read_text())['stages']['segmentation']['status']=='failed'


def test_old_pipeline_needs_two_year_folders():
    with pytest.raises(ValueError, match="--folders"):
        ps._years({})


def test_cli_pipeline_commands_parse():
    a = cli.parser().parse_args(["pipeline", "run", "--segmentation", "new", "--graph", "new", "--change", "gnn",
                                 "--output", "artifacts/runs/x"])
    assert (a.stage, a.phase, a.segmentation, a.graph, a.change) == ("pipeline", "run", "new", "new", "gnn")
    assert a.data is None and a.checkpoints is None
    a = cli.parser().parse_args(["pipeline", "segment", "--method", "old", "--output", "o", "--folders", "x/2013", "y/2023"])
    assert a.method == "old" and a.folders == ["x/2013", "y/2023"]
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["pipeline", "change", "--method", "deep", "--output", "o"])


def test_cli_dispatches_to_selector(tmp_path, fakes, capsys):
    cli.main(["pipeline", "run", "--segmentation", "new", "--graph", "new", "--change", "classical",
              "--output", str(tmp_path / "run"), "--project-root", str(tmp_path)])
    out = json.loads(capsys.readouterr().out)
    assert out["change"]["method"] == "classical" and fakes[-1] == ("change", "classical")


def test_experiment_package_is_complete():
    pkg = ps.PACKAGE
    for f in ("run_experiment.py", "verify_outputs.py", "README.md", "pipeline/run_tile_pipeline.py",
              "reference/input_manifest.json", "reference/expected_outputs.json", "reference/expected_results.json",
              "reference/audit_verdicts.csv", "reference/shift_samples.csv", "environment/environment_windows_exact.yml"):
        assert (pkg / f).is_file(), f
    assert len(json.loads((pkg / "reference" / "input_manifest.json").read_text())) == 24
    assert len(json.loads((pkg / "reference" / "expected_outputs.json").read_text())) == 551
