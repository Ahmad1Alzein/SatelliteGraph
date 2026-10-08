import json
from types import SimpleNamespace
import pytest
from src import pipeline_select as ps


def test_changed_input_rejects_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, '_run_new', lambda *a: {})
    data = tmp_path / 'data'; data.mkdir()
    image = data / 'tile.png'; image.write_bytes(b'original pixels')
    ps.segment('new', tmp_path / 'run', data=data)
    image.write_bytes(b'changed pixels')
    with pytest.raises(ValueError, match='changed'):
        ps.graph('new', tmp_path / 'run')


def test_graph_failure_invalidates_completed_change(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, '_run_new', lambda *a: {})
    ps.run('new', 'new', 'classical', tmp_path)
    def fail(*args): raise RuntimeError('broken graph')
    monkeypatch.setattr(ps, '_run_new', fail)
    with pytest.raises(RuntimeError): ps.graph('new', tmp_path)
    state = json.loads((tmp_path / 'run.json').read_text())
    assert state['stages']['graph']['status'] == 'failed'
    assert 'change' not in state['stages']


def test_failed_process_cannot_use_stale_complete_ledger(tmp_path, monkeypatch):
    logs = tmp_path / 'work' / 'logs'; logs.mkdir(parents=True)
    (logs / 'stages.json').write_text(json.dumps({'completed':['segmentation']}))
    monkeypatch.setattr(ps.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError, match='exit 1'):
        ps._run_new(tmp_path, 'segmentation', {})
