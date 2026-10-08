import json
from types import SimpleNamespace
import pytest
from src import pipeline_select as ps


def test_changed_input_rejects_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, '_execute_stage', lambda *a: {})
    data = tmp_path / 'data'; data.mkdir()
    image = data / 'tile.png'; image.write_bytes(b'original pixels')
    ps.segment('new', tmp_path / 'run', data=data)
    image.write_bytes(b'changed pixels')
    with pytest.raises(ValueError, match='changed'):
        ps.graph('new', tmp_path / 'run')


def test_graph_failure_invalidates_completed_change(tmp_path, monkeypatch):
    monkeypatch.setattr(ps, '_execute_stage', lambda *a: {})
    ps.run('new', 'new', 'classical', tmp_path)
    def fail(*args): raise RuntimeError('broken graph')
    monkeypatch.setattr(ps, '_execute_stage', fail)
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


def test_selected_image_fingerprint_includes_discovered_placement_and_source(tmp_path):
    from src.pipeline_state import input_fingerprint
    from tests.semantic_fixtures import make_year
    folder=make_year(tmp_path);options={'images':[str(folder/'a.png')]}
    original=input_fingerprint(options)
    (folder/'metadata.csv').write_text((folder/'metadata.csv').read_text()+'\n')
    assert input_fingerprint(options)!=original
    original=input_fingerprint(options)
    with (folder/'source.tif').open('ab') as stream: stream.write(b'changed')
    assert input_fingerprint(options)!=original


def test_old_model_config_fingerprint_binds_referenced_weights(tmp_path):
    from src.pipeline_state import input_fingerprint
    import yaml
    checkpoint=tmp_path/'custom.pt';checkpoint.write_bytes(b'old weights')
    config=tmp_path/'custom.yaml';config.write_text(yaml.safe_dump({'stage2':{'instance_model':{'checkpoint':'custom.pt'}}}))
    options={'project_root':str(tmp_path),'config':str(config)}
    original=input_fingerprint(options);checkpoint.write_bytes(b'changed weights')
    assert input_fingerprint(options)!=original


def test_same_path_gnn_checkpoint_change_recomputes_only_change(tmp_path,monkeypatch):
    from pipelines.semantic_temporal.runtime.storage import atomic_json,save_manifest
    checkpoint=tmp_path/'external.pt';checkpoint.write_bytes(b'first model')
    checkpoint.with_suffix('.json').write_text('{}');calls=[]
    def execute(output,stage,method,options,state):
        calls.append(stage);directory=output/stage;directory.mkdir(exist_ok=True)
        report=directory/'report.json';atomic_json(report,{'stage':stage})
        paths={'report':report};atomic_json(directory/'artifact.json',{'paths':paths});save_manifest(directory,paths)
        return {'artifacts':[str(directory/'artifact.json')]}
    monkeypatch.setattr(ps,'_execute_stage',execute)
    output=tmp_path/'run'
    ps.run('new','new','gnn',output,project_root=tmp_path,gnn_checkpoint=checkpoint)
    checkpoint.write_bytes(b'second model')
    ps.run('new','new','gnn',output,resume=True)
    assert calls==['segmentation','graph','change','change']
