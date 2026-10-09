import shutil
from pathlib import Path
import pytest
from tests.test_general_pipeline import input_pair
from pipelines.semantic_temporal.runtime.storage import read_json,sha256,atomic_json


def test_stage_options_and_standalone_results_are_retained(tmp_path,input_pair):
    from src import pipeline_select as ps
    output=tmp_path/'run';ps.segment('new',output,**input_pair,max_report_pixels=3_000_000)
    before={p:sha256(p) for p in (output/'results/segmentation').rglob('*.png')}
    assert before
    ps.graph('new',output);graphs=list((output/'results/graphs').rglob('*.png'));assert graphs
    ps.change('gnn',output)
    assert all(sha256(path)==digest for path,digest in before.items())
    state=read_json(output/'run.json');assert state['options']['max_report_pixels']==3_000_000
    index=read_json(output/'results/index.json');assert 'change_report' in index['images']
    assert state['stages']['change']['scientific_status']=='complete'


def test_render_failure_keeps_science_complete(tmp_path,input_pair,monkeypatch):
    from src import pipeline_select as ps
    output=tmp_path/'run'
    with pytest.raises(ValueError,match='required.*pixels'):
        ps.run('new','new','gnn',output,**input_pair,max_report_pixels=100)
    state=read_json(output/'run.json')
    assert state['stages']['change']['scientific_status']=='complete'
    assert state['stages']['change']['render_status']=='failed'
    assert state['stages']['change']['status']=='render_failed'
    scientific=sha256(output/'temporal/gnn/predictions.csv')
    import pipelines.semantic_temporal.runtime.gnn as gnn
    monkeypatch.setattr(gnn,'run_gnn',lambda *a,**k:pytest.fail('Render retry ran GNN'))
    ps.run('new','new','gnn',output,resume=True,max_report_pixels=3_000_000)
    assert sha256(output/'temporal/gnn/predictions.csv')==scientific
    assert read_json(output/'run.json')['stages']['change']['render_status']=='complete'


def test_change_only_override_keeps_segmentation(tmp_path,input_pair):
    from src import pipeline_select as ps
    output=tmp_path/'run';ps.run('new','new','classical',output,**input_pair)
    artifact=read_json(read_json(output/'run.json')['stages']['segmentation']['result']['artifacts'][0])
    segmentation=Path(artifact['paths']['classes'])
    segmentation_hash_before=sha256(segmentation)
    change_fingerprint_before=read_json(output/'run.json')['stages']['change']['settings_fingerprint']
    policy=tmp_path/'policy.json';atomic_json(policy,{'presence_probability':.7})
    ps.run('new','new','classical',output,resume=True,change_policy=policy)
    segmentation_hash_after=sha256(segmentation)
    change_fingerprint_after=read_json(output/'run.json')['stages']['change']['settings_fingerprint']
    assert segmentation_hash_after==segmentation_hash_before
    assert change_fingerprint_after!=change_fingerprint_before


def test_moved_result_manifests_validate(tmp_path,input_pair):
    from src import pipeline_select as ps
    from pipelines.semantic_temporal.runtime.storage import verify_result_manifest
    output=tmp_path/'run';ps.run('new','new','classical',output,**input_pair)
    moved=tmp_path/'moved';shutil.copytree(output,moved)
    ps.run('new','new','classical',moved,resume=True)
    for stage in ('segmentation','graph','change'):verify_result_manifest(moved,stage)
    from pipelines.semantic_temporal.runtime.storage import verify_manifest
    for path in moved.rglob('manifest.json'):
        if 'files' in read_json(path):verify_manifest(path.parent)


def test_fingerprints_bind_external_files_and_separate_render_budget(tmp_path):
    from src.pipeline_state import graph_fingerprint,render_fingerprint,change_fingerprint
    policy=tmp_path/'policy.json';atomic_json(policy,{'presence_probability':.6})
    opts={'change_policy':str(policy),'max_report_pixels':1000}
    first=change_fingerprint(opts);render=render_fingerprint(opts,'change');graph=graph_fingerprint(opts)
    atomic_json(policy,{'presence_probability':.8})
    assert first!=change_fingerprint(opts)
    assert render==render_fingerprint(opts,'change') and graph==graph_fingerprint(opts)
    opts['max_report_pixels']=2000
    assert render!=render_fingerprint(opts,'change')


def test_graph_profile_change_rebuilds_graph_only(tmp_path,input_pair,monkeypatch):
    from src import pipeline_select as ps,pipeline_state
    output=tmp_path/'run';ps.run('new','new','classical',output,**input_pair)
    original=ps._execute_stage;calls=[]
    def execute(out,stage,*args):calls.append(stage);return original(out,stage,*args)
    monkeypatch.setattr(ps,'_execute_stage',execute)
    monkeypatch.setattr(pipeline_state,'GRAPH_PROFILE','test-new-profile')
    ps.run('new','new','classical',output,resume=True)
    assert calls==['graph','change']


def test_upstream_harmonization_change_rejects_stale_reuse(tmp_path,input_pair):
    from src import pipeline_select as ps
    output=tmp_path/'run';ps.segment('new',output,**input_pair)
    with pytest.raises(ValueError,match='changed'):ps.graph('new',output,harmonize_images=True)


def test_temporal_schema_validation_rejects_wrong_task_order():
    from pipelines.semantic_temporal.runtime.visualization import validate_temporal_schema
    from pipelines.semantic_temporal.runtime.change_contracts import TASKS,RELATIONS,RELATION_PROFILE
    import numpy as np
    specification={'tasks':list(reversed(TASKS)),'relations':list(RELATIONS),'relation_profile':RELATION_PROFILE}
    tensors={'x':np.zeros((0,1)),'edge_attr':np.zeros((0,10)),
        'edge_index':np.empty((2,0),int),'edge_type':np.empty(0,int),'node_ids':np.array([],str)}
    with pytest.raises(ValueError,match='task order'):validate_temporal_schema({'nodes':[]},tensors,specification)


def test_atomic_json_retries_transient_windows_lock(tmp_path,monkeypatch):
    path=tmp_path/'index.json';atomic_json(path,{'state':'old'})
    original=Path.replace;attempts=[]
    def locked(self,target):
        attempts.append(self)
        if len(attempts)<3:raise PermissionError('temporary OneDrive lock')
        return original(self,target)
    monkeypatch.setattr(Path,'replace',locked)
    atomic_json(path,{'state':'new'})
    assert read_json(path)=={'state':'new'} and len(attempts)==3
