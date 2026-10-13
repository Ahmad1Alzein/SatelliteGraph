from pathlib import Path
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from tests.test_general_pipeline import input_pair

ROOT = Path(__file__).resolve().parents[1]


def test_dashboard_exposes_independent_stage_choices(monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.delenv('SATELLITEGRAPH_RUN', raising=False)
    app = AppTest.from_file(str(ROOT / 'thesis_dashboard.py')).run(timeout=120)
    app.switch_page('dashboard/pages/pipelines.py')
    # AppTest hashes the filename, while this mounted page has a custom URL.
    from streamlit.util import calc_md5
    app._page_hash = calc_md5('image_comparison')
    app.run(timeout=60)
    assert not app.exception
    for key, choices in [('pipeline_segmentation', ['new', 'old']),
                         ('pipeline_graph', ['new', 'old']),
                         ('pipeline_change', ['gnn', 'classical'])]:
        selector = app.selectbox(key=key)
        assert len(selector.options) == 2
        selector.set_value(choices[-1])
    app.run(timeout=60)
    assert not app.exception
    assert app.session_state['pipeline_segmentation'] == 'old'
    assert app.session_state['pipeline_graph'] == 'old'
    assert app.session_state['pipeline_change'] == 'classical'


def test_selectable_gnn_page_works_without_frozen_results(tmp_path,monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv('SG_LEGACY_RUN',str(tmp_path/'absent_frozen_run'))
    monkeypatch.setenv('SG_RUNS',str(tmp_path/'absent_pipeline_runs'))
    app=AppTest.from_file(str(ROOT/'thesis_dashboard.py')).run(timeout=120)
    app.switch_page('dashboard/pages/gnn.py').run(timeout=60)
    assert not app.exception
    assert any('pipelines/legacy_building_stgnn/original/gnn_change.py' in code.value for code in app.code)


@pytest.mark.parametrize('graph_method',['old','new'])
def test_retained_gnn_reuses_original_training_and_has_no_experimental_heads(tmp_path,monkeypatch,graph_method):
    from pipelines.semantic_temporal.runtime.gnn import run_gnn
    from pipelines.semantic_temporal.runtime.temporal import detect_classical
    from pipelines.legacy_building_stgnn import runner
    from tests.test_semantic_temporal import temporal_fixture
    from pipelines.legacy_building_stgnn.original import gnn_change as original
    calls=[]
    train=original.train_fold
    def tracked(*args,**kwargs):
        calls.append(kwargs['epochs'])
        np.testing.assert_array_equal(args[3],args[4])  # explicitly a same-input fit
        return train(*args,**kwargs)
    monkeypatch.setattr(original,'train_fold',tracked)
    graph, context = temporal_fixture(tmp_path,graph_method)
    detect_classical(graph, context)
    result = run_gnn(graph, replace(context, options={'gnn_epochs': 5}))
    assert result['model_source'] == 'pipelines/legacy_building_stgnn/original/gnn_change.py'
    assert result['status'] == 'training_only'
    assert set(result['head_status']) == {'building_gain', 'built_over'}
    assert result['checkpoint'] is None  # original implementation does not store weights
    predictions = pd.read_csv(result['predictions'])
    assert set(predictions.task) <= {'building_gain', 'built_over'}
    assert np.isfinite(predictions.probability.dropna()).all()
    assert runner.check_code() == []
    assert calls==[5]
    from pipelines.semantic_temporal.runtime.comparison import load_comparison
    from pipelines.semantic_temporal.runtime.reports.changes import render_change_results
    from pipelines.semantic_temporal.runtime.storage import read_json
    paths=render_change_results(graph,context,load_comparison(graph,context),
        read_json(graph.directory/'classical_changes.json'),result)
    index=read_json(paths['change_index'])
    assert {t['task'] for t in index['tasks']}=={'building_gain','built_over'}


def test_experimental_checkpoints_are_rejected_without_altering_results(tmp_path):
    from pipelines.semantic_temporal.runtime.gnn import run_gnn
    from tests.test_semantic_temporal import temporal_fixture
    graph, context = temporal_fixture(tmp_path)
    with pytest.raises(ValueError, match='experimental|checkpoint'):
        run_gnn(graph, context, checkpoint=tmp_path / 'experimental.pt')


def test_classical_dashboard_loader_needs_no_gnn_report(tmp_path, monkeypatch):
    import json
    from dashboard import data as D
    run = tmp_path / 'arbitrary_name'
    (run / 'temporal').mkdir(parents=True)
    (run / 'run.json').write_text(json.dumps({'stages': {'change': {'method': 'classical', 'result': {}}}}))
    (run / 'temporal/classical_changes.json').write_text(json.dumps({'events': []}))
    monkeypatch.setenv('SG_RUNS', str(tmp_path))
    assert D.run_gnn('arbitrary_name')['status'] == 'not_selected'
    assert D.run_events('arbitrary_name').empty
    assert 'status' in D.run_events('arbitrary_name')


def test_dashboard_loaders_refresh_when_same_run_changes_method(tmp_path,monkeypatch):
    import json
    from dashboard import data as D
    run=tmp_path/'switched_run';(run/'temporal').mkdir(parents=True)
    monkeypatch.setenv('SG_RUNS',str(tmp_path))
    state={'stages':{'change':{'method':'gnn','result':{'status':'training_only'}}}}
    (run/'run.json').write_text(json.dumps(state))
    (run/'temporal/classical_changes.json').write_text(json.dumps({'events':[]}))
    assert D.run_state('switched_run')['stages']['change']['method']=='gnn'
    assert D.run_gnn('switched_run')['status']=='training_only'
    assert D.run_events('switched_run').empty
    state['stages']['change']={'method':'classical','result':{}}
    (run/'run.json').write_text(json.dumps(state))
    event=dict(event_id='event1',source_class=1,target_class=1,kind='gain',status='candidate',
        season_sensitive=False,area=1.,area_units='pixels squared',related_tasks=['building_gain'],source_nodes=[],target_nodes=[],
        geometry={'type':'Polygon','coordinates':[[[0,0],[1,0],[1,1],[0,0]]]})
    (run/'temporal/classical_changes.json').write_text(json.dumps({'events':[event]}))
    assert D.run_state('switched_run')['stages']['change']['method']=='classical'
    assert D.run_gnn('switched_run')['status']=='not_selected'
    assert len(D.run_events('switched_run'))==1
    assert D.run_events('switched_run').iloc[0].area_units=='pixels squared'
    assert 'area_m2' not in D.run_events('switched_run')


@pytest.mark.parametrize('method', ['old', 'new'])
def test_old_change_engine_is_independent_of_graph_method(tmp_path, method):
    from tests.test_semantic_temporal import temporal_fixture
    from pipelines.semantic_temporal.runtime.temporal import detect_classical
    graph, context = temporal_fixture(tmp_path, method)
    result = detect_classical(graph, context, engine='stage5_classical')
    assert result['engine'] == 'stage5_classical'
    assert any(e['kind'] == 'gain' and e['class_id'] == 1 for e in result['events'])
    assert all('event_id' in e and 'related_tasks' in e for e in result['events'])


@pytest.mark.parametrize('segmentation', ['old','new'])
@pytest.mark.parametrize('graph', ['old','new'])
@pytest.mark.parametrize('change', ['classical','gnn'])
def test_dashboard_command_uses_selected_methods(tmp_path,segmentation,graph,change):
    from dashboard.pipeline_controls import pipeline_command
    from src.cli import parser
    before=tmp_path/'before.png';after=tmp_path/'after.png'
    before.touch();after.touch()
    command=pipeline_command(tmp_path/'output',segmentation,graph,change,[before,after])
    args=parser().parse_args(command[4:])
    assert (args.segmentation,args.graph,args.change)==(segmentation,graph,change)
    assert args.years==[2013,2023]
    assert args.gnn_checkpoint is None


@pytest.mark.parametrize('graph_method',['old','new'])
def test_original_model_feature_and_relation_contract(tmp_path,graph_method):
    from tests.test_semantic_temporal import temporal_fixture
    from pipelines.legacy_building_stgnn.adapter import original_inputs
    from pipelines.semantic_temporal.runtime.storage import read_json
    graph,_=temporal_fixture(tmp_path,graph_method)
    document=read_json(graph.paths['graph'])
    x,relations=original_inputs(document)
    assert x.shape==(len(document['nodes']),38)
    assert set(relations)=={'spatial_touches','spatial_near','t_primary_fwd','t_primary_bwd','t_overlap_fwd','t_overlap_bwd'}
    assert all(attrs.shape[1]==10 for _,attrs in relations.values())
    assert np.isfinite(x.numpy()).all()
    assert relations['t_primary_fwd'][0].shape[1]==sum(e['edge_role']=='primary' for e in document['temporal_edges'])


def test_classical_run_after_gnn_ignores_saved_gnn_scores(tmp_path,input_pair):
    from src import pipeline_select as ps
    from pipelines.semantic_temporal.runtime.storage import read_json
    ps.run('new','new','gnn',tmp_path/'output',**input_pair)
    ps.change('classical',tmp_path/'output')
    index=read_json(tmp_path/'output/results/changes/index.json')
    assert all(row['score'] is None for row in index['tasks'])
    assert {row['status'] for row in index['tasks']}=={'not_selected'}
    assert not index['predicted_crops']


@pytest.mark.parametrize('probability', [.93, None])
def test_crop_browser_shows_change_scores_without_segmentation_confidence(tmp_path,input_pair,probability):
    from src import pipeline_select as ps
    from pipelines.semantic_temporal.runtime.storage import read_json,atomic_json,bind_result_manifest
    from streamlit.testing.v1 import AppTest
    from streamlit.util import calc_md5
    output=tmp_path/'crop_run'
    ps.run('new','new','classical',output,**input_pair)
    document=read_json(output/'temporal/graph.json')
    node=next(n for n in document['nodes'] if n['year']==2017)
    catalog_path=output/'temporal/classical_changes.json'
    catalog=read_json(catalog_path)
    catalog['events']=[dict(event_id='scored-change',kind='loss',class_id=node['class_id'],
        geometry=node['geometry'],area=node['area'],area_units='metres squared',
        source_nodes=[node['id']],target_nodes=[],related_tasks=['built_over'],status='reliable')]
    atomic_json(catalog_path,catalog)
    state=read_json(output/'run.json')
    if probability is not None:
        predictions=output/'fixture_predictions.csv'
        pd.DataFrame([dict(gid=node['id'],task='built_over',probability=probability,
            prediction=1,status='training_only',probability_source='training_only')]).to_csv(predictions,index=False)
        state['stages']['change'].update(method='gnn',result={
            **state['stages']['change']['result'],
            'model_source':'pipelines/legacy_building_stgnn/original/gnn_change.py',
            'predictions':str(predictions)})
    atomic_json(output/'run.json',state)
    manifest=read_json(output/'results/rendering/change/manifest.json')
    bind_result_manifest(output,'change',
        {str(i):output/r['path'] for i,r in enumerate(manifest['outputs'])},[catalog_path])
    app=AppTest.from_file(str(ROOT/'thesis_dashboard.py'))
    app._page_hash=calc_md5('image_comparison')
    app.session_state['selected_run']=str(output)
    app.run(timeout=90)
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]
    captions=[c.value for c in app.caption]
    assert not any('segment confidence' in c.lower() for c in captions)
    assert not any('segment confidence' in slider.label.lower() for slider in app.slider)
    change_tables=[df.value for df in app.dataframe if 'event_id' in df.value]
    assert change_tables and all('segment_confidence' not in df for df in change_tables)
    expected='GNN change score: 0.930' if probability is not None else 'GNN change score: unavailable'
    assert any(expected in c for c in captions)


@pytest.mark.parametrize('segmentation',['old','new'])
@pytest.mark.parametrize('graph',['old','new'])
def test_old_change_method_accepts_plain_pngs_with_both_upstream_methods(tmp_path,input_pair,segmentation,graph):
    from PIL import Image
    from src import pipeline_select as ps
    from pipelines.semantic_temporal.runtime.storage import read_json
    images=[tmp_path/'before.png',tmp_path/'after.png']
    for image in images:Image.new('RGB',(48,40),(100,110,120)).save(image)
    options={k:v for k,v in input_pair.items() if k!='folders'}
    output=tmp_path/'pixel_run'
    ps.run(segmentation,graph,'classical',output,images=images,years=[2013,2023],**options)
    report=read_json(output/'temporal/classical_changes.json')
    assert report['engine']=='stage5_classical'
    assert report['units']=='pixels'
    assert (output/'results/change_report.png').is_file()
