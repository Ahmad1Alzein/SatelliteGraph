import json
import shutil
from pathlib import Path
import pytest
from tests.semantic_fixtures import make_year
from tests.test_semantic_segmentation import TreeModels


@pytest.fixture
def input_pair(tmp_path,monkeypatch):
    from src.config import load_config
    import pipelines.semantic_temporal.runtime.segmentation as segmentation
    import pipelines.semantic_temporal.runtime.models as models
    monkeypatch.setattr(segmentation,'load_models',lambda context:TreeModels())
    monkeypatch.setattr(models,'load_models',lambda context:TreeModels())
    folders=[make_year(tmp_path,year=year) for year in (2017,2023)]
    config=load_config();config['stage2']['instance_model']['backend']='fake';config['stage2']['semantic_model']['backend']='fake'
    config['stage2']['filtering']['minimum_area']=1
    config['stage2']['fusion']['minimum_sam_component_area']=1
    config['stage2']['fusion']['minimum_component_area']={name:1 for name in ['bareland','rangeland','developed space','road','tree','water','agriculture land','buildings','unknown']}
    return {'folders':[str(p) for p in folders],'project_root':str(Path(__file__).resolve().parents[1]),'old_config':config,'gnn_epochs':2,'gnn_hidden':8,'inference_window':64,'inference_overlap':16,'tta':False}


@pytest.mark.parametrize('segmentation',['old','new'])
@pytest.mark.parametrize('graph',['old','new'])
@pytest.mark.parametrize('change',['classical','gnn'])
def test_all_eight_combinations_produce_real_artifacts(tmp_path,input_pair,segmentation,graph,change):
    from pipelines.semantic_temporal.runtime.runner import run_pipeline
    output=tmp_path/'run';result=run_pipeline(segmentation,graph,change,output,**input_pair)
    state=json.loads((output/'run.json').read_text())
    assert state['stages']['segmentation']['status']=='complete'
    assert state['stages']['graph']['status']=='complete'
    assert state['stages']['change']['status'] in ('complete','insufficient_data','training_only')
    assert (output/'temporal/graph_tensors.npz').is_file()
    assert (output/'validation.json').is_file()
    assert list((output/'results/segmentation').rglob('*.png'))
    assert list((output/'results/graphs').rglob('*.png'))
    assert (output/'results/change_report.png').is_file()


def test_separate_stages_inherit_single_images_and_relocated_output(tmp_path,input_pair):
    from src import pipeline_select as ps
    options=dict(input_pair);options['images']=[str(Path(p)/'a.png') for p in options.pop('folders')];options['skip_input_check']=True
    old=tmp_path/'original';ps.segment('new',old,**options)
    relocated=tmp_path/'moved run é';shutil.copytree(old,relocated);shutil.rmtree(old)
    ps.graph('new',relocated);ps.change('classical',relocated)
    state=json.loads((relocated/'run.json').read_text())
    assert state['options']['images']==options['images'] and state['options']['skip_input_check']
    assert state['stages']['change']['status']=='complete'


def test_cli_single_year_and_images_folders_exclusion(tmp_path,input_pair):
    from src import cli,pipeline_select as ps
    options=dict(input_pair);options['folders']=options['folders'][:1]
    ps.segment('new',tmp_path/'run',**options);ps.graph('old',tmp_path/'run')
    with pytest.raises(ValueError,match='two'): ps.change('classical',tmp_path/'run')
    with pytest.raises(SystemExit): cli.parser().parse_args(['pipeline','segment','--method','new','--output','o','--images','a.png','--folders','2017'])


def test_resume_rebuilds_dependents_when_graph_method_changes(tmp_path,input_pair):
    from src import pipeline_select as ps
    output=tmp_path/'run'
    ps.run('new','old','classical',output,**input_pair)
    ps.run('new','new','classical',output,resume=True)
    state=json.loads((output/'run.json').read_text())
    assert state['stages']['change']['status']=='complete'
    graph=json.loads((output/'temporal/graph.json').read_text())
    assert graph['method']=='new'


def test_resume_checks_artifact_description_and_data(tmp_path,input_pair):
    from src import pipeline_select as ps
    output=tmp_path/'run'
    ps.run('new','new','classical',output,**input_pair)
    state=json.loads((output/'run.json').read_text())
    artifact=Path(state['stages']['segmentation']['result']['artifacts'][0])
    value=json.loads(artifact.read_text());value['year']=1999
    artifact.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='changed|missing'):
        ps.run('new','new','classical',output,resume=True)


def test_cli_omitted_project_root_inherits_saved_value():
    from src import cli
    args=cli.parser().parse_args(['pipeline','graph','--method','new','--output','o'])
    assert args.project_root is None


def test_staged_and_one_pass_produce_same_graph_tensors(tmp_path,input_pair):
    import numpy as np
    from src import pipeline_select as ps
    first=tmp_path/'one';second=tmp_path/'staged'
    ps.run('new','new','classical',first,**input_pair)
    ps.segment('new',second,**input_pair);ps.graph('new',second);ps.change('classical',second)
    with np.load(first/'temporal/graph_tensors.npz',allow_pickle=False) as a, np.load(second/'temporal/graph_tensors.npz',allow_pickle=False) as b:
        for name in ('x','edge_index','edge_attr','edge_type','node_ids'): np.testing.assert_array_equal(a[name],b[name])


def test_interrupted_segmentation_resumes_verified_tiles(tmp_path,input_pair,monkeypatch):
    from src import pipeline_select as ps
    import pipelines.semantic_temporal.runtime.segmentation as segmentation
    original=segmentation.segment_tile;calls=[]
    def interrupted(tile,context,**options):
        calls.append(tile.year)
        if tile.year==2023: raise RuntimeError('test interruption')
        return original(tile,context,**options)
    monkeypatch.setattr(segmentation,'segment_tile',interrupted)
    output=tmp_path/'run'
    with pytest.raises(RuntimeError,match='interruption'): ps.segment('new',output,**input_pair)
    def resumed(tile,context,**options):
        calls.append(tile.year);return original(tile,context,**options)
    monkeypatch.setattr(segmentation,'segment_tile',resumed)
    ps.run('new','new','classical',output,resume=True)
    assert calls==[2017,2023,2023]
