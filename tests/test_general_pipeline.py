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
