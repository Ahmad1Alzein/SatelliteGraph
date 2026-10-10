import numpy as np
import pytest
from tests.test_all_class_evidence import frame_for,small_policy
from tests.test_all_class_catalog import catalog
from tests.semantic_fixtures import make_year
from tests.test_general_pipeline import input_pair
from pipelines.semantic_temporal.runtime.storage import read_json,atomic_json


def test_boundary_tolerance_removes_small_displacements_from_candidates():
    from pipelines.semantic_temporal.runtime.all_class_evidence import detect_evidence
    before=np.full((30,40),6,np.uint8);after=before.copy()
    before[:,10:20]=2;after[:,12:22]=2
    frame=frame_for(before,after)
    initial=detect_evidence(frame,small_policy())
    assert initial['candidate_masks']['road_gain'].any()
    result=detect_evidence(frame,small_policy(),boundary_tolerance=2)
    assert not result['candidate_masks']['road_gain'].any()
    assert not result['candidate_masks']['road_loss'].any()
    assert not result['masks']['road_gain'].any()


@pytest.mark.parametrize('harmonize',[False,True])
def test_prepared_context_tampering_is_rejected(tmp_path,harmonize):
    from pipelines.semantic_temporal.runtime.context import build_context,load_context
    from pipelines.semantic_temporal.runtime.registration import prepare_comparison
    from pipelines.semantic_temporal.runtime.preprocessing import prepare_model_inputs
    folders=[make_year(tmp_path,y) for y in (2013,2023)]
    context=prepare_model_inputs(prepare_comparison(build_context(
        {'folders':list(map(str,folders)),'harmonize_images':harmonize},tmp_path/'run')))
    path=context.output/'context.json';value=read_json(path)
    value['years'][0]['tiles'][0]['grid']['affine'][2]+=100
    atomic_json(path,value)
    with pytest.raises(ValueError,match='context|alignment|changed'):
        load_context(context.output)


def test_checkpoint_caption_does_not_claim_current_area_training():
    from pipelines.semantic_temporal.runtime.reports.changes import _event_text
    event={'event_id':'a','kind':'gain','class_id':1,'status':'predicted',
        'area':10,'related_tasks':['building_gain'],'target_nodes':['n']}
    scores={('n','building_gain'):{'probability':.8,'status':'inference','probability_source':'deployment'}}
    lines,_=_event_text(event,scores)
    assert any('saved checkpoint' in line.lower() for line in lines)
    assert not any('trained on this area' in line for line in lines)


def test_one_season_sensitive_class_keeps_ambiguity_badge(tmp_path):
    result=catalog(tmp_path,np.full((24,24),3,np.uint8),np.full((24,24),4,np.uint8))
    assert result['events'][0]['season_sensitive']
    assert result['events'][0]['status']=='season_sensitive'


def test_segmentation_restart_keeps_prepared_alignment(tmp_path,input_pair,monkeypatch):
    from src import pipeline_select as ps
    import pipelines.semantic_temporal.runtime.runner as runner
    output=tmp_path/'run';ps.segment('new',output,**input_pair)
    monkeypatch.setattr(runner,'prepare_comparison',lambda *a:pytest.fail('Repeated upstream alignment'))
    ps.segment('new',output)
