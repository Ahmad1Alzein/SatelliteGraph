import numpy as np
import pytest
from PIL import Image
from shapely.geometry import box,mapping
from tests.test_matching_pngs import matching_fixture
from pipelines.semantic_temporal.runtime.storage import read_json,atomic_json
from pipelines.semantic_temporal.runtime.change_contracts import TASKS


def report_fixture(tmp_path,count=3):
    graph,context,frame=matching_fixture(tmp_path,max(1,count))
    events=[{'event_id':f'event-{i}','kind':'conversion','source_class':3,'target_class':1,
             'geometry':mapping(box(i,89,i+2,91)),'area':4,'area_units':'metres squared',
             'source_nodes':[f'2013-{i}'],'target_nodes':[f'2023-{i}'],
             'related_tasks':['tree_loss','building_gain'],'status':'season_sensitive' if i==0 else 'candidate',
             'descriptors':{'building_gain':['appearance']}} for i in range(count)]
    catalog={'years':[2013,2023],'events':events,'compound_events':[],
             'structural_observations':[],'transition_matrix':{'pixels':np.zeros((8,8),int).tolist()},'coverage':{}}
    atomic_json(graph.directory/'classical_changes.json',catalog)
    return graph,context,frame,catalog


def test_combined_report_has_every_event_once(tmp_path):
    from pipelines.semantic_temporal.runtime.reports.changes import render_change_results
    graph,context,frame,catalog=report_fixture(tmp_path)
    catalog['compound_events']=[{'event_id':'parent','kind':'reconfiguration','class_id':1,'event_ids':['event-0','event-1']}]
    paths=render_change_results(graph,context,frame,catalog,None)
    index=read_json(paths['change_index']);crop_ids=[r['event_id'] for r in index['crops']]
    event_ids=[e['event_id'] for e in catalog['events']]
    assert set(crop_ids)==set(event_ids) and len(crop_ids)==len(set(crop_ids))
    for crop in index['crops']:
        assert crop['years']==[2013,2023]
        assert crop['before_bounds']==crop['after_bounds']
    assert 'Season may explain this' in index['crops'][0]['badge']
    assert len(index['compound_groups'])==1
    assert paths['change_report'].is_file()


def test_edge_crop_uses_same_world_window_without_wrapping(tmp_path):
    from pipelines.semantic_temporal.runtime.reports.changes import event_crops
    graph,context,frame,catalog=report_fixture(tmp_path,1)
    for year in frame['years']:frame['data'][year]['rgb_original'][10,0]=[250,30,20]
    panels,record=event_crops(frame,catalog['events'][0],size=(224,224))
    assert record['before_bounds']==record['after_bounds']
    a,b=map(np.asarray,panels)
    np.testing.assert_array_equal(a,b)
    assert not np.array_equal(a[:,0],a[:,-1]) or record['before_bounds'][0]<0


def test_unsupported_scores_are_unavailable_and_all_classes_visible(tmp_path):
    from pipelines.semantic_temporal.runtime.reports.changes import render_change_results
    graph,context,frame,catalog=report_fixture(tmp_path,0)
    result={'status':'insufficient_data','head_status':{task:{'available':False,'status':'insufficient_data'} for task in TASKS}}
    paths=render_change_results(graph,context,frame,catalog,result)
    index=read_json(paths['change_index'])
    assert len(index['tasks'])==16
    assert all(row['score'] is None and not row['available'] for row in index['tasks'])
    assert len(index['transition_matrix']['classes'])==8
    assert np.asarray(index['transition_matrix']['pixels']).shape==(8,8)
    assert not np.asarray(index['transition_matrix']['pixels']).any()
    assert index['crops']==[]
    with Image.open(paths['change_report']) as image:assert image.height>100


def test_structural_observations_stay_separate(tmp_path):
    from pipelines.semantic_temporal.runtime.reports.changes import render_change_results
    graph,context,frame,catalog=report_fixture(tmp_path,0)
    catalog['structural_observations']=[{'id':'structure-1','kind':'split','status':'matching_difference'}]
    paths=render_change_results(graph,context,frame,catalog,None)
    index=read_json(paths['change_index'])
    assert index['crops']==[] and len(index['structural_observations'])==1


def test_report_budget_fails_without_truncation(tmp_path):
    from pipelines.semantic_temporal.runtime.reports.changes import render_change_results
    graph,context,frame,catalog=report_fixture(tmp_path,5)
    context.options['max_report_pixels']=100
    with pytest.raises(ValueError,match='required.*pixels'):
        render_change_results(graph,context,frame,catalog,None)
    assert (graph.directory/'classical_changes.json').is_file()
    assert not (context.output/'results/change_report.png').exists()


def test_long_names_wrap_and_gnn_only_scores_do_not_invent_subtypes(tmp_path):
    from pipelines.semantic_temporal.runtime.reports.changes import render_change_results
    graph,context,frame,catalog=report_fixture(tmp_path,0)
    import pandas as pd
    path=tmp_path/'predictions.csv'
    pd.DataFrame([{'gid':'2023-0','task':'building_gain','probability':.9,'prediction':1,'eligible':True}]).to_csv(path,index=False)
    result={'head_status':{'building_gain':{'available':True,'status':'training_only'}},'predictions':str(path)}
    paths=render_change_results(graph,context,frame,catalog,result)
    index=read_json(paths['change_index'])
    assert len(index['predicted_crops'])==1
    assert index['predicted_crops'][0]['kind']=='gain'
    assert index['predicted_crops'][0]['badge']=='GNN prediction'
    assert not any('New building' in label for label in index['predicted_crops'][0]['labels'])
