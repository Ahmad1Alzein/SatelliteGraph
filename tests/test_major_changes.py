from shapely.geometry import box,mapping


def test_default_focus_excludes_thin_edges_and_vegetation_subclass_switches():
    from pipelines.semantic_temporal.runtime.preview import major_change
    thin=dict(kind='conversion',source_class=1,target_class=4,geometry=mapping(box(0,0,1,100)),source_nodes=[],target_nodes=[])
    assert not major_change(thin,{'nodes':[],'temporal_edges':[]},1)['include']
    seasonal=dict(thin,source_class=4,target_class=5,geometry=mapping(box(0,0,40,40)))
    assert not major_change(seasonal,{'nodes':[],'temporal_edges':[]},1)['include']


def test_major_building_needs_most_of_object_and_no_substantial_same_class_match():
    from pipelines.semantic_temporal.runtime.preview import major_change
    node=dict(id='b',class_id=1,area=400,geometry=mapping(box(0,0,20,20)))
    event=dict(kind='conversion',source_class=4,target_class=1,geometry=node['geometry'],source_nodes=[],target_nodes=['b'])
    document={'nodes':[node],'temporal_edges':[]}
    assert major_change(event,document,1)['include']
    document['nodes'].append(dict(node,id='old'))
    document['temporal_edges']=[dict(source='old',target='b',fraction_source=.8,fraction_target=.8)]
    assert not major_change(event,document,1)['include']


def test_major_road_and_land_changes_have_clear_interior():
    from pipelines.semantic_temporal.runtime.preview import major_change
    document={'nodes':[],'temporal_edges':[]}
    for a,b in [(6,2),(6,3),(5,7),(7,8)]:
        event=dict(kind='conversion',source_class=a,target_class=b,geometry=mapping(box(0,0,30,30)),source_nodes=[],target_nodes=[])
        assert major_change(event,document,1)['include']


def test_many_small_building_matches_still_mean_the_building_persisted():
    from pipelines.semantic_temporal.runtime.preview import major_change
    node=dict(id='after',class_id=1,geometry=mapping(box(0,0,20,20)))
    previous=[dict(id=f'before-{i}',class_id=1,geometry=mapping(box(i*2,0,(i+1)*2,20))) for i in range(10)]
    doc={'nodes':[node]+previous,'temporal_edges':[
        dict(source=n['id'],target='after',fraction_source=1.,fraction_target=.1) for n in previous]}
    event=dict(kind='gain',class_id=1,geometry=node['geometry'],source_nodes=[],target_nodes=['after'])
    assert not major_change(event,doc,1)['include']


def test_original_built_over_predictions_are_visible_in_major_change_focus():
    from pipelines.semantic_temporal.runtime.reports.changes import _predicted_events
    from pipelines.semantic_temporal.runtime.preview import event_rows,filter_rows
    land=dict(id='earlier-land',year=2013,class_id=3,confidence=.9,
        area=900,geometry=mapping(box(0,0,30,30)))
    document={'nodes':[land],'temporal_edges':[]}
    scores={('earlier-land','built_over'):dict(probability=.9,prediction=1,status='training_only')}
    events=_predicted_events(document,{'events':[]},scores)
    rows=event_rows(events,document,scores,native_pixel=1.)
    assert rows[0]['gnn_score']==.9
    assert len(filter_rows(rows,major_only=True))==1


def test_report_always_provides_major_change_summary_png(tmp_path):
    from tests.test_change_report_pngs import report_fixture
    from pipelines.semantic_temporal.runtime.reports.changes import render_change_results
    from pipelines.semantic_temporal.runtime.storage import read_json
    graph,context,frame,catalog=report_fixture(tmp_path,2)
    paths=render_change_results(graph,context,frame,catalog,None)
    assert paths['major_changes'].is_file()
    index=read_json(paths['change_index'])
    assert index['major_change_focus']['shown']==0
    assert len(index['crops'])==2  # complete scientific review output is preserved
