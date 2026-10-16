"""Review-only regressions: synthetic data, no pipeline or model execution."""
from shapely.geometry import box, mapping
import pytest


def landscape_event(footprint, source='land', target='patch'):
    return dict(event_id='patch', kind='conversion', source_class=3, target_class=7,
        source_nodes=[source], target_nodes=[target], related_tasks=[],
        geometry=mapping(footprint), area=footprint.area, status='candidate')


def test_small_patch_in_large_landscape_is_optional_not_a_major_change():
    from pipelines.semantic_temporal.runtime.preview import event_rows, filter_rows
    patch=box(0,0,30,30)
    event=landscape_event(patch)
    document={'nodes':[
        dict(id='land',class_id=3,geometry=mapping(box(0,0,300,300)),confidence=.8),
        dict(id='patch',class_id=7,geometry=mapping(patch),confidence=.7)]}
    rows=event_rows([event],document,{})
    assert filter_rows(rows,major_only=True)==[]
    assert len(filter_rows(rows))==1  # candidate remains available for review
    assert '10%' in rows[0]['major_change']['reason']


def test_landscape_focus_compares_each_side_as_a_whole_not_each_tiny_sliver():
    from pipelines.semantic_temporal.runtime.preview import major_change
    event=landscape_event(box(0,0,50,50))
    event['source_nodes']=['land','sliver']
    document={'nodes':[
        dict(id='land',class_id=3,geometry=mapping(box(0,0,100,100))),
        dict(id='sliver',class_id=3,geometry=mapping(box(49,49,500,50))),
        dict(id='patch',class_id=7,geometry=event['geometry'])]}
    assert major_change(event,document)['include']


def test_segment_score_keeps_raw_semantic_support_and_limiting_side():
    from pipelines.semantic_temporal.runtime.preview import event_rows
    event=landscape_event(box(0,0,30,30))
    document={'nodes':[dict(id='land',class_id=3,confidence=.81),
        dict(id='patch',class_id=7,confidence=.24,rgb_rule=True,rule_support=.97)]}
    row=event_rows([event],document,{})[0]
    assert row['segment_confidence']==.24
    assert row['before_segment_confidence']==.81
    assert row['after_segment_confidence']==.24
    assert row['gnn_score'] is None
    assert 'segment_confidence' not in event  # source artifact is not rewritten


def test_full_list_sorts_by_segment_score_before_pagination_and_keeps_unknown_last():
    from pipelines.semantic_temporal.runtime.preview import filter_rows
    def row(ident,score):
        return dict(event_id=ident,status='candidate',classes=[3],area=100,
            segment_confidence=score,gnn_score=None)
    rows=[row(str(i),i/20) for i in range(15)]+[
        row('unknown',None),row('nan',float('nan')),row('infinite',float('inf'))]
    selected=filter_rows(rows)
    assert [r['event_id'] for r in selected[:12]]==[str(i) for i in range(14,2,-1)]
    assert [r['event_id'] for r in selected[-3:]]==['unknown','nan','infinite']
    assert [r['event_id'] for r in filter_rows(rows,min_segment_confidence=.6)]==['14','13','12']


def test_grid_caption_includes_honest_segment_score(tmp_path):
    from tests.test_change_report_pngs import report_fixture
    from pipelines.semantic_temporal.runtime.preview import event_rows, grid_image
    _,_,frame,catalog=report_fixture(tmp_path,1)
    row=event_rows(catalog['events'],{'nodes':[dict(id='2023-0',confidence=.35)]},{})[0]
    _,records=grid_image(frame,[row['event']],{})
    assert any('Segmentation score: 0.350' in line for line in records[0]['labels'])
    assert any('GNN change score: unavailable' in line for line in records[0]['labels'])


@pytest.mark.parametrize('score',[float('inf'),float('nan'),-.1,1.1])
def test_invalid_gnn_scores_stay_unavailable_in_cards_and_grid(tmp_path,score):
    from tests.test_change_report_pngs import report_fixture
    from pipelines.semantic_temporal.runtime.preview import event_rows,grid_image
    _,_,frame,catalog=report_fixture(tmp_path,1)
    scores={('2023-0','building_gain'):dict(probability=score)}
    row=event_rows(catalog['events'],{'nodes':[]},scores)[0]
    assert row['gnn_score'] is None
    _,records=grid_image(frame,[row['event']],scores)
    assert any('GNN change score: unavailable' in line for line in records[0]['labels'])


def test_saved_result_cache_refreshes_when_review_profile_changes(tmp_path,monkeypatch):
    import ast
    import inspect
    from pathlib import Path
    import streamlit as st
    from tests.test_change_report_pngs import report_fixture
    from pipelines.semantic_temporal.runtime import context,comparison
    from pipelines.semantic_temporal.runtime.storage import read_json,atomic_json
    from pipelines.semantic_temporal.runtime.preview import event_rows
    from pipelines.semantic_temporal.runtime.reports.changes import _scores,_predicted_events
    graph,run_context,frame,_=report_fixture(tmp_path,1)
    frame['source_pixel_size']=1.
    artifact=graph.directory/'artifact.json';atomic_json(artifact,graph)
    atomic_json(tmp_path/'run.json',{'options':{},'stages':{'change':{
        'render_status':'complete','result':{'artifacts':[str(artifact)]}}}})
    monkeypatch.setattr(context,'load_context',lambda *args:run_context)
    monkeypatch.setattr(comparison,'load_comparison',lambda *args:frame)
    calls=[]
    def tracked_rows(*args,**kwargs):
        calls.append(True)
        return event_rows(*args,**kwargs)
    # Load the actual cached function without executing the interactive page.
    page=Path(__file__).resolve().parents[1]/'dashboard/pages/pipelines.py'
    tree=ast.parse(page.read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='bundle')
    namespace=dict(st=st,Path=Path,read_json=read_json,verify_result_manifest=lambda *args:None,
        _scores=_scores,_predicted_events=_predicted_events,event_rows=tracked_rows)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(page),'exec'),namespace)
    bundle=namespace['bundle']
    assert 'review_profile' in inspect.signature(bundle).parameters
    bundle(str(tmp_path),1,'old-review')
    bundle(str(tmp_path),1,'old-review')
    assert len(calls)==1
    bundle(str(tmp_path),1,'new-review')
    assert len(calls)==2


def test_new_report_pages_share_global_score_order_and_keep_every_event(tmp_path):
    from tests.test_change_report_pngs import report_fixture
    from pipelines.semantic_temporal.runtime.reports.changes import render_change_results
    from pipelines.semantic_temporal.runtime.storage import read_json,atomic_json
    graph,context,frame,catalog=report_fixture(tmp_path,15)
    document=read_json(graph.paths['graph'])
    for node in document['nodes']:
        node['confidence']=.9 if node['year']==2013 else (int(node['id'].split('-')[-1])+1)/20
    atomic_json(graph.paths['graph'],document)
    paths=render_change_results(graph,context,frame,catalog,None)
    index=read_json(paths['change_index'])
    assert [gid for p in index['grid_pages'] for gid in p['event_ids']]==[f'event-{i}' for i in range(14,-1,-1)]
    assert any('Segmentation score: 0.750' in line for line in index['crops'][0]['labels'])
    assert len(index['crops'])==15


def test_browser_filters_and_paginates_all_scores_without_running_a_pipeline(tmp_path):
    from streamlit.testing.v1 import AppTest
    from tests.test_change_report_pngs import report_fixture
    from pipelines.semantic_temporal.runtime.preview import event_rows
    _,_,frame,catalog=report_fixture(tmp_path,15)
    document={'nodes':[dict(id=f'2023-{i}',confidence=(i+1)/20) for i in range(15)]}
    rows=event_rows(catalog['events'],document,{})
    app=AppTest.from_string('''
import streamlit as st
from dashboard.change_review import render_crop_review
render_crop_review(st.session_state.frame, st.session_state.result, {},
    st.session_state.rows, {'change':'classical'}, original_run=False, retired_run=False)
''')
    app.session_state['frame']=frame
    app.session_state['result']={'method':'classical'}
    app.session_state['rows']=rows
    app.run(timeout=60)
    assert not app.exception
    next(c for c in app.checkbox if c.label=='Focus on major changes').uncheck().run()
    score_slider=next(s for s in app.slider if s.label=='Minimum segmentation score')
    assert score_slider.value==0.
    ids=[c.value for c in app.caption if c.value.startswith('event-')]
    assert ids==[f'event-{i}' for i in range(14,2,-1)]
    next(n for n in app.number_input if n.label.startswith('Page (')).set_value(2).run()
    assert [c.value for c in app.caption if c.value.startswith('event-')]==['event-2','event-1','event-0']
    next(s for s in app.slider if s.label=='Minimum segmentation score').set_value(.6).run()
    assert not app.exception
    assert [c.value for c in app.caption if c.value.startswith('event-')]==['event-14','event-13','event-12','event-11']
    assert next(n for n in app.number_input if n.label.startswith('Page (')).value==1
