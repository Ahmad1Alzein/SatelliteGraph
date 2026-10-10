from tests.test_change_report_pngs import report_fixture


def test_filters_do_not_invent_missing_gnn_scores(tmp_path):
    from pipelines.semantic_temporal.runtime.preview import event_rows, filter_rows
    _, _, _, catalog = report_fixture(tmp_path)
    document={'nodes':[{'id':'2023-1','confidence':.7}]}
    rows=event_rows(catalog['events'],document,{})
    assert all(r['gnn_score'] is None for r in rows)
    assert filter_rows(rows,min_gnn_score=.5)==[]
    assert len(filter_rows(rows,statuses=['candidate'],min_area=4))==2
    assert filter_rows(rows,min_area=5)==[]


def test_grid_export_contains_paired_local_cells(tmp_path):
    from pipelines.semantic_temporal.runtime.preview import grid_image
    _, _, frame, catalog = report_fixture(tmp_path)
    image,records=grid_image(frame,catalog['events'],{})
    assert image.width>=900 and image.height<2000
    assert [r['event_id'] for r in records]==[e['event_id'] for e in catalog['events']]
    assert all(r['before_bounds']==r['after_bounds'] for r in records)


def test_pipeline_retains_every_event_in_readable_grid_pages(tmp_path):
    from pipelines.semantic_temporal.runtime.reports.changes import render_change_results
    from pipelines.semantic_temporal.runtime.storage import read_json
    graph,context,frame,catalog=report_fixture(tmp_path,13)
    paths=render_change_results(graph,context,frame,catalog,None)
    pages=read_json(paths['change_index'])['grid_pages']
    assert [len(p['event_ids']) for p in pages]==[12,1]
    assert [gid for p in pages for gid in p['event_ids']]==[e['event_id'] for e in catalog['events']]
    assert all((context.output/p['path']).is_file() for p in pages)


def test_streamlit_upload_page_starts_without_extra_data(monkeypatch):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    monkeypatch.delenv('SATELLITEGRAPH_RUN',raising=False)
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'streamlit_app.py')).run(timeout=30)
    assert not app.exception
    assert app.button[0].label=='Run full comparison' and app.button[0].disabled


def test_external_gnn_support_is_visible_without_rewriting_classical_events():
    from shapely.geometry import box,mapping
    from pipelines.semantic_temporal.runtime.preview import event_rows
    from pipelines.semantic_temporal.runtime.reports.changes import _event_text
    event=dict(event_id='e',kind='conversion',source_class=4,target_class=1,class_id=1,
        status='season_sensitive',source_nodes=['a'],target_nodes=['b'],related_tasks=['grass_shrub_loss','building_gain'],
        geometry=mapping(box(0,0,30,30)),area=900)
    score={('b','building_gain'):dict(probability=.95,prediction=1,status='inference')}
    rows=event_rows([event],{'nodes':[]},score,external_predictions=True)
    assert rows[0]['status']=='season_sensitive'
    assert event['status']=='season_sensitive'
    assert rows[0]['event']['image_evidence_status']=='season_sensitive'
    assert 'Class-label difference' in _event_text(rows[0]['event'],score)[0][0]
    assert 'Removed / decreased' not in rows[0]['type']
    assert event_rows([event],{'nodes':[]},score)[0]['status']=='season_sensitive'


def test_boundary_candidate_cannot_replace_a_whole_node_gnn_prediction():
    from shapely.geometry import box,mapping,shape
    from pipelines.semantic_temporal.runtime.reports.changes import _predicted_events
    from pipelines.semantic_temporal.runtime.preview import event_rows
    node=dict(id='new',year=2,class_id=1,confidence=.9,geometry=mapping(box(0,0,30,30)),area=900)
    edge=dict(event_id='thin',kind='conversion',source_class=4,target_class=1,
        status='candidate',geometry=mapping(box(0,0,2,30)),area=60,
        source_nodes=[],target_nodes=['new'],related_tasks=['building_gain'])
    score={('new','building_gain'):dict(probability=.95,prediction=1,status='inference')}
    doc={'nodes':[node],'temporal_edges':[]}
    predictions=_predicted_events(doc,{'events':[edge]},score)
    assert len(predictions)==1
    assert shape(predictions[0]['geometry']).equals(shape(node['geometry']))
    rows=event_rows([edge]+predictions,doc,score,external_predictions=True)
    assert rows[0]['status']=='candidate' and not rows[0]['major_change']['include']
    assert rows[1]['status']=='predicted' and rows[1]['major_change']['include']
    # A supported whole-footprint image event already represents the same task.
    whole={**edge,'status':'reliable','geometry':node['geometry'],'area':900}
    assert _predicted_events(doc,{'events':[whole]},score)==[]
