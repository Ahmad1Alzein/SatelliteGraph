from pathlib import Path
import csv
import numpy as np
from shapely.geometry import box,mapping
from pipelines.semantic_temporal.runtime.contracts import RunContext,TemporalArtifact,TileInput,YearInput
from pipelines.semantic_temporal.runtime.storage import atomic_json,read_json
from tests.test_all_class_evidence import frame_for


def matching_fixture(tmp_path,count=1):
    frame=frame_for(np.ones((20,40),np.uint8),np.ones((20,40),np.uint8))
    context=RunContext(tmp_path,(),{}, {},'fixture')
    nodes=[{'id':f'{year}-{i}','year':year,'class_id':1,'position':[2+i%35,90],
            'geometry':mapping(box(1+i%35,89,3+i%35,91))} for year in frame['years'] for i in range(count)]
    edges=[{'source':f'2013-{i}','target':f'2023-{i}','edge_role':'primary','confidence':'high','iou':.8} for i in range(count)]
    document={'nodes':nodes,'temporal_edges':edges,'primary_components':[
        {'id':f'match-{i}','kind':['one_to_one','split','merge','many_to_many','unmatched'][i%5],
         'source_nodes':[f'2013-{i}'],'target_nodes':[f'2023-{i}']} for i in range(count)]}
    path=tmp_path/'temporal/graph.json';atomic_json(path,document)
    return TemporalArtifact(frame['years'],path.parent,{'graph':path}),context,frame


def test_control_endpoint_projection(tmp_path):
    from pipelines.semantic_temporal.runtime.reports.matching import render_matching_results
    graph,context,frame=matching_fixture(tmp_path)
    atomic_json(tmp_path/'prepared/registration_controls.json',{'method':'supplied_controls','pairs':[
        {'id':'c1','moving_xy':[10,90],'reference_xy':[12,90],'inlier':True,'residual':2}]})
    render_matching_results(graph,context,frame,{'derived_edges':[]})
    record=read_json(tmp_path/'results/matching/index.json')['registration']
    from pipelines.semantic_temporal.runtime.reports.common import world_pixel
    expected=world_pixel(10,90,record['bounds'],record['panel_size'])
    np.testing.assert_allclose(record['endpoints'][0]['source'],expected,atol=1)
    assert record['endpoints'][0]['inlier']


def test_georeference_only_and_conversion_are_explicit(tmp_path):
    from pipelines.semantic_temporal.runtime.reports.matching import render_matching_results
    graph,context,frame=matching_fixture(tmp_path,5)
    atomic_json(tmp_path/'prepared/registration_controls.json',{'method':'georeference_only','pairs':[]})
    result=render_matching_results(graph,context,frame,{'derived_edges':[{'source':'2013-0','target':'2023-1','event_id':'event-1'}]})
    index=read_json(tmp_path/'results/matching/index.json')
    assert 'not performed' in index['registration']['status']
    assert set(index['component_types'])=={'one_to_one','split','merge','many_to_many','unmatched'}
    assert index['conversions']['edge_kind']=='Class change'
    assert index['temporal']['total_count']==5
    assert all(p.is_file() for p in result.values())


def test_link_display_cap_preserves_full_table(tmp_path):
    from pipelines.semantic_temporal.runtime.reports.matching import render_matching_results
    graph,context,frame=matching_fixture(tmp_path,317)
    result=render_matching_results(graph,context,frame,{'derived_edges':[]})
    record=read_json(tmp_path/'results/matching/index.json')['temporal']
    with result['match_table'].open(newline='') as stream:table=list(csv.DictReader(stream))
    assert record['displayed_count']<=300
    assert len(table)==record['total_count']==317
    assert record['omitted_count']==record['total_count']-record['displayed_count']


def test_unequal_filenames_pair_by_coverage(tmp_path):
    from pipelines.semantic_temporal.runtime.reports.matching import render_matching_results
    graph,context,frame=matching_fixture(tmp_path)
    a=TileInput('2013:left.png',2013,tmp_path/'left.png',None,'a',(0,0),(20,40),frame['grid'],(20,40))
    b=TileInput('2023:different.png',2023,tmp_path/'different.png',None,'b',(0,0),(20,40),frame['grid'],(20,40))
    context=RunContext(tmp_path,(YearInput(2013,(a,)),YearInput(2023,(b,))),{}, {},'fixture')
    render_matching_results(graph,context,frame,{'derived_edges':[]})
    pairs=read_json(tmp_path/'results/matching/index.json')['coverage_pairs']
    assert len(pairs)==1 and pairs[0]['source']==a.key and pairs[0]['target']==b.key
    assert (tmp_path/pairs[0]['path']).is_file()
