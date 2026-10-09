import numpy as np
from tests.test_semantic_graphs import make_artifact
from pipelines.semantic_temporal.runtime.contracts import RunContext
from pipelines.semantic_temporal.runtime.storage import read_json


def temporal_fixture(tmp_path,method='new'):
    from pipelines.semantic_temporal.runtime.graphs import build_tile_graph,combine_year_graphs
    from pipelines.semantic_temporal.runtime.temporal import build_temporal_graph
    context=RunContext(tmp_path/'run',(),{}, {},'fixture')
    labels=np.ones((20,30),np.int32);labels[:,15:]=2
    old_classes=np.full(labels.shape,3,np.uint8);old_classes[:,:15]=1
    new_classes=old_classes.copy();new_classes[:,15:]=1
    old=make_artifact(tmp_path/'inputs',labels,old_classes,year=2017)
    new=make_artifact(tmp_path/'inputs',labels,new_classes,year=2023)
    graphs=tuple(combine_year_graphs([build_tile_graph(a,method,context)],context) for a in (old,new))
    return build_temporal_graph(graphs,context),context


def test_actual_years_construction_and_pseudo_target_provenance(tmp_path):
    from pipelines.semantic_temporal.runtime.temporal import detect_classical
    artifact,context=temporal_fixture(tmp_path)
    result=detect_classical(artifact,context)
    assert result['years']==[2017,2023]
    assert any(n['event']=='new_building' and n['year']==2023 for n in result['node_changes'])
    assert any(n['event']=='persistent' for n in result['node_changes'])
    import pandas as pd
    targets=pd.read_csv(artifact.directory/'training_targets.csv')
    assert set(targets.loc[targets.task=='building_gain','label'].dropna())=={0,1}
    assert targets.source.str.contains('rule').all()


def test_missing_coverage_and_unknown_are_never_negative_changes(tmp_path):
    from pipelines.semantic_temporal.runtime.graphs import build_tile_graph,combine_year_graphs
    from pipelines.semantic_temporal.runtime.temporal import build_temporal_graph,detect_classical
    context=RunContext(tmp_path/'run',(),{}, {},'fixture')
    earlier=make_artifact(tmp_path/'inputs',np.ones((10,20),np.int32),np.ones((10,20),np.uint8),year=2005)
    later=make_artifact(tmp_path/'inputs',np.ones((10,10),np.int32),np.zeros((10,10),np.uint8),year=2021,x=500010)
    graphs=tuple(combine_year_graphs([build_tile_graph(a,'new',context)],context) for a in (earlier,later))
    artifact=build_temporal_graph(graphs,context);result=detect_classical(artifact,context)
    assert all(n['event'] in ('coverage_limited','unknown_observation') for n in result['node_changes'])
    assert all(n['label'] is None for n in result['node_changes'])


def test_old_graph_uses_existing_classical_event_engine(tmp_path):
    from pipelines.semantic_temporal.runtime.temporal import detect_classical
    artifact,context=temporal_fixture(tmp_path,'old');result=detect_classical(artifact,context)
    assert result['engine']=='all_class_image_evidence'
    assert result['legacy_engine']=='stage5_classical'
    legacy=read_json(result['legacy_report'])
    assert any(event['relation']=='constructed' for event in legacy['events'])
    nodes={n['id']:n for n in read_json(artifact.paths['graph'])['nodes']}
    for row in legacy['node_changes']:
        if row['label'] is not None: assert nodes[row['gid']]['class_id']==1
        if row['label']==1: assert row['year']==2023


def test_evidence_clips_invalid_border_outside_observed_mosaic(tmp_path):
    from pipelines.semantic_temporal.runtime.graphs import build_tile_graph,combine_year_graphs
    from pipelines.semantic_temporal.runtime.temporal import build_temporal_graph,detect_classical
    context=RunContext(tmp_path/'run',(),{}, {},'fixture')
    labels=np.zeros((20,30),np.int32);labels[1:-1,1:-1]=1
    classes=np.where(labels,1,255).astype(np.uint8)
    artifacts=[make_artifact(tmp_path/'inputs',labels,classes,year=y) for y in (2017,2023)]
    graphs=tuple(combine_year_graphs([build_tile_graph(a,'new',context)],context) for a in artifacts)
    report=detect_classical(build_temporal_graph(graphs,context),context)
    assert report['status']=='complete'


def test_tree_gain_raster_and_json_events_agree(tmp_path):
    from pipelines.semantic_temporal.runtime.graphs import build_tile_graph,combine_year_graphs
    from pipelines.semantic_temporal.runtime.temporal import build_temporal_graph,detect_classical
    context=RunContext(tmp_path/'run',(),{}, {},'fixture')
    labels=np.ones((20,30),np.int32)
    artifacts=[make_artifact(tmp_path/'inputs',labels,np.full(labels.shape,klass,np.uint8),year=y)
               for y,klass in ((2017,6),(2023,3))]
    graphs=tuple(combine_year_graphs([build_tile_graph(a,'new',context)],context) for a in artifacts)
    report=detect_classical(build_temporal_graph(graphs,context),context)
    assert any(e['target_class']==3 and 'tree_gain' in e['related_tasks'] for e in report['events'])
    assert next(r for r in report['node_changes'] if r['year']==2023)['event']=='tree_gain'
    assert all(r['event']!='persistent' for r in report['node_changes'])


def test_old_classical_uses_explicit_boundary_tolerance(tmp_path,monkeypatch):
    from dataclasses import replace
    from pipelines.semantic_temporal.runtime.temporal import detect_classical
    import src.stage5.events as events
    artifact,context=temporal_fixture(tmp_path,'old');original=events.detect_change_events;seen=[]
    def capture(*args):
        seen.append(args[-1].boundary_tolerance_m);return original(*args)
    monkeypatch.setattr(events,'detect_change_events',capture)
    detect_classical(artifact,replace(context,options={'boundary_tolerance':7.5}))
    assert seen==[7.5]


def test_evidence_field_names_are_year_independent(tmp_path):
    from pipelines.semantic_temporal.runtime.temporal import detect_classical
    artifact,context=temporal_fixture(tmp_path)
    report=detect_classical(artifact,context)
    assert all(not any('2013' in key or '2023' in key for key in row.get('evidence',{})) for row in report['node_changes'])


def test_connected_building_extension_keeps_construction_target(tmp_path):
    from pipelines.semantic_temporal.runtime.graphs import build_tile_graph,combine_year_graphs
    from pipelines.semantic_temporal.runtime.temporal import build_temporal_graph,detect_classical
    context=RunContext(tmp_path/'run',(),{}, {},'fixture')
    before=np.ones((20,30),np.int32);before[:,15:]=2
    earlier_classes=np.where(before==1,1,6).astype(np.uint8)
    later=np.ones_like(before)
    artifacts=[make_artifact(tmp_path/'inputs',before,earlier_classes,year=2017),
               make_artifact(tmp_path/'inputs',later,np.ones_like(later,dtype=np.uint8),year=2023)]
    graphs=tuple(combine_year_graphs([build_tile_graph(a,'new',context)],context) for a in artifacts)
    report=detect_classical(build_temporal_graph(graphs,context),context)
    assert any(r['year']==2023 and r['event']=='extension' and r['label']==1 for r in report['node_changes'])


def test_new_building_correspondence_compensates_current_roof_displacement(tmp_path):
    from pipelines.semantic_temporal.runtime.graphs import build_tile_graph,combine_year_graphs
    from pipelines.semantic_temporal.runtime.temporal import build_temporal_graph
    context=RunContext(tmp_path/'run',(),{}, {},'fixture')
    labels=np.ones((20,20),np.int32);labels[5:15,5:7]=2
    classes=np.where(labels==2,1,3).astype(np.uint8)
    shifted=np.ones_like(labels);shifted[5:15,7:9]=2
    later_classes=np.where(shifted==2,1,3).astype(np.uint8)
    artifacts=[make_artifact(tmp_path/'inputs',lab,cls,year=year) for lab,cls,year in [(labels,classes,2017),(shifted,later_classes,2023)]]
    annual=tuple(combine_year_graphs([build_tile_graph(a,'new',context)],context) for a in artifacts)
    temporal=read_json(build_temporal_graph(annual,context).paths['graph'])
    lookup={n['id']:n for n in temporal['nodes']}
    links=[e for e in temporal['temporal_edges'] if lookup[e['source']]['class_id']==lookup[e['target']]['class_id']==1]
    assert len(links)==1 and links[0]['frame']=='roof'
    assert links[0]['roof_shift']==[2.,0.]
    assert links[0]['iou']==1
