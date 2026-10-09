import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from affine import Affine
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from pipelines.semantic_temporal.runtime.contracts import TemporalArtifact,CLASSES
from pipelines.semantic_temporal.runtime.storage import atomic_json
from pipelines.semantic_temporal.runtime.change_contracts import load_change_policy
from tests.test_all_class_evidence import frame_for


def catalog_fixture(tmp_path,before,after,links=None):
    frame=frame_for(before,after);nodes=[]
    for year in frame['years']:
        d=frame['data'][year];labels=np.zeros(before.shape,np.int32);lookup={}
        for klass in np.unique(d['classes']):
            if klass==255:continue
            mask=d['classes']==klass
            ident=len(lookup)+1;labels[mask]=ident;gid=f'{year}:{int(klass)}';lookup[ident]=gid
            geom=unary_union([shape(g) for g,v in shapes(mask.astype(np.uint8),mask=mask,transform=Affine(*frame['grid'].affine)) if v])
            nodes.append({'id':gid,'year':year,'class_id':int(klass),'confidence':1.,'geometry':mapping(geom),'area':geom.area,
                          'observed_fraction':1.,'known_counterpart_fraction':1.})
        d['labels']=labels;d['node_ids']=lookup
    edges=[] if links is None else links
    directory=tmp_path/'temporal';directory.mkdir(exist_ok=True)
    path=directory/'graph.json';atomic_json(path,{'years':list(frame['years']),'units':'metres','nodes':nodes,
        'temporal_edges':edges,'primary_components':[]})
    graph=TemporalArtifact(frame['years'],directory,{'graph':path})
    return graph,frame


def catalog(tmp_path,before,after,links=None,reviewed_targets=None):
    from pipelines.semantic_temporal.runtime.all_class_evidence import detect_evidence
    from pipelines.semantic_temporal.runtime.change_catalog import build_change_catalog
    graph,frame=catalog_fixture(tmp_path,before,after,links)
    policy=load_change_policy(None)
    return build_change_catalog(graph,frame,detect_evidence(frame,policy),policy,reviewed_targets=reviewed_targets)


@pytest.mark.parametrize('source,target', [(a,b) for a in range(1,9) for b in range(1,9) if a!=b])
def test_each_conversion_is_one_event_with_two_tasks(tmp_path,source,target):
    result=catalog(tmp_path,np.full((24,24),source,np.uint8),np.full((24,24),target,np.uint8))
    assert len(result['events'])==1
    event=result['events'][0]
    assert event['kind']=='conversion'
    assert event['source_class']==source and event['target_class']==target
    assert set(event['related_tasks'])=={CLASSES[source]+'_loss',CLASSES[target]+'_gain'}
    assert event['area']==576
    assert result['transition_matrix']['area'][source-1][target-1]==576


def test_partial_conversion_is_one_event_without_primary_link(tmp_path):
    before=np.full((20,20),3,np.uint8);after=before.copy();after[:,0:4]=1
    result=catalog(tmp_path,before,after)
    assert len(result['events'])==1
    assert len(result['events'][0]['related_tasks'])==2
    assert result['events'][0]['area']==80
    assert result['derived_edges'][0]['edge_type']=='change_conversion'


def test_structural_difference_is_not_a_positive(tmp_path):
    before=np.full((20,20),2,np.uint8)
    link={'source':'2013:2','target':'2023:2','edge_role':'primary','relation':'identity'}
    graph,frame=catalog_fixture(tmp_path,before,before, [link])
    doc=json.loads(graph.paths['graph'].read_text());doc['primary_components']=[{'id':'m1','kind':'split','source_nodes':['2013:2'],'target_nodes':['2023:2']}]
    atomic_json(graph.paths['graph'],doc)
    from pipelines.semantic_temporal.runtime.all_class_evidence import detect_evidence
    from pipelines.semantic_temporal.runtime.change_catalog import build_change_catalog
    p=load_change_policy(None);result=build_change_catalog(graph,frame,detect_evidence(frame,p),p)
    assert not any(row['label']==1 for row in result['targets'])
    assert result['events']==[]
    assert result['structural_observations'][0]['kind']=='split'
    assert any(row['label']==0 for row in result['targets'])


def test_road_expansion_and_contraction_are_shape_descriptors(tmp_path):
    before=np.full((30,30),6,np.uint8);before[:,10:20]=2
    after=np.full_like(before,6);after[:,10:25]=2
    links=[{'source':'2013:2','target':'2023:2','edge_role':'primary','relation':'identity'}]
    result=catalog(tmp_path,before,after,links)
    assert any('expansion' in e['descriptors'].get('road_gain',[]) for e in result['events'])
    reverse=catalog(tmp_path,after,before,links)
    assert any('contraction' in e['descriptors'].get('road_loss',[]) for e in reverse['events'])


def test_reconfiguration_keeps_both_footprints_in_one_parent_group(tmp_path):
    before=np.full((30,30),6,np.uint8);before[:,5:15]=2
    after=np.full_like(before,6);after[:,10:20]=2
    links=[{'source':'2013:2','target':'2023:2','edge_role':'primary','relation':'identity'}]
    result=catalog(tmp_path,before,after,links)
    parents=[p for p in result['compound_events'] if p['kind']=='reconfiguration']
    assert len(parents)==1
    assert len(parents[0]['event_ids'])==2


def test_seasonal_conversion_visible_but_masked_from_training(tmp_path):
    result=catalog(tmp_path,np.full((24,24),4,np.uint8),np.full((24,24),6,np.uint8))
    assert result['events'][0]['status']=='season_sensitive'
    assert all(r['label'] is None for r in result['targets'])


def test_reviewed_labels_can_resolve_seasonal_candidate(tmp_path):
    path=tmp_path/'reviewed.csv'
    path.write_text('gid,task,label,confidence,source\n2023:6,bare_soil_gain,1,1,manual review\n')
    result=catalog(tmp_path,np.full((24,24),4,np.uint8),np.full((24,24),6,np.uint8),reviewed_targets=path)
    row=next(r for r in result['targets'] if r['gid']=='2023:6' and r['task']=='bare_soil_gain')
    assert row['label']==1 and row['source']=='manual review'
    assert result['events'][0]['status']=='reliable'


@pytest.mark.parametrize('rows',[
    '2023:1,building_gain,1,1,review\n2023:1,building_gain,0,1,review\n',
    'missing,building_gain,1,1,review\n', '2023:1,unknown_gain,1,1,review\n',
    '2023:1,building_gain,1,nan,review\n','2023:1,building_gain,2,1,review\n',
    '2023:1,building_gain,1,1, \n', '2013:6,building_gain,1,1,review\n',
])
def test_invalid_reviewed_rows_are_rejected(tmp_path,rows):
    path=tmp_path/'reviewed.csv';path.write_text('gid,task,label,confidence,source\n'+rows)
    with pytest.raises(ValueError):
        catalog(tmp_path,np.full((24,24),6,np.uint8),np.full((24,24),1,np.uint8),reviewed_targets=path)


def test_unknown_coverage_cannot_be_overridden(tmp_path):
    path=tmp_path/'reviewed.csv';path.write_text('gid,task,label,confidence,source\n2023:1,building_gain,1,1,review\n')
    with pytest.raises(ValueError,match='coverage|unknown'):
        catalog(tmp_path,np.zeros((24,24),np.uint8),np.ones((24,24),np.uint8),reviewed_targets=path)


def test_absence_of_match_is_not_a_persistence_negative(tmp_path):
    image=np.ones((24,24),np.uint8)
    result=catalog(tmp_path,image,image)
    assert not any(r['label']==0 for r in result['targets'])


def test_connectivity_is_computed_from_class_pixels(tmp_path):
    before=np.full((30,30),6,np.uint8);before[:,5:10]=2;before[:,20:25]=2
    after=before.copy();after[5:25,5:25]=2
    result=catalog(tmp_path,before,after)
    observation=next(r for r in result['structural_observations'] if r['kind']=='connectivity' and r['class_id']==2)
    assert observation['earlier_components']==2 and observation['later_components']==1
    assert observation['footprint_evidence']


def test_building_minimum_override_cannot_create_target_without_qualifying_mask(tmp_path):
    from pipelines.semantic_temporal.runtime.all_class_evidence import detect_evidence
    from pipelines.semantic_temporal.runtime.change_catalog import build_change_catalog
    graph,frame=catalog_fixture(tmp_path,np.full((24,24),6,np.uint8),np.ones((24,24),np.uint8))
    p=load_change_policy(None);p['minimum_area_m2']['building']=1000
    result=build_change_catalog(graph,frame,detect_evidence(frame,p),p)
    assert not any(r['label']==1 for r in result['targets'])
