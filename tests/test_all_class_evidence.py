import numpy as np
import pytest
from affine import Affine
from pipelines.semantic_temporal.runtime.contracts import GridSpec
from pipelines.semantic_temporal.runtime.change_contracts import load_change_policy


def frame_for(before, after):
    years = (2013,2023); data = {}
    for year,classes in zip(years,(before,after)):
        probabilities = np.stack([classes==k for k in range(9)]).astype(np.float32)
        data[year] = {'classes':classes,'labels':np.ones(classes.shape,np.int32),
            'valid':classes!=255,'rgb_original':np.full((*classes.shape,3),100,np.uint8),
            'shape_probabilities':probabilities,'max_probabilities':probabilities,
            'node_ids':{1:str(year)}}
    return {'years':years,'grid':GridSpec('EPSG:32636',(1,0,0,0,-1,100),'metres',(1,1)),
            'shape_hw':before.shape,'data':data,'roof_offsets':(np.zeros(before.shape[1]),np.zeros(before.shape[1]))}


def small_policy():
    p=load_change_policy(None);p['minimum_area_m2']={k:0 for k in p['minimum_area_m2']};p['absence_window_m']=1
    return p


@pytest.mark.parametrize('source,target', [(a,b) for a in range(1,9) for b in range(1,9) if a!=b])
def test_known_transition_encoding(source,target):
    from pipelines.semantic_temporal.runtime.all_class_evidence import detect_evidence
    frame=frame_for(np.full((5,7),source,np.uint8),np.full((5,7),target,np.uint8))
    result=detect_evidence(frame,small_policy())
    assert result['transition'][2,3] == 16*source+target
    from pipelines.semantic_temporal.runtime.contracts import CLASSES
    assert result['masks'][CLASSES[target]+'_gain'].any()
    assert result['masks'][CLASSES[source]+'_loss'].any()


def test_missing_observations_are_masked():
    from pipelines.semantic_temporal.runtime.all_class_evidence import detect_evidence
    frame=frame_for(np.full((8,12),6,np.uint8),np.full((8,12),1,np.uint8))
    frame['data'][2013]['valid'][:,:3]=False
    frame['data'][2023]['classes'][:,3:5]=0
    frame['data'][2013]['rgb_original'][:,5:7]=0
    result=detect_evidence(frame,small_policy())
    for mask in result['masks'].values(): assert not mask[:,:7].any()
    assert (result['transition'][:,:3]==65535).all()
    assert (result['transition'][:,3:5]==0).all()


def test_vegetation_conversion_is_not_aggregate_loss():
    from pipelines.semantic_temporal.runtime.all_class_evidence import detect_evidence
    frame=frame_for(np.full((5,7),3,np.uint8),np.full((5,7),4,np.uint8))
    result=detect_evidence(frame,small_policy())
    assert result['masks']['tree_loss'].all()
    assert result['masks']['grass_shrub_gain'].all()
    assert not result['vegetation_gain'].any() and not result['vegetation_loss'].any()


def test_roof_compensation_does_not_move_roads():
    from pipelines.semantic_temporal.runtime.all_class_evidence import detect_evidence
    before=np.full((12,28),6,np.uint8);after=before.copy()
    before[2:10,3:6]=1;after[2:10,5:8]=1
    before[2:10,16:18]=2;after[2:10,18:20]=2
    frame=frame_for(before,after);frame['roof_offsets']=(np.zeros(28),np.full(28,2.))
    result=detect_evidence(frame,small_policy())
    assert not result['masks']['building_gain'].any()
    assert not result['masks']['building_loss'].any()
    assert result['masks']['road_gain'].any() and result['masks']['road_loss'].any()


def test_comparison_owns_overlapping_pixels_once(tmp_path):
    from pipelines.semantic_temporal.runtime.comparison import load_comparison
    from tests.test_semantic_graphs import make_artifact
    from pipelines.semantic_temporal.runtime.graphs import build_tile_graph,combine_year_graphs
    from pipelines.semantic_temporal.runtime.temporal import build_temporal_graph
    from pipelines.semantic_temporal.runtime.contracts import RunContext
    context=RunContext(tmp_path/'run',(),{}, {},'fixture')
    annual=[]
    for year in (2013,2023):
        artifacts=[make_artifact(tmp_path/'inputs',np.ones((20,30),np.int32),year=year,key=key) for key in ('a','b')]
        annual.append(combine_year_graphs([build_tile_graph(a,'new',context) for a in artifacts],context))
    graph=build_temporal_graph(tuple(annual),context)
    frame=load_comparison(graph,context)
    assert frame['data'][2013]['valid'].sum()==600
    assert frame['shape_hw']==(20,30)


def test_nine_pixel_high_rgb_is_not_treated_as_nine_probability_channels(tmp_path):
    from pipelines.semantic_temporal.runtime.comparison import load_comparison
    from tests.test_semantic_graphs import make_artifact
    from pipelines.semantic_temporal.runtime.graphs import build_tile_graph,combine_year_graphs
    from pipelines.semantic_temporal.runtime.temporal import build_temporal_graph
    from pipelines.semantic_temporal.runtime.contracts import RunContext
    context=RunContext(tmp_path/'run',(),{}, {},'fixture')
    annual=tuple(combine_year_graphs([build_tile_graph(make_artifact(tmp_path/'inputs',np.ones((9,12),np.int32),year=y),'new',context)],context) for y in (2013,2023))
    frame=load_comparison(build_temporal_graph(annual,context),context)
    assert (frame['data'][2013]['rgb_original']==100).all()


def test_rotated_rectangular_sampling_uses_pixel_centres():
    from pipelines.semantic_temporal.runtime.comparison import warp_array
    data=np.zeros((3,7),np.uint8);data[1,2]=9
    source=Affine(0,-1,5,1,0,0);target=Affine.identity()
    result=warp_array(data,source,target,(8,8),nearest=True)
    assert result[2,3]==9
    assert (result==9).sum()==1


def test_comparison_budget_fails_before_reading_probabilities(tmp_path):
    from dataclasses import replace
    from pipelines.semantic_temporal.runtime.comparison import load_comparison
    from tests.test_semantic_temporal import temporal_fixture
    graph,context=temporal_fixture(tmp_path)
    with pytest.raises(ValueError,match='max-comparison-pixels'):
        load_comparison(graph,replace(context,options={'max_comparison_pixels':100}))
