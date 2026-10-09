from pathlib import Path
import numpy as np
from pipelines.semantic_temporal.runtime.contracts import GridSpec,SegmentationArtifact,RunContext
from pipelines.semantic_temporal.runtime.storage import write_raster,atomic_json,read_json


def make_artifact(root,labels,classes=None,year=2017,key='a',x=500000,units='metres'):
    directory=Path(root)/str(year)/key;directory.mkdir(parents=True,exist_ok=True)
    grid=GridSpec('EPSG:32636' if units=='metres' else None,(1,0,x,0,-1,3700000),units,(1,1) if units=='metres' else None)
    classes=np.where(labels>0,3,255).astype(np.uint8) if classes is None else classes
    paths={k:directory/name for k,name in {'labels':'labels.tif','classes':'classes.tif','valid':'valid.tif','image':'rgb.tif','probabilities':'probabilities.npy','validation':'validation.json','lineage':'lineage.json'}.items()}
    for name,data in [('labels',labels.astype(np.int32)),('classes',classes),('valid',(labels>0).astype(np.uint8)),('image',np.full((3,*labels.shape),100,np.uint8))]: write_raster(paths[name],data,grid)
    probabilities=np.zeros((9,*labels.shape),np.float32)
    for ident in range(9): probabilities[ident]=classes==ident
    np.save(paths['probabilities'],probabilities);atomic_json(paths['validation'],{'probability_evidence':'full'} )
    lineage={str(i):[{'year':year,'tile':key,'segment':int(i),'method':'fixture'}] for i in np.unique(labels[labels>0])}
    atomic_json(paths['lineage'],lineage)
    artifact=SegmentationArtifact(year,key,directory,grid,labels.shape,paths,lineage);atomic_json(directory/'artifact.json',artifact)
    return artifact


def test_distinct_graph_methods_corner_contact_and_finite_empty_edges(tmp_path):
    from pipelines.semantic_temporal.runtime.graphs import build_tile_graph
    labels=np.array([[1,0,0],[0,2,0],[0,0,0]])
    artifact=make_artifact(tmp_path/'inputs',labels,np.where(labels>0,1,255).astype(np.uint8))
    context=RunContext(tmp_path/'run',(),{}, {},'fixture')
    old=read_json(build_tile_graph(artifact,'old',context).paths['graph'])
    new=build_tile_graph(artifact,'new',context);document=read_json(new.paths['graph'])
    assert old['edges']==[]
    assert len(document['edges'])==1 and document['edges'][0]['edge_type']=='near'
    assert document['edges'][0]['corner_contact'] is True
    with np.load(new.paths['tensors'],allow_pickle=False) as data: assert np.isfinite(data['x']).all()


def test_overlap_is_owned_once_with_members_and_schema_validation(tmp_path):
    from pipelines.semantic_temporal.runtime.graphs import build_tile_graph,combine_year_graphs
    from src.stage4.schema import validate_combined_graph
    context=RunContext(tmp_path/'run',(),{}, {},'fixture')
    first=make_artifact(tmp_path/'inputs',np.ones((4,5),np.int32),key='one')
    second=make_artifact(tmp_path/'inputs',np.ones((4,5),np.int32),key='two')
    graphs=[build_tile_graph(a,'new',context) for a in (first,second)]
    combined=combine_year_graphs(graphs,context);doc=read_json(combined.paths['graph'])
    assert len(doc['nodes'])==1 and doc['nodes'][0]['area']==20
    assert len(doc['nodes'][0]['members'])==2
    validate_combined_graph(read_json(combined.paths['old_contract']))


def test_annual_rgb_io_does_not_grow_with_node_count(tmp_path,monkeypatch):
    import rasterio
    from pipelines.semantic_temporal.runtime.graphs import build_tile_graph,combine_year_graphs
    labels=np.ones((8,12),np.int32);labels[:,6:]=2
    artifact=make_artifact(tmp_path/'inputs',labels)
    context=RunContext(tmp_path/'run',(),{}, {},'fixture')
    graph=build_tile_graph(artifact,'new',context);original=rasterio.open;reads=[]
    def opened(path,*args,**kwargs):
        if Path(path)==artifact.paths['image']:reads.append(path)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(rasterio,'open',opened)
    annual=combine_year_graphs([graph],context)
    assert len(reads)<=1
    assert all(node['rgb_mean']==[100.,100.,100.] for node in read_json(annual.paths['graph'])['nodes'])


def test_annual_overlap_uses_saved_custom_config(tmp_path,monkeypatch):
    from pipelines.semantic_temporal.runtime.graphs import build_tile_graph,combine_year_graphs
    from src.config import load_config
    import src.stage4.pipeline as stage4
    config=load_config();config['stage4']['overlap']['minimum_intersection_over_union']=.91
    context=RunContext(tmp_path/'run',(),{'old_config':config},{},'fixture')
    artifact=make_artifact(tmp_path/'inputs',np.ones((5,7),np.int32))
    graph=build_tile_graph(artifact,'old',context);original=stage4._overlap_candidates;seen=[]
    def capture(tiles,policy):
        seen.append(policy['minimum_intersection_over_union']);return original(tiles,policy)
    monkeypatch.setattr(stage4,'_overlap_candidates',capture)
    combine_year_graphs([graph],context)
    assert seen==[.91]


def test_original_pixels_unknown_one_node_and_no_fake_metres(tmp_path):
    from pipelines.semantic_temporal.runtime.graphs import build_tile_graph
    labels=np.ones((2,3),np.int32);artifact=make_artifact(tmp_path/'inputs',labels,np.zeros_like(labels,np.uint8),units='pixels',x=4)
    context=RunContext(tmp_path/'run',(),{}, {},'fixture')
    result=build_tile_graph(artifact,'new',context);doc=read_json(result.paths['graph'])
    assert doc['units']=='pixels' and doc['nodes'][0]['class_id']==0
    assert not any('m2' in name for name in doc['nodes'][0])
    with np.load(result.paths['tensors'],allow_pickle=False) as data:
        assert data['edge_index'].shape==(2,0) and np.isfinite(data['x']).all()
