"""Distinct graph methods with shared, exact annual pixel ownership."""
from pathlib import Path
from types import SimpleNamespace
import math
import numpy as np
import pandas as pd
import cv2
import rasterio
from affine import Affine
from shapely import STRtree
from shapely.geometry import shape,mapping
from shapely.ops import transform as geometry_transform
from .contracts import CLASSES,SCHEMA,GraphArtifact
from .storage import atomic_json,read_json,save_manifest
from .adapters import read_band,CANONICAL_TO_OLD
from .segmentation.parcels import shape_metrics


def transform_geometry(geom,affine):
    return geometry_transform(lambda x,y,z=None:(affine.a*np.asarray(x)+affine.b*np.asarray(y)+affine.c,
                                                affine.d*np.asarray(x)+affine.e*np.asarray(y)+affine.f),geom)


def _old_tile(labels,classes,probabilities,key,grid,offset=(0,0),source_grid=None):
    from src.stage3.geometry import build_node_geometries,spatial_context
    from src.stage3.adjacency import build_adjacency_edges
    from src.stage3.pipeline import _build_nodes,_build_edges,_annotate_graph,_annotate_analysis_view
    from src.stage3.schema import SCHEMA_VERSION,ADJACENCY_DEFINITION,validate_graph_document
    metadata={'tile_id':key,'source_image_id':'runtime','global_x0':offset[0],'global_y0':offset[1],
              'tile_affine_transform':list(grid.affine),'source_affine_transform':list((source_grid or grid).affine),'crs':grid.crs}
    records=[]
    for ident in range(1,int(labels.max())+1):
        selected=labels==ident
        if not selected.any(): raise ValueError('Graph requires contiguous segmentation IDs')
        klass=int(classes[selected][0]);confidence=float(probabilities[klass,selected].mean())
        records.append({'instance_id':ident,'segment_id':f'segment_{ident}','area_pixels':int(selected.sum()),
                        'label':{'id':CANONICAL_TO_OLD[klass],'name':CLASSES[klass]},'mean_semantic_confidence':confidence})
    geometries=build_node_geometries(labels,metadata)
    result=SimpleNamespace(tile_id=key,segments=records,tile_metadata=metadata)
    nodes=_build_nodes(result,geometries)
    edges=_build_edges(key,nodes,build_adjacency_edges(labels,node_geometries=geometries,tile_metadata=metadata),geometries)
    _annotate_graph(nodes,edges);_annotate_analysis_view(nodes,edges,{})
    spatial=spatial_context(metadata)
    document={'schema_version':SCHEMA_VERSION,'graph':{'tile_id':key,'source_image_id':'runtime','tile_metadata':metadata,
              'projected_measurements':{'unit':spatial.linear_unit if spatial.projected_available else None,'crs':spatial.source_crs_text if spatial.projected_available else None},
              'directed':False,'multigraph':False,'adjacency_definition':ADJACENCY_DEFINITION,'minimum_shared_boundary_pixels':1,
              'node_count':len(nodes),'edge_count':len(edges)},'nodes':nodes,'edges':edges}
    validate_graph_document(document)
    return document,geometries


def _row(ident,klass,geom,rgb,probabilities,members,year,evidence):
    hsv=cv2.cvtColor(rgb.reshape(-1,1,3),cv2.COLOR_RGB2HSV).reshape(-1,3) if len(rgb) else np.zeros((1,3))
    metrics=shape_metrics(geom);point=geom.representative_point()
    row={'id':ident,'year':year,'class_id':klass,'class':CLASSES[klass],'area':float(geom.area),
         'perimeter':float(geom.length),'geometry':mapping(geom),'position':[point.x,point.y],
         'centroid':[geom.centroid.x,geom.centroid.y],'members':members,
         'confidence':float(probabilities[klass]),'full_probability_evidence':bool(evidence),
         'image_evidence':len(rgb)>0,'probabilities':probabilities.tolist(),
         **{k:v for k,v in metrics.items() if k!='perimeter_m'}}
    row['rgb_mean']=rgb.mean(0).tolist() if len(rgb) else [0.,0.,0.]
    row['rgb_std']=rgb.std(0).tolist() if len(rgb) else [0.,0.,0.]
    row['hsv_mean']=hsv.mean(0).tolist()
    return row


def _edges(nodes,touches,method,units,near_distance=25.,near_k=6):
    by_id={n['id']:n for n in nodes};geoms={n['id']:shape(n['geometry']) for n in nodes}
    pairs={(min(a,b),max(a,b)):('touches',length,0.) for a,b,length in touches}
    if method=='new' and units=='metres':
        ids=[n['id'] for n in nodes if n['class_id']==1];shapes=[geoms[i] for i in ids];tree=STRtree(shapes)
        for i,ident in enumerate(ids):
            candidates=[]
            for raw in tree.query(shapes[i].buffer(near_distance)):
                other=ids[int(raw)];key=tuple(sorted((ident,other)))
                distance=shapes[i].distance(shapes[int(raw)])
                if ident!=other and key not in pairs and distance<=near_distance: candidates.append((distance,other))
            for distance,other in sorted(candidates)[:near_k]: pairs[tuple(sorted((ident,other)))]=('near',0.,distance)
    edges=[]
    for (a,b),(kind,shared,gap) in sorted(pairs.items()):
        source,target=by_id[a],by_id[b];delta=np.array(target['position'])-source['position'];angle=math.atan2(delta[1],delta[0])
        edges.append({'id':f'{a}|{b}','source':a,'target':b,'edge_type':kind,'shared_boundary':float(shared),'gap':float(gap),
                      'distance':float(np.linalg.norm(delta)),'boundary_frac_source':shared/max(source['perimeter'],1e-9),
                      'boundary_frac_target':shared/max(target['perimeter'],1e-9),
                      'color_contrast':float(np.linalg.norm(np.array(source['rgb_mean'])-target['rgb_mean'])/255),
                      'same_class':source['class_id']==target['class_id'],'dir_sin':math.sin(angle),'dir_cos':math.cos(angle),
                      'corner_contact':kind=='near' and gap<1e-9})
    return edges


NODE_FEATURES=[f'class_{CLASSES[i]}' for i in range(9)]+['log_area','log_perimeter','compactness','solidity','log_elongation',
    'orient_sin2','orient_cos2']+[f'{stat}_{channel}' for stat in ('mean','std') for channel in ('r','g','b')]+[
    'mean_s','mean_v','hue_sin','hue_cos']+[f'p_{CLASSES[i]}' for i in range(9)]+[
    'confidence','full_probability_evidence','image_evidence','pos_x','pos_y']
EDGE_FEATURES=['shared_boundary','gap','distance','boundary_frac_source','boundary_frac_target','color_contrast','same_class','dir_sin','dir_cos']


def _persist(directory,nodes,edges,grid,year,method,**extra):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    paths={k:directory/name for k,name in {'graph':'graph.json','nodes':'nodes.csv','edges':'edges.csv','features':'feature_spec.json','tensors':'graph_tensors.npz'}.items()}
    positions=np.array([n['position'] for n in nodes],np.float64).reshape(-1,2)
    lo=positions.min(0) if len(nodes) else np.zeros(2);extent=np.ptp(positions,axis=0) if len(nodes) else np.ones(2)
    features=[]
    for node in nodes:
        angle=math.radians(node['orientation_deg'])*2;hue=math.radians(node['hsv_mean'][0]*2)
        features.append([float(node['class_id']==i) for i in range(9)]+[
            math.log(max(node['area'],1e-9)),math.log(max(node['perimeter'],1e-9)),node['compactness'],node['solidity'],
            math.log(max(1,node['elongation'])),math.sin(angle),math.cos(angle)]+[
            v/255 for v in node['rgb_mean']+node['rgb_std']]+[node['hsv_mean'][1]/255,node['hsv_mean'][2]/255,math.sin(hue),math.cos(hue)]+
            node['probabilities']+[node['confidence'],float(node['full_probability_evidence']),float(node['image_evidence'])]+
            ((np.array(node['position'])-lo)/np.maximum(extent,1e-9)).tolist())
    x=np.array(features,np.float32).reshape(len(nodes),len(NODE_FEATURES));index={n['id']:i for i,n in enumerate(nodes)}
    source=np.array([index[e['source']] for e in edges],np.int64);target=np.array([index[e['target']] for e in edges],np.int64)
    attributes=np.array([[float(e[name]) for name in EDGE_FEATURES] for e in edges],np.float32).reshape(len(edges),len(EDGE_FEATURES))
    reverse=attributes.copy();reverse[:,[3,4]]=attributes[:,[4,3]];reverse[:,-2:]*=-1
    edge_index=np.stack((np.r_[source,target],np.r_[target,source]))
    np.savez_compressed(paths['tensors'],x=x,edge_index=edge_index,edge_attr=np.concatenate((attributes,reverse)),
                        edge_type=np.array([0 if e['edge_type']=='touches' else 1 for e in edges]*2,np.int64),pos=positions,
                        node_ids=np.array([n['id'] for n in nodes],dtype=str))
    if not np.isfinite(x).all() or not np.isfinite(attributes).all(): raise ValueError('Nonfinite graph features')
    document={'schema':SCHEMA,'year':year,'method':method,'units':grid.units,'grid':grid,'nodes':nodes,'edges':edges,**extra}
    atomic_json(paths['graph'],document);atomic_json(paths['features'],{'node_features':NODE_FEATURES,'edge_features':EDGE_FEATURES,'normalized':False,'units':grid.units})
    pd.DataFrame(nodes).to_csv(paths['nodes'],index=False);pd.DataFrame(edges,columns=list(edges[0]) if edges else ['id','source','target','edge_type']).to_csv(paths['edges'],index=False)
    artifact=GraphArtifact(year,method,directory,grid,paths);atomic_json(directory/'artifact.json',artifact);save_manifest(directory,paths,year=year,method=method)
    return artifact


def build_tile_graph(artifact,method,context):
    if method not in ('old','new'): raise ValueError('Graph method must be old or new')
    if method=='new':
        from .adapters import connected_region_view
        artifact=connected_region_view(artifact)
    labels=read_band(artifact.paths['labels']);classes=read_band(artifact.paths['classes']);probabilities=np.load(artifact.paths['probabilities'],allow_pickle=False)
    image=None
    if 'image' in artifact.paths:
        with rasterio.open(artifact.paths['image']) as src: image=src.read().transpose(1,2,0)
    key=f'{artifact.year}:{artifact.tile_key}'
    old,geometries=_old_tile(labels,classes,probabilities,key,artifact.grid)
    nodes=[];lookup={}
    validation=read_json(artifact.paths['validation']) if 'validation' in artifact.paths else {}
    evidence=validation.get('probability_evidence')!='assigned_label_confidence_only'
    for record in old['nodes']:
        ident=record['instance_id'];mask=labels==ident;klass=int(classes[mask][0]);node_id=record['node_id'];lookup[ident]=node_id
        geom=transform_geometry(shape(record['geometry']),Affine(*artifact.grid.affine))
        rgb=image[mask] if image is not None else np.zeros((0,3),np.uint8)
        nodes.append(_row(node_id,klass,geom,rgb,probabilities[:,mask].mean(1),artifact.lineage.get(str(ident),[]),artifact.year,evidence))
    touches=[(lookup[e['source_instance_id']],lookup[e['target_instance_id']],e['shared_boundary_length_projected'] if artifact.grid.crs else e['shared_boundary_length_pixels']*math.sqrt(abs(Affine(*artifact.grid.affine).determinant))) for e in old['edges']]
    edges=_edges(nodes,touches,method,artifact.grid.units)
    graph=_persist(context.output/'graphs'/method/str(artifact.year)/artifact.tile_key,nodes,edges,artifact.grid,artifact.year,method,segmentation_artifact=str(artifact.directory/'artifact.json'))
    paths=dict(graph.paths);paths['old_contract']=graph.directory/'old_graph.json';atomic_json(paths['old_contract'],old)
    graph=GraphArtifact(graph.year,graph.method,graph.directory,graph.grid,paths);atomic_json(graph.directory/'artifact.json',graph);save_manifest(graph.directory,paths)
    return graph


def combine_year_graphs(graphs,context):
    if not graphs or len({(g.year,g.method) for g in graphs})!=1: raise ValueError('Combine one year and graph method at a time')
    from src.stage4.pipeline import _TileInput,_overlap_candidates,_resolve_matches,_assign_ownership,_build_combined_nodes,_build_combined_edges,_graph_metadata
    from src.stage4.schema import SCHEMA_VERSION,validate_combined_graph
    from src.config import load_config,select_stage4_config
    from .storage import load_segmentation
    graphs=sorted(graphs,key=lambda g:str(g.directory));base=graphs[0].grid;reference=Affine(*base.affine);tiles=[];inputs={}
    for graph in graphs:
        if graph.grid.crs!=base.crs or graph.grid.units!=base.units: raise ValueError('Annual graphs must share CRS and units')
        document=read_json(graph.paths['graph']);artifact=load_segmentation(document['segmentation_artifact'])
        affine=Affine(*artifact.grid.affine);relative=(~reference)*affine
        if not np.allclose([relative.a,relative.b,relative.d,relative.e],[1,0,0,1],atol=1e-6) or not np.allclose([relative.c,relative.f],np.round([relative.c,relative.f]),atol=1e-6): raise ValueError('Annual tiles must lie on the shared prepared grid')
        x,y=round(relative.c),round(relative.f);labels=read_band(artifact.paths['labels']);classes=read_band(artifact.paths['classes']);prob=np.load(artifact.paths['probabilities'],allow_pickle=False)
        key=f'{artifact.year}:{artifact.tile_key}';old,_=_old_tile(labels,classes,prob,key,artifact.grid,(x,y),base)
        old['graph']['source_image_id']=str(graph.year)
        confidence=np.zeros(labels.shape,np.float32)
        for klass in range(9): confidence[classes==klass]=prob[klass,classes==klass]
        result=SimpleNamespace(graph=old['graph'],edges=old['edges'])
        tiles.append(_TileInput(key,result,graph.directory,labels,confidence,{n['instance_id']:n for n in old['nodes']},x,y,labels.shape[1],labels.shape[0],np.ones(labels.shape,bool)))
        inputs[key]=(artifact,classes,prob)
    from .context import old_configuration
    cfg=select_stage4_config(old_configuration(context));matches=_overlap_candidates(tiles,cfg['overlap']);union,accepted=_resolve_matches(tiles,matches);_assign_ownership(tiles)
    old_nodes,geoms,roots,dropped=_build_combined_nodes(tiles,union,str(graphs[0].year))
    old_edges=_build_combined_edges(old_nodes,geoms,tiles,union,roots)
    old={'schema_version':SCHEMA_VERSION,'graph':_graph_metadata(tiles,old_nodes,old_edges,str(graphs[0].year),cfg,accepted_count=accepted),'nodes':old_nodes,'edges':old_edges}
    validate_combined_graph(old)
    nodes=[]
    for record,geom in zip(old_nodes,geoms):
        rgbs=[];probs=[];members=[];full=True
        for member in record['members']:
            tile=next(t for t in tiles if t.tile_id==member['tile_id']);artifact,classes,prob=inputs[tile.tile_id]
            mask=tile.ownership&(tile.instance_map==member['instance_id'])
            if 'image' in artifact.paths:
                with rasterio.open(artifact.paths['image']) as src: rgbs.append(src.read().transpose(1,2,0)[mask])
            probs.append(prob[:,mask]);members.extend(artifact.lineage.get(str(member['instance_id']),[]))
            if 'validation' in artifact.paths: full &= read_json(artifact.paths['validation']).get('probability_evidence')!='assigned_label_confidence_only'
        klass={v:k for k,v in CANONICAL_TO_OLD.items()}[record['segment_type']['id']]
        rgb=np.concatenate(rgbs) if rgbs else np.zeros((0,3),np.uint8)
        probability=np.concatenate(probs,axis=1).mean(1)
        nodes.append(_row(record['node_id'],klass,transform_geometry(geom,reference),rgb,probability,members,graphs[0].year,full))
    scale=math.sqrt(abs(reference.determinant));touches=[(e['source'],e['target'],e['shared_boundary_length_pixels']*scale) for e in old_edges]
    edges=_edges(nodes,touches,graphs[0].method,base.units)
    destination=context.output/'annual'/graphs[0].method/str(graphs[0].year)
    ownership={}
    from .storage import write_raster
    for i,tile in enumerate(tiles):
        path=destination/f'ownership_{i}.tif'
        write_raster(path,(tile.ownership&(tile.instance_map>0)).astype(np.uint8),inputs[tile.tile_id][0].grid,nodata=0)
        ownership[tile.tile_id]=str(path)
    combined=_persist(destination,nodes,edges,base,graphs[0].year,graphs[0].method,annual=True,accepted_overlap_matches=accepted,source_tiles=[str(g.directory) for g in graphs],source_ownership=ownership)
    paths=dict(combined.paths);paths['old_contract']=combined.directory/'old_combined_graph.json';atomic_json(paths['old_contract'],old)
    paths.update({f'ownership_{i}':Path(path) for i,path in enumerate(ownership.values())})
    combined=GraphArtifact(combined.year,combined.method,combined.directory,base,paths);atomic_json(combined.directory/'artifact.json',combined);save_manifest(combined.directory,paths)
    return combined


def export_old_graph_contract(graph,context):
    del context
    return graph.paths['old_contract']
