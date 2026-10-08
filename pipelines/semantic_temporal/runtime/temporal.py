"""Observed temporal correspondences and explicit classical pseudo labels."""
from pathlib import Path
from dataclasses import replace
import math
import numpy as np
import pandas as pd
import cv2
import rasterio
from rasterio.features import rasterize
from rasterio.warp import reproject,Resampling
from affine import Affine
from shapely import STRtree
from shapely.geometry import shape
from shapely.ops import unary_union
from .contracts import TemporalArtifact,SCHEMA
from .storage import atomic_json,read_json,save_manifest,load_segmentation,write_raster
from .graphs import NODE_FEATURES,EDGE_FEATURES
from .adapters import CANONICAL_TO_OLD


def _roof_polygons(docs,polygons,grid,context):
    from .roof_offsets import roof_offset_field,offset_building_labels
    from shapely.affinity import translate
    if grid.units!='metres': return polygons[0],{}
    bounds=unary_union(polygons[0]+polygons[1]).bounds;resolution=abs(grid.affine[0])
    sx=1;sy=-1 if grid.crs else 1
    affine=Affine(resolution,0,bounds[0],0,sy*resolution,bounds[3] if sy<0 else bounds[1])
    shape_hw=(round((bounds[3]-bounds[1])/resolution),round((bounds[2]-bounds[0])/resolution))
    if np.prod(shape_hw)>int(context.options.get('max_comparison_pixels') or 4_000_000):
        raise ValueError('Roof correspondence exceeds --max-comparison-pixels; select overlapping subsets or raise the budget')
    bids=[[i+1 for i,n in enumerate(d['nodes']) if n['class_id']==1] for d in docs]
    labels=[rasterize([(p,i+1) for i,p in enumerate(ps)],out_shape=shape_hw,transform=affine,dtype='int32') for ps in polygons]
    dy,dx,_=roof_offset_field(np.isin(labels[0],bids[0]),np.isin(labels[1],bids[1]),seg=max(32,round(384/resolution)),search=max(1,round(9/resolution)))
    _,shifts=offset_building_labels(labels[0],bids[0],np.isin(labels[1],bids[1]),dy,dx,local=max(1,round(3/resolution)))
    moved=list(polygons[0]);offsets={}
    for ident,(y,x) in shifts.items():
        offsets[ident-1]=[float(x*resolution),float(y*sy*resolution)]
        moved[ident-1]=translate(polygons[0][ident-1],xoff=offsets[ident-1][0],yoff=offsets[ident-1][1])
    return moved,offsets


def _observations(document):
    nodes=[]
    for node in document['nodes']:
        nodes.append({'node_id':node['id'],'year':node['year'],
                      'segment_type':{'id':CANONICAL_TO_OLD[node['class_id']],'name':node['class']},
                      'mean_semantic_confidence':node['confidence'],'area_m2':node['area'],'geometry_world':node['geometry']})
    return nodes


def build_temporal_graph(graphs,context):
    if len(graphs)!=2 or graphs[0].year>=graphs[1].year: raise ValueError('Change requires two distinct chronological years')
    if graphs[0].grid.crs!=graphs[1].grid.crs or graphs[0].grid.units!=graphs[1].grid.units: raise ValueError('Temporal graphs need a common prepared frame')
    docs=[read_json(g.paths['graph']) for g in graphs]
    polygons=[[shape(n['geometry']) for n in d['nodes']] for d in docs]
    footprints=[unary_union(p) for p in polygons];common=footprints[0].intersection(footprints[1])
    nodes=[];observations=[]
    for side,doc in enumerate(docs):
        known=unary_union([p for n,p in zip(docs[1-side]['nodes'],polygons[1-side]) if n['class_id']!=0 and n['confidence']>=.6])
        for node,polygon in zip(doc['nodes'],polygons[side]):
            row=dict(node);row['observed_fraction']=polygon.intersection(common).area/max(polygon.area,1e-9)
            row['known_counterpart_fraction']=polygon.intersection(known).area/max(polygon.area,1e-9)
            nodes.append(row);observations.append(row)
    edges=[]
    if graphs[0].method=='old':
        from src.stage5.matching import match_snapshots,MatchPolicy
        from .context import old_configuration
        documents=[read_json(g.paths['old_contract']) for g in graphs]
        matched=match_snapshots(documents[0],documents[1],polygons[0],polygons[1],MatchPolicy(**old_configuration(context).get('stage5',{}).get('matching',{})))
        for link in matched['links']:
            a,b=docs[0]['nodes'][link['i']],docs[1]['nodes'][link['j']]
            edges.append({'source':a['id'],'target':b['id'],'relation':link['relation'],**link['evidence']})
    else:
        tree=STRtree(polygons[1])
        roof_polygons,roof_offsets=_roof_polygons(docs,polygons,graphs[0].grid,context)
        for i,poly in enumerate(polygons[0]):
            for raw in sorted(tree.query(poly.union(roof_polygons[i]))):
                j=int(raw);other=polygons[1][j]
                roof=docs[0]['nodes'][i]['class_id']==docs[1]['nodes'][j]['class_id']==1
                compared=roof_polygons[i] if roof else poly
                shared=compared.intersection(other).area
                if shared<6*abs(Affine(*graphs[0].grid.affine).determinant) or shared<.25*min(poly.area,other.area): continue
                a,b=docs[0]['nodes'][i],docs[1]['nodes'][j];fa,fb=shared/poly.area,shared/other.area;iou=shared/(poly.area+other.area-shared)
                same=a['class_id']==b['class_id'] and a['class_id']!=0;score=math.sqrt(fa*fb)
                confidence='high' if iou>=.5 and same else 'medium' if (same and score>=.3) or (not same and score>=.5) else 'low'
                relation='identity' if confidence!='low' else 'overlap'
                if confidence=='low' and same and max(fa,fb)>=.8 and min(poly.area,other.area)>=50: relation='contained_in';confidence='medium'
                edges.append({'source':a['id'],'target':b['id'],'relation':relation,'shared_area':shared,'iou':iou,
                              'frame':'roof' if roof else 'ground','roof_shift':roof_offsets.get(i,[0.,0.]) if roof else [0.,0.],
                              'fraction_source':fa,'fraction_target':fb,'confidence':confidence,'edge_role':'primary' if confidence!='low' else 'secondary'})
    for e in edges: e.update(source_year=graphs[0].year,target_year=graphs[1].year,delta_years=graphs[1].year-graphs[0].year)
    directory=context.output/'temporal';directory.mkdir(parents=True,exist_ok=True)
    paths={k:directory/name for k,name in {'graph':'graph.json','tensors':'graph_tensors.npz','features':'feature_spec.json','targets':'training_targets.csv'}.items()}
    atomic_json(paths['graph'],{'schema':SCHEMA,'years':[g.year for g in graphs],'method':graphs[0].method,'units':graphs[0].grid.units,
                'annual_artifacts':[str(g.directory/'artifact.json') for g in graphs],'nodes':nodes,'temporal_edges':edges,
                'common_coverage_area':common.area,'registration':context.options.get('registration',{'method':'georeference_only','accuracy_status':'not_independently_verified'})})
    x=[];ei=[];ea=[];et=[];node_ids=[];offset=0
    for g in graphs:
        with np.load(g.paths['tensors'],allow_pickle=False) as t:
            x.append(t['x']);ei.append(t['edge_index']+offset);ea.append(t['edge_attr']);et.append(t['edge_type']);node_ids.extend(t['node_ids'].tolist());offset+=len(t['x'])
    index={ident:i for i,ident in enumerate(node_ids)}
    if edges:
        source=[index[e['source']] for e in edges];target=[index[e['target']] for e in edges]
        temporal=np.zeros((len(edges),len(EDGE_FEATURES)),np.float32)
        for i,e in enumerate(edges): temporal[i,3:5]=[e.get('fraction_source',e.get('coverage_source',0)),e.get('fraction_target',e.get('coverage_target',0))]
        reverse=temporal.copy();reverse[:,[3,4]]=temporal[:,[4,3]]
        ei.append(np.array([source+target,target+source],np.int64));ea.append(np.r_[temporal,reverse]);et.append(np.full(len(edges)*2,2,np.int64))
    np.savez_compressed(paths['tensors'],x=np.concatenate(x),edge_index=np.concatenate(ei,axis=1),edge_attr=np.concatenate(ea),edge_type=np.concatenate(et),node_ids=np.array(node_ids,dtype=str))
    atomic_json(paths['features'],{'node_features':NODE_FEATURES,'edge_features':EDGE_FEATURES,'relations':['touches','near','temporal'],'target':'building_change','units':graphs[0].grid.units})
    artifact=TemporalArtifact((graphs[0].year,graphs[1].year),directory,paths);atomic_json(directory/'artifact.json',artifact)
    save_manifest(directory,{k:v for k,v in paths.items() if k!='targets'})
    return artifact


def _image_evidence(document,grid,shape_hw,context):
    """Paint owned annual polygons, then registered image/probability evidence in that frame."""
    affine=Affine(*grid.affine);directory=Path(document['annual_artifacts'][0]).parent
    annuals=[read_json(p)['paths']['graph'] for p in document['annual_artifacts']]
    arrays=[]
    for annual_path in annuals:
        annual=read_json(annual_path);nodes=annual['nodes'];labels=rasterize([(shape(n['geometry']),i+1) for i,n in enumerate(nodes)],out_shape=shape_hw,transform=affine,dtype='int32') if nodes else np.zeros(shape_hw,np.int32)
        rgb=np.zeros((*shape_hw,3),np.uint8);shape_prob=np.zeros((9,*shape_hw),np.float32);max_prob=np.zeros_like(shape_prob)
        lookup={n['id']:i+1 for i,n in enumerate(nodes)}
        for tile_path in annual['source_tiles']:
            tile_doc=read_json(Path(tile_path)/'graph.json');artifact=load_segmentation(tile_doc['segmentation_artifact'])
            with rasterio.open(artifact.paths['image']) as src: image=src.read().transpose(1,2,0)
            base_probability=np.load(artifact.paths['probabilities'],allow_pickle=False)
            native_shape=np.load(artifact.paths.get('shape_probabilities',artifact.paths['probabilities']),allow_pickle=False)
            maxima=np.load(artifact.paths.get('max_probabilities',artifact.paths['probabilities']),allow_pickle=False)
            relative=(~affine)*Affine(*artifact.grid.affine);x,y=round(relative.c),round(relative.f);h,w=image.shape[:2]
            if not np.allclose([relative.a,relative.b,relative.d,relative.e],[1,0,0,1],atol=1e-6): raise ValueError('Comparison evidence must share resolution')
            from .adapters import read_band
            owner=read_band(annual['source_ownership'][f'{artifact.year}:{artifact.tile_key}'])>0
            left,top=max(0,x),max(0,y);right,bottom=min(shape_hw[1],x+w),min(shape_hw[0],y+h)
            if left>=right or top>=bottom: continue
            destination=(slice(top,bottom),slice(left,right))
            source_slice=(slice(top-y,bottom-y),slice(left-x,right-x))
            own=(labels[destination]>0)&owner[source_slice]
            quality=read_json(artifact.paths['validation'])
            if quality.get('probability_evidence')=='assigned_label_confidence_only':
                from .models import load_models
                models=load_models(context);samples=[]
                from .segmentation.semantic import OEM_TO_CLASS
                for scale in (1.,1.5,2.):
                    values=models.probabilities(image,scale)
                    canonical=np.zeros((9,h,w),np.float32);canonical[0]=cv2.resize(values[0],(w,h))
                    for source,target in OEM_TO_CLASS.items(): canonical[target]=cv2.resize(values[source],(w,h))
                    samples.append(canonical)
                native_shape=samples[-1];maxima=np.maximum.reduce(samples)
            rgb[destination][own]=image[source_slice][own]
            shape_prob[(slice(None),*destination)][:,own]=native_shape[(slice(None),*source_slice)][:,own]
            max_prob[(slice(None),*destination)][:,own]=maxima[(slice(None),*source_slice)][:,own]
        known=np.isin(labels,[i+1 for i,n in enumerate(nodes) if n['class_id']!=0])
        evidence={}
        for name,klass in [('building',1),('tree',3),('road',2)]:
            evidence[name+'_2x']=shape_prob[klass]
            evidence[name+'_max']=np.where(known,max_prob[klass],1.)
        arrays.append((labels,rgb,evidence))
    return arrays


def detect_classical(graph,context):
    document=read_json(graph.paths['graph']);nodes=document['nodes'];years=document['years'];method=document['method']
    earlier=[n for n in nodes if n['year']==years[0]];later=[n for n in nodes if n['year']==years[1]]
    if not earlier or not later: raise ValueError('Both years need observed nodes')
    source={n['id']:n for n in nodes};events=[];changes=[];units=document['units']
    if method=='old' and units!='pixels':
        from src.stage5.events import detect_change_events,ChangePolicy
        from .context import old_configuration
        settings=dict(old_configuration(context).get('stage5',{}).get('changes',{}))
        if context.options.get('boundary_tolerance') is not None: settings['boundary_tolerance_m']=context.options['boundary_tolerance']
        observations=[_observations({'nodes':v}) for v in (earlier,later)]
        events_result=detect_change_events(observations[0],observations[1],[shape(n['geometry']) for n in earlier],[shape(n['geometry']) for n in later],ChangePolicy(**settings))
        events=events_result['events'];by_node={}
        for event in events:
            if event['class_id']==7:
                for ident in event['target_node_ids']+event['source_node_ids']: by_node.setdefault(ident,[]).append(event)
        for node in nodes:
            candidates=by_node.get(node['id'],[])
            positive=next((e for e in candidates if e['relation'] in ('constructed','expanded')
                           and node['class_id']==1 and node['year']==years[1]
                           and node['id'] in e['target_node_ids']),None)
            persistent=next((e for e in candidates if e['relation']=='persisted'),None)
            event='new_building' if positive and positive['relation']=='constructed' else 'extension' if positive else 'persistent' if persistent else 'unexplained_no_evidence'
            changes.append({'gid':node['id'],'year':node['year'],'event':event,'label':1 if positive else 0 if persistent and node['class_id']==1 else None,'confidence':node['confidence'],'source':'stage5_classical_rule'})
    elif units=='pixels':
        changes=[{'gid':n['id'],'year':n['year'],'event':'correspondence_only','label':None,'confidence':n['confidence'],'source':'original_pixel_rule'} for n in nodes]
    else:
        from .change_evidence import EvidenceRules
        from .roof_offsets import roof_offset_field
        bounds=unary_union([shape(n['geometry']) for n in nodes]).bounds
        base=read_json(document['annual_artifacts'][0])['grid'];resolution=abs(base['affine'][0])
        affine=Affine(resolution,0,bounds[0],0,-resolution if base['crs'] else resolution,bounds[3] if base['crs'] else bounds[1])
        height=round((bounds[3]-bounds[1])/resolution);width=round((bounds[2]-bounds[0])/resolution)
        if height*width>int(context.options.get('max_comparison_pixels') or 4_000_000): raise ValueError('Comparison exceeds memory budget; select overlapping subsets or raise --max-comparison-pixels')
        from .contracts import GridSpec
        grid=GridSpec(base['crs'],tuple(affine)[:6],units,(resolution,resolution))
        arrays=_image_evidence(document,grid,(height,width),context);(a,rgb_a,ev_a),(b,rgb_b,ev_b)=arrays
        tables=[]
        for subset in (earlier,later):
            tables.append(pd.DataFrame([{'node_id':i+1,'object_type':n['class'],'area_m2':n['area'],'mean_conf':n['confidence']} for i,n in enumerate(subset)]).set_index('node_id'))
        dy,dx,_=roof_offset_field(np.isin(a,[i+1 for i,n in enumerate(earlier) if n['class_id']==1]),np.isin(b,[i+1 for i,n in enumerate(later) if n['class_id']==1]),seg=max(32,round(384/resolution)),search=max(1,round(9/resolution)))
        change_map,after,before=EvidenceRules(resolution**2).detect(ev_a,ev_b,a,tables[0],b,tables[1],dy,dx,rgb_a,rgb_b)
        common=(a>0)&(b>0);change_map[~common]=0
        write_raster(graph.directory/'change_map.tif',change_map,grid,nodata=0)
        # Persist every named raster event, including non-building rules whose
        # detector does not return a per-node evidence table.
        from rasterio.features import shapes
        from .change_evidence import CODES
        for relation,code in CODES.items():
            region=change_map==code
            for geometry,value in shapes(region.astype(np.uint8),mask=region,transform=affine):
                if not value: continue
                polygon=shape(geometry)
                events.append({'event_id':f'{years[0]}:{years[1]}:{relation}:{len(events)}',
                               'relation':relation,'source_year':years[0],'target_year':years[1],
                               'area':polygon.area,'area_units':'metres squared','geometry':geometry,
                               'source':'semantic_image_evidence_rule','confidence':'rule_candidate'})
        evidence_tables=[before,after]
        for side,subset in enumerate((earlier,later)):
            evidence=evidence_tables[side];by_id=evidence.set_index('node_id').to_dict('index') if len(evidence) else {}
            for i,node in enumerate(subset,1):
                ev=by_id.get(i,{});status=ev.get('change','unchanged');label=None
                predecessor=any(e['target']==node['id'] and source[e['source']]['class_id']==1 and e['relation'] in ('identity','contained_in') for e in document['temporal_edges'])
                if node['class_id']==1 and ev.get('confidence')=='high' and status in ('new_building','extension'):
                    event='extension' if predecessor else 'new_building';label=1
                elif node['class_id']==1 and status=='unchanged' and node['confidence']>=.6 and any(
                    (e['source']==node['id'] and source[e['target']]['class_id']==1) or
                    (e['target']==node['id'] and source[e['source']]['class_id']==1) for e in document['temporal_edges'] if e['relation'] in ('identity','contained_in')):
                    event='persistent';label=0
                elif status!='unchanged': event=status
                else:
                    same_class=any(((e['source']==node['id'] and source[e['target']]['class_id']==node['class_id']) or
                                    (e['target']==node['id'] and source[e['source']]['class_id']==node['class_id']))
                                   and e['relation'] in ('identity','contained_in') for e in document['temporal_edges'])
                    event='persistent' if same_class else 'not_assessed' if node['class_id']!=0 else 'unknown_observation'
                nonbuilding_code={(0,3):('tree_loss',4),(1,3):('tree_gain',5),(1,2):('new_road',6)}.get((side,node['class_id']))
                if nonbuilding_code:
                    relation,code=nonbuilding_code;count=int(((change_map==code)&(arrays[side][0]==i)).sum())
                    if count: event=relation;ev={**ev,'changed_area_m2':count*resolution**2,'rule_code':code}
                changes.append({'gid':node['id'],'year':node['year'],'event':event,'label':label,'confidence':node['confidence'],'source':'semantic_image_evidence_rule','evidence':ev})
    # Final observability gates override every engine, including pseudo negatives.
    for row in changes:
        node=source[row['gid']]
        if node['observed_fraction']<.95: row.update(event='coverage_limited',label=None)
        elif node['class_id']==0 or node['known_counterpart_fraction']<.95: row.update(event='unknown_observation',label=None)
        elif node['confidence']<.6: row['label']=None
    result={'status':'complete','years':years,'units':units,'engine':'stage5_classical' if method=='old' else 'semantic_image_evidence',
            'node_changes':changes,'events':events,'audit_status':'not_provided','target_definition':'building construction/extension pseudo labels; agreement is not independently verified accuracy'}
    atomic_json(graph.directory/'classical_changes.json',result)
    pd.DataFrame([{'gid':r['gid'],'label':r['label'],'confidence':r['confidence'],'source':r['source'],'year':r['year']} for r in changes if r['label'] is not None],columns=['gid','label','confidence','source','year']).to_csv(graph.paths['targets'],index=False)
    paths={**graph.paths,'classical':graph.directory/'classical_changes.json'}
    if (graph.directory/'change_map.tif').exists(): paths['change_map']=graph.directory/'change_map.tif'
    save_manifest(graph.directory,paths)
    return result
