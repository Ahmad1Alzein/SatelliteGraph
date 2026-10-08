"""Lossless class/coordinate adapters between established and general artifacts."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import copy
import csv
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from scipy import ndimage
from .contracts import GridSpec,SegmentationArtifact
from .storage import atomic_json,read_json,write_raster,save_manifest
from .registration import prepare_tile
from .segmentation.parcels import FOUR

OLD_TO_CANONICAL={0:6,1:4,2:7,3:2,4:3,5:8,6:5,7:1,8:0,255:255}
CANONICAL_TO_OLD={value:key for key,value in OLD_TO_CANONICAL.items()}


def read_band(path):
    with rasterio.open(path) as src: return src.read(1)


def load_old_segmentation(directory,tile):
    from src.refinement import load_refinement_result
    old=load_refinement_result(directory);directory=Path(directory)
    grid=tile.grid
    if (directory/'runtime_grid.json').exists():
        value=read_json(directory/'runtime_grid.json');grid=GridSpec(value['crs'],tuple(value['affine']),value['units'],tuple(value['physical_pixel_size']) if value['physical_pixel_size'] else None)
    elif old.tile_metadata.get('tile_affine_transform'):
        grid=replace(grid,affine=tuple(old.tile_metadata['tile_affine_transform']))
    classes=np.full(old.semantic_map.shape,255,np.uint8)
    for source,target in OLD_TO_CANONICAL.items(): classes[old.semantic_map==source]=target
    probabilities=np.zeros((9,*classes.shape),np.float32)
    for target in range(9): probabilities[target]=np.where(classes==target,old.confidence_map,0)
    # Only confidence for the assigned label exists in old fused artifacts. Never invent other channels.
    out=directory/'canonical';out.mkdir(exist_ok=True)
    paths={k:out/name for k,name in {'labels':'segment_ids.tif','classes':'segment_classes.tif','valid':'valid_mask.tif','probabilities':'probabilities.npy','nodes':'nodes.csv','lineage':'lineage.json','validation':'validation.json'}.items()}
    write_raster(paths['labels'],old.instance_map.astype(np.int32),grid,nodata=0)
    write_raster(paths['classes'],classes,grid,nodata=255);write_raster(paths['valid'],old.valid_mask.astype(np.uint8),grid,nodata=0)
    np.save(paths['probabilities'],probabilities)
    lineage={str(i):[{'year':tile.year,'tile':tile.key,'segment':record['segment_id'],'method':'old','old_lineage':record.get('lineage',{})}] for i,record in enumerate(old.segments,1)}
    atomic_json(paths['lineage'],lineage);atomic_json(paths['validation'],{'method':'old','probability_evidence':'assigned_label_confidence_only','valid_pixels':int(old.valid_mask.sum())})
    pd.DataFrame([{'segment_id':i,'class_id':OLD_TO_CANONICAL[int(record['label']['id'])],'pixels':record['area_pixels'],'confidence':record.get('mean_semantic_confidence',0)} for i,record in enumerate(old.segments,1)]).to_csv(paths['nodes'],index=False)
    artifact=SegmentationArtifact(tile.year,tile.key,out,grid,old.instance_map.shape,paths,lineage)
    atomic_json(out/'artifact.json',artifact);save_manifest(out,paths,method='old')
    return artifact


def connected_region_view(artifact):
    labels=read_band(artifact.paths['labels']);result=np.zeros_like(labels);lineage={};next_id=0
    for parent in np.unique(labels[labels>0]):
        components,count=ndimage.label(labels==parent,structure=FOUR)
        for ident in range(1,count+1):
            next_id+=1;result[components==ident]=next_id
            lineage[str(next_id)]=artifact.lineage.get(str(parent),[{'segment':int(parent),'tile':artifact.tile_key,'year':artifact.year}])
    if np.array_equal(result,labels): return artifact
    directory=artifact.directory/'connected';directory.mkdir(exist_ok=True)
    paths=dict(artifact.paths);paths['labels']=directory/'segment_ids.tif';paths['lineage']=directory/'lineage.json'
    write_raster(paths['labels'],result,artifact.grid,nodata=0);atomic_json(paths['lineage'],lineage)
    return replace(artifact,directory=directory,paths=paths,lineage=lineage)


def export_old_refinement_contract(artifact,context):
    """Publish the real Stage 2R contract without applying old refinement to new masks."""
    from src.stage2.classes import class_registry
    from src.refinement.records import refresh_segment_record
    from src.refinement.validation import validate_refined_segmentation
    from src.refinement.storage import save_segmentation_bundle
    from src.refinement.topology import RefinementResult
    from src.refinement.pipeline import _Stage2Source,_persist_complete_result
    from src.config import load_config,select_refinement_config
    from src.stage2.artifacts import now
    artifact=connected_region_view(artifact)
    labels=read_band(artifact.paths['labels']);canonical=read_band(artifact.paths['classes']);valid=read_band(artifact.paths['valid'])>0
    semantics=np.full(labels.shape,255,np.uint8)
    for source,target in CANONICAL_TO_OLD.items(): semantics[canonical==source]=target
    probabilities=np.load(artifact.paths['probabilities'],allow_pickle=False)
    confidence=np.zeros(labels.shape,np.float32)
    for target in range(9): confidence[canonical==target]=probabilities[target,canonical==target]
    source_map=np.where(valid,2,255).astype(np.uint8);source_map[canonical==0]=3
    tile_id=f'{artifact.year}_{artifact.tile_key}'
    metadata={'tile_id':tile_id,'source_image_id':str(artifact.year),'width':labels.shape[1],'height':labels.shape[0],
              'tile_affine_transform':list(artifact.grid.affine),'global_x0':0,'global_y0':0,'source_crs':artifact.grid.crs}
    records=[];masks=[]
    names={int(r['id']):r['name'] for r in class_registry()}
    for ident in range(1,int(labels.max())+1):
        mask=labels==ident;klass=int(semantics[mask][0])
        ancestor={'instance_id':ident,'external_lineage':artifact.lineage.get(str(ident),[])}
        record={'instance_id':ident,'mask_index':ident-1,'tile_id':tile_id,'segment_id':f'segment_{ident}',
                'entity_id':f'{tile_id}::entity::{ident}','label':{'id':klass,'name':names[klass]},
                'stage2_metadata':ancestor,'stage2_metadata_records':[ancestor],'source_stage2_instance_ids':[ident],
                'lineage':{'source_stage2_instance_ids':[ident],'refinement_operation_ids':[]},'refinement_history':[]}
        refresh_segment_record(record,mask,semantic_class_id=klass,confidence_map=confidence,source_map=source_map,tile_metadata=metadata)
        records.append(record);masks.append(mask)
    validation=validate_refined_segmentation(labels,semantics,valid,records,masks,confidence_map=confidence,source_map=source_map)
    directory=context.output/'adapters'/str(artifact.year)/artifact.tile_key
    source_dir=directory/'source';source_dir.mkdir(parents=True,exist_ok=True)
    save_segmentation_bundle(source_dir/'segmentation.npz',instance_map=labels,semantic_map=semantics,confidence_map=confidence,source_map=source_map,valid_mask=valid)
    atomic_json(source_dir/'class_registry.json',class_registry())
    rgb=None
    if 'image' in artifact.paths:
        with rasterio.open(artifact.paths['image']) as src: rgb=src.read().transpose(1,2,0)
    result=SimpleNamespace(tile_id=tile_id,segments=records,instance_map=labels,semantic_map=semantics,confidence_map=confidence,source_map=source_map,valid_mask=valid,tile_metadata=metadata,report={'coordinate_reference':{'units':artifact.grid.units,'crs':artifact.grid.crs}})
    source=_Stage2Source(source_dir,{'schema_version':'satellitegraph.stage2.full-segmentation.v1'},result,[source_dir/'segmentation.npz',source_dir/'class_registry.json'],rgb,None,rgb is not None,None,None)
    refined=RefinementResult(masks,records,labels,semantics,confidence,source_map,valid,[],{'external_lineage':artifact.lineage},{'object_identity':{'require_every_object_four_connected':True}})
    cfg=select_refinement_config(load_config())
    _persist_complete_result(output_directory=directory/'13_refinement',source=source,refined=refined,proposals=[],decisions=[],proposal_report={},validation=validation,refinement_config=cfg,workflow_mode='external_adapter',automatic_selection_report=None,started=now())
    atomic_json(directory/'13_refinement/runtime_grid.json',artifact.grid)
    return directory/'13_refinement'


def run_old_segmentation(context):
    """Use unchanged Stage 2 + automatic Stage 2R on explicitly prepared inputs."""
    from src.folder_pipeline import run_folder
    from src.config import load_config
    artifacts=[]
    root=Path(context.options.get('project_root') or Path(__file__).resolve().parents[3])
    for year in context.years:
        for tile in year.tiles:
            rgb,valid,grid=prepare_tile(tile,context)
            stage=context.output/'old'/str(year.year)/tile.key;inputs=stage/'prepared';inputs.mkdir(parents=True,exist_ok=True)
            image=inputs/'tile_00000.png';Image.fromarray(np.dstack((rgb,valid.astype(np.uint8)*255))).save(image)
            raster=None
            if grid.crs:
                raster=inputs/'source.tif';write_raster(raster,rgb.transpose(2,0,1),grid)
                h,w=valid.shape
                with (inputs/'metadata.csv').open('w',newline='',encoding='utf-8') as stream:
                    writer=csv.DictWriter(stream,fieldnames=['tile_id','tile_name','x0','y0','width','height','alpha_band','valid_pixel_count'])
                    writer.writeheader();writer.writerow(dict(tile_id=0,tile_name=image.name,x0=0,y0=0,width=w,height=h,alpha_band=4,valid_pixel_count=int(valid.sum())))
            cfg=copy.deepcopy(context.options.get('old_config') or load_config(context.options.get('config') or root/'config.yaml'))
            for part in ('instance_model','semantic_model'): cfg['stage2'][part]['device']=context.options.get('device') or 'cpu'
            run_folder(inputs,year.year,stage/'run',source_raster=raster,selected_image=image,config=cfg,project_root=root,show_progress=bool(context.options.get('progress')),stop_after='segmentation')
            directories=list((stage/'run/stage2').glob('*/13_refinement'))
            if len(directories)!=1: raise ValueError('Expected one old refined tile')
            atomic_json(directories[0]/'runtime_grid.json',grid)
            artifact=load_old_segmentation(directories[0],replace(tile,grid=grid))
            paths=dict(artifact.paths);paths['image']=artifact.directory/'rgb.tif';write_raster(paths['image'],rgb.transpose(2,0,1),grid)
            artifact=replace(artifact,paths=paths);atomic_json(artifact.directory/'artifact.json',artifact);save_manifest(artifact.directory,paths,method='old')
            artifacts.append(artifact)
    return artifacts
