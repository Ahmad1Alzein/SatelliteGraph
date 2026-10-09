"""Masked and georeferenced semantic/object segmentation artifacts."""
from dataclasses import replace
import math
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from affine import Affine
from ..contracts import SegmentationArtifact
from ..registration import prepare_tile
from ..storage import atomic_json,write_raster,save_manifest
from ..models import load_models
from .regions import segment,DEFAULT_SETTINGS
from .parcels import FOUR


def segment_tile(tile,context,*,models=None):
    rgb,valid,grid=prepare_tile(tile,context)
    if grid.units!='metres': raise ValueError('Segmentation metric rules require physical pixel size; supply --pixel-size')
    transform=Affine(*grid.affine)*Affine.scale(.5,.5)
    area=abs(transform.determinant);pixel=math.sqrt(area)
    shape=(valid.shape[0]*2,valid.shape[1]*2)
    budget=int(context.options.get('max_pixels') or 16_000_000)
    if shape[0]*shape[1]>budget: raise ValueError('Tile exceeds --max-pixels memory budget; use metadata-based smaller tiles or raise the budget')
    observed=cv2.resize(valid.astype(np.uint8),shape[::-1],interpolation=cv2.INTER_NEAREST)>0
    settings={**DEFAULT_SETTINGS,'pixel_area':area,'smooth_px':max(1,round(5.25/pixel)),
              'road_close_px':max(1,round(3.75/pixel))}
    building={'building_prob_2x':.4,'min_building_m2':27.,'neck_min_core_m2':20.,'neck_split_radius_px':max(0,round(1.5/pixel))}
    log={}
    labels,classes,sources,probabilities,image,rule,_,merge_log,shape_probability,max_probability=segment(rgb,area,log,P=settings,
        models=models or load_models(context),valid=observed,building_settings=building)
    labels[~observed]=0
    # Preserve unknown observations rather than folding them into bare soil/land.
    unknown=observed & (probabilities.argmax(0)==0)
    labels[unknown]=0
    result=np.zeros(shape,np.int32);class_map=np.full(shape,255,np.uint8);lineage={};rows=[]
    for original in np.unique(labels[observed]):
        mask=(labels==original)&observed
        if original==0:
            groups=[(mask&unknown,0),(mask&~unknown,-1)]
        else: groups=[(mask,int(classes[original]))]
        for selected,klass in groups:
            if klass==-1:
                # Growth's finite iteration limit must not leave large isolated observed areas blank.
                candidates=range(1,9)
            else: candidates=[klass]
            for target in candidates:
                selected_class=selected if klass!=-1 else selected&(probabilities[1:].argmax(0)+1==target)
                components,count=ndimage.label(selected_class,structure=FOUR)
                for ident in range(1,count+1):
                    piece=components==ident;new=len(rows)+1;result[piece]=new;class_map[piece]=target
                    parent={'year':tile.year,'tile':tile.key,'segment':int(original),'method':'new'}
                    lineage[str(new)]=[parent]
                    rows.append({'segment_id':new,'class_id':target,'pixels':int(piece.sum()),'area':float(piece.sum()*area),
                                 'confidence':float(probabilities[target,piece].mean()),'source':sources.get(int(original),'observed_fallback')})
    if np.any(observed&(result==0)): raise ValueError('Segmentation failed observed-pixel coverage')
    directory=context.output/'segmentation'/str(tile.year)/tile.key
    paths={k:directory/name for k,name in {'labels':'segment_ids.tif','classes':'segment_classes.tif','valid':'valid_mask.tif',
          'probabilities':'probabilities.npy','image':'rgb.tif','nodes':'nodes.csv','lineage':'lineage.json','validation':'validation.json','merges':'merge_log.csv'}.items()}
    outgrid=replace(grid,affine=tuple(transform)[:6],physical_pixel_size=(pixel,pixel))
    write_raster(paths['labels'],result,outgrid,nodata=0);write_raster(paths['classes'],class_map,outgrid,nodata=255)
    write_raster(paths['valid'],observed.astype(np.uint8),outgrid,nodata=0);write_raster(paths['image'],image.transpose(2,0,1),outgrid)
    original,_,_=prepare_tile(tile,context,model=False)
    original=cv2.resize(original,shape[::-1],interpolation=cv2.INTER_LINEAR)
    original[~observed]=0;paths['original_image']=directory/'original_rgb.tif'
    write_raster(paths['original_image'],original.transpose(2,0,1),outgrid)
    probabilities[:,~observed]=0;np.save(paths['probabilities'],probabilities)
    for key,data in [('shape_probabilities',shape_probability),('max_probabilities',max_probability)]:
        data[:,~observed]=0;paths[key]=directory/(key+'.npy');np.save(paths[key],data)
    pd.DataFrame(rows).to_csv(paths['nodes'],index=False);merge_log.to_csv(paths['merges'],index=False)
    atomic_json(paths['lineage'],lineage);atomic_json(paths['validation'],{**log,'valid_pixels':int(observed.sum()),'segments':len(rows),'units':grid.units,'upsampling_adds_detail':False})
    artifact=SegmentationArtifact(tile.year,tile.key,directory,outgrid,shape,paths,lineage)
    atomic_json(directory/'artifact.json',artifact);save_manifest(directory,paths,method='new')
    return artifact
