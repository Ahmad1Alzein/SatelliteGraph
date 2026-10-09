"""Symmetric class gain/loss evidence with explicit unknown and observation masks."""
import cv2
import numpy as np
from scipy import ndimage
from affine import Affine
from .change_contracts import KNOWN_CLASSES
from .roof_offsets import roof_offset_field


def _components(mask,pixel_area,minimum):
    labels,_=ndimage.label(mask,structure=ndimage.generate_binary_structure(2,1))
    sizes=np.bincount(labels.ravel());keep=sizes*pixel_area>=minimum;keep[0]=False
    return keep[labels]


def detect_evidence(frame,policy):
    first,last=frame['years'];a,b=frame['data'][first],frame['data'][last]
    area=abs(Affine(*frame['grid'].affine).determinant);pixel=np.sqrt(area)
    common=a['valid']&b['valid'];known=common&np.isin(a['classes'],range(1,9))&np.isin(b['classes'],range(1,9))
    clear=[cv2.cvtColor(d['rgb_original'],cv2.COLOR_RGB2GRAY)>=policy['shadow_cutoff'] for d in (a,b)]
    eligible=known&clear[0]&clear[1]
    transition=np.zeros(common.shape,np.uint16);transition[~common]=65535
    transition[known]=16*a['classes'][known].astype(np.uint16)+b['classes'][known]
    window=max(1,round(policy['absence_window_m']/pixel))
    roof=frame.get('roof_offsets')
    if roof is None and (a['classes']==1).any() and (b['classes']==1).any():
        dy,dx,_=roof_offset_field(a['classes']==1,b['classes']==1,seg=max(32,round(384/pixel)),search=max(1,round(9/pixel)))
        roof=(dy,dx)
    if roof is None: roof=(np.zeros(common.shape[1]),np.zeros(common.shape[1]))
    yy,xx=np.mgrid[:common.shape[0],:common.shape[1]].astype(np.float32)
    mapx=xx-np.asarray(roof[1],np.float32)[None,:];mapy=yy-np.asarray(roof[0],np.float32)[None,:]
    def remap(array,nearest=False,border=0):
        return cv2.remap(array.astype(np.float32),mapx,mapy,cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,borderValue=border)
    aligned_labels=remap(a['labels'],nearest=True).astype(np.int32)
    aligned_building=remap(a['classes']==1,nearest=True)>0
    aligned_valid=remap(a['valid'],nearest=True)>0
    aligned_known=remap(np.isin(a['classes'],range(1,9)),nearest=True)>0
    aligned_clear=remap(clear[0],nearest=True)>0
    roof_eligible=b['valid']&aligned_valid&aligned_known&np.isin(b['classes'],range(1,9))&aligned_clear&clear[1]
    # Roof differences with a stable aligned roof are not evidence for land change.
    roof_disagreement=((a['classes']==1)!=(b['classes']==1)) & (aligned_building==(b['classes']==1))
    raw={};masks={};candidates={};stable={}
    for klass,name in KNOWN_CLASSES.items():
        present_threshold=policy['building_presence_probability'] if klass==1 else policy['presence_probability']
        old=a['shape_probabilities'][klass];old_max=a['max_probabilities'][klass]
        old_class=a['classes']==klass;ok=eligible.copy()
        if klass==1:
            old=remap(old);old_max=remap(old_max,border=1);old_class=aligned_building;ok=roof_eligible
        else:
            ok&=~roof_disagreement
        new=b['shape_probabilities'][klass];new_max=b['max_probabilities'][klass]
        absent_old=ndimage.maximum_filter(np.where(ok,old_max,1.),size=window,mode='constant',cval=1.)<=policy['absence_probability']
        absent_new=ndimage.maximum_filter(np.where(ok,new_max,1.),size=window,mode='constant',cval=1.)<=policy['absence_probability']
        present_old=old>=present_threshold;present_new=new>=present_threshold
        gain=ok&(b['classes']==klass)&present_new&absent_old
        loss=ok&old_class&present_old&absent_new
        if klass==1:
            gain=ndimage.binary_opening(gain,structure=ndimage.generate_binary_structure(2,1))
            loss=ndimage.binary_opening(loss,structure=ndimage.generate_binary_structure(2,1))
        for direction,mask in [('gain',gain),('loss',loss)]:
            task=f'{name}_{direction}';raw[task]=mask
            masks[task]=_components(mask,area,policy['minimum_area_m2'][name])
            changed=(~old_class&(b['classes']==klass)) if direction=='gain' else (old_class&(b['classes']!=klass))
            candidates[task]=_components(ok&changed,area,policy['minimum_area_m2'][name])
        stable[name]=ok&old_class&(b['classes']==klass)&present_old&present_new
    old_veg=a['shape_probabilities'][3:6].sum(0);new_veg=b['shape_probabilities'][3:6].sum(0)
    old_veg_max=a['max_probabilities'][3:6].sum(0);new_veg_max=b['max_probabilities'][3:6].sum(0)
    veg_ok=eligible&~roof_disagreement
    vegetation_gain=veg_ok&(new_veg>=policy['presence_probability'])&(ndimage.maximum_filter(np.where(veg_ok,old_veg_max,1.),size=window,mode='constant',cval=1.)<=policy['absence_probability'])
    vegetation_loss=veg_ok&(old_veg>=policy['presence_probability'])&(ndimage.maximum_filter(np.where(veg_ok,new_veg_max,1.),size=window,mode='constant',cval=1.)<=policy['absence_probability'])
    return {'masks':masks,'raw_masks':raw,'candidate_masks':candidates,'stable':stable,
            'transition':transition,'transition_encoding':{'known':'16 * source_class + target_class','unknown':0,'unobserved':65535},
            'common_valid':common,'known':known,'eligible':eligible,'roof_eligible':roof_eligible,
            'roof_labels_earlier':aligned_labels,'roof_offsets':roof,'roof_disagreement':roof_disagreement,
            'vegetation_gain':vegetation_gain,'vegetation_loss':vegetation_loss,'pixel_area':area}
