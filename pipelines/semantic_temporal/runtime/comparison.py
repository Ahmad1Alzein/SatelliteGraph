"""Owned scientific pixels and original/model RGB on one comparison frame."""
from pathlib import Path
import math
import cv2
import numpy as np
import rasterio
from affine import Affine
from rasterio.features import rasterize
from shapely.geometry import shape
from shapely.ops import unary_union
from .contracts import GridSpec
from .storage import read_json,load_segmentation
from .adapters import read_band


def warp_array(data, source_affine, target_affine, shape_hw, *, nearest=False):
    """Warp HxW or CxHxW arrays; affine coordinates locate pixel edges."""
    matrix=Affine.translation(-.5,-.5)*(~target_affine)*source_affine*Affine.translation(.5,.5)
    matrix=np.asarray(tuple(matrix)[:6]).reshape(2,3)
    flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    if data.ndim==2:
        return cv2.warpAffine(data,matrix,shape_hw[::-1],flags=flags,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return np.stack([cv2.warpAffine(channel,matrix,shape_hw[::-1],flags=flags,
                     borderMode=cv2.BORDER_CONSTANT,borderValue=0) for channel in data])


def _probabilities(artifact,image,context,models):
    quality=read_json(artifact.paths['validation']) if 'validation' in artifact.paths else {}
    if quality.get('probability_evidence')!='assigned_label_confidence_only':
        return (np.load(artifact.paths.get('shape_probabilities',artifact.paths['probabilities']),allow_pickle=False),
                np.load(artifact.paths.get('max_probabilities',artifact.paths['probabilities']),allow_pickle=False),models)
    if models is None:
        from .models import load_models
        models=load_models(context)
    from .segmentation.semantic import OEM_TO_CLASS
    h,w=image.shape[:2];samples=[]
    for scale in (1.,1.5,2.):
        values=models.probabilities(image,scale);canonical=np.zeros((9,h,w),np.float32)
        canonical[0]=cv2.resize(values[0],(w,h))
        for source,target in OEM_TO_CLASS.items(): canonical[target]=cv2.resize(values[source],(w,h))
        samples.append(canonical)
    return samples[-1],np.maximum.reduce(samples),models


def load_comparison(graph,context):
    return _load_document(read_json(graph.paths['graph']),context)


def load_annual_frame(graph,context):
    doc=read_json(graph.paths['graph'])
    return _load_document({'annual_artifacts':[str(graph.directory/'artifact.json')],
                           'nodes':doc['nodes'],'units':doc['units'],'years':[graph.year]},context)


def _load_document(doc,context):
    annuals=[read_json(read_json(p)['paths']['graph']) for p in doc['annual_artifacts']]
    nodes=doc['nodes'];geometries=[shape(n['geometry']) for n in nodes]
    base=annuals[0]['grid'];resolution=abs(base['affine'][0])
    bounds=unary_union(geometries).bounds if geometries else (base['affine'][2],base['affine'][5],base['affine'][2]+resolution,base['affine'][5]+resolution)
    sign=-1 if base['crs'] else 1
    affine=Affine(resolution,0,bounds[0],0,sign*resolution,bounds[3] if sign<0 else bounds[1])
    height=max(1,round((bounds[3]-bounds[1])/resolution));width=max(1,round((bounds[2]-bounds[0])/resolution))
    if height*width>int(context.options.get('max_comparison_pixels') or 4_000_000):
        raise ValueError(f'Comparison requires {height*width} pixels; raise --max-comparison-pixels')
    shape_hw=(height,width);data={};models=None
    for annual in annuals:
        subset=annual['nodes'];labels=rasterize([(shape(n['geometry']),i+1) for i,n in enumerate(subset)],
            out_shape=shape_hw,transform=affine,dtype='int32') if subset else np.zeros(shape_hw,np.int32)
        classes=np.full(shape_hw,255,np.uint8)
        class_lookup=np.array([255]+[n['class_id'] for n in subset],np.uint8)
        classes[:]=class_lookup[labels]
        original=np.zeros((height,width,3),np.uint8);model=np.zeros_like(original)
        probabilities=np.zeros((9,height,width),np.float32);maxima=np.zeros_like(probabilities)
        observed=np.zeros(shape_hw,bool)
        for tile_path in annual['source_tiles']:
            tile_doc=read_json(Path(tile_path)/'graph.json');artifact=load_segmentation(tile_doc['segmentation_artifact'])
            source_affine=Affine(*artifact.grid.affine)
            with rasterio.open(artifact.paths['image']) as src: image=src.read().transpose(1,2,0)
            with rasterio.open(artifact.paths.get('original_image',artifact.paths['image'])) as src: rgb=src.read().transpose(1,2,0)
            valid=read_band(artifact.paths['valid'])>0
            owner=read_band(annual['source_ownership'][f'{artifact.year}:{artifact.tile_key}'])>0
            use=valid&owner
            # A target pixel is assigned once even if warped edge footprints overlap.
            painted=warp_array(use.astype(np.uint8),source_affine,affine,shape_hw,nearest=True)>0
            painted&=~observed&(labels>0)
            if not painted.any(): continue
            native_shape,native_max,models=_probabilities(artifact,image,context,models)
            weight=warp_array(use.astype(np.float32),source_affine,affine,shape_hw)
            def paint(values,destination,*,channels=False):
                warped=warp_array(values.astype(np.float32)*use,source_affine,affine,shape_hw)
                warped=np.divide(warped,weight,out=np.zeros_like(warped),where=weight>1e-8)
                if channels:
                    destination[:,painted]=warped[:,painted]
                else:
                    destination[painted]=np.clip(np.rint(warped[:,painted].T),0,255).astype(np.uint8)
            paint(image.transpose(2,0,1),model);paint(rgb.transpose(2,0,1),original)
            paint(native_shape,probabilities,channels=True);paint(native_max,maxima,channels=True)
            observed|=painted
        classes[~observed]=255;labels[~observed]=0
        data[annual['year']]={'labels':labels,'classes':classes,'valid':observed,
            'rgb_original':original,'rgb_model':model,'shape_probabilities':probabilities,
            'max_probabilities':maxima,'node_ids':{i+1:n['id'] for i,n in enumerate(subset)}}
    return {'grid':GridSpec(base['crs'],tuple(affine)[:6],doc['units'],(resolution,resolution) if doc['units']=='metres' else None),
            'years':tuple(doc['years']),'shape_hw':shape_hw,'data':data}
