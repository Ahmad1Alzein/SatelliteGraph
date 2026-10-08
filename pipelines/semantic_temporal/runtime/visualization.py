"""Validation and previews sized from actual imagery and coordinates."""
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import transform_geom
from PIL import Image,ImageDraw
from shapely.geometry import shape
from scipy import ndimage
from .storage import atomic_json,read_json,verify_manifest
from .contracts import CLASSES
from .adapters import read_band

COLORS=np.array([[130,130,130],[230,57,70],[90,90,100],[27,120,55],[150,200,90],[238,200,70],
                 [196,150,100],[170,140,200],[40,120,230]],np.uint8)


def validate_run(context,stage):
    artifacts=[]
    for path in sorted(context.output.rglob('artifact.json')):
        document=read_json(path);directory=path.parent
        if (directory/'manifest.json').exists(): verify_manifest(directory)
        paths=document['paths']
        if 'shape_hw' in document:
            labels=read_band(paths['labels']);valid=read_band(paths['valid'])>0;classes=read_band(paths['classes'])
            if not np.array_equal(labels>0,valid): raise ValueError(f'Segmentation coverage gap: {directory}')
            if np.any(classes[~valid]!=255) or np.any(classes[valid]>8): raise ValueError('Invalid semantic/nodata classes')
            quality=read_json(paths['validation'])
            for ident in np.unique(labels[valid]):
                if quality.get('method')!='old' and ndimage.label(labels==ident)[1]!=1: raise ValueError(f'Disconnected region {ident}: {directory}')
            probability=np.load(paths['probabilities'],allow_pickle=False)
            if probability.shape!=(9,*labels.shape) or not np.isfinite(probability).all(): raise ValueError('Invalid probabilities')
            if any(str(int(i)) not in document['lineage'] for i in np.unique(labels[valid])): raise ValueError('Missing segment lineage')
        elif 'graph' in paths:
            graph=read_json(paths['graph']);nodes=graph['nodes'];ids={n['id'] for n in nodes}
            if len(ids)!=len(nodes): raise ValueError('Duplicate graph node IDs')
            for edge in graph.get('edges',graph.get('temporal_edges',[])):
                if edge['source'] not in ids or edge['target'] not in ids: raise ValueError('Missing graph edge endpoint')
            if any(not shape(n['geometry']).is_valid or n['area']<=0 for n in nodes): raise ValueError('Invalid graph geometry')
            with np.load(paths['tensors'],allow_pickle=False) as tensors:
                if not np.isfinite(tensors['x']).all() or not np.isfinite(tensors['edge_attr']).all(): raise ValueError('Nonfinite graph features')
        artifacts.append(str(path.relative_to(context.output)))
    if not artifacts: raise ValueError('No completed scientific artifacts to validate')
    result={'status':'pass','stage':stage,'artifacts':artifacts,'audit_status':'not_provided'}
    atomic_json(context.output/'validation.json',result);return result


def _title(image,title):
    canvas=Image.new('RGB',(image.width,max(60,image.height+28)),'white');canvas.paste(image,(0,28))
    ImageDraw.Draw(canvas).text((8,8),title,fill='black');return canvas


def _polygon(draw,polygon,to_pixel,fill,outline):
    parts=[polygon] if polygon.geom_type=='Polygon' else list(polygon.geoms)
    for part in parts:
        draw.polygon([to_pixel(x,y) for x,y in part.exterior.coords],fill=fill,outline=outline)
        for ring in part.interiors: draw.polygon([to_pixel(x,y) for x,y in ring.coords],fill='white',outline=outline)


def render_outputs(context,stage):
    directory=context.output/'previews';directory.mkdir(exist_ok=True);paths={};years=set();units=set()
    for number,path in enumerate(sorted(context.output.rglob('artifact.json'))):
        artifact=read_json(path);grid=artifact.get('grid',{});units.add(grid.get('units',''))
        if 'shape_hw' in artifact:
            year=artifact['year'];years.add(year);classes=read_band(artifact['paths']['classes']);valid=read_band(artifact['paths']['valid'])>0
            color=COLORS[np.minimum(classes,8)];color[~valid]=[245,245,245]
            image=Image.fromarray(color);image.thumbnail((1200,800))
            title=f'{year} | {artifact["tile_key"]} | {grid["units"]} | invalid pixels shown in white'
        elif 'graph' in artifact['paths']:
            graph=read_json(artifact['paths']['graph']);nodes=graph['nodes'];years.update(graph.get('years',[graph.get('year')]))
            if not nodes: continue
            units.add(graph['units']);polygons=[shape(n['geometry']) for n in nodes]
            lo=np.min([p.bounds[:2] for p in polygons],0);hi=np.max([p.bounds[2:] for p in polygons],0);extent=np.maximum(hi-lo,1e-9)
            scale=min(1200/extent[0],800/extent[1]);width=max(120,round(extent[0]*scale));height=max(50,round(extent[1]*scale))
            panels=graph.get('years',[graph.get('year')])
            image=Image.new('RGB',(width*len(panels),height+24),'white')
            for panel,year in enumerate(panels):
                draw=ImageDraw.Draw(image)
                draw.text((panel*width+8,4),str(year),fill='black')
                to_pixel=lambda x,y:((x-lo[0])*scale+panel*width,(hi[1]-y)*scale+24)
                for node,polygon in zip(nodes,polygons):
                    if node['year']==year: _polygon(draw,polygon,to_pixel,tuple(COLORS[node['class_id']]),'gray')
                lookup={n['id']:n for n in nodes if n['year']==year}
                for edge in graph.get('edges',[]):
                    if edge['source'] in lookup and edge['target'] in lookup:
                        draw.line([to_pixel(*lookup[edge['source']]['position']),to_pixel(*lookup[edge['target']]['position'])],fill='black',width=1)
            title=f'{graph.get("years",graph.get("year"))} | {graph.get("method")} graph | {graph["units"]}'
            if grid.get('crs'):
                features=[{'type':'Feature','geometry':transform_geom(grid['crs'],'EPSG:4326',n['geometry']),
                    'properties':{'id':n['id'],'year':n['year'],'class':n['class'],'area':n['area'],'area_units':graph['units']+' squared'}} for n in nodes]
                atomic_json(path.parent/'nodes.geojson',{'type':'FeatureCollection','features':features})
        else: continue
        preview=directory/f'{number:04d}.png';_title(image,title).save(preview);paths[str(number)]=preview
    index={'stage':stage,'years':sorted(y for y in years if y is not None),'units':sorted(u for u in units if u),
           'classes':CLASSES,'previews':{k:str(v.relative_to(context.output)) for k,v in paths.items()}}
    atomic_json(context.output/'preview_index.json',index);return paths
