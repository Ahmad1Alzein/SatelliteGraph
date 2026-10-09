"""Independent, retained segmentation and spatial graph verification PNGs."""
from pathlib import Path
from collections import Counter
import numpy as np
import rasterio
from PIL import Image,ImageDraw
from shapely.geometry import shape
from ..contracts import GraphArtifact,GridSpec
from ..storage import read_json,atomic_json,load_segmentation,sha256,save_manifest
from ..adapters import read_band
from ..comparison import load_annual_frame
from ..change_contracts import CLASS_LABELS,EDGE_LABELS
from .common import (class_color,frame_bounds,sample_rgb,sample_classes,display_rgb,
    paint_geometry,styled_line,world_pixel,compose,horizontal_panels,EDGE_STYLES)


def _frame(artifact):
    with rasterio.open(artifact.paths.get('original_image',artifact.paths['image'])) as src:rgb=src.read().transpose(1,2,0)
    return {'grid':artifact.grid,'years':(artifact.year,),'shape_hw':artifact.shape_hw,
        'data':{artifact.year:{'rgb_original':rgb,'valid':read_band(artifact.paths['valid'])>0,
                               'classes':read_band(artifact.paths['classes'])}}}


def _size(frame,maximum=(600,600)):
    bounds=frame_bounds(frame);w=bounds[2]-bounds[0];h=bounds[3]-bounds[1]
    scale=min(maximum[0]/max(w,1e-9),maximum[1]/max(h,1e-9),1/max(abs(frame['grid'].affine[0]),1e-9))
    scale=max(scale,min(320/max(w,1e-9),180/max(h,1e-9)))
    return (max(1,round(w*scale)),max(1,round(h*scale)))


def _class_legend(frame,year):
    classes=frame['data'][year]['classes'];counts=Counter(classes.ravel().tolist())
    return [(f'{name}: {counts.get(klass,0):,} pixels',class_color(klass),None) for klass,name in CLASS_LABELS.items()]


def _inputs(context,artifacts):
    paths=set()
    for artifact in artifacts:
        paths.add(artifact.directory/'artifact.json');paths.update(artifact.paths.values())
    return [{'path':str(path.relative_to(context.output)) if path.is_relative_to(context.output) else str(path),
             'external':not path.is_relative_to(context.output),'sha256':sha256(path)} for path in sorted(paths) if path.is_file()]


def _graph(value):
    g=value['grid'];grid=GridSpec(g['crs'],tuple(g['affine']),g['units'],tuple(g['physical_pixel_size']) if g.get('physical_pixel_size') else None)
    return GraphArtifact(value['year'],value['method'],Path(value['directory']),grid,{k:Path(p) for k,p in value['paths'].items()})


def render_stage_results(context,stage):
    if stage not in ('segmentation','graph'):raise ValueError('Stage PNG rendering requires segmentation or graph')
    run_path=context.output/'run.json';run=read_json(run_path) if run_path.is_file() else {}
    selected=run.get('stages',{}).get(stage,{}).get('method')
    paths={};records=[];dependencies=[]
    if stage=='segmentation':
        chosen={}
        for path in sorted(context.output.rglob('artifact.json'),key=lambda p:(len(p.parts),str(p))):
            v=read_json(path)
            if 'shape_hw' not in v:continue
            artifact=load_segmentation(path)
            quality=read_json(artifact.paths['validation']) if 'validation' in artifact.paths else {}
            method=quality.get('method','new')
            if selected and method!=selected:continue
            chosen.setdefault((artifact.year,artifact.tile_key),artifact)
        for (year,key),artifact in sorted(chosen.items()):
            frame=_frame(artifact);bounds=frame_bounds(frame);size=_size(frame)
            rgb,valid=sample_rgb(frame,year,bounds,size);original=display_rgb(rgb,valid)
            classes=sample_classes(frame,year,bounds,size)
            colors=np.zeros_like(rgb)
            for klass in CLASS_LABELS:colors[classes==klass]=class_color(klass)
            semantic=rgb.copy();semantic[valid]=np.rint(.45*rgb[valid]+.55*colors[valid]).astype(np.uint8)
            boundaries=np.zeros(valid.shape,bool)
            labels=read_band(artifact.paths['labels'])
            from ..comparison import warp_array
            from affine import Affine
            from .common import view_affine
            visible=warp_array(labels,Affine(*artifact.grid.affine),view_affine(bounds,size,flip_y=bool(artifact.grid.crs) or artifact.grid.affine[4]<0),size[::-1],nearest=True)
            boundaries[1:]|=visible[1:]!=visible[:-1];boundaries[:,1:]|=visible[:,1:]!=visible[:,:-1]
            semantic[boundaries&valid]=[255,255,255]
            body=horizontal_panels([original,display_rgb(semantic,valid)],['Original image','Segments and boundaries'])
            image,layout=compose(body,f'Segmentation | {year} | {Path(key).name} | {artifact.grid.units}',
                ['Segment types',*_class_legend(frame,year)])
            path=context.output/'results/segmentation'/str(year)/Path(key).stem/'segmentation.png'
            path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
            paths[f'segmentation:{year}:{key}']=path;dependencies.append(artifact)
            records.append({'path':str(path.relative_to(context.output)),'year':year,'image':Path(key).stem,
                            'class_legend':list(CLASS_LABELS.values()),**layout})
    else:
        graph_paths=list((context.output/'graphs').rglob('artifact.json'))+list((context.output/'annual').rglob('artifact.json'))
        for path in sorted(graph_paths):
            artifact=_graph(read_json(path))
            if selected and artifact.method!=selected:continue
            document=read_json(artifact.paths['graph']);annual=bool(document.get('annual'))
            if annual:
                frame=load_annual_frame(artifact,context);key='annual'
                segmentation=[load_segmentation(read_json(Path(tile)/'graph.json')['segmentation_artifact']) for tile in document['source_tiles']]
            else:
                segmentation=[load_segmentation(document['segmentation_artifact'])]
                frame=_frame(segmentation[0]);key=Path(segmentation[0].tile_key).stem
            year=artifact.year;bounds=frame_bounds(frame);size=_size(frame,(1200,800))
            rgb,valid=sample_rgb(frame,year,bounds,size);image=display_rgb(rgb,valid)
            flip_y=bool(frame['grid'].crs) or frame['grid'].affine[4]<0
            for node in document['nodes']:
                paint_geometry(image,shape(node['geometry']),bounds,class_color(node['class_id']),alpha=42,flip_y=flip_y,width=1)
            draw=ImageDraw.Draw(image);lookup={n['id']:n for n in document['nodes']}
            for edge in document['edges']:
                a,b=lookup[edge['source']]['position'],lookup[edge['target']]['position']
                color,style=EDGE_STYLES[edge['edge_type']]
                styled_line(draw,world_pixel(*a,bounds,size,flip_y=flip_y),world_pixel(*b,bounds,size,flip_y=flip_y),color,style)
            for node in document['nodes']:
                x,y=world_pixel(*node['position'],bounds,size,flip_y=flip_y)
                draw.ellipse((x-3,y-3,x+3,y+3),fill=class_color(node['class_id']),outline='white',width=1)
            counts=Counter(e['edge_type'] for e in document['edges'])
            kinds=['touches','near','context_near'] if artifact.method=='new' else ['touches']
            legend=['Segment types',*_class_legend(frame,year),'Edge kinds',
                *[(f'{EDGE_LABELS[kind]}: {counts[kind]:,}',*EDGE_STYLES[kind]) for kind in kinds],
                f'{len(document["nodes"]):,} nodes | {len(document["edges"]):,} edges | distances in {artifact.grid.units}']
            canvas,layout=compose(image,f'{"Annual" if annual else "Image"} graph | {year} | {key} | method {artifact.method}',legend)
            destination=context.output/'results/graphs'/artifact.method/str(year)
            output=destination/'annual_graph.png' if annual else destination/key/'graph.png'
            output.parent.mkdir(parents=True,exist_ok=True);canvas.save(output)
            paths[f'graph:{artifact.method}:{year}:{key}']=output;dependencies.extend([artifact,*segmentation])
            records.append({'path':str(output.relative_to(context.output)),'year':year,'image':key,'annual':annual,
                'method':artifact.method,'node_count':len(document['nodes']),'edge_count':len(document['edges']),
                'class_legend':list(CLASS_LABELS.values()),'edge_legend':[EDGE_LABELS[k] for k in kinds],**layout})
    directory=context.output/('results/segmentation' if stage=='segmentation' else 'results/graphs')
    index=directory/'index.json';atomic_json(index,{'stage':stage,'images':records,'inputs':_inputs(context,dependencies)})
    paths['index']=index;save_manifest(directory,paths,stage=stage)
    return paths
