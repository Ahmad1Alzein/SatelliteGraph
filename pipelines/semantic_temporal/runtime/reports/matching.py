"""Measured image controls, object correspondences, and separate class changes."""
from collections import defaultdict,Counter
from pathlib import Path
import csv
import hashlib
import numpy as np
from PIL import Image,ImageDraw
from shapely.geometry import shape,Polygon
from shapely.ops import unary_union
from ..storage import read_json,atomic_json,save_manifest
from ..registration import coverage_pairs,_world_points
from ..change_contracts import EDGE_LABELS
from .common import (frame_bounds,sample_rgb,display_rgb,paint_geometry,world_pixel,
    styled_line,compose,font,wrap_text,EDGE_STYLES)

COMPONENT_LABELS={'one_to_one':'One area matched to one area','split':'One area matched to several areas',
    'merge':'Several areas matched to one area','many_to_many':'Several areas matched across years',
    'unmatched':'No matching area'}
COMPONENT_COLORS={'one_to_one':'#009b79','split':'#e48017','merge':'#8a40c9',
    'many_to_many':'#c13a74','unmatched':'#6a7482'}


def select_links(links,lookup,bounds,limit=300):
    """Round-robin spatial bins, strongest first inside each bin."""
    bins=defaultdict(list);left,bottom,right,top=bounds
    for index,edge in enumerate(links):
        point=lookup[edge['source']]['position']
        key=(min(11,max(0,int(12*(point[0]-left)/max(right-left,1e-9)))),
             min(11,max(0,int(12*(point[1]-bottom)/max(top-bottom,1e-9)))))
        bins[key].append(index)
    for values in bins.values():values.sort(key=lambda i:(-float(links[i].get('iou',0)),
        -{'high':2,'medium':1,'low':0}.get(links[i].get('confidence'),0),i))
    selected=[];offset=0
    while len(selected)<limit:
        available=[values[offset] for _,values in sorted(bins.items()) if len(values)>offset]
        if not available:break
        selected.extend(available[:limit-len(selected)]);offset+=1
    return [links[i] for i in selected]


def _pair_body(frame,bounds,size=(600,400)):
    panels=[display_rgb(*sample_rgb(frame,year,bounds,size)) for year in frame['years']]
    header=30;gap=24;body=Image.new('RGB',(2*size[0]+gap,size[1]+header),'white');draw=ImageDraw.Draw(body)
    for side,(year,panel) in enumerate(zip(frame['years'],panels)):
        x=side*(size[0]+gap);body.paste(panel,(x,header));draw.text((x+4,3),str(year),font=font(18),fill='#182331')
    return body,panels,[(0,header),(size[0]+gap,header)]


def _object_view(frame,document,links,bounds,title,*,component_colors=None,edge_kind=None,size=(600,400)):
    body,panels,offsets=_pair_body(frame,bounds,size);lookup={n['id']:n for n in document['nodes']}
    selected=select_links(links,lookup,bounds);flip=bool(frame['grid'].crs) or frame['grid'].affine[4]<0
    # Draw footprints separately on their original-year RGB, then overlay cross-year arrows.
    for node in document['nodes']:
        side=frame['years'].index(node['year']);color=(component_colors or {}).get(node['id'],'#eeeeee')
        if shape(node['geometry']).intersects(Polygon.from_bounds(*bounds)):
            paint_geometry(panels[side],shape(node['geometry']),bounds,color,flip_y=flip,width=1)
    for panel,offset in zip(panels,offsets):body.paste(panel,offset)
    draw=ImageDraw.Draw(body);endpoints=[]
    for edge in selected:
        kind=edge_kind or ('t_primary_fwd' if edge.get('edge_role')=='primary' else 't_overlap_fwd')
        color,style=EDGE_STYLES[kind];points=[]
        for side,key in enumerate(('source','target')):
            point=world_pixel(*lookup[edge[key]]['position'],bounds,size,flip_y=flip)
            points.append(point)
        styled_line(draw,tuple(np.array(points[0])+offsets[0]),tuple(np.array(points[1])+offsets[1]),color,style,arrow=True)
        endpoints.append({'source_node':edge['source'],'target_node':edge['target'],
                          'source':list(points[0]),'target':list(points[1]),'edge_kind':EDGE_LABELS[kind]})
    legend=[f'{len(links):,} links total | {len(selected):,} displayed | {len(links)-len(selected):,} omitted from this view',
            'Arrows connect the same coordinate frame across the two original images.']
    kinds=[edge_kind] if edge_kind else ['t_primary_fwd','t_overlap_fwd']
    legend.extend((EDGE_LABELS[k],*EDGE_STYLES[k]) for k in kinds)
    if component_colors:legend.extend((COMPONENT_LABELS[k],v,None) for k,v in COMPONENT_COLORS.items())
    image,layout=compose(body,title,legend)
    return image,{'total_count':len(links),'displayed_count':len(selected),'omitted_count':len(links)-len(selected),
        'bounds':list(bounds),'panel_size':list(size),'panel_offsets':offsets,'endpoints':endpoints,**layout}


def render_matching_results(graph,context,frame,catalog):
    directory=context.output/'results';directory.mkdir(parents=True,exist_ok=True)
    document=read_json(graph.paths['graph']);lookup={n['id']:n for n in document['nodes']}
    bounds=frame_bounds(frame);paths={};index={};size=(600,400);flip=bool(frame['grid'].crs) or frame['grid'].affine[4]<0
    controls_path=context.output/'prepared/registration_controls.json'
    controls=read_json(controls_path) if controls_path.is_file() else {'method':'georeference_only','pairs':[]}
    body,_,offsets=_pair_body(frame,bounds,size);draw=ImageDraw.Draw(body)
    pairs=controls.get('pairs',[])
    # Controls live in the unregistered moving frame; display its fitted location on registered imagery.
    affine=controls.get('diagnostics',{}).get('affine',context.options.get('registration',{}).get('affine'))
    matrix=np.asarray(affine).reshape(2,3) if affine is not None else None
    chosen=sorted(pairs,key=lambda p:(not p.get('inlier',False),p.get('residual',0),p['id']))[:300]
    endpoints=[]
    for pair in chosen:
        moving=np.asarray(pair['moving_xy']);moving=moving@matrix[:,:2].T+matrix[:,2] if matrix is not None else moving
        a=world_pixel(*moving,bounds,size,flip_y=flip);b=world_pixel(*pair['reference_xy'],bounds,size,flip_y=flip)
        color='#009b79' if pair.get('inlier') else '#db5266'
        styled_line(draw,tuple(np.array(a)+offsets[0]),tuple(np.array(b)+offsets[1]),color,arrow=True)
        for point,offset in zip((a,b),offsets):
            x,y=np.array(point)+offset;draw.ellipse((x-3,y-3,x+3,y+3),fill=color,outline='white')
        endpoints.append({'id':pair['id'],'source':list(a),'target':list(b),'inlier':bool(pair.get('inlier'))})
    status='Feature registration was not performed; placement uses georeferencing.' if controls['method']=='georeference_only' else 'Measured feature controls; residuals are fitted, not independent accuracy.'
    image,layout=compose(body,'Image landmark matching',[
        status,f'{len(pairs):,} controls total | {len(chosen):,} displayed | {len(pairs)-len(chosen):,} omitted',
        ('Used in the fitted alignment','#009b79',None),('Rejected by the fitted alignment','#db5266',None)])
    paths['registration_matches']=directory/'registration_matches.png';image.save(paths['registration_matches'])
    index['registration']={'status':status,'method':controls['method'],'bounds':bounds,'panel_size':size,
        'panel_offsets':offsets,'endpoints':endpoints,'total_count':len(pairs),'displayed_count':len(chosen),**layout}
    links=document['temporal_edges'];components=document.get('primary_components',[])
    colors={gid:COMPONENT_COLORS[c['kind']] for c in components for gid in c['source_nodes']+c['target_nodes']}
    for name,title,edges,kwargs in (
        ('temporal_matches','Object matches across years',links,{}),
        ('temporal_match_types','How areas match across years',links,{'component_colors':colors}),
        ('class_change_matches','Class changes (separate from object matches)',catalog.get('derived_edges',[]),{'edge_kind':'change_conversion'})):
        image,record=_object_view(frame,document,edges,bounds,title,**kwargs)
        paths[name]=directory/(name+'.png');image.save(paths[name]);index['conversions' if name=='class_change_matches' else 'temporal' if name=='temporal_matches' else 'match_types']=record
    index['conversions']['edge_kind']='Class change';index['component_types']=dict(Counter(c['kind'] for c in components))
    matching_dir=directory/'matching';matching_dir.mkdir(exist_ok=True)
    paths['match_table']=matching_dir/'all_matches.csv'
    fields=sorted({key for edge in links for key in edge}) or ['source','target','edge_role']
    with paths['match_table'].open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(links)
    paths['all_controls']=matching_dir/'all_registration_controls.json';atomic_json(paths['all_controls'],controls)
    paths['all_components']=matching_dir/'all_components.json';atomic_json(paths['all_components'],components)
    index['coverage_pairs']=[]
    by_tile={tile.key:tile for year in context.years for tile in year.tiles}
    for a,b in coverage_pairs(context):
        footprints=[]
        for key in (a,b):
            tile=by_tile[key];h,w=tile.shape_hw
            footprints.append(Polygon(_world_points(tile,[(0,0),(w,0),(w,h),(0,h)],context)))
        coverage=footprints[0].intersection(footprints[1]);crop=coverage.bounds
        local=[e for e in links if coverage.covers(shape(lookup[e['source']]['geometry']).representative_point()) or coverage.covers(shape(lookup[e['target']]['geometry']).representative_point())]
        name='pair-'+hashlib.sha256((a+'\0'+b).encode()).hexdigest()[:12]
        image,record=_object_view(frame,document,local,crop,f'Coverage match | {Path(a).name} | {Path(b).name}')
        path=matching_dir/(name+'.png');image.save(path);paths[name]=path
        index['coverage_pairs'].append({'source':a,'target':b,'path':str(path.relative_to(context.output)),**record})
    index['component_galleries']=[]
    for kind in COMPONENT_LABELS:
        group=[c for c in components if c['kind']==kind];examples=group[:12]
        if not examples:continue
        cells=[]
        for component in examples:
            members=set(component['source_nodes']+component['target_nodes'])
            geometry=unary_union([shape(lookup[gid]['geometry']) for gid in members]);l,b,r,t=geometry.bounds
            padding=max((r-l)*.15,(t-b)*.15,2);crop=(l-padding,b-padding,r+padding,t+padding)
            local=[e for e in links if e['source'] in members and e['target'] in members]
            cell,_=_object_view(frame,document,local,crop,component['id'],component_colors=colors,size=(224,224));cells.append(cell)
        gallery=Image.new('RGB',(max(c.width for c in cells),sum(c.height for c in cells)), 'white');y=0
        for cell in cells:gallery.paste(cell,(0,y));y+=cell.height
        image,layout=compose(gallery,COMPONENT_LABELS[kind],[f'{len(group)} components total | {len(examples)} examples shown. Complete component table is retained.'])
        path=matching_dir/(kind+'_examples.png');image.save(path);paths['gallery:'+kind]=path
        index['component_galleries'].append({'kind':kind,'total_count':len(group),'displayed_count':len(examples),'path':str(path.relative_to(context.output)),**layout})
    paths['matching_index']=matching_dir/'index.json';atomic_json(paths['matching_index'],index)
    # Only files inside this stage directory belong to this manifest; main PNGs are bound by the render-stage manifest.
    save_manifest(matching_dir,{k:v for k,v in paths.items() if v.parent==matching_dir},stage='matching')
    return paths
