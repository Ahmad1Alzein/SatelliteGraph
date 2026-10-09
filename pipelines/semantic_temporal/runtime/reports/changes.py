"""All-class scores, complete transitions, and one paired-crop change report."""
from pathlib import Path
import math
import numpy as np
import pandas as pd
from PIL import Image,ImageDraw
from shapely.geometry import shape
from ..storage import read_json,atomic_json,save_manifest
from ..change_contracts import TASKS,KNOWN_CLASSES,CLASS_LABELS,change_label,status_label
from .common import (frame_bounds,fit_bounds,sample_rgb,display_rgb,paint_geometry,
                     compose,horizontal_panels,font,wrap_text,class_color)


def event_crops(frame,event,size=(224,224)):
    geometry=shape(event['geometry']);left,bottom,right,top=geometry.bounds
    padding=max((right-left)*.2,(top-bottom)*.2,abs(frame['grid'].affine[0])*4)
    bounds=fit_bounds((left-padding,bottom-padding,right+padding,top+padding),size)
    panels=[];flip=bool(frame['grid'].crs) or frame['grid'].affine[4]<0
    for year in frame['years']:
        image=display_rgb(*sample_rgb(frame,year,bounds,size))
        paint_geometry(image,geometry,bounds,'#ffdc18',alpha=25,flip_y=flip,width=2);panels.append(image)
    return panels,{'event_id':event['event_id'],'years':list(frame['years']),
        'before_bounds':list(bounds),'after_bounds':list(bounds),'panel_size':list(size)}


def _scores(result):
    rows={}
    for name in ('predictions','evaluation_predictions'):
        path=(result or {}).get(name)
        if path and Path(path).is_file():
            for row in pd.read_csv(path).to_dict('records'):
                if row.get('task') in TASKS and pd.notna(row.get('probability')):
                    rows[row['gid'],row['task']]=row
    return rows


def _task_rows(result,scores):
    heads=(result or {}).get('head_status',{});rows=[]
    for task in TASKS:
        head=heads.get(task,{});values=[float(r['probability']) for (gid,t),r in scores.items() if t==task]
        rows.append({'task':task,'available':bool(head.get('available',False)),
            'status':head.get('status','insufficient_data'),'score':max(values) if values else None,
            'positive_targets':head.get('positive_targets',head.get('positive',0)),
            'negative_targets':head.get('negative_targets',head.get('negative',0))})
    return rows


def _task_text(row):
    name,direction=row['task'].rsplit('_',1);klass=next(k for k,v in KNOWN_CLASSES.items() if v==name)
    state=status_label(row['status']) if row['available'] else 'Unavailable: not enough training examples'
    return f'{CLASS_LABELS[klass]} | {"Added / increased" if direction=="gain" else "Removed / decreased"}: {state}'


def _event_text(event,scores):
    title=change_label(event);descriptors=[]
    if event.get('status')=='predicted':
        title=f'{CLASS_LABELS[event["class_id"]]} | {"Added / increased" if event["kind"]=="gain" else "Removed / decreased"}'
    for task,kinds in event.get('descriptors',{}).items():
        name=task.rsplit('_',1)[0];klass=next((k for k,v in KNOWN_CLASSES.items() if v==name),event.get('class_id',0))
        descriptors.extend(change_label({'kind':kind,'class_id':klass}) for kind in kinds)
    if descriptors:title+=' | '+'; '.join(dict.fromkeys(descriptors))
    badge=status_label(event.get('status','candidate'))
    texts=[title,badge,f'{event.get("area",0):,.1f} {event.get("area_units","pixels squared")}']
    related=[]
    for task in event.get('related_tasks',[]):
        for gid in event.get('source_nodes',[])+event.get('target_nodes',[]):
            if (gid,task) in scores:related.append(scores[gid,task])
    if related:
        best=max(related,key=lambda r:r['probability']);held=best.get('probability_source')=='held_out'
        texts.append(f'GNN change score: {best["probability"]:.3f} ({"held out" if held else "trained on this area"})')
    else:texts.append('GNN change score: unavailable')
    if event.get('parent_event_ids'):texts.append('Part of a shape change; counted once here.')
    texts.append(event['event_id'])
    return texts,badge


def _text_block(lines,width,size=16):
    return [part for line in lines for part in wrap_text(line,width,size=size)]


def _predicted_events(document,catalog,scores):
    covered={(gid,task) for event in catalog.get('events',[]) for gid in event.get('source_nodes',[])+event.get('target_nodes',[]) for task in event.get('related_tasks',[])}
    nodes={n['id']:n for n in document['nodes']};events=[]
    for (gid,task),row in sorted(scores.items()):
        if gid not in nodes or (gid,task) in covered or int(row.get('prediction',-1))!=1:continue
        node=nodes[gid];direction=task.rsplit('_',1)[1]
        events.append({'event_id':'gnn:'+gid+':'+task,'kind':direction,'class_id':node['class_id'],
            'geometry':node['geometry'],'area':node.get('area',shape(node['geometry']).area),
            'area_units':catalog.get('transition_matrix',{}).get('area_units','pixels squared'),
            'status':'predicted','related_tasks':[task],'source_nodes':[gid] if direction=='loss' else [],
            'target_nodes':[gid] if direction=='gain' else []})
    return events


def _score_view(frame,document,scores,tasks,*,probabilities):
    bounds=frame_bounds(frame);size=(400,230);flip=bool(frame['grid'].crs) or frame['grid'].affine[4]<0
    rows=[]
    for klass,name in KNOWN_CLASSES.items():
        panels=[];captions=[]
        for direction,year in [('loss',frame['years'][0]),('gain',frame['years'][1])]:
            task=name+'_'+direction;status=next(r for r in tasks if r['task']==task)
            image=display_rgb(*sample_rgb(frame,year,bounds,size));count=0
            for node in document['nodes']:
                row=scores.get((node['id'],task))
                if node['year']!=year or node['class_id']!=klass or row is None:continue
                if not probabilities and int(row.get('prediction',-1))!=1:continue
                score=float(row['probability']);color=(round(255*score),60,round(255*(1-score))) if probabilities else '#ffda18'
                paint_geometry(image,shape(node['geometry']),bounds,color,alpha=100,flip_y=flip,width=2);count+=1
            captions.append(f'{CLASS_LABELS[klass]} | {year} | {"Removed / decreased" if direction=="loss" else "Added / increased"} | {str(count)+" scored areas" if status["available"] else "Unavailable"}')
            panels.append(image)
        rows.append(horizontal_panels(panels,captions))
    body=Image.new('RGB',(max(r.width for r in rows),sum(r.height for r in rows)+12*(len(rows)-1)),'white');y=0
    for row in rows:body.paste(row,(0,y));y+=row.height+12
    legend=['GNN change scores are model outputs, not accuracy. Held-out scores are used when available.',
        'Unavailable tasks have no score; they are not zero. All eight known classes are listed.',
        'Yellow outlines: predicted gain or loss.' if not probabilities else 'Blue: lower score. Red: higher score. Original RGB stays underneath.',
        *[_task_text(r) for r in tasks]]
    return compose(body,'All-class GNN '+('change scores' if probabilities else 'predictions'),legend)[0]


def _transition_view(catalog,years):
    matrix=np.asarray(catalog.get('transition_matrix',{}).get('pixels',np.zeros((8,8))),dtype=np.int64)
    if matrix.shape!=(8,8):raise ValueError('Transition matrix must contain all eight classes')
    labels=list(KNOWN_CLASSES);cell=100;left=180;top=110
    image=Image.new('RGB',(left+8*cell+20,top+8*55+20),'white');draw=ImageDraw.Draw(image);maximum=max(1,int(matrix.max()))
    for j,klass in enumerate(labels):
        for line,text in enumerate(wrap_text(CLASS_LABELS[klass],cell-6,size=14)):
            draw.text((left+j*cell+3,top-65+line*18),text,font=font(14),fill='#182331')
    for i,klass in enumerate(labels):
        draw.text((5,top+i*55+16),CLASS_LABELS[klass],font=font(),fill='#182331')
        for j in range(8):
            ratio=math.sqrt(float(matrix[i,j])/maximum);color=(round(245-100*ratio),round(248-70*ratio),250)
            x=left+j*cell;y=top+i*55;draw.rectangle((x,y,x+cell,y+55),fill=color,outline='#c8d0d8')
            draw.text((x+8,y+16),f'{matrix[i,j]:,}',font=font(),fill='#182331')
    coverage=catalog.get('coverage',{})
    canvas,_=compose(image,f'Class transitions | rows: {years[0]} | columns: {years[1]}',[
        'Values are owned common-grid pixels; diagonal cells keep the same class.',
        f'Unknown class pixels: {coverage.get("unknown_pixels",0):,} | Excluded empty pixels: {coverage.get("unobserved_pixels",0):,}'])
    return canvas,{'classes':[CLASS_LABELS[k] for k in labels],'pixels':matrix.tolist()}


def render_change_results(graph,context,frame,catalog,gnn_result):
    document=read_json(graph.paths['graph']);scores=_scores(gnn_result);tasks=_task_rows(gnn_result,scores)
    events=catalog.get('events',[]);predicted=_predicted_events(document,catalog,scores)
    if len({e['event_id'] for e in events})!=len(events):raise ValueError('Duplicate canonical event IDs')
    # Plan the complete sheet before allocating it. No event is silently truncated.
    width=960;cell_width=460;row_plans=[];cell_specs=[]
    for event in events+predicted:
        texts,badge=_event_text(event,scores);lines=_text_block(texts,cell_width-12)
        cell_specs.append((event,lines,badge,224+32+len(lines)*22+16))
    for i in range(0,len(cell_specs),2):row_plans.append(cell_specs[i:i+2])
    summary=[f'{len(events):,} image-evidence candidates | {len(predicted):,} additional GNN predictions',
        'Yellow highlights show the same footprint on both original images. Paired crops use the same location and scale.',
        'Image evidence and GNN predictions need review; these are not independently verified ground truth.',
        *[_task_text(r) for r in tasks]]
    summary_lines=_text_block(summary,width-40);header_height=60+len(summary_lines)*23
    structures=catalog.get('structural_observations',[]);compound=catalog.get('compound_events',[])
    structural_lines=_text_block(['Matching and neighborhood observations (separate from physical changes)',
        *[f'{change_label(item)} | {item.get("id",item.get("event_id",""))} | {item.get("status","Needs review")}' for item in structures],
        *[f'{change_label(item)}: {", ".join(item.get("event_ids",[]))} (already shown above)' for item in compound]],width-40)
    height=header_height+sum(max(c[3] for c in row)+20 for row in row_plans)+len(structural_lines)*23+50
    if not cell_specs:height+=60
    required=width*height;budget=context.options.get('max_report_pixels',128_000_000)
    if isinstance(budget,bool) or not isinstance(budget,int) or budget<=0:raise ValueError('max_report_pixels must be a positive integer')
    if required>budget:raise ValueError(f'Complete change report required {required:,} pixels ({width} x {height}); budget is {budget:,}. Increase --max-report-pixels and rerender; scientific results are retained.')
    directory=context.output/'results';directory.mkdir(parents=True,exist_ok=True);paths={}
    sheet=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(sheet)
    draw.text((20,14),f'Change report | {frame["years"][0]} to {frame["years"][1]}',font=font(22),fill='#182331')
    for i,line in enumerate(summary_lines):draw.text((20,50+i*23),line,font=font(),fill='#182331')
    y=header_height;crops=[];predicted_crops=[]
    for row in row_plans:
        for col,(event,lines,badge,cell_height) in enumerate(row):
            x=16+col*480;panels,record=event_crops(frame,event)
            for side,(panel,year) in enumerate(zip(panels,frame['years'])):
                px=x+side*236;draw.text((px,y+2),str(year),font=font(18),fill='#182331');sheet.paste(panel,(px,y+28))
            for i,line in enumerate(lines):draw.text((x,y+258+i*22),line,font=font(),fill='#182331')
            record.update(badge=badge,kind=event['kind'],sheet_bounds=[x,y,x+cell_width,y+cell_height],labels=lines)
            (predicted_crops if event['status']=='predicted' else crops).append(record)
        y+=max(c[3] for c in row)+20
    if not cell_specs:draw.text((20,y+12),'No reportable physical change candidates or GNN predictions.',font=font(18),fill='#182331');y+=60
    for i,line in enumerate(structural_lines):draw.text((20,y+12+i*23),line,font=font(),fill='#182331')
    paths['change_report']=directory/'change_report.png';sheet.save(paths['change_report'])
    for name,probabilities in [('gnn_predictions',False),('gnn_probabilities',True)]:
        paths[name]=directory/(name+'.png');_score_view(frame,document,scores,tasks,probabilities=probabilities).save(paths[name])
    transition,transition_record=_transition_view(catalog,frame['years']);paths['class_transitions']=directory/'class_transitions.png';transition.save(paths['class_transitions'])
    # Overview PNGs combine the independently retained stage products without removing them.
    for name,stage in [('segmentation_overview','segmentation'),('spatial_graph','graphs')]:
        index_path=directory/stage/'index.json'
        if not index_path.is_file():continue
        records=read_json(index_path).get('images',[])
        chosen=[r for r in records if r.get('annual')] if stage=='graphs' else records
        if not chosen:chosen=records
        thumbnails=[]
        for record in chosen:
            with Image.open(context.output/record['path']) as src:
                image=src.convert('RGB');image.thumbnail((960,700));thumbnails.append(image.copy())
        if thumbnails:
            body=Image.new('RGB',(max(p.width for p in thumbnails),sum(p.height for p in thumbnails)+16*len(thumbnails)),'white');offset=0
            for image in thumbnails:body.paste(image,(0,offset));offset+=image.height+16
            paths[name]=directory/(name+'.png');body.save(paths[name])
    report_dir=directory/'changes';report_dir.mkdir(exist_ok=True)
    paths['change_index']=report_dir/'index.json';atomic_json(paths['change_index'],{
        'years':frame['years'],'tasks':tasks,'crops':crops,'predicted_crops':predicted_crops,
        'compound_groups':compound,'structural_observations':structures,'transition_matrix':transition_record,
        'required_pixels':required,'canvas_size':[width,height],
        'images':{name:str(path.relative_to(context.output)) for name,path in paths.items() if path.suffix=='.png'}})
    save_manifest(report_dir,{'index':paths['change_index']},stage='changes')
    return paths
