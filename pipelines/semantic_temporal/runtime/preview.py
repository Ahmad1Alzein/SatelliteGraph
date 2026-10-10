"""Shared event filtering and local paired grid export for the review page."""
from PIL import Image,ImageDraw
from .change_contracts import change_label
from .reports.changes import event_crops,_event_text,_text_block
from .reports.common import font
from shapely.geometry import shape


def major_change(event,document,native_pixel=1.):
    """Conservative display focus, separate from scientific candidate retention."""
    geometry=shape(event['geometry']);margin=max(1e-6,2*float(native_pixel))
    classes={event[k] for k in ('class_id','source_class','target_class') if k in event}
    def result(include,reason):return dict(include=include,reason=reason)
    if not geometry.area or geometry.buffer(-margin).area/geometry.area<.2:
        return result(False,'Thin boundary or small footprint')
    if event.get('kind')=='conversion' and {event['source_class'],event['target_class']}<={3,4,5}:
        return result(False,'Vegetation subclass difference; season may explain it')
    if 1 in classes:
        nodes={n['id']:n for n in document['nodes']}
        ids=[g for g in event.get('source_nodes',[])+event.get('target_nodes',[]) if g in nodes and nodes[g]['class_id']==1]
        for gid in ids:
            node=nodes[gid];area=shape(node['geometry']).area
            fraction=geometry.intersection(shape(node['geometry'])).area/max(area,1e-9)
            matches=[e for e in document.get('temporal_edges',[]) if gid in (e['source'],e['target'])
                and all(nodes.get(g,{}).get('class_id')==1 for g in (e['source'],e['target']))]
            # Annual segments partition each image. Several small counterpart
            # matches can together cover an unchanged building after a split.
            persistent=min(1.,sum(e.get('fraction_source',0) if gid==e['source'] else e.get('fraction_target',0) for e in matches))
            if fraction>=.6 and persistent<.15:return result(True,'Most of a building footprint changed; no substantial building match')
        return result(False,'Building outline, size variation or unresolved match')
    return result(True,'Substantial road or land-cover footprint')


def display_event(event,scores,*,external_predictions=False):
    """Attach independent model support for display; keep source evidence intact."""
    if not external_predictions or event.get('status') not in ('candidate','season_sensitive'):return event
    support=[scores[g,t] for g in event.get('source_nodes',[])+event.get('target_nodes',[])
        for t in event.get('related_tasks',[]) if (g,t) in scores and
        scores[g,t].get('status')=='inference' and scores[g,t].get('prediction')==1]
    if not support:return event
    return {**event,'image_evidence_status':event['status'],'model_supported':True}


def event_rows(events,document,scores,*,native_pixel=1.,external_predictions=False):
    nodes={n['id']:n for n in document['nodes']};rows=[]
    for event in events:
        event=display_event(event,scores,external_predictions=external_predictions)
        gids=event.get('source_nodes',[])+event.get('target_nodes',[])
        confidence=[nodes[g]['confidence'] for g in gids if g in nodes and 'confidence' in nodes[g]]
        probabilities=[float(scores[g,t]['probability']) for g in gids for t in event.get('related_tasks',[]) if (g,t) in scores]
        rows.append({'event_id':event['event_id'],'type':_event_text(event,{})[0][0],'status':event.get('status','candidate'),
            'classes':list({event[k] for k in ('class_id','source_class','target_class') if k in event}),
            'area':float(event.get('area',0)),'area_units':event.get('area_units','pixels squared'),
            'segment_confidence':min(confidence) if confidence else None,
            'gnn_score':max(probabilities) if probabilities else None,'event':event,
            'major_change':major_change(event,document,native_pixel)})
    return rows


def filter_rows(rows,*,statuses=None,classes=None,min_area=0,min_segment_confidence=0,min_gnn_score=None,major_only=False):
    return [r for r in rows if (statuses is None or r['status'] in statuses)
        and (not major_only or r.get('major_change',{}).get('include',False))
        and (classes is None or set(classes)&set(r['classes'])) and r['area']>=min_area
        and (min_segment_confidence<=0 or r['segment_confidence'] is not None and r['segment_confidence']>=min_segment_confidence)
        and (min_gnn_score is None or r['gnn_score'] is not None and r['gnn_score']>=min_gnn_score)]


def grid_image(frame,events,scores,*,columns=2,panel_size=(224,224)):
    if not events:raise ValueError('Select at least one change for grid export')
    if len(events)>24:raise ValueError('Grid export supports up to 24 changes per page')
    cell_width=panel_size[0]*2+24;plans=[]
    for event in events:
        texts,badge=_event_text(event,scores);lines=_text_block(texts,cell_width-12)
        plans.append((event,lines,badge,panel_size[1]+48+22*len(lines)))
    rows=[plans[i:i+columns] for i in range(0,len(plans),columns)]
    width=columns*(cell_width+16)+16;height=70+sum(max(p[3] for p in row)+16 for row in rows)
    image=Image.new('RGB',(width,height),'#f2f5f8');draw=ImageDraw.Draw(image);records=[];y=60
    draw.text((16,12),'Local change crops | yellow outline = change footprint',font=font(21),fill='#182331')
    for row in rows:
        for col,(event,lines,badge,ch) in enumerate(row):
            x=16+col*(cell_width+16)
            draw.rounded_rectangle((x,y,x+cell_width,y+ch),radius=8,fill='white')
            panels,record=event_crops(frame,event,panel_size)
            for side,panel in enumerate(panels):
                px=x+6+side*(panel_size[0]+12)
                draw.text((px,y+3),'Before' if side==0 else 'After',font=font(),fill='#182331')
                image.paste(panel,(px,y+27))
            for i,line in enumerate(lines):draw.text((x+6,y+panel_size[1]+34+i*22),line,font=font(),fill='#182331')
            record.update(badge=badge,labels=lines);records.append(record)
        y+=max(p[3] for p in row)+16
    return image,records
