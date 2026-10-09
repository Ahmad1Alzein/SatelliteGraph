"""Canonical physical candidates, structural observations, and masked task targets."""
import hashlib
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import ndimage
from affine import Affine
from rasterio.features import shapes
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from .change_contracts import KNOWN_CLASSES,TASKS,CONVERSIONS,CHANGE_SCHEMA,change_label
from .storage import read_json

SEASONAL={4,5,6}
FOUR=ndimage.generate_binary_structure(2,1)


def _pieces(mask,affine,minimum,pixel_area):
    labels,_=ndimage.label(mask,structure=FOUR)
    counts=np.bincount(labels.ravel())
    for ident,window in enumerate(ndimage.find_objects(labels),1):
        if window is None or counts[ident]*pixel_area<minimum: continue
        piece=labels[window]==ident
        local=affine*Affine.translation(window[1].start,window[0].start)
        geometry=unary_union([shape(g) for g,v in shapes(piece.astype(np.uint8),mask=piece,transform=local) if v])
        yield window,piece,geometry


def _same_class_links(document):
    nodes={n['id']:n for n in document['nodes']};links={gid:set() for gid in nodes}
    for edge in document['temporal_edges']:
        a,b=edge['source'],edge['target']
        if edge.get('edge_role')=='primary' and nodes[a]['class_id']==nodes[b]['class_id']:
            links[a].add(b);links[b].add(a)
    return links


def _counts(labels,mask,size):
    return np.bincount(labels[mask].ravel(),minlength=size)


def _largest_pieces(labels,mask,size,pixel_area):
    components,_=ndimage.label(mask,structure=FOUR)
    pairs=components[mask].astype(np.int64)*size+labels[mask]
    unique,counts=np.unique(pairs,return_counts=True)
    maxima=np.zeros(size,float)
    np.maximum.at(maxima,unique%size,counts*pixel_area)
    return maxima


def _reviewed(path,targets):
    if path is None: return
    table=pd.read_csv(path,dtype={'gid':str,'task':str,'source':str},keep_default_na=False)
    required={'gid','task','label','confidence','source'}
    if not required<=set(table): raise ValueError('Reviewed targets require gid,task,label,confidence,source')
    if table.duplicated(['gid','task']).any(): raise ValueError('Duplicate/conflicting reviewed target rows')
    lookup={(r['gid'],r['task']):r for r in targets}
    for row in table.to_dict('records'):
        key=(row['gid'],row['task'])
        if key not in lookup: raise ValueError(f'Unknown node/task in reviewed targets: {key}')
        target=lookup[key]
        try: label=float(row['label']);confidence=float(row['confidence'])
        except (ValueError,TypeError) as exc: raise ValueError('Invalid reviewed target label/confidence') from exc
        if label not in (0,1) or not math.isfinite(confidence) or not 0<=confidence<=1 or not str(row['source']).strip():
            raise ValueError('Invalid reviewed target label/confidence/source')
        if not target['eligible']:
            raise ValueError(f'Review cannot override unknown semantics, coverage, or wrong class/year: {key} ({target["reason"]})')
        target.update(label=int(label),confidence=confidence,source=str(row['source']).strip(),reason='reviewed')


def build_change_catalog(graph,frame,evidence,policy,*,reviewed_targets=None):
    document=read_json(graph.paths['graph']);nodes=document['nodes'];by_id={n['id']:n for n in nodes}
    first,last=frame['years'];area=evidence['pixel_area'];affine=Affine(*frame['grid'].affine)
    same=_same_class_links(document);task_stats={};targets=[];events=[]
    # Per-label aggregation is linear in pixels, rather than one full raster scan per node.
    for year,direction in ((first,'loss'),(last,'gain')):
        data=frame['data'][year];other=frame['data'][last if year==first else first]
        size=max(data['node_ids'],default=0)+1
        counterpart=np.take_along_axis(other['shape_probabilities'],np.minimum(other['classes'],8)[None],axis=0)[0]
        known=evidence['known']&(counterpart>=policy['reliable_confidence'])
        for klass,name in KNOWN_CLASSES.items():
            labels=evidence['roof_labels_earlier'] if year==first and klass==1 else data['labels']
            eligible=evidence['roof_eligible'] if klass==1 else evidence['eligible']&~evidence['roof_disagreement']
            mask=evidence['masks'][name+'_'+direction]
            raw=evidence['raw_masks'][name+'_'+direction]
            task_stats[name+'_'+direction]={
                'total':np.bincount(labels.ravel(),minlength=size),
                'observed':_counts(labels,evidence['common_valid'],size),
                'known':_counts(labels,known,size),'clear':_counts(labels,eligible,size),
                'changed':_counts(labels,mask,size)*area,
                'stable':_counts(labels,evidence['stable'][name],size),
                'largest':_largest_pieces(labels,raw,size,area),
                'label_by_gid':{gid:i for i,gid in data['node_ids'].items()}}
    for node in nodes:
        klass=node['class_id'];direction='gain' if node['year']==last else 'loss'
        applicable=f'{KNOWN_CLASSES[klass]}_{direction}' if klass in KNOWN_CLASSES else None
        reason='unknown_semantics';eligible=False;label=None
        if applicable:
            stats=task_stats[applicable];ident=stats['label_by_gid'][node['id']]
            expected=max(node['area']/area,1.);denom=max(stats['total'][ident],expected)
            observed=stats['observed'][ident]/denom;known=stats['known'][ident]/denom
            clear=stats['clear'][ident]/denom
            reason=('coverage_limited' if observed<policy['observed_fraction'] else
                    'unknown_counterpart' if known<policy['known_fraction'] else
                    'shadow_or_alignment' if clear<policy['observed_fraction'] else
                    'low_confidence' if node['confidence']<policy['reliable_confidence'] else
                    'physical_scale_unavailable' if frame['grid'].units!='metres' else 'no_reliable_change_or_stability')
            eligible=reason=='no_reliable_change_or_stability'
            changed=stats['changed'][ident];largest=stats['largest'][ident]
            if eligible:
                positive=(largest>=max(policy['building_reliable_area_m2'],policy['minimum_area_m2']['building']) and changed>0 if klass==1 else
                          largest>=policy['minimum_area_m2'][KNOWN_CLASSES[klass]] and changed>=policy['positive_fraction']*denom*area)
                if positive and klass not in SEASONAL: label=1;reason='reliable_image_change'
                elif positive: reason='season_sensitive'
                elif (same[node['id']] and stats['stable'][ident]/denom>=policy['stable_fraction']
                      and changed==0): label=0;reason='supported_stability'
        for task in TASKS:
            selected=task==applicable
            targets.append({'gid':node['id'],'task':task,'label':label if selected else None,
                'confidence':node['confidence'],'source':'semantic_image_evidence_rule',
                'year':node['year'],'eligible':bool(eligible and selected),
                'reason':reason if selected or applicable is None else 'different_class_or_year'})
    _reviewed(reviewed_targets,targets)
    target_lookup={(r['gid'],r['task']):r for r in targets}
    # Conversion geometry uses roof-aligned buildings; other classes remain in the ground frame.
    source_classes=frame['data'][first]['classes'].copy()
    old_labels=frame['data'][first]['labels'].copy()
    building_ids=[i for i,gid in frame['data'][first]['node_ids'].items() if by_id[gid]['class_id']==1]
    aligned_building=np.isin(evidence['roof_labels_earlier'],building_ids)
    if building_ids:
        displaced=(source_classes==1)&~aligned_building
        source_classes[displaced]=0
        source_classes[aligned_building]=1
        old_labels[aligned_building]=evidence['roof_labels_earlier'][aligned_building]
    target_classes=frame['data'][last]['classes'];new_labels=frame['data'][last]['labels']
    for source_class,target_class in CONVERSIONS:
        source_task=KNOWN_CLASSES[source_class]+'_loss';target_task=KNOWN_CLASSES[target_class]+'_gain'
        selected=(source_classes==source_class)&(target_classes==target_class)
        selected&=(evidence['candidate_masks'][source_task]|evidence['candidate_masks'][target_task]|
                   evidence['masks'][source_task]|evidence['masks'][target_task])
        minimum=min(policy['minimum_area_m2'][KNOWN_CLASSES[source_class]],policy['minimum_area_m2'][KNOWN_CLASSES[target_class]])
        for window,piece,geometry in _pieces(selected,affine,minimum,area):
            source_ids=sorted(frame['data'][first]['node_ids'][int(i)] for i in np.unique(old_labels[window][piece])
                              if i and by_id[frame['data'][first]['node_ids'][int(i)]]['class_id']==source_class)
            target_ids=sorted(frame['data'][last]['node_ids'][int(i)] for i in np.unique(new_labels[window][piece])
                              if i and by_id[frame['data'][last]['node_ids'][int(i)]]['class_id']==target_class)
            descriptors={source_task:['contraction' if any(same[i] for i in source_ids) else 'disappearance'],
                         target_task:['expansion' if any(same[i] for i in target_ids) else 'appearance']}
            supporting=[target_lookup[gid,task] for task,gids in [(source_task,source_ids),(target_task,target_ids)] for gid in gids]
            seasonal=source_class in SEASONAL and target_class in SEASONAL
            reviewed_positive=any(r['label']==1 and r['reason']=='reviewed' and r['confidence']>=policy['reliable_confidence'] for r in supporting)
            status='reliable' if reviewed_positive else 'season_sensitive' if seasonal else 'reliable' if any(r['label']==1 for r in supporting) else 'candidate'
            ident=hashlib.sha256(f'{first}:{last}:{source_class}:{target_class}:'.encode()+geometry.wkb).hexdigest()[:16]
            event={'event_id':'change-'+ident,'kind':'conversion','source_class':source_class,'target_class':target_class,
                'class_id':target_class,'geometry':mapping(geometry),'area':float(piece.sum()*area),
                'area_units':'metres squared' if frame['grid'].units=='metres' else 'pixels squared',
                'source_year':first,'target_year':last,'source_nodes':source_ids,'target_nodes':target_ids,
                'related_tasks':[source_task,target_task],'descriptors':descriptors,'status':status,
                'source':'semantic_image_evidence_rule','season_sensitive':seasonal,
                'season_resolved_by_review':bool(seasonal and reviewed_positive),
                'lineage':{'source_nodes':source_ids,'target_nodes':target_ids},
                'reliable_area':float((evidence['raw_masks'][source_task][window]&evidence['raw_masks'][target_task][window]&piece).sum()*area)}
            event['display_name']=change_label(event);events.append(event)
    events.sort(key=lambda e:(e['source_class'],e['target_class'],e['event_id']))
    node_events={gid:[] for gid in by_id}
    for e in events:
        for gid in e['source_nodes']+e['target_nodes']: node_events[gid].append(e)
    compound=[];seen_pairs=set()
    for gid,linked in same.items():
        if by_id[gid]['year']!=first: continue
        klass=by_id[gid]['class_id']
        if klass not in KNOWN_CLASSES: continue
        loss_task=KNOWN_CLASSES[klass]+'_loss';gain_task=KNOWN_CLASSES[klass]+'_gain'
        loss=[e for e in node_events[gid] if loss_task in e['related_tasks']]
        gain=[e for other in linked for e in node_events[other] if gain_task in e['related_tasks']]
        if not loss or not gain: continue
        event_ids=tuple(sorted({e['event_id'] for e in loss+gain}))
        if event_ids in seen_pairs: continue
        seen_pairs.add(event_ids)
        ident='shape-'+hashlib.sha256('|'.join(event_ids).encode()).hexdigest()[:16]
        compound.append({'event_id':ident,'kind':'reconfiguration','class_id':klass,'event_ids':list(event_ids),
                         'source_nodes':[gid],'target_nodes':sorted(linked),
                         'geometry':mapping(unary_union([shape(e['geometry']) for e in loss+gain]))})
        for event in loss+gain:
            event.setdefault('parent_event_ids',[]).append(ident)
            task=loss_task if event in loss else gain_task
            event['descriptors'][task].append('reconfiguration')
    structural=[]
    for component in document.get('primary_components',[]):
        if component['kind'] not in ('split','merge','many_to_many'): continue
        members=component['source_nodes']+component['target_nodes']
        structural.append({**component,'status':'matching_difference',
            'geometry':mapping(unary_union([shape(by_id[g]['geometry']) for g in members]))})
    for klass,name in KNOWN_CLASSES.items():
        counts=[]
        for year in (first,last):
            counts.append(ndimage.label((frame['data'][year]['classes']==klass)&evidence['eligible'],structure=FOUR)[1])
        if counts[0]!=counts[1]:
            physical=bool(evidence['masks'][name+'_gain'].any() or evidence['masks'][name+'_loss'].any())
            structural.append({'id':f'connectivity-{klass}','kind':'connectivity','class_id':klass,
                'status':'context_observation','earlier_components':counts[0],'later_components':counts[1],
                'footprint_evidence':physical,'source_nodes':[],'target_nodes':[]})
    # Neighborhood comparisons describe graph context only, never training positives.
    spatial=document.get('spatial_edges',[])
    neighbors={gid:[] for gid in by_id}
    for e in spatial:
        neighbors[e['source']].append(by_id[e['target']]['class_id']);neighbors[e['target']].append(by_id[e['source']]['class_id'])
    for gid,linked in same.items():
        if by_id[gid]['year']!=first or len(linked)!=1:continue
        other=next(iter(linked))
        if sorted(neighbors[gid])!=sorted(neighbors[other]):
            structural.append({'id':f'neighborhood-{gid}','kind':'neighborhood','status':'context_observation',
                'source_nodes':[gid],'target_nodes':[other],'earlier_neighbor_classes':sorted(neighbors[gid]),
                'later_neighbor_classes':sorted(neighbors[other])})
    node_changes=[]
    for node in nodes:
        klass=node['class_id'];direction='gain' if node['year']==last else 'loss'
        task=KNOWN_CLASSES[klass]+'_'+direction if klass in KNOWN_CLASSES else None
        target=target_lookup.get((node['id'],task));related=node_events[node['id']]
        reason=target['reason'] if target else 'unknown_semantics'
        event=('persistent' if target and target['label']==0 else 'unknown_observation' if reason in ('unknown_semantics','unknown_counterpart')
               else 'coverage_limited' if reason=='coverage_limited' else 'not_assessed')
        if related and target and target['eligible']:
            if klass==1:
                event='demolished' if direction=='loss' else 'extension' if any('expansion' in e['descriptors'].get(task,[]) for e in related) else 'new_building'
            else:event=task
        node_changes.append({'gid':node['id'],'year':node['year'],'class_id':klass,'task':task,
            'event':event,'event_ids':[e['event_id'] for e in related],
            'label':target['label'] if target else None,'eligible':target['eligible'] if target else False,
            'reason':reason,'confidence':node['confidence'],'source':target['source'] if target else 'unassessed',
            'status':'unassessed' if not target or not target['eligible'] else 'reliable' if target['label']==1 else 'candidate' if related else 'persistent' if target['label']==0 else 'uncertain'})
    transition=evidence['transition'];matrix=np.zeros((8,8),np.int64)
    for a,b in ((a,b) for a in range(1,9) for b in range(1,9)):matrix[a-1,b-1]=(transition==16*a+b).sum()
    derived=[]
    for event in events:
        for source in event['source_nodes']:
            for target in event['target_nodes']:
                derived.append({'source':source,'target':target,'edge_type':'change_conversion','event_id':event['event_id']})
    return {'schema':CHANGE_SCHEMA,'status':'complete','engine':'all_class_image_evidence','years':[first,last],
        'units':frame['grid'].units,'events':events,'compound_events':compound,'derived_edges':derived,
        'structural_observations':structural,'node_changes':node_changes,'targets':targets,
        'transition_matrix':{'classes':list(KNOWN_CLASSES.values()),'pixels':matrix.tolist(),'area':(matrix*area).tolist(),
            'area_units':'metres squared' if frame['grid'].units=='metres' else 'pixels squared'},
        'coverage':{'unknown_pixels':int((transition==0).sum()),'unobserved_pixels':int((transition==65535).sum()),
                    'known_pixels':int(((transition>0)&(transition<65535)).sum()),
                    'vegetation_gain_area':float(evidence['vegetation_gain'].sum()*area),
                    'vegetation_loss_area':float(evidence['vegetation_loss'].sum()*area)},
        'policy':policy,'target_definition':'Per-class gain/loss pseudo labels; agreement is not independently verified accuracy'}
