"""Relational GNN training with grouped, isolated evaluation and reusable state."""
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from affine import Affine
from shapely.geometry import Polygon,shape
from .storage import atomic_json,read_json,sha256,save_manifest
from .registration import prepare_tile
from .gnn_model import RGNN


def evaluation_groups(context):
    tiles=[t for y in context.years for t in y.tiles];parent={t.key:t.key for t in tiles};polygons={}
    def find(key):
        while parent[key]!=key: parent[key]=parent[parent[key]];key=parent[key]
        return key
    for tile in tiles:
        _,valid,grid=prepare_tile(tile,context);h,w=valid.shape;affine=Affine(*grid.affine)
        polygons[tile.key]=Polygon([affine*p for p in ((0,0),(w,0),(w,h),(0,h))])
    for i,tile in enumerate(tiles):
        for other in tiles[i+1:]:
            if polygons[tile.key].intersection(polygons[other.key]).area>0:
                a,b=sorted((find(tile.key),find(other.key)));parent[b]=a
    return {key:find(key) for key in parent}


def save_gnn_checkpoint(path,model,schema,normalization,settings):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp');torch.save(model.state_dict(),temporary);temporary.replace(path)
    atomic_json(path.with_suffix('.json'),{'schema':schema,'normalization':normalization,'settings':settings,
                 'state_sha256':sha256(path),'target':'building construction/extension pseudo labels'})


def _persist_result(directory,result):
    atomic_json(directory/'report.json',result)
    paths={'report':directory/'report.json'}
    if result.get('predictions'): paths['predictions']=Path(result['predictions'])
    if result.get('evaluation_predictions'): paths['evaluation_predictions']=Path(result['evaluation_predictions'])
    if result.get('checkpoint'):
        checkpoint=Path(result['checkpoint'])
        if checkpoint.is_relative_to(directory):
            paths.update(checkpoint=checkpoint,checkpoint_metadata=checkpoint.with_suffix('.json'))
        else:
            provenance=directory/'checkpoint_provenance.json'
            atomic_json(provenance,{'checkpoint':str(checkpoint),'sha256':sha256(checkpoint),
                                    'metadata_sha256':sha256(checkpoint.with_suffix('.json'))})
            paths['checkpoint_provenance']=provenance
    atomic_json(directory/'artifact.json',{'kind':'gnn','directory':directory,'paths':paths})
    save_manifest(directory,paths)


def _normalize(x,edge_attr,edge_index,train_nodes):
    train_edges=train_nodes[edge_index[0]]&train_nodes[edge_index[1]]
    mu=x[train_nodes].mean(0);sd=np.maximum(x[train_nodes].std(0),1e-6)
    em=edge_attr[train_edges].mean(0) if train_edges.any() else np.zeros(edge_attr.shape[1])
    es=np.maximum(edge_attr[train_edges].std(0),1e-6) if train_edges.any() else np.ones(edge_attr.shape[1])
    return {'node_mean':mu.tolist(),'node_std':sd.tolist(),'edge_mean':em.tolist(),'edge_std':es.tolist()}


def _tensors(x,edge_attr,edge_index,edge_type,norm,groups=None):
    normalized=(x-norm['node_mean'])/norm['node_std'];attributes=(edge_attr-norm['edge_mean'])/norm['edge_std']
    keep=np.ones(edge_type.shape,bool) if groups is None else groups[edge_index[0]]==groups[edge_index[1]]
    relations={}
    for ident,name in enumerate(('touches','near','temporal')):
        mask=(edge_type==ident)&keep
        relations[name]=(torch.from_numpy(edge_index[:,mask]),torch.from_numpy(attributes[mask].astype(np.float32)))
    return torch.from_numpy(normalized.astype(np.float32)),relations


def _metrics(labels,predictions):
    truth=labels.astype(int);pred=predictions.astype(int)
    tp=int(((truth==1)&(pred==1)).sum());fp=int(((truth==0)&(pred==1)).sum());fn=int(((truth==1)&(pred==0)).sum())
    precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn)
    return {'precision':precision,'recall':recall,'f1':2*precision*recall/max(precision+recall,1e-9),'count':len(truth)}


def _train(x,relations,labels,train,val,settings):
    torch.manual_seed(0)
    model=RGNN(x.shape[1],relations,hidden=settings['hidden'],edge_dim=settings['edge_dim'])
    optimizer=torch.optim.Adam(model.parameters(),lr=.005,weight_decay=.0005)
    truth=torch.from_numpy(np.maximum(labels,0).astype(np.float32));indices=torch.from_numpy(np.flatnonzero(train))
    positive=float((labels[train]==0).sum()/max(1,(labels[train]==1).sum()));best=None;best_loss=float('inf')
    for epoch in range(settings['epochs']):
        model.train();optimizer.zero_grad();logits=model(x,relations)
        loss=F.binary_cross_entropy_with_logits(logits[indices],truth[indices],pos_weight=torch.tensor(positive))
        loss.backward();optimizer.step()
        if val.any():
            model.eval()
            with torch.no_grad(): score=float(F.binary_cross_entropy_with_logits(model(x,relations)[val],truth[val]))
            if score<best_loss: best_loss=score;best=copy.deepcopy(model.state_dict())
    if best is not None: model.load_state_dict(best)
    model.eval()
    with torch.no_grad(): probabilities=torch.sigmoid(model(x,relations)).numpy()
    threshold=.5
    if val.any():
        candidates=np.unique(np.r_[.5,probabilities[val]])
        threshold=float(max(candidates,key=lambda t:(_metrics(labels[val],probabilities[val]>=t)['f1'],-abs(t-.5))))
    return model,probabilities,threshold


def _run_legacy_gnn(graph,context,*,checkpoint=None):
    schema=read_json(graph.paths['features']);document=read_json(graph.paths['graph'])
    with np.load(graph.paths['tensors'],allow_pickle=False) as data:
        x=data['x'];edge_attr=data['edge_attr'];edge_index=data['edge_index'];edge_type=data['edge_type'];ids=data['node_ids']
    directory=graph.directory/'gnn';directory.mkdir(exist_ok=True)
    labels=np.full(len(ids),-1,np.int64);index={ident:i for i,ident in enumerate(ids)}
    eligible=np.array([n['class_id']==1 and n['observed_fraction']>=.95
                       and n['known_counterpart_fraction']>=.95 and n['confidence']>=.6 for n in document['nodes']])
    targets=pd.read_csv(graph.paths['targets']) if graph.paths['targets'].exists() else pd.DataFrame()
    for row in targets.to_dict('records'):
        if row['gid'] in index and eligible[index[row['gid']]] and row['label'] in (0,1) and row['confidence']>=.6:
            labels[index[row['gid']]]=int(row['label'])
    positive=int((labels==1).sum());negative=int((labels==0).sum())
    result={'positive_targets':positive,'negative_targets':negative,'audit_status':'not_provided',
            'evaluation':{'status':'not_independently_evaluated','reason':'pseudo labels are classical rule outputs'},'predictions':None,'checkpoint':None}
    settings={'hidden':int(context.options.get('gnn_hidden') or 64),'edge_dim':edge_attr.shape[1],'epochs':int(context.options.get('gnn_epochs') or 300),'seed':0}
    if settings['epochs']<1: raise ValueError('GNN epochs must be positive')
    if checkpoint:
        checkpoint=Path(checkpoint);metadata=read_json(checkpoint.with_suffix('.json'))
        if metadata['schema']!=schema: raise ValueError('GNN checkpoint feature/target/unit schema is incompatible')
        if sha256(checkpoint)!=metadata['state_sha256']: raise ValueError('GNN checkpoint hash mismatch')
        normalization=metadata['normalization'];settings=metadata['settings']
        tensor,relations=_tensors(x,edge_attr,edge_index,edge_type,normalization)
        model=RGNN(x.shape[1],relations,hidden=settings['hidden'],edge_dim=settings['edge_dim'])
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.eval()
        with torch.no_grad(): probabilities=torch.sigmoid(model(tensor,relations)).numpy()
        threshold=float(settings.get('threshold',.5));result['status']='inference';result['checkpoint']=str(checkpoint)
    else:
        if not positive or not negative:
            result.update(status='insufficient_data',reason='Training needs reliable positive and negative building targets; supply a compatible --gnn-checkpoint or additional labelled image pairs')
            _persist_result(directory,result);return result
        tile_groups=evaluation_groups(context);groups=[]
        # Union group membership again at combined-object level, protecting multi-tile objects.
        parent={g:g for g in set(tile_groups.values())}
        def find(g):
            while parent[g]!=g: parent[g]=parent[parent[g]];g=parent[g]
            return g
        memberships=[]
        for node in document['nodes']:
            candidates=sorted({tile_groups[m['tile']] for m in node['members'] if m.get('tile') in tile_groups})
            memberships.append(candidates)
            for other in candidates[1:]: a,b=sorted((find(candidates[0]),find(other)));parent[b]=a
        for candidates in memberships: groups.append(find(candidates[0]) if candidates else 'unresolved_shared_group')
        groups=np.array(groups,dtype=str);usable=sorted(set(groups[labels>=0]))
        folds=[]
        support=all(set(labels[(groups==g)&(labels>=0)])=={0,1} for g in usable)
        if len(usable)>=3 and support:
            for i,test_group in enumerate(usable):
                val_group=usable[(i+1)%len(usable)];train_groups=set(usable)-{test_group,val_group}
                train_nodes=np.isin(groups,list(train_groups));train=train_nodes&(labels>=0);val=(groups==val_group)&(labels>=0);test=(groups==test_group)&(labels>=0)
                norm=_normalize(x,edge_attr,edge_index,train_nodes);tensor,relations=_tensors(x,edge_attr,edge_index,edge_type,norm,groups)
                _,pred,threshold=_train(tensor,relations,labels,train,val,settings)
                folds.append({'train_groups':sorted(train_groups),'validation_group':val_group,'test_group':test_group,
                              'threshold':threshold,'normalization':norm,'metrics':_metrics(labels[test],pred[test]>=threshold),
                              'cross_group_messages':False})
            result['evaluation']={'status':'grouped_pseudo_label_agreement','folds':folds,'independent_human_accuracy':False}
            result['status']='trained'
        else: result.update(status='training_only',reason='Independent train/validation/test evaluation requires >=3 nonoverlapping groups, each with both target classes')
        norm=_normalize(x,edge_attr,edge_index,np.ones(len(x),bool));tensor,relations=_tensors(x,edge_attr,edge_index,edge_type,norm)
        model,probabilities,threshold=_train(tensor,relations,labels,labels>=0,np.zeros(len(x),bool),settings)
        settings['threshold']=threshold;checkpoint=directory/'model.pt';save_gnn_checkpoint(checkpoint,model,schema,norm,settings)
        result.update(checkpoint=str(checkpoint),groups=sorted(set(groups)),folds=len(folds))
    predictions=directory/'predictions.csv'
    pd.DataFrame({'gid':ids,'probability':probabilities,'prediction':np.where(eligible,(probabilities>=threshold).astype(int),-1),'eligible_building_target':eligible}).to_csv(predictions,index=False)
    result['predictions']=str(predictions);result['threshold']=threshold
    _persist_result(directory,result)
    return result


def _node_groups(document,context,ids):
    tile_groups=evaluation_groups(context);parent={g:g for g in tile_groups.values()}
    def find(g):
        while parent[g]!=g:parent[g]=parent[parent[g]];g=parent[g]
        return g
    memberships={}
    for node in document['nodes']:
        candidates=sorted({tile_groups[m['tile']] for m in node.get('members',[]) if m.get('tile') in tile_groups})
        memberships[node['id']]=candidates
        for other in candidates[1:]:
            a,b=sorted((find(candidates[0]),find(other)));parent[b]=a
    return np.array([find(memberships[gid][0]) if memberships[gid] else 'unresolved_shared_group' for gid in ids],str)


def _check_task_normalization(norm,x,relations):
    def vector(value,length,positive=False):
        array=np.asarray(value,float)
        if array.shape!=(length,) or not np.isfinite(array).all() or (positive and (array<=0).any()):
            raise ValueError('GNN checkpoint normalization schema is invalid')
    vector(norm.get('node_mean'),x.shape[1]);vector(norm.get('node_std'),x.shape[1],True)
    if set(norm.get('relations',{}))!=set(relations):raise ValueError('GNN checkpoint relation schema is invalid')
    for name,(_,ea) in relations.items():
        vector(norm['relations'][name].get('mean'),ea.shape[1]);vector(norm['relations'][name].get('std'),ea.shape[1],True)


def run_gnn(graph,context,*,checkpoint=None):
    from .change_contracts import TASKS,RELATIONS,RELATION_PROFILE,CHANGE_SCHEMA,KNOWN_CLASSES,load_change_policy
    from .gnn_training import train_tasks,tensor_inputs
    schema=read_json(graph.paths['features'])
    if schema.get('relation_profile')!=RELATION_PROFILE:
        return _run_legacy_gnn(graph,context,checkpoint=checkpoint)
    if schema.get('tasks')!=list(TASKS) or schema.get('relations')!=list(RELATIONS):
        raise ValueError('All-class GNN feature/target schema is incompatible')
    document=read_json(graph.paths['graph']);by_id={n['id']:n for n in document['nodes']}
    with np.load(graph.paths['tensors'],allow_pickle=False) as t:
        x=t['x'];ei=t['edge_index'];ea=t['edge_attr'];et=t['edge_type'];ids=t['node_ids']
    if len(set(ids))!=len(ids) or set(ids)!=set(by_id):raise ValueError('GNN tensor node IDs do not match the graph')
    if not np.isin(et,range(len(RELATIONS))).all():raise ValueError('Unknown GNN input relation; derived changes cannot be model inputs')
    relations={name:(ei[:,et==i],ea[et==i]) for i,name in enumerate(RELATIONS)}
    policy=load_change_policy(context.options.get('change_policy'))
    eligible=np.zeros((len(ids),len(TASKS)),bool);labels=np.full(eligible.shape,-1,np.int64)
    index={gid:i for i,gid in enumerate(ids)};reasons={};first,last=document['years']
    catalog_path=graph.directory/'classical_changes.json'
    catalog=read_json(catalog_path) if catalog_path.is_file() else {}
    node_changes={r['gid']:r for r in catalog.get('node_changes',[])}
    for i,gid in enumerate(ids):
        node=by_id[gid];klass=node['class_id']
        if klass not in KNOWN_CLASSES:continue
        task=KNOWN_CLASSES[klass]+('_gain' if node['year']==last else '_loss')
        base=(node['observed_fraction']>=policy['observed_fraction'] and
              node['known_counterpart_fraction']>=policy['known_fraction'] and node['confidence']>=policy['reliable_confidence'])
        eligible[i,TASKS.index(task)]=node_changes.get(gid,{}).get('eligible',base)
    targets=pd.read_csv(graph.paths['targets']) if graph.paths['targets'].is_file() else pd.DataFrame()
    if 'task' in targets:
        if targets.duplicated(['gid','task']).any():raise ValueError('Duplicate GNN target node/task rows')
        for row in targets.to_dict('records'):
            if row['gid'] not in index or row['task'] not in TASKS:raise ValueError('Unknown GNN target node/task')
            i=index[row['gid']];j=TASKS.index(row['task']);reasons[i,j]=str(row.get('reason','unassessed'))
            if eligible[i,j] and row['label'] in (0,1) and row['confidence']>=policy['reliable_confidence']:
                labels[i,j]=int(row['label'])
    groups=_node_groups(document,context,ids)
    settings={'hidden':int(context.options.get('gnn_hidden') or 64),'epochs':int(context.options.get('gnn_epochs') or 300),
              'edge_dim':ea.shape[1],'seed':0}
    directory=graph.directory/'gnn';directory.mkdir(exist_ok=True)
    result={'positive_targets':int((labels==1).sum()),'negative_targets':int((labels==0).sum()),
        'audit_status':'not_provided','target_schema':CHANGE_SCHEMA,'predictions':None,'checkpoint':None,
        'evaluation':{'status':'not_independently_evaluated','reason':'Pseudo labels are image-rule outputs',
                      'independent_human_accuracy':False},'groups':sorted(set(groups))}
    if checkpoint:
        checkpoint=Path(checkpoint);metadata=read_json(checkpoint.with_suffix('.json'))
        if (metadata.get('schema')!=schema or metadata.get('target_schema')!=CHANGE_SCHEMA
                or metadata.get('task_order')!=list(TASKS) or metadata.get('relation_profile')!=RELATION_PROFILE):
            raise ValueError('GNN checkpoint feature/target schema is incompatible; legacy building checkpoints require a compatible legacy graph')
        if sha256(checkpoint)!=metadata.get('state_sha256'):raise ValueError('GNN checkpoint hash mismatch')
        heads=copy.deepcopy(metadata.get('head_status',{}));norm=metadata['normalization'];settings=metadata['settings']
        if set(heads)!=set(TASKS):raise ValueError('GNN checkpoint task schema is incompatible')
        for task,head in heads.items():
            if head.get('available'):
                threshold=head.get('threshold')
                if (not head.get('positive_targets',0) or not head.get('negative_targets',0)
                        or not isinstance(threshold,(int,float)) or not np.isfinite(threshold) or not 0<=threshold<=1):
                    raise ValueError('GNN checkpoint head support/threshold schema is invalid')
                head.update(status='inference',reason='Compatible saved class-change head')
        _check_task_normalization(norm,x,relations)
        tensor,r=tensor_inputs(x,relations,norm)
        model=RGNN(x.shape[1],r,hidden=settings['hidden'],edge_dim=settings['edge_dim'],tasks=TASKS)
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);model.eval()
        with torch.no_grad():probabilities=torch.sigmoid(model(tensor,r)).numpy()
        evaluation=np.full(probabilities.shape,np.nan);eval_decisions=np.full(probabilities.shape,-1)
        for j,task in enumerate(TASKS):
            if not heads[task]['available']:probabilities[:,j]=np.nan
        result.update(status='inference',checkpoint=str(checkpoint),folds=0)
    else:
        trained=train_tasks(x,relations,labels,eligible,groups,settings)
        heads=trained['heads'];probabilities=trained['probabilities'];evaluation=trained['evaluation_probabilities'];eval_decisions=trained['evaluation_predictions']
        result.update(status=trained['status'],folds=len(trained['folds']))
        if trained['folds']:
            result['evaluation']={'status':'grouped_pseudo_label_agreement','folds':trained['folds'],'independent_human_accuracy':False}
        if trained['model'] is not None:
            checkpoint=directory/'model.pt'
            save_gnn_checkpoint(checkpoint,trained['model'],schema,trained['normalization'],trained['settings'])
            metadata=read_json(checkpoint.with_suffix('.json'))
            metadata.update(target_schema=CHANGE_SCHEMA,task_order=list(TASKS),relation_profile=RELATION_PROFILE,
                            head_status=heads,target='Per-class gain/loss pseudo labels',
                            threshold_source='Fixed 0.5 for deployment; held-out fold thresholds are stored separately')
            atomic_json(checkpoint.with_suffix('.json'),metadata);result['checkpoint']=str(checkpoint)
    rows=[];evaluation_rows=[]
    for i,gid in enumerate(ids):
        for j,task in enumerate(TASKS):
            available=bool(heads[task]['available'] and eligible[i,j])
            score=float(probabilities[i,j]) if available else None
            row={'gid':gid,'task':task,'probability':score,
                 'prediction':int(score>=heads[task]['threshold']) if score is not None else -1,
                 'eligible':bool(eligible[i,j]),'reason':reasons.get((i,j),'different_class_or_year') if not available else heads[task]['reason'],
                 'status':heads[task]['status'] if eligible[i,j] else 'ineligible','probability_source':'deployment'}
            rows.append(row)
            if available and np.isfinite(evaluation[i,j]):
                evaluation_rows.append({**row,'probability':float(evaluation[i,j]),'prediction':int(eval_decisions[i,j]),
                                        'probability_source':'held_out'})
    predictions=directory/'predictions.csv';pd.DataFrame(rows,columns=['gid','task','probability','prediction','eligible','reason','status','probability_source']).to_csv(predictions,index=False)
    evaluation_path=directory/'evaluation_predictions.csv'
    pd.DataFrame(evaluation_rows,columns=['gid','task','probability','prediction','eligible','reason','status','probability_source']).to_csv(evaluation_path,index=False)
    result.update(predictions=str(predictions),evaluation_predictions=str(evaluation_path),head_status=heads,
                  reason='See per-task support; unavailable heads produce no scores')
    _persist_result(directory,result)
    return result
