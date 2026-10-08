"""Relational GNN training with grouped, isolated evaluation and reusable state."""
from pathlib import Path
import copy
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from affine import Affine
from shapely.geometry import Polygon,shape
from .storage import atomic_json,read_json,sha256
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


def run_gnn(graph,context,*,checkpoint=None):
    schema=read_json(graph.paths['features']);document=read_json(graph.paths['graph'])
    with np.load(graph.paths['tensors'],allow_pickle=False) as data:
        x=data['x'];edge_attr=data['edge_attr'];edge_index=data['edge_index'];edge_type=data['edge_type'];ids=data['node_ids']
    directory=graph.directory/'gnn';directory.mkdir(exist_ok=True)
    labels=np.full(len(ids),-1,np.int64);index={ident:i for i,ident in enumerate(ids)}
    targets=pd.read_csv(graph.paths['targets']) if graph.paths['targets'].exists() else pd.DataFrame()
    for row in targets.to_dict('records'):
        if row['gid'] in index and row['label'] in (0,1) and row['confidence']>=.6: labels[index[row['gid']]]=int(row['label'])
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
            atomic_json(directory/'report.json',result);return result
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
    eligible=np.array([n['class_id']==1 and n['observed_fraction']>=.95 and n['known_counterpart_fraction']>=.95 for n in document['nodes']])
    pd.DataFrame({'gid':ids,'probability':probabilities,'prediction':np.where(eligible,(probabilities>=threshold).astype(int),-1),'eligible_building_target':eligible}).to_csv(predictions,index=False)
    result['predictions']=str(predictions);result['threshold']=threshold
    atomic_json(directory/'report.json',result)
    return result
