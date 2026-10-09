"""Joint RGNN task training with masked labels and isolated grouped evaluation."""
import copy
from numbers import Integral
import numpy as np
import torch
from torch.nn import functional as F
from .gnn_model import RGNN
from .change_contracts import TASKS


def normalization(x,relations,train_nodes):
    if not train_nodes.any(): raise ValueError('Normalization requires training context nodes')
    result={'node_mean':x[train_nodes].mean(0).tolist(),
            'node_std':np.maximum(x[train_nodes].std(0),1e-6).tolist(),'relations':{}}
    for name,(ei,ea) in relations.items():
        keep=train_nodes[ei[0]]&train_nodes[ei[1]]
        mean=ea[keep].mean(0) if keep.any() else np.zeros(ea.shape[1])
        std=np.maximum(ea[keep].std(0),1e-6) if keep.any() else np.ones(ea.shape[1])
        result['relations'][name]={'mean':mean.tolist(),'std':std.tolist()}
    return result


def tensor_inputs(x,relations,norm,groups=None):
    tensor=torch.from_numpy(((x-norm['node_mean'])/norm['node_std']).astype(np.float32))
    output={}
    for name,(ei,ea) in relations.items():
        keep=np.ones(ei.shape[1],bool) if groups is None else groups[ei[0]]==groups[ei[1]]
        values=norm['relations'][name]
        attrs=((ea[keep]-values['mean'])/values['std']).astype(np.float32)
        output[name]=(torch.from_numpy(ei[:,keep].astype(np.int64)),torch.from_numpy(attrs))
    return tensor,output


def _metrics(truth,prediction):
    tp=int(((truth==1)&(prediction==1)).sum());fp=int(((truth==0)&(prediction==1)).sum());fn=int(((truth==1)&(prediction==0)).sum())
    precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn)
    return {'precision':precision,'recall':recall,'f1':2*precision*recall/max(precision+recall,1e-9),'count':len(truth)}


def _average_precision(truth,scores):
    order=np.argsort(-scores,kind='stable');ordered=scores[order]
    ends=np.r_[np.flatnonzero(np.diff(ordered)!=0),len(order)-1]
    tp=np.cumsum(truth[order]==1)[ends];delta=np.diff(np.r_[0,tp])
    return float(np.sum(tp/(ends+1)*delta)/max(1,(truth==1).sum()))


def _fit(x,relations,labels,train,val,settings):
    torch.manual_seed(0)
    model=RGNN(x.shape[1],relations,hidden=settings['hidden'],edge_dim=settings['edge_dim'],tasks=TASKS)
    optimizer=torch.optim.Adam(model.parameters(),lr=.005,weight_decay=.0005)
    usable=[i for i in range(len(TASKS)) if set(labels[train[:,i],i])=={0,1}]
    if not usable: raise ValueError('Training requires at least one supported task')
    truth=torch.from_numpy(np.maximum(labels,0).astype(np.float32))
    indices={i:torch.from_numpy(np.flatnonzero(train[:,i])) for i in usable}
    weights={i:torch.tensor(float((labels[train[:,i],i]==0).sum()/max(1,(labels[train[:,i],i]==1).sum()))) for i in usable}
    snapshots={};best={}
    for epoch in range(settings['epochs']):
        model.train();optimizer.zero_grad();logits=model(x,relations)
        loss=torch.stack([F.binary_cross_entropy_with_logits(logits[indices[i],i],truth[indices[i],i],pos_weight=weights[i]) for i in usable]).mean()
        if not torch.isfinite(loss): raise ValueError('Nonfinite RGNN training loss')
        loss.backward();optimizer.step()
        if val.any():
            model.eval()
            with torch.no_grad(): logits=model(x,relations);prob=torch.sigmoid(logits).numpy()
            for i in usable:
                mask=val[:,i]
                if set(labels[mask,i])!={0,1}:continue
                ap=_average_precision(labels[mask,i],prob[mask,i])
                bce=float(F.binary_cross_entropy_with_logits(logits[mask,i],truth[mask,i]))
                score=(ap,-bce)
                if i not in best or score>best[i]:
                    best[i]=score;snapshots[i]=(copy.deepcopy(model.state_dict()),epoch+1)
    model.eval()
    with torch.no_grad(): probabilities=torch.sigmoid(model(x,relations)).numpy()
    thresholds=np.full(len(TASKS),.5);epochs={}
    final_state=copy.deepcopy(model.state_dict())
    for i,(state,epoch) in snapshots.items():
        model.load_state_dict(state);model.eval()
        with torch.no_grad(): probabilities[:,i]=torch.sigmoid(model(x,relations)[:,i]).numpy()
        mask=val[:,i];candidates=np.unique(np.r_[.5,probabilities[mask,i]])
        thresholds[i]=max(candidates,key=lambda t:(_metrics(labels[mask,i],probabilities[mask,i]>=t)['f1'],-abs(t-.5)))
        epochs[i]=epoch
    model.load_state_dict(final_state);model.eval()
    return model,probabilities,thresholds,epochs


def train_tasks(x,relations,labels,eligibility,groups,settings):
    labels=np.asarray(labels)
    if not np.isin(labels,[-1,0,1]).all():raise ValueError('GNN targets must be -1, 0, or 1')
    x=np.asarray(x,np.float32);labels=labels.astype(np.int64);eligibility=np.asarray(eligibility,bool);groups=np.asarray(groups,str)
    settings={**settings,'hidden':settings.get('hidden',64),'epochs':settings.get('epochs',300),
              'edge_dim':settings.get('edge_dim',10),'seed':0}
    for key in ('hidden','epochs','edge_dim'):
        if isinstance(settings[key],bool) or not isinstance(settings[key],Integral) or settings[key]<1:
            raise ValueError(f'GNN {key} must be a positive integer')
    if (x.ndim!=2 or labels.shape!=(len(x),len(TASKS)) or eligibility.shape!=labels.shape
            or groups.shape!=(len(x),) or not np.isfinite(x).all() or not np.isin(labels,[-1,0,1]).all()):
        raise ValueError('Invalid all-class GNN feature/target arrays')
    for name,(ei,ea) in relations.items():
        if ei.shape!=(2,len(ea)) or ea.shape[1:]!=(settings['edge_dim'],) or not np.isfinite(ea).all():
            raise ValueError(f'Invalid relation tensors: {name}')
        if ei.size and (ei.min()<0 or ei.max()>=len(x)):raise ValueError('Relation node index out of range')
    torch.set_num_threads(4)
    known=eligibility&(labels>=0);heads={};supported=[]
    for i,task in enumerate(TASKS):
        positive=int((known[:,i]&(labels[:,i]==1)).sum());negative=int((known[:,i]&(labels[:,i]==0)).sum())
        available=bool(positive and negative)
        heads[task]={'available':available,'positive_targets':positive,'negative_targets':negative,
            'status':'training_only' if available else 'insufficient_data',
            'threshold':.5 if available else None,
            'reason':'Independent train/validation/test evaluation needs at least three groups with both labels' if available else 'Training needs reliable changed and unchanged examples'}
        if available:supported.append(i)
    probabilities=np.full(labels.shape,np.nan,np.float32);evaluation=np.full_like(probabilities,np.nan)
    decisions=np.full(labels.shape,-1,np.int64);folds=[];model=None;norm=None
    if supported:
        for i in supported:
            usable=sorted(g for g in set(groups[known[:,i]]) if set(labels[(groups==g)&known[:,i],i])=={0,1})
            if len(usable)<3:continue
            for j,test_group in enumerate(usable):
                val_group=usable[(j+1)%len(usable)];train_groups=sorted(set(usable)-{test_group,val_group})
                train_nodes=np.isin(groups,train_groups);train=known&train_nodes[:,None]
                val=np.zeros_like(known);val[:,i]=known[:,i]&(groups==val_group)
                fold_norm=normalization(x,relations,train_nodes)
                tensor,r=tensor_inputs(x,relations,fold_norm,groups)
                _,scores,thresholds,best_epochs=_fit(tensor,r,labels,train,val,settings)
                test=(groups==test_group)&known[:,i];threshold=float(thresholds[i])
                output=(groups==test_group)&eligibility[:,i]
                evaluation[output,i]=scores[output,i];decisions[output,i]=scores[output,i]>=threshold
                folds.append({'task':TASKS[i],'train_groups':train_groups,'validation_group':val_group,
                    'test_group':test_group,'threshold':threshold,'best_epoch':best_epochs.get(i),
                    'normalization':fold_norm,'metrics':_metrics(labels[test,i],scores[test,i]>=threshold),
                    'cross_group_messages':False,'independent_human_accuracy':False})
            heads[TASKS[i]].update(status='trained',reason='Grouped pseudo-label agreement; not independent human accuracy')
        norm=normalization(x,relations,np.ones(len(x),bool));tensor,r=tensor_inputs(x,relations,norm)
        model,scores,_,_=_fit(tensor,r,labels,known,np.zeros_like(known),settings)
        probabilities[:,supported]=scores[:,supported]
    return {'model':model,'normalization':norm,'settings':settings,'heads':heads,'folds':folds,
        'probabilities':probabilities,'evaluation_probabilities':evaluation,'evaluation_predictions':decisions,
        'status':'trained' if folds else 'training_only' if supported else 'insufficient_data'}
