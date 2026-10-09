import ast
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from pipelines.semantic_temporal.runtime.change_contracts import TASKS


def training_fixture():
    x=np.column_stack([np.repeat([0,100,200],4)+np.tile(np.arange(4),3),np.ones(12),np.tile([0,1,0,1],3)]).astype(np.float32)
    labels=np.full((12,16),-1,np.int64)
    labels[:,0]=np.tile([0,1,-1,-1],3);labels[:,2]=np.tile([-1,-1,0,1],3)
    labels[0,1]=1
    eligible=labels>=0
    edges=np.array([[0,1,4,5,8,9,1],[1,0,5,4,9,8,4]],np.int64)
    relations={'touches':(edges,np.repeat(np.arange(7,dtype=np.float32)[:,None],10,axis=1)),
               'near':(np.empty((2,0),np.int64),np.empty((0,10),np.float32))}
    return x,relations,labels,eligible,np.repeat(['a','b','c'],4)


def test_unsupported_head_has_no_score():
    from pipelines.semantic_temporal.runtime.gnn_training import train_tasks
    args=training_fixture()
    result=train_tasks(*args,{'hidden':8,'epochs':2,'edge_dim':10})
    assert result['heads']['building_gain']['available']
    assert result['heads']['road_gain']['available']
    assert not result['heads']['building_loss']['available']
    assert np.isnan(result['probabilities'][:,1]).all()
    assert np.isnan(result['probabilities'][:,3:]).all()


def test_grouped_evaluation_is_isolated_and_normalized_on_training_only():
    from pipelines.semantic_temporal.runtime.gnn_training import train_tasks
    result=train_tasks(*training_fixture(),{'hidden':8,'epochs':2,'edge_dim':10})
    assert result['heads']['building_gain']['status']=='trained'
    folds=[f for f in result['folds'] if f['task']=='building_gain']
    assert len(folds)==3
    for fold in folds:
        assert fold['cross_group_messages'] is False
        assert fold['normalization']['node_mean'][0] == ['a','b','c'].index(fold['train_groups'][0])*100+1.5
        assert fold['normalization']['relations']['near']['std']==[1]*10
    assert np.isfinite(result['evaluation_probabilities'][:2,0]).all()


def test_masked_labels_cannot_affect_training():
    from pipelines.semantic_temporal.runtime.gnn_training import train_tasks
    x,r,y,e,g=training_fixture();g[:]='one'
    original=train_tasks(x,r,y,e,g,{'hidden':8,'epochs':2,'edge_dim':10})
    poisoned=y.copy();poisoned[~e]=1
    changed=train_tasks(x,r,poisoned,e,g,{'hidden':8,'epochs':2,'edge_dim':10})
    np.testing.assert_allclose(original['probabilities'],changed['probabilities'],equal_nan=True)
    assert original['heads']==changed['heads']


def test_checkpoint_roundtrip_and_wrong_class_year_outputs(tmp_path):
    from tests.test_semantic_temporal import temporal_fixture
    from pipelines.semantic_temporal.runtime.temporal import detect_classical
    from pipelines.semantic_temporal.runtime.gnn import run_gnn
    graph,context=temporal_fixture(tmp_path);detect_classical(graph,context)
    context=replace(context,options={'gnn_epochs':2,'gnn_hidden':8})
    first=run_gnn(graph,context);pred=pd.read_csv(first['predictions'])
    assert set(pred.task)==set(TASKS)
    unsupported=pred.task=='water_gain'
    assert pred.loc[unsupported,'probability'].isna().all()
    assert (pred.loc[unsupported,'prediction']==-1).all()
    first_scores=pred.probability.to_numpy()
    second=run_gnn(graph,context,checkpoint=first['checkpoint'])
    loaded=pd.read_csv(second['predictions'])
    np.testing.assert_allclose(first_scores,loaded.probability.to_numpy(),equal_nan=True)
    assert pred.loc[~pred.eligible,'probability'].isna().all()
    assert first['head_status']['building_gain']['positive_targets']==1


def test_original_encoder_and_scalar_head_weights_are_unchanged():
    from pipelines.semantic_temporal.runtime.gnn_model import RGNN
    source=Path(__file__).resolve().parents[1]/'pipelines/semantic_temporal/pipeline/gnn_change.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RGNN')
    scope={'torch':torch,'nn':nn,'F':F}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),scope)
    rels=['t_primary_fwd','t_primary_bwd'];torch.manual_seed(1)
    old=scope['RGNN'](4,rels,hidden=8);new=RGNN(4,rels,hidden=8,edge_dim=10)
    new.load_state_dict(old.state_dict(),strict=True);old.eval();new.eval()
    x=torch.arange(12,dtype=torch.float32).reshape(3,4)/12
    r={name:(torch.tensor([[0,1],[1,2]]),torch.ones(2,10)) for name in rels}
    assert torch.equal(old(x,r),new(x,r))


def test_multitask_encoder_preserves_message_passing_parameters():
    from pipelines.semantic_temporal.runtime.gnn_model import RGNN
    model=RGNN(4,['touches'],hidden=8,edge_dim=10,tasks=TASKS)
    x=torch.ones(3,4);r={'touches':(torch.empty((2,0),dtype=torch.int64),torch.empty((0,10)))}
    assert model(x,r).shape==(3,16)
    assert 'self_w.0.weight' in model.state_dict()


def test_legacy_binary_pipeline_still_roundtrips_compatible_profile(tmp_path):
    from tests.test_semantic_temporal import temporal_fixture
    from pipelines.semantic_temporal.runtime.temporal import detect_classical
    from pipelines.semantic_temporal.runtime.gnn import run_gnn
    from pipelines.semantic_temporal.runtime.storage import atomic_json
    graph,context=temporal_fixture(tmp_path);detect_classical(graph,context)
    with np.load(graph.paths['tensors']) as t:data={key:t[key] for key in t.files}
    data['x']=data['x'][:,:-1];data['edge_attr']=data['edge_attr'][:,:9];data['edge_type']=np.where(data['edge_type']>=3,2,data['edge_type'])
    np.savez_compressed(graph.paths['tensors'],**data)
    atomic_json(graph.paths['features'],{'node_features':['legacy']*data['x'].shape[1],
        'edge_features':['legacy']*9,'relations':['touches','near','temporal'],'target':'building_change','units':'metres'})
    graph=replace(graph,paths={**graph.paths,'targets':graph.directory/'legacy_training_targets.csv'})
    result=run_gnn(graph,replace(context,options={'gnn_epochs':1,'gnn_hidden':8}))
    first=pd.read_csv(result['predictions'])
    loaded=run_gnn(graph,context,checkpoint=result['checkpoint'])
    np.testing.assert_allclose(first.probability,pd.read_csv(loaded['predictions']).probability)


def test_fractional_labels_are_rejected_instead_of_rounded():
    from pipelines.semantic_temporal.runtime.gnn_training import train_tasks
    x,r,y,e,g=training_fixture();y=y.astype(float);y[0,0]=.5
    with pytest.raises(ValueError,match='target'):
        train_tasks(x,r,y,e,g,{'hidden':8,'epochs':1,'edge_dim':10})


def test_checkpoint_declared_task_order_is_verified(tmp_path):
    from tests.test_semantic_temporal import temporal_fixture
    from pipelines.semantic_temporal.runtime.temporal import detect_classical
    from pipelines.semantic_temporal.runtime.gnn import run_gnn
    from pipelines.semantic_temporal.runtime.storage import read_json,atomic_json
    graph,context=temporal_fixture(tmp_path);detect_classical(graph,context)
    result=run_gnn(graph,replace(context,options={'gnn_epochs':1,'gnn_hidden':8}))
    path=Path(result['checkpoint']).with_suffix('.json');metadata=read_json(path)
    metadata['task_order']=list(reversed(metadata['task_order']));atomic_json(path,metadata)
    with pytest.raises(ValueError,match='schema'):
        run_gnn(graph,context,checkpoint=result['checkpoint'])
