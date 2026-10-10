import torch
from pipelines.semantic_temporal.runtime.gnn_model import RGNN


def test_temporal_messages_distinguish_changes_hidden_by_neighbor_averaging():
    # Both graphs deliver the same average neighbor value to node0. One is a
    # mixture of different counterparts; the other has matching counterparts.
    torch.manual_seed(11)
    model=RGNN(1,['t_primary_fwd'],hidden=8,layers=1,drop=0,edge_dim=10,
        tasks=['building_gain'],message_mode='temporal_difference.v1').eval()
    edges=torch.tensor([[1,2],[0,0]])
    relations={'t_primary_fwd':(edges,torch.zeros(2,10))}
    mixed=torch.tensor([[1.],[0.],[2.]])
    matched=torch.tensor([[1.],[1.],[1.]])
    assert not torch.allclose(model(mixed,relations)[0],model(matched,relations)[0])


def test_missing_temporal_match_has_an_explicit_signal():
    torch.manual_seed(12)
    model=RGNN(2,['t_primary_fwd'],hidden=8,layers=1,drop=0,edge_dim=10,
        tasks=['building_gain'],message_mode='temporal_difference.v1').eval()
    x=torch.zeros(2,2)
    empty={'t_primary_fwd':(torch.empty((2,0),dtype=torch.long),torch.empty(0,10))}
    matched={'t_primary_fwd':(torch.tensor([[1],[0]]),torch.zeros(1,10))}
    assert not torch.allclose(model(x,empty)[0],model(x,matched)[0])
    assert torch.isfinite(model(x,empty)).all()


def test_temporal_model_is_equivariant_to_graph_node_reordering():
    torch.manual_seed(13)
    model=RGNN(3,['touches','t_primary_fwd'],hidden=8,drop=0,edge_dim=10,
        tasks=['building_gain','road_gain'],message_mode='temporal_difference.v1').eval()
    x=torch.randn(4,3);edges=torch.tensor([[0,1,3],[1,2,2]]);attrs=torch.randn(3,10)
    rels={name:(edges,attrs) for name in model.rels}
    permutation=torch.tensor([2,0,3,1]);inverse=torch.argsort(permutation)
    reordered={name:(inverse[edges],attrs) for name in model.rels}
    torch.testing.assert_close(model(x[permutation],reordered),model(x,rels)[permutation])


def test_temporal_mode_checkpoint_roundtrips_through_pipeline(tmp_path):
    from dataclasses import replace
    import pandas as pd
    import numpy as np
    from tests.test_semantic_temporal import temporal_fixture
    from pipelines.semantic_temporal.runtime.temporal import detect_classical
    from pipelines.semantic_temporal.runtime.gnn import run_gnn,save_gnn_checkpoint
    from pipelines.semantic_temporal.runtime.storage import read_json,atomic_json
    from pipelines.semantic_temporal.runtime.change_contracts import TASKS,RELATIONS
    graph,context=temporal_fixture(tmp_path);detect_classical(graph,context)
    initial=run_gnn(graph,replace(context,options={'gnn_epochs':1,'gnn_hidden':8}))
    from pathlib import Path
    checkpoint=Path(initial['checkpoint']);metadata=read_json(checkpoint.with_suffix('.json'))
    with np.load(graph.paths['tensors']) as data:dim=data['x'].shape[1]
    model=RGNN(dim,RELATIONS,hidden=8,edge_dim=10,tasks=TASKS,message_mode='temporal_difference.v1')
    settings={**metadata['settings'],'message_mode':'temporal_difference.v1'}
    save_gnn_checkpoint(checkpoint,model,metadata['schema'],metadata['normalization'],settings)
    updated=read_json(checkpoint.with_suffix('.json'))
    for key in ('head_status','target_schema','task_order','relation_profile'):updated[key]=metadata[key]
    atomic_json(checkpoint.with_suffix('.json'),updated)
    first=run_gnn(graph,context,checkpoint=checkpoint)
    probabilities=pd.read_csv(first['predictions']).probability.to_numpy()
    second=run_gnn(graph,context,checkpoint=checkpoint)
    np.testing.assert_allclose(probabilities,pd.read_csv(second['predictions']).probability,equal_nan=True)


def test_graph_task_expert_only_replaces_its_declared_head():
    from pipelines.semantic_temporal.runtime.gnn_model import model_from_settings
    tasks=['building_gain','building_loss','road_gain']
    settings=dict(hidden=8,edge_dim=10,message_mode='temporal_difference.v1',
        task_experts=[dict(tasks=['building_loss'],message_mode='legacy')])
    model=model_from_settings(3,['touches','t_primary_fwd'],settings,tasks).eval()
    x=torch.randn(4,3);edges=torch.tensor([[0,1,3],[1,2,2]]);attrs=torch.randn(3,10)
    relations={name:(edges,attrs) for name in model.base.rels}
    actual=model(x,relations);base=model.base(x,relations);expert=model.experts[0](x,relations)
    torch.testing.assert_close(actual[:,[0,2]],base[:,[0,2]])
    torch.testing.assert_close(actual[:,1],expert[:,1])
    # The selected expert remains graph-dependent, not a scalar override.
    no_edges={name:(torch.empty((2,0),dtype=torch.long),torch.empty(0,10)) for name in relations}
    assert not torch.allclose(model(x,no_edges)[:,1],actual[:,1])


def test_graph_task_expert_settings_reject_ambiguous_or_unknown_owners():
    import pytest
    from pipelines.semantic_temporal.runtime.gnn_model import model_from_settings
    settings=dict(hidden=8,edge_dim=10)
    for experts in ([dict(tasks=['unknown'],message_mode='legacy')],
                    [dict(tasks=['building_gain'],message_mode='legacy')]*2,
                    [dict(tasks=[],message_mode='legacy')],
                    [dict(tasks=['building_gain'],message_mode='invalid')]):
        with pytest.raises(ValueError):
            model_from_settings(3,['touches'],dict(settings,task_experts=experts),['building_gain'])


def test_model_factory_preserves_ordinary_rgnn_state_and_output():
    from pipelines.semantic_temporal.runtime.gnn_model import model_from_settings
    settings=dict(hidden=8,edge_dim=10,message_mode='legacy');tasks=['building_gain']
    torch.manual_seed(91);expected=RGNN(3,['touches'],hidden=8,edge_dim=10,tasks=tasks).eval()
    torch.manual_seed(91);actual=model_from_settings(3,['touches'],settings,tasks).eval()
    assert type(actual) is RGNN
    assert actual.state_dict().keys()==expected.state_dict().keys()
    x=torch.randn(2,3);r={'touches':(torch.tensor([[0],[1]]),torch.zeros(1,10))}
    torch.testing.assert_close(actual(x,r),expected(x,r),rtol=0,atol=0)


def test_graph_expert_bundle_roundtrips_through_native_checkpoint_inference(tmp_path):
    from dataclasses import replace
    from pathlib import Path
    import numpy as np
    import pandas as pd
    from tests.test_semantic_temporal import temporal_fixture
    from pipelines.semantic_temporal.runtime.temporal import detect_classical
    from pipelines.semantic_temporal.runtime.gnn import run_gnn,save_gnn_checkpoint
    from pipelines.semantic_temporal.runtime.gnn_model import model_from_settings
    from pipelines.semantic_temporal.runtime.storage import read_json,atomic_json
    from pipelines.semantic_temporal.runtime.change_contracts import TASKS,RELATIONS
    graph,context=temporal_fixture(tmp_path);detect_classical(graph,context)
    initial=run_gnn(graph,replace(context,options={'gnn_epochs':1,'gnn_hidden':8}))
    checkpoint=Path(initial['checkpoint']);metadata=read_json(checkpoint.with_suffix('.json'))
    with np.load(graph.paths['tensors']) as arrays:dim=arrays['x'].shape[1]
    settings=dict(metadata['settings'],message_mode='temporal_difference.v1',
        task_experts=[dict(tasks=['building_gain'],message_mode='legacy')])
    model=model_from_settings(dim,RELATIONS,settings,TASKS)
    with torch.no_grad():
        for parameter in model.experts[0].heads['building_gain'].parameters():parameter.zero_()
        model.experts[0].heads['building_gain'][-1].bias.fill_(2.)
    save_gnn_checkpoint(checkpoint,model,metadata['schema'],metadata['normalization'],settings)
    updated=read_json(checkpoint.with_suffix('.json'))
    for key in ('head_status','target_schema','task_order','relation_profile'):updated[key]=metadata[key]
    updated['head_status']['building_gain'].update(available=True,positive_targets=1,negative_targets=1,threshold=.5)
    updated['inference_eligibility']='observed_known_class_node.v1'
    atomic_json(checkpoint.with_suffix('.json'),updated)
    result=run_gnn(graph,context,checkpoint=checkpoint)
    rows=pd.read_csv(result['predictions']);scores=rows[(rows.task=='building_gain')&rows.probability.notna()]
    assert len(scores)>0
    np.testing.assert_allclose(scores.probability,torch.sigmoid(torch.tensor(2.)).item(),rtol=1e-6)
    assert (scores.prediction==1).all()
