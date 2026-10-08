import numpy as np
import pandas as pd
import pytest
from dataclasses import replace
from tests.test_semantic_temporal import temporal_fixture
from pipelines.semantic_temporal.runtime.storage import read_json,atomic_json


def test_single_pair_trains_without_heldout_claims_and_checkpoint_roundtrip(tmp_path):
    from pipelines.semantic_temporal.runtime.temporal import detect_classical
    from pipelines.semantic_temporal.runtime.gnn import run_gnn
    artifact,context=temporal_fixture(tmp_path);detect_classical(artifact,context)
    context=replace(context,options={'gnn_epochs':3,'gnn_hidden':8})
    first=run_gnn(artifact,context)
    assert first['status']=='training_only' and first['evaluation']['status']=='not_independently_evaluated'
    assert first['audit_status']=='not_provided'
    trained=pd.read_csv(first['predictions'])
    second=run_gnn(artifact,context,checkpoint=first['checkpoint'])
    loaded=pd.read_csv(second['predictions'])
    assert second['status']=='inference'
    assert np.allclose(trained.probability,loaded.probability)
    features=read_json(artifact.paths['features']);features['units']='pixels';atomic_json(artifact.paths['features'],features)
    with pytest.raises(ValueError,match='schema'): run_gnn(artifact,context,checkpoint=first['checkpoint'])


def test_no_positive_targets_returns_counts_and_no_predictions(tmp_path):
    from pipelines.semantic_temporal.runtime.gnn import run_gnn
    artifact,context=temporal_fixture(tmp_path)
    pd.DataFrame([{'gid':'irrelevant','label':0,'confidence':1,'source':'rule'}]).to_csv(artifact.paths['targets'],index=False)
    result=run_gnn(artifact,context)
    assert result['status']=='insufficient_data' and result['positive_targets']==0
    assert result['predictions'] is None


def test_overlapping_input_observations_share_evaluation_group(tmp_path):
    from tests.semantic_fixtures import make_year
    from pipelines.semantic_temporal.runtime.context import build_context
    from pipelines.semantic_temporal.runtime.registration import prepare_comparison
    from pipelines.semantic_temporal.runtime.gnn import evaluation_groups
    earlier=make_year(tmp_path,year=2017);later=make_year(tmp_path,year=2023,names=('different.png',))
    context=prepare_comparison(build_context({'folders':[str(earlier),str(later)]},tmp_path/'run'))
    groups=evaluation_groups(context)
    assert len(groups)==2 and len(set(groups.values()))==1


def test_three_groups_use_train_only_normalization_and_isolated_folds(tmp_path):
    import copy
    from tests.semantic_fixtures import make_year
    from pipelines.semantic_temporal.runtime.context import build_context
    from pipelines.semantic_temporal.runtime.registration import prepare_comparison
    from pipelines.semantic_temporal.runtime.temporal import detect_classical
    from pipelines.semantic_temporal.runtime.gnn import run_gnn
    artifact,context=temporal_fixture(tmp_path/'pair');detect_classical(artifact,context)
    names=('a.png','b.png','c.png');offsets=((3,2),(30,2),(60,2))
    earlier=make_year(tmp_path/'groups',2017,names=names,offsets=offsets)
    later=make_year(tmp_path/'groups',2023,names=names,offsets=offsets)
    context=prepare_comparison(build_context({'folders':[str(earlier),str(later)],'gnn_epochs':2,'gnn_hidden':8},context.output))
    document=read_json(artifact.paths['graph']);new_nodes=[];new_edges=[]
    original_targets=pd.read_csv(artifact.paths['targets']);targets=[]
    with np.load(artifact.paths['tensors'],allow_pickle=False) as data: original={key:data[key] for key in data.files}
    xs=[];eis=[];eas=[];ets=[];ids=[]
    for group,name in enumerate(names):
        prefix=f'{group}:'
        for node in document['nodes']:
            node=copy.deepcopy(node);node['id']=prefix+node['id']
            for member in node['members']: member['tile']=f'{member["year"]}/{name}'
            new_nodes.append(node)
        for edge in document['temporal_edges']:
            edge=copy.deepcopy(edge);edge['source']=prefix+edge['source'];edge['target']=prefix+edge['target'];new_edges.append(edge)
        target=original_targets.copy();target.gid=prefix+target.gid;targets.append(target)
        x=original['x'].copy();x[:,9]=group*100;xs.append(x)
        eis.append(original['edge_index']+group*len(x));eas.append(original['edge_attr']);ets.append(original['edge_type']);ids.extend(prefix+ident for ident in original['node_ids'])
    document['nodes']=new_nodes;document['temporal_edges']=new_edges;atomic_json(artifact.paths['graph'],document)
    np.savez_compressed(artifact.paths['tensors'],x=np.concatenate(xs),edge_index=np.concatenate(eis,axis=1),edge_attr=np.concatenate(eas),edge_type=np.concatenate(ets),node_ids=np.array(ids))
    pd.concat(targets).to_csv(artifact.paths['targets'],index=False)
    result=run_gnn(artifact,context)
    assert result['status']=='trained' and result['folds']==3
    for fold in result['evaluation']['folds']:
        assert not fold['cross_group_messages']
        assert len(fold['train_groups'])==1
        group=names.index(fold['train_groups'][0].split('/')[-1])
        assert fold['normalization']['node_mean'][9]==group*100
