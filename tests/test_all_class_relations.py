import numpy as np
from shapely.geometry import box, mapping
from tests.test_semantic_temporal import temporal_fixture


def test_context_neighbors_preserve_building_neighbors_and_exclude_unknown():
    from pipelines.semantic_temporal.runtime.graphs import _edges
    nodes = [{'id': str(i), 'class_id': klass, 'geometry': mapping(box(x,0,x+1,1)),
              'position': [x+.5,.5], 'perimeter': 4., 'rgb_mean': [100,100,100]}
             for i,(klass,x) in enumerate([(1,0),(1,20),(2,3),(3,6),(0,9)])]
    edges = _edges(nodes, [], 'new', 'metres')
    by_pair = {(e['source'],e['target']):e['edge_type'] for e in edges}
    assert by_pair['0','1'] == 'near'
    assert by_pair['0','2'] == 'context_near'
    assert by_pair['2','3'] == 'context_near'
    assert not any('4' in pair for pair in by_pair)
    assert len(edges) == len(by_pair)


def test_directional_temporal_features():
    from pipelines.semantic_temporal.runtime.temporal import temporal_edge_features
    row = {'iou': .3,'fraction_source':.4,'fraction_target':.7,
           'shared_area':25.,'confidence':'medium','frame':'roof','roof_shift':[2.,-3.]}
    forward, backward = temporal_edge_features(row, same_class=True)
    assert forward.shape == backward.shape == (10,)
    assert backward[1] == forward[2]
    assert backward[2] == forward[1]
    assert backward[8] == -forward[8]
    assert backward[9] == -forward[9]
    assert forward[4] == 25 and forward[5] == 1 and forward[7] == 1


def test_primary_components_ignore_secondary_chaining():
    from pipelines.semantic_temporal.runtime.temporal import build_primary_components
    nodes = [{'id':i,'year':y} for y,ids in [(2013,['a','b','c']), (2023,['x','y','z'])] for i in ids]
    edges = [{'source':a,'target':b,'edge_role':role} for a,b,role in
             [('a','x','primary'),('a','y','primary'),('b','z','primary'),('c','z','primary'),('a','z','secondary')]]
    components = build_primary_components(nodes,edges,(2013,2023))
    assert {c['kind'] for c in components} == {'split','merge'}
    assert len(components) == 2


def test_role_feature_and_separate_temporal_relations_are_persisted(tmp_path):
    from pipelines.semantic_temporal.runtime.storage import read_json
    artifact,_ = temporal_fixture(tmp_path)
    spec = read_json(artifact.paths['features'])
    assert spec['node_features'][-1] == 'is_later'
    assert len(spec['edge_features']) == 10
    assert spec['relations'] == ['touches','near','context_near','t_primary_fwd','t_primary_bwd','t_overlap_fwd','t_overlap_bwd']
    with np.load(artifact.paths['tensors'],allow_pickle=False) as t:
        assert np.array_equal(t['x'][:2,-1], [0,0])
        assert np.array_equal(t['x'][2:,-1], [1,1])
        assert {3,4} <= set(t['edge_type'])
        assert t['edge_attr'].shape[1] == 10
    assert artifact.paths['components'].is_file()


def test_change_edges_are_not_model_inputs(tmp_path):
    from pipelines.semantic_temporal.runtime.storage import read_json
    artifact,_ = temporal_fixture(tmp_path)
    input_relations = read_json(artifact.paths['features'])['relations']
    assert 'change_conversion' not in input_relations
