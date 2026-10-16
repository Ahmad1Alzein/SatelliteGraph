"""Synthetic adapter contracts; no training, inference or pipeline execution."""
from unittest.mock import Mock

import numpy as np
import pytest
from shapely.geometry import box, mapping

from pipelines.legacy_building_stgnn import adapter
from pipelines.legacy_building_stgnn.original import gnn_change as original


def _document(edges):
    nodes = [dict(id=name, year=year, class_id=1, geometry=mapping(box(x, 0, x+10, 10)),
                  area=100., perimeter=40., confidence=.8, position=[x+5, 5],
                  probabilities=[0., 1., 0., 0., 0., 0., 0., 0., 0.])
             for name, year, x in [('before', 2013, 0), ('after', 2023, 1)]]
    return dict(years=[2013, 2023], nodes=nodes, spatial_edges=[], temporal_edges=edges)


@pytest.mark.parametrize('fractions,explicit,expected', [
    ({'fraction_source': .25, 'fraction_target': 1.}, None, .5),
    ({'fraction_source': .25, 'fraction_target': 1.}, 0., 0.),
    ({'fraction_source': .25, 'fraction_target': 1.}, .7, .7),
    ({'fraction_source': -.25, 'fraction_target': 1.}, None, 0.),
    ({'fraction_source': .25}, None, 0.),
])
def test_temporal_overlap_fallback_reaches_original_relations(monkeypatch, fractions, explicit, expected):
    edge = dict(source='before', target='after', relation='identity', edge_role='primary',
                iou=.2, shared_area=25., confidence='high', frame='roof',
                roof_shift=[1.5, -2.], **fractions)
    if explicit is not None:
        edge['overlap_score'] = explicit
    captured = []
    relations = original.relations

    def capture(table, mode):
        captured.append((table.copy(deep=True), mode))
        return relations(table, mode)

    monkeypatch.setattr(original, 'relations', capture)
    features, mapped = adapter.original_inputs(_document([edge]))

    table, mode = captured[0]
    row = table.iloc[0]
    assert row.overlap_score == pytest.approx(expected)
    assert mode == 'gnn_full'
    assert row.relation == 'temporal_identity'
    assert row.frac_2013 == fractions.get('fraction_source', 0.)
    assert row.frac_2023 == fractions.get('fraction_target', 0.)
    assert row.shared_m2 == 25. and row.confidence_ord == 2.
    assert row.frame_roof == 1. and row.roof_shift_dx_m == 1.5 and row.roof_shift_dy_m == -2.
    assert features.shape == (2, 38)
    assert set(mapped) == {'spatial_touches', 'spatial_near', 't_primary_fwd',
                           't_primary_bwd', 't_overlap_fwd', 't_overlap_bwd'}
    np.testing.assert_array_equal(mapped['t_primary_fwd'][0].numpy(), [[0], [1]])
    np.testing.assert_array_equal(mapped['t_primary_bwd'][0].numpy(), [[1], [0]])
    assert mapped['t_primary_fwd'][1].shape == (1, 10)


def test_missing_overlap_scores_keep_original_relation_attribute_variation():
    edges = [dict(source='before', target='after', relation='identity', edge_role='primary',
                  fraction_source=fraction, fraction_target=fraction)
             for fraction in (.2, .4, .6)]

    _, relations = adapter.original_inputs(_document(edges))

    # Original relations normalizes each attribute. Derived scores must reach
    # that function as varying values, rather than an all-zero missing column.
    overlap = relations['t_primary_fwd'][1][:, 3].numpy()
    np.testing.assert_allclose(overlap, [-1.22474, 0., 1.22474], atol=1e-5)
    np.testing.assert_array_equal(overlap, relations['t_primary_bwd'][1][:, 3].numpy())


@pytest.mark.parametrize('class_id,rgb_rule,support,expected', [
    (6, True, .9, .9),
    (8, True, .7, .7),
    (6, True, 0., 0.),
    (3, True, .9, .15),
    (6, False, .9, .15),
    (6, None, .9, .15),
    (6, True, 'missing', .15),
    (6, True, None, .15),
    (6, True, float('nan'), .15),
    (6, True, float('inf'), .15),
    (6, True, -.1, .15),
    (6, True, 1.1, .15),
])
def test_original_confidence_feature_uses_only_known_valid_rgb_rule_support(
        monkeypatch, class_id, rgb_rule, support, expected):
    document = _document([])
    node = document['nodes'][0]
    node.update(class_id=class_id, confidence=.15, rgb_rule=rgb_rule)
    if support != 'missing':
        node['rule_support'] = support
    original_features = adapter._feature_function()
    feature_spy = Mock(wraps=original_features)
    feature_spy.__globals__ = original_features.__globals__
    monkeypatch.setattr(adapter, '_feature_function', lambda: feature_spy)

    features, _ = adapter.original_inputs(document)

    feature_nodes = feature_spy.call_args.args[0]
    assert feature_nodes.iloc[0].confidence == pytest.approx(expected)
    assert feature_nodes.iloc[1].confidence == .8
    assert features.shape == (2, 38)
    assert node['confidence'] == .15
    assert node['probabilities'] == [0., 1., 0., 0., 0., 0., 0., 0., 0.]


def test_original_feature_rule_support_does_not_change_semantic_target_gate():
    document = _document([])
    node = document['nodes'][0]
    node.update(class_id=6, confidence=.15, rgb_rule=True, rule_support=1.,
                observed_fraction=1., known_counterpart_fraction=1.)
    document['nodes'][1].update(observed_fraction=1., known_counterpart_fraction=1.)
    catalog = {'node_changes': [dict(gid=n['id'], label=0, eligible=True)
                                for n in document['nodes']]}

    adapter.original_inputs(document)
    labels, _ = adapter.building_targets(document, catalog)

    assert labels.tolist() == [-1, 0]
    assert node['confidence'] == .15
