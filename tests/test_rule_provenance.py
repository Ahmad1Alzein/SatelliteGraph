"""Synthetic artifact checks; no segmentation model, inference or training runs."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from pipelines.semantic_temporal.runtime.contracts import GridSpec, RunContext
from pipelines.semantic_temporal.runtime.storage import atomic_json, read_json, write_raster
from tests.test_semantic_graphs import make_artifact


def rule_artifact(root, *, key='a', x=500000, support=None, labels=None):
    labels = np.ones((4, 5), np.int32) if labels is None else labels
    classes = np.where(labels > 0, 6, 255).astype(np.uint8)
    artifact = make_artifact(root, labels, classes, key=key, x=x)
    probabilities = np.zeros((9, *labels.shape), np.float32)
    probabilities[6, labels > 0] = .15
    probabilities[3, labels > 0] = .85
    np.save(artifact.paths['probabilities'], probabilities)
    if support is not None:
        artifact.paths['rule_support'] = artifact.directory/'rule_support.tif'
        write_raster(artifact.paths['rule_support'], np.asarray(support, np.float32), artifact.grid)
    atomic_json(artifact.directory/'artifact.json', artifact)
    return artifact


def test_segmentation_persists_rule_support_without_replacing_semantic_confidence(tmp_path, monkeypatch):
    import pipelines.semantic_temporal.runtime.segmentation as segmentation
    from pipelines.semantic_temporal.runtime.adapters import read_band

    grid = GridSpec('EPSG:32636', (1, 0, 500000, 0, -1, 3700000), 'metres', (1, 1))
    tile = SimpleNamespace(year=2017, key='synthetic', grid=grid)
    context = RunContext(tmp_path/'run', (), {}, {}, 'synthetic')
    rgb = np.full((2, 4, 3), 100, np.uint8)
    monkeypatch.setattr(segmentation, 'prepare_tile', lambda *args, **kwargs: (rgb.copy(), np.ones((2, 4), bool), grid))
    labels = np.ones((4, 8), np.int32)
    labels[:, 3:5] = 2
    labels[:, 5:] = 3
    probabilities = np.zeros((9, 4, 8), np.float32)
    probabilities[6] = .1
    probabilities[8] = .2
    probabilities[3] = .7
    support = np.zeros((4, 8), np.float32)
    support[:, :3] = .25
    support[:, 3:5] = .75
    monkeypatch.setattr(segmentation, 'segment', lambda *args, **kwargs: (
        labels.copy(), {1: 6, 2: 8, 3: 3}, {1: 'class_region', 2: 'water', 3: 'class_region'},
        probabilities.copy(), np.full((4, 8, 3), 100, np.uint8), support.copy(), labels.copy(),
        pd.DataFrame(), probabilities.copy(), probabilities.copy()))

    artifact = segmentation.segment_tile(tile, context, models=object())

    assert 'rule_support' in artifact.paths
    np.testing.assert_array_equal(read_band(artifact.paths['rule_support']), support)
    np.testing.assert_array_equal(np.load(artifact.paths['probabilities']), probabilities)
    np.testing.assert_array_equal(read_band(artifact.paths['labels']), labels)
    rows = pd.read_csv(artifact.paths['nodes']).set_index('class_id')
    assert rows.loc[6, 'rgb_rule'] and rows.loc[8, 'rgb_rule']
    assert not rows.loc[3, 'rgb_rule']
    assert rows.loc[6, 'rule_support'] == .25
    assert rows.loc[8, 'rule_support'] == .75
    assert rows.loc[6, 'confidence'] == pytest.approx(.1)
    assert rows.loc[8, 'confidence'] == pytest.approx(.2)


def test_owned_rule_support_reaches_original_node_feature_without_changing_confidence(tmp_path):
    from pipelines.semantic_temporal.runtime.graphs import build_tile_graph, combine_year_graphs
    from pipelines.legacy_building_stgnn.adapter import original_inputs

    context = RunContext(tmp_path/'run', (), {}, {}, 'synthetic')
    first = rule_artifact(tmp_path/'inputs', key='one', support=np.ones((4, 5)))
    second_support = np.ones((4, 5)); second_support[:, :2] = 0
    second = rule_artifact(tmp_path/'inputs', key='two', x=500002, support=second_support)
    graphs = [build_tile_graph(artifact, 'new', context) for artifact in (first, second)]
    for graph in graphs:
        node = read_json(graph.paths['graph'])['nodes'][0]
        assert node['rgb_rule'] is True
        assert node['confidence'] == pytest.approx(.15)

    annual = read_json(combine_year_graphs(graphs, context).paths['graph'])
    assert len(annual['nodes']) == 1
    node = annual['nodes'][0]
    assert node['area'] == 28
    assert node['rgb_rule'] is True
    # The second tile's two unsupported columns lose ownership in the overlap.
    # Averaging both complete input tiles instead would incorrectly produce .8.
    assert node['rule_support'] == 1.
    assert node['confidence'] == pytest.approx(.15)
    x, _ = original_inputs({'nodes': annual['nodes'], 'years': [2017, 2023],
                            'spatial_edges': annual['edges'], 'temporal_edges': []})
    assert x.shape == (1, 38)
    assert x[0, 34].item() == 1.  # Original unscaled rule_based_class feature.


def test_connected_regions_keep_separate_support_and_raw_confidence(tmp_path):
    from pipelines.semantic_temporal.runtime.adapters import connected_region_view, read_band

    labels = np.array([[1, 0, 1], [1, 0, 1]], np.int32)
    support = np.array([[1, 0, 0], [1, 0, 0]], np.float32)
    original = rule_artifact(tmp_path/'inputs', labels=labels, support=support)
    connected = connected_region_view(original)
    np.testing.assert_array_equal(read_band(connected.paths['rule_support']), support)
    rows = pd.read_csv(connected.paths['nodes']).sort_values('segment_id')
    assert rows.rgb_rule.tolist() == [True, True]
    assert rows.rule_support.tolist() == [1., 0.]
    np.testing.assert_allclose(rows.confidence, [.15, .15])


@pytest.mark.parametrize('partial', [False, True])
def test_historical_missing_rule_provenance_stays_unknown(tmp_path, partial):
    from pipelines.semantic_temporal.runtime.graphs import build_tile_graph, combine_year_graphs

    context = RunContext(tmp_path/'run', (), {}, {}, 'synthetic')
    known = rule_artifact(tmp_path/'inputs', key='one', support=np.ones((4, 5)))
    unknown = rule_artifact(tmp_path/'inputs', key='two', x=500002)
    unknown_graph = build_tile_graph(unknown, 'new', context)
    node = read_json(unknown_graph.paths['graph'])['nodes'][0]
    assert node['rgb_rule'] is None
    assert node['rule_support'] is None
    graphs = [build_tile_graph(known, 'new', context), unknown_graph] if partial else [unknown_graph]
    node = read_json(combine_year_graphs(graphs, context).paths['graph'])['nodes'][0]
    assert node['rgb_rule'] is None
    assert node['rule_support'] is None
    assert node['confidence'] == pytest.approx(.15)
