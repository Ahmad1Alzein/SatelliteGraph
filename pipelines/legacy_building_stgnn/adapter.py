"""Adapt general pipeline artifacts to the unmodified thesis building ST-GNN.

The frozen reproduction runner and its data are never used as output folders.
New imagery is a new experiment; its training-only predictions are not the
published leave-one-tile-out results. No experimental multi-head model is used.
"""
import ast
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd
import torch
from shapely.geometry import shape
from shapely.ops import unary_union

from . import runner
from pipelines.semantic_temporal.runtime.storage import atomic_json, read_json, save_manifest

MODEL_SOURCE = 'pipelines/legacy_building_stgnn/original/gnn_change.py'


@lru_cache(maxsize=1)
def _feature_function():
    # build_graph.py imports GDAL for exports. Its pure feature function does
    # not use GDAL: load that exact function without requiring the export stack.
    path = runner.ORIGINAL / 'build_graph.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'node_features')
    namespace = {'np': np, 'pd': pd, 'CLASSES': {
        1: 'building', 2: 'road', 3: 'tree', 4: 'grass_shrub',
        5: 'cropland', 6: 'bare_soil', 7: 'paved_developed', 8: 'water'}}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['node_features']


def original_inputs(document):
    """Original feature order, scaling and relations, for either spatial graph."""
    from .original import gnn_change as original
    nodes = document['nodes']
    if not nodes:
        return torch.empty((0, 38)), {}
    bounds = unary_union([shape(n['geometry']) for n in nodes]).bounds
    # Nonempty polygons give nonzero extent in both axes.
    rows = []
    for i, n in enumerate(nodes):
        hsv = n.get('hsv_mean', [0., 0., 0.])
        row = dict(segment_id=i, class_id=n['class_id'], area_m2=max(n['area'], 1e-9),
            perimeter_m=max(n['perimeter'], 1e-9), confidence=n['confidence'],
            rgb_rule=bool(n.get('rgb_rule', False)),
            node_x=n['position'][0], node_y=n['position'][1],
            compactness=n.get('compactness', 0.), solidity=n.get('solidity', 0.),
            elongation=n.get('elongation', 1.), orientation_deg=n.get('orientation_deg', 0.),
            mean_h=hsv[0], mean_s=hsv[1], mean_v=hsv[2])
        for prefix, values in [('mean', n.get('rgb_mean', [0.]*3)), ('std', n.get('rgb_std', [0.]*3))]:
            row.update({prefix+'_'+c: v for c, v in zip('rgb', values)})
        for klass, name in _feature_function().__globals__['CLASSES'].items():
            row['p_'+name] = n['probabilities'][klass]
        rows.append(row)
    features = _feature_function()(pd.DataFrame(rows), bounds).drop(columns='segment_id')
    x = features.to_numpy(dtype=np.float32)
    continuous = [i for i, name in enumerate(features) if not name.startswith('class_') and name != 'rule_based_class']
    x[:, continuous] = (x[:, continuous]-x[:, continuous].mean(0))/(x[:, continuous].std(0)+1e-6)
    x = np.column_stack([x, [float(n['year'] == document['years'][-1]) for n in nodes]]).astype(np.float32)
    index = {n['id']: i for i, n in enumerate(nodes)}
    spatial = ['shared_boundary_m', 'gap_m', 'distance_m', 'boundary_frac_source',
        'boundary_frac_target', 'color_contrast', 'same_class', 'dir_sin', 'dir_cos']
    temporal = ['iou', 'frac_2013', 'frac_2023', 'overlap_score', 'shared_m2',
        'class_agreement', 'confidence_ord', 'frame_roof', 'roof_shift_dx_m', 'roof_shift_dy_m']
    edges = []
    for e in document.get('spatial_edges', []):
        # context_near belongs to the retired enhanced model, not this model.
        if e['edge_type'] not in ('touches', 'near'):
            continue
        names = ['shared_boundary', 'gap', 'distance', *spatial[3:]]
        edges.append(dict(src=index[e['source']], dst=index[e['target']],
            relation='spatial_'+e['edge_type'], **{a: e.get(b, 0.) for a, b in zip(spatial, names)}))
    for e in document.get('temporal_edges', []):
        # Old graphs call accepted links correspondence/split/merge. Their
        # primary role must reach the same original channels as new identity
        # links, rather than being silently downgraded to contextual overlap.
        rel = ('contained_in' if e['relation'] == 'contained_in' else 'identity') if e.get('edge_role') == 'primary' else 'overlap'
        shift = e.get('roof_shift', [0., 0.])
        a, b = nodes[index[e['source']]], nodes[index[e['target']]]
        edges.append(dict(src=index[e['source']], dst=index[e['target']], relation='temporal_'+rel,
            iou=e.get('iou', 0.), frac_2013=e.get('fraction_source', 0.),
            frac_2023=e.get('fraction_target', 0.), overlap_score=e.get('overlap_score', 0.),
            shared_m2=e.get('shared_area', 0.), class_agreement=float(a['class_id'] == b['class_id']),
            confidence_ord={'high': 2., 'medium': 1., 'low': 0.}.get(e.get('confidence'), 1.),
            frame_roof=float(e.get('frame') == 'roof'), roof_shift_dx_m=shift[0], roof_shift_dy_m=shift[1]))
    table = pd.DataFrame(edges, columns=['src', 'dst', 'relation', *spatial, *temporal])
    table = table.astype({'src':'int64','dst':'int64'})
    # Original relations keeps empty relation tensors; numpy warns when computing
    # their unused statistics. Preserve those tensors and suppress only warnings.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        relations = original.relations(table, 'gnn_full')
    return torch.from_numpy(x), relations


def building_targets(document, catalog):
    """Keep the original roles: later buildings and earlier land built over."""
    nodes = document['nodes']; first, last = document['years']
    changes = {r['gid']: r for r in catalog.get('node_changes', [])}
    positive_buildings = [shape(n['geometry']) for n in nodes if n['year'] == last
        and n['class_id'] == 1 and changes.get(n['id'], {}).get('label') == 1
        and changes.get(n['id'], {}).get('eligible', False)]
    built = unary_union(positive_buildings)
    labels = np.full(len(nodes), -1, np.int64); tasks = []
    for i, n in enumerate(nodes):
        applicable = (n['year'] == last and n['class_id'] == 1) or (n['year'] == first and n['class_id'] in range(2, 9))
        tasks.append('building_gain' if n['year'] == last else 'built_over')
        if not applicable or n['observed_fraction'] < .95 or n['known_counterpart_fraction'] < .95 or n['confidence'] < .6:
            continue
        change = changes.get(n['id'], {})
        if not change.get('eligible', False):
            continue
        if n['year'] == last and change.get('label') in (0, 1):
            labels[i] = int(change['label'])
        elif n['year'] == first:
            geometry = shape(n['geometry'])
            if geometry.intersection(built).area >= .5*geometry.area:
                labels[i] = 1
            elif change.get('label') == 0:
                labels[i] = 0
    return labels, tasks


def run_gnn(graph, context, *, checkpoint=None):
    """Fit the original binary ST-GNN on current data without touching the study."""
    if checkpoint:
        raise ValueError('The thesis ST-GNN has no saved checkpoint; experimental RGNN checkpoints were retired. Use a fresh run without --gnn-checkpoint.')
    if runner.check_code():
        raise ValueError('Original thesis GNN code failed its frozen integrity check')
    from .original import gnn_change as original
    document = read_json(graph.paths['graph'])
    catalog = read_json(graph.directory/'classical_changes.json')
    labels, tasks = building_targets(document, catalog)
    positive = int((labels == 1).sum()); negative = int((labels == 0).sum())
    epochs = int(context.options.get('gnn_epochs') or 300)
    if epochs < 1:
        raise ValueError('GNN epochs must be positive')
    # The original early-stopping implementation selects a state every 5 epochs.
    epochs = max(5, epochs)
    result = dict(model_source=MODEL_SOURCE, model='Original thesis building ST-GNN',
        positive_targets=positive, negative_targets=negative, checkpoint=None,
        audit_status='not_provided', epochs=epochs, hidden=64, threshold=.5,
        evaluation={'status': 'not_independently_evaluated', 'independent_human_accuracy': False,
            'reason': 'New-input fit; frozen leave-one-tile-out results are a separate experiment.'})
    probability = np.full(len(labels), np.nan)
    if positive and negative:
        x, relations = original_inputs(document)
        # With no independent validation set, the original function otherwise
        # may select no state at all (its initial best score is -1). Select and
        # threshold on the same labeled input, explicitly training only.
        probability, threshold, history = original.train_fold(x, relations, labels,
            labels >= 0, labels >= 0, epochs=epochs)
        result.update(status='training_only', threshold=float(threshold),
            threshold_source='same-input rule labels; training only',
            reason='Original binary ST-GNN fitted and scored on this input. No held-out accuracy is claimed.')
    else:
        history = []
        result.update(status='insufficient_data', reason='Original ST-GNN needs reliable changed and unchanged building/built-over targets.')
    available = bool(positive and negative)
    result['head_status'] = {task: dict(available=available, status=result['status'],
        positive_targets=positive, negative_targets=negative, threshold=result['threshold'],
        reason=result['reason']) for task in ('building_gain', 'built_over')}
    rows = [dict(gid=n['id'], task=task, label=int(label),
        probability=float(probability[i]) if label >= 0 and available else None,
        prediction=int(probability[i] >= result['threshold']) if label >= 0 and available else -1,
        eligible=bool(label >= 0), status=result['status'], probability_source='training_only')
        for i, (n, task, label) in enumerate(zip(document['nodes'], tasks, labels))
        if (n['year'] == document['years'][-1] and n['class_id'] == 1)
        or (n['year'] == document['years'][0] and n['class_id'] in range(2, 9))]
    directory = graph.directory/'gnn'; directory.mkdir(exist_ok=True)
    predictions = directory/'predictions.csv'
    pd.DataFrame(rows, columns=['gid', 'task', 'label', 'probability', 'prediction', 'eligible', 'status', 'probability_source']).to_csv(predictions, index=False)
    result['predictions'] = str(predictions)
    atomic_json(directory/'history.json', [{k: (None if isinstance(v, float) and not np.isfinite(v) else v)
        for k, v in entry.items()} for entry in history])
    atomic_json(directory/'report.json', result)
    paths = {'predictions': predictions, 'report': directory/'report.json', 'history': directory/'history.json'}
    atomic_json(directory/'artifact.json', {'kind': 'gnn', 'directory': directory, 'paths': paths})
    save_manifest(directory, paths, model_source=MODEL_SOURCE)
    return result
