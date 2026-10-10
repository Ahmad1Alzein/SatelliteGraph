"""Versioned change policy, task order, and user-facing terminology."""
from copy import deepcopy
import json
import math
from pathlib import Path

from .contracts import CLASSES

CHANGE_SCHEMA = 'satellitegraph.all-class-change.v1'
RELATION_PROFILE = 'directional-all-class.v1'
KNOWN_CLASSES = {k: v for k, v in CLASSES.items() if 1 <= k <= 8}
TASKS = tuple(f'{name}_{direction}' for name in KNOWN_CLASSES.values()
              for direction in ('gain', 'loss'))
CONVERSIONS = tuple((a, b) for a in KNOWN_CLASSES for b in KNOWN_CLASSES if a != b)
RELATIONS = ('touches', 'near', 'context_near', 't_primary_fwd', 't_primary_bwd',
             't_overlap_fwd', 't_overlap_bwd')
CLASS_LABELS = {0: 'Unknown', 1: 'Buildings', 2: 'Roads', 3: 'Trees',
                4: 'Grass/shrubs', 5: 'Cropland', 6: 'Bare ground',
                7: 'Paved areas', 8: 'Water'}
EDGE_LABELS = {'touches': 'Share a border', 'near': 'Nearby buildings',
               'context_near': 'Nearby areas',
               't_primary_fwd': 'Main match across years',
               't_primary_bwd': 'Main match across years',
               't_overlap_fwd': 'Partial overlap across years',
               't_overlap_bwd': 'Partial overlap across years',
               'change_conversion': 'Class change'}
DEFAULT_POLICY = {
    'presence_probability': .6, 'building_presence_probability': .5,
    'absence_probability': .2, 'absence_window_m': 3.75, 'shadow_cutoff': 20,
    'reliable_confidence': .6, 'observed_fraction': .95, 'known_fraction': .95,
    'positive_fraction': .3, 'stable_fraction': .95, 'building_reliable_area_m2': 100.,
    'minimum_area_m2': dict(zip(KNOWN_CLASSES.values(), (40., 100., 150., 150., 150., 100., 100., 100.))),
}


def load_change_policy(path: Path | None) -> dict:
    """Resolve partial overrides without mutating defaults; reject ambiguous units."""
    policy = deepcopy(DEFAULT_POLICY)
    if path is not None:
        try:
            supplied = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        except (json.JSONDecodeError, UnicodeError) as exc:
            raise ValueError(f'Invalid change policy: {exc}') from exc
        if not isinstance(supplied, dict) or supplied.keys() - policy.keys():
            raise ValueError('Change policy must be an object with recognized threshold names')
        for key, value in supplied.items():
            if key == 'minimum_area_m2':
                if not isinstance(value, dict) or value.keys() - policy[key].keys():
                    raise ValueError('minimum_area_m2 must use known class names')
                policy[key].update(value)
            else:
                policy[key] = value
    probabilities = {'presence_probability', 'building_presence_probability',
                     'absence_probability', 'reliable_confidence', 'observed_fraction',
                     'known_fraction', 'positive_fraction', 'stable_fraction'}
    def finite(value, key, low, high=None):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < low
                or (high is not None and value > high)):
            raise ValueError(f'Invalid change policy {key}: expected finite value in [{low}, {high}]')
    for key, value in policy.items():
        if key == 'minimum_area_m2':
            for name, area in value.items():
                finite(area, f'{key}.{name}', 0)
        else:
            finite(value, key, 0, 1 if key in probabilities else 255 if key == 'shadow_cutoff' else None)
    if (policy['absence_probability'] >= policy['presence_probability']
            or policy['absence_probability'] >= policy['building_presence_probability']):
        raise ValueError('Absence probability must be lower than presence probabilities')
    if policy['absence_window_m'] <= 0:
        raise ValueError('absence_window_m must be positive')
    return policy


def change_label(event: dict) -> str:
    kind = event.get('kind', 'uncertain')
    if kind in ('conversion', 'class_conversion'):
        return f"{CLASS_LABELS[int(event['source_class'])]} \u2192 {CLASS_LABELS[int(event['target_class'])]}"
    structural = {'split': 'One area split', 'merge': 'Areas joined',
                  'many_to_many': 'Complex match', 'connectivity': 'Connections changed',
                  'neighborhood': 'Nearby areas changed'}
    if kind in structural:
        return structural[kind]
    kind = {'gain': 'appearance', 'loss': 'disappearance', 'persistent': 'unchanged',
            'unassessed': 'cannot_assess'}.get(kind, kind)
    klass = int(event.get('class_id', event.get('target_class', 0)))
    specific = {
        1: {'appearance': 'New building', 'disappearance': 'Building disappeared',
            'expansion': 'Building grew', 'contraction': 'Building shrank'},
        2: {'appearance': 'Road added', 'disappearance': 'Road removed',
            'expansion': 'Road got wider', 'contraction': 'Road got narrower'},
        3: {'appearance': 'More tree cover', 'disappearance': 'Less tree cover',
            'expansion': 'More tree cover', 'contraction': 'Less tree cover'},
        8: {'appearance': 'Water appeared', 'disappearance': 'Water disappeared',
            'expansion': 'Water expanded', 'contraction': 'Water shrank'},
    }
    if kind in specific.get(klass, {}):
        return specific[klass][kind]
    generic = {'appearance': 'Appeared', 'disappearance': 'Disappeared',
               'expansion': 'Expanded', 'contraction': 'Shrank',
               'reconfiguration': 'Changed shape', 'unchanged': 'Unchanged',
               'uncertain': 'Needs review', 'cannot_assess': 'Cannot assess'}
    return f"{CLASS_LABELS.get(klass, 'Unknown')}: {generic.get(kind, 'Needs review')}"


def status_label(status: str) -> str:
    return {'reliable': 'Strong image evidence', 'candidate': 'Candidate',
            'predicted': 'GNN prediction', 'uncertain': 'Needs review',
            'unassessed': 'Cannot assess', 'insufficient_data': 'Not enough training examples',
            'training_only': 'Trained on this area; not independently tested',
            'season_sensitive': 'Season may explain this', 'complete': 'Ready',
            'ineligible': 'Cannot assess', 'inference': 'Saved model prediction',
            'trained': 'Trained; see evaluation',
            'supervised_external': 'Supervised external model',
            'insufficient_validation': 'Not enough validated examples'}.get(status, 'Needs review')
