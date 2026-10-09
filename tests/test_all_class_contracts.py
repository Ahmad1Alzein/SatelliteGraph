"""Public all-class policy validation and readable report contracts."""
import json
import pytest


def contracts():
    from pipelines.semantic_temporal.runtime import change_contracts
    return change_contracts


def test_registry_covers_every_class_and_direction():
    c = contracts()
    assert len(c.TASKS) == 16
    assert c.TASKS[0] == 'building_gain'
    assert c.TASKS[-1] == 'water_loss'
    assert set(c.KNOWN_CLASSES) == set(range(1, 9))
    assert len(c.CONVERSIONS) == 56
    assert (3, 1) in c.CONVERSIONS and (1, 3) in c.CONVERSIONS
    assert (0, 1) not in c.CONVERSIONS


@pytest.mark.parametrize('override', [
    {'presence_probability': 1.1}, {'absence_probability': float('nan')},
    {'minimum_area_m2': {'forest': 10}}, {'minimum_area_m2': {'road': -1}},
    {'mystery': 42}, {'shadow_cutoff': 256}, {'presence_probability': True},
    {'minimum_area_m2': []}, {'absence_window_m': 'wide'},
    {'presence_probability': .1, 'absence_probability': .2},
])
def test_invalid_policy_is_rejected(tmp_path, override):
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps(override))
    with pytest.raises(ValueError):
        contracts().load_change_policy(path)


def test_partial_policy_override_keeps_other_class_defaults(tmp_path):
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps({'minimum_area_m2': {'road': 120}}))
    resolved = contracts().load_change_policy(path)
    assert resolved['minimum_area_m2']['road'] == 120
    assert resolved['minimum_area_m2']['tree'] == 150
    assert contracts().load_change_policy(None)['minimum_area_m2']['road'] == 100


@pytest.mark.parametrize('event, expected', [
    ({'kind': 'appearance', 'class_id': 1}, 'New building'),
    ({'kind': 'expansion', 'class_id': 1}, 'Building grew'),
    ({'kind': 'expansion', 'class_id': 2}, 'Road got wider'),
    ({'kind': 'loss', 'class_id': 3}, 'Less tree cover'),
    ({'kind': 'conversion', 'source_class': 3, 'target_class': 1}, 'Trees → Buildings'),
    ({'kind': 'reconfiguration', 'class_id': 8}, 'Water: Changed shape'),
    ({'kind': 'split'}, 'One area split'),
])
def test_change_names_explain_class_and_change(event, expected):
    assert contracts().change_label(event) == expected


def test_model_status_is_not_a_physical_change_name():
    c = contracts()
    assert c.status_label('insufficient_data') == 'Not enough training examples'
    assert c.status_label('training_only') == 'Trained on this area; not independently tested'
    assert c.EDGE_LABELS['t_primary_bwd'] == 'Main match across years'
