"""Partial historical exports must remain readable without claiming verification."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from streamlit.util import calc_md5

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def archived_run(tmp_path, monkeypatch):
    run = tmp_path / 'runs' / 'tile_00001_geo'
    (run / 'temporal').mkdir(parents=True)
    state = {'output': r'D:\old_project\artifacts\tile_00001_geo', 'stages': {
        'segmentation': {'method': 'new'}, 'graph': {'method': 'new'},
        'change': {'method': 'gnn', 'render_status': 'complete', 'result': {
            'artifacts': [r'D:\old_project\artifacts\tile_00001_geo\temporal\artifact.json']}}}}
    event = dict(event_id='historical-conversion', source_class=4, target_class=1,
                 kind='conversion', status='candidate', season_sensitive=False,
                 area=9, area_units='m2', related_tasks=['building_gain'],
                 source_nodes=['old-1'], target_nodes=['new-1'],
                 geometry={'type': 'Polygon', 'coordinates': [[[0, 0], [3, 0], [3, 3], [0, 3], [0, 0]]]})
    (run / 'run.json').write_text(json.dumps(state), encoding='utf-8')
    (run / 'temporal/classical_changes.json').write_text(json.dumps({'events': [event]}), encoding='utf-8')
    monkeypatch.setenv('SG_RUNS', str(run.parent))
    monkeypatch.setenv('SG_LEGACY_RUN', str(tmp_path / 'no_frozen_run'))
    return run


@pytest.mark.parametrize('page', ['changes', 'pipelines'])
def test_partial_historical_run_shows_events_without_missing_manifest_error(archived_run, page):
    before = {p: p.read_bytes() for p in archived_run.rglob('*') if p.is_file()}
    app = AppTest.from_file(str(ROOT / 'thesis_dashboard.py'))
    app._page_hash = calc_md5('image_comparison' if page == 'pipelines' else page)
    app.session_state['selected_run'] = str(archived_run)
    app.run(timeout=90)
    assert not app.exception, [e.value for e in app.exception]
    assert not app.error, [e.value for e in app.error]
    assert any('historical-conversion' in df.value.event_id.values
               for df in app.dataframe if 'event_id' in df.value)
    assert any('not verified' in info.value.lower() for info in app.info)
    assert any('retired experimental' in warning.value.lower() for warning in app.warning)
    assert before == {p: p.read_bytes() for p in archived_run.rglob('*') if p.is_file()}


def test_missing_manifest_in_current_run_still_blocks_report_display(archived_run):
    from dashboard.run_results import change_view_status
    (archived_run / 'results').mkdir()
    state = json.loads((archived_run / 'run.json').read_text())
    with pytest.raises(ValueError, match='manifest.*missing'):
        change_view_status(archived_run, state)


def test_changed_report_is_not_treated_as_a_historical_export(archived_run):
    from dashboard.run_results import change_view_status
    from pipelines.semantic_temporal.runtime.storage import bind_result_manifest
    report = archived_run / 'results/change_report.png'
    report.parent.mkdir()
    report.write_bytes(b'original report')
    bind_result_manifest(archived_run, 'change', {'report': report}, [])
    state = json.loads((archived_run / 'run.json').read_text())
    assert change_view_status(archived_run, state) == 'verified'
    report.write_bytes(b'changed report')
    with pytest.raises(ValueError, match='changed or missing'):
        change_view_status(archived_run, state)
