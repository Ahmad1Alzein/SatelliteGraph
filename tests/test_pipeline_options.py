"""Omitted CLI flags must inherit the effective saved run configuration."""
import json
from pathlib import Path

from src import cli, pipeline_select as ps


def test_separate_cli_stages_inherit_inputs_and_flags(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ps, '_execute_stage', lambda out, stage, method, opts, state: {'effective': opts})
    output = tmp_path / 'run'
    cli.main(['pipeline', 'segment', '--method', 'new', '--output', str(output),
              '--project-root', str(tmp_path), '--data', 'données with spaces',
              '--checkpoints', 'models', '--config', 'custom.yaml',
              '--skip-input-check', '--no-progress'])
    cli.main(['pipeline', 'graph', '--method', 'new', '--output', str(output),
              '--project-root', str(tmp_path)])
    cli.main(['pipeline', 'change', '--method', 'classical', '--output', str(output),
              '--project-root', str(tmp_path)])
    state = json.loads((output / 'run.json').read_text())
    for phase in ('segmentation', 'graph', 'change'):
        received = state['stages'][phase]['result']['effective']
        assert received['data'] == str(tmp_path / 'données with spaces')
        assert received['checkpoints'] == str(tmp_path / 'models')
        assert received['config'] == str(tmp_path / 'custom.yaml')
        assert received['skip_input_check'] is True
        assert received['no_progress'] is True


def test_explicit_inverse_flags_override_saved_values(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ps, '_execute_stage', lambda out, stage, method, opts, state: {'effective': opts})
    cli.main(['pipeline', 'segment', '--method', 'new', '--output', str(tmp_path),
              '--skip-input-check', '--no-progress'])
    cli.main(['pipeline', 'graph', '--method', 'new', '--output', str(tmp_path),
              '--no-skip-input-check', '--progress'])
    received = json.loads((tmp_path / 'run.json').read_text())['options']
    assert received['skip_input_check'] is False
    assert received['no_progress'] is False


def test_registration_override_clears_inherited_alternative(tmp_path):
    options=ps.resolve_options({'register_images':True},{'registration_json':tmp_path/'controls.json'},initial=False,project_root=tmp_path)
    assert options['register_images'] is False
    assert options['registration_json']==str(tmp_path/'controls.json')
