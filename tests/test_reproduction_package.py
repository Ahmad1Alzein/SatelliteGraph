from pathlib import Path
import json
import subprocess
import sys


def test_renamed_package_retains_reproduction_interface():
    package = Path(__file__).resolve().parents[1] / 'pipelines' / 'semantic_temporal'
    assert (package / 'run_experiment.py').is_file()
    result = subprocess.run([sys.executable, str(package / 'run_experiment.py'), '--help'], capture_output=True, text=True)
    assert result.returncode == 0
    assert '--resume' in result.stdout
    assert len(json.loads((package/'reference/input_manifest.json').read_text())) == 24
    assert len(json.loads((package/'reference/expected_outputs.json').read_text())) == 551
