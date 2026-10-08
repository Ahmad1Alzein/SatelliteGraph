"""Run project tests with optional workspace-local test dependencies."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / '.tools' / 'python'))
import pytest

if __name__ == '__main__':
    raise SystemExit(pytest.main(sys.argv[1:]))
