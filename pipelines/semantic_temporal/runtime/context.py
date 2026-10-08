from pathlib import Path
from .contracts import RunContext
from .inputs import inspect_inputs
from .storage import atomic_json
from src.pipeline_state import input_fingerprint


def build_context(options: dict, output: Path) -> RunContext:
    root=Path(options.get('project_root') or Path(__file__).resolve().parents[3])
    years=inspect_inputs(options,root)
    context=RunContext(Path(output).resolve(),years,dict(options),{},input_fingerprint(options))
    atomic_json(context.output/'context.json',context)
    return context
