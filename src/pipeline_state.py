"""Hash-bound run state, independent of either scientific pipeline."""
from pathlib import Path
import hashlib
import json


def stage_fingerprint(inputs: list[Path], settings: dict) -> str:
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode())
    for path in sorted(set(inputs), key=str):
        digest.update(str(path.resolve()).encode())
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def input_fingerprint(options: dict) -> str:
    keys = ('folders', 'images', 'years', 'data', 'metadata', 'source_rasters',
            'checkpoints', 'config', 'pixel_size', 'processing_scale', 'target_resolution')
    settings = {k: options[k] for k in keys if options.get(k) is not None}
    files = []
    for key, value in settings.items():
        if key == 'years' or isinstance(value, dict) or key in ('pixel_size', 'processing_scale', 'target_resolution'):
            continue
        for item in value if isinstance(value, list) else [value]:
            if item is None: continue
            p = Path(item)
            if p.is_file(): files.append(p)
            elif p.is_dir():
                for f in p.rglob('*'):
                    if (f.is_file() and not any(part.startswith('.') for part in f.relative_to(p).parts)
                            and f.suffix.lower() in ('.png', '.tif', '.tiff', '.csv', '.yaml', '.pt', '.pth', '.safetensors')):
                        files.append(f)
                    elif f.is_file() and f.name in ('config.json', 'preprocessor_config.json'):
                        files.append(f)
    return stage_fingerprint(files, settings)


def validate_resume(state: dict, stage: str, fingerprint: str) -> None:
    if state.get('input_fingerprint') not in (None, fingerprint):
        raise ValueError('Inputs or scientific settings changed; choose a new output folder to recompute dependent stages')


def record_stage(output: Path, stage: str, status: str, **details) -> dict:
    path = output / 'run.json'
    state = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    state.setdefault('stages', {}).setdefault(stage, {}).update(status=status, **details)
    output.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state, indent=2, default=str), encoding='utf-8')
    temporary.replace(path)
    return state
