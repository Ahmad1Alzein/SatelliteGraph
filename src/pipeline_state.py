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
            'checkpoints', 'semantic_models','config','old_config','pixel_size','processing_scale','target_resolution',
            'target_crs','registration_json','register_images','inference_window','inference_overlap','tta')
    settings = {k: options[k] for k in keys if options.get(k) is not None}
    if options.get('harmonize_images'):
        settings['harmonize_images'] = True
        settings['preprocessing_profile'] = 'paired-harmonization.v1'
    files = []
    # Selected PNGs still consume their discovered placement and original raster.
    for index, image in enumerate(options.get('images') or []):
        p=Path(image)
        if p.suffix.lower()!='.png': continue
        metadata=options.get('metadata') or []
        rasters=options.get('source_rasters') or []
        if index>=len(metadata) or metadata[index] is None:
            if (p.parent/'metadata.csv').is_file(): files.append(p.parent/'metadata.csv')
        if index>=len(rasters) or rasters[index] is None:
            files.extend(f for f in p.parent.iterdir() if f.is_file() and f.suffix.lower() in ('.tif','.tiff'))
    # The old backends load assets referenced in configuration, outside the new
    # semantic model directory. Bind those bytes as well as the YAML itself.
    config=options.get('old_config')
    config_path=options.get('config')
    if config is None and config_path and Path(config_path).is_file():
        import yaml
        config=yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    if isinstance(config,dict):
        root=Path(options.get('project_root') or Path(__file__).resolve().parents[1])
        for name in ('instance_model','semantic_model'):
            model=config.get('stage2',{}).get(name,{})
            if model.get('backend')=='fake': continue
            for key in ('architecture','checkpoint'):
                if model.get(key):
                    p=Path(model[key]);p=p if p.is_absolute() else root/p
                    if p.is_file(): files.append(p)
    for key, value in settings.items():
        if key == 'years' or isinstance(value, dict) or key in ('pixel_size','processing_scale','target_resolution','target_crs','register_images','inference_window','inference_overlap','tta','harmonize_images','preprocessing_profile'):
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


def change_fingerprint(options: dict) -> str:
    settings={k:options.get(k) for k in ('gnn_checkpoint','gnn_epochs','gnn_hidden','boundary_tolerance')}
    files=[]
    if options.get('gnn_checkpoint'):
        checkpoint=Path(options['gnn_checkpoint'])
        for path in (checkpoint,checkpoint.with_suffix('.json')):
            settings[str(path)]=path.is_file()
            if path.is_file(): files.append(path)
    return stage_fingerprint(files,settings)


def record_stage(output: Path, stage: str, status: str, **details) -> dict:
    path = output / 'run.json'
    state = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    state.setdefault('stages', {}).setdefault(stage, {}).update(status=status, **details)
    output.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state, indent=2, default=str), encoding='utf-8')
    temporary.replace(path)
    return state
