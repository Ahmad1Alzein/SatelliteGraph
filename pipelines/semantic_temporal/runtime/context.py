from pathlib import Path
from .contracts import RunContext
from .inputs import inspect_inputs
from .storage import atomic_json,sha256
from src.pipeline_state import input_fingerprint


def old_configuration(context):
    from src.config import load_config,DEFAULT_CONFIG
    return context.options.get('old_config') or load_config(context.options.get('config') or DEFAULT_CONFIG)


def model_provenance(options,root):
    from src.config import load_config,DEFAULT_CONFIG
    models={}
    def record(name,path,backend):
        path=Path(path).resolve()
        models[name]={'path':str(path),'backend':backend,'status':'available' if path.is_file() else 'missing',
                      'size':path.stat().st_size if path.is_file() else None,
                      'sha256':sha256(path) if path.is_file() else None}
    directory=Path(options.get('semantic_models') or root/'checkpoints/semantic_temporal')
    for name in ('DelineateAnythingv2.pt','satlens-segformer/config.json',
                 'satlens-segformer/preprocessor_config.json','satlens-segformer/model.safetensors'):
        record('new.'+name,directory/name,'delineate_anything' if name.endswith('.pt') else 'segformer')
    config=options.get('old_config')
    if config is None:
        config_path=Path(options.get('config') or DEFAULT_CONFIG)
        if config_path.is_file(): config=load_config(config_path)
        else:
            models['old.configuration']={'path':str(config_path.resolve()),'status':'missing'}
            config={}
    for name in ('instance_model','semantic_model'):
        model=config.get('stage2',{}).get(name,{})
        if model.get('backend')=='fake': continue
        for key in ('architecture','checkpoint'):
            if model.get(key):
                path=Path(model[key]);record(f'old.{name}.{key}',path if path.is_absolute() else root/path,model.get('backend'))
    if options.get('gnn_checkpoint'):
        path=Path(options['gnn_checkpoint']);record('gnn.checkpoint',path,'relational_gnn')
        record('gnn.metadata',path.with_suffix('.json'),'relational_gnn')
    return models


def build_context(options: dict, output: Path) -> RunContext:
    root=Path(options.get('project_root') or Path(__file__).resolve().parents[3])
    years=inspect_inputs(options,root)
    context=RunContext(Path(output).resolve(),years,dict(options),model_provenance(options,root),input_fingerprint(options))
    atomic_json(context.output/'context.json',context)
    return context


def load_context(output,options=None):
    from .storage import read_json
    from .contracts import GridSpec,TileInput,YearInput
    value=read_json(Path(output)/'context.json');years=[]
    for year in value['years']:
        tiles=[]
        for tile in year['tiles']:
            grid=tile['grid'];grid=GridSpec(grid['crs'],tuple(grid['affine']),grid['units'],tuple(grid['physical_pixel_size']) if grid.get('physical_pixel_size') else None)
            tiles.append(TileInput(tile['key'],tile['year'],Path(tile['image']),Path(tile['metadata']) if tile['metadata'] else None,
                tile['source_id'],tuple(tile['offset_xy']),tuple(tile['shape_hw']),grid,tuple(tile['valid_extent_hw'])))
        years.append(YearInput(year['year'],tuple(tiles),Path(year['source_raster']) if year.get('source_raster') else None))
    return RunContext(Path(output).resolve(),tuple(years),{**value['options'],**(options or {})},value['models'],value['fingerprint'])
