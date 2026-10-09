"""Native general stages; published reproduction has a separate runner."""
from pathlib import Path
from .context import build_context
from .registration import prepare_comparison
from .storage import read_json,atomic_json,load_segmentation,verify_manifest,sha256
from .contracts import GraphArtifact,GridSpec


def _graph(path):
    v=read_json(path);g=v['grid']
    grid=GridSpec(g['crs'],tuple(g['affine']),g['units'],tuple(g['physical_pixel_size']) if g.get('physical_pixel_size') else None)
    return GraphArtifact(v['year'],v['method'],Path(v['directory']),grid,{k:Path(p) for k,p in v['paths'].items()})


def relocate_run(output,state):
    old=state.get('output')
    if not old or Path(old).resolve()==output: return state
    old=Path(old);manifests=[p for p in output.rglob('manifest.json') if 'files' in read_json(p)]
    for manifest in manifests: verify_manifest(manifest.parent)
    def relocated(v):
        if isinstance(v,dict): return {k:relocated(value) for k,value in v.items()}
        if isinstance(v,list): return [relocated(value) for value in v]
        if isinstance(v,str):
            try:
                p=Path(v)
                if p.is_absolute() and p.is_relative_to(old): return str(output/p.relative_to(old))
            except (ValueError,OSError): pass
        return v
    for path in output.rglob('*.json'):
        value=read_json(path);updated=relocated(value)
        if updated!=value: atomic_json(path,updated)
    for path in manifests:
        v=read_json(path);v['files']={name:sha256(path.parent/name) for name in v['files']};atomic_json(path,v)
    state=relocated(state);state['output']=str(output);atomic_json(output/'run.json',state);return state


def execute_stage(output,stage,method,options,state):
    from .visualization import validate_run,render_outputs
    context=prepare_comparison(build_context(options,output))
    from .preprocessing import prepare_model_inputs
    context=prepare_model_inputs(context)
    if stage=='segmentation':
        if method=='new':
            from .segmentation import segment_tile,load_models
            models=load_models(context);artifacts=[]
            for year in context.years:
                for tile in year.tiles:
                    destination=output/'segmentation'/str(tile.year)/tile.key
                    if (destination/'artifact.json').exists() and (destination/'manifest.json').exists():
                        verify_manifest(destination);artifacts.append(load_segmentation(destination/'artifact.json'));continue
                    print(f'Segment {tile.year}: {tile.image.name}',flush=True)
                    artifacts.append(segment_tile(tile,context,models=models))
        else:
            from .adapters import run_old_segmentation
            artifacts=run_old_segmentation(context)
        info={'artifacts':[str(a.directory/'artifact.json') for a in artifacts],'years':[y.year for y in context.years],'tiles':len(artifacts)}
    elif stage=='graph':
        from .graphs import build_tile_graph,combine_year_graphs
        artifacts=[load_segmentation(p) for p in state['stages']['segmentation']['result']['artifacts']]
        for artifact in artifacts: verify_manifest(artifact.directory)
        tiles=[build_tile_graph(a,method,context) for a in artifacts]
        annual=[combine_year_graphs([g for g in tiles if g.year==y.year],context) for y in context.years]
        info={'artifacts':[str(g.directory/'artifact.json') for g in tiles+annual],'annual':[str(g.directory/'artifact.json') for g in annual]}
    else:
        from .temporal import build_temporal_graph,detect_classical
        graphs=[_graph(p) for p in state['stages']['graph']['result']['annual']]
        if len(graphs)!=2: raise ValueError('Change requires two chronological input years; single-year segmentation/graph remain available')
        for graph in graphs: verify_manifest(graph.directory)
        temporal=build_temporal_graph(tuple(graphs),context);classical=detect_classical(temporal,context)
        if method=='gnn':
            from .gnn import run_gnn
            info=run_gnn(temporal,context,checkpoint=options.get('gnn_checkpoint'))
        else: info={'status':'complete','report':str(temporal.directory/'classical_changes.json'),'years':classical['years']}
        info['artifacts']=[str(temporal.directory/'artifact.json')]
        if method=='gnn': info['artifacts'].append(str(temporal.directory/'gnn/artifact.json'))
    validate_run(context,stage);render_outputs(context,stage);return info


def run_stage(stage,method,output,**options):
    from src import pipeline_select
    return getattr(pipeline_select,'segment' if stage=='segmentation' else stage)(method,output,**options)


def run_pipeline(segmentation,graph,change,output,**options):
    from src import pipeline_select
    return pipeline_select.run(segmentation,graph,change,output,**options)
