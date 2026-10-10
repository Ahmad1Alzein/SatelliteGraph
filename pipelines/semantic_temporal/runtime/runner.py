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
    from .context import verify_prepared_context
    verify_prepared_context(output)
    from .storage import verify_result_manifest
    result_stages=[p.parent.name for p in (output/'results/rendering').glob('*/manifest.json')]
    for stage in result_stages:verify_result_manifest(output,stage)
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
    # Internal JSON descriptions may change during rebasing. Rebind verified dependencies,
    # then index outputs, in that order; external file hashes are never rewritten.
    for path in output.glob('results/**/index.json'):
        value=read_json(path)
        for item in value.get('inputs',[]):
            if not item['external']:item['sha256']=sha256(output/item['path'])
        atomic_json(path,value)
    for path in manifests:
        v=read_json(path);v['files']={name:sha256(path.parent/name) for name in v['files']};atomic_json(path,v)
    for stage in result_stages:
        path=output/'results/rendering'/stage/'manifest.json';value=read_json(path)
        for item in value['inputs']+value['outputs']:
            if not item['external']:item['sha256']=sha256(output/item['path'])
        atomic_json(path,value)
    if (output/'results/manifest.json').is_file():
        path=output/'results/manifest.json';value=read_json(path)
        value['files']={name:sha256(path.parent/name) for name in value['files']};atomic_json(path,value)
    state=relocated(state);state['output']=str(output);atomic_json(output/'run.json',state);return state


def execute_stage(output,stage,method,options,state):
    from .visualization import validate_run,render_outputs
    # Alignment is an upstream scientific input, not a per-stage estimate.
    # Re-fitting here can move graph geometry away from segmented pixels.
    if stage=='segmentation':
        if (output/'prepared/context/manifest.json').is_file():
            from .context import load_context
            context=load_context(output,options)
        else:context=prepare_comparison(build_context(options,output))
    else:
        from .context import load_context
        context=load_context(output,options)
    from .preprocessing import prepare_model_inputs
    context=prepare_model_inputs(context)
    if stage=='segmentation':
        from .context import seal_prepared_context
        seal_prepared_context(context)
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
        tiles=[];annual=[]
        for artifact in artifacts:
            print(f'Graph {artifact.year}: {artifact.tile_key}',flush=True)
            tiles.append(build_tile_graph(artifact,method,context))
        for year in context.years:
            print(f'Combine annual graph: {year.year}',flush=True)
            annual.append(combine_year_graphs([g for g in tiles if g.year==year.year],context))
        info={'artifacts':[str(g.directory/'artifact.json') for g in tiles+annual],'annual':[str(g.directory/'artifact.json') for g in annual]}
    else:
        from .temporal import build_temporal_graph,detect_classical
        graphs=[_graph(p) for p in state['stages']['graph']['result']['annual']]
        if len(graphs)!=2: raise ValueError('Change requires two chronological input years; single-year segmentation/graph remain available')
        for graph in graphs: verify_manifest(graph.directory)
        print('Build temporal correspondences',flush=True)
        temporal=build_temporal_graph(tuple(graphs),context)
        print('Measure all-class image changes and training targets',flush=True)
        classical=detect_classical(temporal,context)
        if method=='gnn':
            from .gnn import run_gnn
            print('Run existing RGNN with supported class gain/loss heads',flush=True)
            info=run_gnn(temporal,context,checkpoint=options.get('gnn_checkpoint'))
        else: info={'status':'complete','report':str(temporal.directory/'classical_changes.json'),'years':classical['years']}
        info['artifacts']=[str(temporal.directory/'artifact.json')]
        if method=='gnn': info['artifacts'].append(str(temporal.directory/'gnn/artifact.json'))
    print(f'Validate {stage} scientific artifacts',flush=True)
    validate_run(context,stage)
    state['stages'][stage].update(result=info,scientific_status='complete',render_status='pending')
    atomic_json(output/'run.json',state)
    render_completed_stage(output,stage,state)
    return info


def _scientific_inputs(output,state,stage):
    from .context import verify_prepared_context
    verify_prepared_context(output)
    paths=[output/'context.json',output/'prepared/context/manifest.json',
           output/'prepared/context/context.json']
    if (output/'prepared/registration_controls.json').is_file():
        paths.extend([output/'prepared/registration_controls.json',output/'prepared/context/registration_controls.json'])
    for name in ('segmentation','graph','change'):
        for artifact in state.get('stages',{}).get(name,{}).get('result',{}).get('artifacts',[]):
            path=Path(artifact);verify_manifest(path.parent);paths.append(path)
            manifest=path.parent/'manifest.json';paths.append(manifest)
            paths.extend(path.parent/key for key in read_json(manifest)['files'])
        if name==stage:break
    if stage=='change' and (output/'prepared/registration_controls.json').is_file():paths.append(output/'prepared/registration_controls.json')
    for key in ('change_policy','reviewed_targets','gnn_checkpoint') if stage=='change' else ():
        value=state['options'].get(key)
        if value:paths.append(Path(value))
        if key=='gnn_checkpoint' and value:paths.append(Path(value).with_suffix('.json'))
    return paths


def _result_index(output,state):
    from .storage import save_manifest
    directory=output/'results';directory.mkdir(exist_ok=True);images={};indexes={};science={};statuses={}
    for stage in ('segmentation','graph','change'):
        saved=state.get('stages',{}).get(stage,{})
        statuses[stage]={key:saved.get(key) for key in ('scientific_status','render_status','status')}
        if saved.get('scientific_status')!='complete':continue
        science[stage]=[str(Path(p).relative_to(output)) for p in saved.get('result',{}).get('artifacts',[])]
        if saved.get('render_status')!='complete':continue
        manifest=output/'results/rendering'/stage/'manifest.json'
        if not manifest.is_file():continue
        for item in read_json(manifest)['outputs']:
            if item['path'].endswith('.png'):images[Path(item['path']).stem if stage=='change' else item['path']]=item['path']
            elif item['path'].endswith('index.json'):indexes[item['path']]=item['path']
    index=directory/'index.json';atomic_json(index,{'stages':statuses,'images':images,'indexes':indexes,'scientific_artifacts':science,
        'task_status':state.get('stages',{}).get('change',{}).get('result',{}).get('head_status',{})})
    save_manifest(directory,{'index':index},stage='results_index')


def render_completed_stage(output,stage,state):
    from .context import load_context
    from .visualization import validate_run,render_outputs
    from .storage import bind_result_manifest
    from src.pipeline_state import render_fingerprint
    output=Path(output).resolve();saved=state['stages'][stage]
    if saved.get('scientific_status')!='complete':raise ValueError('Render retry requires completed scientific artifacts')
    inputs=_scientific_inputs(output,state,stage);context=load_context(output,state['options'])
    saved['render_status']='running';atomic_json(output/'run.json',state)
    try:
        validate_run(context,stage)
        print(f'Render {stage} verification images',flush=True)
        paths=render_outputs(context,stage,include_results=True)
        paths={k:v for k,v in paths.items() if Path(v).is_relative_to(output/'results')}
        bind_result_manifest(output,stage,paths,inputs)
        saved.update(render_status='complete',render_fingerprint=render_fingerprint(state['options'],stage))
        saved.pop('error',None)
        _result_index(output,state);atomic_json(output/'run.json',state)
        print(f'{stage.capitalize()} images: {output/"results/index.json"}',flush=True)
        if stage=='change':
            print(f'Combined change report: {output/"results/change_report.png"}',flush=True)
            for task,head in saved.get('result',{}).get('head_status',{}).items():
                print(f'{task}: {head.get("status")} | {head.get("reason","")}',flush=True)
        return paths
    except Exception as exc:
        saved.update(status='render_failed',render_status='failed',error=f'{type(exc).__name__}: {exc}')
        _result_index(output,state);atomic_json(output/'run.json',state);raise


def run_stage(stage,method,output,**options):
    from src import pipeline_select
    return getattr(pipeline_select,'segment' if stage=='segmentation' else stage)(method,output,**options)


def run_pipeline(segmentation,graph,change,output,**options):
    from src import pipeline_select
    return pipeline_select.run(segmentation,graph,change,output,**options)
