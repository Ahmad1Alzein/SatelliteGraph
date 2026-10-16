"""Selectable pipeline workspace, mounted by thesis_dashboard.py."""
from pathlib import Path
import os
import subprocess
import sys
import uuid
import html
from collections import Counter

import streamlit as st
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pipelines.semantic_temporal.runtime.storage import read_json,verify_result_manifest
from pipelines.semantic_temporal.runtime.change_contracts import CLASS_LABELS,EDGE_LABELS
from pipelines.semantic_temporal.runtime.reports.common import class_color,EDGE_STYLES
from pipelines.semantic_temporal.runtime.reports.changes import _scores,_predicted_events
from pipelines.semantic_temporal.runtime.preview import event_rows
from dashboard.pipeline_controls import METHOD_LABELS,pipeline_command
from src.pipeline_state import RENDER_PROFILE

st.set_page_config(page_title='SatelliteGraph | Image comparison',layout='wide')
st.title('SatelliteGraph · Image comparison')
st.caption('Choose segmentation, spatial graph and change detection independently, then inspect their common results.')


def launch(command,output):
    if st.session_state.get('job') and st.session_state.job.poll() is None:
        raise ValueError('A pipeline is already running in this session')
    log=output/'console.log';log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('ab') as stream:
        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,
            env={**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONUTF8':'1'},
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    st.session_state.update(job=process,job_output=str(output),job_log=str(log),job_reported=False)


@st.cache_data(show_spinner='Reading verified results…',max_entries=2)
def bundle(output,stamp,review_profile):
    from pipelines.semantic_temporal.runtime.context import load_context
    from pipelines.semantic_temporal.runtime.comparison import load_comparison
    from pipelines.semantic_temporal.runtime.contracts import TemporalArtifact
    path=Path(output);state=read_json(path/'run.json')
    if state['stages']['change'].get('render_status')!='complete':raise ValueError('The change stage has not finished rendering')
    verify_result_manifest(path,'change')
    context=load_context(path,state['options'])
    artifact=read_json(state['stages']['change']['result']['artifacts'][0])
    graph=TemporalArtifact(tuple(artifact['years']),Path(artifact['directory']),{k:Path(v) for k,v in artifact['paths'].items()})
    document=read_json(graph.paths['graph']);catalog=read_json(graph.directory/'classical_changes.json')
    result=state['stages']['change']['result'];scores=_scores(result)
    events=catalog.get('events',[])+_predicted_events(document,catalog,scores)
    frame=load_comparison(graph,context)
    return context,frame,document,catalog,result,scores,event_rows(events,document,scores,native_pixel=frame['source_pixel_size'],
        external_predictions=result.get('evaluation',{}).get('status')=='external_annotated_subset')


with st.expander('Choose two images and run the full pipeline',expanded=True):
    methods=st.columns(3)
    segmentation=methods[0].selectbox('Segmentation',list(METHOD_LABELS['segmentation']),
        format_func=METHOD_LABELS['segmentation'].get,key='pipeline_segmentation')
    graph_method=methods[1].selectbox('Spatial graph',list(METHOD_LABELS['graph']),
        format_func=METHOD_LABELS['graph'].get,key='pipeline_graph')
    change_method=methods[2].selectbox('Change detection',list(METHOD_LABELS['change']),
        format_func=METHOD_LABELS['change'].get,key='pipeline_change')
    if change_method=='gnn':
        st.caption('Uses the original thesis building ST-GNN: later new/extended buildings and earlier land built over. '
            'Other classes are reviewed as image-evidence candidates. New-input fits are separate from the preserved thesis results.')
    input_mode=st.radio('Input source',['Upload images','Local image paths','Local tile folders'],horizontal=True)
    cols=st.columns(2)
    if input_mode=='Upload images':
        before=cols[0].file_uploader('Before image',type=['png','tif','tiff'],key='before')
        after=cols[1].file_uploader('After image',type=['png','tif','tiff'],key='after')
    else:
        before=cols[0].text_input('Before folder' if input_mode=='Local tile folders' else 'Before image path')
        after=cols[1].text_input('After folder' if input_mode=='Local tile folders' else 'After image path')
    st.caption('PNG pairs are aligned automatically in image coordinates. Tile folders use their placement metadata. '
        'Mapped GeoTIFFs keep their map information; pixel-only measurements remain in pixels.')
    options=st.columns(4)
    first_year=options[0].number_input('Before year',min_value=1,max_value=9998,value=2013)
    last_year=options[1].number_input('After year',min_value=2,max_value=9999,value=2023)
    device=options[2].selectbox('Processing device',['cpu','auto','cuda'])
    pixel_size=options[3].number_input('Known metres per pixel (0 = unknown)',min_value=0.,value=0.,step=.25)
    register_images=st.checkbox('Estimate image registration',value=False,
        help='For mapped inputs, estimate alignment from current image landmarks. Unmapped PNG pairs align automatically.')
    harmonize_images=st.checkbox('Harmonize brightness and sharpness before segmentation',value=False)
    active=st.session_state.get('job') is not None and st.session_state.job.poll() is None
    if st.button('Run full comparison',type='primary',disabled=not(before and after) or active):
        try:
            output=ROOT/'artifacts/image_comparisons'/uuid.uuid4().hex[:12]
            if input_mode=='Upload images':
                inputs=output/'inputs';inputs.mkdir(parents=True)
                paths=[]
                for name,upload in [('before',before),('after',after)]:
                    path=inputs/(name+Path(upload.name).suffix.lower());path.write_bytes(upload.getvalue());paths.append(path)
            else:
                paths=[Path(p).expanduser().resolve() for p in (before,after)]
            command=pipeline_command(output,segmentation,graph_method,change_method,paths,
                input_kind='folders' if input_mode=='Local tile folders' else 'images',
                years=(first_year,last_year),device=device,pixel_size=pixel_size or None,
                register_images=register_images,harmonize_images=harmonize_images)
            launch(command,output)
            st.rerun()
        except Exception as exc:st.error(str(exc))


@st.fragment(run_every=2)
def progress():
    job=st.session_state.get('job')
    if job is None:return
    code=job.poll();path=Path(st.session_state.job_log)
    if code is None:st.info('Pipeline running locally. CPU segmentation can take several minutes.')
    elif not st.session_state.get('job_reported'):
        st.session_state.job_reported=True
        if code==0:
            st.session_state.selected_run=st.session_state.job_output
            st.rerun()
    if code is not None:
        (st.success if code==0 else st.error)(f'Pipeline finished (exit {code}).')
    if path.is_file():
        with st.expander('Pipeline log',expanded=code not in (None,0)):
            st.code(path.read_text(encoding='utf-8',errors='replace')[-12000:],language=None)
progress()

default=os.environ.get('SATELLITEGRAPH_RUN','')
selected=st.text_input('Open a completed run folder',key='selected_run',value=st.session_state.get('selected_run',default))
if not selected:
    st.info('Upload two images above, or enter a completed run folder to inspect its results.')
    st.stop()
output=Path(selected).expanduser().resolve()
try:
    from dashboard.run_results import change_view_status,show_archived_results
    state=read_json(output/'run.json')
    if change_view_status(output,state)=='archived':
        show_archived_results(output,state)
        st.stop()
    context,frame,document,catalog,result,scores,rows=bundle(str(output),(output/'run.json').stat().st_mtime_ns,RENDER_PROFILE)
except Exception as exc:
    st.error(f'Results are not ready: {exc}')
    st.stop()

st.caption(f'Run: {output} · Measurements: {frame["grid"].units}. Pixel areas are image-relative, not square metres.')
state=read_json(output/'run.json')
selected_methods={stage:state['stages'][stage]['method'] for stage in ('segmentation','graph','change')}
original_run=result.get('model_source')=='pipelines/legacy_building_stgnn/original/gnn_change.py'
retired_run=selected_methods['change']=='gnn' and not original_run
method_labels={stage:METHOD_LABELS[stage].get(method,method) for stage,method in selected_methods.items()}
if retired_run:
    method_labels['change']='Historical experimental all-class RGNN (retired)'
    st.warning('This saved run used the retired experimental RGNN. Its historical outputs are retained; new GNN runs use the original thesis building ST-GNN.')
st.caption(' · '.join(method_labels.values()))
tabs=st.tabs(['Changes','Segmentation','Graphs','Matching','GNN training'])

with tabs[0]:
    from dashboard.change_review import render_crop_review
    render_crop_review(frame,result,scores,rows,selected_methods,
        original_run=original_run,retired_run=retired_run)

with tabs[1]:
    st.subheader('Segmentation and class data')
    records=read_json(output/'results/segmentation/index.json')['images']
    for year,col in zip(frame['years'],st.columns(2)):
        with col:
            st.markdown(f'**{"Before" if year==frame["years"][0] else "After"} · {year}**')
            for record in records:
                if record['year']==year:st.image(str(output/record['path']),width='stretch')
            labels=frame['data'][year]['classes'];pixel_area=abs(frame['grid'].affine[0]*frame['grid'].affine[4])
            counts=Counter(n['class_id'] for n in document['nodes'] if n['year']==year)
            values=[]
            for klass,name in CLASS_LABELS.items():
                color='#%02x%02x%02x'%class_color(klass);count=int((labels==klass).sum())
                st.markdown(f'<span style="color:{color}">■</span> {html.escape(name)} · {count:,} comparison pixels · {counts[klass]} regions',unsafe_allow_html=True)
                values.append({'Class':name,'Pixels':count,'Regions':counts[klass],'Area':count*pixel_area,'Color':color})
            st.dataframe(pd.DataFrame(values),hide_index=True,width='stretch')
    st.caption('Regions are segmentation outputs, not verified object counts. Legends exclude empty coverage.')

with tabs[2]:
    records=read_json(output/'results/graphs/index.json')['images']
    for year,col in zip(frame['years'],st.columns(2)):
        with col:
            for record in records:
                if record['year']==year and not record.get('annual'):
                    st.image(str(output/record['path']),width='stretch')
                    st.caption(f'{record["node_count"]} nodes · {record["edge_count"]} edges')
    for kind in (('touches',) if selected_methods['graph']=='old' else ('touches','near','context_near')):
        color,style=EDGE_STYLES[kind]
        st.markdown(f'<span style="color:{color}">━━</span> **{EDGE_LABELS[kind]}** · {style}',unsafe_allow_html=True)

with tabs[3]:
    st.caption('Registration aligns image landmarks. Temporal matching links segmented areas. Fitted landmark residuals are not independent accuracy.')
    for name in ['registration_matches','temporal_matches','temporal_match_types','class_change_matches']:
        path=output/'results'/f'{name}.png'
        if path.is_file():st.image(str(path),caption=name.replace('_',' ').capitalize(),width='stretch')

with tabs[4]:
    st.subheader('Historical experimental RGNN (retired)' if retired_run else 'Original thesis ST-GNN')
    if selected_methods['change']=='classical':
        st.info('Rule-based change detection selected. No GNN was trained for this run.')
    else:
        st.caption(result.get('reason','Saved experimental run; this model is no longer used.' if retired_run else 'Original binary building ST-GNN; separate new-input experiment.'))
        st.code(result.get('model_source','Historical run: experimental model has been retired'),language=None)
    st.dataframe(pd.DataFrame([dict(task=task,**head) for task,head in result.get('head_status',{}).items()]),width='stretch')
    st.caption('The preserved thesis experiment, figures and held-out predictions remain on the Spatio-temporal GNN page. '
        'The original model has no downloaded weights or saved checkpoint.')
