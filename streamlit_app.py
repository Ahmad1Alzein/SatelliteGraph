"""Run: python -m streamlit run streamlit_app.py"""
from pathlib import Path
import io
import math
import os
import subprocess
import sys
import uuid
import html
from collections import Counter

import streamlit as st
import pandas as pd

ROOT=Path(__file__).resolve().parent
EXTERNAL_CHECKPOINT=ROOT/'checkpoints/semantic_temporal/gnn/lsscd_major_pixels/model.pt'
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pipelines.semantic_temporal.runtime.storage import read_json,verify_result_manifest
from pipelines.semantic_temporal.runtime.change_contracts import CLASS_LABELS,EDGE_LABELS,TASKS,status_label
from pipelines.semantic_temporal.runtime.reports.common import class_color,EDGE_STYLES
from pipelines.semantic_temporal.runtime.reports.changes import event_crops,_scores,_predicted_events
from pipelines.semantic_temporal.runtime.preview import event_rows,filter_rows,grid_image

st.set_page_config(page_title='SatelliteGraph | Image comparison',layout='wide')
st.title('SatelliteGraph · Image comparison')
st.caption('Original images → segmentation → spatial graphs → matching → all-class change review')


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
def bundle(output,stamp):
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
    cols=st.columns(2)
    before=cols[0].file_uploader('Before image',type=['png','tif','tiff'],key='before')
    after=cols[1].file_uploader('After image',type=['png','tif','tiff'],key='after')
    st.caption('PNG pairs need no extra files. Alignment is automatic; measurements remain in pixels unless true map information is available.')
    external_ready=EXTERNAL_CHECKPOINT.is_file() and EXTERNAL_CHECKPOINT.with_suffix('.json').is_file()
    optical_png_pair=bool(before and after and Path(before.name).suffix.lower()=='.png' and Path(after.name).suffix.lower()=='.png')
    use_external=st.checkbox('Use the experimental externally supervised GNN for optical PNG pairs',value=external_ready,
        disabled=not(external_ready and optical_png_pair),
        help='Uses the local LsSCD-Ex checkpoint. Only supported class heads produce scores. Academic/noncommercial data license; accuracy on these images is unmeasured.')
    if external_ready and optical_png_pair and use_external:
        st.caption('Using experimental external weights. Validation support does not establish dependable detection; inspect held-out metrics in Training and review transfer to different satellites/views.')
    active=st.session_state.get('job') is not None and st.session_state.job.poll() is None
    if st.button('Run full comparison',type='primary',disabled=not(before and after) or active):
        try:
            output=ROOT/'artifacts/image_comparisons'/uuid.uuid4().hex[:12]
            inputs=output/'inputs';inputs.mkdir(parents=True)
            paths=[]
            for name,upload in [('before',before),('after',after)]:
                path=inputs/(name+Path(upload.name).suffix.lower());path.write_bytes(upload.getvalue());paths.append(path)
            launch([sys.executable,'-u','-m','src','pipeline','run','--segmentation','new','--graph','new','--change','gnn',
                '--images',*map(str,paths),'--output',str(output),'--device','cpu','--max-report-pixels','256000000',
                *(['--gnn-checkpoint',str(EXTERNAL_CHECKPOINT)] if use_external and external_ready and optical_png_pair else [])],output)
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
    context,frame,document,catalog,result,scores,rows=bundle(str(output),(output/'run.json').stat().st_mtime_ns)
except Exception as exc:
    st.error(f'Results are not ready: {exc}')
    st.stop()

st.caption(f'Run: {output} · Measurements: {frame["grid"].units}. Pixel areas are image-relative, not square metres.')
tabs=st.tabs(['Changes','Segmentation','Graphs','Matching','GNN training'])

with tabs[0]:
    st.subheader('Local before / after crops')
    st.caption('Yellow outlines mark the same change footprint. Crops use native original RGB, local context and at most 2× enlargement. No artificial sharpening.')
    show_weak=st.checkbox('Include weak and seasonal class differences',value=False,
        help='Raw segmentation differences are review candidates, not confirmed physical changes.')
    major_only=st.checkbox('Focus on major changes',value=True,
        help='Show whole-building appearance/disappearance and substantial road or land-cover changes. Hide thin outlines, small size changes and vegetation subclass swaps.')
    allowed=sorted({r['status'] for r in rows}) if show_weak else sorted({r['status'] for r in rows}&{'reliable','predicted'})
    filter_cols=st.columns(4)
    statuses=filter_cols[0].multiselect('Evidence status',sorted({r['status'] for r in rows}),
        default=allowed,format_func=status_label,key='statuses_'+str(show_weak))
    classes=filter_cols[1].multiselect('Classes',list(range(1,9)),default=list(range(1,9)),format_func=lambda k:CLASS_LABELS[k])
    area=filter_cols[2].number_input('Minimum change area',min_value=0.,value=0.,step=10.)
    confidence=filter_cols[3].slider('Minimum segment confidence',0.,1.,0.,.05,
        help='Segmentation model confidence, not a GNN change score or measured accuracy.')
    has_scores=any(r['gnn_score'] is not None for r in rows)
    use_score=st.checkbox('Filter by GNN change score',disabled=not has_scores)
    threshold=st.slider('Minimum GNN score',0.,1.,.5,.05,disabled=not(use_score and has_scores))
    if not has_scores:st.info('No eligible, validation-supported GNN scores are available for this comparison. Check the per-class head status and training details below. Image-evidence candidates remain reviewable.')
    filtered=filter_rows(rows,statuses=statuses,classes=classes,min_area=area,min_segment_confidence=confidence,
        min_gnn_score=threshold if use_score and has_scores else None,major_only=major_only)
    st.write(f'{len(filtered):,} of {len(rows):,} changes shown · {dict(Counter(r["status"] for r in filtered))}')
    st.dataframe(pd.DataFrame([{'Class':CLASS_LABELS[k],
        'Visible changes':sum(k in r['classes'] for r in filtered),
        'Image-evidence candidates':sum(k in r['classes'] for r in rows),
        'GNN gain head':result.get('head_status',{}).get(TASKS[(k-1)*2],{}).get('status','unavailable'),
        'GNN loss head':result.get('head_status',{}).get(TASKS[(k-1)*2+1],{}).get('status','unavailable')}
        for k in range(1,9)]),hide_index=True,width='stretch')
    if not filtered:
        st.info('No changes pass the current filters. Weak or seasonal differences can be inspected with the checkbox above; they are not confirmed changes.')
    pages=max(1,math.ceil(len(filtered)/12));page=st.number_input('Page (12 changes per page)',min_value=1,max_value=pages,value=1)
    visible=filtered[(page-1)*12:page*12]
    for start in range(0,len(visible),2):
        for col,row in zip(st.columns(2),visible[start:start+2]):
            with col:
                with st.container(border=True):
                    st.markdown(f'**{row["type"]}** · {status_label(row["status"])}')
                    panels,_=event_crops(frame,row['event'],(256,256))
                    for side,panel,title in zip(st.columns(2),panels,['Before','After']):side.image(panel,caption=title,width='stretch')
                    segment='unavailable' if row['segment_confidence'] is None else f'{row["segment_confidence"]:.3f}'
                    gnn='unavailable' if row['gnn_score'] is None else f'{row["gnn_score"]:.3f}'
                    st.caption(f'{row["area"]:,.1f} {row["area_units"]} · Segment confidence {segment} · GNN score {gnn}')
                    st.caption(row['event_id'])
    if visible:
        grid,_=grid_image(frame,[r['event'] for r in visible],scores);buffer=io.BytesIO();grid.save(buffer,format='PNG')
        st.download_button('Download this page as one grid PNG',buffer.getvalue(),file_name=f'change_grid_{page:03d}.png',mime='image/png')
    with st.expander('Change data'):
        st.dataframe(pd.DataFrame([{k:v for k,v in r.items() if k!='event'} for r in filtered]),width='stretch')

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
    for kind in ('touches','near','context_near'):
        color,style=EDGE_STYLES[kind]
        st.markdown(f'<span style="color:{color}">━━</span> **{EDGE_LABELS[kind]}** · {style}',unsafe_allow_html=True)

with tabs[3]:
    st.caption('Registration aligns image landmarks. Temporal matching links segmented areas. Fitted landmark residuals are not independent accuracy.')
    for name in ['registration_matches','temporal_matches','temporal_match_types','class_change_matches']:
        path=output/'results'/f'{name}.png'
        if path.is_file():st.image(str(path),caption=name.replace('_',' ').capitalize(),width='stretch')

with tabs[4]:
    st.subheader('GNN task support and review')
    st.caption('Training needs changed and unchanged examples for each class/direction. One image pair supports training only; it cannot establish held-out accuracy.')
    st.dataframe(pd.DataFrame([dict(task=task,**head) for task,head in result.get('head_status',{}).items()]),width='stretch')
    if result.get('evaluation',{}).get('status')=='external_annotated_subset':
        st.caption('External held-out annotated-node metrics. These do not measure accuracy on the uploaded images.')
        st.dataframe(pd.DataFrame([dict(task=task,**metrics) for task,metrics in result['evaluation']['per_task'].items()]),width='stretch')
    eligible=[r for r in catalog['targets'] if r['eligible']]
    if not eligible:st.info('No node/task passes the coverage, class and confidence gates. Review labels cannot override those gates.')
    else:
        node_lookup={n['id']:n for n in document['nodes']}
        options={f'{r["task"]} | {r["gid"]}':r for r in eligible}
        choice=st.selectbox('Review an eligible training example',list(options));target=options[choice];node=node_lookup[target['gid']]
        direction=target['task'].rsplit('_',1)[1]
        event=dict(event_id='review:'+target['gid']+':'+target['task'],kind=direction,class_id=node['class_id'],geometry=node['geometry'],area=node['area'])
        panels,_=event_crops(frame,event,(320,320))
        for side,panel,title in zip(st.columns(2),panels,['Before','After']):side.image(panel,caption=title,width='stretch')
        st.caption(f'Current label: {target.get("label")} · Reason: {target["reason"]}. Judge this class/direction inside the highlighted footprint.')
        decision=st.radio('Reviewed outcome',['Unsure','Changed','Unchanged'],horizontal=True)
        review_path=output/'reviewed_targets.csv'
        if st.button('Save reviewed label',disabled=decision=='Unsure'):
            existing=pd.read_csv(review_path).to_dict('records') if review_path.is_file() else []
            existing=[r for r in existing if (r['gid'],r['task'])!=(target['gid'],target['task'])]
            existing.append(dict(gid=target['gid'],task=target['task'],label=int(decision=='Changed'),confidence=1.,source='user_visual_review'))
            pd.DataFrame(existing).to_csv(review_path,index=False);st.success('Reviewed label saved.')
        if review_path.is_file():
            st.dataframe(pd.read_csv(review_path),width='stretch')
            fixed_checkpoint=bool(context.options.get('gnn_checkpoint'))
            if fixed_checkpoint:
                st.caption('This run uses fixed external weights. Reviewed-label training requires a fresh run with the external GNN option unchecked; it does not fine-tune these weights.')
            if st.button('Train supported GNN heads and refresh changes',disabled=active or fixed_checkpoint):
                try:
                    launch([sys.executable,'-u','-m','src','pipeline','change','--method','gnn','--output',str(output),
                        '--reviewed-targets',str(review_path)],output);st.rerun()
                except Exception as exc:st.error(str(exc))
