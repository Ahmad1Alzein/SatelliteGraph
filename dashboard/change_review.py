"""Read-only crop review, shared by the dashboard and focused UI tests."""
from collections import Counter
import io
import math

import pandas as pd
import streamlit as st

from pipelines.semantic_temporal.runtime.change_contracts import CLASS_LABELS,status_label
from pipelines.semantic_temporal.runtime.preview import filter_rows,grid_image
from pipelines.semantic_temporal.runtime.reports.changes import event_crops


def render_crop_review(frame,result,scores,rows,selected_methods,*,original_run,retired_run):
    st.subheader('Local before / after crops')
    st.caption('Yellow outlines mark the same change footprint. Crops use native original RGB, local context and at most 2× enlargement. No artificial sharpening.')
    show_weak=st.checkbox('Include weak and seasonal class differences',value=False,
        help='Raw segmentation differences are review candidates, not confirmed physical changes.')
    major_only=st.checkbox('Focus on major changes',value=True,
        help='Show whole-building changes and substantial road or land-cover changes. Hide thin outlines, vegetation subclass swaps and land-cover patches affecting less than 10% of participating regions on either date.')
    allowed=sorted({r['status'] for r in rows}) if show_weak or selected_methods['change']=='classical' else sorted({r['status'] for r in rows}&{'reliable','predicted'})
    filter_cols=st.columns(3)
    statuses=filter_cols[0].multiselect('Evidence status',sorted({r['status'] for r in rows}),
        default=allowed,format_func=status_label,key='statuses_'+str(show_weak))
    classes=filter_cols[1].multiselect('Classes',list(range(1,9)),default=list(range(1,9)),format_func=lambda k:CLASS_LABELS[k])
    area=filter_cols[2].number_input('Minimum change area',min_value=0.,value=0.,step=10.)
    score_cols=st.columns(2)
    segment_threshold=score_cols[0].slider('Minimum segmentation score',0.,1.,0.,.05,
        help='Zero keeps all scores, including unavailable ones. Higher values retain stronger semantic class support.')
    has_scores=any(r['gnn_score'] is not None for r in rows)
    use_score=score_cols[1].checkbox('Filter by GNN change score',disabled=not has_scores)
    threshold=score_cols[1].slider('Minimum GNN change score',0.,1.,.5,.05,disabled=not(use_score and has_scores))
    st.caption('Sorted by segmentation score, highest first, across all pages. Each segment score is its mean semantic class probability; the change uses the lowest available score among its before/after segments. Colour rules can assign bare ground or water despite low semantic probability. Physical-change confidence is separate.')
    if has_scores:
        st.caption('Higher GNN change scores indicate greater model support for a change. Evidence labels distinguish strong image evidence from weak or seasonal candidates.')
        if original_run:st.caption('These scores come from a fit trained on this area; they are not independently tested confidence or accuracy.')
    else:
        st.info('This comparison has no GNN scores. Classical runs use rule-based events; the original ST-GNN needs reliable changed and unchanged training targets. Segmentation scores remain available for review.')
    filtered=filter_rows(rows,statuses=statuses,classes=classes,min_area=area,
        min_segment_confidence=segment_threshold,
        min_gnn_score=threshold if use_score and has_scores else None,major_only=major_only)
    st.write(f'{len(filtered):,} of {len(rows):,} changes match the filters · {dict(Counter(r["status"] for r in filtered))}')
    st.caption('To inspect every candidate: include weak/seasonal differences, turn off major-change focus, select all evidence statuses and classes, and set minimum area and segmentation score to zero with the GNN filter off.')
    st.dataframe(pd.DataFrame([{'Class':CLASS_LABELS[k],
        'Visible changes':sum(k in r['classes'] for r in filtered),
        'All review events':sum(k in r['classes'] for r in rows),
        'GNN scope / status':('Rule-based run; no GNN' if selected_methods['change']=='classical' else
            'Historical experimental scores' if retired_run else
            'Later building gain: '+result.get('head_status',{}).get('building_gain',{}).get('status','unavailable') if k==1 else
            'Earlier land built over: '+result.get('head_status',{}).get('built_over',{}).get('status','unavailable'))}
        for k in range(1,9)]),hide_index=True,width='stretch')
    if not filtered:st.info('No changes pass the current filters. Adjust the controls above to review other candidates.')
    pages=max(1,math.ceil(len(filtered)/12))
    signature=tuple(r['event_id'] for r in filtered)
    page_key='change_review_page'
    if st.session_state.get('change_review_selection')!=signature:
        st.session_state[page_key]=1
        st.session_state['change_review_selection']=signature
    st.session_state[page_key]=max(1,min(pages,st.session_state.get(page_key,1)))
    page=st.number_input('Page (12 changes per page)',min_value=1,max_value=pages,key=page_key)
    visible=filtered[(page-1)*12:page*12]
    def score_text(value):return 'unavailable' if value is None else f'{value:.3f}'
    for start in range(0,len(visible),2):
        for col,row in zip(st.columns(2),visible[start:start+2]):
            with col:
                with st.container(border=True):
                    st.markdown(f'**{row["type"]}** · {status_label(row["status"])}')
                    panels,_=event_crops(frame,row['event'],(256,256))
                    for side,panel,title in zip(st.columns(2),panels,['Before','After']):side.image(panel,caption=title,width='stretch')
                    st.caption(f'{row["area"]:,.1f} {row["area_units"]} · Segmentation score: {score_text(row["segment_confidence"])}')
                    st.caption(f'Before segment: {score_text(row.get("before_segment_confidence"))} · After segment: {score_text(row.get("after_segment_confidence"))}')
                    if row['gnn_score'] is not None:st.caption(f'GNN change score: {row["gnn_score"]:.3f}')
                    st.caption(row['event_id'])
    if visible:
        grid,_=grid_image(frame,[r['event'] for r in visible],scores);buffer=io.BytesIO();grid.save(buffer,format='PNG')
        st.download_button('Download this page as one grid PNG',buffer.getvalue(),file_name=f'change_grid_{page:03d}.png',mime='image/png')
    with st.expander('Change data'):
        st.dataframe(pd.DataFrame([{k:v for k,v in r.items() if k!='event'} for r in filtered]),width='stretch')
