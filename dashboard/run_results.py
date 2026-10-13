"""Distinguish verified reports from read-only, partial historical exports."""
from pathlib import Path

import streamlit as st

from dashboard import data as D
from pipelines.semantic_temporal.runtime.storage import verify_result_manifest


def change_view_status(run, state):
    run = Path(run)
    if state.get('stages', {}).get('change', {}).get('render_status') != 'complete':
        return 'pending'
    if (run / 'results/rendering/change/manifest.json').is_file():
        verify_result_manifest(run, 'change')
        return 'verified'
    # Published data-only exports omit imagery, prepared context and renders.
    # Their copied ledger describes the original machine, not this export.
    if ((run / 'temporal/classical_changes.json').is_file()
            and not (run / 'results').exists()
            and not (run / 'context.json').exists()
            and not (run / 'prepared').exists()):
        return 'archived'
    raise ValueError('Change report verification manifest is missing. Open a complete run export '
                     'or regenerate its reports from the original inputs.')


def show_archived_results(run, state):
    run = Path(run)
    st.info('Partial historical export: saved event data are available, but rendered reports, '
            'source imagery and verification files were not included. These saved data are not verified. '
            'Run a new comparison with the original images to generate current reports and local crops.')
    change = state.get('stages', {}).get('change', {})
    if change.get('method') == 'gnn' and not change.get('result', {}).get('model_source'):
        st.warning('This export used the retired experimental RGNN. Its saved events do not represent '
                   'results from the original thesis building ST-GNN.')
    st.subheader('Saved change events')
    events = D.events_from_run(run).drop(columns='geometry')
    st.dataframe(events, hide_index=True, width='stretch')
    st.download_button('Download saved events (CSV)', events.to_csv(index=False),
                       file_name=run.name+'_events.csv', mime='text/csv')
