"""Streamlit dashboard for live monitoring of telemetry, predictions, confidence, and RUL."""
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / 'outputs' / 'predictions.db'
TELEMETRY_PATH = BASE_DIR / 'outputs' / 'latest_telemetry.npz'
EXPERIMENTS_DIR = BASE_DIR / 'outputs' / 'experiments'

st.set_page_config(page_title='Predictive Maintenance Dashboard', layout='wide', initial_sidebar_state='expanded')

st.markdown("""
<style>
body {background-color:#0f1218; color:#f2f5f9;}
.stApp {background-color:#0f1218;}
section.main {padding-top: 0rem;}
.css-1d391kg, .css-1d391kg * {background-color: #11171f !important; color: #f2f5f9 !important;}
</style>
""", unsafe_allow_html=True)

st.sidebar.title('Realtime Monitoring')
refresh = st.sidebar.slider('Refresh interval (seconds)', 2, 10, 3)
show_telemetry = st.sidebar.checkbox('Show latest telemetry waveform', value=True)
show_eval = st.sidebar.checkbox('Show latest evaluation summary', value=True)

st.title('Industrial Predictive Maintenance Dashboard')
st.markdown('Monitor live predictions, model confidence, RUL trends, and experiment evaluation assets for publication-quality reporting.')


def latest_experiment_path():
    if not EXPERIMENTS_DIR.exists():
        return None
    experiments = [p for p in EXPERIMENTS_DIR.iterdir() if p.is_dir()]
    if not experiments:
        return None
    return max(experiments, key=lambda p: p.name)


@st.cache_data(ttl=refresh)
def load_predictions(limit: int = 300):
    if not DB_PATH.exists():
        return pd.DataFrame(columns=['id', 'ts', 'prediction', 'confidence', 'severity', 'rul'])
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM predictions ORDER BY ts DESC LIMIT {limit}", conn)
    conn.close()
    return df


@st.cache_data(ttl=refresh)
def load_telemetry():
    if not TELEMETRY_PATH.exists():
        return None
    data = np.load(TELEMETRY_PATH, allow_pickle=True)
    return {
        'channels': data['channels'],
        'meta': data['meta'].item() if hasattr(data['meta'], 'item') else data['meta'],
        'ts': float(data['ts'])
    }


@st.cache_data(ttl=refresh)
def load_evaluation_metrics():
    exp_path = latest_experiment_path()
    if exp_path is None:
        return None, None
    eval_file = exp_path / 'eval_metrics.json'
    report_file = exp_path / 'classification_report.txt'
    if not eval_file.exists():
        return None, None
    with open(eval_file, 'r') as f:
        metrics = json.load(f)
    report = None
    if report_file.exists():
        with open(report_file, 'r') as f:
            report = f.read()
    return metrics, report


predictions = load_predictions()
telemetry = load_telemetry()
eval_metrics, eval_report = load_evaluation_metrics()

if predictions.empty:
    st.warning('No predictions logged yet. Run `stream_demo.py` and refresh.')
else:
    predictions['ts_readable'] = pd.to_datetime(predictions['ts'], unit='s')
    latest = predictions.iloc[0]
    recent = predictions.head(20)[::-1]

    top_left, top_right = st.columns([3, 1])
    with top_left:
        fig_conf = go.Figure()
        fig_conf.add_trace(go.Scatter(x=predictions['ts_readable'], y=predictions['confidence'], mode='lines+markers', line=dict(color='#22d3ee')))
        fig_conf.update_layout(template='plotly_dark', title='Confidence Trend', xaxis_title='Time', yaxis_title='Confidence', height=330, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_conf, width='stretch')

        fig_rul = go.Figure()
        fig_rul.add_trace(go.Scatter(x=predictions['ts_readable'], y=predictions['rul'], mode='lines', line=dict(color='#fb7185')))
        fig_rul.update_layout(template='plotly_dark', title='RUL Trend', xaxis_title='Time', yaxis_title='Estimated RUL', height=330, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_rul, width='stretch')

    with top_right:
        st.metric('Latest Prediction', int(latest['prediction']))
        st.metric('Latest Confidence', f"{latest['confidence']:.3f}")
        st.metric('Latest RUL', f"{latest['rul']:.1f}")
        st.markdown(f"**Severity level:** {int(latest['severity'])}")
        alert_level = ('NORMAL' if latest['confidence'] > 0.6 else 'ADVISORY' if latest['confidence'] > 0.4 else 'WARNING')
        st.markdown(f"### Alert State: {alert_level}")

    if show_eval and eval_metrics:
        st.markdown('### Latest Model Evaluation Metrics')
        metric_cards = st.columns(4)
        metric_items = [
            ('Accuracy', eval_metrics.get('accuracy')), 
            ('Balanced Accuracy', eval_metrics.get('balanced_accuracy')),
            ('Macro F1', eval_metrics.get('f1_macro')),
            ('ROC AUC OVR', eval_metrics.get('roc_auc_ovr'))
        ]
        for col, (label, value) in zip(metric_cards, metric_items):
            col.metric(label, f"{value:.3f}" if value is not None else 'N/A')

        if eval_report:
            with st.expander('Classification report'): 
                st.text(eval_report)

    if show_telemetry and telemetry is not None:
        st.markdown('### Latest Sensor Waveforms')
        ch_names = ['Vibration', 'Current', 'Temperature', 'RPM']
        fig_wave = go.Figure()
        palette = ['#7c3aed', '#22d3ee', '#f97316', '#10b981']
        for i, name in enumerate(ch_names):
            fig_wave.add_trace(go.Scatter(y=telemetry['channels'][i], mode='lines', name=name, line=dict(color=palette[i], width=1)))
        fig_wave.update_layout(template='plotly_dark', height=460, margin=dict(l=10, r=10, t=40, b=10), xaxis_title='Sample', yaxis_title='Amplitude')
        st.plotly_chart(fig_wave, width='stretch')

        meta = telemetry['meta']
        st.markdown(
            f"**Telemetry Timestamp:** {pd.to_datetime(telemetry['ts'], unit='s')}  \n"
            f"**Fault Mode:** {meta.get('fault', 'unknown')}  \n"
            f"**Severity:** {meta.get('severity', 'unknown')}  \n"
            f"**Speed RPM:** {meta.get('speed_rpm', 0)}  \n"
            f"**Load:** {meta.get('load_fraction', 0.0)}"
        )

    st.markdown('### Recent Prediction Log')
    st.dataframe(
        recent[['ts_readable', 'prediction', 'confidence', 'severity', 'rul']].rename(columns={
            'ts_readable': 'Timestamp', 'prediction': 'Prediction', 'confidence': 'Confidence', 'severity': 'Severity', 'rul': 'RUL'
        }),
        width='stretch'
    )

    label_counts = predictions['prediction'].value_counts().sort_index()
    fig_dist = go.Figure([go.Bar(x=label_counts.index, y=label_counts.values, marker_color='#22d3ee')])
    fig_dist.update_layout(template='plotly_dark', title='Prediction Distribution', xaxis_title='Predicted Class', yaxis_title='Count', height=320)
    st.plotly_chart(fig_dist, width='stretch')

st.sidebar.markdown('---')
st.sidebar.write('Last refresh: ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
if st.sidebar.button('Refresh'):
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.warning("Please refresh the browser manually.")
from pathlib import Path
from PIL import Image

st.subheader("Confusion Matrix")

exp_root = Path("outputs/experiments")

if not exp_root.exists():
    st.warning("No experiment outputs found in deployment.")
else:

    runs = sorted(
        [p for p in exp_root.iterdir() if p.is_dir()],
        reverse=True
    )

    if not runs:
        st.warning("No experiment folders found.")
    else:

        latest = runs[0]

        candidates = [
            latest / "confusion_matrix_eval.png",
            latest / "assets" / "confusion_matrix.png"
        ]

        found = next((p for p in candidates if p.exists()), None)

        if found:
            st.image(
                Image.open(found),
                caption=f"Experiment: {latest.name}"
            )
        else:
            st.warning("Confusion matrix not found.")
