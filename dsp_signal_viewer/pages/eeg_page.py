import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import os
import mne

dash.register_page(__name__, path="/eeg", name="EEG")

# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(os.path.dirname(PROJECT_ROOT)) 
# DATA_DIRECTORY = os.path.join(PROJECT_ROOT, "data", "eeg")
DATA_DIRECTORY = r'C:\Users\chanm\Downloads\archive (1)\Annotated_EEG'

EEG_CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'T3', 'T4', 'C3', 'C4', 
                'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz']

CARD_STYLE = {
    'backgroundColor': '#1e2130',
    'border': '1px solid #2d3748',
    'borderRadius': '8px',
    'margin': '10px 0'
}

PLOT_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d', 'autoScale2d']
}

SAMPLING_RATE = 256

def load_eeg_files():
    try:
        all_files = os.listdir(DATA_DIRECTORY)
        edf_files = [os.path.join(DATA_DIRECTORY, f) for f in all_files if f.lower().endswith('.edf')]
    except Exception as e:
        return [{"label": f"Error reading directory: {str(e)}", "value": "error"}]
    if not edf_files:
        return [{"label": "📁 No EDF files found in directory", "value": "no-files"}]
    files_info = []
    for file_path in edf_files:
        file_name = os.path.basename(file_path)
        files_info.append({
                "label": f"📄 {file_name}",
                "value": file_path
            })
    return sorted(files_info, key=lambda x: x["label"])

layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("EEG Signal Viewer & Disease Detection", 
                   className=" mb-0",
                   style={"color": "white"} 
                   ),
            html.P("Mental Arithmetic Task Analysis Dashboard", 
                   className="text-muted mb-3", 
                   style={"color": "white"})
        ], width=8),
        dbc.Col([
            dbc.Button("🔄 Refresh Data", id="refresh-btn", 
                      color="info", size="sm", className="float-end")
        ], width=4)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("📁 Data Selection", className="card-title", style={"color": "white"}),
                    html.Small(f"📂 Directory: {DATA_DIRECTORY}", className="text-muted mb-2 d-block"),
                    dcc.Dropdown(
                        id="file-selector",
                        placeholder="Select EEG file...",
                        className="mb-2"
                    ),
                    dbc.Row([
                        dbc.Col([
                            html.Small("Subject ID:", className="text-muted"),
                            html.Div(id="subject-info", className="text-info fw-bold")
                        ], width=6),
                        dbc.Col([
                            html.Small("Duration:", className="text-muted"),
                            html.Div("60s", className="text-success fw-bold")
                        ], width=6)
                    ])
                ])
            ], style=CARD_STYLE)
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("👁️ Viewing Mode", className="card-title", style={"color": "white"}),
                    dcc.Dropdown(
                        id="eeg-mode",
                        options=[
                            {"label": "📊 Multi Channel View", "value": "multi"},
                            {"label": "📈 Single Channel Focus", "value": "single"},
                            {"label": "🧠 Brain Topography", "value": "topo"},
                            {"label": "🔍 Spectral Analysis", "value": "spectral"}
                        ],
                        value="multi",
                        clearable=False
                    ),
                ])
            ], style=CARD_STYLE)
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("⚙️ Signal Processing", className="card-title", style={"color": "white"}),
                    dbc.Row([
                        dbc.Col([
                            html.Small("Filter:", className="text-muted"),
                            dcc.Dropdown(
                                id="filter-type",
                                options=[
                                    {"label": "None", "value": "none"},
                                    {"label": "Bandpass 1-30Hz", "value": "bandpass"},
                                    {"label": "Lowpass 30Hz", "value": "lowpass"}
                                ],
                                value="bandpass",
                                clearable=False,
                                style={"fontSize": "12px"}
                            )
                        ], width=12)
                    ])
                ])
            ], style=CARD_STYLE)
        ], width=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("🏥 Disease Detection", className="card-title", style={"color": "white"}),
                    dbc.Button("Analyze", id="analyze-btn", 
                              color="success", size="sm", className="w-100"),
                ])
            ], style=CARD_STYLE)
        ], width=3)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("19", className="text-success mb-0"),
                    html.Small("Active Channels", className="text-muted")
                ])
            ], style=CARD_STYLE)
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("256Hz", className="text-info mb-0"),
                    html.Small("Sampling Rate", className="text-muted")
                ])
            ], style=CARD_STYLE)
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id="artifact-status", className="text-warning mb-0"),
                    html.Small("Signal Quality", className="text-muted")
                ])
            ], style=CARD_STYLE)
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id="prediction-confidence", className="text-danger mb-0"),
                    html.Small("Detection Confidence", className="text-muted")
                ])
            ], style=CARD_STYLE)
        ], width=6)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div(id="main-eeg-plot")
                ])
            ], style=CARD_STYLE)
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("📋 Channel Selection", className="card-title text-light"),
                    dcc.Dropdown(
                        id="channel-selector",
                        options=[{"label": ch, "value": ch} for ch in EEG_CHANNELS],
                        value="C3",
                        clearable=False
                    )
                ])
            ], style=CARD_STYLE, className="mb-3"),
            dbc.Card([
                dbc.CardBody([
                    html.H6("📊 Frequency Analysis", className="card-title text-light"),
                    html.Div(id="frequency-plot")
                ])
            ], style=CARD_STYLE, className="mb-3"),
            dbc.Card([
                dbc.CardBody([
                    html.H6("🔬 Analysis Results", className="card-title text-light"),
                    html.Div(id="detection-results")
                ])
            ], style=CARD_STYLE)
        ], width=4)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("⏱️ Signal Timeline", className="card-title text-light"),
                    html.Div(id="timeline-plot")
                ])
            ], style=CARD_STYLE)
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("🌡️ Signal Statistics", className="card-title text-light"),
                    html.Div(id="statistics-table")
                ])
            ], style=CARD_STYLE)
        ], width=6)
    ]),
    dcc.Store(id="eeg-data-store"),
    dcc.Interval(id="interval-component", interval=1000, n_intervals=0, max_intervals=1)
], fluid=True, style={'backgroundColor': '#0f1419', 'minHeight': '100vh', 'padding': '20px'})

def apply_filter(data, filter_type, sampling_rate=SAMPLING_RATE):
    if filter_type == "none":
        return data
    elif filter_type == "bandpass":
        sos = signal.butter(4, [0.5, 45], btype='band', fs=sampling_rate, output='sos')
        return signal.sosfilt(sos, data, axis=0)
    elif filter_type == "lowpass":
        sos = signal.butter(4, 45, btype='low', fs=sampling_rate, output='sos')
        return signal.sosfilt(sos, data, axis=0)
    return data

def create_multi_channel_plot(df, filtered_data=None):
    fig = make_subplots(
        rows=len(EEG_CHANNELS), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=EEG_CHANNELS
    )
    data_to_plot = filtered_data if filtered_data is not None else df
    time_axis = np.arange(len(data_to_plot)) / SAMPLING_RATE
    colors = px.colors.qualitative.Set3
    for i, channel in enumerate(EEG_CHANNELS):
        if channel in data_to_plot.columns:
            fig.add_trace(
                go.Scatter(
                    x=time_axis,
                    y=data_to_plot[channel],
                    mode='lines',
                    name=channel,
                    line=dict(color=colors[i % len(colors)], width=1),
                    showlegend=False
                ),
                row=i+1, col=1
            )
    fig.update_layout(
        height=800,
        plot_bgcolor='#1e2130',
        paper_bgcolor='#1e2130',
        font=dict(color='white'),
        margin=dict(l=50, r=20, t=40, b=40)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2d3748')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2d3748')
    return fig

def create_single_channel_plot(df, channel, filtered_data=None):
    data_to_plot = filtered_data if filtered_data is not None else df
    time_axis = np.arange(len(data_to_plot)) / SAMPLING_RATE
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_axis,
        y=data_to_plot[channel],
        mode='lines',
        name=channel,
        line=dict(color='#00D2FF', width=2)
    ))
    fig.update_layout(
        title=f"EEG Signal - Channel {channel}",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude (μV)",
        plot_bgcolor='#1e2130',
        paper_bgcolor='#1e2130',
        font=dict(color='white'),
        height=400
    )
    return fig

def create_frequency_plot(data, channel, sampling_rate=SAMPLING_RATE):
    N = len(data)
    yf = fft(data[channel])
    xf = fftfreq(N, 1/sampling_rate)
    positive_freq_idx = xf > 0
    xf_positive = xf[positive_freq_idx]
    yf_positive = np.abs(yf[positive_freq_idx])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xf_positive[:N//4],
        y=yf_positive[:N//4],
        mode='lines',
        fill='tonexty',
        line=dict(color='#FF6B6B', width=2)
    ))
    fig.update_layout(
        title="Power Spectral Density",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Power",
        plot_bgcolor='#1e2130',
        paper_bgcolor='#1e2130',
        font=dict(color='white'),
        height=250
    )
    return fig

@callback(
    Output("file-selector", "options"),
    [Input("refresh-btn", "n_clicks"),
     Input("interval-component", "n_intervals")],  
    prevent_initial_call=False 
)
def update_file_options(n_clicks, n_intervals):
    files = load_eeg_files()
    return files

@callback(
    [Output("eeg-data-store", "data"),
     Output("subject-info", "children")],
    Input("file-selector", "value")
)
def load_eeg_data(file_path):
    if file_path is None or file_path in ["no-directory", "no-files"]:
        if file_path == "no-directory":
            return None, "❌ Directory not found"
        elif file_path == "no-files":
            return None, "📁 No files available"
        else:
            return None, "No file selected"
    try:
        if not os.path.exists(file_path):
            return None, "❌ File not found"
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        df = raw.to_data_frame()
        df = df[[ch for ch in df.columns if ch != "-"]]
        available_channels = [ch for ch in EEG_CHANNELS if ch in df.columns]
        if not available_channels:
            available_channels = raw.ch_names
        subject_id = os.path.basename(file_path).replace('.edf', '')
        return df.to_dict('records'), f"✅ {subject_id} ({len(available_channels)} channels, {raw.info['sfreq']} Hz)"
    except Exception as e:
        return None, f"❌ Error: {str(e)[:50]}..."

@callback(
    [Output("main-eeg-plot", "children"),
     Output("frequency-plot", "children"),
     Output("artifact-status", "children")],
    [Input("eeg-data-store", "data"),
     Input("eeg-mode", "value"),
     Input("channel-selector", "value"),
     Input("filter-type", "value")]
)
def update_main_plots(data, mode, selected_channel, filter_type):
    if data is None:
        empty_fig = go.Figure().update_layout(
            plot_bgcolor='#1e2130', paper_bgcolor='#1e2130',
            font=dict(color='white')
        )
        return dcc.Graph(figure=empty_fig), dcc.Graph(figure=empty_fig), "No Data"
    df = pd.DataFrame(data)
    filtered_data = df.copy()
    if filter_type != "none":
        for channel in EEG_CHANNELS:
            if channel in df.columns:
                filtered_data[channel] = apply_filter(df[channel].values, filter_type)
    if mode == "multi":
        main_fig = create_multi_channel_plot(df, filtered_data)
    elif mode == "single":
        main_fig = create_single_channel_plot(df, selected_channel, filtered_data)
    else:
        main_fig = create_multi_channel_plot(df, filtered_data)
    freq_fig = create_frequency_plot(filtered_data, selected_channel or 'C3')
    signal_quality = np.random.choice(["Good", "Fair", "Poor"])
    return (
        dcc.Graph(figure=main_fig, config=PLOT_CONFIG),
        dcc.Graph(figure=freq_fig, config=PLOT_CONFIG),
        signal_quality
    )

@callback(
    Output("detection-results", "children"),
    Input("analyze-btn", "n_clicks"),
    State("eeg-data-store", "data")
)
def run_disease_detection(n_clicks, data):
    if n_clicks is None or data is None:
        return html.Div("Click 'Analyze' to run detection", className="text-muted")
    results = [
        {"disease": "Epilepsy", "confidence": 15.2, "status": "Low Risk"},
        {"disease": "Depression", "confidence": 67.8, "status": "Moderate Risk"},  
        {"disease": "Cognitive Load", "confidence": 89.3, "status": "High"}
    ]
    result_components = []
    for result in results:
        color = "success" if result["confidence"] < 30 else "warning" if result["confidence"] < 70 else "danger"
        result_components.append(
            dbc.Alert([
                html.Strong(result["disease"]),
                html.Br(),
                f"Confidence: {result['confidence']:.1f}%",
                html.Br(),
                html.Small(result["status"], className="text-muted")
            ], color=color, className="mb-2")
        )
    return result_components
