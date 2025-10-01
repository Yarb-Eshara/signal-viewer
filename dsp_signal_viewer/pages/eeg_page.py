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

DATA_DIRECTORY = r'C:\Users\chanm\Downloads\archive (1)\Annotated_EEG'

# This will be updated dynamically from the loaded file
EEG_CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'T3', 'T4', 'C3', 'C4', 
                'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz']

CARD_STYLE = {
    'backgroundColor': '#1e2130',
    'border': '1px solid #2d3748',
    'borderRadius': '8px',
    'margin': '10px 0'
}

PLOT_CONFIG = {
    'displayModeBar': False,  # Hide toolbar for better performance
    'displaylogo': False
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
                    html.H4(id="channel-count", className="text-success mb-0"),
                    html.Small("Active Channels", className="text-muted")
                ])
            ], style=CARD_STYLE)
        ], width=2),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id="sampling-rate-display", className="text-info mb-0"),
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
                        options=[],  # Will be populated dynamically
                        placeholder="Select a channel...",
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
    dcc.Store(id="eeg-metadata-store"),  # NEW: Store metadata separately
    dcc.Interval(id="interval-component", interval=1000, n_intervals=0, max_intervals=1)
], fluid=True, style={'backgroundColor': '#0f1419', 'minHeight': '100vh', 'padding': '20px'})

def apply_filter(data, filter_type, sampling_rate=SAMPLING_RATE):
    """Apply filter to signal data"""
    if filter_type == "none":
        return data
    elif filter_type == "bandpass":
        sos = signal.butter(4, [0.5, 45], btype='band', fs=sampling_rate, output='sos')
        return signal.sosfilt(sos, data, axis=0)
    elif filter_type == "lowpass":
        sos = signal.butter(4, 45, btype='low', fs=sampling_rate, output='sos')
        return signal.sosfilt(sos, data, axis=0)
    return data

def create_multi_channel_plot(df, channels, filtered_data=None, sampling_rate=SAMPLING_RATE):
    """Create multi-channel EEG plot"""
    # Limit to first 10 channels if too many
    display_channels = channels[:10] if len(channels) > 10 else channels
    
    # OPTIMIZATION: Use Scattergl for hardware-accelerated rendering
    fig = make_subplots(
        rows=len(display_channels), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.015,
        subplot_titles=display_channels
    )
    data_to_plot = filtered_data if filtered_data is not None else df
    time_axis = np.arange(len(data_to_plot)) / sampling_rate
    colors = px.colors.qualitative.Set3
    
    for i, channel in enumerate(display_channels):
        if channel in data_to_plot.columns:
            signal_data = data_to_plot[channel].values
            
            # Use Scattergl for better performance
            fig.add_trace(
                go.Scattergl(  # Changed from Scatter to Scattergl
                    x=time_axis,
                    y=signal_data,
                    mode='lines',
                    name=channel,
                    line=dict(color=colors[i % len(colors)], width=1),
                    showlegend=False
                ),
                row=i+1, col=1
            )
    
    if len(channels) > 10:
        note = f"Displaying first 10 of {len(channels)} channels"
    else:
        note = f"{len(channels)} channels"
    
    fig.update_layout(
        height=max(500, len(display_channels) * 60),
        plot_bgcolor='#1e2130',
        paper_bgcolor='#1e2130',
        font=dict(color='white', size=10),
        margin=dict(l=80, r=20, t=60, b=40),
        title=dict(text=note, font=dict(size=12), x=0.5, xanchor='center')
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2d3748', title_text="Time (s)", row=len(display_channels))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2d3748', title_text="μV", title_standoff=5)
    
    # Make subplot titles smaller
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=9)
    
    return fig

def create_single_channel_plot(df, channel, filtered_data=None, sampling_rate=SAMPLING_RATE):
    """Create single channel EEG plot"""
    data_to_plot = filtered_data if filtered_data is not None else df
    time_axis = np.arange(len(data_to_plot)) / sampling_rate
    signal_data = data_to_plot[channel].values
    
    fig = go.Figure()
    # Use Scattergl for better performance
    fig.add_trace(go.Scattergl(
        x=time_axis,
        y=signal_data,
        mode='lines',
        name=channel,
        line=dict(color='#00D2FF', width=1.5)
    ))
    
    fig.update_layout(
        title=f"EEG Signal - Channel {channel}",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude (μV)",
        plot_bgcolor='#1e2130',
        paper_bgcolor='#1e2130',
        font=dict(color='white'),
        height=400,
        margin=dict(l=60, r=20, t=60, b=60)
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2d3748')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2d3748')
    return fig

def create_frequency_plot(data, channel, sampling_rate=SAMPLING_RATE):
    """Create frequency domain plot"""
    if channel not in data.columns:
        # Return empty plot if channel not found
        fig = go.Figure()
        fig.update_layout(
            title="Channel not found",
            plot_bgcolor='#1e2130',
            paper_bgcolor='#1e2130',
            font=dict(color='white'),
            height=250
        )
        return fig
    
    N = len(data)
    yf = fft(data[channel].values)
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
    """Update file dropdown options"""
    files = load_eeg_files()
    return files

@callback(
    Output("channel-selector", "value"),
    Input("channel-selector", "options"),
    prevent_initial_call=True
)
def set_default_channel(options):
    """Set default channel when options are updated"""
    if options and len(options) > 0:
        return options[0]["value"]
    return None

@callback(
    [Output("eeg-data-store", "data"),
     Output("eeg-metadata-store", "data"),
     Output("subject-info", "children"),
     Output("channel-selector", "options")],
    Input("file-selector", "value")
)
def load_eeg_data(file_path):
    """Load EEG data from selected file"""
    if file_path is None or file_path in ["no-directory", "no-files", "error"]:
        if file_path == "no-directory":
            return None, None, "❌ Directory not found", []
        elif file_path == "no-files":
            return None, None, "📁 No files available", []
        else:
            return None, None, "No file selected", []
    
    try:
        if not os.path.exists(file_path):
            return None, None, "❌ File not found", []
        
        print(f"\n{'='*60}")
        print(f"Loading file: {file_path}")
        
        # Load EDF file
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        
        # Get sampling rate
        sfreq = float(raw.info['sfreq'])
        
        # CRITICAL OPTIMIZATION: Downsample data to reduce size
        # Target: max 5000 points per channel for smooth performance
        n_samples = len(raw.times)
        target_samples = 5000
        
        if n_samples > target_samples:
            # Calculate decimation factor
            decim_factor = n_samples // target_samples
            print(f"Downsampling from {n_samples} to ~{n_samples//decim_factor} samples (factor: {decim_factor})")
            raw = raw.resample(sfreq / decim_factor)
            sfreq = float(raw.info['sfreq'])
        
        # Convert to DataFrame
        df = raw.to_data_frame()
        
        print(f"Original columns: {df.columns.tolist()}")
        print(f"Optimized shape: {df.shape}")
        
        # Remove the 'time' column if it exists
        if 'time' in df.columns:
            df = df.drop('time', axis=1)
        
        # Handle duplicate column names by making them unique
        cols = df.columns.tolist()
        seen = {}
        new_cols = []
        for col in cols:
            if col in seen:
                seen[col] += 1
                new_cols.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                new_cols.append(col)
        df.columns = new_cols
        
        # Get actual channel names from the file (excluding empty strings)
        actual_channels = [ch for ch in df.columns if ch and ch.strip()]
        
        print(f"Processed channels ({len(actual_channels)}): {actual_channels}")
        print(f"Final data shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"{'='*60}\n")
        
        if not actual_channels:
            return None, None, "❌ No valid channels found", []
        
        # Store metadata
        metadata = {
            'channels': actual_channels,
            'sampling_rate': sfreq,
            'duration': float(raw.times[-1]),
            'original_samples': n_samples,
            'stored_samples': len(df)
        }
        
        subject_id = os.path.basename(file_path).replace('.edf', '').replace('_annotated', '')
        
        # Update channel selector options with cleaner labels
        channel_options = [{"label": f"📡 {ch}", "value": ch} for ch in actual_channels]
        
        # CRITICAL: Use 'split' orientation for much faster serialization
        return (
            df.to_dict('split'), 
            metadata,
            f"✅ {subject_id} ({len(actual_channels)} ch, {int(sfreq)} Hz, {int(raw.times[-1])}s)",
            channel_options
        )
        
    except Exception as e:
        import traceback
        print(f"Error loading file: {str(e)}")
        print(traceback.format_exc())
        return None, None, f"❌ Error: {str(e)[:50]}...", []

@callback(
    [Output("main-eeg-plot", "children"),
     Output("frequency-plot", "children"),
     Output("artifact-status", "children"),
     Output("channel-count", "children"),
     Output("sampling-rate-display", "children")],
    [Input("eeg-data-store", "data"),
     Input("eeg-metadata-store", "data"),
     Input("eeg-mode", "value"),
     Input("channel-selector", "value"),
     Input("filter-type", "value")]
)
def update_main_plots(data, metadata, mode, selected_channel, filter_type):
    """Update main plots based on data and settings"""
    if data is None or metadata is None:
        empty_fig = go.Figure().update_layout(
            plot_bgcolor='#1e2130', paper_bgcolor='#1e2130',
            font=dict(color='white'),
            annotations=[dict(
                text="No data loaded. Please select a file.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14, color="white")
            )]
        )
        return (
            dcc.Graph(figure=empty_fig), 
            dcc.Graph(figure=empty_fig), 
            "No Data",
            "0",
            "0 Hz"
        )
    
    # OPTIMIZATION: Use 'split' format for much faster DataFrame reconstruction
    df = pd.DataFrame(data['data'], columns=data['columns'])
    channels = metadata['channels']
    sampling_rate = metadata['sampling_rate']
    
    print(f"Plotting {len(df)} samples across {len(channels)} channels")
    
    # OPTIMIZATION: Only filter if needed and cache results
    filtered_data = df.copy()
    if filter_type != "none":
        print(f"Applying {filter_type} filter...")
        for channel in channels:
            if channel in df.columns:
                try:
                    filtered_data[channel] = apply_filter(
                        df[channel].values, 
                        filter_type, 
                        sampling_rate
                    )
                except Exception as e:
                    print(f"Error filtering channel {channel}: {str(e)}")
    
    # Ensure selected channel exists
    if selected_channel not in channels:
        selected_channel = channels[0] if channels else None
    
    # Create main plot based on mode
    if mode == "multi":
        main_fig = create_multi_channel_plot(df, channels, filtered_data, sampling_rate)
    elif mode == "single" and selected_channel:
        main_fig = create_single_channel_plot(df, selected_channel, filtered_data, sampling_rate)
    else:
        main_fig = create_multi_channel_plot(df, channels, filtered_data, sampling_rate)
    
    # Create frequency plot
    if selected_channel and selected_channel in filtered_data.columns:
        freq_fig = create_frequency_plot(filtered_data, selected_channel, sampling_rate)
    else:
        freq_fig = create_frequency_plot(filtered_data, channels[0], sampling_rate)
    
    # Calculate signal quality (basic check)
    signal_quality = "Good"
    try:
        if selected_channel and selected_channel in filtered_data.columns:
            std_val = filtered_data[selected_channel].std()
            if std_val > 100:
                signal_quality = "Poor"
            elif std_val > 50:
                signal_quality = "Fair"
    except:
        signal_quality = "Unknown"
    
    return (
        dcc.Graph(figure=main_fig, config=PLOT_CONFIG),
        dcc.Graph(figure=freq_fig, config=PLOT_CONFIG),
        signal_quality,
        str(len(channels)),
        f"{int(sampling_rate)} Hz"
    )

@callback(
    Output("detection-results", "children"),
    Input("analyze-btn", "n_clicks"),
    State("eeg-data-store", "data")
)
def run_disease_detection(n_clicks, data):
    """Run disease detection analysis"""
    if n_clicks is None or data is None:
        return html.Div("Click 'Analyze' to run detection", className="text-muted")
    
    # Mock results - replace with actual ML model
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