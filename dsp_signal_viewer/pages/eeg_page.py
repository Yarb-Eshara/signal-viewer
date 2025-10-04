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
from warnings import filterwarnings
filterwarnings("ignore")
import torch
import torch.nn as nn
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

dash.register_page(__name__, path="/eeg", name="EEG")

# DATA_DIRECTORY = r'C:\Users\chanm\Downloads\archive (1)\Annotated_EEG'
DATA_DIRECTORY = r'data/Annotated_EEG'

EEG_CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'F7', 'F8', 'T3', 'T4', 'C3', 'C4', 
                'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz']

CARD_STYLE = {
    'backgroundColor': '#1e2130',
    'border': '1px solid #2d3748',
    'borderRadius': '8px',
    'margin': '10px 0'
}

PLOT_CONFIG = {
    'displayModeBar': False,
    'displaylogo': False
}

SAMPLING_RATE = 256

# ============================================================================
# DETECTION MODEL CLASSES
# ============================================================================

class CHBMITPreprocessor:
    """Preprocessing for EEG data"""
    
    def __init__(self, fs=256):
        self.fs = fs
        self.standard_channels = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
            'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4', 
            'T3', 'T4', 'T5', 'T6', 'O1', 'O2'
        ]
        self.scaler = StandardScaler()
    
    def map_channels_to_standard(self, data):
        """Map available channels to standard 19 channels"""
        mapped_data = pd.DataFrame()
        available_channels = data.columns.tolist()
        
        # Simple mapping - use available channels or fill with zeros
        for std_ch in self.standard_channels[:min(19, len(available_channels))]:
            if std_ch in available_channels:
                mapped_data[std_ch] = data[std_ch]
            else:
                # Try to find similar channel names
                similar = [ch for ch in available_channels if std_ch.lower() in ch.lower()]
                if similar:
                    mapped_data[std_ch] = data[similar[0]]
                else:
                    mapped_data[std_ch] = 0.0
        
        # Fill remaining channels with zeros if needed
        for std_ch in self.standard_channels[len(mapped_data.columns):]:
            mapped_data[std_ch] = 0.0
        
        return mapped_data
    
    def segment_data(self, data, window_size=1024, overlap=0.5):
        """Segment data into windows"""
        step_size = int(window_size * (1 - overlap))
        segments = []
        
        for i in range(0, len(data) - window_size + 1, step_size):
            segment = data.iloc[i:i + window_size]
            segments.append(segment.values)
        
        return np.array(segments) if segments else np.array([data.values])

class SeizureDetector:
    """Seizure Detection Model"""
    
    def __init__(self):
        self.model = None
        self.is_loaded = False
    
    def extract_features(self, segments):
        """Extract seizure-specific features"""
        features = []
        
        for segment in segments:
            segment_features = []
            
            for ch in range(min(segment.shape[1], 19)):
                signal_data = segment[:, ch]
                
                # Statistical features
                variance = np.var(signal_data)
                
                # Frequency features
                fft_vals = np.abs(fft(signal_data))
                freqs = fftfreq(len(signal_data), 1/256)
                
                # Band powers
                delta = np.sum(fft_vals[(freqs >= 1) & (freqs <= 4)])
                theta = np.sum(fft_vals[(freqs >= 4) & (freqs <= 8)])
                alpha = np.sum(fft_vals[(freqs >= 8) & (freqs <= 13)])
                beta = np.sum(fft_vals[(freqs >= 13) & (freqs <= 30)])
                gamma = np.sum(fft_vals[(freqs >= 30) & (freqs <= 100)])
                
                high_freq_ratio = (beta + gamma) / (delta + theta + alpha + 1e-8)
                
                segment_features.extend([variance, high_freq_ratio])
            
            features.append(segment_features)
        
        return np.array(features)
    
    def predict(self, segments):
        """Predict seizure probability"""
        features = self.extract_features(segments)
        predictions = []
        
        # Collect all features to calculate normalization parameters
        all_variance = []
        all_high_freq = []
        
        for feat_vec in features:
            variance_scores = feat_vec[::2]
            high_freq_scores = feat_vec[1::2]
            all_variance.extend(variance_scores)
            all_high_freq.extend(high_freq_scores)
        
        # Calculate robust normalization thresholds
        variance_75th = np.percentile(all_variance, 75) if all_variance else 1.0
        high_freq_75th = np.percentile(all_high_freq, 75) if all_high_freq else 1.0
        
        # Avoid division by zero
        variance_75th = max(variance_75th, 0.01)
        high_freq_75th = max(high_freq_75th, 0.01)
        
        for feat_vec in features:
            variance_score = np.mean(feat_vec[::2])
            high_freq_score = np.mean(feat_vec[1::2])
            
            # Normalize by 75th percentile and scale
            normalized_variance = min(1.0, variance_score / (variance_75th * 3))
            normalized_high_freq = min(1.0, high_freq_score / (high_freq_75th * 3))
            
            # More conservative scoring
            seizure_score = (
                0.3 * normalized_variance +
                0.4 * normalized_high_freq
            )
            
            # Add baseline noise threshold
            seizure_score = max(0.1, min(0.9, seizure_score))
            
            predictions.append([1-seizure_score, seizure_score])
        
        return np.array(predictions)

class AlzheimerDetector:
    """Alzheimer's Detection Model"""
    
    def extract_features(self, segments):
        """Extract Alzheimer's-specific features"""
        features = []
        
        for segment in segments:
            segment_features = []
            
            for ch in range(min(segment.shape[1], 19)):
                signal_data = segment[:, ch]
                freqs = fftfreq(len(signal_data), 1/256)
                psd = np.abs(fft(signal_data))
                
                # Band powers
                delta = np.sum(psd[(freqs >= 1) & (freqs <= 4)])
                theta = np.sum(psd[(freqs >= 4) & (freqs <= 8)])
                alpha = np.sum(psd[(freqs >= 8) & (freqs <= 13)])
                beta = np.sum(psd[(freqs >= 13) & (freqs <= 30)])
                
                theta_alpha_ratio = theta / (alpha + 1e-8)
                
                # Spectral entropy
                psd_norm = psd / (np.sum(psd) + 1e-8)
                spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-8))
                
                segment_features.extend([theta_alpha_ratio, spectral_entropy])
            
            features.append(segment_features)
        
        return np.array(features)
    
    def predict(self, segments):
        """Predict Alzheimer's probability"""
        features = self.extract_features(segments)
        predictions = []
        
        # Collect all features for normalization
        all_theta_alpha = []
        all_complexity = []
        
        for feat_vec in features:
            theta_alpha_scores = feat_vec[::2]
            complexity_scores = feat_vec[1::2]
            all_theta_alpha.extend(theta_alpha_scores)
            all_complexity.extend(complexity_scores)
        
        # Calculate median values as reference points
        median_theta_alpha = np.median(all_theta_alpha) if all_theta_alpha else 1.0
        median_complexity = np.median(all_complexity) if all_complexity else 4.0
        
        for feat_vec in features:
            theta_alpha_scores = feat_vec[::2]
            complexity_scores = feat_vec[1::2]
            
            avg_theta_alpha = np.mean(theta_alpha_scores)
            avg_complexity = np.mean(complexity_scores)
            
            # Compare to baseline: elevated theta/alpha ratio indicates AD
            theta_alpha_deviation = (avg_theta_alpha - median_theta_alpha) / (median_theta_alpha + 0.1)
            theta_alpha_component = min(1.0, max(0.0, theta_alpha_deviation))
            
            # Lower complexity indicates AD
            complexity_component = max(0.0, (median_complexity - avg_complexity) / median_complexity)
            
            # Conservative scoring
            alzheimer_score = (
                0.4 * theta_alpha_component +
                0.3 * complexity_component
            )
            
            # Add baseline and cap
            alzheimer_score = max(0.05, min(0.85, alzheimer_score))
            
            predictions.append([1-alzheimer_score, alzheimer_score])
        
        return np.array(predictions)

class ParkinsonDetector:
    """Parkinson's Detection Model"""
    
    def extract_features(self, segments):
        """Extract Parkinson's-specific features"""
        features = []
        
        for segment in segments:
            segment_features = []
            
            for ch in range(min(segment.shape[1], 19)):
                signal_data = segment[:, ch]
                freqs = fftfreq(len(signal_data), 1/256)
                psd = np.abs(fft(signal_data))
                
                # Band powers
                delta = np.sum(psd[(freqs >= 1) & (freqs <= 4)])
                theta = np.sum(psd[(freqs >= 4) & (freqs <= 8)])
                alpha = np.sum(psd[(freqs >= 8) & (freqs <= 13)])
                beta = np.sum(psd[(freqs >= 13) & (freqs <= 30)])
                
                total = delta + theta + alpha + beta + 1e-8
                
                beta_ratio = beta / total
                tremor_ratio = theta / total
                
                segment_features.extend([beta_ratio, tremor_ratio])
            
            features.append(segment_features)
        
        return np.array(features)
    
    def predict(self, segments):
        """Predict Parkinson's probability"""
        features = self.extract_features(segments)
        predictions = []
        
        # Collect all features for normalization
        all_beta = []
        all_tremor = []
        
        for feat_vec in features:
            beta_ratios = feat_vec[::2]
            tremor_ratios = feat_vec[1::2]
            all_beta.extend(beta_ratios)
            all_tremor.extend(tremor_ratios)
        
        # Calculate median reference values
        median_beta = np.median(all_beta) if all_beta else 0.2
        median_tremor = np.median(all_tremor) if all_tremor else 0.15
        
        for feat_vec in features:
            beta_ratios = feat_vec[::2]
            tremor_ratios = feat_vec[1::2]
            
            avg_beta = np.mean(beta_ratios)
            avg_tremor = np.mean(tremor_ratios)
            
            # Beta suppression (lower than normal indicates PD)
            beta_suppression = max(0.0, (median_beta - avg_beta) / (median_beta + 0.01))
            
            # Elevated tremor activity
            tremor_elevation = max(0.0, (avg_tremor - median_tremor) / (median_tremor + 0.01))
            
            # Conservative scoring
            parkinson_score = (
                0.35 * min(1.0, beta_suppression) +
                0.35 * min(1.0, tremor_elevation)
            )
            
            # Add baseline and cap
            parkinson_score = max(0.05, min(0.85, parkinson_score))
            
            predictions.append([1-parkinson_score, parkinson_score])
        
        return np.array(predictions)

# Add this import at the top with other imports
import base64
import io
import tempfile

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_eeg_files():
    try:
        all_files = os.listdir(DATA_DIRECTORY)
        edf_files = [os.path.join(DATA_DIRECTORY, f) for f in all_files if f.lower().endswith('.edf')]
    except Exception as e:
        return [{"label": f"Error reading directory: {str(e)}", "value": "error"}]
    
    if not edf_files:
        return [{"label": "No EDF files found", "value": "no-files"}]
    
    files_info = []
    for file_path in edf_files:
        file_name = os.path.basename(file_path)
        files_info.append({
            "label": f"📄 {file_name}",
            "value": file_path
        })
    
    return sorted(files_info, key=lambda x: x["label"])

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
    display_channels = channels[:10] if len(channels) > 10 else channels
    
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
            
            fig.add_trace(
                go.Scattergl(
                    x=time_axis,
                    y=signal_data,
                    mode='lines',
                    name=channel,
                    line=dict(color=colors[i % len(colors)], width=1),
                    showlegend=False
                ),
                row=i+1, col=1
            )
    
    fig.update_layout(
        height=max(500, len(display_channels) * 60),
        plot_bgcolor='#1e2130',
        paper_bgcolor='#1e2130',
        font=dict(color='white', size=10),
        margin=dict(l=80, r=20, t=60, b=40),
        title=dict(text=f"Multi-Channel View ({len(display_channels)} channels)", 
                   font=dict(size=12), x=0.5, xanchor='center')
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2d3748', 
                     title_text="Time (s)", row=len(display_channels))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2d3748', 
                     title_text="μV", title_standoff=5)
    
    return fig

def create_single_channel_plot(df, channel, filtered_data=None, sampling_rate=SAMPLING_RATE):
    """Create single channel EEG plot"""
    data_to_plot = filtered_data if filtered_data is not None else df
    
    if channel not in data_to_plot.columns:
        # Fallback to first channel if selected channel not found
        channel = data_to_plot.columns[0]
    
    time_axis = np.arange(len(data_to_plot)) / sampling_rate
    signal_data = data_to_plot[channel].values
    
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=time_axis,
        y=signal_data,
        mode='lines',
        name=channel,
        line=dict(color='#00D2FF', width=1.5)
    ))
    
    fig.update_layout(
        title=f"Single Channel View - {channel}",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude (μV)",
        plot_bgcolor='#1e2130',
        paper_bgcolor='#1e2130',
        font=dict(color='white'),
        height=600,
        margin=dict(l=60, r=20, t=60, b=60)
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2d3748')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2d3748')
    
    return fig

def create_frequency_plot(data, channel, sampling_rate=SAMPLING_RATE):
    """Create frequency domain plot"""
    if channel not in data.columns:
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

# ============================================================================
# DASH LAYOUT
# ============================================================================

layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("EEG Signal Viewer & Disease Detection", 
                   className="mb-0",
                   style={"color": "white"}),
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
                        placeholder="Select EEG file from directory...",
                        className="mb-2"
                    ),
                    html.Div([
                        html.Small("Or drag & drop EDF file:", className="text-muted d-block mb-1"),
                        dcc.Upload(
                            id='upload-edf',
                            children=html.Div([
                                '📤 Drag and Drop or ',
                                html.A('Select EDF File', style={'color': '#00D2FF', 'cursor': 'pointer'})
                            ]),
                            style={
                                'width': '100%',
                                'height': '50px',
                                'lineHeight': '50px',
                                'borderWidth': '2px',
                                'borderStyle': 'dashed',
                                'borderRadius': '8px',
                                'borderColor': '#2d3748',
                                'textAlign': 'center',
                                'backgroundColor': '#0f1419',
                                'color': 'white',
                                'cursor': 'pointer',
                                'fontSize': '12px'
                            },
                            multiple=False,
                            accept='.edf'
                        )
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col([
                            html.Small("Subject ID:", className="text-muted"),
                            html.Div(id="subject-info", className="text-info fw-bold")
                        ], width=6),
                        dbc.Col([
                            html.Small("Duration:", className="text-muted"),
                            html.Div(id="duration-info", className="text-success fw-bold")
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
                            {"label": "📈 Single Channel Focus", "value": "single"}
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
                    dcc.Dropdown(
                        id="filter-type",
                        options=[
                            {"label": "None", "value": "none"},
                            {"label": "Bandpass 1-30Hz", "value": "bandpass"},
                            {"label": "Lowpass 30Hz", "value": "lowpass"}
                        ],
                        value="bandpass",
                        clearable=False
                    )
                ])
            ], style=CARD_STYLE)
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("🏥 Disease Detection", className="card-title", style={"color": "white"}),
                    dbc.Button("Analyze", id="analyze-btn", 
                              color="success", size="sm", className="w-100",
                              n_clicks=0),
                    dbc.Spinner(html.Div(id="analyze-status"), size="sm", color="success")
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
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id="sampling-rate-display", className="text-info mb-0"),
                    html.Small("Sampling Rate", className="text-muted")
                ])
            ], style=CARD_STYLE)
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id="artifact-status", className="text-warning mb-0"),
                    html.Small("Signal Quality", className="text-muted")
                ])
            ], style=CARD_STYLE)
        ], width=3),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4(id="prediction-confidence", className="text-danger mb-0"),
                    html.Small("Detection Status", className="text-muted")
                ])
            ], style=CARD_STYLE)
        ], width=3)
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
                        options=[],
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
    ]),
    
    dcc.Store(id="eeg-data-store"),
    dcc.Store(id="eeg-metadata-store"),
    dcc.Store(id="detection-results-store"),
    dcc.Store(id="uploaded-file-store"),  # NEW: Store uploaded file data
    dcc.Interval(id="interval-component", interval=1000, n_intervals=0, max_intervals=1)
], fluid=True, style={'backgroundColor': '#0f1419', 'minHeight': '100vh', 'padding': '20px'})

# ============================================================================
# CALLBACKS
# ============================================================================

@callback(
    Output("file-selector", "options"),
    [Input("refresh-btn", "n_clicks"),
     Input("interval-component", "n_intervals")],
    prevent_initial_call=False
)
def update_file_options(n_clicks, n_intervals):
    """Update file dropdown options"""
    return load_eeg_files()

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
    [Output("uploaded-file-store", "data"),
     Output("file-selector", "value")],
    Input("upload-edf", "contents"),
    State("upload-edf", "filename"),
    prevent_initial_call=True
)
def handle_file_upload(contents, filename):
    """Handle uploaded EDF file"""
    if contents is None:
        return None, None
    
    try:
        # Parse the uploaded file
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Check if it's an EDF file
        if not filename.lower().endswith('.edf'):
            return None, None
        
        # Store the file data
        uploaded_data = {
            'filename': filename,
            'content': content_string,  # Store base64 encoded content
            'is_uploaded': True
        }
        
        return uploaded_data, None  # Clear dropdown selection
        
    except Exception as e:
        print(f"Error handling upload: {e}")
        return None, None

@callback(
    [Output("eeg-data-store", "data"),
     Output("eeg-metadata-store", "data"),
     Output("subject-info", "children"),
     Output("duration-info", "children"),
     Output("channel-selector", "options")],
    [Input("file-selector", "value"),
     Input("uploaded-file-store", "data")]
)
def load_eeg_data(file_path, uploaded_data):
    """Load EEG data from selected file or uploaded file"""
    
    # Determine which input triggered the callback
    ctx = dash.callback_context
    if not ctx.triggered:
        return None, None, "No file selected", "N/A", []
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Handle uploaded file
    if trigger_id == "uploaded-file-store" and uploaded_data is not None:
        try:
            # Decode the uploaded file
            content_string = uploaded_data['content']
            decoded = base64.b64decode(content_string)
            filename = uploaded_data['filename']
            
            # Create a temporary file to load with MNE
            with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp_file:
                tmp_file.write(decoded)
                tmp_file_path = tmp_file.name
            
            try:
                # Load EDF file
                raw = mne.io.read_raw_edf(tmp_file_path, preload=True, verbose=False)
                sfreq = float(raw.info['sfreq'])
                
                # Downsample if needed
                n_samples = len(raw.times)
                target_samples = 5000
                
                if n_samples > target_samples:
                    decim_factor = n_samples // target_samples
                    raw = raw.resample(sfreq / decim_factor)
                    sfreq = float(raw.info['sfreq'])
                
                # Convert to DataFrame
                df = raw.to_data_frame()
                
                if 'time' in df.columns:
                    df = df.drop('time', axis=1)
                
                # Handle duplicate columns
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
                
                actual_channels = [ch for ch in df.columns if ch and ch.strip()]
                
                if not actual_channels:
                    return None, None, "❌ No valid channels", "N/A", []
                
                duration = float(raw.times[-1])
                
                metadata = {
                    'channels': actual_channels,
                    'sampling_rate': sfreq,
                    'duration': duration,
                    'original_samples': n_samples,
                    'stored_samples': len(df)
                }
                
                subject_id = filename.replace('.edf', '').replace('_annotated', '')
                duration_str = f"{int(duration)}s"
                
                channel_options = [{"label": f"📡 {ch}", "value": ch} for ch in actual_channels]
                
                return (
                    df.to_dict('split'),
                    metadata,
                    f"📤 {subject_id}",
                    duration_str,
                    channel_options
                )
                
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                
        except Exception as e:
            return None, None, f"❌ Upload Error: {str(e)[:30]}", "N/A", []
    
    # Handle file from directory (existing logic)
    if file_path is None or file_path in ["no-directory", "no-files", "error"]:
        return None, None, "No file selected", "N/A", []
    
    try:
        if not os.path.exists(file_path):
            return None, None, "❌ File not found", "N/A", []
        
        # Load EDF file
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        sfreq = float(raw.info['sfreq'])
        
        # Downsample if needed
        n_samples = len(raw.times)
        target_samples = 5000
        
        if n_samples > target_samples:
            decim_factor = n_samples // target_samples
            raw = raw.resample(sfreq / decim_factor)
            sfreq = float(raw.info['sfreq'])
        
        # Convert to DataFrame
        df = raw.to_data_frame()
        
        if 'time' in df.columns:
            df = df.drop('time', axis=1)
        
        # Handle duplicate columns
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
        
        actual_channels = [ch for ch in df.columns if ch and ch.strip()]
        
        if not actual_channels:
            return None, None, "❌ No valid channels", "N/A", []
        
        duration = float(raw.times[-1])
        
        metadata = {
            'channels': actual_channels,
            'sampling_rate': sfreq,
            'duration': duration,
            'original_samples': n_samples,
            'stored_samples': len(df)
        }
        
        subject_id = os.path.basename(file_path).replace('.edf', '').replace('_annotated', '')
        duration_str = f"{int(duration)}s"
        
        channel_options = [{"label": f"📡 {ch}", "value": ch} for ch in actual_channels]
        
        return (
            df.to_dict('split'),
            metadata,
            f"✅ {subject_id}",
            duration_str,
            channel_options
        )
        
    except Exception as e:
        return None, None, f"❌ Error: {str(e)[:30]}", "N/A", []

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
        return dcc.Graph(figure=empty_fig), dcc.Graph(figure=empty_fig), "No Data", "0", "0 Hz"
    
    df = pd.DataFrame(data['data'], columns=data['columns'])
    channels = metadata['channels']
    sampling_rate = metadata['sampling_rate']
    
    # Apply filtering
    filtered_data = df.copy()
    if filter_type != "none":
        for channel in channels:
            if channel in df.columns:
                try:
                    filtered_data[channel] = apply_filter(
                        df[channel].values, filter_type, sampling_rate
                    )
                except:
                    pass
    
    if selected_channel not in channels:
        selected_channel = channels[0] if channels else None
    
    # Create plots
    if mode == "multi":
        main_fig = create_multi_channel_plot(df, channels, filtered_data, sampling_rate)
    elif mode == "single" and selected_channel:
        main_fig = create_single_channel_plot(df, selected_channel, filtered_data, sampling_rate)
    else:
        # Fallback to multi-channel if single mode but no channel selected
        main_fig = create_multi_channel_plot(df, channels, filtered_data, sampling_rate)
    
    if selected_channel and selected_channel in filtered_data.columns:
        freq_fig = create_frequency_plot(filtered_data, selected_channel, sampling_rate)
    else:
        freq_fig = create_frequency_plot(filtered_data, channels[0], sampling_rate)
    
    # Signal quality
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
    [Output("detection-results", "children"),
     Output("prediction-confidence", "children"),
     Output("analyze-status", "children"),
     Output("detection-results-store", "data")],
    Input("analyze-btn", "n_clicks"),
    [State("eeg-data-store", "data"),
     State("eeg-metadata-store", "data")],
    prevent_initial_call=True
)
def run_disease_detection(n_clicks, data, metadata):
    """Run disease detection analysis when button is clicked"""
    if n_clicks == 0 or data is None or metadata is None:
        return (
            html.Div("Click 'Analyze' to run detection", className="text-muted"),
            "Not Analyzed",
            "",
            None
        )
    
    try:
        # Reconstruct DataFrame
        df = pd.DataFrame(data['data'], columns=data['columns'])
        
        # Initialize preprocessor and detectors
        preprocessor = CHBMITPreprocessor()
        seizure_detector = SeizureDetector()
        alzheimer_detector = AlzheimerDetector()
        parkinson_detector = ParkinsonDetector()
        
        # Map channels and segment data
        mapped_data = preprocessor.map_channels_to_standard(df)
        segments = preprocessor.segment_data(mapped_data, window_size=1024, overlap=0.5)
        
        # Run predictions
        seizure_preds = seizure_detector.predict(segments)
        alzheimer_preds = alzheimer_detector.predict(segments)
        parkinson_preds = parkinson_detector.predict(segments)
        
        # Calculate results
        results = {
            'seizure': {
                'probability': float(np.mean(seizure_preds[:, 1])),
                'segments_detected': int(np.sum(seizure_preds[:, 1] > 0.5)),
                'total_segments': len(segments)
            },
            'alzheimer': {
                'probability': float(np.mean(alzheimer_preds[:, 1])),
                'segments_detected': int(np.sum(alzheimer_preds[:, 1] > 0.5)),
                'total_segments': len(segments)
            },
            'parkinson': {
                'probability': float(np.mean(parkinson_preds[:, 1])),
                'segments_detected': int(np.sum(parkinson_preds[:, 1] > 0.5)),
                'total_segments': len(segments)
            }
        }
        
        # Determine overall risk level
        max_prob = max(
            results['seizure']['probability'],
            results['alzheimer']['probability'],
            results['parkinson']['probability']
        )
        
        if max_prob > 0.7:
            overall_status = "HIGH RISK"
            status_color = "danger"
        elif max_prob > 0.4:
            overall_status = "MODERATE"
            status_color = "warning"
        else:
            overall_status = "LOW RISK"
            status_color = "success"
        
        # Create result display components
        result_components = []
        
        # Seizure results
        seizure_prob = results['seizure']['probability']
        seizure_color = "danger" if seizure_prob > 0.7 else "warning" if seizure_prob > 0.4 else "success"
        result_components.append(
            dbc.Alert([
                html.Div([
                    html.Strong("Seizure Detection", className="d-block mb-2"),
                    html.Div([
                        html.Span("Probability: ", className="text-muted"),
                        html.Span(f"{seizure_prob:.1%}", className="fw-bold")
                    ]),
                    html.Div([
                        html.Span("Segments: ", className="text-muted"),
                        html.Span(f"{results['seizure']['segments_detected']}/{results['seizure']['total_segments']}")
                    ]),
                    dbc.Progress(
                        value=seizure_prob * 100,
                        color=seizure_color,
                        className="mt-2",
                        style={"height": "8px"}
                    )
                ])
            ], color=seizure_color, className="mb-2")
        )
        
        # Alzheimer results
        alzheimer_prob = results['alzheimer']['probability']
        alzheimer_color = "danger" if alzheimer_prob > 0.7 else "warning" if alzheimer_prob > 0.4 else "success"
        result_components.append(
            dbc.Alert([
                html.Div([
                    html.Strong("Alzheimer's Detection", className="d-block mb-2"),
                    html.Div([
                        html.Span("Probability: ", className="text-muted"),
                        html.Span(f"{alzheimer_prob:.1%}", className="fw-bold")
                    ]),
                    html.Div([
                        html.Span("Segments: ", className="text-muted"),
                        html.Span(f"{results['alzheimer']['segments_detected']}/{results['alzheimer']['total_segments']}")
                    ]),
                    dbc.Progress(
                        value=alzheimer_prob * 100,
                        color=alzheimer_color,
                        className="mt-2",
                        style={"height": "8px"}
                    )
                ])
            ], color=alzheimer_color, className="mb-2")
        )
        
        # Parkinson results
        parkinson_prob = results['parkinson']['probability']
        parkinson_color = "danger" if parkinson_prob > 0.7 else "warning" if parkinson_prob > 0.4 else "success"
        result_components.append(
            dbc.Alert([
                html.Div([
                    html.Strong("Parkinson's Detection", className="d-block mb-2"),
                    html.Div([
                        html.Span("Probability: ", className="text-muted"),
                        html.Span(f"{parkinson_prob:.1%}", className="fw-bold")
                    ]),
                    html.Div([
                        html.Span("Segments: ", className="text-muted"),
                        html.Span(f"{results['parkinson']['segments_detected']}/{results['parkinson']['total_segments']}")
                    ]),
                    dbc.Progress(
                        value=parkinson_prob * 100,
                        color=parkinson_color,
                        className="mt-2",
                        style={"height": "8px"}
                    )
                ])
            ], color=parkinson_color, className="mb-2")
        )
        
        # Add summary alert
        result_components.append(
            dbc.Alert([
                html.Strong("Overall Assessment:", className="d-block mb-1"),
                html.Div([
                    html.Span("Risk Level: ", className="text-muted"),
                    html.Span(overall_status, className="fw-bold")
                ]),
                html.Small(
                    f"Analyzed {len(segments)} segments from EEG data",
                    className="text-muted d-block mt-2"
                )
            ], color=status_color, className="mb-0")
        )
        
        return (
            html.Div(result_components),
            overall_status,
            html.Small("Analysis complete!", className="text-success"),
            results
        )
        
    except Exception as e:
        return (
            dbc.Alert([
                html.Strong("Error during analysis"),
                html.Br(),
                html.Small(str(e))
            ], color="danger"),
            "Error",
            html.Small("Analysis failed", className="text-danger"),
            None
        )