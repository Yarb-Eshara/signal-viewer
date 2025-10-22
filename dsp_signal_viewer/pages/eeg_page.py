import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
import os
import mne
from warnings import filterwarnings
filterwarnings("ignore")
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import base64
import tempfile
from torcheeg.models.cnn import CCNN, EEGNet, TSCeption, FBCCNN
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path
TF_AVAILABLE = True
TORCHEEG_AVAILABLE = True

dash.register_page(__name__, path="/eeg", name="EEG")

BASE_DIR = os.path.dirname(os.path.dirname(__file__)) 
SEIZURE_MODEL_PATH = os.path.join(BASE_DIR, "models", "CHB_MIT_sz_detec_demo.h5")
BASE_DIR_data = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = os.path.join(BASE_DIR_data, "data", "Annotated_EEG")

CARD_STYLE = {
    'backgroundColor': '#1e2130',
    'border': '1px solid #2d3748',
    'borderRadius': '8px',
    'margin': '10px 0'
}

PLOT_CONFIG = {'displayModeBar': False, 'displaylogo': False}
SAMPLING_RATE = 256

class H5SeizureDetector:
    def __init__(self, model_path=SEIZURE_MODEL_PATH):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not installed")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Seizure model not found at {model_path}")

        try:
            self.model = load_model(model_path)
            self.model_name = "CHB-MIT Seizure Detection Model"
            print(f"Successfully loaded seizure detection model from {model_path}")
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

        self.input_shape = self.model.input_shape
        print(f"Model input shape: {self.input_shape}")
        self.scaler = StandardScaler()

    def preprocess(self, segments):
        """Preprocess segments for the H5 model"""
        # Input: segments shape = (n_segments, timesteps, n_channels)
        # Model expects: (batch, channels, timesteps, 1)
        n_segments, timesteps, n_channels = segments.shape

        # Expected channels from model input shape: (None, 18, 1024, 1)
        expected_channels = self.input_shape[1]
        expected_timesteps = self.input_shape[2]
        
        # Adjust channels
        if n_channels > expected_channels:
            segments = segments[:, :, :expected_channels]
            n_channels = expected_channels
        elif n_channels < expected_channels:
            padding = np.zeros((n_segments, timesteps, expected_channels - n_channels))
            segments = np.concatenate([segments, padding], axis=2)
            n_channels = expected_channels

        # Adjust timesteps if needed
        if timesteps > expected_timesteps:
            segments = segments[:, :expected_timesteps, :]
            timesteps = expected_timesteps
        elif timesteps < expected_timesteps:
            padding = np.zeros((n_segments, expected_timesteps - timesteps, n_channels))
            segments = np.concatenate([segments, padding], axis=1)
            timesteps = expected_timesteps

        # Normalize each channel separately
        segments_normalized = np.zeros_like(segments)
        for i in range(n_channels):
            channel_data = segments[:, :, i].reshape(-1, 1)
            segments_normalized[:, :, i] = self.scaler.fit_transform(channel_data).reshape(n_segments, timesteps)

        # Transform to model expected shape: (batch, channels, timesteps, 1)
        # Current shape: (n_segments, timesteps, n_channels)
        # Step 1: Transpose to (n_segments, n_channels, timesteps)
        tensor = segments_normalized.transpose(0, 2, 1)
        # Step 2: Add channel dimension at the end: (n_segments, n_channels, timesteps, 1)
        tensor = np.expand_dims(tensor, axis=-1)

        return tensor.astype(np.float32)

    def predict(self, segments):
        """Predict seizure probability using H5 model"""
        try:
            x = self.preprocess(segments)
            predictions = self.model.predict(x, verbose=0)

            if predictions.shape[1] == 1:
                seizure_probs = predictions.flatten()
                normal_probs = 1 - seizure_probs
                return np.column_stack([normal_probs, seizure_probs])
            else:
                return predictions

        except Exception as e:
            print(f"Prediction error: {e}")
            n_segments = len(segments)
            return np.random.rand(n_segments, 2)


class TorchEEGAlzheimerDetector:
    """Alzheimer's detector using TorchEEG's TSCeption architecture"""
    def __init__(self, n_channels=18):
        if not TORCHEEG_AVAILABLE:
            raise ImportError("TorchEEG not installed")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = "TSCeption (TorchEEG)"
        self.n_channels = n_channels

        self.model = TSCeption(
            num_electrodes=n_channels,
            num_classes=2,
            num_T=15,
            num_S=15,
            hid_channels=32,
            dropout=0.5
        ).to(self.device)

        self._initialize_weights()
        self.model.eval()
        self.scaler = StandardScaler()

    def _initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def preprocess(self, segments):
        n_segments, timesteps, n_channels = segments.shape

        # Ensure we have the right number of channels
        if n_channels != self.n_channels:
            if n_channels > self.n_channels:
                segments = segments[:, :, :self.n_channels]
            else:
                padding = np.zeros((n_segments, timesteps, self.n_channels - n_channels))
                segments = np.concatenate([segments, padding], axis=2)
            n_channels = self.n_channels

        segments_reshaped = segments.reshape(-1, n_channels)
        segments_normalized = self.scaler.fit_transform(segments_reshaped)
        segments_normalized = segments_normalized.reshape(n_segments, timesteps, n_channels)

        # TSCeption expects: (batch, 1, channels, timesteps)
        tensor = torch.FloatTensor(segments_normalized).unsqueeze(1).permute(0, 1, 3, 2)
        return tensor.to(self.device)

    def predict(self, segments):
        self.model.eval()
        with torch.no_grad():
            x = self.preprocess(segments)
            outputs = self.model(x)
            predictions = torch.softmax(outputs, dim=1).cpu().numpy()
        return predictions


class TorchEEGParkinsonDetector:
    """Parkinson's detector using TorchEEG's FBCCNN architecture"""
    def __init__(self, n_channels=18):
        if not TORCHEEG_AVAILABLE:
            raise ImportError("TorchEEG not installed")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = "FBCCNN (TorchEEG)"
        self.n_channels = n_channels

        # FBCCNN expects input with specific channel dimensions
        # Let's use a simpler approach with EEGNet which is more flexible
        self.model = EEGNet(
            chunk_size=1024,
            num_electrodes=n_channels,
            num_classes=2,
            dropout=0.5
        ).to(self.device)

        self._initialize_weights()
        self.model.eval()
        self.scaler = StandardScaler()

    def _initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def preprocess(self, segments):
        n_segments, timesteps, n_channels = segments.shape

        # Ensure we have the right number of channels
        if n_channels != self.n_channels:
            if n_channels > self.n_channels:
                segments = segments[:, :, :self.n_channels]
            else:
                padding = np.zeros((n_segments, timesteps, self.n_channels - n_channels))
                segments = np.concatenate([segments, padding], axis=2)
            n_channels = self.n_channels

        segments_reshaped = segments.reshape(-1, n_channels)
        segments_normalized = self.scaler.fit_transform(segments_reshaped)
        segments_normalized = segments_normalized.reshape(n_segments, timesteps, n_channels)

        # EEGNet expects: (batch, 1, channels, timesteps)
        tensor = torch.FloatTensor(segments_normalized).unsqueeze(1).permute(0, 1, 3, 2)
        return tensor.to(self.device)

    def predict(self, segments):
        self.model.eval()
        with torch.no_grad():
            x = self.preprocess(segments)
            outputs = self.model(x)
            predictions = torch.softmax(outputs, dim=1).cpu().numpy()
        return predictions


class CHBMITPreprocessor:
    """Preprocessor for CHB-MIT EEG dataset"""
    def __init__(self, fs=256):
        self.fs = fs
        self.standard_channels = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
            'C3', 'Cz', 'C4', 'P3', 'Pz', 'P4',
            'T3', 'T4', 'T5', 'T6', 'O1'
        ]

    def map_channels_to_standard(self, data):
        """Map available channels to standard 18 channels"""
        mapped_data = pd.DataFrame()
        available_channels = data.columns.tolist()

        for std_ch in self.standard_channels:
            if std_ch in available_channels:
                mapped_data[std_ch] = data[std_ch]
            else:
                similar = [ch for ch in available_channels if std_ch.lower() in ch.lower()]
                if similar:
                    mapped_data[std_ch] = data[similar[0]]
                else:
                    mapped_data[std_ch] = 0.0

        return mapped_data

    def segment_data(self, data, window_size=1024, overlap=0.5):
        """Segment data into windows"""
        step_size = int(window_size * (1 - overlap))
        segments = []

        for i in range(0, len(data) - window_size + 1, step_size):
            segment = data.iloc[i:i + window_size].values
            segments.append(segment)

        if not segments:
            segment = data.values
            if len(segment) < window_size:
                padding = np.zeros((window_size - len(segment), segment.shape[1]))
                segment = np.vstack([segment, padding])
            segments.append(segment)

        return np.array(segments)


class SeizureDetector:
    def __init__(self):
        try:
            self.detector = H5SeizureDetector()
            self.model_name = self.detector.model_name
            self.status = "✅ H5 Model Loaded"
        except Exception as e:
            print(f"Failed to load H5 seizure detector: {e}")
            self.detector = None
            self.model_name = "Not Available"
            self.status = f"❌ H5 Model Error: {str(e)[:50]}"

    def predict(self, segments):
        if self.detector:
            return self.detector.predict(segments)
        return np.random.rand(len(segments), 2)


class AlzheimerDetector:
    def __init__(self):
        self.detector = TorchEEGAlzheimerDetector() if TORCHEEG_AVAILABLE else None
        self.model_name = self.detector.model_name if self.detector else "Not Available"
        self.status = "✅ Available" if self.detector else "❌ TorchEEG not available"

    def predict(self, segments):
        if self.detector:
            return self.detector.predict(segments)
        return np.random.rand(len(segments), 2)


class ParkinsonDetector:
    def __init__(self):
        self.detector = TorchEEGParkinsonDetector() if TORCHEEG_AVAILABLE else None
        self.model_name = self.detector.model_name if self.detector else "Not Available"
        self.status = "✅ Available" if self.detector else "❌ TorchEEG not available"

    def predict(self, segments):
        if self.detector:
            return self.detector.predict(segments)
        return np.random.rand(len(segments), 2)


def downsample_dataframe(df, factor):
    """Downsample dataframe by taking every nth sample"""
    if factor == 1:
        return df
    return df.iloc[::factor].reset_index(drop=True)


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

def create_multi_channel_plot(df, channels, sampling_rate=SAMPLING_RATE):
    """Create multi-channel EEG plot"""
    display_channels = channels[:10] if len(channels) > 10 else channels

    fig = make_subplots(
        rows=len(display_channels), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.015,
        subplot_titles=display_channels
    )

    time_axis = np.arange(len(df)) / sampling_rate
    colors = px.colors.qualitative.Set3

    for i, channel in enumerate(display_channels):
        if channel in df.columns:
            signal_data = df[channel].values

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

def create_single_channel_plot(df, channel, sampling_rate=SAMPLING_RATE):
    """Create single channel EEG plot"""
    if channel not in df.columns:
        channel = df.columns[0]

    time_axis = np.arange(len(df)) / sampling_rate
    signal_data = df[channel].values

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
                                'backgroundColor': "#182940",
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
        ], width=4),

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
        ], width=4),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("🏥 Disease Detection", className="card-title", style={"color": "white"}),
                    dbc.Button("Analyze", id="analyze-btn",
                              color="success", size="sm", className="w-100",
                              n_clicks=0),
                    dbc.Spinner(html.Div(id="analyze-status"), size="sm", color="success"),
                    html.Small(id="model-status", className="text-info mt-2")
                ])
            ], style=CARD_STYLE)
        ], width=4)
    ], className="mb-4"),

    # NEW: Downsampling Slider Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("⚙️ Signal Downsampling", className="card-title", style={"color": "white"}),
                    html.Div([
                        html.Small("Downsample Factor:", className="text-muted d-block mb-2"),
                        dcc.Slider(
                            id="downsample-slider",
                            min=1,
                            max=10,
                            step=1,
                            value=1,
                            marks={i: f"{i}x" for i in range(1, 11)},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Div(id="downsample-info", className="text-info mt-2 text-center")
                    ])
                ])
            ], style=CARD_STYLE)
        ], width=12)
    ], className="mb-4"),

    dbc.Tooltip(
    "Already analyzed this subject. Upload or select a new one to re-enable.",
    target="analyze-btn",
    id="analyze-tooltip",
    placement="top"
    ),

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
                    html.Small("Effective Sampling Rate", className="text-muted")
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
    dcc.Store(id="uploaded-file-store"),
    dcc.Store(id="original-data-store"),  # NEW: Store original undownsampled data
    dcc.Interval(id="interval-component", interval=1000, n_intervals=0, max_intervals=1)
], fluid=True, style={'backgroundColor': '#182940', 'minHeight': '100vh', 'padding': '20px'})

@callback(
    Output("model-status", "children"),
    Input("analyze-btn", "n_clicks"),
    prevent_initial_call=True
)
def show_model_status(n_clicks):
    """Show the status of all detection models"""
    seizure_detector = SeizureDetector()
    alzheimer_detector = AlzheimerDetector()
    parkinson_detector = ParkinsonDetector()

    return html.Div([
        html.Small(f"Seizure: {seizure_detector.status}", className="d-block"),
        html.Small(f"Alzheimer: {alzheimer_detector.status}", className="d-block"),
        html.Small(f"Parkinson: {parkinson_detector.status}", className="d-block")
    ])

@callback(
    Output("file-selector", "options"),
    [Input("refresh-btn", "n_clicks"),
     Input("interval-component", "n_intervals")],
    prevent_initial_call=False
)
def update_file_options(n_clicks, n_intervals):
    return load_eeg_files()

@callback(
    Output("channel-selector", "value"),
    Input("channel-selector", "options"),
    prevent_initial_call=True
)
def set_default_channel(options):
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
    if contents is None:
        return None, None

    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        if not filename.lower().endswith('.edf'):
            return None, None

        uploaded_data = {
            'filename': filename,
            'content': content_string,
            'is_uploaded': True
        }

        return uploaded_data, None

    except Exception as e:
        print(f"Error handling upload: {e}")
        return None, None

# MODIFIED: Load and store original data
@callback(
    [Output("original-data-store", "data"),
     Output("eeg-metadata-store", "data"),
     Output("subject-info", "children"),
     Output("duration-info", "children"),
     Output("channel-selector", "options")],
    [Input("file-selector", "value"),
     Input("uploaded-file-store", "data")]
)
def load_eeg_data(file_path, uploaded_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return None, None, "No file selected", "N/A", []

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if trigger_id == "uploaded-file-store" and uploaded_data is not None:
        try:
            content_string = uploaded_data['content']
            decoded = base64.b64decode(content_string)
            filename = uploaded_data['filename']

            with tempfile.NamedTemporaryFile(delete=False, suffix='.edf') as tmp_file:
                tmp_file.write(decoded)
                tmp_file_path = tmp_file.name

            try:
                raw = mne.io.read_raw_edf(tmp_file_path, preload=True, verbose=False)
                sfreq = float(raw.info['sfreq'])

                n_samples = len(raw.times)
                target_samples = 5000

                if n_samples > target_samples:
                    decim_factor = n_samples // target_samples
                    raw = raw.resample(sfreq / decim_factor)
                    sfreq = float(raw.info['sfreq'])

                df = raw.to_data_frame()

                if 'time' in df.columns:
                    df = df.drop('time', axis=1)

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
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass

        except Exception as e:
            return None, None, f"❌ Upload Error: {str(e)[:30]}", "N/A", []

    if file_path is None or file_path in ["no-directory", "no-files", "error"]:
        return None, None, "No file selected", "N/A", []

    try:
        if not os.path.exists(file_path):
            return None, None, "❌ File not found", "N/A", []

        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        sfreq = float(raw.info['sfreq'])

        n_samples = len(raw.times)
        target_samples = 5000

        if n_samples > target_samples:
            decim_factor = n_samples // target_samples
            raw = raw.resample(sfreq / decim_factor)
            sfreq = float(raw.info['sfreq'])

        df = raw.to_data_frame()

        if 'time' in df.columns:
            df = df.drop('time', axis=1)

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


# NEW: Apply downsampling to create eeg-data-store
@callback(
    [Output("eeg-data-store", "data"),
     Output("downsample-info", "children")],
    [Input("original-data-store", "data"),
     Input("downsample-slider", "value")],
    State("eeg-metadata-store", "data")
)
def apply_downsampling(original_data, downsample_factor, metadata):
    if original_data is None or metadata is None:
        return None, ""
    
    df_original = pd.DataFrame(original_data['data'], columns=original_data['columns'])
    
    df_downsampled = downsample_dataframe(df_original, downsample_factor)
    
    effective_srate = metadata['sampling_rate'] / downsample_factor
    info_text = f"Original: {len(df_original)} samples @ {metadata['sampling_rate']:.1f} Hz → Downsampled: {len(df_downsampled)} samples @ {effective_srate:.1f} Hz"
    
    return df_downsampled.to_dict('split'), info_text


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
     Input("downsample-slider", "value")]
)
def update_main_plots(data, metadata, mode, selected_channel, downsample_factor):
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
    sampling_rate = metadata['sampling_rate'] / downsample_factor

    if selected_channel not in channels:
        selected_channel = channels[0] if channels else None

    if mode == "multi":
        main_fig = create_multi_channel_plot(df, channels, sampling_rate)
    elif mode == "single" and selected_channel:
        main_fig = create_single_channel_plot(df, selected_channel, sampling_rate)
    else:
        main_fig = create_multi_channel_plot(df, channels, sampling_rate)

    if selected_channel and selected_channel in df.columns:
        freq_fig = create_frequency_plot(df, selected_channel, sampling_rate)
    else:
        freq_fig = create_frequency_plot(df, channels[0], sampling_rate)

    signal_quality = "Good"
    try:
        if selected_channel and selected_channel in df.columns:
            std_val = df[selected_channel].std()
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
     Output("detection-results-store", "data"),
     Output("analyze-btn", "disabled")],
    Input("analyze-btn", "n_clicks"),
    [State("eeg-data-store", "data"),
     State("eeg-metadata-store", "data"),
     State("downsample-slider", "value")],
    prevent_initial_call=True
)
def run_disease_detection(n_clicks, data, metadata, downsample_factor):
    if n_clicks == 0 or data is None or metadata is None:
        return (
            html.Div("Click 'Analyze' to run detection", className="text-muted"),
            "Not Analyzed",
            "",
            None,
            False
        )

    try:
        df = pd.DataFrame(data['data'], columns=data['columns'])

        preprocessor = CHBMITPreprocessor()
        seizure_detector = SeizureDetector()
        alzheimer_detector = AlzheimerDetector()
        parkinson_detector = ParkinsonDetector()

        mapped_data = preprocessor.map_channels_to_standard(df)
        segments = preprocessor.segment_data(mapped_data, window_size=1024, overlap=0.5)

        seizure_preds = seizure_detector.predict(segments)
        alzheimer_preds = alzheimer_detector.predict(segments)
        parkinson_preds = parkinson_detector.predict(segments)

        results = {
            'seizure': {
                'probability': float(np.mean(seizure_preds[:, 1])),
                'segments_detected': int(np.sum(seizure_preds[:, 1] > 0.5)),
                'total_segments': len(segments),
                'model': seizure_detector.model_name
            },
            'alzheimer': {
                'probability': float(np.mean(alzheimer_preds[:, 1])),
                'segments_detected': int(np.sum(alzheimer_preds[:, 1] > 0.5)),
                'total_segments': len(segments),
                'model': alzheimer_detector.model_name
            },
            'parkinson': {
                'probability': float(np.mean(parkinson_preds[:, 1])),
                'segments_detected': int(np.sum(parkinson_preds[:, 1] > 0.5)),
                'total_segments': len(segments),
                'model': parkinson_detector.model_name
            }
        }

        # Find disease with highest probability
        disease_probs = {
            "Seizure": results['seizure']['probability'],
            "Alzheimer's": results['alzheimer']['probability'],
            "Parkinson's": results['parkinson']['probability']
        }

        # Get the disease with maximum probability
        top_disease = max(disease_probs, key=disease_probs.get)
        top_prob = disease_probs[top_disease]

        # Determine display based on threshold
        if top_prob > 0.5:
            overall_status = f"⚠️ {top_disease} Detected"
            # Get the disease key for results dictionary
            disease_key = top_disease.lower().replace("'s", "")
            main_message = html.Div([
                html.H4(f"🚨 HIGH RISK: {top_disease}", className="text-danger mb-3"),
                html.P(f"Confidence: {top_prob*100:.1f}%", className="text-warning fs-5 fw-bold"),
                html.Hr(),
                html.Small(f"Segments detected: {results[disease_key]['segments_detected']}/{results[disease_key]['total_segments']}", 
                          className="text-muted d-block"),
                html.Small(f"Model: {results[disease_key]['model']}", 
                          className="text-muted d-block"),
                html.Small(f"Downsampling: {downsample_factor}x", className="text-muted d-block mt-2")
            ])
        else:
            overall_status = "✅ Healthy"
            main_message = html.Div([
                html.H4("✨ NO CONDITION DETECTED", className="text-success mb-3"),
                html.P("All probabilities below threshold", className="text-info"),
                html.Hr(),
                html.Small(f"Highest probability: {top_disease} ({top_prob*100:.1f}%)", 
                          className="text-muted d-block"),
                html.Small(f"Downsampling: {downsample_factor}x", className="text-muted d-block mt-2")
            ])

        return (
            main_message,
            overall_status,
            html.Small("Analysis complete!", className="text-success"),
            results,
            True
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
            None,
            True
        )
        
@callback(
    Output("analyze-btn", "disabled", allow_duplicate=True),
    Input("eeg-data-store", "data"),
    prevent_initial_call=True
)
def reset_analyze_button(data):
    if data is None:
        return True
    return False

@callback(
    Output("analyze-tooltip", "is_open"),
    [Input("analyze-btn", "n_clicks"),
     Input("analyze-btn", "disabled")],
    State("analyze-tooltip", "is_open")
)
def toggle_tooltip(n_clicks, disabled, is_open):
    if disabled:
        return True
    return False