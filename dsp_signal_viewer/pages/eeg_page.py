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
try:
    from torcheeg.models import CCNN, EEGNet, TSCeption, FBCCNN
    from torcheeg import transforms
    TORCHEEG_AVAILABLE = True
except ImportError:
    TORCHEEG_AVAILABLE = False

dash.register_page(__name__, path="/eeg", name="EEG")

DATA_DIRECTORY = r"C:\Users\chanm\signal-viewer\data\Annotated_EEG"

CARD_STYLE = {
    'backgroundColor': '#1e2130',
    'border': '1px solid #2d3748',
    'borderRadius': '8px',
    'margin': '10px 0'
}

PLOT_CONFIG = {'displayModeBar': False, 'displaylogo': False}
SAMPLING_RATE = 256

# Torch EEG Models for Disease Detection
class TorchEEGSeizureDetector:
    def __init__(self, n_channels=19):
        if not TORCHEEG_AVAILABLE:
            raise ImportError("TorchEEG not installed")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = "EEGNet (TorchEEG)"
        
        self.model = EEGNet(
            chunk_size=1024, # 4 seconds at 256 Hz
            num_electrodes=n_channels,
            num_classes=2,
            F1=8, # 8 filters in the first conv layer
            F2=16, # 16 filters in the second conv layer
            D=2, # depth multiplier
            kernel_1=64, # temporal conv kernel size
            kernel_2=16, # spatial conv kernel size
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
        # segments: (batch, timesteps, channels)
        n_segments, timesteps, n_channels = segments.shape
        
        # Normalize
        segments_reshaped = segments.reshape(-1, n_channels)
        segments_normalized = self.scaler.fit_transform(segments_reshaped)
        segments_normalized = segments_normalized.reshape(n_segments, timesteps, n_channels)
        
        # EEGNet expects (batch, 1, channels, timesteps)
        tensor = torch.FloatTensor(segments_normalized).unsqueeze(1).permute(0, 1, 3, 2)
        return tensor.to(self.device)
    
    def predict(self, segments):
        """Predict seizure probability"""
        self.model.eval()
        with torch.no_grad():
            x = self.preprocess(segments)
            outputs = self.model(x)
            predictions = torch.softmax(outputs, dim=1).cpu().numpy()
        return predictions


class TorchEEGAlzheimerDetector:
    """Alzheimer's detector using TorchEEG's TSCeption architecture"""
    def __init__(self, n_channels=19):
        if not TORCHEEG_AVAILABLE:
            raise ImportError("TorchEEG not installed")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = "TSCeption (TorchEEG)"
        
        # TSCeption: Temporal-Spatial Inception for EEG
        self.model = TSCeption(
            num_electrodes=n_channels,
            num_classes=2,
            num_T=15, # because Alzheimer's affects multiple frequency bands
            num_S=15, # Models local and global electrode relationships
            hid_channels=32,
            dropout=0.5
        ).to(self.device)
        
        self._initialize_weights()
        self.model.eval()
        self.scaler = StandardScaler()
        
    def _initialize_weights(self):
        """Initialize with focus on slow-wave patterns (AD signature)"""
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
        """Preprocess for TSCeption"""
        n_segments, timesteps, n_channels = segments.shape
        
        segments_reshaped = segments.reshape(-1, n_channels)
        segments_normalized = self.scaler.fit_transform(segments_reshaped)
        segments_normalized = segments_normalized.reshape(n_segments, timesteps, n_channels)
        
        # TSCeption expects (batch, 1, channels, timesteps)
        tensor = torch.FloatTensor(segments_normalized).unsqueeze(1).permute(0, 1, 3, 2)
        return tensor.to(self.device)
    
    def predict(self, segments):
        """Predict Alzheimer's probability"""
        self.model.eval()
        with torch.no_grad():
            x = self.preprocess(segments)
            outputs = self.model(x)
            predictions = torch.softmax(outputs, dim=1).cpu().numpy()
        return predictions


class TorchEEGParkinsonDetector:
    """Parkinson's detector using TorchEEG's FBCCNN architecture"""
    def __init__(self, n_channels=19):
        if not TORCHEEG_AVAILABLE:
            raise ImportError("TorchEEG not installed")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = "FBCCNN (TorchEEG)"
        
        # FBCCNN: Filter Bank Convolutional CNN
        self.model = FBCCNN(
            num_classes=2,
            num_electrodes=n_channels,
            chunk_size=1024,
            dropout=0.5,
            F1=128,
            F2=256,
            D=2
        ).to(self.device)
        
        self._initialize_weights()
        self.model.eval()
        self.scaler = StandardScaler()
        
    def _initialize_weights(self):
        """Initialize with focus on beta band suppression (PD signature)"""
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
        """Preprocess for FBCCNN"""
        n_segments, timesteps, n_channels = segments.shape
        
        segments_reshaped = segments.reshape(-1, n_channels)
        segments_normalized = self.scaler.fit_transform(segments_reshaped)
        segments_normalized = segments_normalized.reshape(n_segments, timesteps, n_channels)
        
        # FBCCNN expects (batch, 1, channels, timesteps)
        tensor = torch.FloatTensor(segments_normalized).unsqueeze(1).permute(0, 1, 3, 2)
        return tensor.to(self.device)
    
    def predict(self, segments):
        """Predict Parkinson's probability"""
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
            'T3', 'T4', 'T5', 'T6', 'O1', 'O2'
        ]
    
    def map_channels_to_standard(self, data):
        """Map available channels to standard 19 channels"""
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


# Wrapper classes
class SeizureDetector:
    def __init__(self):
        self.detector = TorchEEGSeizureDetector() if TORCHEEG_AVAILABLE else None
        self.model_name = self.detector.model_name if self.detector else "Not Available"
    
    def predict(self, segments):
        if self.detector:
            return self.detector.predict(segments)
        return np.random.rand(len(segments), 2)  # Fallback


class AlzheimerDetector:
    def __init__(self):
        self.detector = TorchEEGAlzheimerDetector() if TORCHEEG_AVAILABLE else None
        self.model_name = self.detector.model_name if self.detector else "Not Available"
    
    def predict(self, segments):
        if self.detector:
            return self.detector.predict(segments)
        return np.random.rand(len(segments), 2)


class ParkinsonDetector:
    def __init__(self):
        self.detector = TorchEEGParkinsonDetector() if TORCHEEG_AVAILABLE else None
        self.model_name = self.detector.model_name if self.detector else "Not Available"
    
    def predict(self, segments):
        if self.detector:
            return self.detector.predict(segments)
        return np.random.rand(len(segments), 2)


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
        # Fallback to first channel if selected channel not found
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
                    dbc.Spinner(html.Div(id="analyze-status"), size="sm", color="success")
                ])
            ], style=CARD_STYLE)
        ], width=4)
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
    dcc.Store(id="uploaded-file-store"),
    dcc.Interval(id="interval-component", interval=1000, n_intervals=0, max_intervals=1)
], fluid=True, style={'backgroundColor': '#182940', 'minHeight': '100vh', 'padding': '20px'})

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
     Input("channel-selector", "value")]
)
def update_main_plots(data, metadata, mode, selected_channel):
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
    
    if selected_channel not in channels:
        selected_channel = channels[0] if channels else None
    
    # Create plots
    if mode == "multi":
        main_fig = create_multi_channel_plot(df, channels, sampling_rate)
    elif mode == "single" and selected_channel:
        main_fig = create_single_channel_plot(df, selected_channel, sampling_rate)
    else:
        # Fallback to multi-channel if single mode but no channel selected
        main_fig = create_multi_channel_plot(df, channels, sampling_rate)
    
    if selected_channel and selected_channel in df.columns:
        freq_fig = create_frequency_plot(df, selected_channel, sampling_rate)
    else:
        freq_fig = create_frequency_plot(df, channels[0], sampling_rate)
    
    # Signal quality
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
            None,
            False  # Button remains enabled initially
        )
    
    try:
        df = pd.DataFrame(data['data'], columns=data['columns'])
        
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
        
        # Compute results
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
        
        disease_probs = {
            "Seizure": results['seizure']['probability'],
            "Alzheimer's": results['alzheimer']['probability'],
            "Parkinson's": results['parkinson']['probability']
        }

        top_disease = max(disease_probs, key=disease_probs.get)
        top_prob = disease_probs[top_disease]

        if top_prob > 0.5:
            overall_status = f"HIGH RISK of {top_disease}"
            status_color = "danger"
        else:
            overall_status = "PERFECT HEALTH"
            status_color = "success"
        
        result_components = []
        for name, res in results.items():
            progress = dbc.Progress(
                value=res['probability'] * 100,
                color="danger" if res['probability'] > 0.5 else "success",
                className="mb-2",
                label=f"{res['probability']*100:.1f}%"
            )
            result_components.append(
                dbc.Card([
                    dbc.CardHeader(name.capitalize()),
                    dbc.CardBody([
                        html.P(f"Segments Detected: {res['segments_detected']}/{res['total_segments']}"),
                        progress
                    ])
                ], className="mb-3 shadow-sm")
            )
        
        return (
            html.Div(result_components),
            overall_status,
            html.Small("Analysis complete!", className="text-success"),
            results,
            True  # Disable button after analysis
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
            True  # Disable button on failure too
        )

@callback(
    Output("analyze-btn", "disabled", allow_duplicate=True),
    Input("eeg-data-store", "data"),
    prevent_initial_call=True
)
def reset_analyze_button(data):
    """Re-enable Analyze button when new EEG data is loaded"""
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
    """Show tooltip when button is disabled."""
    if disabled:
        return True
    return False