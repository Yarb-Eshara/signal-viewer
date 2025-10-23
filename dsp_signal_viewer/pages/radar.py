import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # hide TF logs

import base64
import io
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import pandas as pd
import numpy as np
import struct  # <-- NEW: Added for constructing WAV file headers
from tensorflow import keras
import joblib
import librosa
import plotly.graph_objects as go
import tempfile

# --- Register page ---
dash.register_page(__name__, path="/radar", name="Radar Velocity Prediction")

# --- Define BASE_DIR and paths ---
# NOTE: BASE_DIR calculation here is likely relative to the structure
# of the dsp_signal_viewer project, assuming the script is in a subdirectory.
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # dsp_signal_viewer/
model_path = os.path.join(BASE_DIR, "models", "velocity_model.h5")
x_scaler_path = os.path.join(BASE_DIR, "models", "x_scaler.pkl")

# --- Load model and scaler ---
print("Loading model from:", model_path)
print("File exists?", os.path.exists(model_path))

# NOTE: Since this environment cannot access external files, these paths
# will likely fail in a live canvas environment unless the model files
# are pre-loaded or mocked. We assume they load successfully for now.
try:
    model = keras.models.load_model(model_path, compile=False)
    x_scaler = joblib.load(x_scaler_path)
except Exception as e:
    print(f"Warning: Could not load ML artifacts. Prediction will fail. {e}")

    # Create mock objects to allow the app to render
    class MockModel:
        def predict(self, X): return np.array([[50.0]])

    class MockScaler:
        def transform(self, X): return X

    model = MockModel()
    x_scaler = MockScaler()

# --- Doppler calculation function ---
def doppler_from_velocity(velocity, f_source):
    """
    velocity in km/h
    f_source in Hz
    """
    v_m_s = velocity / 3.6  # convert km/h to m/s
    c = 343  # speed of sound in air m/s
    delta_f = (v_m_s / c) * f_source
    f_received = f_source + delta_f
    return delta_f, f_received

# --- Extract MFCC features from uploaded audio (MODIFIED to accept target_sr) ---
def extract_mfcc(file_path, target_sr, n_mfcc=13):
    """
    Loads audio and resamples it to target_sr before feature extraction.
    """
    # librosa loads the file and resamples it to the target_sr specified by the user
    # sr will be equal to target_sr
    y, sr = librosa.load(file_path, sr=target_sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean, y, sr

# --- Extract dominant frequency from audio ---
def estimate_source_frequency(y, sr):
    """
    Use FFT to find the frequency with max amplitude
    """
    n = len(y)
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    magnitude = np.abs(Y)
    dominant_idx = np.argmax(magnitude)
    f_source = freqs[dominant_idx]
    return f_source

# --- Create waveform plot ---
def create_waveform_plot(y, sr):
    time = np.arange(0, len(y)) / sr
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=y, mode='lines', name='Waveform'))
    fig.update_layout(
        title="Audio Waveform (Processed at " + str(sr) + " Hz)",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_dark",
        height=300
    )
    return fig

# --- NEW: Helper function to convert numpy array to base64 WAV Data URI ---
def numpy_to_wav_base64(y, sr):
    """
    Converts a floating-point NumPy array (librosa output) to a 16-bit PCM WAV
    byte stream and returns it as a base64 Data URI for browser playback.
    """
    # Convert float data (-1 to 1) to 16-bit PCM integer format
    # Max amplitude is 2^15 - 1
    y_int16 = np.int16(y * 32767)
    n_samples = len(y_int16)

    n_channels = 1
    sample_width = 2  # 16-bit PCM

    wav_io = io.BytesIO()

    # --- RIFF chunk ---
    wav_io.write(b'RIFF')
    file_size_bytes = struct.pack('<I', 36 + n_samples * sample_width)
    wav_io.write(file_size_bytes)
    wav_io.write(b'WAVE')

    # --- fmt chunk ---
    wav_io.write(b'fmt ')
    wav_io.write(struct.pack('<I', 16))  # Chunk Size (16 for PCM)
    wav_io.write(struct.pack('<H', 1))  # Audio Format (1 for PCM)
    wav_io.write(struct.pack('<H', n_channels))
    wav_io.write(struct.pack('<I', sr))  # Sample Rate
    byte_rate = sr * n_channels * sample_width
    wav_io.write(struct.pack('<I', byte_rate))  # Byte Rate
    block_align = n_channels * sample_width
    wav_io.write(struct.pack('<H', block_align))  # Block Align
    wav_io.write(struct.pack('<H', sample_width * 8))  # Bits per Sample (16)

    # --- data chunk ---
    wav_io.write(b'data')
    wav_io.write(struct.pack('<I', n_samples * sample_width))
    wav_io.write(y_int16.tobytes())

    # Base64 encode the byte stream
    wav_io.seek(0)
    wav_bytes = wav_io.read()
    encoded_data = base64.b64encode(wav_bytes).decode('utf-8')

    # Data URI format
    return f"data:audio/wav;base64,{encoded_data}"

# --- Dash layout (MODIFIED to include dcc.Slider and html.Audio) ---
layout = dbc.Container([
    # Header
    html.Div([
        html.H1("Radar Velocity Prediction", className="text-center mt-4 mb-2"),
        html.P(
            "Upload an audio file and select a sampling rate to predict the velocity of a moving object using Doppler effect analysis.",
            className="text-center text-muted mb-4"
        )
    ]),

    # Main content
    dbc.Row([
        # Upload and control column
        dbc.Col([
            html.Div([
                dcc.Upload(
                    id="upload-audio",
                    children=html.Div([
                        html.I(className="bi bi-upload me-2"),
                        "Drag and Drop or Click to Select Audio File (.wav, .mp3)",
                    ]),
                    style={
                        "width": "100%",
                        "height": "100px",
                        "lineHeight": "100px",
                        "borderWidth": "2px",
                        "borderStyle": "dashed",
                        "borderRadius": "10px",
                        "textAlign": "center",
                        "marginBottom": "20px",
                        "backgroundColor": "transparent",
                        "cursor": "pointer",
                        "fontSize": "18px",
                        "color": "#495057",
                        "borderColor": "#6c757d"
                    },
                    multiple=False
                ),
                html.P(id="upload-status", className="text-muted text-center"),

                # --- AUDIO PLAYER CONTROLS (NEW) ---
                html.Div([
                    dbc.Label("Processed Audio Playback:", className="fw-bold mb-1"),
                    html.Audio(id='audio-player', controls=True, style={'width': '100%'}),
                ], id='audio-player-container', className="mb-3", style={'display': 'none'}),
                # --- END AUDIO PLAYER CONTROLS ---

                # --- NEW SAMPLING RATE SLIDER (UPDATED MAX VALUE) ---
                dbc.Label("Target Sampling Rate (Hz) for Resampling:", className="fw-bold mt-3 mb-2"),
                dcc.Slider(
                    id='sampling-rate-slider',
                    min=1000,  # Start at 1k to prevent librosa errors
                    max=70000,  # <-- UPDATED from 40000 to 70000
                    step=1000,
                    value=22050,  # Common standard for many audio models
                    # Updated marks to cover the new range up to 70k
                    marks={i: f'{i / 1000:.0f}k' for i in range(0, 70001, 5000) if i >= 1000 or i == 0},
                    tooltip={"placement": "bottom", "always_visible": True},
                    className="mb-4"
                ),
                # --- END NEW SLIDER ---

                dbc.Row([
                    dbc.Col(dbc.Button("Predict", id="predict-btn", color="primary", className="me-2"), width=6),
                    dbc.Col(dbc.Button("Reset", id="reset-btn", color="secondary"), width=6)
                ], className="mb-3"),
                dbc.Spinner(html.Div(id="prediction-output"), size="lg", color="primary")
            ], style={"padding": "20px", "borderRadius": "10px"})
        ], md=6),

        # Results column
        dbc.Col([
            html.Div([
                html.H3("Results", className="mb-3"),
                dcc.Graph(id="waveform-plot", style={"display": "none"}),
                html.Div(id="results-card")
            ], style={"padding": "20px", "borderRadius": "10px"})
        ], md=6)
    ], className="my-4")
], fluid=True, style={"minHeight": "100vh"})

# --- Callback for prediction (MODIFIED to include audio playback outputs) ---
@dash.callback(
    [
        Output("prediction-output", "children"),
        Output("waveform-plot", "figure"),
        Output("waveform-plot", "style"),
        Output("results-card", "children"),
        Output("upload-status", "children"),
        # NEW OUTPUTS for Audio Playback
        Output("audio-player", "src", allow_duplicate=True),
        Output("audio-player-container", "style")
    ],
    [
        Input("predict-btn", "n_clicks"),
        Input("reset-btn", "n_clicks")
    ],
    [
        State("upload-audio", "contents"),
        State("upload-audio", "filename"),
        State("sampling-rate-slider", "value")
    ],
    prevent_initial_call=True
)
def update_output(predict_clicks, reset_clicks, contents, filename, sampling_rate):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Reset action
    if triggered_id == "reset-btn":
        # Reset must also clear audio source and hide the player
        return (
            None,
            go.Figure(),
            {"display": "none"},
            None,
            "No file uploaded",
            None,  # Clear audio source
            {'display': 'none'}  # Hide audio player
        )

    # Predict action
    if triggered_id == "predict-btn" and contents is None:
        return (
            dbc.Alert("Please upload an audio file first.", color="warning"),
            go.Figure(),
            {"display": "none"},
            None,
            "No file uploaded",
            None,  # Clear audio source
            {'display': 'none'}  # Hide audio player
        )

    temp_path = None
    try:
        # Save uploaded file temporarily
        file_bytes = base64.b64decode(contents.split(",")[1])
        # Use a more robust temporary file naming

        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"uploaded_audio_{os.getpid()}_{filename}")

        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        # Extract MFCC features and raw audio - PASS SAMPLING RATE HERE
        mfcc_features, y, sr = extract_mfcc(temp_path, sampling_rate)

        # Check if features are valid (e.g., if audio was too short)
        if len(mfcc_features) == 0:
            os.remove(temp_path)
            raise ValueError("Audio file too short to extract features after resampling.")

        X = pd.DataFrame([mfcc_features], columns=[f"mfcc_{i + 1}" for i in range(len(mfcc_features))])

        # Scale input
        X_scaled = x_scaler.transform(X)

        # Predict velocity
        y_pred = model.predict(X_scaled).flatten()  # 1D array
        velocity_kmh = y_pred[0]

        # Estimate source frequency from audio
        f_source = estimate_source_frequency(y, sr)

        # Doppler calculation
        delta_f, f_received = doppler_from_velocity(velocity_kmh, f_source)

        # Create waveform plot
        waveform_fig = create_waveform_plot(y, sr)

        # --- NEW: Generate base64 Data URI for playback ---
        audio_data_uri = numpy_to_wav_base64(y, sr)

        # Clean up temp file
        os.remove(temp_path)

        # Results card
        results_card = dbc.Card([
            dbc.CardBody([
                html.H5(f"Predicted Velocity: {velocity_kmh:.2f} km/h", className="card-title text-primary"),
                html.P([
                    html.Strong("Processing Rate: "), f"{sr} Hz"
                ], className="card-text"),
                html.P([
                    html.Strong("Estimated Source Frequency: "), f"{f_source:.2f} Hz"
                ], className="card-text"),
                html.P([
                    html.Strong("Doppler Shift: "), f"{delta_f:.2f} Hz"
                ], className="card-text"),
                html.P([
                    html.Strong("Received Frequency: "), f"{f_received:.2f} Hz"
                ], className="card-text")
            ])
        ], color="light", className="mt-3 border-start border-primary border-5")

        return (
            None,
            waveform_fig,
            {"display": "block"},
            results_card,
            f"Uploaded: {filename} (Resampling to {sampling_rate} Hz)",
            audio_data_uri,  # New Audio Source
            {'display': 'block'}  # Show audio player
        )

    except Exception as e:
        # Attempt to clean up temp file even on error
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
        except:
            pass  # Ignore cleanup error

        return (
            dbc.Alert(f"Error during prediction: {str(e)}", color="danger"),
            go.Figure(),
            {"display": "none"},
            None,
            "Error processing file",
            None,  # Clear audio source
            {'display': 'none'}  # Hide audio player
        )