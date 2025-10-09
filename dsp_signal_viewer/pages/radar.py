import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # hide TF logs

import base64
import io
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State
import pandas as pd
import numpy as np
from tensorflow import keras
import joblib
import librosa

# --- Register page ---
dash.register_page(__name__, path="/radar", name="Radar")

# --- Define BASE_DIR and paths ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # dsp_signal_viewer/
model_path = os.path.join(BASE_DIR, "models", "velocity_model.h5")
x_scaler_path = os.path.join(BASE_DIR, "models", "x_scaler.pkl")

# --- Load model and scaler ---
print("Loading model from:", model_path)
print("File exists?", os.path.exists(model_path))

model = keras.models.load_model(model_path, compile=False)
x_scaler = joblib.load(x_scaler_path)

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

# --- Extract MFCC features from uploaded audio ---
def extract_mfcc(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=None)
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
    freqs = np.fft.rfftfreq(n, 1/sr)
    magnitude = np.abs(Y)
    dominant_idx = np.argmax(magnitude)
    f_source = freqs[dominant_idx]
    return f_source

# --- Dash layout ---
layout = dbc.Container([
    html.H2("Radar Velocity Prediction", className="text-center my-3"),

    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id="upload-audio",
                children=html.Div([
                    "Drag and Drop or ",
                    html.A("Select Audio File")
                ]),
                style={
                    "width": "100%",
                    "height": "60px",
                    "lineHeight": "60px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                    "textAlign": "center",
                    "margin-bottom": "10px"
                },
                multiple=False
            ),
            dbc.Button("Predict", id="predict-btn", color="primary", className="mt-2"),
            dbc.Spinner(html.Div(id="prediction-output"), size="lg", color="primary")
        ], md=6)
    ])
], fluid=True)

# --- Callback for prediction ---
@dash.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("upload-audio", "contents"),
    State("upload-audio", "filename"),
    prevent_initial_call=True
)
def predict_velocity(n_clicks, contents, filename):
    if contents is None:
        return dbc.Alert("Please upload an audio file first.", color="warning")

    try:
        # Save uploaded file temporarily
        file_bytes = base64.b64decode(contents.split(",")[1])
        temp_path = os.path.join("/tmp", filename)
        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        # Extract MFCC features and raw audio
        mfcc_features, y, sr = extract_mfcc(temp_path)
        X = pd.DataFrame([mfcc_features], columns=[f"mfcc_{i+1}" for i in range(len(mfcc_features))])

        # Scale input
        X_scaled = x_scaler.transform(X)

        # Predict velocity
        y_pred = model.predict(X_scaled).flatten()  # 1D array
        velocity_kmh = y_pred[0]

        # Estimate source frequency from audio
        f_source = estimate_source_frequency(y, sr)

        # Doppler calculation
        delta_f, f_received = doppler_from_velocity(velocity_kmh, f_source)

        return dbc.Card([
            dbc.CardBody([
                html.H5(f"Predicted Velocity: {velocity_kmh:.2f} km/h"),
                html.H5(f"Estimated Source Frequency: {f_source:.2f} Hz"),

            ])
        ], color="dark", className="mt-3")

    except Exception as e:
        return dbc.Alert(f"Error during prediction: {e}", color="danger")
