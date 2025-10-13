import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import base64
import io
import time as pytime
from keras.models import load_model
from scipy.signal import resample, decimate
import wfdb

dash.register_page(__name__, path="/ecg-downsampling", name="ECG Down Sampling")

# --- Parameters ---
WINDOW_SIZE = 2500
STEP = 250
FS = 250
DOWNSAMPLE = 4
ECG_LEADS = ["i", "ii", "iii", "avr", "avl", "avf",
             "v1", "v2", "v3", "v4", "v5", "v6"]

# --- Colors ---
ACTIVE_STYLE = {"backgroundColor": "#2c7f91", "borderColor": "#2c7f91", "color": "white", "marginRight": "10px"}
INACTIVE_STYLE = {"backgroundColor": "#3E9AAB", "borderColor": "#3E9AAB", "color": "white", "marginRight": "10px"}

# --- Load model ---
MODEL_PATH = r"models\model.hdf5"
model = load_model(MODEL_PATH, compile=False)
ABNORMALITIES = ["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"]
THRESHOLD = 0.3

# Downsampling options - FIXED: Only standard Dropdown format
DOWNSAMPLING_OPTIONS = [
    {"label": "Original (250 Hz)", "value": "1"},
    {"label": "2x Downsample (125 Hz)", "value": "2"},
    {"label": "4x Downsample (62.5 Hz)", "value": "4"},
    {"label": "8x Downsample (31.25 Hz)", "value": "8"},
    {"label": "16x Downsample (15.625 Hz)", "value": "16"},
]

# Nyquist rates mapping
NYQUIST_RATES = {
    "1": 125,
    "2": 62.5,
    "4": 31.25,
    "8": 15.625,
    "16": 7.8125
}

# --- Layout ---
layout = dbc.Container([
    html.H2("ECG Down Sampling - Nyquist Rate Effect", className="text-center"),
    html.Hr(),

    # --- File Upload ---
    dcc.Upload(
        id="upload-ecg-downsample",
        children=html.Div(["Drag and Drop or Select a CSV File"]),
        style={
            "width": "100%", "height": "60px", "lineHeight": "60px",
            "borderWidth": "1px", "borderStyle": "dashed",
            "borderRadius": "5px", "textAlign": "center",
            "marginBottom": "10px"
        },
        multiple=False
    ),

    # --- Controls Row ---
    dbc.Row([
        # Channels selector
        dbc.Col([
            html.Label("Select Channels"),
            dcc.Dropdown(
                id="channel-select-downsample",
                options=[{"label": ch.upper(), "value": ch} for ch in ECG_LEADS],
                value=ECG_LEADS,
                multi=True,
                clearable=False,
                style={"color": "#182940", "width": "100%"}
            )
        ], width=4),  # Increased width

        # Downsampling selector
        dbc.Col([
            html.Label("Down Sampling Factor"),
            dcc.Dropdown(
                id="downsample-select",
                options=DOWNSAMPLING_OPTIONS,
                value="1",
                clearable=False,
                style={"color": "#182940", "width": "100%"}
            ),
            html.Div(id="nyquist-display", style={
                "fontWeight": "bold", 
                "marginTop": "5px", 
                "color": "#3E9AAB",
                "fontSize": "14px"
            })
        ], width=3),

        # Start/Stop button with elapsed time
        dbc.Col([
            html.Div([
                dbc.Button(
                    "▶ Start",
                    id="start-stop-btn-downsample",
                    size="md",
                    style={
                        "backgroundColor": "#3E9AAB",
                        "borderColor": "#3E9AAB",
                        "color": "white",
                        "fontSize": "14px",
                        "height": "38px",
                        "padding": "0 15px"
                    }
                ),
                html.Div(id="timer-display-downsample", style={"fontWeight": "bold", "marginTop": "5px", "textAlign": "center"})
            ], style={"display": "flex", "flexDirection": "column", "alignItems": "center"})
        ], width=2),

        # Predict button + prediction text
        dbc.Col([
            dbc.Button("Predict ECG", id="predict-btn-downsample", color="primary", n_clicks=0, style={"marginRight": "10px"}),
            html.Div(id="ecg-prediction-output-downsample", style={
                "fontWeight": "bold", "fontSize": "16px", "display": "inline-block"
            })
        ], width=3)  # Increased width
    ], align="center", className="mb-4"),

    # --- Information Display ---
    dbc.Row([
        dbc.Col([
            dbc.Alert(
                id="nyquist-info",
                color="info",
                style={"marginBottom": "20px"}
            )
        ], width=12)
    ]),

    html.Br(),
    dcc.Graph(id="ecg-graph-downsample", style={"height": "900px"}),

    dcc.Interval(id="interval-downsample", interval=1000, n_intervals=0, disabled=True),
    dcc.Store(id="is-running-downsample", data=False),
    dcc.Store(id="start-time-downsample", data=None),
    dcc.Store(id="ecg-data-downsample", data=None),
    dcc.Store(id="current-position-downsample", data=0),
    dcc.Store(id="ecg-prediction-result-downsample", data=""),  # Store the prediction result
], fluid=True)

# --- Initialize the app with callback exception suppression ---
app = dash.get_app()
app.config.suppress_callback_exceptions = True

# --- Update Nyquist display ---
@dash.callback(
    Output("nyquist-display", "children"),
    Input("downsample-select", "value")
)
def update_nyquist_display(downsample_factor):
    original_fs = FS
    new_fs = original_fs / int(downsample_factor)
    nyquist_rate = new_fs / 2
    
    # Get nyquist rate from mapping
    nyquist = NYQUIST_RATES.get(downsample_factor, nyquist_rate)
    return f"Nyquist: {nyquist} Hz"

# --- Update Nyquist information alert ---
@dash.callback(
    Output("nyquist-info", "children"),
    Input("downsample-select", "value"),
    Input("ecg-prediction-result-downsample", "data")
)
def update_nyquist_info(downsample_factor, prediction_result):
    downsample_factor_int = int(downsample_factor)
    original_fs = FS
    new_fs = original_fs / downsample_factor_int
    nyquist_rate = new_fs / 2
    
    # ECG frequency components information
    ecg_components = {
        "P Wave": "0.67-5 Hz",
        "QRS Complex": "10-50 Hz", 
        "T Wave": "1-7 Hz",
        "ST Segment": "0.67-5 Hz",
        "High Frequency QRS": "150-250 Hz"
    }
    
    # Determine which components are affected
    affected_components = []
    preserved_components = []
    
    for component, freq_range in ecg_components.items():
        # Extract max frequency from range
        max_freq = float(freq_range.split('-')[-1].split(' ')[0])
        if max_freq > nyquist_rate:
            affected_components.append(f"{component} ({freq_range})")
        else:
            preserved_components.append(f"{component} ({freq_range})")
    
    # Create information message
    info_parts = [
        f"Current Sampling: {new_fs} Hz | Nyquist Rate: {nyquist_rate:.2f} Hz"
    ]
    
    if affected_components:
        info_parts.append(f"⚠️ Affected: {', '.join(affected_components)}")
    
    if preserved_components:
        info_parts.append(f"✓ Preserved: {', '.join(preserved_components)}")
    
    # Add prediction effect information
    if prediction_result:
        info_parts.append(f"📊 Prediction: {prediction_result.upper()}")
        if affected_components and prediction_result == "abnormal":
            info_parts.append("🔍 Downsampling may affect abnormality detection!")
    
    return " | ".join(info_parts)

# --- Run prediction on button click ---
@dash.callback(
    Output("ecg-prediction-output-downsample", "children"),
    Output("ecg-prediction-result-downsample", "data"),
    Input("predict-btn-downsample", "n_clicks"),
    State("ecg-data-downsample", "data"),
    State("channel-select-downsample", "value"),
    State("downsample-select", "value"),
    prevent_initial_call=True
)
def predict_ecg(n_clicks, data_json, selected_channels, downsample_factor):
    if data_json is None or not selected_channels:
        return "⚠️ No ECG data or channels selected.", ""
    
    df = pd.read_json(data_json, orient="split")
    df.columns = [c.lower() for c in df.columns]
    
    # Apply downsampling to the data before prediction
    downsample_factor_int = int(downsample_factor)
    if downsample_factor_int > 1:
        df = apply_downsampling(df, downsample_factor_int)
    
    # Keep only selected channels
    df = df[selected_channels]

    try:
        result_text, is_normal = run_ecg_prediction(df)
        return result_text, "normal" if is_normal else "abnormal"
    except Exception as e:
        error_text = f"⚠️ Prediction failed: {e}"
        return error_text, "error"

# --- Helper: Apply downsampling to ECG data ---
def apply_downsampling(df, downsample_factor):
    """Apply downsampling to all channels in the DataFrame"""
    if downsample_factor <= 1:
        return df
    
    downsampled_data = {}
    for column in df.columns:
        signal = df[column].values
        # Use scipy's decimate for better anti-aliasing
        downsampled_signal = decimate(signal, downsample_factor, zero_phase=True)
        downsampled_data[column] = downsampled_signal
    
    return pd.DataFrame(downsampled_data)

# --- Helper: Run ECG model prediction ---
def run_ecg_prediction(df):
    # Ensure all 12 leads exist
    lead_map = {"i":"I","ii":"II","iii":"III","avr":"AVR","avl":"AVL","avf":"AVF",
                "v1":"V1","v2":"V2","v3":"V3","v4":"V4","v5":"V5","v6":"V6"}
    df_model = pd.DataFrame()
    for ch in lead_map:
        df_model[lead_map[ch]] = df[ch] if ch in df.columns else 0.0

    # Resample to 4096
    target_length = 4096
    num_samples = df_model.shape[0]
    if num_samples != target_length:
        df_resampled = resample(df_model.to_numpy(), target_length, axis=0)
    else:
        df_resampled = df_model.to_numpy()

    # Normalize
    df_norm = (df_resampled - np.mean(df_resampled, axis=0)) / (np.std(df_resampled, axis=0) + 1e-8)
    input_tensor = np.expand_dims(df_norm, axis=0)

    # Predict
    pred = model.predict(input_tensor)[0]
    detected = [(ab, p) for ab, p in zip(ABNORMALITIES, pred) if p >= THRESHOLD]

    if len(detected) == 0:
        return " Normal (no significant abnormality detected)", True
    else:
        diseases = [ab for ab, _ in detected]
        return f" Abnormal: Detected conditions → {', '.join(diseases)}", False

# --- CSV Upload callback ---
@dash.callback(
    Output("ecg-data-downsample", "data"),
    Output("channel-select-downsample", "options"),
    Output("channel-select-downsample", "value"),
    Output("current-position-downsample", "data", allow_duplicate=True),
    Input("upload-ecg-downsample", "contents"),
    State("upload-ecg-downsample", "filename"),
    prevent_initial_call=True
)
def load_csv(contents, filename):
    if contents is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8-sig')))
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None, [], [], 0

    df.columns = [c.lower().strip() for c in df.columns]
    available = [lead for lead in ECG_LEADS if lead in df.columns]
    df = df[available]

    options = [{"label": ch.upper(), "value": ch} for ch in available]
    return df.to_json(date_format="iso", orient="split"), options, available, 0

# --- Start/Stop toggle ---
@dash.callback(
    Output("is-running-downsample", "data"),
    Output("interval-downsample", "disabled"),
    Output("start-stop-btn-downsample", "children"),
    Output("start-stop-btn-downsample", "style"),
    Output("start-time-downsample", "data"),
    Input("start-stop-btn-downsample", "n_clicks"),
    State("is-running-downsample", "data"),
    State("start-time-downsample", "data"),
    prevent_initial_call=True
)
def toggle_run(n_clicks, is_running, start_time):
    if not is_running:
        style = {"backgroundColor": "#d9534f", "borderColor": "#d9534f", "color": "white",
                 "height": "38px", "padding": "0 15px", "marginTop": "5px", "display": "block"}
        return True, False, "⏸ Stop", style, pytime.time()
    else:
        style = {"backgroundColor": "#3E9AAB", "borderColor": "#3E9AAB", "color": "white",
                 "height": "38px", "padding": "0 15px", "marginTop": "5px", "display": "block"}
        return False, True, "▶ Start", style, start_time

# --- Helper: Get looping data ---
def get_looping_data(df, current_position, window_size, step):
    """Get data that loops when reaching the end"""
    total_samples = len(df)
    
    if total_samples == 0:
        return pd.DataFrame(), current_position
    
    # Calculate end position
    end_position = current_position + window_size
    
    if end_position <= total_samples:
        # Normal case - no looping needed
        data_slice = df.iloc[current_position:end_position]
        new_position = current_position + step
    else:
        # Need to loop - combine end and beginning
        samples_from_end = total_samples - current_position
        samples_from_start = window_size - samples_from_end
        
        part1 = df.iloc[current_position:]
        part2 = df.iloc[:samples_from_start]
        
        data_slice = pd.concat([part1, part2], ignore_index=True)
        new_position = samples_from_start
    
    # Ensure we don't exceed data length
    if new_position >= total_samples:
        new_position = new_position % total_samples
    
    return data_slice, new_position

# --- Combined callback for timer and graph updates ---
@dash.callback(
    Output("ecg-graph-downsample", "figure"),
    Output("timer-display-downsample", "children"),
    Output("current-position-downsample", "data"),
    Input("interval-downsample", "n_intervals"),
    Input("upload-ecg-downsample", "contents"),
    Input("downsample-select", "value"),
    Input("channel-select-downsample", "value"),
    State("start-time-downsample", "data"),
    State("is-running-downsample", "data"),
    State("ecg-data-downsample", "data"),
    State("current-position-downsample", "data"),
    prevent_initial_call=True
)
def update_graph_and_timer(n_intervals, upload_contents, downsample_factor, selected_channels, 
                          start_time, is_running, data_json, current_position):
    
    # Use callback context to determine which input triggered the callback
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    line_width = 1.5 
    
    # Update timer display
    timer_text = ""
    if is_running and start_time:
        elapsed = int(pytime.time() - start_time)
        timer_text = f"Elapsed Time: {elapsed} s"
    
    # If no data or channels selected, return empty figure
    if data_json is None or not selected_channels:
        fig = go.Figure()
        fig.update_layout(
            title="No data loaded or channel selected",
            template="plotly_white",
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        return fig, timer_text, current_position

    df = pd.read_json(data_json, orient="split")
    df.columns = [c.lower() for c in df.columns]
    df = df[selected_channels]

    # Apply downsampling for display
    downsample_factor_int = int(downsample_factor)
    if downsample_factor_int > 1:
        df = apply_downsampling(df, downsample_factor_int)

    total_samples = df.shape[0]
    current_fs = FS / downsample_factor_int
    current_window_size = int(WINDOW_SIZE / downsample_factor_int)  # Adjust window size for downsampled data
    
    # Only update position if the interval triggered the callback and is running
    if triggered_id == "interval-downsample" and is_running:
        # Get data with looping and update position
        data_slice, new_position = get_looping_data(df, current_position, current_window_size, STEP // downsample_factor_int)
        current_position = new_position
    else:
        # For other triggers (like upload or parameter changes), just get current data without updating position
        data_slice, _ = get_looping_data(df, current_position, current_window_size, 0)
    
    t_window = np.arange(len(data_slice)) / current_fs

    # Create regular ECG plot
    fig = make_subplots(
        rows=len(selected_channels), cols=1, shared_xaxes=True,
        subplot_titles=[f"{ch.upper()} (Downsampled {downsample_factor_int}x)" for ch in selected_channels]
    )
    
    for i, ch in enumerate(selected_channels):
        y = data_slice[ch].values
        x = t_window
        fig.add_trace(
            go.Scatter(x=x, y=y, mode="lines",
                       line=dict(color="blue", width=line_width),
                       name=ch.upper(), showlegend=False),
            row=i+1, col=1
        )
    
    for i in range(len(selected_channels)):
        fig.update_xaxes(row=i+1, col=1, tickfont=dict(size=9, color="black"))
        fig.update_yaxes(row=i+1, col=1, tickfont=dict(size=9, color="black"),
                         title_standoff=50)
    
    # Get current downsampling option for display
    current_option = next((opt for opt in DOWNSAMPLING_OPTIONS if opt["value"] == downsample_factor), None)
    sampling_info = current_option["label"] if current_option else f"Downsampled {downsample_factor_int}x"
    
    fig.update_layout(
        height=250 * len(selected_channels),
        template="plotly_white",
        margin=dict(l=60, r=40, t=100, b=40),
        plot_bgcolor="white", paper_bgcolor="white",
        title=f"ECG Signal - {sampling_info} | Position: {current_position}/{total_samples} | Effective FS: {current_fs} Hz"
    )

    return fig, timer_text, current_position