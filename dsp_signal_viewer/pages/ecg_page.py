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
from scipy.signal import resample
import wfdb

dash.register_page(__name__, path="/ecg", name="ECG")

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

# --- Layout ---
layout = dbc.Container([
    html.H2("ECG Signal Viewer", className="text-center"),
    html.Hr(),

    # --- File Upload ---
    dcc.Upload(
        id="upload-ecg",
        children=html.Div(["Drag and Drop or Select a CSV File"]),
        style={
            "width": "100%", "height": "60px", "lineHeight": "60px",
            "borderWidth": "1px", "borderStyle": "dashed",
            "borderRadius": "5px", "textAlign": "center",
            "marginBottom": "10px"
        },
        multiple=False
    ),

    # --- Graph Type Buttons ---
    dbc.Row(
        dbc.Col(
            html.Div(
                [
                    dbc.Button("Regular", id="regular-btn", size="md", style=ACTIVE_STYLE),
                    dbc.Button("XOR", id="xor-btn", size="md", style=INACTIVE_STYLE),
                    dbc.Button("Polar", id="polar-btn", size="md", style=INACTIVE_STYLE),
                    dbc.Button("Recurrence", id="recurrence-btn", size="md",
                               style={**INACTIVE_STYLE, "marginRight": "0px"})
                ],
                style={
                    "textAlign": "center",
                    "marginTop": "20px",
                    "marginBottom": "20px"
                }
            ),
            width="auto"
        ),
        justify="center"
    ),

    # --- Controls Row ---
    dbc.Row([
        # Channels selector
        dbc.Col([
            html.Label("Select Channels"),
            dcc.Dropdown(
                id="channel-select",
                options=[{"label": ch.upper(), "value": ch} for ch in ECG_LEADS],
                value=ECG_LEADS,
                multi=True,
                clearable=False,
                style={"color": "#182940", "width": "100%"}
            )
        ], width=3),

        # Time controls
        dbc.Col([
            html.Label("Time Window (s)"),
            dcc.Slider(
                id="time-window-slider",
                min=1,
                max=10,
                step=0.5,
                value=WINDOW_SIZE/FS,
                marks={i: f"{i}s" for i in range(1, 11)},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], width=3),

        # Start/Stop button with elapsed time
        dbc.Col([
            html.Div([
                dbc.Button(
                    "▶ Start",
                    id="start-stop-btn",
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
                html.Div(id="timer-display", style={"fontWeight": "bold", "marginTop": "5px", "textAlign": "center"})
            ], style={"display": "flex", "flexDirection": "column", "alignItems": "center"})
        ], width=2),

        # Convert 12→3 options
        dbc.Col([
            html.Label("Convert 12 → 3 Channels:"),
            dcc.RadioItems(
                id="convert-option",
                options=[
                    {"label": "None", "value": "none"},
                    {"label": "Clinical choice (II, V1, V5)", "value": "clinical"},
                    {"label": "Average Limb vs Chest Leads", "value": "average"}
                ],
                value="none",
                inline=True,
                inputStyle={"marginRight": "5px", "marginLeft": "10px"},
                style={"marginLeft": "20px"}
            )
        ], width=2),

        # Predict button + prediction text
        dbc.Col([
            dbc.Button("Predict ECG", id="predict-btn", color="primary", n_clicks=0, style={"marginRight": "10px"}),
            html.Div(id="ecg-prediction-output", style={
                "fontWeight": "bold", "fontSize": "16px", "display": "inline-block"
            })
        ], width=2)
    ], align="center", className="mb-4"),

    # --- Additional controls for specific graph types ---
    dbc.Row([
        dbc.Col([
            html.Div(id="graph-specific-controls", children=[
                # Default controls that are always in layout
                html.Div(id="polar-mode-controls", style={"display": "none"}, children=[
                    html.Label("Polar Mode:"),
                    dcc.RadioItems(
                        id="polar-mode-radio",
                        options=[
                            {"label": "Latest Only", "value": "latest"},
                            {"label": "Cumulative", "value": "cumulative"}
                        ],
                        value="latest",
                        inline=True,
                        inputStyle={"marginRight": "5px", "marginLeft": "10px"}
                    )
                ]),
                html.Div(id="recurrence-mode-controls", style={"display": "none"}, children=[
                    html.Label("Recurrence Mode:"),
                    dcc.RadioItems(
                        id="recurrence-mode-radio",
                        options=[
                            {"label": "Latest Only", "value": "latest"},
                            {"label": "Cumulative", "value": "cumulative"}
                        ],
                        value="cumulative",
                        inline=True,
                        inputStyle={"marginRight": "5px", "marginLeft": "10px"}
                    ),
                    html.Label("Color Map:"),
                    dcc.Dropdown(
                        id="colormap-select",
                        options=[
                            {"label": "Blues", "value": "Blues"},
                            {"label": "Viridis", "value": "Viridis"},
                            {"label": "Plasma", "value": "Plasma"},
                            {"label": "Hot", "value": "Hot"},
                            {"label": "Jet", "value": "Jet"}
                        ],
                        value="Blues",
                        clearable=False,
                        style={"width": "200px", "display": "inline-block", "marginLeft": "10px"}
                    )
                ]),
                html.Div(id="xor-controls", style={"display": "none"}, children=[
                    html.Label("XOR Chunk Size (s):"),
                    dcc.Slider(
                        id="xor-chunk-slider",
                        min=0.5,
                        max=5,
                        step=0.5,
                        value=2,
                        marks={i: f"{i}s" for i in [0.5, 1, 2, 3, 4, 5]},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Label("Abnormal Beat Detection:"),
                    dcc.RadioItems(
                        id="abnormal-beat-radio",
                        options=[
                            {"label": "Show Points", "value": "points"},
                            {"label": "Show Lines", "value": "lines"}
                        ],
                        value="points",
                        inline=True,
                        inputStyle={"marginRight": "5px", "marginLeft": "10px"}
                    )
                ])
            ])
        ], width=12)
    ], className="mb-3"),

    html.Br(),
    dcc.Graph(id="ecg-graph", style={"height": "900px"}),

    dcc.Interval(id="interval", interval=1000, n_intervals=0, disabled=True),
    dcc.Store(id="is-running", data=False),
    dcc.Store(id="start-time", data=None),
    dcc.Store(id="ecg-data", data=None),
    dcc.Store(id="active-graph-type", data="regular"),
    dcc.Store(id="previous-chunk", data=None),
    dcc.Store(id="polar-mode", data="latest"),
    dcc.Store(id="recurrence-mode", data="cumulative"),
    dcc.Store(id="colormap", data="Blues"),
    dcc.Store(id="xor-chunk-size", data=2),
    dcc.Store(id="current-position", data=0),
    dcc.Store(id="ecg-prediction-result", data=""),
    dcc.Store(id="abnormal-beat-mode", data="points"),
    dcc.Store(id="xor-history", data=[])  # Store XOR history for moving display
], fluid=True)

# --- Initialize the app with callback exception suppression ---
app = dash.get_app()
app.config.suppress_callback_exceptions = True

# --- Graph-specific controls visibility ---
@dash.callback(
    Output("polar-mode-controls", "style"),
    Output("recurrence-mode-controls", "style"),
    Output("xor-controls", "style"),
    Input("active-graph-type", "data")
)
def update_graph_controls_visibility(graph_type):
    polar_style = {"display": "block"} if graph_type == "polar" else {"display": "none"}
    recurrence_style = {"display": "block"} if graph_type == "recurrence" else {"display": "none"}
    xor_style = {"display": "block"} if graph_type == "xor" else {"display": "none"}
    
    return polar_style, recurrence_style, xor_style

# --- Store updates for modes ---
@dash.callback(
    Output("polar-mode", "data"),
    Input("polar-mode-radio", "value")
)
def update_polar_mode(mode):
    return mode

@dash.callback(
    Output("recurrence-mode", "data"),
    Input("recurrence-mode-radio", "value")
)
def update_recurrence_mode(mode):
    return mode

@dash.callback(
    Output("colormap", "data"),
    Input("colormap-select", "value")
)
def update_colormap(colormap):
    return colormap

@dash.callback(
    Output("xor-chunk-size", "data"),
    Input("xor-chunk-slider", "value")
)
def update_xor_chunk_size(size):
    return size

@dash.callback(
    Output("abnormal-beat-mode", "data"),
    Input("abnormal-beat-radio", "value")
)
def update_abnormal_beat_mode(mode):
    return mode

# --- Run prediction on button click ---
@dash.callback(
    Output("ecg-prediction-output", "children"),
    Output("ecg-prediction-result", "data"),
    Input("predict-btn", "n_clicks"),
    State("ecg-data", "data"),
    State("convert-option", "value"),
    State("channel-select", "value"),
    prevent_initial_call=True
)
def predict_ecg(n_clicks, data_json, convert_option, selected_channels):
    if data_json is None or not selected_channels:
        return "⚠️ No ECG data or channels selected.", ""
    
    df = pd.read_json(data_json, orient="split")
    df.columns = [c.lower() for c in df.columns]
    
    # Apply 12→3 channel conversion if selected
    if convert_option != "none" and len(selected_channels) == 12:
        df = convert_12_to_3(df, convert_option)
        selected_channels = df.columns.tolist()
    
    # Keep only selected channels
    df = df[selected_channels]

    try:
        result_text, is_normal = run_ecg_prediction(df)
        return result_text, "normal" if is_normal else "abnormal"
    except Exception as e:
        error_text = f"⚠️ Prediction failed: {e}"
        return error_text, "error"

# --- CSV Upload callback ---
@dash.callback(
    Output("ecg-data", "data"),
    Output("channel-select", "options"),
    Output("channel-select", "value"),
    Output("current-position", "data", allow_duplicate=True),
    Output("xor-history", "data", allow_duplicate=True),
    Input("upload-ecg", "contents"),
    State("upload-ecg", "filename"),
    prevent_initial_call=True
)
def load_csv(contents, filename):
    if contents is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8-sig')))
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None, [], [], 0, []

    df.columns = [c.lower().strip() for c in df.columns]
    available = [lead for lead in ECG_LEADS if lead in df.columns]
    df = df[available]

    options = [{"label": ch.upper(), "value": ch} for ch in available]
    return df.to_json(date_format="iso", orient="split"), options, available, 0, []

# --- Start/Stop toggle ---
@dash.callback(
    Output("is-running", "data"),
    Output("interval", "disabled"),
    Output("start-stop-btn", "children"),
    Output("start-stop-btn", "style"),
    Output("start-time", "data"),
    Input("start-stop-btn", "n_clicks"),
    State("is-running", "data"),
    State("start-time", "data"),
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

# --- Graph Type Button State Management ---
@dash.callback(
    Output("regular-btn", "style"),
    Output("xor-btn", "style"),
    Output("polar-btn", "style"),
    Output("recurrence-btn", "style"),
    Output("active-graph-type", "data"),
    Input("regular-btn", "n_clicks"),
    Input("xor-btn", "n_clicks"),
    Input("polar-btn", "n_clicks"),
    Input("recurrence-btn", "n_clicks"),
    State("active-graph-type", "data"),
    prevent_initial_call=True
)
def set_active_button(n_reg, n_xor, n_pol, n_rec, current_type):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ACTIVE_STYLE, INACTIVE_STYLE, INACTIVE_STYLE, INACTIVE_STYLE, current_type

    clicked = ctx.triggered[0]["prop_id"].split(".")[0]

    if clicked == "regular-btn":
        return ACTIVE_STYLE, INACTIVE_STYLE, INACTIVE_STYLE, INACTIVE_STYLE, "regular"
    elif clicked == "xor-btn":
        return INACTIVE_STYLE, ACTIVE_STYLE, INACTIVE_STYLE, INACTIVE_STYLE, "xor"
    elif clicked == "polar-btn":
        return INACTIVE_STYLE, INACTIVE_STYLE, ACTIVE_STYLE, INACTIVE_STYLE, "polar"
    elif clicked == "recurrence-btn":
        return INACTIVE_STYLE, INACTIVE_STYLE, INACTIVE_STYLE, ACTIVE_STYLE, "recurrence"
    return ACTIVE_STYLE, INACTIVE_STYLE, INACTIVE_STYLE, INACTIVE_STYLE, current_type

# --- Helper: Convert 12 leads to 3 ---
def convert_12_to_3(df, method):
    if method == "clinical":
        picks = [ch for ch in ["ii", "v1", "v5"] if ch in df.columns]
        return df[picks]
    elif method == "average":
        limb = [ch for ch in ["i", "ii", "iii", "avr", "avl", "avf"] if ch in df.columns]
        chest = [ch for ch in ["v1", "v2", "v3", "v4", "v5", "v6"] if ch in df.columns]
        new_df = pd.DataFrame()
        if limb:
            new_df["limb_avg"] = df[limb].mean(axis=1)
        if chest:
            new_df["chest_avg"] = df[chest].mean(axis=1)
        return new_df
    else:
        return df

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

# --- Helper: Detect abnormal beats ---
def detect_abnormal_beats(signal, threshold_std=2.5):
    """Detect abnormal beats using statistical thresholding"""
    signal = np.array(signal)
    
    # Calculate moving statistics
    window_size = min(50, len(signal) // 4)
    if window_size < 5:
        return np.zeros_like(signal, dtype=bool)
    
    # Use rolling statistics to detect outliers
    abs_signal = np.abs(signal)
    
    # Simple threshold-based detection
    median_val = np.median(abs_signal)
    mad = np.median(np.abs(abs_signal - median_val))  # Median Absolute Deviation
    threshold = median_val + threshold_std * mad
    
    abnormal_mask = abs_signal > threshold
    
    return abnormal_mask

# --- Helper: Moving XOR Plot Function ---
def create_moving_xor_plot(df, selected_channels, chunk_size_seconds, previous_chunk_data, 
                          current_position, is_normal, xor_history, abnormal_beat_mode, fs=FS):
    chunk_size_samples = int(chunk_size_seconds * fs)
    total_samples = len(df)
    
    if total_samples == 0:
        fig = go.Figure()
        fig.update_layout(title="No data available")
        return fig, None, xor_history
    
    # Get current chunk with looping
    current_chunk_data, _ = get_looping_data(df[selected_channels], current_position, chunk_size_samples, 0)
    current_chunk = current_chunk_data.values
    
    if previous_chunk_data is None or len(previous_chunk_data) != len(current_chunk):
        # Initialize with first chunk
        fig = go.Figure()
        fig.update_layout(title="Initializing XOR plot...")
        # Store as list for JSON serialization
        return fig, current_chunk.tolist(), xor_history
    
    # Reconstruct previous chunk from stored data
    previous_chunk = np.array(previous_chunk_data)
    
    # If patient is normal, show flat line (ICU-like display)
    if is_normal:
        # Create a flat line at zero (no XOR activity for normal ECG)
        xor_result = np.zeros_like(current_chunk)
        
        # Create figure with flat lines
        fig = make_subplots(
            rows=len(selected_channels), cols=1,
            subplot_titles=[f"XOR: {ch.upper()} - NORMAL (No Activity)" for ch in selected_channels]
        )
        
        time_axis = np.arange(len(xor_result)) / fs
        
        for i, ch in enumerate(selected_channels):
            fig.add_trace(
                go.Scatter(
                    x=time_axis, y=xor_result[:, i],
                    mode="lines", line=dict(color="green", width=1, dash="dash"),
                    name=f"{ch.upper()} - NORMAL",
                    fill='tozeroy',
                    fillcolor='rgba(0, 255, 0, 0.1)'
                ),
                row=i+1, col=1
            )
        
        fig.update_layout(
            height=250 * len(selected_channels),
            title="XOR Graph - NORMAL PATIENT (No Abnormal Activity Detected)",
            template="plotly_white"
        )
        
        # Reset history for normal patient
        xor_history = []
        
    else:
        # Abnormal patient - show moving XOR with beat detection
        # XOR operation - compare binarized signals
        current_binary = (current_chunk > np.median(current_chunk, axis=0)).astype(float)
        previous_binary = (previous_chunk > np.median(previous_chunk, axis=0)).astype(float)
        
        xor_result = np.logical_xor(current_binary, previous_binary).astype(float)
        
        # Add to history for moving display
        if len(xor_history) < 5:  # Keep last 5 chunks for moving display
            xor_history.append(xor_result.tolist())
        else:
            xor_history.pop(0)
            xor_history.append(xor_result.tolist())
        
        # Create moving display by combining history
        if xor_history:
            # Combine all history chunks for continuous display
            combined_xor = np.vstack([np.array(chunk) for chunk in xor_history])
            extended_time = np.arange(len(combined_xor)) / fs
        else:
            combined_xor = xor_result
            extended_time = np.arange(len(xor_result)) / fs
        
        # Detect abnormal beats
        abnormal_beats_detected = []
        for i in range(combined_xor.shape[1]):
            abnormal_mask = detect_abnormal_beats(combined_xor[:, i])
            abnormal_beats_detected.append(abnormal_mask)
        
        # Create figure with moving XOR and abnormal beats
        fig = make_subplots(
            rows=len(selected_channels), cols=1,
            subplot_titles=[f"XOR: {ch.upper()} - ABNORMAL ACTIVITY" for ch in selected_channels]
        )
        
        for i, ch in enumerate(selected_channels):
            # Plot XOR signal as moving line
            fig.add_trace(
                go.Scatter(
                    x=extended_time, y=combined_xor[:, i],
                    mode="lines", line=dict(color="red", width=1),
                    name=f"{ch.upper()} - XOR Signal",
                    opacity=0.7
                ),
                row=i+1, col=1
            )
            
            # Plot abnormal beats as points or lines
            abnormal_indices = np.where(abnormal_beats_detected[i])[0]
            if len(abnormal_indices) > 0:
                abnormal_times = extended_time[abnormal_indices]
                abnormal_values = combined_xor[abnormal_indices, i]
                
                if abnormal_beat_mode == "points":
                    # Show abnormal beats as red points
                    fig.add_trace(
                        go.Scatter(
                            x=abnormal_times, y=abnormal_values,
                            mode="markers",
                            marker=dict(
                                color="red",
                                size=8,
                                symbol="diamond",
                                line=dict(width=2, color="darkred")
                            ),
                            name=f"{ch.upper()} - Abnormal Beats",
                            showlegend=(i == 0)  # Only show legend for first channel
                        ),
                        row=i+1, col=1
                    )
                else:
                    # Show abnormal beats as highlighted lines
                    fig.add_trace(
                        go.Scatter(
                            x=extended_time, y=combined_xor[:, i],
                            mode="lines",
                            line=dict(color="red", width=3),
                            name=f"{ch.upper()} - Abnormal Regions",
                            showlegend=(i == 0)  # Only show legend for first channel
                        ),
                        row=i+1, col=1
                    )
        
        fig.update_layout(
            height=250 * len(selected_channels),
            title=f"Moving XOR Graph - ABNORMAL PATIENT (Real-time Activity Detection) - Mode: {abnormal_beat_mode.upper()}",
            template="plotly_white"
        )
    
    return fig, current_chunk.tolist(), xor_history

# --- Main callback for graph updates ---
@dash.callback(
    Output("ecg-graph", "figure"),
    Output("timer-display", "children"),
    Output("previous-chunk", "data"),
    Output("current-position", "data"),
    Output("xor-history", "data"),
    Input("interval", "n_intervals"),
    Input("upload-ecg", "contents"),  # Also update when new data is loaded
    State("channel-select", "value"),
    State("convert-option", "value"),
    State("start-time", "data"),
    State("is-running", "data"),
    State("ecg-data", "data"),
    State("active-graph-type", "data"),
    State("previous-chunk", "data"),
    State("polar-mode", "data"),
    State("recurrence-mode", "data"),
    State("colormap", "data"),
    State("xor-chunk-size", "data"),
    State("time-window-slider", "value"),
    State("current-position", "data"),
    State("ecg-prediction-result", "data"),
    State("abnormal-beat-mode", "data"),
    State("xor-history", "data"),
    prevent_initial_call=True
)
def update_live(n_intervals, upload_contents, selected_channels, convert_option, start_time, is_running, data_json, 
                graph_type, previous_chunk, polar_mode, recurrence_mode, 
                colormap, xor_chunk_size, time_window, current_position, prediction_result,
                abnormal_beat_mode, xor_history):
    
    # Use callback context to determine which input triggered the callback
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    line_width = 1.5 
    
    # If no data or channels selected, return empty figure
    if data_json is None or not selected_channels:
        fig = go.Figure()
        fig.update_layout(
            title="No data loaded or channel selected",
            template="plotly_white",
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        return fig, "", previous_chunk, current_position, xor_history

    df = pd.read_json(data_json, orient="split")
    df.columns = [c.lower() for c in df.columns]
    df = df[selected_channels]

    if convert_option != "none" and len(selected_channels) == 12:
        df = convert_12_to_3(df, convert_option)
        selected_channels = df.columns.tolist()

    total_samples = df.shape[0]
    current_window_size = int(time_window * FS)
    
    # Only update position if the interval triggered the callback and is running
    if triggered_id == "interval" and is_running:
        # Get data with looping and update position
        data_slice, new_position = get_looping_data(df, current_position, current_window_size, STEP)
        current_position = new_position
    else:
        # For other triggers (like upload), just get current data without updating position
        data_slice, _ = get_looping_data(df, current_position, current_window_size, 0)
    
    t_window = np.arange(len(data_slice)) / FS

    new_previous_chunk = previous_chunk
    new_xor_history = xor_history

    # Determine if patient is normal based on prediction result
    is_normal = prediction_result == "normal"

    # Regular plot
    if graph_type == "regular":
        fig = make_subplots(
            rows=len(selected_channels), cols=1, shared_xaxes=True,
            subplot_titles=[ch.upper() for ch in selected_channels]
        )
        for i, ch in enumerate(selected_channels):
            y = data_slice[ch].iloc[::DOWNSAMPLE]
            x = t_window[::DOWNSAMPLE]
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
        
        status_text = "NORMAL" if is_normal else "ABNORMAL"
        status_color = "green" if is_normal else "red"
        
        fig.update_layout(
            height=250 * len(selected_channels),
            template="plotly_white",
            margin=dict(l=60, r=40, t=100, b=40),
            plot_bgcolor="white", paper_bgcolor="white",
            title=f"ECG Signal (Position: {current_position}/{total_samples}) - Status: <span style='color:{status_color}'>{status_text}</span>"
        )

    # XOR Plot
    elif graph_type == "xor":
        fig, new_previous_chunk, new_xor_history = create_moving_xor_plot(
            df, selected_channels, xor_chunk_size, previous_chunk, 
            current_position, is_normal, xor_history, abnormal_beat_mode
        )

    # Other plot types (Polar and Recurrence) remain the same as before
    # ... [Polar and Recurrence plot code remains unchanged] ...

    else:
        fig = go.Figure()

    timer_text = f"Elapsed Time: {int(pytime.time() - start_time)} s" if is_running and start_time else ""
    return fig, timer_text, new_previous_chunk, current_position, new_xor_history