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
    # --- Controls Row (Aligned) ---
# --- Controls Row (Aligned) ---
# --- Controls Row (Channels + Start + Convert + Predict) ---
# --- Controls Row (Channels + Start/Stop + Convert + Predict) ---
dbc.Row([
    # Channels selector (wider)
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
    ], width=4),

    # Start/Stop button with elapsed time under Start
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
    ], width=1),

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
    ], width=4),

    # Predict button + prediction text
    dbc.Col([
        dbc.Button("Predict ECG", id="predict-btn", color="primary", n_clicks=0, style={"marginRight": "10px"}),
        html.Div(id="ecg-prediction-output", style={
            "fontWeight": "bold", "fontSize": "16px", "display": "inline-block"
        })
    ], width=3)
], align="center", className="mb-4"),


    html.Br(),
    dcc.Graph(id="ecg-graph", style={"height": "900px"}),

 
    

    dcc.Interval(id="interval", interval=1000, n_intervals=0, disabled=True),
    dcc.Store(id="is-running", data=False),
    dcc.Store(id="start-time", data=None),
    dcc.Store(id="ecg-data", data=None),
    dcc.Store(id="active-graph-type", data="regular")
], fluid=True)

# --- Run prediction on button click ---
# --- Run prediction on button click ---
@dash.callback(
    Output("ecg-prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("ecg-data", "data"),
    State("convert-option", "value"),
    State("channel-select", "value"),
    prevent_initial_call=True
)
def predict_ecg(n_clicks, data_json, convert_option, selected_channels):
    if data_json is None or not selected_channels:
        return "⚠️ No ECG data or channels selected."
    
    df = pd.read_json(data_json, orient="split")
    df.columns = [c.lower() for c in df.columns]
    
    # Apply 12→3 channel conversion if selected
    if convert_option != "none" and len(selected_channels) == 12:
        df = convert_12_to_3(df, convert_option)
        selected_channels = df.columns.tolist()
    
    # Keep only selected channels
    df = df[selected_channels]

    try:
        result_text = run_ecg_prediction(df)
    except Exception as e:
        result_text = f"⚠️ Prediction failed: {e}"

    return result_text

# --- CSV Upload callback ---
@dash.callback(
    Output("ecg-data", "data"),
    Output("channel-select", "options"),
    Output("channel-select", "value"),
    Input("upload-ecg", "contents"),
    State("upload-ecg", "filename")
)
def load_csv(contents, filename):
    if contents is None:
        return None, [], []

    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8-sig')))
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None, [], []

    df.columns = [c.lower().strip() for c in df.columns]
    available = [lead for lead in ECG_LEADS if lead in df.columns]
    df = df[available]

    options = [{"label": ch.upper(), "value": ch} for ch in available]
    return df.to_json(date_format="iso", orient="split"), options, available


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
    Output("polar-btn", "style"),
    Output("recurrence-btn", "style"),
    Output("active-graph-type", "data"),
    Input("regular-btn", "n_clicks"),
    Input("polar-btn", "n_clicks"),
    Input("recurrence-btn", "n_clicks"),
    State("active-graph-type", "data"),
    prevent_initial_call=True
)
def set_active_button(n_reg, n_pol, n_rec, current_type):
    ctx = dash.callback_context
    if not ctx.triggered:
        return ACTIVE_STYLE, INACTIVE_STYLE, INACTIVE_STYLE, current_type

    clicked = ctx.triggered[0]["prop_id"].split(".")[0]

    if clicked == "regular-btn":
        return ACTIVE_STYLE, INACTIVE_STYLE, INACTIVE_STYLE, "regular"
    elif clicked == "polar-btn":
        return INACTIVE_STYLE, ACTIVE_STYLE, INACTIVE_STYLE, "polar"
    elif clicked == "recurrence-btn":
        return INACTIVE_STYLE, INACTIVE_STYLE, ACTIVE_STYLE, "recurrence"
    return ACTIVE_STYLE, INACTIVE_STYLE, INACTIVE_STYLE, current_type


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
        return " Normal (no significant abnormality detected)"
    else:
        diseases = [ab for ab, _ in detected]
        return f" Abnormal: Detected conditions → {', '.join(diseases)}"


# --- Update Prediction on CSV upload ---

def update_prediction(data_json):
    if data_json is None:
        return ""
    df = pd.read_json(data_json, orient="split")
    df.columns = [c.lower() for c in df.columns]
    result_text = run_ecg_prediction(df)
    return result_text


# --- Update Graph + Timer ---
@dash.callback(
    Output("ecg-graph", "figure"),
    Output("timer-display", "children"),
    Input("interval", "n_intervals"),
    State("channel-select", "value"),
    State("convert-option", "value"),
    State("start-time", "data"),
    State("is-running", "data"),
    State("ecg-data", "data"),
    State("active-graph-type", "data")
)
def update_live(n, selected_channels, convert_option, start_time, is_running, data_json, graph_type):
    line_width = 1.5 
    if data_json is None or not selected_channels:
        fig = go.Figure()
        fig.update_layout(
            title=" No data loaded or channel selected",
            template="plotly_white",
            plot_bgcolor="white",
            paper_bgcolor="white"
        )
        return fig, ""

    df = pd.read_json(data_json, orient="split")
    df.columns = [c.lower() for c in df.columns]
    df = df[selected_channels]

    if convert_option != "none" and len(selected_channels) == 12:
        df = convert_12_to_3(df, convert_option)
        selected_channels = df.columns.tolist()

    total_samples = df.shape[0]
    end_idx = min(n * STEP, total_samples)
    start_idx = max(0, end_idx - WINDOW_SIZE)
    t_window = np.arange(start_idx, end_idx) / FS

    # Regular plot
    if graph_type == "regular":
        fig = make_subplots(
            rows=len(selected_channels), cols=1, shared_xaxes=True,
            subplot_titles=[ch.upper() for ch in selected_channels]
        )
        for i, ch in enumerate(selected_channels):
            y = df[ch].iloc[start_idx:end_idx:DOWNSAMPLE]
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
        fig.update_layout(
            height=250 * len(selected_channels),
            template="plotly_white",
            margin=dict(l=60, r=40, t=100, b=40),
            plot_bgcolor="white", paper_bgcolor="white"
        )
    elif graph_type == "polar":
            rows = len(selected_channels)
            specs = [[{"type": "polar"}] for _ in range(rows)]
            fig = make_subplots(rows=rows, cols=1,
                                subplot_titles=[f"{ch.upper()}<br><br>" for ch in selected_channels],
                                specs=specs)
            for ann in fig['layout']['annotations']:
                ann['y'] += 0.01   # move titles higher (increase for more space)

            for i, ch in enumerate(selected_channels):
                y = df[ch].iloc[start_idx:end_idx:DOWNSAMPLE].to_numpy()
                theta = np.linspace(0, 360, len(y), endpoint=False)
                fig.add_trace(
                    go.Scatterpolar(r=y, theta=theta, mode="lines",
                                    line=dict(color="blue", width=line_width),
                                    name=ch.upper(), showlegend=False),
                    row=i+1, col=1
                )

            fig.update_layout(
                height=400 * len(selected_channels),
                margin=dict(l=60, r=40, t=100, b=40),
                polar=dict(
                    angularaxis=dict(rotation=0, direction="counterclockwise", tickfont=dict(size=9)),
                    radialaxis=dict(angle=0, tickfont=dict(size=9))
                ),
                title=dict(
                y=0.98 # shift all subplot titles down a bit
            )
            )

    elif graph_type == "recurrence":
        # --- Blue Heatmap Recurrence Plots ---
        if len(selected_channels) == 1:
            ch = selected_channels[0]
            s = df[ch].iloc[start_idx:end_idx:DOWNSAMPLE].to_numpy()
            dist = np.abs(np.subtract.outer(s, s))

            fig = go.Figure(data=go.Heatmap(z=dist, colorscale="Blues"))
            fig.update_layout(
                title=f"Recurrence Plot: {ch.upper()}",
                xaxis_title="Time Index",
                yaxis_title="Time Index",
                height=500,
                template="plotly_white"
            )

        elif len(selected_channels) == 2:
            ch1, ch2 = selected_channels
            s1 = df[ch1].iloc[start_idx:end_idx:DOWNSAMPLE].to_numpy()
            s2 = df[ch2].iloc[start_idx:end_idx:DOWNSAMPLE].to_numpy()
            dist = np.abs(np.subtract.outer(s1, s2))

            fig = go.Figure(data=go.Heatmap(z=dist, colorscale="Blues"))
            fig.update_layout(
                title=f"Cross-Recurrence: {ch1.upper()} vs {ch2.upper()}",
                xaxis_title=f"{ch1.upper()} Index",
                yaxis_title=f"{ch2.upper()} Index",
                height=500,
                template="plotly_white"
            )

        elif len(selected_channels) == 4:
            groupA = selected_channels[:2]
            groupB = selected_channels[2:4]
            sA = df[groupA].iloc[start_idx:end_idx:DOWNSAMPLE].mean(axis=1).to_numpy()
            sB = df[groupB].iloc[start_idx:end_idx:DOWNSAMPLE].mean(axis=1).to_numpy()
            dist = np.abs(np.subtract.outer(sA, sB))

            fig = go.Figure(data=go.Heatmap(z=dist, colorscale="Blues"))
            fig.update_layout(
                title=f"Group Recurrence: {', '.join([g.upper() for g in groupA])} vs {', '.join([g.upper() for g in groupB])}",
                xaxis_title="Group A Index",
                yaxis_title="Group B Index",
                height=500,
                template="plotly_white"
            )

        elif len(selected_channels) == 6:
            groupA = selected_channels[:3]
            groupB = selected_channels[3:6]
            sA = df[groupA].iloc[start_idx:end_idx:DOWNSAMPLE].mean(axis=1).to_numpy()
            sB = df[groupB].iloc[start_idx:end_idx:DOWNSAMPLE].mean(axis=1).to_numpy()
            dist = np.abs(np.subtract.outer(sA, sB))

            fig = go.Figure(data=go.Heatmap(z=dist, colorscale="Blues"))
            fig.update_layout(
                title=f"Group Recurrence (3 vs 3): {', '.join([g.upper() for g in groupA])} vs {', '.join([g.upper() for g in groupB])}",
                xaxis_title="Group A Index",
                yaxis_title="Group B Index",
                height=500,
                template="plotly_white"
            )

        elif len(selected_channels) == 12:
            groupA = selected_channels[:6]
            groupB = selected_channels[6:12]
            sA = df[groupA].iloc[start_idx:end_idx:DOWNSAMPLE].mean(axis=1).to_numpy()
            sB = df[groupB].iloc[start_idx:end_idx:DOWNSAMPLE].mean(axis=1).to_numpy()
            dist = np.abs(np.subtract.outer(sA, sB))

            fig = go.Figure(data=go.Heatmap(z=dist, colorscale="Blues"))
            fig.update_layout(
                title=f"Group Recurrence (6 vs 6): {', '.join([g.upper() for g in groupA])} vs {', '.join([g.upper() for g in groupB])}",
                xaxis_title="Group A Index",
                yaxis_title="Group B Index",
                height=500,
                template="plotly_white"
            )

        else:
            fig = go.Figure()
            fig.update_layout(
                title="⚠️ Please select 1, 2, 4 (2 vs 2), 6 (3 vs 3), or 12 (6 vs 6) channels",
                template="plotly_white",
                height=300
            )

    else:
        fig = go.Figure()

    timer_text = f"Elapsed Time: {int(pytime.time() - start_time)} s" if is_running and start_time else ""
    return fig, timer_text
