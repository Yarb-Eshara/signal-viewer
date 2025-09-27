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

dash.register_page(__name__, path="/ecg", name="ECG")

# --- Parameters ---
WINDOW_SIZE = 2500   # ~10s if fs=250Hz
STEP = 250           # shift by 1s
FS = 250             # sampling frequency
DOWNSAMPLE = 4       # plot every 4th point (lighter)
ECG_LEADS = ["i", "ii", "iii", "avr", "avl", "avf",
             "v1", "v2", "v3", "v4", "v5", "v6"]

# --- Colors ---
ACTIVE_STYLE = {"backgroundColor": "#2c7f91", "borderColor": "#2c7f91", "color": "white", "marginRight": "10px"}
INACTIVE_STYLE = {"backgroundColor": "#3E9AAB", "borderColor": "#3E9AAB", "color": "white", "marginRight": "10px"}

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

    # --- Controls ---
    dbc.Row([
        dbc.Col([
            html.Label("Select Channels"),
            dcc.Dropdown(
                id="channel-select",
                options=[{"label": ch.upper(), "value": ch} for ch in ECG_LEADS],
                value=ECG_LEADS,
                multi=True,
                clearable=False,
                style={"color": "#182940"}
            ),
        ], width=10),
        dbc.Col([
            dbc.Button(
                "▶ Start",
                id="start-stop-btn",
                size="md",
                style={
                    "backgroundColor": "#3E9AAB",
                    "borderColor": "#3E9AAB",
                    "color": "white",
                    "fontSize": "14px",
                    "height": "38px",      # align with dropdown
                    "padding": "0 15px",
                    "marginTop": "23px"    # slightly lower to align vertically
                }
            ),
            html.Div(id="timer-display", className="mt-2", style={"fontWeight": "bold"})
        ], width=2),
    ]),

    html.Br(),
    dcc.Graph(id="ecg-graph", style={"height": "900px"}),

    dcc.Interval(id="interval", interval=1000, n_intervals=0, disabled=True),
    dcc.Store(id="is-running", data=False),
    dcc.Store(id="start-time", data=None),
    dcc.Store(id="ecg-data", data=None),
    dcc.Store(id="active-graph-type", data="regular")
], fluid=True)


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
        style = {
            "backgroundColor": "#d9534f",
            "borderColor": "#d9534f",
            "color": "white",
            "height": "38px",
            "padding": "0 15px",
            "marginTop": "23px"
        }  # red Stop aligned
        return True, False, "⏸ Stop", style, pytime.time()
    else:
        style = {
            "backgroundColor": "#3E9AAB",
            "borderColor": "#3E9AAB",
            "color": "white",
            "height": "38px",
            "padding": "0 15px",
            "marginTop": "23px"
        }  # teal Start aligned
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


# --- Update Graph + Timer ---
@dash.callback(
    Output("ecg-graph", "figure"),
    Output("timer-display", "children"),
    Input("interval", "n_intervals"),
    State("channel-select", "value"),
    State("start-time", "data"),
    State("is-running", "data"),
    State("ecg-data", "data"),
    State("active-graph-type", "data")
)
def update_live(n, selected_channels, start_time, is_running, data_json, graph_type):
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

    total_samples = df.shape[0]
    end_idx = min(n * STEP, total_samples)
    start_idx = max(0, end_idx - WINDOW_SIZE)
    t_window = np.arange(start_idx, end_idx) / FS

    if graph_type == "regular":
        fig = make_subplots(
            rows=len(selected_channels), cols=1, shared_xaxes=True,
            subplot_titles=[ch.upper() for ch in selected_channels]
        )
        for i, ch in enumerate(selected_channels):
            y = df[ch].iloc[start_idx:end_idx:DOWNSAMPLE]
            x = t_window[::DOWNSAMPLE]
            fig.add_trace(
                go.Scatter(x=x, y=y, mode="lines", line=dict(color="blue", width=1), name=ch.upper(), showlegend=False),
                row=i+1, col=1
            )
        fig.update_layout(height=250 * len(selected_channels), template="plotly_white",
                          margin=dict(l=30, r=30, t=30, b=30),
                          plot_bgcolor="white", paper_bgcolor="white")

    elif graph_type == "polar":
        fig = go.Figure()
        for ch in selected_channels:
            y = df[ch].iloc[start_idx:end_idx:DOWNSAMPLE]
            theta = np.linspace(0, 360, len(y))
            fig.add_trace(go.Scatterpolar(r=y, theta=theta, mode="lines", name=ch.upper()))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)

    elif graph_type == "recurrence":
        segment = df[selected_channels[0]].iloc[start_idx:end_idx:DOWNSAMPLE]
        dist = np.abs(np.subtract.outer(segment, segment))
        fig = go.Figure(data=go.Heatmap(z=dist))
        fig.update_layout(title="Recurrence Plot", template="plotly_white")
    else:
        fig = go.Figure()

    timer_text = f"Elapsed Time: {int(pytime.time() - start_time)} s" if is_running and start_time else ""
    return fig, timer_text
