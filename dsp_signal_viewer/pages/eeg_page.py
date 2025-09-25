import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/eeg", name="EEG")

layout = dbc.Container([
    html.H2("EEG Signal Viewer"),
    html.Hr(),
    dcc.Dropdown(
        id="eeg-mode",
        options=[
            {"label": "Single Channel", "value": "single"},
            {"label": "Multi Channel", "value": "multi"},
            {"label": "Reduced 3 Channels (PCA)", "value": "pca"}
        ],
        value="multi",
        clearable=False
    ),
    html.Div(id="eeg-output"),
])
