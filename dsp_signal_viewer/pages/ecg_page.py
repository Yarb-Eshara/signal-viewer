import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/ecg", name="ECG")

layout = dbc.Container([
    html.H2("ECG Signal Viewer"),
    html.Hr(),
    dcc.Upload(
        id="upload-ecg",
        children=html.Div(["Drag and Drop or Select ECG File"]),
        style={
            "width": "100%", "height": "60px", "lineHeight": "60px",
            "borderWidth": "1px", "borderStyle": "dashed",
            "borderRadius": "5px", "textAlign": "center"
        }
    ),
    html.Div(id="ecg-output"),
])
