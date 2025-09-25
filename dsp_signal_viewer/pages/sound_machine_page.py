import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/sound-machines", name="Sound Machines")

layout = dbc.Container([
    html.H2("Machine Sound Classification"),
    html.Hr(),
    dcc.Upload(
        id="upload-sound-machine",
        children=html.Div(["Upload a sound file (wav, mp3)"]),
        style={
            "width": "100%", "height": "60px", "lineHeight": "60px",
            "borderWidth": "1px", "borderStyle": "dashed",
            "borderRadius": "5px", "textAlign": "center"
        }
    ),
    html.Div(id="sound-machine-output"),
])
