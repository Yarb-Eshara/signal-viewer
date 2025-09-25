import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/sound-doppler", name="Sound Doppler")

layout = dbc.Container([
    html.H2("Doppler Effect Simulator"),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.Label("Base Frequency (Hz)"),
            dcc.Slider(20, 2000, 10, value=440, id="doppler-freq"),
            html.Div(id="doppler-freq-output"),
        ], width=6),
        dbc.Col([
            html.Label("Velocity (m/s)"),
            dcc.Slider(-50, 50, 1, value=0, id="doppler-vel"),
            html.Div(id="doppler-vel-output"),
        ], width=6),
    ]),
    html.Br(),
    html.Button("Generate Sound", id="doppler-generate", n_clicks=0, className="btn btn-primary"),
    html.Div(id="doppler-output"),
])
