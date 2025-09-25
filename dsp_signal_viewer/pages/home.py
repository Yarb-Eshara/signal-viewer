import dash
from dash import html
import dash_bootstrap_components as dbc


dash.register_page(__name__, path="/", name="Home")


layout = dbc.Container([
    html.H1("Welcome to Signal Viewer"),
    html.P("Use the navigation bar above to select ECG, EEG, or Sound pages.")
], style={"textAlign": "center", "marginTop": "50px", "color": "white"})