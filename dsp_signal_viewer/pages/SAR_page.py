import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

# Register the page
dash.register_page(__name__, path="/sar", name="Sound Machines")

# Define the layout
layout = dbc.Container([
    # Use className for basic spacing/text formatting with Bootstrap classes
    html.H2("jaime lannister", className="mb-4"),
    html.Hr(),

    # Upload Component - Inline styling cleanup and using a tuple for 'children'
    dcc.Upload(
        id="upload-sound-machine",
        # Use a list/tuple for children for cleaner structure
        children=[
            html.Div("Upload a sound file (wav, mp3)")
        ],
        # Simplify style properties (e.g., use percentages for width/height)
        style={
            "width": "100%",
            "height": "60px",
            "lineHeight": "60px",
            "border": "1px dashed #ccc",  # Shorthand for border properties
            "borderRadius": "5px",
            "textAlign": "center"
        },
        # Consider using a className for styling if the style is reused
        # className="dcc-upload-style"
    ),

    # Output Div
    html.Div(id="sound-machine-output"),
],
    # Add a bit of top margin to the container itself
    className="py-4")