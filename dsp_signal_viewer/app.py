import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    use_pages=True,
    pages_folder="./pages"
)

# ECG Dropdown Component
ecg_dropdown = dbc.DropdownMenu(
    children=[
        dbc.DropdownMenuItem("ECG Viewer", href="/ecg"),
        dbc.DropdownMenuItem("ECG Down Sampling", href="/ecg-downsampling"),
    ],
    nav=True,
    in_navbar=True,
    label="ECG",
    style={"marginRight": "10px"}
)

navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("Signal Viewer", href="/"),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink("Home", href="/")),
            dbc.NavItem(ecg_dropdown),  # Replace the simple ECG link with dropdown
            dbc.NavItem(dbc.NavLink("EEG", href="/eeg")),
            dbc.NavItem(dbc.NavLink("Sound Doppler", href="/sound-doppler")),
            dbc.NavItem(dbc.NavLink("Radar", href="/radar")),
            dbc.NavItem(dbc.NavLink("Drone", href="/drone")),
            dbc.NavItem(dbc.NavLink("SAR", href="/sar")),
        ], className="ms-auto", navbar=True),
    ]),
    dark=True,
    color="#3E9AAB"
)

# Layout - ADD SPACING HERE
app.layout = html.Div(
    [
        navbar,
        html.Div(
            dash.page_container,
            className="mt-4"  # This adds vertical space
        )
    ],
    style={
        "backgroundColor": "#182940",
        "minHeight": "100vh",
        "color": "white"
    }
)

if __name__ == "__main__":
   # app.run_server(debug=True)
    app.run(debug=True)