import dash
import dash_bootstrap_components as dbc
from dash import html, dcc

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    use_pages=True,
    pages_folder="./pages"
)

navbar = dbc.Navbar(
    dbc.Container([
        dbc.NavbarBrand("Signal Viewer", href="/"),
        dbc.Nav([
            dbc.NavItem(dbc.NavLink("Home", href="/")),
            dbc.NavItem(dbc.NavLink("ECG", href="/ecg")),
            dbc.NavItem(dbc.NavLink("EEG", href="/eeg")),
            dbc.NavItem(dbc.NavLink("Sound Doppler", href="/sound-doppler")),
            dbc.NavItem(dbc.NavLink("Sound Machines", href="/sound-machines")),
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
    app.run_server(debug=True)