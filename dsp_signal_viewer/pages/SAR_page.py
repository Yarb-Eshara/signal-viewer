import dash
from dash import html, dcc, Input, Output, State
import os
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import rasterio
from skimage.transform import downscale_local_mean
import dash_bootstrap_components as dbc

# =====================
# Styles
# =====================
CARD_STYLE = {
    'backgroundColor': '#1e2130',
    'border': '1px solid #2d3748',
    'borderRadius': '8px',
    'margin': '10px 0',
    'padding': '15px'
}

# =====================
# Register page
# =====================
dash.register_page(__name__, path="/sar", name="Level-1 GRD SAR Viewer")

# =====================
# Layout
# =====================
layout = dbc.Container([
    # Header with GIF
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Span(
                    "Level-1 GRD SAR Viewer",
                    style={"color": "white", "fontSize": "24px", "alignSelf": "center"}
                ),
                html.Img(
                    src="/assets/SAR.gif",  # your GIF file here
                    style={"height": "80px", "marginLeft": "10px", "marginTop": "0px"}
                )
            ], style={"display": "flex", "alignItems": "center"})
        ])
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("📂 Load SAR Data", style={"color": "white"}),
                    dcc.Input(
                        id='safe-folder-path',
                        type='text',
                        placeholder='Enter full path to .SAFE folder',
                        style={
                            'width': '100%',
                            'marginTop': '10px',
                            'marginBottom': '10px',
                            'backgroundColor': '#182940',
                            'color': 'white',
                            'border': '1px solid #2d3748',
                            'borderRadius': '5px',
                            'padding': '8px'
                        }
                    ),
                    html.Button('Load SAR Data', id='load-button', n_clicks=0,
                                style={'width': '100%', 'backgroundColor': '#00D2FF', 'color': 'black',
                                       'border': 'none', 'borderRadius': '5px'}),
                ])
            ], style=CARD_STYLE)
        ], width=4),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("🛰️ SAR Image", style={"color": "white"}),
                    html.Div(id='output-image', style={'marginTop': '10px'})
                ])
            ], style={**CARD_STYLE, "height": "auto"})
        ], width=8)
    ])
], fluid=True, style={'backgroundColor': '#182940', 'minHeight': '100vh', 'padding': '20px'})

# =====================
# Callback
# =====================
@dash.callback(
    Output('output-image', 'children'),
    Input('load-button', 'n_clicks'),
    State('safe-folder-path', 'value')
)
def display_grd(n_clicks, folder_path):
    if n_clicks == 0 or not folder_path:
        return dbc.Alert("Enter a folder path and click 'Load SAR Data'.", color="black")

    try:
        if not os.path.exists(folder_path):
            return dbc.Alert(f"Folder not found: {folder_path}", color="warning")

        # Find the first .tiff inside the folder recursively
        tiff_file = None
        for root, dirs, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith((".tiff", ".tif")):
                    tiff_file = os.path.join(root, f)
                    break
            if tiff_file:
                break

        if not tiff_file:
            return dbc.Alert("No .tiff measurement file found in folder.", color="warning")

        # Open with rasterio
        with rasterio.open(tiff_file) as src:
            image_data = src.read(1).astype(np.float32)  # first band

        # Downsample if too large
        max_dim = 2048
        if image_data.shape[0] > max_dim or image_data.shape[1] > max_dim:
            factor_row = max(1, image_data.shape[0] // max_dim)
            factor_col = max(1, image_data.shape[1] // max_dim)
            image_data = downscale_local_mean(image_data, (factor_row, factor_col))

        # Convert to dB (GRD = detected intensity)
        amplitude_db = 10 * np.log10(image_data + 1e-6)

        # Rotate 180 degrees
        amplitude_db_rotated = np.rot90(amplitude_db, 2)

        # Compute figure size based on image aspect ratio
        height, width = amplitude_db_rotated.shape
        figsize = (width / 200, height / 200)  # scale factor 200

        # Plot with terrain colormap
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(amplitude_db_rotated, cmap='terrain', aspect='auto')
        ax.axis('off')  # remove axes

        # Save figure to PNG buffer with no padding
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        img_b64 = base64.b64encode(buf.getbuffer()).decode("utf8")

        return html.Img(
            src=f"data:image/png;base64,{img_b64}",
            style={'width': '100%', 'height': 'auto', 'display': 'block', 'borderRadius': '5px'}
        )

    except Exception as e:
        return dbc.Alert(f"Error processing GRD .SAFE folder: {e}", color="danger")