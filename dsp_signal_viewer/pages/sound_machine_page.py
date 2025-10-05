import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import base64
import tempfile
import torch
import librosa
import numpy as np
from transformers import AutoModelForAudioClassification, AutoProcessor
import soundfile as sf
import plotly.graph_objects as go
import os

# =====================
# Constants & Styles
# =====================
CARD_STYLE = {
    'backgroundColor': '#1e2130',
    'border': '1px solid #2d3748',
    'borderRadius': '8px',
    'margin': '10px 0'
}

PLOT_CONFIG = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d', 'autoScale2d']
}

# Dark mode default empty figure
def empty_dark_fig():
    fig = go.Figure()
    fig.update_layout(
        plot_bgcolor='#1e2130',
        paper_bgcolor='#1e2130',
        font=dict(color='white'),
        xaxis=dict(showgrid=True, gridcolor='#2d3748'),
        yaxis=dict(showgrid=True, gridcolor='#2d3748')
    )
    return fig

# =====================
# Load Model
# =====================
model_name = "preszzz/drone-audio-detection-05-12"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name)
model.eval()
print("Model loaded successfully!")

# =====================
# Helper Functions
# =====================


def pad_or_repeat_audio(file_path, target_duration_sec=3, padded_folder=r"assets/padded_sounds"):
    # Make sure the padded folder exists
    os.makedirs(padded_folder, exist_ok=True)

    # Create padded file path in the padded folder
    base_name = os.path.basename(file_path).replace(".wav", "-padded.wav")
    padded_file_path = os.path.join(padded_folder, base_name)

    # If already exists, just return it
    if os.path.exists(padded_file_path):
        return padded_file_path

    # Read original audio
    data, sr = sf.read(file_path)
    desired_length = sr * target_duration_sec

    # Repeat or pad if too short
    if len(data) < desired_length:
        repeats = int(np.ceil(desired_length / len(data)))
        data = np.tile(data, repeats)[:desired_length]

    # Write padded file once
    sf.write(padded_file_path, data, sr)
    return padded_file_path



def create_waveform_plot(file_path):
    data, sr = librosa.load(file_path, sr=None)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Waveform', line=dict(color='#00D2FF')))
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=10),
        xaxis_title='Sample',
        yaxis_title='Amplitude',
        plot_bgcolor='#1e2130',
        paper_bgcolor='#1e2130',
        font=dict(color='white')
    )
    return fig

# =====================
# Predefined sounds
# =====================
SOUNDS_FOLDER = r'dsp_signal_viewer\assets\sounds'
#SOUNDS_FOLDER = r'assets/sounds'  # your 4 original sounds
sound_files = [f for f in os.listdir(SOUNDS_FOLDER) if f.endswith(".wav")]

# Only use original filenames in dropdown
sound_options = [{"label": f, "value": os.path.join(SOUNDS_FOLDER, f)} for f in sound_files]

# =====================
# Layout
# =====================
dash.register_page(__name__, path="/drone", name="Drone Detection")

layout = dbc.Container([
    # Header with GIF on the right
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Span(
                    "Drone Sound Detection",
                    style={"color": "white", "fontSize": "24px", "alignSelf": "center"}
                ),
                html.Img(
                    src="/assets/Drone.gif",
                    style={"height": "80px", "marginLeft": "10px", "marginTop": "10px"}
                )
            ], style={"display": "flex", "alignItems": "center"})
        ])
    ], className="mb-4"),

    # Upload & Controls + Dropdown
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("🔊 Upload or Select Sound", style={"color": "white"}),

                    # Upload
                    dcc.Upload(
                        id="upload-sound-machine",
                        children=html.Div(["📂 Drag or click to upload wav/mp3"]),
                        style={
                            "width": "100%", "height": "60px", "lineHeight": "60px",
                            "borderWidth": "1px", "borderStyle": "dashed",
                            "borderRadius": "5px", "textAlign": "center",
                            "color": "white",
                            "backgroundColor": "#1e2130"
                        }
                    ),
                    html.Br(),

                    # Dropdown (dark-themed)
                    dcc.Dropdown(
                        id="dropdown-sounds",
                        options=sound_options,
                        placeholder="Or select a predefined sound",
                        style={
                            "backgroundColor": "#1e2130",
                            "color": "white",
                            "border": "1px solid #2d3748",
                            "borderRadius": "5px",
                            "padding": "5px"
                        },
                        className="dark-dropdown"
                    ),
                    html.Br(),

                    # Audio player
                    html.Div(
                        html.Audio(id="audio-player", controls=True, style={
                            "width": "100%",
                            "backgroundColor": "#1e2130",
                            "color": "black"
                        }),
                        style={"backgroundColor": "#1e2130", "padding": "10px", "borderRadius": "5px"}
                    ),
                    html.Br(),

                    # Remove button
                    dbc.Button("Remove", id="remove-button", color="danger", className="w-100", n_clicks=0),
                    html.Br(), html.Br(),

                    # Predict button
                    dbc.Button("Predict", id="predict-button", color="primary", disabled=True, className="w-100"),
                    html.Br(), html.Br(),

                    html.Div(id="sound-machine-output")
                ])
            ], style=CARD_STYLE)
        ], width=4),

        # Visualization Cards
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H6("📈 Waveform", style={"color": "white"}),
                    dcc.Graph(id="waveform-plot", figure=empty_dark_fig(), config=PLOT_CONFIG, style={"height": "350px"})
                ])
            ], style={**CARD_STYLE, "height": "400px"}),

            dbc.Card([
                dbc.CardBody([
                    html.H6("🎯 Prediction", style={"color": "white"}),
                    dcc.Graph(id="prediction-plot", figure=empty_dark_fig(), config=PLOT_CONFIG, style={"height": "350px"})
                ])
            ], style={**CARD_STYLE, "height": "400px"})
        ], width=8)
    ]),

    # Hidden Store
    dcc.Store(id="uploaded-file-path", data=None)
], fluid=True, style={'backgroundColor': '#0f1419', 'minHeight': '100vh', 'padding': '20px'})

# =====================
# Callbacks
# =====================
@dash.callback(
    Output("audio-player", "src"),
    Output("waveform-plot", "figure"),
    Output("predict-button", "disabled"),
    Output("uploaded-file-path", "data"),
    Input("upload-sound-machine", "contents"),
    Input("dropdown-sounds", "value"),
    Input("remove-button", "n_clicks"),
    State("upload-sound-machine", "filename"),
    prevent_initial_call=True
)
def update_audio(contents, dropdown_value, remove_clicks, filename):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # Remove button clicked
    if triggered_id == "remove-button":
        return None, empty_dark_fig(), True, None

    file_path = None
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(decoded)
            file_path = tmp_file.name
    elif dropdown_value is not None:
        file_path = dropdown_value

    if file_path is None:
        return None, empty_dark_fig(), True, None

    audio_src = f"data:audio/wav;base64,{base64.b64encode(open(file_path, 'rb').read()).decode()}"
    fig = create_waveform_plot(file_path)
    return audio_src, fig, False, file_path

@dash.callback(
    Output("sound-machine-output", "children"),
    Output("prediction-plot", "figure"),
    Input("predict-button", "n_clicks"),
    State("uploaded-file-path", "data"),
    prevent_initial_call=True
)
def handle_predict(n_clicks, file_path):
    if not file_path:
        return dbc.Alert("No file uploaded or selected.", color="warning"), empty_dark_fig()

    try:
        file_path = pad_or_repeat_audio(file_path, target_duration_sec=3)
        waveform, sr = librosa.load(file_path, sr=16000, mono=True)
        waveform = torch.tensor(waveform).unsqueeze(0)
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).squeeze().numpy()
        labels = [model.config.id2label[i] for i in range(len(probs))]
        top_indices = probs.argsort()[-3:][::-1]
        top_probs = probs[top_indices]
        top_labels = [labels[i] for i in top_indices]

        predicted_label = top_labels[0]
        confidence = top_probs[0]
        color = "green" if predicted_label.lower() == "drone" else "red"
        alert = dbc.Alert(f"Prediction: {predicted_label} (Confidence: {confidence:.6f})", color=color)

        # Bar chart with half-width bars
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_labels,
            y=top_probs,
            width=[0.3]*3,
            marker_color=['green' if i==0 else 'lightblue' for i in range(3)]
        ))
        fig.update_layout(
            title="",
            yaxis=dict(title="Probability", range=[0,1]),
            xaxis=dict(title=""),
            plot_bgcolor='#1e2130',
            paper_bgcolor='#1e2130',
            font=dict(color='white')
        )

        return alert, fig
    except Exception as e:
        return dbc.Alert(f"Error processing file: {str(e)}", color="danger"), empty_dark_fig()