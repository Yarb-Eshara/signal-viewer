import dash
from dash import html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc
import numpy as np
from scipy.interpolate import interp1d
import plotly.graph_objects as go
import soundfile as sf
from io import BytesIO
import base64

dash.register_page(__name__, path="/sound-doppler", name="Sound Doppler")

layout = dbc.Container([
    html.H2("Doppler Effect Simulator"),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.Label("Source Velocity (m/s)"),
            dcc.Slider(id="vs-slider", min=0, max=50, step=1, value=20,
                       marks={i: str(i) for i in range(0, 51, 10)}),
        ], width=4),
        dbc.Col([
            html.Label("Closest Distance (m)"),
            dcc.Slider(id="d-slider", min=0.1, max=20, step=0.1, value=5,
                       marks={i: str(i) for i in [0.1, 5, 10, 15, 20]}),
        ], width=4),
        dbc.Col([
            html.Label("Duration (s)"),
            dcc.Slider(id="dur-slider", min=5, max=30, step=1, value=10,
                       marks={i: str(i) for i in range(5, 31, 5)}),
        ], width=4),
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.Label("Amplitude Model"),
            dcc.Dropdown(
                id="amp-model",
                options=[
                    {'label': 'Inverse Distance (1/r)', 'value': 'inverse'},
                    {'label': 'Gaussian', 'value': 'gaussian'},
                ],
                value='inverse',
                clearable=False
            ),
        ], width=4),
    ]),
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.Label("Base Car Frequency (Hz)"),
            dcc.Input(id="car-freq", type="number", value=250, min=50, max=2000, step=10,
                      style={"width": "100%"}),
        ], width=4)
    ]),
    html.Br(),
    dcc.Loading(
        id="loading",
        type="default",
        children=html.Div(id="doppler-output")
    ),
])

# ---------------------------
# Doppler schematic diagram
# ---------------------------
def doppler_diagram(vs, d, duration):
    x_source = np.linspace(-duration*vs/2, duration*vs/2, 5)
    y_source = np.zeros_like(x_source)
    x_obs, y_obs = 0, d

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_source, y=y_source, mode="lines",
                             line=dict(color="blue", dash="dash"), name="Source Path"))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text",
                             marker=dict(size=12, color="blue"), text=["Source"],
                             textposition="top center", name="Source"))
    fig.add_trace(go.Scatter(x=[x_obs], y=[y_obs], mode="markers+text",
                             marker=dict(size=12, color="red"), text=["Observer"],
                             textposition="bottom center", name="Observer"))
    c = 343
    times = [0.5, 1.0, 1.5]
    for t in times:
        x_emit = -vs * t
        r = c * t
        theta = np.linspace(0, 2*np.pi, 200)
        x_circ = x_emit + r * np.cos(theta)
        y_circ = r * np.sin(theta)
        fig.add_trace(go.Scatter(x=x_circ, y=y_circ, mode="lines",
                                 line=dict(color="gray", dash="dot", width=1),
                                 showlegend=False))
    fig.update_layout(
        title="Doppler Effect Geometry (Schematic)",
        xaxis_title="x (m)",
        yaxis_title="y (m)",
        xaxis=dict(scaleanchor="y", scaleratio=1),
        template="simple_white",
        height=500
    )
    return fig

# ---------------------------
# Main callback
# ---------------------------
@callback(
    Output("doppler-output", "children"),
    Input("vs-slider", "value"),
    Input("d-slider", "value"),
    Input("dur-slider", "value"),
    Input("amp-model", "value"),
    Input("car-freq", "value")
)
def update_doppler(vs, d, duration, amp_model, car_freq):
    if vs == 0:
        return html.Div("No motion, no Doppler effect.")

    v = 343.0  # speed of sound
    sr = 22050
    t_emit_min = -3 * duration - 30
    t_emit_max = 3 * duration + 30

    # ---- Synthesize sound using user input frequency ----
    num_samples_emit = int(sr * (t_emit_max - t_emit_min)) + 1
    t_emit_grid = np.linspace(t_emit_min, t_emit_max, num_samples_emit)
    f_mean = car_freq or 250
    dt = 1.0 / sr
    phase = np.cumsum(2 * np.pi * f_mean * dt * np.ones_like(t_emit_grid))
    s_emit = np.sin(phase) + 0.05 * np.random.normal(0, 1, len(phase))
    s_emit /= np.max(np.abs(s_emit))

    interp_s = interp1d(t_emit_grid, s_emit, kind='linear', fill_value=0, bounds_error=False)
    interp_f_emit = interp1d(t_emit_grid, f_mean*np.ones_like(t_emit_grid),
                             kind='linear', fill_value=f_mean, bounds_error=False)

    num_samples_rec = int(sr * duration) + 1
    t_rec = np.linspace(0, duration, num_samples_rec)
    t_emit = np.zeros_like(t_rec)
    approx_factor = 1 / (1 - vs / v)
    te_guess = -2 * duration * approx_factor

    for i, tr in enumerate(t_rec):
        def f(te):
            r = np.sqrt((vs * te)**2 + d**2)
            return te + r / v - tr + duration/2
        def df(te):
            r = np.sqrt((vs * te)**2 + d**2)
            return 1 + (vs**2 * te) / (v * r)
        te = te_guess
        for _ in range(100):
            delta = f(te) / df(te)
            te -= delta
            if abs(delta) < 1e-10:
                break
        t_emit[i] = te
        te_guess = te

    r = np.sqrt((vs * t_emit)**2 + d**2)
    if amp_model == 'inverse':
        amp = 1 / (r + 0.1)
    else:
        sigma = duration / 4
        amp = np.exp(-((t_rec - duration/2)**2) / (2*sigma**2))
    amp /= np.max(amp)
    s_rec = interp_s(t_emit) * amp
    s_rec /= np.max(np.abs(s_rec)) * 1.1

    v_s = - (vs**2 * t_emit) / r
    k = v / (v - v_s)
    f_emit_inst = interp_f_emit(t_emit)
    f_rec_inst = f_emit_inst * k
    smooth_window = 200
    f_rec_inst_smooth = np.convolve(f_rec_inst, np.ones(smooth_window)/smooth_window, mode='same')

    # ---- Encode audio for playback ----
    def encode_audio(waveform):
        buf = BytesIO()
        sf.write(buf, waveform, sr, subtype='PCM_16', format='WAV')
        return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode('ascii')

    audio_src = encode_audio(s_rec)
    audio_orig_src = encode_audio(s_emit[:num_samples_rec])

    # ---- Plots ----
    fig_wave = go.Figure()
    ds_factor_wave = max(1, num_samples_rec // 50000)
    fig_wave.add_trace(go.Scatter(x=t_rec[::ds_factor_wave], y=s_rec[::ds_factor_wave],
                                  mode='lines', line=dict(width=0.5), name='Received'))
    fig_wave.update_layout(title="Received Audio Waveform",
                           xaxis_title="Time (s)", yaxis_title="Amplitude")

    fig_freq = go.Figure()
    ds_factor = max(1, num_samples_rec // 1000)
    fig_freq.add_trace(go.Scatter(x=t_rec[::ds_factor], y=f_rec_inst_smooth[::ds_factor],
                                  mode='lines', name='Frequency'))
    fig_freq.update_layout(title="Received Instantaneous Frequency",
                           xaxis_title="Time (s)", yaxis_title="Hz")

    return [
        html.H4("Original Source Sound (stationary)"),
        html.Audio(src=audio_orig_src, controls=True),
        html.H4("Observed Sound with Doppler Effect"),
        html.Audio(src=audio_src, controls=True),
        dcc.Graph(figure=doppler_diagram(vs, d, duration)),
        dcc.Graph(figure=fig_wave),
        dcc.Graph(figure=fig_freq),
        html.H4(f"Base Car Frequency: {car_freq} Hz"),
    ]
