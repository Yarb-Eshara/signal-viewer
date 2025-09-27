import dash
from dash import html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc
import numpy as np
from scipy.interpolate import interp1d
import base64
from io import BytesIO
import soundfile as sf
import plotly.graph_objects as go
import librosa

dash.register_page(__name__, path="/sound-doppler", name="Sound Doppler")

layout = dbc.Container([
    html.H2("Doppler Effect Simulator"),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.Label("Source Velocity (m/s)"),
            dcc.Slider(id="vs-slider", min=0, max=50, step=1, value=20, marks={i: str(i) for i in range(0, 51, 10)}),
        ], width=4),
        dbc.Col([
            html.Label("Closest Distance (m)"),
            dcc.Slider(id="d-slider", min=0.1, max=20, step=0.1, value=5, marks={i: str(i) for i in [0.1, 5, 10, 15, 20]}),
        ], width=4),
        dbc.Col([
            html.Label("Duration (s)"),
            dcc.Slider(id="dur-slider", min=5, max=30, step=1, value=10, marks={i: str(i) for i in range(5, 31, 5)}),
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
    html.H4("Upload Custom Car Sound (optional)"),
    dcc.Upload(
        id='upload-audio',
        children=html.Div(['Drag and Drop or ', html.A('Select a File')]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Br(),
    dcc.Loading(
        id="loading",
        type="default",
        children=html.Div(id="doppler-output")
    ),
])

@callback(
    Output("doppler-output", "children"),
    Input("vs-slider", "value"),
    Input("d-slider", "value"),
    Input("dur-slider", "value"),
    Input("amp-model", "value"),
    Input('upload-audio', 'contents'),
    Input('upload-audio', 'filename')
)
def update_doppler(vs, d, duration, amp_model, contents, filename):
    if vs == 0:
        return html.Div("No motion, no Doppler effect.")
    
    v = 343.0  # speed of sound in m/s
    sr = 22050  # sample rate for audio
    
    # Define emission time range to cover the full motion, increased buffer for robustness
    t_emit_min = -3 * duration - 30  # larger buffer for longer durations
    t_emit_max = 3 * duration + 30   # larger buffer for longer durations
    
    if contents is not None:
        if filename is None or not filename.lower().endswith(('.wav', '.mp3', '.ogg', '.flac')):
            return html.Div("Invalid or no audio file uploaded. Using synthetic sound.")
        
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        buf = BytesIO(decoded)
        
        s_short, sr_load = librosa.load(buf, sr=None, mono=True)
        if sr_load != sr:
            s_short = librosa.resample(s_short, orig_sr=sr_load, target_sr=sr)
        s_short /= np.max(np.abs(s_short)) or 1
        
        period = len(s_short) / sr
        if period == 0:
            contents = None  # fallback to synthetic
        
        else:
            # Tile to cover t_emit_min to t_emit_max
            num_repeats = np.ceil((t_emit_max - t_emit_min) / period) + 2
            s_emit = np.tile(s_short, int(num_repeats))
            t_emit_grid = np.arange(len(s_emit)) / sr + t_emit_min
            idx = np.searchsorted(t_emit_grid, t_emit_max)
            s_emit = s_emit[:idx]
            t_emit_grid = t_emit_grid[:idx]
            
            # Estimate frequency for plot (used internally for Doppler)
            f0 = librosa.yin(s_short, fmin=100, fmax=2000, sr=sr)
            f0_valid = f0[f0 > 0]
            f_mean = np.median(f0_valid) if len(f0_valid) > 0 else 250  # Adjusted for car sound
            hop_length = 512
            t_f0 = librosa.times_like(f0, sr=sr, hop_length=hop_length)
            interp_f = interp1d(t_f0, f0, kind='linear', bounds_error=False, fill_value=f_mean)
            f_inst = interp_f(np.mod(t_emit_grid - t_emit_min, period))
    
    if contents is None:
        # Generate synthetic car engine sound
        num_samples_emit = int(sr * (t_emit_max - t_emit_min)) + 1
        t_emit_grid = np.linspace(t_emit_min, t_emit_max, num_samples_emit)
        
        f_mean = 250  # Hz (typical car engine idle or low horn)
        f_variation = 20  # Hz (random variation for engine rumble)
        f_inst = f_mean + np.random.normal(0, f_variation / 3, len(t_emit_grid))  # Slight random fluctuation
        
        # Generate phase and signal with noise
        dt = 1.0 / sr
        phase = np.cumsum(2 * np.pi * f_inst * dt)
        s_emit = np.sin(phase) + 0.1 * np.random.normal(0, 1, len(phase))  # Add noise for engine texture
        s_emit /= np.max(np.abs(s_emit))  # normalize
    
    # Interpolator for source signal
    interp_s = interp1d(t_emit_grid, s_emit, kind='linear', fill_value=0, bounds_error=False)
    
    # Interpolator for emit frequency (used internally)
    interp_f_emit = interp1d(t_emit_grid, f_inst, kind='linear', fill_value=f_mean, bounds_error=False)
    
    # Received times, covering full duration
    num_samples_rec = int(sr * duration) + 1
    t_rec = np.linspace(0, duration, num_samples_rec)
    plot_x = t_rec  # Use t_rec directly for full duration
    
    # Compute t_emit for each t_rec using Newton's method
    t_emit = np.zeros_like(t_rec)
    approx_factor = 1 / (1 - vs / v)
    te_guess = -2 * duration * approx_factor  # improved initial guess for longer durations
    
    for i, tr in enumerate(t_rec):
        def f(te):
            r = np.sqrt((vs * te)**2 + d**2)
            return te + r / v - tr + duration / 2  # shift so closest approach at t_rec = duration/2
        
        def df(te):
            r = np.sqrt((vs * te)**2 + d**2)
            if r == 0:
                return 1 + 1e-10  # avoid division by zero
            return 1 + (vs**2 * te) / (v * r)
        
        te = te_guess
        for _ in range(100):  # increased iterations for robustness
            delta = f(te) / df(te)
            te -= delta
            if abs(delta) < 1e-10 or abs(delta) > 1e6:  # prevent divergence
                break
        t_emit[i] = te
        te_guess = te  # use as next guess
    
    # Compute received signal
    r = np.sqrt((vs * t_emit)**2 + d**2)
    if amp_model == 'inverse':
        amp = 1 / (r + 0.1)  # inverse distance, small offset to prevent infinity
    else:  # gaussian
        sigma = duration / 4  # standard deviation for Gaussian
        amp = np.exp(-((t_rec - duration / 2)**2) / (2 * sigma**2))
    amp /= np.max(amp)  # normalize amplitude envelope
    s_rec = interp_s(t_emit) * amp
    s_rec /= np.max(np.abs(s_rec)) * 1.1  # normalize with headroom
    
    # Compute instantaneous received frequency for visualization (standard formula)
    v_s = - (vs**2 * t_emit) / r  # radial velocity (positive for approaching)
    k = v / (v - v_s)  # Doppler factor
    f_emit_inst = interp_f_emit(t_emit)
    f_rec_inst = f_emit_inst * k
    
    # Smooth frequency data to highlight Doppler trend
    smooth_window = 200
    f_rec_inst_smooth = np.convolve(f_rec_inst, np.ones(smooth_window)/smooth_window, mode='same')
    
    # Create waveform plot with downsampling for full duration
    ds_factor_wave = max(1, num_samples_rec // 50000)  # Aim for ~50,000 points for performance
    fig_wave = go.Figure()
    fig_wave.add_trace(go.Scatter(x=plot_x[::ds_factor_wave], y=s_rec[::ds_factor_wave], mode='lines', name='Received Signal', line=dict(width=0.5)))
    fig_wave.update_layout(
        title='Received Audio Waveform',
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        xaxis_range=[0, duration],
        yaxis_range=[-1.1, 1.1],
        shapes=[dict(type='line', x0=duration/2, x1=duration/2, y0=-1.1, y1=1.1, yref='y', line=dict(color='red', dash='dash'))]
    )
    
    # Create frequency plot (smoothed for clarity)
    ds_factor = max(1, num_samples_rec // 1000)
    fig_freq = go.Figure()
    fig_freq.add_trace(go.Scatter(x=plot_x[::ds_factor], y=f_rec_inst_smooth[::ds_factor], mode='lines', name='Received Frequency (Smoothed)'))
    fig_freq.update_layout(title='Received Instantaneous Frequency', xaxis_title='Time (s)', yaxis_title='Frequency (Hz)', xaxis_range=[0, duration])
    
    # Create amplitude envelope plot
    fig_amp = go.Figure()
    fig_amp.add_trace(go.Scatter(x=plot_x[::ds_factor], y=amp[::ds_factor], mode='lines', name='Amplitude Envelope'))
    fig_amp.update_layout(title=f'Amplitude Envelope ({amp_model.capitalize()})', xaxis_title='Time (s)', yaxis_title='Relative Amplitude', xaxis_range=[0, duration])
    
    # Create observer frequency vs. time plot (smoothed)
    fig_obs_freq = go.Figure()
    fig_obs_freq.add_trace(go.Scatter(x=plot_x[::ds_factor], y=f_rec_inst_smooth[::ds_factor], mode='lines', name='Observer Frequency'))
    fig_obs_freq.update_layout(
        title='Observer Frequency vs. Time',
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        xaxis_range=[0, duration]
    )
    
    # Calculate observer frequencies at 5 evenly spaced points (10% to 90% of duration)
    time_points = np.linspace(duration * 0.1, duration * 0.9, 5).round().astype(int)
    t_emit_points = np.interp(time_points, t_rec, t_emit)
    r_points = np.sqrt((vs * t_emit_points)**2 + d**2)
    v_s_points = - (vs**2 * t_emit_points) / r_points
    k_points = v / (v - v_s_points)
    f_observer_points = interp_f_emit(t_emit_points) * k_points
    
    # Create frequency table (only observer frequency)
    freq_table = html.Table(
        [html.Tr([html.Th("Time (s)"), html.Th("Observer Freq (Hz)")])] +
        [html.Tr([html.Td(f"{t}"), html.Td(f"{fo:.1f}")]) 
         for t, fo in zip(time_points, f_observer_points)]
    )
    
    # Generate audio for received signal
    buf = BytesIO()
    sf.write(buf, s_rec, sr, subtype='PCM_16', format='WAV')
    buf.seek(0)
    wav_data = buf.read()
    base64_wav = base64.b64encode(wav_data).decode('ascii')
    audio_src = f"data:audio/wav;base64,{base64_wav}"
    
    # Generate audio for original signal (clipped to same length)
    t_start = 0  # adjusted to start at t=0 in emitted time
    orig_start_idx = np.searchsorted(t_emit_grid, t_start)
    orig_end_idx = orig_start_idx + num_samples_rec
    s_emit_clip = s_emit[orig_start_idx:orig_end_idx]
    if len(s_emit_clip) < num_samples_rec:
        s_emit_clip = np.pad(s_emit_clip, (0, num_samples_rec - len(s_emit_clip)))
    s_emit_clip /= np.max(np.abs(s_emit_clip)) * 1.1
    
    buf_orig = BytesIO()
    sf.write(buf_orig, s_emit_clip, sr, subtype='PCM_16', format='WAV')
    buf_orig.seek(0)
    wav_orig = buf_orig.read()
    base64_orig = base64.b64encode(wav_orig).decode('ascii')
    audio_orig_src = f"data:audio/wav;base64,{base64_orig}"
    
    return [
        html.H4("Original Source Sound (stationary)"),
        html.Audio(src=audio_orig_src, controls=True),
        html.H4("Observed Sound with Doppler Effect"),
        html.Audio(src=audio_src, controls=True),
        dcc.Graph(figure=fig_wave),
        dcc.Graph(figure=fig_freq),
        dcc.Graph(figure=fig_amp),
        dcc.Graph(figure=fig_obs_freq),  # Observer frequency graph
        html.H4("Observer Frequency Values at 5 Time Points"),
        freq_table
    ]