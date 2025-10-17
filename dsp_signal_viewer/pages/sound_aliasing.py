import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback
from dash import register_page
import plotly.graph_objs as go
import numpy as np
import librosa
import soundfile as sf
import io
import base64
from scipy import signal
import tempfile
import os
from voicefixer import VoiceFixer as VoiceFixerModel


VOICEFIXER_AVAILABLE = True
VOICEFIXER_MODEL = None

register_page(
    __name__,
    path="/sound-aliasing",
    name="Sound Aliasing",
    title="Sound Aliasing",
    description="Explore sound aliasing effects with adjustable parameters."
)

def load_voicefixer():
    """Load VoiceFixer model - automatically downloads models on first run"""
    global VOICEFIXER_MODEL, VOICEFIXER_AVAILABLE
    try:
        if VoiceFixerModel is None:
            print("voicefixer not installed")
            return False
        
        # Initialize VoiceFixer - models auto-download on first run
        VOICEFIXER_MODEL = VoiceFixerModel()
        VOICEFIXER_AVAILABLE = True
        print("VoiceFixer model loaded successfully")
        return True
        
    except Exception as e:
        print(f"Error loading VoiceFixer: {e}")
        import traceback
        traceback.print_exc()
        VOICEFIXER_AVAILABLE = False
        return False

layout = dbc.Container(
    [
        html.H1("Sound Aliasing & Anti-Aliasing Demo", className="text-center my-4"),
        html.Hr(),
        
        # Alert for VoiceFixer availability
        dbc.Alert(
            "VoiceFixer model loaded successfully!" if VOICEFIXER_AVAILABLE else 
            "VoiceFixer not available. Install with: pip install voicefixer",
            color="success" if VOICEFIXER_AVAILABLE else "warning",
            className="mb-3"
        ),

        # Upload Section
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Step 1: Upload Audio", className="card-title"),
                                    dcc.Upload(
                                        id='upload-audio',
                                        children=html.Div([
                                            html.I(className="bi bi-cloud-upload me-2"),
                                            'Drag and Drop or Click to Select Audio File'
                                        ]),
                                        style={
                                            'width': '100%',
                                            'height': '60px',
                                            'lineHeight': '60px',
                                            'borderWidth': '2px',
                                            'borderStyle': 'dashed',
                                            'borderRadius': '5px',
                                            'textAlign': 'center',
                                            'cursor': 'pointer'
                                        },
                                        multiple=False
                                    ),
                                    html.Div(id='upload-status', className="mt-2"),
                                ]
                            )
                        ),
                    ],
                    width=12,
                    className="mb-3"
                ),
            ]
        ),

        # Controls Section
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Step 2: Downsample Audio", className="card-title"),
                                    html.Label("Downsampling Factor:", className="mt-3"),
                                    dcc.Slider(
                                        id="downsample-slider",
                                        min=1,
                                        max=16,
                                        step=1,
                                        value=1,
                                        marks={i: f'{i}x' for i in [1, 2, 4, 8, 16]},
                                        tooltip={"placement": "bottom", "always_visible": True}
                                    ),
                                    html.Div(id="sample-rate-info", className="mt-3 text-muted"),
                                    html.Hr(),
                                    html.Label("Playback Controls:", className="mt-2"),
                                    dbc.ButtonGroup(
                                        [
                                            dbc.Button("Play Original", id="play-original-btn", 
                                                      color="primary", className="me-2"),
                                            dbc.Button("Play Downsampled", id="play-downsampled-btn", 
                                                      color="warning", className="me-2"),
                                            dbc.Button("Play Anti-Aliased", id="play-antialiased-btn", 
                                                      color="success", disabled=not VOICEFIXER_AVAILABLE),
                                        ],
                                        className="mt-2"
                                    ),
                                    html.Audio(id='audio-player', controls=True, className="w-100 mt-3"),
                                    html.Hr(),
                                    html.Label("Step 3: Apply Anti-Aliasing:", className="mt-2"),
                                    html.Label("VoiceFixer Mode:", className="mt-2"),
                                    dcc.Dropdown(
                                        id="voicefixer-mode",
                                        options=[
                                            {'label': 'Mode 0: Basic restoration', 'value': 0},
                                            {'label': 'Mode 1: Mild enhancement', 'value': 1},
                                            {'label': 'Mode 2: Strong enhancement', 'value': 2},
                                        ],
                                        value=2,
                                        clearable=False,
                                        className="mb-2"
                                    ),
                                    dbc.Checklist(
                                        options=[
                                            {"label": "Apply pre-emphasis filter", "value": "preemph"},
                                            {"label": "Normalize before processing", "value": "normalize"},
                                        ],
                                        value=["normalize"],
                                        id="preprocessing-options",
                                        className="mb-2"
                                    ),
                                    dbc.Button("Apply VoiceFixer", id="apply-voicefixer-btn", 
                                              color="success", className="w-100 mt-2",
                                              disabled=not VOICEFIXER_AVAILABLE),
                                    html.Div(id="voicefixer-status", className="mt-2"),
                                    html.Hr(),
                                    dbc.Button("Download Processed Audio", id="download-btn", 
                                              color="info", className="w-100 mt-2"),
                                    dcc.Download(id="download-audio"),
                                ]
                            )
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Waveform Visualization", className="card-title"),
                                    dcc.Graph(id="waveform-graph", style={'height': '400px'}),
                                ]
                            )
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Signal Comparison: Original vs Anti-Aliased", className="card-title"),
                                    html.P("Green = Similar | Yellow = Moderate | Red = Different", 
                                           className="text-muted small"),
                                    dcc.Graph(id="comparison-graph", style={'height': '400px'}),
                                ]
                            ),
                            className="mt-3"
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Similarity Metrics", className="card-title"),
                                    html.Div(id="metrics-display", className="mt-2"),
                                ]
                            ),
                            className="mt-3"
                        ),
                    ],
                    width=8,
                ),
            ]
        ),
        
        # Hidden divs to store audio data
        html.Div(id='original-audio-store', style={'display': 'none'}),
        html.Div(id='downsampled-audio-store', style={'display': 'none'}),
        html.Div(id='antialiased-audio-store', style={'display': 'none'}),
    ],
    fluid=True,
)

# Helper functions
def parse_audio_contents(contents):
    """Parse uploaded audio file"""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    audio_data, sr = librosa.load(io.BytesIO(decoded), sr=None, mono=False)
    
    # Convert to mono if stereo
    if len(audio_data.shape) > 1:
        audio_data = librosa.to_mono(audio_data)
    
    return audio_data, sr

def downsample_audio(audio_data, original_sr, factor):
    """Downsample audio by a given factor"""
    if factor == 1:
        return audio_data, original_sr
    
    new_sr = original_sr // factor
    downsampled = signal.resample(audio_data, len(audio_data) // factor)
    return downsampled, new_sr

def audio_to_base64(audio_data, sr):
    """Convert audio array to base64 encoded WAV"""
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sr, format='WAV')
    buffer.seek(0)
    audio_base64 = base64.b64encode(buffer.read()).decode()
    return f"data:audio/wav;base64,{audio_base64}"

def create_waveform_plot(audio_data, sr, title="Waveform"):
    """Create plotly waveform visualization"""
    time = np.linspace(0, len(audio_data) / sr, len(audio_data))
    
    # Downsample for visualization if too long
    max_points = 10000
    if len(time) > max_points:
        step = len(time) // max_points
        time = time[::step]
        audio_data = audio_data[::step]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=audio_data, mode='lines', name='Amplitude'))
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_white",
        margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig

def create_comparison_plot(original_data, antialiased_data, sr, title="Signal Comparison"):
    """Create comparison plot with original and anti-aliased signals overlaid with color-coded differences"""
    time = np.linspace(0, len(original_data) / sr, len(original_data))
    
    # Calculate point-wise difference
    difference = np.abs(original_data - antialiased_data)
    
    # Calculate similarity metrics
    # 1. Mean Squared Error (MSE)
    mse = np.mean((original_data - antialiased_data) ** 2)
    
    # 2. Signal-to-Noise Ratio (SNR)
    signal_power = np.mean(original_data ** 2)
    noise_power = np.mean((original_data - antialiased_data) ** 2)
    snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # 3. Pearson Correlation Coefficient
    correlation = np.corrcoef(original_data, antialiased_data)[0, 1]
    
    # 4. Percentage similarity (based on normalized difference)
    max_possible_diff = np.max(np.abs(original_data)) + np.max(np.abs(antialiased_data))
    similarity_percent = (1 - np.mean(difference) / (max_possible_diff + 1e-10)) * 100
    
    # Downsample for visualization if too long
    max_points = 10000
    if len(time) > max_points:
        step = len(time) // max_points
        time = time[::step]
        original_data = original_data[::step]
        antialiased_data = antialiased_data[::step]
        difference = difference[::step]
    
    # Normalize difference for color mapping (0 = similar, 1 = different)
    diff_normalized = difference / (np.max(difference) + 1e-10)
    
    # Create color array: green for similar, red for different
    # Using a gradient from green (low difference) to red (high difference)
    colors = []
    for d in diff_normalized:
        if d < 0.2:  # Very similar
            colors.append(f'rgba(0, 255, 0, 0.7)')  # Green
        elif d < 0.5:  # Somewhat similar
            colors.append(f'rgba(255, 255, 0, 0.7)')  # Yellow
        else:  # Different
            colors.append(f'rgba(255, 0, 0, 0.7)')  # Red
    
    fig = go.Figure()
    
    # Add original signal (thin line in background)
    fig.add_trace(go.Scatter(
        x=time, 
        y=original_data, 
        mode='lines', 
        name='Original Signal',
        line=dict(color='lightblue', width=1),
        opacity=0.5
    ))
    
    # Add anti-aliased signal with color-coded segments
    # We'll use a scatter plot with markers to show color differences
    fig.add_trace(go.Scatter(
        x=time, 
        y=antialiased_data, 
        mode='markers',
        name='Anti-Aliased (Color-coded)',
        marker=dict(
            size=3,
            color=diff_normalized,
            colorscale=[[0, 'green'], [0.5, 'yellow'], [1, 'red']],
            showscale=True,
            colorbar=dict(
                title="Difference<br>Magnitude",
                titleside="right",
                tickmode="linear",
                tick0=0,
                dtick=0.25,
                tickvals=[0, 0.25, 0.5, 0.75, 1],
                ticktext=['Very Similar', 'Similar', 'Moderate', 'Different', 'Very Different']
            )
        ),
        hovertemplate='Time: %{x:.3f}s<br>Amplitude: %{y:.3f}<br>Difference: %{marker.color:.3f}<extra></extra>'
    ))
    
    # Add metrics as annotation
    metrics_text = (
        f"<b>Similarity Metrics:</b><br>"
        f"Overall Similarity: {similarity_percent:.2f}%<br>"
        f"Correlation: {correlation:.4f}<br>"
        f"SNR: {snr_db:.2f} dB<br>"
        f"MSE: {mse:.6f}"
    )
    
    fig.add_annotation(
        text=metrics_text,
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        align="left",
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        font=dict(size=10)
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        template="plotly_white",
        margin=dict(l=50, r=100, t=50, b=50),
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )
    return fig

# Callbacks
@callback(
    [Output('upload-status', 'children'),
     Output('original-audio-store', 'children'),
     Output('waveform-graph', 'figure'),
     Output('comparison-graph', 'figure'),
     Output('sample-rate-info', 'children')],
    Input('upload-audio', 'contents'),
    State('upload-audio', 'filename')
)
def upload_audio(contents, filename):
    if contents is None:
        empty_fig = go.Figure()
        empty_fig.update_layout(title="Upload audio to see visualization")
        return "", "", empty_fig, empty_fig, ""
    
    try:
        audio_data, sr = parse_audio_contents(contents)
        
        # Store original audio as JSON
        audio_store = {
            'data': audio_data.tolist(),
            'sr': sr
        }
        
        # Create visualizations
        waveform_fig = create_waveform_plot(audio_data, sr, "Original Waveform")
        
        # Create empty comparison plot initially
        comparison_fig = go.Figure()
        comparison_fig.update_layout(
            title="Apply anti-aliasing to see comparison",
            template="plotly_white"
        )
        
        status = dbc.Alert(f"✓ Loaded: {filename} (Sample Rate: {sr} Hz)", color="success")
        info = f"Original Sample Rate: {sr} Hz | Duration: {len(audio_data)/sr:.2f}s"
        
        return status, str(audio_store), waveform_fig, comparison_fig, info
    except Exception as e:
        return dbc.Alert(f"Error: {str(e)}", color="danger"), "", go.Figure(), go.Figure(), ""

@callback(
    [Output('downsampled-audio-store', 'children'),
     Output('waveform-graph', 'figure', allow_duplicate=True),
     Output('sample-rate-info', 'children', allow_duplicate=True)],
    Input('downsample-slider', 'value'),
    State('original-audio-store', 'children'),
    prevent_initial_call=True
)
def update_downsample(factor, audio_store_str):
    if not audio_store_str:
        empty_fig = go.Figure()
        return "", empty_fig, ""
    
    try:
        audio_store = eval(audio_store_str)
        audio_data = np.array(audio_store['data'])
        sr = audio_store['sr']
        
        # Downsample
        downsampled, new_sr = downsample_audio(audio_data, sr, factor)
        
        # Upsample back to original rate for visualization
        upsampled = signal.resample(downsampled, len(audio_data))
        
        # Store downsampled audio
        ds_store = {
            'data': downsampled.tolist(),
            'sr': new_sr,
            'upsampled': upsampled.tolist()
        }
        
        # Create visualizations
        waveform_fig = create_waveform_plot(upsampled, sr, f"Downsampled Waveform ({factor}x)")
        
        info = f"Original SR: {sr} Hz | Downsampled SR: {new_sr} Hz | Factor: {factor}x"
        
        return str(ds_store), waveform_fig, info
    except Exception as e:
        print(f"Error in downsample: {e}")
        return "", go.Figure(), ""

@callback(
    Output('audio-player', 'src'),
    [Input('play-original-btn', 'n_clicks'),
     Input('play-downsampled-btn', 'n_clicks'),
     Input('play-antialiased-btn', 'n_clicks')],
    [State('original-audio-store', 'children'),
     State('downsampled-audio-store', 'children'),
     State('antialiased-audio-store', 'children')],
    prevent_initial_call=True
)
def play_audio(play_orig, play_ds, play_aa, orig_store, ds_store, aa_store):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    try:
        if button_id == 'play-original-btn' and orig_store:
            store = eval(orig_store)
            audio_data = np.array(store['data'])
            sr = store['sr']
            return audio_to_base64(audio_data, sr)
        
        elif button_id == 'play-downsampled-btn' and ds_store:
            store = eval(ds_store)
            audio_data = np.array(store['upsampled'])
            sr = eval(orig_store)['sr']
            return audio_to_base64(audio_data, sr)
        
        elif button_id == 'play-antialiased-btn' and aa_store:
            store = eval(aa_store)
            audio_data = np.array(store['data'])
            sr = store['sr']
            return audio_to_base64(audio_data, sr)
    except Exception as e:
        print(f"Error in playback: {e}")
    
    return dash.no_update

@callback(
    [Output('antialiased-audio-store', 'children'),
     Output('voicefixer-status', 'children'),
     Output('comparison-graph', 'figure', allow_duplicate=True),
     Output('metrics-display', 'children')],
    Input('apply-voicefixer-btn', 'n_clicks'),
    [State('downsampled-audio-store', 'children'),
     State('original-audio-store', 'children'),
     State('voicefixer-mode', 'value'),
     State('preprocessing-options', 'value')],
    prevent_initial_call=True
)
def apply_voicefixer(n_clicks, ds_store, orig_store, vf_mode, preprocess_opts):
    if not n_clicks or not ds_store:
        return "", dbc.Alert("No downsampled audio available", color="warning"), dash.no_update, ""
    
    if not VOICEFIXER_AVAILABLE or VoiceFixerModel is None:
        return "", dbc.Alert("VoiceFixer not available. Install with: pip install voicefixer", color="danger"), dash.no_update, ""
    
    try:
        # Ensure VoiceFixer is loaded
        if VOICEFIXER_MODEL is None:
            if not load_voicefixer():
                return "", dbc.Alert("Failed to load VoiceFixer model. Please check installation.", color="danger"), dash.no_update, ""
        
        ds = eval(ds_store)
        orig = eval(orig_store)
        
        downsampled = np.array(ds['data'])
        ds_sr = ds['sr']
        orig_sr = orig['sr']
        original_audio = np.array(orig['data'])
        
        # Preprocessing options
        preprocess_opts = preprocess_opts or []
        
        # Apply pre-emphasis filter if selected
        if 'preemph' in preprocess_opts:
            # Pre-emphasis filter to boost high frequencies
            pre_emphasis = 0.97
            downsampled = np.append(downsampled[0], downsampled[1:] - pre_emphasis * downsampled[:-1])
        
        # Normalize audio to [-1, 1] range
        downsampled = downsampled.astype(np.float32)
        if 'normalize' in preprocess_opts and np.max(np.abs(downsampled)) > 0:
            downsampled = downsampled / np.max(np.abs(downsampled))
        
        # Ensure audio is in proper range
        downsampled = np.clip(downsampled, -1.0, 1.0)
        
        # Save to temporary file with higher quality
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_input:
            # Use 32-bit float for better quality
            sf.write(tmp_input.name, downsampled, ds_sr, subtype='FLOAT')
            input_path = tmp_input.name
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_output:
            output_path = tmp_output.name
        
        try:
            # Apply VoiceFixer with selected mode
            VOICEFIXER_MODEL.restore(
                input=input_path,
                output=output_path,
                cuda=False,
                mode=vf_mode  # Use selected mode (0, 1, or 2)
            )
            
            # Load processed audio
            processed_audio, processed_sr = librosa.load(output_path, sr=orig_sr, mono=True)
            
            # Post-processing: Match length if needed
            if len(processed_audio) != len(original_audio):
                if len(processed_audio) < len(original_audio):
                    # Pad if shorter
                    processed_audio = np.pad(processed_audio, (0, len(original_audio) - len(processed_audio)))
                else:
                    # Trim if longer
                    processed_audio = processed_audio[:len(original_audio)]
            
            # Optional: Apply de-emphasis if pre-emphasis was used
            if 'preemph' in preprocess_opts:
                pre_emphasis = 0.97
                processed_audio_deemph = np.zeros_like(processed_audio)
                processed_audio_deemph[0] = processed_audio[0]
                for i in range(1, len(processed_audio)):
                    processed_audio_deemph[i] = processed_audio[i] + pre_emphasis * processed_audio_deemph[i-1]
                processed_audio = processed_audio_deemph
            
            # Normalize final output
            if np.max(np.abs(processed_audio)) > 0:
                processed_audio = processed_audio / np.max(np.abs(processed_audio))
                processed_audio = processed_audio * np.max(np.abs(original_audio))
            
            # Calculate detailed metrics
            difference = np.abs(original_audio - processed_audio)
            mse = np.mean((original_audio - processed_audio) ** 2)
            signal_power = np.mean(original_audio ** 2)
            noise_power = np.mean((original_audio - processed_audio) ** 2)
            snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
            correlation = np.corrcoef(original_audio, processed_audio)[0, 1]
            max_possible_diff = np.max(np.abs(original_audio)) + np.max(np.abs(processed_audio))
            similarity_percent = (1 - np.mean(difference) / (max_possible_diff + 1e-10)) * 100
            
            # Calculate percentage of signal that is similar (difference < 10% threshold)
            threshold = 0.1 * np.max(np.abs(original_audio))
            similar_points = np.sum(difference < threshold)
            similar_percent = (similar_points / len(difference)) * 100
            
            # Store antialiased audio
            aa_store = {
                'data': processed_audio.tolist(),
                'sr': processed_sr
            }
            
            # Create comparison plot
            comparison_fig = create_comparison_plot(
                original_audio, 
                processed_audio, 
                orig_sr, 
                f"Original (Blue) vs Anti-Aliased Mode {vf_mode} (Color-coded)"
            )
            
            # Create detailed metrics display
            metrics_display = html.Div([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.H6("Overall Similarity", className="text-muted mb-1"),
                            html.H3(f"{similarity_percent:.2f}%", className="text-success mb-0"),
                        ], className="text-center p-3 border rounded bg-light")
                    ], width=6),
                    dbc.Col([
                        html.Div([
                            html.H6("Points Within Threshold", className="text-muted mb-1"),
                            html.H3(f"{similar_percent:.2f}%", className="text-info mb-0"),
                        ], className="text-center p-3 border rounded bg-light")
                    ], width=6),
                ], className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Strong("Correlation Coefficient:"),
                        html.Span(f" {correlation:.4f}", className="ms-2"),
                        dbc.Progress(value=correlation*100, color="primary", className="mt-1"),
                    ], width=6, className="mb-2"),
                    dbc.Col([
                        html.Strong("Signal-to-Noise Ratio:"),
                        html.Span(f" {snr_db:.2f} dB", className="ms-2"),
                        dbc.Progress(
                            value=min(max((snr_db + 20) / 60 * 100, 0), 100), 
                            color="success" if snr_db > 20 else "warning" if snr_db > 10 else "danger",
                            className="mt-1"
                        ),
                    ], width=6, className="mb-2"),
                ]),
                dbc.Row([
                    dbc.Col([
                        html.Strong("Mean Squared Error:"),
                        html.Span(f" {mse:.6f}", className="ms-2"),
                    ], width=12, className="mb-2"),
                ]),
                html.Hr(className="my-2"),
                html.P([
                    html.Strong("Interpretation: "),
                    "Green regions show where signals match closely. ",
                    "Red regions indicate significant differences. ",
                    f"Higher correlation ({correlation:.2f}) and SNR ({snr_db:.1f} dB) indicate better reconstruction."
                ], className="small text-muted mb-0")
            ])
            
            status = dbc.Alert(f"✓ VoiceFixer applied successfully (Mode {vf_mode})!", color="success")
            
            return str(aa_store), status, comparison_fig, metrics_display
        finally:
            # Clean up temp files
            try:
                if os.path.exists(input_path):
                    os.unlink(input_path)
                if os.path.exists(output_path):
                    os.unlink(output_path)
            except:
                pass
        
    except Exception as e:
        print(f"Error applying VoiceFixer: {e}")
        import traceback
        traceback.print_exc()
        error_msg = dbc.Alert(f"Error applying VoiceFixer: {str(e)}", color="danger")
        return "", error_msg, dash.no_update, ""

@callback(
    Output('download-audio', 'data'),
    Input('download-btn', 'n_clicks'),
    State('antialiased-audio-store', 'children'),
    prevent_initial_call=True
)
def download_audio(n_clicks, aa_store):
    if not n_clicks or not aa_store:
        return dash.no_update
    
    try:
        store = eval(aa_store)
        audio_data = np.array(store['data'])
        sr = store['sr']
        
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sr, format='WAV')
        buffer.seek(0)
        
        return dcc.send_bytes(buffer.getvalue(), "antialiased_audio.wav")
    except Exception as e:
        print(f"Error downloading: {e}")
        return dash.no_update