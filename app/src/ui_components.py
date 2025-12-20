"""
UI components and layout for GAINED application
"""
from dash import html, dcc
import dash


def create_upload_section():
    """Create the file upload section"""
    return html.Div([
        html.H2("Upload Session Files"),
        html.Div([
            html.Label("📁 Upload Audio (MP3/WAV) and Transcript (XLSX/CSV) separately:"),
            html.Div([
                html.Div([
                    html.Div('🎵 Audio File', style={'fontSize': '16px', 'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Upload(
                        id='upload-audio',
                        children=html.Div([
                            html.Div('Drop MP3/WAV here', style={'fontSize': '14px', 'color': '#6c757d'}),
                            html.Div('or click to browse', style={'fontSize': '12px', 'color': '#adb5bd', 'marginTop': '5px'})
                        ]),
                        style={
                            'width': '100%',
                            'height': '100px',
                            'lineHeight': '30px',
                            'borderWidth': '3px',
                            'borderStyle': 'dashed',
                            'borderRadius': '10px',
                            'textAlign': 'center',
                            'margin': '10px 0',
                            'backgroundColor': '#f8f9fa',
                            'display': 'flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                            'borderColor': '#007bff'
                        },
                        multiple=False,
                        accept='.mp3,.wav'
                    ),
                ], style={'flex': '1', 'minWidth': '250px', 'marginRight': '10px'}),
                html.Div([
                    html.Div('📄 Transcript File', style={'fontSize': '16px', 'fontWeight': 'bold', 'marginBottom': '5px'}),
                    dcc.Upload(
                        id='upload-transcript',
                        children=html.Div([
                            html.Div('Drop XLSX/CSV here', style={'fontSize': '14px', 'color': '#6c757d'}),
                            html.Div('or click to browse', style={'fontSize': '12px', 'color': '#adb5bd', 'marginTop': '5px'})
                        ]),
                        style={
                            'width': '100%',
                            'height': '100px',
                            'lineHeight': '30px',
                            'borderWidth': '3px',
                            'borderStyle': 'dashed',
                            'borderRadius': '10px',
                            'textAlign': 'center',
                            'margin': '10px 0',
                            'backgroundColor': '#f8f9fa',
                            'display': 'flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                            'borderColor': '#17a2b8'
                        },
                        multiple=False,
                        accept='.xlsx,.csv'
                    ),
                ], style={'flex': '1', 'minWidth': '250px', 'marginLeft': '10px'}),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'}),
            html.Div(id='upload-status', style={'marginTop': '10px', 'fontSize': '14px'})
        ]),
    ], style={'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px', 'marginBottom': '20px', 
              'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'})


def create_audio_section():
    """Create the audio player and waveform section"""
    return html.Div([
        html.H2("Audio Waveform"),
        html.Div(id='audio-player-container'),
        html.Div(id='waveform-container', children=[
            html.Div(id='waveform', style={
                'marginBottom': '20px', 
                'minHeight': '128px',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '5px',
                'border': '1px solid #dee2e6',
                'padding': '10px',
                'textAlign': 'center'
            }, children=[
                html.P("Waveform will appear here when audio is loaded", 
                       style={'color': '#6c757d', 'margin': '50px 0'})
            ]),
        ]),
        html.Div(id='playback-controls', style={'marginTop': '10px'}, children=[
            html.Button('🔄 Init Waveform', id='init-waveform-btn', n_clicks=0, style={
                'marginRight': '10px', 'padding': '8px 15px', 'border': 'none', 
                'borderRadius': '5px', 'cursor': 'pointer', 'backgroundColor': '#28a745', 'color': 'white'
            }),
            html.Button('⏮ -10s', id='rewind-btn', n_clicks=0, style={
                'marginRight': '10px', 'padding': '8px 15px', 'border': 'none', 
                'borderRadius': '5px', 'cursor': 'pointer', 'backgroundColor': '#6c757d', 'color': 'white'
            }),
            html.Button('▶ Play', id='play-pause-btn', n_clicks=0, style={
                'marginRight': '10px', 'padding': '8px 15px', 'border': 'none', 
                'borderRadius': '5px', 'cursor': 'pointer', 'backgroundColor': '#007bff', 'color': 'white'
            }),
            html.Button('⏭ +10s', id='forward-btn', n_clicks=0, style={
                'padding': '8px 15px', 'border': 'none', 
                'borderRadius': '5px', 'cursor': 'pointer', 'backgroundColor': '#6c757d', 'color': 'white'
            }),
        ])
    ], id='audio-section', style={'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px', 
                                  'marginBottom': '20px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'display': 'none'})


def create_transcript_section():
    """Create the transcript display section"""
    return html.Div([
        html.H2("Session Transcript", style={'marginBottom': '20px'}),
        html.Div(id='transcription-display', style={
            'padding': '15px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '5px',
            'minHeight': '200px',
            'maxHeight': '600px',
            'overflowY': 'auto'
        })
    ], id='transcript-section', style={'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px', 
                                       'marginBottom': '20px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'display': 'none'})


def create_metrics_section():
    """Create the metrics chart section"""
    return html.Div([
        html.H2("Session Metrics Over Time"),
        dcc.Graph(id='metrics-chart')
    ], id='chart-section', style={'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px', 
                                  'marginBottom': '20px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'display': 'none'})


def create_pie_chart_section():
    """Create the pie chart section for patient speech-turn fields"""
    return html.Div([
        html.H2("Patient Speech-Turn Analysis", style={'marginBottom': '20px'}),
        html.Div([
            html.Label("Select field to visualize:", style={'fontSize': '14px', 'fontWeight': 'bold', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='pie-chart-field-selector',
                placeholder="Select a categorical field (e.g., emotion, sentiment)...",
                style={'marginBottom': '20px'}
            ),
        ]),
        dcc.Graph(id='pie-chart')
    ], id='pie-chart-section', style={'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px', 
                                      'marginBottom': '20px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'display': 'none'})


def create_layout():
    """Create the main application layout"""
    return html.Div([
        html.H1("GAINED - Therapy Session Analysis", style={'textAlign': 'center', 'marginBottom': '30px'}),
        
        create_upload_section(),
        create_audio_section(),
        create_transcript_section(),
        create_metrics_section(),
        create_pie_chart_section(),
        
        # Hidden divs to store data
        dcc.Store(id='audio-data'),
        dcc.Store(id='transcript-data'),
        dcc.Store(id='current-time', data=0),
        dcc.Interval(id='playback-interval', interval=100, disabled=False),
        
        # Hidden components for development/testing (load existing sessions)
        html.Div([
            dcc.Dropdown(id='patient-dropdown', options=[], value='012', style={'display': 'none'}),
            dcc.Dropdown(id='session-dropdown', value=0, style={'display': 'none'}),
            html.Button('Load', id='load-button', n_clicks=0, style={'display': 'none'})
        ], style={'display': 'none'}),
    ], style={'maxWidth': '1400px', 'margin': '0 auto', 'padding': '20px', 'backgroundColor': '#f0f2f5', 
              'minHeight': '100vh'})

