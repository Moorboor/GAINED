"""
UI components and layout for GAINED application
"""
from dash import html, dcc


# Design System Constants
COLORS = {
    'primary': '#2563eb',
    'primary_dark': '#1d4ed8',
    'primary_light': '#dbeafe',
    'success': '#059669',
    'success_light': '#d1fae5',
    'gray_50': '#f9fafb',
    'gray_100': '#f3f4f6',
    'gray_200': '#e5e7eb',
    'gray_300': '#d1d5db',
    'gray_400': '#9ca3af',
    'gray_500': '#6b7280',
    'gray_600': '#4b5563',
    'gray_700': '#374151',
    'gray_800': '#1f2937',
    'gray_900': '#111827',
    'white': '#ffffff',
}

CARD_STYLE = {
    'padding': '24px',
    'backgroundColor': COLORS['white'],
    'borderRadius': '8px',
    'marginBottom': '16px',
    'boxShadow': '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px -1px rgba(0, 0, 0, 0.1)',
    'border': f'1px solid {COLORS["gray_200"]}'
}

HIDDEN_STYLE = {'display': 'none'}


def create_upload_section():
    """Create the file upload section"""
    upload_box_style = {
        'width': '100%',
        'height': '88px',
        'borderWidth': '2px',
        'borderStyle': 'dashed',
        'borderRadius': '8px',
        'textAlign': 'center',
        'backgroundColor': COLORS['gray_50'],
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'center',
        'cursor': 'pointer',
        'transition': 'all 0.15s ease',
    }
    
    return html.Div([
        html.H2("Upload Session Files"),
        html.P(
            "Upload audio and transcript files separately",
            style={'color': COLORS['gray_500'], 'fontSize': '14px', 'marginBottom': '16px'}
        ),
        html.Div([
            html.Div([
                html.Div('Audio File', style={
                    'fontSize': '14px',
                    'fontWeight': '500',
                    'color': COLORS['gray_700'],
                    'marginBottom': '8px'
                }),
                dcc.Upload(
                    id='upload-audio',
                    children=html.Div([
                        html.Div('Drop MP3 or WAV file here', style={
                            'fontSize': '13px',
                            'color': COLORS['gray_500']
                        }),
                        html.Div('or click to browse', style={
                            'fontSize': '12px',
                            'color': COLORS['gray_400'],
                            'marginTop': '4px'
                        })
                    ]),
                    style={**upload_box_style, 'borderColor': COLORS['gray_300']},
                    multiple=False,
                    accept='.mp3,.wav'
                ),
            ], style={'flex': '1', 'minWidth': '240px', 'marginRight': '12px'}),
            html.Div([
                html.Div('Transcript File', style={
                    'fontSize': '14px',
                    'fontWeight': '500',
                    'color': COLORS['gray_700'],
                    'marginBottom': '8px'
                }),
                dcc.Upload(
                    id='upload-transcript',
                    children=html.Div([
                        html.Div('Drop XLSX or CSV file here', style={
                            'fontSize': '13px',
                            'color': COLORS['gray_500']
                        }),
                        html.Div('or click to browse', style={
                            'fontSize': '12px',
                            'color': COLORS['gray_400'],
                            'marginTop': '4px'
                        })
                    ]),
                    style={**upload_box_style, 'borderColor': COLORS['gray_300']},
                    multiple=False,
                    accept='.xlsx,.csv'
                ),
            ], style={'flex': '1', 'minWidth': '240px', 'marginLeft': '12px'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '8px'}),
        html.Div(id='upload-status', style={'marginTop': '12px', 'fontSize': '13px'})
    ], style=CARD_STYLE)


def create_audio_section():
    """Create the audio player and waveform section"""
    button_base = {
        'padding': '8px 16px',
        'border': 'none',
        'borderRadius': '6px',
        'cursor': 'pointer',
        'fontSize': '13px',
        'fontWeight': '500',
        'marginRight': '8px',
    }
    
    return html.Div([
        html.H2("Audio Waveform"),
        html.Div(id='audio-player-container'),
        html.Div(id='waveform-container', children=[
            html.Div(id='waveform', style={
                'marginBottom': '16px',
                'minHeight': '120px',
                'backgroundColor': COLORS['gray_50'],
                'borderRadius': '6px',
                'border': f'1px solid {COLORS["gray_200"]}',
                'padding': '8px',
            }, children=[
                html.P("Waveform will appear when audio is loaded",
                       style={'color': COLORS['gray_400'], 'margin': '48px 0', 'textAlign': 'center', 'fontSize': '13px'})
            ]),
        ]),
        html.Div(id='playback-controls', style={'marginTop': '12px', 'display': 'flex', 'alignItems': 'center'}, children=[
            html.Button('Init Waveform', id='init-waveform-btn', n_clicks=0, style={
                **button_base, 'backgroundColor': COLORS['success'], 'color': 'white'
            }),
            html.Button('−10s', id='rewind-btn', n_clicks=0, style={
                **button_base, 'backgroundColor': COLORS['gray_200'], 'color': COLORS['gray_700']
            }),
            html.Button('▶ Play', id='play-pause-btn', n_clicks=0, style={
                **button_base, 'backgroundColor': COLORS['primary'], 'color': 'white'
            }),
            html.Button('+10s', id='forward-btn', n_clicks=0, style={
                **button_base, 'backgroundColor': COLORS['gray_200'], 'color': COLORS['gray_700'], 'marginRight': '0'
            }),
        ])
    ], id='audio-section', style={**CARD_STYLE, 'display': 'none'})


def create_transcript_section():
    """Create the transcript display section"""
    return html.Div([
        html.H2("Session Transcript"),
        html.Div(id='transcription-display', style={
            'padding': '12px',
            'backgroundColor': COLORS['gray_50'],
            'borderRadius': '6px',
            'minHeight': '200px',
            'maxHeight': '500px',
            'overflowY': 'auto',
            'border': f'1px solid {COLORS["gray_200"]}'
        })
    ], id='transcript-section', style={**CARD_STYLE, 'display': 'none'})


def create_metrics_section():
    """Create the metrics chart section"""
    return html.Div([
        html.H2("Session Metrics"),
        dcc.Graph(id='metrics-chart', config={'displayModeBar': True, 'displaylogo': False})
    ], id='chart-section', style={**CARD_STYLE, 'display': 'none'})


def create_pie_chart_section():
    """Create the pie chart section for patient speech-turn fields"""
    return html.Div([
        html.H2("Patient Speech Analysis"),
        html.Div([
            html.Label("Select field to visualize", style={
                'fontSize': '13px',
                'fontWeight': '500',
                'color': COLORS['gray_600'],
                'marginBottom': '8px',
                'display': 'block'
            }),
            dcc.Dropdown(
                id='pie-chart-field-selector',
                placeholder="Select a field...",
                style={'marginBottom': '16px'}
            ),
        ]),
        dcc.Graph(id='pie-chart', config={'displayModeBar': True, 'displaylogo': False})
    ], id='pie-chart-section', style={**CARD_STYLE, 'display': 'none'})


def create_interventions_pie_section():
    """Create the interventions pie chart section showing Therapist and Patient intervention means"""
    return html.Div([
        html.H2("Interventions Overview"),
        html.P("Mean of intervention scores across session segments", style={
            'color': COLORS['gray_500'], 'fontSize': '13px', 'marginBottom': '16px'
        }),
        html.Div([
            # Therapist pie
            html.Div([
                html.H4("Therapist", style={'textAlign': 'center', 'fontSize': '14px', 'marginBottom': '8px'}),
                dcc.Graph(id='therapist-interventions-pie', config={'displayModeBar': False},
                          style={'height': '350px'})
            ], style={'flex': '1', 'minWidth': '300px'}),
            # Patient pie
            html.Div([
                html.H4("Patient", style={'textAlign': 'center', 'fontSize': '14px', 'marginBottom': '8px'}),
                dcc.Graph(id='patient-interventions-pie', config={'displayModeBar': False},
                          style={'height': '350px'})
            ], style={'flex': '1', 'minWidth': '300px'}),
        ], style={'display': 'flex', 'flexDirection': 'row', 'flexWrap': 'wrap', 'gap': '24px'}),
    ], id='interventions-pie-section', style={**CARD_STYLE, 'display': 'none'})


def create_session_rationale_section():
    """Create the session-level rationale section displaying summary rationale from the rationale sheet"""
    return html.Div([
        html.H2("Session Rationale Summary", style={'marginBottom': '16px'}),
        html.P("Summary rationale for session-level metrics (from rationale sheet)", style={
            'color': COLORS['gray_500'], 'fontSize': '13px', 'marginBottom': '16px'
        }),
        html.Div(id='session-rationale-content', children=[
            html.Div("Upload a session file to see rationale",
                    style={'textAlign': 'center', 'color': COLORS['gray_400'], 'padding': '40px', 'fontSize': '14px'})
        ])
    ], id='session-rationale-section', style={**CARD_STYLE, 'display': 'none'})


def create_field_plots_section():
    """Create the field plots section with rationale display"""
    return html.Div([
        html.H2("Field Analysis with Rationale"),
        html.Div([
            html.Label("Select up to 4 fields to plot", style={
                'fontSize': '13px',
                'fontWeight': '500',
                'color': COLORS['gray_600'],
                'marginBottom': '8px',
                'display': 'block'
            }),
            dcc.Dropdown(
                id='field-plots-selector',
                placeholder="Select fields...",
                multi=True,
                maxHeight=200,
                style={'marginBottom': '16px'}
            ),
        ]),
        html.Div(id='field-plots-container', children=[
            html.Div("Select fields above to display plots",
                    style={'textAlign': 'center', 'color': COLORS['gray_400'], 'padding': '40px', 'fontSize': '14px'})
        ])
    ], id='field-plots-section', style={**CARD_STYLE, 'display': 'none'})




def create_sessions_upload_section():
    """Create the multi-file upload section for sessions"""
    upload_box_style = {
        'width': '100%',
        'height': '120px',
        'borderWidth': '2px',
        'borderStyle': 'dashed',
        'borderRadius': '8px',
        'textAlign': 'center',
        'backgroundColor': COLORS['gray_50'],
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'center',
        'cursor': 'pointer',
        'transition': 'all 0.15s ease',
        'flexDirection': 'column'
    }
    
    return html.Div([
        html.H2("Upload Multiple Sessions"),
        html.P(
            "Upload multiple session files (xlsx/csv) to analyze trends",
            style={'color': COLORS['gray_500'], 'fontSize': '14px', 'marginBottom': '16px'}
        ),
        dcc.Upload(
            id='upload-sessions',
            children=html.Div([
                html.Div('Drop multiple session files here', style={
                    'fontSize': '15px',
                    'fontWeight': '500',
                    'color': COLORS['gray_700'],
                    'marginBottom': '4px'
                }),
                html.Div('Accepts .xlsx and .csv files', style={
                    'fontSize': '13px',
                    'color': COLORS['gray_500']
                }),
                html.Div('or click to browse', style={
                    'fontSize': '12px',
                    'color': COLORS['gray_400'],
                    'marginTop': '8px'
                })
            ]),
            style={**upload_box_style, 'borderColor': COLORS['gray_300']},
            multiple=True,
            accept='.xlsx,.csv'
        ),
        html.Div(id='sessions-upload-status', style={'marginTop': '12px', 'fontSize': '13px'})
    ], style=CARD_STYLE)


def create_sessions_detailed_charts_section():
    """Create the specific detailed charts section with rationale boxes"""
    
    def create_chart_with_rationale(title, chart_id, rationale_id):
        badge_id = f'{rationale_id}-badge'
        return html.Div([
            html.H3(title, style={'fontSize': '16px', 'marginBottom': '12px'}),
            html.Div([
                # Chart Container
                html.Div([
                    dcc.Graph(id=chart_id, config={'displayModeBar': True, 'displaylogo': False}, style={'height': '400px'})
                ], style={
                    'flex': '3', 
                    'minWidth': '500px',
                    'padding': '12px',
                    'border': f'1px solid {COLORS["gray_200"]}',
                    'borderRadius': '6px',
                    'backgroundColor': COLORS['white']
                }), 
                
                # Rationale Box Container
                html.Div([
                    # Sticky header with rationale title + session badge
                    html.Div([
                        html.H4("Rationale", style={
                            'fontSize': '14px', 'color': COLORS['gray_600'], 'margin': '0'
                        }),
                        html.Span(id=badge_id, children="No session", style={
                            'display': 'inline-block',
                            'padding': '2px 10px',
                            'fontSize': '11px',
                            'fontWeight': '500',
                            'color': COLORS['gray_600'],
                            'backgroundColor': COLORS['gray_100'],
                            'border': f'1px solid {COLORS["gray_200"]}',
                            'borderRadius': '12px',
                            'whiteSpace': 'nowrap',
                        })
                    ], style={
                        'display': 'flex',
                        'alignItems': 'center',
                        'justifyContent': 'space-between',
                        'position': 'sticky',
                        'top': '0',
                        'zIndex': '10',
                        'backgroundColor': COLORS['white'],
                        'padding': '8px 0',
                        'borderBottom': f'1px solid {COLORS["gray_200"]}',
                        'marginBottom': '8px',
                    }),
                    # Scrollable rationale content
                    html.Div(id=rationale_id, style={
                        'overflowY': 'auto',
                        'padding': '12px',
                        'backgroundColor': COLORS['gray_50'],
                        'border': f'1px solid {COLORS["gray_200"]}',
                        'borderRadius': '6px',
                        'fontSize': '13px',
                        'lineHeight': '1.5',
                        'color': COLORS['gray_700'],
                        'whiteSpace': 'pre-line',
                        'flex': '1',
                    }, children="Click on a point to see rationale.")
                ], style={
                    'flex': '1', 'minWidth': '300px',
                    'display': 'flex', 'flexDirection': 'column',
                    'maxHeight': '430px',
                }) 
                
            ], style={'display': 'flex', 'flexDirection': 'row', 'flexWrap': 'wrap', 'gap': '24px', 'padding': '0 12px'})
        ], style={**CARD_STYLE, 'marginBottom': '24px', 'marginTop': '12px'})

    return html.Div([
        # 1. Challenging vs Supporting
        create_chart_with_rationale("Therapist Contribution (TCCS)", 'tccs-chart', 'tccs-rationale'),
        
        # 2. Activation vs Engagement
        create_chart_with_rationale("Patient State (Activation & Engagement)", 'activation-engagement-chart', 'activation-rationale'),
        
        # 3. CTS Breakdown
        create_chart_with_rationale("Competence Scale (CTS)", 'cts-chart', 'cts-rationale'),
        
    ], id='sessions-detailed-charts-section', style={'display': 'none', 'marginTop': '24px', 'padding': '0 12px'})


def create_sessions_layout():
    """Create the sessions page layout"""
    return html.Div([
        # Header
        html.Div([
            html.H1("GAINED - Sessions Analysis", style={
                'textAlign': 'center',
                'marginBottom': '4px',
                'color': COLORS['gray_900']
            }),
            html.P("Analyze trends across multiple therapy sessions", style={
                'textAlign': 'center',
                'color': COLORS['gray_500'],
                'fontSize': '14px',
                'marginBottom': '24px'
            }),
            html.Div([
                dcc.Link("← Back to Single Session Analysis", href="/", style={
                    'color': COLORS['primary'],
                    'fontSize': '14px',
                    'textDecoration': 'none',
                    'fontWeight': '500',
                    'padding': '8px 16px',
                    'backgroundColor': COLORS['white'],
                    'borderRadius': '6px',
                    'border': f'1px solid {COLORS["gray_200"]}',
                    'display': 'inline-block'
                })
            ], style={'textAlign': 'center', 'marginBottom': '24px'})
        ]),
        
        # Main Content
        create_sessions_upload_section(),
        # Removed generic chart section
        create_sessions_detailed_charts_section(),
        
        # Data Stores for Sessions
        dcc.Store(id='sessions-data'),
        
    ], style={
        'maxWidth': '1200px',
        'margin': '0 auto',
        'padding': '24px',
        'backgroundColor': COLORS['gray_100'],
        'minHeight': '100vh'
    })


def create_layout():
    """Create the main application layout wrapper with routing"""
    return html.Div([
        dcc.Location(id='url', refresh=False),
        html.Div(id='page-content')
    ])


def create_main_analysis_layout():
    """Create the single session analysis layout (original layout)"""
    return html.Div([
        # Header
        html.Div([
            html.H1("GAINED", style={
                'textAlign': 'center',
                'marginBottom': '4px',
                'color': COLORS['gray_900']
            }),
            html.P("Therapy Session Analysis", style={
                'textAlign': 'center',
                'color': COLORS['gray_500'],
                'fontSize': '14px',
                'marginBottom': '24px'
            })
        ]),
        
        # Main Content
        create_upload_section(),
        create_audio_section(),
        create_transcript_section(),
        create_metrics_section(),
        create_pie_chart_section(),
        create_interventions_pie_section(),
        create_field_plots_section(),
        create_session_rationale_section(),
        
        # Navigation to Sessions
        html.Div([
            dcc.Link("Go to Multi-Session Analysis →", href="/sessions", style={
                'display': 'block',
                'width': '100%',
                'padding': '16px',
                'textAlign': 'center',
                'backgroundColor': COLORS['white'],
                'color': COLORS['primary'],
                'fontWeight': '600',
                'textDecoration': 'none',
                'borderRadius': '8px',
                'border': f'1px solid {COLORS["gray_200"]}',
                'boxShadow': '0 1px 3px 0 rgba(0, 0, 0, 0.1)'
            })
        ], style={'marginTop': '32px', 'marginBottom': '48px'}),
        
        # Data Stores
        dcc.Store(id='audio-data'),
        dcc.Store(id='transcript-data'),
        dcc.Store(id='rationale-data', data={}),
        dcc.Store(id='current-time', data=0),
        dcc.Interval(id='playback-interval', interval=100, disabled=False),
        
        # Hidden components (development/testing)
        html.Div([
            dcc.Dropdown(id='patient-dropdown', options=[], value='012', style=HIDDEN_STYLE),
            dcc.Dropdown(id='session-dropdown', value=0, style=HIDDEN_STYLE),
            html.Button('Load', id='load-button', n_clicks=0, style=HIDDEN_STYLE)
        ], style=HIDDEN_STYLE),
    ], style={
        'maxWidth': '1200px',
        'margin': '0 auto',
        'padding': '24px',
        'backgroundColor': COLORS['gray_100'],
        'minHeight': '100vh'
    })
