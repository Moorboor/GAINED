"""
UI components and layout for GAINED application
"""
import json
import os
from dash import html, dcc

# Load strings
_STRINGS_PATH = os.path.join(os.path.dirname(__file__), 'strings.json')
with open(_STRINGS_PATH, 'r', encoding='utf-8') as _f:
    STRINGS = json.load(_f)

def t(lang, key):
    return STRINGS.get(lang, STRINGS['de']).get(key, key)


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
        html.H2(id='single-upload-title', children="Sitzungsdateien hochladen"),
        html.P(
            id='single-upload-subtitle',
            children="Audio- und Transkriptdatei separat hochladen",
            style={'color': COLORS['gray_500'], 'fontSize': '14px', 'marginBottom': '16px'}
        ),
        html.Div([
            html.Div([
                html.Div(id='single-audio-label', children='Audiodatei', style={
                    'fontSize': '14px',
                    'fontWeight': '500',
                    'color': COLORS['gray_700'],
                    'marginBottom': '8px'
                }),
                dcc.Upload(
                    id='upload-audio',
                    children=html.Div([
                        html.Div(id='single-audio-drop', children='MP3- oder WAV-Datei hier ablegen', style={
                            'fontSize': '13px',
                            'color': COLORS['gray_500']
                        }),
                        html.Div(id='single-audio-browse', children='oder hier klicken', style={
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
                html.Div(id='single-transcript-label', children='Transkriptdatei', style={
                    'fontSize': '14px',
                    'fontWeight': '500',
                    'color': COLORS['gray_700'],
                    'marginBottom': '8px'
                }),
                dcc.Upload(
                    id='upload-transcript',
                    children=html.Div([
                        html.Div(id='single-transcript-drop', children='XLSX- oder CSV-Datei hier ablegen', style={
                            'fontSize': '13px',
                            'color': COLORS['gray_500']
                        }),
                        html.Div(id='single-transcript-browse', children='oder hier klicken', style={
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
        html.H2(id='single-audio-section-title', children="Audio-Wellenform"),
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
                html.P(id='single-waveform-placeholder', children="Wellenform erscheint, wenn Audio geladen ist",
                       style={'color': COLORS['gray_400'], 'margin': '48px 0', 'textAlign': 'center', 'fontSize': '13px'})
            ]),
        ]),
        html.Div(id='playback-controls', style={'marginTop': '12px', 'display': 'flex', 'alignItems': 'center'}, children=[
            html.Button(id='init-waveform-btn', children=html.Span(id='single-btn-init', children='Wellenform initialisieren'), n_clicks=0, style={
                **button_base, 'backgroundColor': COLORS['success'], 'color': 'white'
            }),
            html.Button('−10s', id='rewind-btn', n_clicks=0, style={
                **button_base, 'backgroundColor': COLORS['gray_200'], 'color': COLORS['gray_700']
            }),
            html.Button(id='play-pause-btn', children=html.Span(id='single-btn-play', children='▶ Abspielen'), n_clicks=0, style={
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
        html.H2(id='single-transcript-title', children="Sitzungsprotokoll"),
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



def create_interventions_pie_section():
    """Create the interventions pie chart section showing Therapist and Patient intervention means"""
    return html.Div([
        html.H2(id='single-interventions-title', children="Interventionsübersicht"),
        html.P(id='single-interventions-subtitle', children="Durchschnitt der Interventionswerte über Sitzungssegmente", style={
            'color': COLORS['gray_500'], 'fontSize': '13px', 'marginBottom': '16px'
        }),
        html.Div([
            # Therapist pie
            html.Div([
                html.H4(id='single-interventions-therapist', children="Therapeut", style={'textAlign': 'center', 'fontSize': '14px', 'marginBottom': '8px'}),
                dcc.Graph(id='therapist-interventions-pie', config={'displayModeBar': False},
                          style={'height': '350px'})
            ], style={'flex': '1', 'minWidth': '300px'}),
            # Patient pie
            html.Div([
                html.H4(id='single-interventions-patient', children="Patient", style={'textAlign': 'center', 'fontSize': '14px', 'marginBottom': '8px'}),
                dcc.Graph(id='patient-interventions-pie', config={'displayModeBar': False},
                          style={'height': '350px'})
            ], style={'flex': '1', 'minWidth': '300px'}),
        ], style={'display': 'flex', 'flexDirection': 'row', 'flexWrap': 'wrap', 'gap': '24px'}),
    ], id='interventions-pie-section', style={**CARD_STYLE, 'display': 'none'})


def create_session_rationale_section():
    """Create the session-level rationale section displaying summary rationale from the rationale sheet"""
    return html.Div([
        html.H2(id='single-rationale-title', children="Sitzungsbegründung", style={'marginBottom': '16px'}),
        html.P(id='single-rationale-subtitle', children="Zusammenfassung der Begründungen für sitzungsbezogene Metriken", style={
            'color': COLORS['gray_500'], 'fontSize': '13px', 'marginBottom': '16px'
        }),
        html.Div(id='session-rationale-content', children=[
            html.Div(id='single-rationale-placeholder', children="Sitzungsdatei hochladen, um Begründungen zu sehen",
                    style={'textAlign': 'center', 'color': COLORS['gray_400'], 'padding': '40px', 'fontSize': '14px'})
        ])
    ], id='session-rationale-section', style={**CARD_STYLE, 'display': 'none'})



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
        html.H2(id='upload-title', children="Mehrere Sitzungen hochladen"),
        html.P(id='upload-subtitle',
            children="Mehrere Sitzungsdateien (xlsx/csv) hochladen, um Verläufe zu analysieren",
            style={'color': COLORS['gray_500'], 'fontSize': '14px', 'marginBottom': '16px'}
        ),
        dcc.Upload(
            id='upload-sessions',
            children=html.Div([
                html.Div(id='upload-drop', children='Sitzungsdateien hier ablegen', style={
                    'fontSize': '15px',
                    'fontWeight': '500',
                    'color': COLORS['gray_700'],
                    'marginBottom': '4px'
                }),
                html.Div(id='upload-formats', children='Akzeptiert .xlsx und .csv Dateien', style={
                    'fontSize': '13px',
                    'color': COLORS['gray_500']
                }),
                html.Div(id='upload-browse', children='oder hier klicken', style={
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
    
    def create_chart_with_rationale(title, chart_id, rationale_id, title_id=None, extra_controls=None):
        badge_id = f'{rationale_id}-badge'
        title_props = {'id': title_id} if title_id else {}
        return html.Div([
            html.Div([
                html.H3(title, style={'fontSize': '16px', 'marginBottom': '12px', 'display': 'inline-block'}, **title_props),
                extra_controls if extra_controls else html.Span()
            ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between', 'flexWrap': 'wrap', 'gap': '12px'}),
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
                        html.H4(id=f'{rationale_id}-title', children="Begründung", style={
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

    def make_legend(selector_id, items):
        """
        items: list of dicts with keys: value, label, color, tooltip
        Renders a custom HTML legend with tooltips. Keeps a hidden dcc.Checklist for callback state.
        """
        default_values = [item['value'] for item in items]

        legend_items = []
        for item in items:
            legend_items.append(
                html.Div([
                    # Colored line + dot swatch
                    html.Span(style={
                        'display': 'inline-block',
                        'width': '24px',
                        'height': '4px',
                        'backgroundColor': item['color'],
                        'borderRadius': '2px',
                        'marginRight': '5px',
                        'verticalAlign': 'middle',
                        'flexShrink': '0',
                    }),
                    html.Span(item['label'], style={'verticalAlign': 'middle'}),
                    html.Span([
                        html.Span('i', style={
                            'display': 'inline-flex',
                            'alignItems': 'center',
                            'justifyContent': 'center',
                            'width': '13px',
                            'height': '13px',
                            'borderRadius': '50%',
                            'backgroundColor': COLORS['gray_300'],
                            'color': COLORS['white'],
                            'fontSize': '9px',
                            'fontStyle': 'italic',
                            'fontWeight': 'bold',
                            'cursor': 'help',
                            'lineHeight': '1',
                        }),
                        html.Span(item['tooltip'], className='legend-popover'),
                    ], className='legend-info-wrap', style={'marginLeft': '4px', 'flexShrink': '0'}),
                ],
                id={'type': 'legend-item', 'selector': selector_id, 'value': item['value']},
                n_clicks=0,
                **{'data-value': item['value'], 'data-selector': selector_id},
                style={
                    'display': 'inline-flex',
                    'alignItems': 'center',
                    'marginRight': '16px',
                    'cursor': 'pointer',
                    'fontSize': '13px',
                    'userSelect': 'none',
                    'opacity': '1',
                    'transition': 'opacity 0.15s',
                })
            )

        return html.Div([
            # Hidden checklist keeps callback state
            dcc.Checklist(
                id=selector_id,
                options=[{'label': '', 'value': item['value']} for item in items],
                value=default_values,
                style={'display': 'none'},
            ),
            # Visible HTML legend
            html.Div(legend_items, id=f'{selector_id}-legend', style={
                'display': 'flex',
                'flexWrap': 'wrap',
                'alignItems': 'center',
                'gap': '4px',
                'padding': '8px 12px',
                'backgroundColor': COLORS['gray_50'],
                'borderRadius': '6px',
                'border': f'1px solid {COLORS["gray_200"]}',
            })
        ])

    # TCCS legend
    tccs_line_selector = make_legend('tccs-line-selector', [
        {'value': 'challenging', 'label': 'Supportiv', 'color': '#2563eb',
         'tooltip': 'Supportiv-stützendes Therapeutenverhalten: Der Therapeut hört aufmerksam zu, zeigt Verständnis und fasst wichtige Punkte zusammen. Skala 0–100 – je höher, desto häufiger dieses Verhalten.'},
        {'value': 'supporting',  'label': 'Herausfordernd', 'color': '#dc2626',
         'tooltip': 'Herausforderndes Therapeutenverhalten: Der Therapeut regt den Patienten an, neue Perspektiven einzunehmen und über bekannte Denk- und Verhaltensmuster hinauszugehen. Skala 0–100.'},
    ])

    # Activation & Engagement legend
    ae_line_selector = make_legend('ae-line-selector', [
        {'value': 'activation', 'label': 'Aktivierung', 'color': '#9333ea',
         'tooltip': 'Verhaltensaktivierung: Wie aktiv war der Patient in der letzten Woche? Skala 0–100 – je höher, desto aktiver laut KI-Assessment.'},
        {'value': 'engagement', 'label': 'Engagement', 'color': '#059669',
         'tooltip': 'Engagement: Wie aktiv bringt sich der Patient in die Therapie ein (Hausaufgaben, Übungen)? Skala 0–100 – je höher, desto stärker das Engagement laut KI-Assessment.'},
        {'value': 'alliance', 'label': 'Allianz', 'color': '#f59e0b',
         'tooltip': 'Allianz (ALLIANCE): Die therapeutische Allianz aus Sicht des Patienten – Vertrauen, gemeinsame Ziele und gegenseitige Unterstützung. Skala 0–5.'},
        {'value': 'epo_1', 'label': 'Wohlbefinden', 'color': '#0ea5e9',
         'tooltip': 'Wohlbefinden (EPO-1): Subjektiv wahrgenommenes emotionales und psychisches Funktionsniveau. Skala 0–100 – je höher, desto besser das Wohlbefinden.'},
    ])

    # CTS legend
    cts_line_selector = make_legend('cts-line-selector', [
        {'value': 'cts_cognitions', 'label': 'Kognitionen', 'color': '#ea580c',
         'tooltip': 'Kognitionen (CTS_Cognitions): Der Therapeut hilft dem Patienten, belastende Gedanken und Grundüberzeugungen zu erkennen. Skala 0–6 – je höher, desto besser gelingt Einsicht oder Veränderung.'},
        {'value': 'cts_behaviours', 'label': 'Verhaltensanalyse', 'color': '#0891b2',
         'tooltip': 'Verhaltensanalyse (CTS_Behaviours): Verhaltensmuster erkennen und neue Verhaltensweisen entwickeln. Skala 0–6 – je höher, desto wirksamer die Verhaltensarbeit.'},
        {'value': 'cts_discovery', 'label': 'Geleitetes Entdecken', 'color': '#db2777',
         'tooltip': 'Geleitetes Entdecken (CTS_Discovery): Der Therapeut arbeitet mit offenen Fragen, um Reflexion und neue Einsichten zu fördern. Skala 0–6.'},
        {'value': 'cts_methods',   'label': 'Änderungsinterventionen', 'color': '#4f46e5',
         'tooltip': 'Änderungsinterventionen (CTS_Methods): Vielfalt und Geschick beim Einsatz von Methoden zur Veränderung von Gedanken oder Verhalten. Skala 0–6.'},
        {'value': 'cts_mean',      'label': 'PT-Kompetenz', 'color': '#111827',
         'tooltip': 'PT-Kompetenz (CTS_MEAN): Gesamtbewertung der therapeutischen Fertigkeiten – Gedanken- und Verhaltensanalyse, geleitetes Entdecken und Änderungsinterventionen. Inklusive Flexibilität und Umgang mit Problemen.'},
    ])

    return html.Div([
        # 1. Challenging vs Supporting
        create_chart_with_rationale("Therapeutenbeitrag (TCCS)", 'tccs-chart', 'tccs-rationale', title_id='chart-title-tccs', extra_controls=tccs_line_selector),

        # 2. Activation vs Engagement
        create_chart_with_rationale("Patientenstatus (Aktivierung & Engagement)", 'activation-engagement-chart', 'activation-rationale', title_id='chart-title-ae', extra_controls=ae_line_selector),

        # 3. CTS Breakdown with line selector
        create_chart_with_rationale("Kompetenzskala (CTS)", 'cts-chart', 'cts-rationale', title_id='chart-title-cts', extra_controls=cts_line_selector),

        # 4. DAG Pipeline
        html.Div([
            html.H3(id='chart-title-dag', children="Psychometrischer Beziehungsgraph (DAG)"),
            html.Iframe(
                id='dag-iframe',
                style={'width': '100%', 'height': '800px', 'border': 'none', 'marginTop': '12px'}
            ),
            html.Div(id='dag-zscores', style={'marginTop': '16px'}),
        ], style={**CARD_STYLE, 'paddingBottom': '200px'})
        
    ], id='sessions-detailed-charts-section', style={'display': 'none', 'marginTop': '24px', 'padding': '0 12px'})


def create_sessions_layout():
    """Create the sessions page layout"""
    return html.Div([
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
        dcc.Store(id='lang', data='de', storage_type='local'),

        # Global navbar
        html.Div([
            html.Span("GAINED", style={
                'fontWeight': '700',
                'fontSize': '18px',
                'color': COLORS['gray_900'],
                'letterSpacing': '0.5px',
            }),
            html.Span(id='main-subtitle-nav', children="Therapiesitzungsanalyse", style={
                'fontSize': '13px',
                'color': COLORS['gray_500'],
                'marginLeft': '12px',
            }),
            html.Div([
                dcc.Link(id='sessions-nav-link-bar', children="→ Einzelsitzungsanalyse", href="/single", style={
                    'color': COLORS['primary'],
                    'fontSize': '13px',
                    'textDecoration': 'none',
                    'fontWeight': '500',
                    'padding': '6px 14px',
                    'backgroundColor': COLORS['white'],
                    'borderRadius': '6px',
                    'border': f'1px solid {COLORS["gray_200"]}',
                    'display': 'inline-block',
                    'marginRight': '16px',
                }),
                dcc.Dropdown(
                    id='lang-dropdown',
                    options=[
                        {'label': '🇩🇪  Deutsch', 'value': 'de'},
                        {'label': '🇬🇧  English', 'value': 'en'},
                    ],
                    value='de',
                    clearable=False,
                    searchable=False,
                    style={'width': '140px', 'fontSize': '13px'},
                ),
            ], style={'display': 'flex', 'alignItems': 'center', 'marginLeft': 'auto'}),
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'padding': '8px 24px',
            'backgroundColor': COLORS['white'],
            'borderBottom': f'1px solid {COLORS["gray_200"]}',
            'position': 'sticky',
            'top': '0',
            'zIndex': '1000',
        }),

        html.Div(id='page-content')
    ])


def create_main_analysis_layout():
    """Create the single session analysis layout (original layout)"""
    return html.Div([
        # Back to main
        html.Div([
            dcc.Link(id='main-nav-link', children="← Mehrsitzungsanalyse", href="/", style={
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
        ], style={'marginBottom': '16px'}),
        
        # Main Content
        create_upload_section(),
        create_audio_section(),
        create_transcript_section(),
        create_interventions_pie_section(),
        create_session_rationale_section(),
        
        # Data Stores
        dcc.Store(id='audio-data'),
        dcc.Store(id='transcript-data'),
        dcc.Store(id='turn-data'),
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
