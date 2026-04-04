"""
Callbacks for GAINED application
"""
import io
import logging

import dash
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import html, dcc, Input, Output, State, callback, no_update
from dash.exceptions import PreventUpdate

from .data_loader import (
    get_patient_sessions, 
    load_session_from_disk,
    load_session_with_rationale,
    encode_audio_to_base64,
    process_audio_upload,
    process_transcript_upload_with_rationale,
)

logger = logging.getLogger(__name__)

import src.dag as dag


def register_callbacks(app):
    """Register all callbacks for the application"""
    
    # Update session dropdown based on patient selection
    @callback(
        Output('session-dropdown', 'options'),
        Input('patient-dropdown', 'value')
    )
    def update_session_dropdown(patient_id):
        return get_patient_sessions(patient_id)
    
    
    # Handle file uploads
    @callback(
        [Output('audio-data', 'data'),
         Output('transcript-data', 'data'),
         Output('rationale-data', 'data'),
         Output('turn-data', 'data'),
         Output('upload-status', 'children')],
        [Input('upload-audio', 'contents'),
         Input('upload-transcript', 'contents')],
        [State('upload-audio', 'filename'),
         State('upload-transcript', 'filename')]
    )
    def handle_file_upload(audio_content, transcript_content, audio_filename, transcript_filename):
        ctx = dash.callback_context
        triggered_id = None
        if ctx.triggered:
            triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        if not triggered_id:
            raise PreventUpdate
        
        audio_output = no_update
        transcript_output = no_update
        rationale_output = no_update
        turn_output = no_update
        status_messages = []
        
        if triggered_id == 'upload-audio' and audio_content and audio_filename:
            new_audio_data, audio_msg = process_audio_upload(audio_content, audio_filename)
            if new_audio_data:
                audio_output = new_audio_data
            if audio_msg:
                status_messages.append(audio_msg)
        elif triggered_id == 'upload-transcript' and transcript_content and transcript_filename:
            # Use the new function that also loads rationale
            new_transcript_data, new_rationale_data, new_turn_data, transcript_msg = process_transcript_upload_with_rationale(
                transcript_content, transcript_filename
            )
            if new_transcript_data:
                transcript_output = new_transcript_data
            if new_rationale_data:
                rationale_output = new_rationale_data
            if new_turn_data:
                turn_output = new_turn_data
            if transcript_msg:
                status_messages.append(transcript_msg)
        else:
            raise PreventUpdate
        
        status_div = no_update
        if status_messages:
            status_div = html.Div([
                html.Div(msg, style={
                    'color': '#059669' if '✅' in msg else '#dc2626' if '❌' in msg else '#d97706'
                }) for msg in status_messages
            ])
        
        return audio_output, transcript_output, rationale_output, turn_output, status_div
    
    
    # Load existing session data
    @callback(
        [Output('audio-data', 'data', allow_duplicate=True),
         Output('transcript-data', 'data', allow_duplicate=True),
         Output('rationale-data', 'data', allow_duplicate=True),
         Output('turn-data', 'data', allow_duplicate=True),
         Output('audio-section', 'style'),
         Output('transcript-section', 'style')],
        Input('load-button', 'n_clicks'),
        [State('patient-dropdown', 'value'),
         State('session-dropdown', 'value')],
        prevent_initial_call=True
    )
    def load_session_data(n_clicks, patient_id, session_idx):
        if n_clicks == 0:
            raise PreventUpdate
        
        visible_style = {
            'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px',
            'marginBottom': '20px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'display': 'block'
        }
        
        # Load with rationale support
        transcript_data, rationale_data, turn_data = load_session_with_rationale(patient_id, session_idx)
        
        # Load audio separately (using old function for now)
        audio_data, _ = load_session_from_disk(patient_id, session_idx)
        
        if audio_data or transcript_data:
            return audio_data, transcript_data, rationale_data, turn_data, visible_style, visible_style
        
        raise PreventUpdate
    
    
    # Show/hide sections when files are uploaded or loaded
    @callback(
        [Output('audio-section', 'style', allow_duplicate=True),
         Output('transcript-section', 'style', allow_duplicate=True)],
        [Input('audio-data', 'data'),
         Input('transcript-data', 'data')],
        prevent_initial_call=True
    )
    def update_section_visibility(audio_data, transcript_data):
        visible_style = {
            'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px',
            'marginBottom': '20px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'display': 'block'
        }
        hidden_style = {'display': 'none'}
        
        audio_visible = visible_style if audio_data else hidden_style
        transcript_visible = visible_style if transcript_data else hidden_style
        
        return audio_visible, transcript_visible
    
    
    # Display audio player
    @callback(
        Output('audio-player-container', 'children'),
        Input('audio-data', 'data')
    )
    def display_audio_player(audio_data):
        if not audio_data:
            return html.Div("No audio loaded", style={
                'color': '#9ca3af', 'textAlign': 'center', 'padding': '20px', 'fontSize': '13px'
            })
        
        audio_src = encode_audio_to_base64(audio_data)
        
        return html.Div([
            html.P(f"Audio loaded: {audio_data.get('filename', 'Unknown')}", 
                   style={'color': '#059669', 'marginBottom': '12px', 'fontWeight': '500', 'fontSize': '13px'}),
            html.Audio(
                id='audio-element',
                src=audio_src,
                controls=True,
                autoPlay=False,
                preload='auto',
                style={'width': '100%', 'marginBottom': '12px'}
            ),
            html.P("Waveform visualization below", 
                   style={'color': '#9ca3af', 'fontSize': '12px', 'marginBottom': '4px'}),
        ])
    
    
    # Display transcription - NEW LIST FORMAT
    @callback(
        Output('transcription-display', 'children'),
        Input('transcript-data', 'data')
    )
    def display_transcription(transcript_data):
        if not transcript_data:
            return html.Div("No transcript loaded")
        
        try:
            df = pd.read_json(io.StringIO(transcript_data), orient='split')
            
            # Check if required columns exist
            if 'text' not in df.columns and 'segment_text' not in df.columns:
                return html.Div("Transcript data doesn't have expected format")
            
            text_col = 'text' if 'text' in df.columns else 'segment_text'
            speaker_col = 'speaker' if 'speaker' in df.columns else None
            
            def resolve_speaker(row):
                speaker_value = row.get('speaker') if speaker_col else None
                if isinstance(speaker_value, str) and speaker_value.strip():
                    return speaker_value
                
                if 'pred' in df.columns:
                    pred_val = row.get('pred')
                    if pd.isna(pred_val):
                        return 'Unknown'
                    
                    try:
                        numeric_pred = float(pred_val)
                        if numeric_pred == 1:
                            return 'Therapist'
                        if numeric_pred == 0:
                            return 'Patient'
                    except (TypeError, ValueError):
                        pass
                    
                    pred_str = str(pred_val).strip().lower()
                    if pred_str in {'1', 'therapist', 't', 'true'}:
                        return 'Therapist'
                    if pred_str in {'0', 'patient', 'p', 'false'}:
                        return 'Patient'
                
                return speaker_value or 'Unknown'
            
            # Create simple list of transcript segments
            segments = []
            for idx, row in df.iterrows():
                speaker = resolve_speaker(row)
                text = row.get(text_col, '')
                start_time = row.get('start', 0) if 'start' in df.columns else 0
                end_time = row.get('end', 0) if 'end' in df.columns else 0
                
                # Determine if therapist or patient
                is_therapist = 'therapist' in speaker.lower()
                
                # Segment styling
                therapist_color = '#2563eb'
                patient_color = '#059669'
                segment_color = therapist_color if is_therapist else patient_color
                
                segment_style = {
                    'padding': '12px',
                    'marginBottom': '8px',
                    'borderRadius': '6px',
                    'cursor': 'pointer',
                    'transition': 'all 0.15s ease',
                    'borderLeft': f'3px solid {segment_color}',
                    'backgroundColor': '#ffffff',
                    'boxShadow': '0 1px 2px 0 rgba(0, 0, 0, 0.05)',
                }
                
                segment = html.Div([
                    html.Div([
                        html.Span(speaker, style={
                            'fontSize': '11px',
                            'fontWeight': '600',
                            'color': segment_color,
                            'textTransform': 'uppercase',
                            'letterSpacing': '0.025em',
                            'marginRight': '12px'
                        }),
                        html.Span(f"{start_time:.1f}s – {end_time:.1f}s", style={
                            'fontSize': '11px',
                            'color': '#9ca3af',
                        })
                    ], style={'marginBottom': '6px'}),
                    html.Div(text, style={
                        'fontSize': '13px',
                        'lineHeight': '1.6',
                        'color': '#374151'
                    })
                ],
                id={'type': 'transcript-segment', 'index': idx},
                n_clicks=0,
                style=segment_style,
                className='transcript-segment',
                **{'data-start': start_time, 'data-end': end_time, 'data-speaker': speaker, 'data-index': idx}
                )
                segments.append(segment)
            
            return segments
        
        except Exception as e:
            logger.exception("Error displaying transcript")
            return html.Div(f"Error loading transcript: {str(e)}")
    
    
    # Interventions pie charts — Therapist and Patient
    @callback(
        [Output('therapist-interventions-pie', 'figure'),
         Output('patient-interventions-pie', 'figure'),
         Output('interventions-pie-section', 'style')],
        [Input('transcript-data', 'data'),
         Input('turn-data', 'data')]
    )
    def update_interventions_pies(transcript_data, turn_data):
        hidden = {'display': 'none'}
        visible = {
            'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px',
            'marginBottom': '20px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'display': 'block'
        }
        empty = (go.Figure(), go.Figure(), hidden)
        
        target_data = turn_data if turn_data else transcript_data
        if not target_data:
            return empty
        
        try:
            df = pd.read_json(io.StringIO(target_data), orient='split')
            
            # Intervention column mappings (column aliases → display label)
            therapist_interventions = {
                'SP': 'Supporting (SP)',
                'supporting': 'Supporting (SP)',
                'TCCS_SP': 'Supporting (SP)',
                'C': 'Challenging (C)',
                'challenging': 'Challenging (C)',
                'TCCS_C': 'Challenging (C)',
                'G': 'Guidance (G)',
                'guidance': 'Guidance (G)',
            }
            patient_interventions = {
                'S': 'Safety (S)',
                'safety': 'Safety (S)',
                'TR': 'Tolerable Risk (TR)',
                'tolerable_risk': 'Tolerable Risk (TR)',
                'A': 'Ambivalence (A)',
                'ambivalence': 'Ambivalence (A)',
            }
            
            # Build a case-insensitive column lookup
            col_lookup = {str(c).lower(): c for c in df.columns}
            
            def build_pie(interventions_map, title_text, colors_list, is_therapist=True):
                labels = []
                values = []
                seen_labels = set()
                
                # Try to find actual columns first
                for col, label in interventions_map.items():
                    actual_col = col_lookup.get(col.lower())
                    if actual_col and label not in seen_labels:
                        col_data = pd.to_numeric(df[actual_col], errors='coerce').dropna()
                        if not col_data.empty:
                            labels.append(label)
                            values.append(round(col_data.mean(), 3))
                            seen_labels.add(label)
                
                # Mock data removed to avoid displaying fake proportions when actual intervention probabilities are missing.
                
                if not labels:
                    fig = go.Figure()
                    fig.add_annotation(text="No data available", showarrow=False,
                                       font=dict(size=14, color='#9ca3af'))
                    fig.update_layout(template='plotly_white', height=300,
                                       margin=dict(l=20, r=20, t=20, b=20))
                    return fig
                
                colors = colors_list[:len(labels)]
                fig = go.Figure(go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.45,
                    textinfo='label+percent',
                    textposition='outside',
                    marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
                    hovertemplate='<b>%{label}</b><br>Mean: %{value:.3f}<br>Share: %{percent}<extra></extra>'
                ))
                fig.update_layout(
                    template='plotly_white',
                    height=300,
                    margin=dict(l=20, r=20, t=20, b=20),
                    showlegend=False,
                )
                return fig
            
            therapist_colors = ['#2563eb', '#dc2626', '#f59e0b']  # Blue, Red, Amber
            patient_colors = ['#059669', '#9333ea', '#6b7280']    # Green, Purple, Gray
            
            fig_t = build_pie(therapist_interventions, 'Therapist', therapist_colors, is_therapist=True)
            fig_p = build_pie(patient_interventions, 'Patient', patient_colors, is_therapist=False)
            
            # Always show the section when transcript data is loaded
            return fig_t, fig_p, visible
            
        except Exception as e:
            logger.exception("Error creating interventions pie charts")
            return empty


    # Update field plots selector dropdown
    @callback(
        [Output('field-plots-selector', 'options'),
         Output('field-plots-selector', 'value')],
        Input('transcript-data', 'data')
    )
    def update_field_plots_selector(transcript_data):
        if not transcript_data:
            return [], []

        try:
            df = pd.read_json(io.StringIO(transcript_data), orient='split')

            # Find numeric fields that can be plotted
            excluded_cols = {'segment_id', 'index', 'start', 'end'}
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            plotable_fields = [col for col in numeric_cols if col.lower() not in excluded_cols]

            # Friendly labels for known metric columns
            friendly_labels = {
                'TCCS_SP': 'Supportive (TCCS_SP)',
                'TCCS_C': 'Challenging (TCCS_C)',
                'challenging': 'Supportive (TCCS_SP)',
                'supporting': 'Challenging (TCCS_C)',
                'activation_mean': 'Activation',
                'engagement_mean': 'Engagement',
                'CTS_Cognitions': 'CTS Cognitions',
                'CTS_Behaviours': 'CTS Behaviours',
                'CTS_Discovery': 'CTS Discovery',
                'CTS_Methods': 'CTS Methods',
            }

            options = [{'label': friendly_labels.get(col, col), 'value': col} for col in plotable_fields]

            # Default selection: prefer 'challenging' and 'supporting', fall back to TCCS_SP/TCCS_C
            default_cols = ['challenging', 'supporting', 'TCCS_SP', 'TCCS_C']
            default_value = [col for col in default_cols if col in plotable_fields][:2]

            return options, default_value

        except Exception as e:
            logger.exception("Error updating field plots selector")
            return [], []
    
    
    # Display field plots with rationale
    @callback(
        Output('field-plots-container', 'children'),
        [Input('field-plots-selector', 'value'),
         Input('transcript-data', 'data'),
         Input('rationale-data', 'data')]
    )
    def display_field_plots(selected_fields, transcript_data, rationale_data):
        if not transcript_data or not selected_fields:
            return html.Div("Select fields above to display plots with rationale", 
                          style={'textAlign': 'center', 'color': '#9ca3af', 'padding': '40px'})
        
        # Limit to 4 fields
        selected_fields = selected_fields[:4] if isinstance(selected_fields, list) else [selected_fields]
        
        try:
            df = pd.read_json(io.StringIO(transcript_data), orient='split')
            
            # Parse rationale data if available
            rationale_dict = {}
            if rationale_data:
                for col_name, rationale_json in rationale_data.items():
                    try:
                        rationale_df = pd.read_json(io.StringIO(rationale_json), orient='split')
                        rationale_dict[col_name] = rationale_df
                    except Exception:
                        logger.warning(f"Could not parse rationale for {col_name}")
            
            # Determine x-axis: prefer time in minutes when available
            if 'start' in df.columns:
                x_data = (df['start'] / 60).round(2)
                x_label = 'Time (minutes)'
            elif 'segment_id' in df.columns:
                x_data = df['segment_id']
                x_label = 'Segment'
            else:
                x_data = df.index
                x_label = 'Index'
            
            # Create plots for each selected field
            plot_components = []
            for field in selected_fields:
                if field not in df.columns:
                    continue
                
                # Get rationale for this field if available (per row)
                rationale_per_row = None
                if field in rationale_dict:
                    rationale_df = rationale_dict[field]
                    
                    # Try to match by segment_id or index if available
                    match_key = None
                    if 'segment_id' in df.columns and 'segment_id' in rationale_df.columns:
                        match_key = 'segment_id'
                    elif 'index' in rationale_df.columns:
                        match_key = 'index'
                    
                    if match_key:
                        # Merge rationale with main data by match key
                        merged = df.merge(
                            rationale_df[[match_key, 'data']] if 'data' in rationale_df.columns else rationale_df[[match_key, field]],
                            on=match_key,
                            how='left',
                            suffixes=('', '_rationale')
                        )
                        rationale_col = 'data' if 'data' in merged.columns else field
                        if rationale_col in merged.columns:
                            rationale_per_row = merged[rationale_col].astype(str).tolist()
                    else:
                        # No match key, assume same order
                        # Check for 'data' column first
                        if 'data' in rationale_df.columns:
                            rationale_per_row = rationale_df['data'].astype(str).tolist()
                        elif field in rationale_df.columns:
                            rationale_per_row = rationale_df[field].astype(str).tolist()
                        else:
                            # Use first column
                            rationale_per_row = rationale_df.iloc[:, 0].astype(str).tolist()
                        
                        # Ensure same length as main data
                        if len(rationale_per_row) != len(df):
                            rationale_per_row = None
                
                # Prepare custom hover text with field information and rationale
                hover_texts = []
                for idx in range(len(df)):
                    # Format the main field value
                    field_value = df[field].iloc[idx]
                    if pd.notna(field_value) and isinstance(field_value, (int, float)):
                        hover_parts = [f"<b>{field}</b>: {field_value:.4f}"]
                    else:
                        hover_parts = [f"<b>{field}</b>: {field_value}"]
                    
                    # Add segment info if available
                    if 'segment_id' in df.columns:
                        hover_parts.append(f"<b>Segment</b>: {df['segment_id'].iloc[idx]}")
                    if 'start' in df.columns and 'end' in df.columns:
                        hover_parts.append(f"<b>Time</b>: {df['start'].iloc[idx]:.1f}s - {df['end'].iloc[idx]:.1f}s")
                    
                    # Add other relevant field information (e.g., emotions, speaker, etc.)
                    info_fields = ['emotion', 'emotions', 'sentiment', 'speaker']
                    for info_field in info_fields:
                        if info_field in df.columns:
                            value = df[info_field].iloc[idx]
                            if pd.notna(value) and str(value).strip():
                                hover_parts.append(f"<b>{info_field}</b>: {value}")
                    
                    # Add text preview (truncated)
                    if 'text' in df.columns:
                        text_value = df['text'].iloc[idx]
                        if pd.notna(text_value) and str(text_value).strip():
                            text_str = str(text_value)
                            if len(text_str) > 100:
                                text_str = text_str[:100] + "..."
                            hover_parts.append(f"<b>Text</b>: {text_str}")
                    
                    # Add rationale if available
                    if rationale_per_row and idx < len(rationale_per_row):
                        rationale_val = rationale_per_row[idx]
                        if pd.notna(rationale_val) and str(rationale_val).strip() and str(rationale_val).lower() != 'nan':
                            rationale_str = str(rationale_val).strip()
                            # Truncate long rationale in tooltip
                            if len(rationale_str) > 200:
                                rationale_str = rationale_str[:200] + "..."
                            hover_parts.append(f"<br><b>Rationale</b>: {rationale_str}")
                    
                    hover_texts.append("<br>".join(hover_parts))
                
                # Create plot with enhanced tooltips
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=df[field],
                    mode='lines+markers',
                    name=field,
                    line=dict(width=2),
                    marker=dict(size=6),
                    hovertemplate='%{text}<extra></extra>',
                    text=hover_texts
                ))
                
                fig.update_layout(
                    title=field,
                    xaxis_title=x_label,
                    yaxis_title='Value',
                    hovermode='closest',
                    template='plotly_white',
                    height=350,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                
                # Get rationale for this field if available (for display below plot)
                rationale_text = None
                if field in rationale_dict:
                    rationale_df = rationale_dict[field]
                    
                    # Check for 'data' column first
                    if 'data' in rationale_df.columns:
                        rationale_values = rationale_df['data'].dropna().astype(str).tolist()
                        if rationale_values:
                            # Filter out 'nan' strings
                            rationale_values = [v for v in rationale_values if v.lower() != 'nan' and v.strip()]
                            if rationale_values:
                                rationale_text = '\n'.join(rationale_values) if len(rationale_values) > 1 else rationale_values[0]
                    elif field in rationale_df.columns:
                        # Get non-null values from the rationale column
                        rationale_values = rationale_df[field].dropna().astype(str).tolist()
                        if rationale_values:
                            rationale_values = [v for v in rationale_values if v.lower() != 'nan' and v.strip()]
                            if rationale_values:
                                rationale_text = '\n'.join(rationale_values) if len(rationale_values) > 1 else rationale_values[0]
                    else:
                        # Try to find a text/description column in rationale
                        text_cols = [col for col in rationale_df.columns 
                                   if any(keyword in col.lower() for keyword in ['text', 'description', 'rationale', 'reason', 'explanation'])]
                        if text_cols:
                            # Combine all rationale text
                            rationale_values = rationale_df[text_cols[0]].dropna().astype(str).tolist()
                            if rationale_values:
                                rationale_values = [v for v in rationale_values if v.lower() != 'nan' and v.strip()]
                                if rationale_values:
                                    rationale_text = '\n'.join(rationale_values) if len(rationale_values) > 1 else rationale_values[0]
                        elif len(rationale_df.columns) > 0:
                            # Use first column if no obvious text column
                            rationale_values = rationale_df.iloc[:, 0].dropna().astype(str).tolist()
                            if rationale_values:
                                rationale_values = [v for v in rationale_values if v.lower() != 'nan' and v.strip()]
                                if rationale_values:
                                    rationale_text = '\n'.join(rationale_values) if len(rationale_values) > 1 else rationale_values[0]
                
                # Create component for this field
                field_component = html.Div([
                    dcc.Graph(figure=fig, id={'type': 'field-plot', 'field': field}),
                    html.Div([
                        html.H4("Rationale", style={'fontSize': '13px', 'fontWeight': '600', 'marginBottom': '8px', 'color': '#374151'}),
                        html.Div(
                            rationale_text if rationale_text else "No rationale available for this field",
                            style={
                                'padding': '12px 16px',
                                'backgroundColor': '#f9fafb',
                                'borderRadius': '6px',
                                'borderLeft': '3px solid #2563eb',
                                'fontSize': '13px',
                                'lineHeight': '1.6',
                                'color': '#374151' if rationale_text else '#9ca3af',
                                'fontStyle': 'italic' if not rationale_text else 'normal',
                                'whiteSpace': 'pre-line'
                            }
                        )
                    ], style={'marginTop': '12px', 'marginBottom': '24px'})
                ], style={
                    'marginBottom': '24px',
                    'padding': '16px',
                    'backgroundColor': '#ffffff',
                    'borderRadius': '8px',
                    'border': '1px solid #e5e7eb',
                    'boxShadow': '0 1px 3px 0 rgba(0, 0, 0, 0.1)'
                })
                
                plot_components.append(field_component)
            
            if not plot_components:
                return html.Div("No valid fields selected", 
                              style={'textAlign': 'center', 'color': '#9ca3af', 'padding': '40px'})
            
            return plot_components
        
        except Exception as e:
            logger.exception("Error creating field plots")
            return html.Div(f"Error displaying plots: {str(e)}", 
                          style={'textAlign': 'center', 'color': '#dc2626', 'padding': '40px', 'fontSize': '13px'})


    # Show/hide session rationale section
    @callback(
        Output('session-rationale-section', 'style'),
        Input('rationale-data', 'data')
    )
    def update_session_rationale_visibility(rationale_data):
        visible_style = {
            'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px',
            'marginBottom': '20px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'display': 'block'
        }
        hidden_style = {'display': 'none'}
        
        if rationale_data and '_session_rationale' in rationale_data:
            return visible_style
        return hidden_style


    # Populate session rationale content
    @callback(
        Output('session-rationale-content', 'children'),
        Input('rationale-data', 'data')
    )
    def update_session_rationale_content(rationale_data):
        if not rationale_data or '_session_rationale' not in rationale_data:
            return html.Div("No session rationale available",
                          style={'textAlign': 'center', 'color': '#9ca3af', 'padding': '40px'})
        
        session_rationale = rationale_data['_session_rationale']
        
        if not session_rationale:
            return html.Div("No session rationale available",
                          style={'textAlign': 'center', 'color': '#9ca3af', 'padding': '40px'})
        
        # Display labels for known metrics
        metric_labels = {
            'activation': 'Activation',
            'engagement': 'Engagement',
            'CTS_Cognitions': 'CTS - Cognitions',
            'CTS_Behaviours': 'CTS - Behaviours', 
            'CTS_Discovery': 'CTS - Discovery',
            'CTS_Methods': 'CTS - Methods'
        }
        
        cards = []
        for metric_key, rationale_text in session_rationale.items():
            label = metric_labels.get(metric_key, metric_key)
            
            card = html.Div([
                html.H4(label, style={
                    'fontSize': '14px', 'fontWeight': '600', 'marginBottom': '8px',
                    'color': '#1f2937'
                }),
                html.Div(rationale_text, style={
                    'padding': '12px 16px',
                    'backgroundColor': '#f9fafb',
                    'borderRadius': '6px',
                    'borderLeft': '3px solid #2563eb',
                    'fontSize': '13px',
                    'lineHeight': '1.6',
                    'color': '#374151',
                    'whiteSpace': 'pre-line'
                })
            ], style={'marginBottom': '16px'})
            
            cards.append(card)
        
        return cards
    
    
    # Show/hide field plots section when transcript data is available
    @callback(
        Output('field-plots-section', 'style'),
        Input('transcript-data', 'data')
    )
    def update_field_plots_section_visibility(transcript_data):
        visible_style = {
            'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px',
            'marginBottom': '20px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'display': 'block'
        }
        hidden_style = {'display': 'none'}
        
        return visible_style if transcript_data else hidden_style


    # Routing callback
    @callback(
        Output('page-content', 'children'),
        Input('url', 'pathname')
    )
    def display_page(pathname):
        from .ui_components import create_main_analysis_layout, create_sessions_layout
        
        if pathname == '/sessions':
            return create_sessions_layout()
        else:
            return create_main_analysis_layout()


    # Handle sessions file upload
    @callback(
        [Output('sessions-data', 'data'),
         Output('sessions-upload-status', 'children')],
        Input('upload-sessions', 'contents'),
        State('upload-sessions', 'filename')
    )
    def handle_sessions_upload(contents, filenames):
        if not contents or not filenames:
            raise PreventUpdate
        
        from .data_loader import process_sessions_upload
        
        sessions_json, status_msg = process_sessions_upload(contents, filenames)
        
        status_div = html.Div(status_msg, style={
            'color': '#059669' if '✅' in status_msg else '#dc2626'
        })
        
        return sessions_json, status_div


    # Show/hide detailed section when data loads
    @callback(
        Output('sessions-detailed-charts-section', 'style'),
        Input('sessions-data', 'data')
    )
    def update_sessions_section_visibility(sessions_json):
        if not sessions_json:
            return {'display': 'none'}
        try:
            df = pd.read_json(io.StringIO(sessions_json), orient='split')
            if df.empty:
                return {'display': 'none'}
            return {'display': 'block', 'marginTop': '24px', 'padding': '0 12px'}
        except Exception:
            return {'display': 'none'}

    def _build_sessions_df_and_trace_fn(sessions_json):
        """Helper: parse sessions JSON, return (df, x_data, x_label, x_axis_cfg, create_trace) or None."""
        if not sessions_json:
            return None
        df = pd.read_json(io.StringIO(sessions_json), orient='split')
        if df.empty:
            return None
        if 'session_number' in df.columns:
            df = df.sort_values('session_number').reset_index(drop=True)
        x_data = [int(s) for s in df['session_number']] if 'session_number' in df.columns else list(range(1, len(df) + 1))
        x_label = 'Session'
        x_axis_cfg = dict(type='linear', tickmode='array', tickvals=x_data, ticktext=[f'S{n}' for n in x_data])

        def create_trace(metric_name, label, color, dash_style=None):
            if metric_name not in df.columns:
                return None
            hover_texts = []
            for _, row in df.iterrows():
                val = row.get(metric_name)
                session_num = row.get('session_number', '')
                duration = row.get('session_duration_min', '')
                if pd.isna(val):
                    hover_texts.append('')
                else:
                    hover_text = f"<b>Session {int(session_num)}</b>"
                    if pd.notna(duration):
                        hover_text += f" ({duration} min)"
                    hover_text += f"<br>{label}: {val}"
                    hover_texts.append(hover_text)
            line_config = dict(width=3, color=color)
            if dash_style:
                line_config['dash'] = dash_style
            return go.Scatter(
                x=x_data, y=df[metric_name], mode='lines+markers', name=label,
                line=line_config, marker=dict(size=8),
                hovertemplate='%{text}<extra></extra>', text=hover_texts
            )

        return df, x_data, x_label, x_axis_cfg, create_trace

    # Update TCCS chart with line selection
    @callback(
        Output('tccs-chart', 'figure'),
        [Input('sessions-data', 'data'),
         Input('tccs-line-selector', 'value')]
    )
    def update_tccs_chart(sessions_json, selected_lines):
        if not selected_lines:
            selected_lines = []
        result = _build_sessions_df_and_trace_fn(sessions_json)
        if result is None:
            return go.Figure()
        try:
            df, x_data, x_label, x_axis_cfg, create_trace = result
            fig = go.Figure()
            tccs_metrics = [
                ('challenging', 'Supportive (TCCS_SP)', '#2563eb'),
                ('supporting', 'Challenging (TCCS_C)', '#dc2626'),
            ]
            for metric, label, color in tccs_metrics:
                if metric in selected_lines:
                    t = create_trace(metric, label, color)
                    if t:
                        fig.add_trace(t)
            fig.update_layout(
                xaxis_title=x_label, xaxis=x_axis_cfg,
                yaxis_title='Score', yaxis=dict(range=[0, 1]),
                hovermode='closest', template='plotly_white',
                margin=dict(l=40, r=40, t=30, b=40),
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
            )
            return fig
        except Exception:
            logger.exception("Error updating TCCS chart")
            return go.Figure()

    # Update Activation/Engagement chart with line selection
    @callback(
        Output('activation-engagement-chart', 'figure'),
        [Input('sessions-data', 'data'),
         Input('ae-line-selector', 'value')]
    )
    def update_ae_chart(sessions_json, selected_lines):
        if not selected_lines:
            selected_lines = []
        result = _build_sessions_df_and_trace_fn(sessions_json)
        if result is None:
            return go.Figure()
        try:
            df, x_data, x_label, x_axis_cfg, create_trace = result
            fig = go.Figure()
            ae_metrics = [
                ('activation', 'Activation', '#9333ea'),
                ('engagement', 'Engagement', '#059669'),
            ]
            for metric, label, color in ae_metrics:
                if metric in selected_lines:
                    t = create_trace(metric, label, color)
                    if t:
                        fig.add_trace(t)
            fig.update_layout(
                xaxis_title=x_label, xaxis=x_axis_cfg,
                yaxis_title='Score', yaxis=dict(range=[0, 100]),
                hovermode='closest', template='plotly_white',
                margin=dict(l=40, r=40, t=30, b=40),
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
            )
            return fig
        except Exception:
            logger.exception("Error updating Activation/Engagement chart")
            return go.Figure()


    # Update CTS chart with line selection
    @callback(
        Output('cts-chart', 'figure'),
        [Input('sessions-data', 'data'),
         Input('cts-line-selector', 'value')]
    )
    def update_cts_chart(sessions_json, selected_lines):
        if not sessions_json:
            return go.Figure()
        
        if not selected_lines:
            selected_lines = []
        
        try:
            df = pd.read_json(io.StringIO(sessions_json), orient='split')
            
            if df.empty:
                return go.Figure()
            
            # Sort by session number
            if 'session_number' in df.columns:
                df = df.sort_values('session_number').reset_index(drop=True)

            # Use numeric session numbers as x values
            if 'session_number' in df.columns:
                x_data = [int(s) for s in df['session_number']]
            else:
                x_data = list(range(1, len(df) + 1))
            x_label = 'Session'

            # Calculate overall mean of all CTS metrics across all sessions (single value)
            cts_cols = ['cts_cognitions', 'cts_behaviours', 'cts_discovery', 'cts_methods']
            existing_cts_cols = [c for c in cts_cols if c in df.columns]
            overall_mean = None
            if existing_cts_cols:
                all_values = df[existing_cts_cols].values.flatten()
                all_values = all_values[~pd.isna(all_values)]
                if len(all_values) > 0:
                    overall_mean = float(all_values.mean())

            # Helper to create trace
            def create_trace(metric_name, label, color, dash_style=None):
                if metric_name not in df.columns:
                    return None

                hover_texts = []
                for _, row in df.iterrows():
                    val = row.get(metric_name)
                    session_num = row.get('session_number', '')
                    duration = row.get('session_duration_min', '')
                    if pd.isna(val):
                        hover_texts.append('')
                    else:
                        hover_text = f"<b>Session {int(session_num)}</b>"
                        if pd.notna(duration):
                            hover_text += f" ({duration} min)"
                        hover_text += f"<br>{label}: {val:.2f}"
                        hover_texts.append(hover_text)

                line_config = dict(width=3, color=color)
                if dash_style:
                    line_config['dash'] = dash_style

                return go.Scatter(
                    x=x_data,
                    y=df[metric_name],
                    mode='lines+markers',
                    name=label,
                    line=line_config,
                    marker=dict(size=8),
                    hovertemplate='%{text}<extra></extra>',
                    text=hover_texts
                )

            # Axis config
            x_axis_cfg = dict(
                type='linear',
                tickmode='array',
                tickvals=x_data,
                ticktext=[f'S{n}' for n in x_data]
            )
            y_cts = dict(range=[0, 7])  # CTS-R scale 0–7

            # CTS Chart with selectable lines
            fig_cts = go.Figure()
            
            cts_metrics = [
                ('cts_cognitions', 'Cognitions', '#ea580c'),     # Orange
                ('cts_behaviours', 'Behaviours', '#0891b2'),     # Cyan
                ('cts_discovery', 'Discovery', '#db2777'),       # Pink
                ('cts_methods', 'Methods', '#4f46e5'),           # Indigo
            ]
            
            for metric, label, color in cts_metrics:
                if metric in selected_lines:
                    t = create_trace(metric, label, color)
                    if t:
                        fig_cts.add_trace(t)
            
            # Add horizontal mean line if selected and data exists
            if 'cts_mean' in selected_lines and overall_mean is not None:
                x_range = [min(x_data), max(x_data)]
                fig_cts.add_trace(go.Scatter(
                    x=x_range,
                    y=[overall_mean, overall_mean],
                    mode='lines',
                    name=f'Mean ({overall_mean:.2f})',
                    line=dict(width=2, color='#111827', dash='dash'),
                    hovertemplate=f'<b>Overall Mean</b>: {overall_mean:.2f}<extra></extra>'
                ))
            
            fig_cts.update_layout(
                xaxis_title=x_label,
                xaxis=x_axis_cfg,
                yaxis_title='Score',
                yaxis=y_cts,
                hovermode='closest',
                template='plotly_white',
                margin=dict(l=40, r=40, t=30, b=40),
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
            )
            
            return fig_cts
            
        except Exception as e:
            logger.exception("Error updating CTS chart")
            return go.Figure()


    # Update rationale text boxes and session badges
    @callback(
        [Output('tccs-rationale', 'children'),
         Output('activation-rationale', 'children'),
         Output('cts-rationale', 'children'),
         Output('tccs-rationale-badge', 'children'),
         Output('activation-rationale-badge', 'children'),
         Output('cts-rationale-badge', 'children')],
        [Input('tccs-chart', 'clickData'),
         Input('activation-engagement-chart', 'clickData'),
         Input('cts-chart', 'clickData'),
         Input('sessions-data', 'data')]
    )
    def update_rationale_boxes(tccs_click, ae_click, cts_click, sessions_json):
        no_badge = "No session"
        if not sessions_json:
            return "No data", "No data", "No data", no_badge, no_badge, no_badge
            
        try:
            df = pd.read_json(io.StringIO(sessions_json), orient='split')
            if df.empty:
                return "No data", "No data", "No data", no_badge, no_badge, no_badge
            
            # Sort by session number
            if 'session_number' in df.columns:
                df = df.sort_values('session_number').reset_index(drop=True)
            
            # Helper to extract rationale text and session label
            def get_rationale(session_num, metrics):
                if session_num is None:
                    row = df.iloc[-1]
                elif 'session_number' in df.columns:
                    match = df[df['session_number'] == session_num]
                    if match.empty:
                        return f"Session {session_num} not found", f"Session {session_num}"
                    row = match.iloc[0]
                else:
                    if session_num >= len(df):
                        return "Session not found", no_badge
                    row = df.iloc[session_num]
                
                snum = int(row.get('session_number', session_num or 0))
                badge = f"Session {snum}"
                
                text_parts = []
                for metric, label in metrics:
                    val = row.get(metric)
                    rationale = row.get(f"{metric}_rationale")
                    
                    if pd.notna(val) or (pd.notna(rationale) and str(rationale).strip()):
                        text_parts.append(f"\n[{label}]")
                        if pd.notna(val):
                            text_parts.append(f"Score: {val}")
                        if pd.notna(rationale) and str(rationale).strip():
                            text_parts.append(f"Rationale: {str(rationale).strip()}")
                        else:
                            text_parts.append("Rationale: N/A")
                
                content = "\n".join(text_parts).strip() if text_parts else "No rationale available."
                return content, badge

            # Extract session number from clickData (x value)
            def get_session_from_click(click_data):
                if click_data and 'points' in click_data:
                    x_val = click_data['points'][0].get('x')
                    if isinstance(x_val, str) and x_val.startswith('S'):
                        try:
                            return int(x_val[1:])
                        except ValueError:
                            pass
                    return x_val
                return None
            
            # TCCS
            tccs_text, tccs_badge = get_rationale(get_session_from_click(tccs_click), [
                ('challenging', 'Supportive (TCCS_SP)'),
                ('supporting', 'Challenging (TCCS_C)')
            ])
            
            # Activation/Engagement
            ae_text, ae_badge = get_rationale(get_session_from_click(ae_click), [
                ('activation', 'Activation'),
                ('engagement', 'Engagement')
            ])
            
            # CTS
            cts_text, cts_badge = get_rationale(get_session_from_click(cts_click), [
                ('cts_cognitions', 'Cognitions'),
                ('cts_behaviours', 'Behaviours'),
                ('cts_discovery', 'Discovery'),
                ('cts_methods', 'Methods')
            ])
            
            return tccs_text, ae_text, cts_text, tccs_badge, ae_badge, cts_badge
            
        except Exception as e:
            logger.exception("Error updating rationale boxes")
            err = f"Error: {e}"
            return err, err, err, no_badge, no_badge, no_badge


    # Highlight clicked dot in each multi-session chart
    def _make_highlight_callback(chart_id):
        @callback(
            Output(chart_id, 'figure', allow_duplicate=True),
            Input(chart_id, 'clickData'),
            State(chart_id, 'figure'),
            prevent_initial_call=True
        )
        def highlight_clicked_dot(click_data, fig):
            if not click_data or not fig:
                raise PreventUpdate
            clicked_x = click_data['points'][0].get('x')
            for trace in fig.get('data', []):
                xs = trace.get('x', [])
                n = len(xs)
                sizes = [16 if xs[i] == clicked_x else 8 for i in range(n)]
                trace.setdefault('marker', {})['size'] = sizes
            return fig

    for _chart_id in ['tccs-chart', 'activation-engagement-chart', 'cts-chart']:
        _make_highlight_callback(_chart_id)

    # Update the DAG Iframe
    @callback(
        Output('dag-iframe', 'srcDoc'),
        Input('sessions-data', 'data')
    )
    def update_session_dag(sessions_json):
        if not sessions_json:
            return ""
        
        try:
            df = pd.read_json(io.StringIO(sessions_json), orient='split')
            html_string = dag.create_session_dag_from_json(df)
            return html_string
        except Exception as e:
            logger.exception("Error creating session DAG:")
            return f"<div style='color:red; padding: 20px;'>Failed to render DAG: {e}</div>"




def register_clientside_callbacks(app):
    """Register all clientside (JavaScript) callbacks"""
    
    # Initialize wavesurfer with speaker regions
    app.clientside_callback(
        """
        function(audioPlayerContent, transcriptData) {
            console.log('🎯 WaveSurfer callback triggered');
            
            if (!audioPlayerContent) {
                console.log('No audio player content, skipping');
                return '';
            }
            
            setTimeout(function() {
                var audioElement = document.getElementById('audio-element');
                var waveformDiv = document.getElementById('waveform');
                
                if (!waveformDiv || !audioElement || !audioElement.src) return;
                
                console.log('✅ Audio element ready! Creating WaveSurfer with regions...');
                
                if (typeof WaveSurfer === 'undefined') {
                    console.error('❌ WaveSurfer is not loaded from CDN!');
                    waveformDiv.innerHTML = '<p style="color: orange; padding: 20px; text-align: center;">⚠️ WaveSurfer.js not loaded. Audio controls above still work!</p>';
                    return;
                }
                
                if (window.wavesurfer) {
                    try { window.wavesurfer.destroy(); } catch(e) {}
                }
                
                try {
                    waveformDiv.innerHTML = '<p style="padding: 20px; text-align: center; color: #6b7280; font-size: 13px;">Loading waveform...</p>';
                    
                    window.wavesurfer = WaveSurfer.create({
                        container: '#waveform',
                        waveColor: '#93c5fd',
                        progressColor: '#2563eb',
                        cursorColor: '#dc2626',
                        barWidth: 3,
                        barGap: 1,
                        barRadius: 2,
                        cursorWidth: 2,
                        height: 200,
                        responsive: true,
                        normalize: false,
                        fillParent: true,
                        minPxPerSec: 50,
                        barHeight: 2
                    });
                    
                    // Register Regions Plugin and store reference
                    window.wsRegions = null;
                    if (typeof WaveSurfer.Regions !== 'undefined') {
                        window.wsRegions = window.wavesurfer.registerPlugin(WaveSurfer.Regions.create());
                        console.log('✅ Regions plugin registered');
                    } else {
                        console.warn('⚠️ Regions plugin not available');
                    }
                    
                    window.wavesurfer.load(audioElement.src);
                    
                    window.wavesurfer.on('ready', function() {
                        console.log('🎉 WaveSurfer ready!');
                        waveformDiv.style.padding = '0';
                        // Remove loading text (WaveSurfer v7 appends to container, doesn't replace innerHTML)
                        var loadingTexts = waveformDiv.querySelectorAll('p');
                        loadingTexts.forEach(function(el) { el.remove(); });
                        
                        // Add speaker regions from transcript
                        if (transcriptData && window.wsRegions) {
                            try {
                                var transcript = JSON.parse(transcriptData);
                                var data = transcript.data;
                                var columns = transcript.columns;
                                
                                var startIdx = columns.indexOf('start');
                                var endIdx = columns.indexOf('end');
                                var speakerIdx = columns.indexOf('speaker');
                                
                                if (startIdx !== -1 && endIdx !== -1) {
                                    console.log('Creating', data.length, 'speaker regions...');
                                    
                                    data.forEach(function(row, idx) {
                                        var start = row[startIdx];
                                        var end = row[endIdx];
                                        var speaker = speakerIdx !== -1 ? row[speakerIdx] : 'Unknown';
                                        var isTherapist = speaker.toLowerCase().includes('therapist');
                                        
                                        console.log('Adding region:', idx, 'Speaker:', speaker, 'Start:', start, 'End:', end);
                                        
                                        // Add region via regions plugin instance
                                        window.wsRegions.addRegion({
                                            start: start,
                                            end: end,
                                            color: isTherapist ? 'rgba(37, 99, 235, 0.15)' : 'rgba(5, 150, 105, 0.15)',
                                            drag: false,
                                            resize: false,
                                            content: speaker.split('_')[0].toUpperCase().substring(0, 1),
                                            id: 'region-' + idx
                                        });
                                    });
                                    
                                    console.log('✅ All speaker regions created!');
                                }
                            } catch(e) {
                                console.error('Error creating regions:', e);
                            }
                        }
                    });
                    
                    // Handle region clicks
                    if (window.wsRegions) {
                        window.wsRegions.on('region-clicked', function(region, e) {
                            e.stopPropagation();
                            console.log('🎯 Region clicked:', region.id);
                            
                            // Extract index from region ID
                            var idx = parseInt(region.id.replace('region-', ''));
                            
                            // Play from this region
                            window.wavesurfer.play(region.start);
                            
                            // Update button
                            var button = document.getElementById('play-pause-btn');
                            if (button) button.innerText = '⏸ Pause';
                            
                            // Highlight corresponding transcript segment
                            var allSegments = document.querySelectorAll('[id*="transcript-segment"]');
                            allSegments.forEach(function(seg, segIdx) {
                                if (segIdx === idx) {
                                    seg.style.backgroundColor = '#dbeafe';
                                    seg.style.boxShadow = '0 2px 8px rgba(0,0,0,0.15)';
                                    seg.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                                } else {
                                    seg.style.backgroundColor = '#ffffff';
                                    seg.style.boxShadow = '0 1px 3px rgba(0,0,0,0.1)';
                                }
                            });
                        });
                    }
                    
                    window.wavesurfer.on('error', function(err) {
                        console.error('WaveSurfer error:', err);
                        waveformDiv.innerHTML = '<p style="color: orange; padding: 20px; text-align: center;">⚠️ Could not load waveform</p>';
                    });
                    
                    window.wavesurfer.on('click', function(progress) {
                        if (audioElement && audioElement.duration) {
                            audioElement.currentTime = progress * audioElement.duration;
                        }
                    });
                    
                } catch(error) {
                    console.error('❌ Error creating WaveSurfer:', error);
                    waveformDiv.innerHTML = '<p style="color: red; padding: 20px;">Error: ' + error.message + '</p>';
                }
            }, 500);
            
            return '';
        }
        """,
        Output('waveform', 'data-wavesurfer'),
        Input('audio-player-container', 'children'),
        Input('transcript-data', 'data')
    )
    
    # Manual waveform initialization
    app.clientside_callback(
        """
        function(n_clicks, transcriptData) {
            if (n_clicks === 0) return '';
            
            console.log('🔄 Manual waveform initialization with speaker regions');
            
            var audioElement = document.getElementById('audio-element');
            var waveformDiv = document.getElementById('waveform');
            
            if (!waveformDiv || !audioElement || !audioElement.src) {
                alert('❌ Audio element not ready! Load a session first.');
                return '';
            }
            
            if (typeof WaveSurfer === 'undefined') {
                alert('❌ WaveSurfer library not loaded!');
                return '';
            }
            
            if (window.wavesurfer) {
                try { window.wavesurfer.destroy(); } catch(e) {}
            }
            
            try {
                waveformDiv.innerHTML = '<p style="padding: 20px; text-align: center; color: #6b7280; font-size: 13px;">Creating waveform...</p>';
                
                // Create WaveSurfer with large visible bars
                window.wavesurfer = WaveSurfer.create({
                    container: '#waveform',
                    waveColor: '#4a90e2',
                    progressColor: '#1e5a9e',
                    cursorColor: '#ff0000',
                    barWidth: 3,
                    barGap: 1,
                    barRadius: 2,
                    height: 400,
                    responsive: true,
                    normalize: false,
                    backend: 'WebAudio',
                    minPxPerSec: 50,
                    barHeight: 2
                });
                
                // Register Regions Plugin and store reference
                window.wsRegions = null;
                if (typeof WaveSurfer.Regions !== 'undefined') {
                    window.wsRegions = window.wavesurfer.registerPlugin(WaveSurfer.Regions.create());
                    console.log('✅ Regions plugin registered');
                } else {
                    console.warn('⚠️ Regions plugin not available');
                }
                
                window.wavesurfer.load(audioElement.src);
                
                window.wavesurfer.on('ready', function() {
                    console.log('🎉 Waveform ready! Adding speaker regions...');
                    waveformDiv.style.padding = '0';
                    // Remove loading text
                    var loadingTexts = waveformDiv.querySelectorAll('p');
                    loadingTexts.forEach(function(el) { el.remove(); });
                    
                    // Add speaker regions from transcript
                    if (transcriptData && window.wsRegions) {
                        try {
                            var transcript = JSON.parse(transcriptData);
                            var data = transcript.data;
                            var columns = transcript.columns;
                            
                            var startIdx = columns.indexOf('start');
                            var endIdx = columns.indexOf('end');
                            var speakerIdx = columns.indexOf('speaker');
                            
                            if (startIdx !== -1 && endIdx !== -1) {
                                console.log('Creating', data.length, 'speaker regions...');
                                
                                data.forEach(function(row, idx) {
                                    var start = row[startIdx];
                                    var end = row[endIdx];
                                    var speaker = speakerIdx !== -1 ? row[speakerIdx] : 'Unknown';
                                    var isTherapist = speaker.toLowerCase().includes('therapist');
                                    
                                    console.log('Adding region:', idx, 'Speaker:', speaker, 'Start:', start, 'End:', end);
                                    
                                    // Add region via regions plugin instance
                                    window.wsRegions.addRegion({
                                        start: start,
                                        end: end,
                                        color: isTherapist ? 'rgba(25, 118, 210, 0.2)' : 'rgba(46, 125, 50, 0.2)',
                                        drag: false,
                                        resize: false,
                                        content: speaker.split('_')[0].toUpperCase().substring(0, 3),
                                        id: 'region-' + idx
                                    });
                                });
                                
                                console.log('✅ All speaker regions created!');
                            }
                        } catch(e) {
                            console.error('Error creating regions:', e);
                        }
                    }
                });
                
                // Handle region clicks
                if (window.wsRegions) {
                    window.wsRegions.on('region-clicked', function(region, e) {
                        e.stopPropagation();
                        console.log('🎯 Region clicked:', region.id);
                        
                        // Extract index from region ID
                        var idx = parseInt(region.id.replace('region-', ''));
                        
                        // Play from this region
                        window.wavesurfer.play(region.start);
                        
                        // Update button
                        var button = document.getElementById('play-pause-btn');
                        if (button) button.innerText = '⏸ Pause';
                        
                        // Highlight corresponding transcript segment
                        var allSegments = document.querySelectorAll('[id*="transcript-segment"]');
                        allSegments.forEach(function(seg, segIdx) {
                            if (segIdx === idx) {
                                seg.style.backgroundColor = '#dbeafe';
                                seg.style.boxShadow = '0 2px 8px rgba(0,0,0,0.15)';
                                seg.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                            } else {
                                seg.style.backgroundColor = '#ffffff';
                                seg.style.boxShadow = '0 1px 3px rgba(0,0,0,0.1)';
                            }
                        });
                    });
                }
                
                window.wavesurfer.on('error', function(err) {
                    console.error('Error:', err);
                    waveformDiv.innerHTML = '<p style="color: red; padding: 20px;">Error: ' + err + '</p>';
                });
                
                window.wavesurfer.on('click', function(progress) {
                    audioElement.currentTime = progress * audioElement.duration;
                });
                
            } catch(error) {
                console.error('❌ Error:', error);
                alert('Error creating waveform: ' + error.message);
            }
            
            return '';
        }
        """,
        Output('init-waveform-btn', 'data-dummy'),
        Input('init-waveform-btn', 'n_clicks'),
        State('transcript-data', 'data')
    )
    
    # Play/pause button
    app.clientside_callback(
        """
        function(n_clicks) {
            if (n_clicks === 0) return '';
            
            var button = document.getElementById('play-pause-btn');
            var audioElement = document.getElementById('audio-element');
            
            if (window.wavesurfer) {
                var isPlaying = window.wavesurfer.isPlaying();
                
                if (isPlaying) {
                    window.wavesurfer.pause();
                    if (button) button.innerText = '▶ Play';
                } else {
                    window.wavesurfer.play();
                    if (button) button.innerText = '⏸ Pause';
                }
            } else if (audioElement) {
                if (audioElement.paused) {
                    audioElement.play();
                    if (button) button.innerText = '⏸ Pause';
                } else {
                    audioElement.pause();
                    if (button) button.innerText = '▶ Play';
                }
            }
            
            return '';
        }
        """,
        Output('play-pause-btn', 'data-dummy'),
        Input('play-pause-btn', 'n_clicks')
    )
    
    # Rewind button
    app.clientside_callback(
        """
        function(n_clicks) {
            if (window.wavesurfer && n_clicks > 0) {
                var currentTime = window.wavesurfer.getCurrentTime();
                window.wavesurfer.setTime(Math.max(0, currentTime - 10));
            }
            return '';
        }
        """,
        Output('rewind-btn', 'data-dummy'),
        Input('rewind-btn', 'n_clicks')
    )
    
    # Forward button
    app.clientside_callback(
        """
        function(n_clicks) {
            if (window.wavesurfer && n_clicks > 0) {
                var currentTime = window.wavesurfer.getCurrentTime();
                var duration = window.wavesurfer.getDuration();
                window.wavesurfer.setTime(Math.min(duration, currentTime + 10));
            }
            return '';
        }
        """,
        Output('forward-btn', 'data-dummy'),
        Input('forward-btn', 'n_clicks')
    )
    
    # Transcript segment clicks - play audio and update button
    app.clientside_callback(
        """
        function(n_clicks_list, transcriptData) {
            if (!n_clicks_list || n_clicks_list.length === 0) return '';
            
            // Find which segment was clicked
            if (!window.previousClicksList) {
                window.previousClicksList = new Array(n_clicks_list.length).fill(0);
            }
            
            var clickedIndex = -1;
            for (var i = 0; i < n_clicks_list.length; i++) {
                if ((n_clicks_list[i] || 0) > (window.previousClicksList[i] || 0)) {
                    clickedIndex = i;
                    break;
                }
            }
            
            window.previousClicksList = n_clicks_list.slice();
            
            if (clickedIndex === -1) return '';
            
            console.log('✅ Segment clicked:', clickedIndex);
            
            if (!transcriptData) return '';
            
            try {
                var transcript = JSON.parse(transcriptData);
                var data = transcript.data;
                var columns = transcript.columns;
                
                var startIdx = columns.indexOf('start');
                if (startIdx === -1 || !data[clickedIndex]) return '';
                
                var startTime = data[clickedIndex][startIdx];
                var audioElement = document.getElementById('audio-element');
                var button = document.getElementById('play-pause-btn');
                
                if (!audioElement || !audioElement.duration) {
                    alert('Audio not loaded yet. Please wait.');
                    return '';
                }
                
                // Pause, seek, then play
                audioElement.pause();
                audioElement.currentTime = startTime;
                
                audioElement.addEventListener('seeked', function onSeeked() {
                    audioElement.removeEventListener('seeked', onSeeked);
                    
                    if (window.wavesurfer && window.wavesurfer.getDuration()) {
                        var progress = audioElement.currentTime / audioElement.duration;
                        window.wavesurfer.pause();
                        window.wavesurfer.seekTo(progress);
                        setTimeout(function() {
                            window.wavesurfer.play();
                            if (button) button.innerText = '⏸ Pause';
                        }, 100);
                    } else {
                        audioElement.play();
                        if (button) button.innerText = '⏸ Pause';
                    }
                });
                
                // Highlight clicked segment
                var allSegments = document.querySelectorAll('[id*="transcript-segment"]');
                allSegments.forEach(function(seg, idx) {
                    if (idx === clickedIndex) {
                        seg.style.backgroundColor = '#dbeafe';
                        seg.style.boxShadow = '0 2px 8px rgba(0,0,0,0.15)';
                        seg.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    } else {
                        seg.style.backgroundColor = '#ffffff';
                        seg.style.boxShadow = '0 1px 3px rgba(0,0,0,0.1)';
                    }
                });
                
            } catch (e) {
                console.error('Error:', e);
            }
            
            return '';
        }
        """,
        Output('current-time', 'data'),
        Input({'type': 'transcript-segment', 'index': dash.dependencies.ALL}, 'n_clicks'),
        State('transcript-data', 'data')
    )

