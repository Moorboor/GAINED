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
        status_messages = []
        
        if triggered_id == 'upload-audio' and audio_content and audio_filename:
            new_audio_data, audio_msg = process_audio_upload(audio_content, audio_filename)
            if new_audio_data:
                audio_output = new_audio_data
            if audio_msg:
                status_messages.append(audio_msg)
        elif triggered_id == 'upload-transcript' and transcript_content and transcript_filename:
            # Use the new function that also loads rationale
            new_transcript_data, new_rationale_data, transcript_msg = process_transcript_upload_with_rationale(
                transcript_content, transcript_filename
            )
            if new_transcript_data:
                transcript_output = new_transcript_data
            if new_rationale_data:
                rationale_output = new_rationale_data
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
        
        return audio_output, transcript_output, rationale_output, status_div
    
    
    # Load existing session data
    @callback(
        [Output('audio-data', 'data', allow_duplicate=True),
         Output('transcript-data', 'data', allow_duplicate=True),
         Output('rationale-data', 'data', allow_duplicate=True),
         Output('audio-section', 'style'),
         Output('transcript-section', 'style'),
         Output('chart-section', 'style')],
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
        transcript_data, rationale_data = load_session_with_rationale(patient_id, session_idx)
        
        # Load audio separately (using old function for now)
        audio_data, _ = load_session_from_disk(patient_id, session_idx)
        
        if audio_data or transcript_data:
            return audio_data, transcript_data, rationale_data, visible_style, visible_style, visible_style
        
        raise PreventUpdate
    
    
    # Show/hide sections when files are uploaded or loaded
    @callback(
        [Output('audio-section', 'style', allow_duplicate=True),
         Output('transcript-section', 'style', allow_duplicate=True),
         Output('chart-section', 'style', allow_duplicate=True)],
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
        chart_visible = visible_style if transcript_data else hidden_style
        
        return audio_visible, transcript_visible, chart_visible
    
    
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
    
    
    # Display metrics chart
    @callback(
        Output('metrics-chart', 'figure'),
        Input('transcript-data', 'data')
    )
    def display_metrics_chart(transcript_data):
        if not transcript_data:
            return go.Figure()
        
        try:
            df = pd.read_json(io.StringIO(transcript_data), orient='split')
            
            # Find metric columns
            metric_columns = []
            possible_metrics = ['LLM_T', 'sentiment', 'emotion', 'engagement', 'score', 'metric']
            
            for col in df.columns:
                for metric in possible_metrics:
                    if metric.lower() in col.lower():
                        metric_columns.append(col)
                        break
            
            if not metric_columns:
                metric_columns = df.select_dtypes(include=['number']).columns.tolist()
                metric_columns = [col for col in metric_columns if col not in ['segment_id', 'index', 'start', 'end']]
            
            fig = go.Figure()
            
            if 'segment_id' in df.columns:
                x_data = df['segment_id']
                x_label = 'Segment'
            else:
                x_data = df.index
                x_label = 'Index'
            
            for metric in metric_columns[:3]:
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=df[metric],
                    mode='lines+markers',
                    name=metric,
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
            
            fig.update_layout(
                title='Session Metrics Over Time',
                xaxis_title=x_label,
                yaxis_title='Metric Value',
                hovermode='x unified',
                template='plotly_white',
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            return fig
        
        except Exception as e:
            logger.exception("Error creating chart")
            return go.Figure()
    
    
    # Update pie chart field selector dropdown
    @callback(
        Output('pie-chart-field-selector', 'options'),
        Output('pie-chart-field-selector', 'value'),
        Input('transcript-data', 'data')
    )
    def update_pie_chart_field_selector(transcript_data):
        if not transcript_data:
            return [], None
        
        try:
            df = pd.read_json(io.StringIO(transcript_data), orient='split')
            
            # Find patient speech-turnvised fields (fields ending with _P)
            # These are fields that have separate values for patient speech turns
            patient_fields = []
            excluded_cols = {'segment_id', 'index', 'start', 'end', 'text', 'segment_text', 'speaker', 'pred', 'speaker_text', 'num_words', 'words', 'wespeaker_div'}
            
            for col in df.columns:
                col_lower = col.lower()
                # Skip excluded columns
                if col_lower in excluded_cols:
                    continue
                
                # Look for fields ending with _P (patient-specific speech-turnvised fields)
                if col.endswith('_P') or col.endswith('_p'):
                    patient_fields.append(col)
                # Also include categorical fields that might be emotion-related
                elif df[col].dtype == 'object':
                    # String columns that might be categorical
                    if any(keyword in col_lower for keyword in ['emotion', 'sentiment', 'feeling', 'mood', 'affect']):
                        patient_fields.insert(0, col)  # Prioritize emotion fields
                    elif col_lower not in excluded_cols:
                        # Other string columns (but we'll filter for patient turns when displaying)
                        pass
            
            # If no _P fields found, look for any numeric fields that could be binned
            if not patient_fields:
                for col in df.columns:
                    if col.lower() in excluded_cols:
                        continue
                    if df[col].dtype in ['int64', 'float64']:
                        # Low-cardinality numeric might be categorical
                        unique_count = df[col].nunique()
                        total_count = len(df[col].dropna())
                        if unique_count < 20 and unique_count < total_count * 0.9:
                            patient_fields.append(col)
            
            options = [{'label': col, 'value': col} for col in patient_fields]
            
            # Auto-select first field if available
            value = patient_fields[0] if patient_fields else None
            
            return options, value
        
        except Exception as e:
            logger.exception("Error updating field selector")
            return [], None
    
    
    # Generate pie chart for patient speech-turn fields
    @callback(
        Output('pie-chart', 'figure'),
        Input('pie-chart-field-selector', 'value'),
        Input('transcript-data', 'data')
    )
    def generate_pie_chart(selected_field, transcript_data):
        if not transcript_data or not selected_field:
            return go.Figure()
        
        try:
            df = pd.read_json(io.StringIO(transcript_data), orient='split')
            
            # Filter for patient speech turns only
            # Check for speaker_text column (contains "P:" for patient, "T:" for therapist)
            # or speaker column
            patient_df = None
            if 'speaker_text' in df.columns:
                # Filter rows where speaker_text contains "P:"
                patient_df = df[df['speaker_text'].astype(str).str.contains('^P:', case=False, na=False, regex=True)].copy()
            elif 'speaker' in df.columns:
                # Filter for patient speech turns
                def is_patient(value):
                    if pd.isna(value):
                        return False
                    value_str = str(value).strip().lower()
                    return 'patient' in value_str or value_str.startswith('p')
                
                patient_df = df[df['speaker'].apply(is_patient)].copy()
            else:
                # If no speaker column, use all data (assume all are patient turns)
                patient_df = df.copy()
            
            if patient_df.empty or selected_field not in patient_df.columns:
                return go.Figure()
            
            # Get field data
            field_data = patient_df[selected_field].dropna()
            
            if field_data.empty:
                return go.Figure()
            
            # Handle numeric fields by binning them
            if field_data.dtype in ['int64', 'float64']:
                # For numeric fields, create bins
                if field_data.min() == field_data.max():
                    # All values are the same, just use the value
                    value_counts = pd.Series([len(field_data)], index=[f'{field_data.iloc[0]:.3f}'])
                else:
                    # Create bins: use quantiles for better distribution
                    try:
                        # Try to create 5-10 bins based on data distribution
                        n_bins = min(10, max(5, int(len(field_data) / 10)))
                        if n_bins < 2:
                            n_bins = 2
                        
                        # Use quantile-based bins for better visualization
                        bins = pd.qcut(field_data, q=n_bins, duplicates='drop', precision=2)
                        value_counts = bins.value_counts().sort_index()
                        # Format bin labels
                        value_counts.index = [str(interval) for interval in value_counts.index]
                    except (ValueError, TypeError):
                        # Fallback to equal-width bins
                        bins = pd.cut(field_data, bins=min(10, field_data.nunique()), precision=2, duplicates='drop')
                        value_counts = bins.value_counts().sort_index()
                        value_counts.index = [str(interval) for interval in value_counts.index]
            else:
                # Categorical/string fields - use value counts directly
                value_counts = field_data.value_counts()
            
            # Create subplots: pie chart + bar chart side by side
            from plotly.subplots import make_subplots
            
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "pie"}, {"type": "bar"}]],
                subplot_titles=[f'Distribution of {selected_field}', f'Distribution of {selected_field}'],
                column_widths=[0.5, 0.5]
            )
            
            # Colors
            colors = px.colors.qualitative.Set3[:len(value_counts)]
            
            # 1. Donut Pie Chart (left)
            fig.add_trace(go.Pie(
                labels=value_counts.index.astype(str),
                values=value_counts.values,
                hole=0.4,
                textinfo='label+percent',
                textposition='outside',
                marker=dict(
                    colors=colors,
                    line=dict(color='#FFFFFF', width=2)
                ),
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            ), row=1, col=1)
            
            # 2. Horizontal Bar Chart (right)
            fig.add_trace(go.Bar(
                y=value_counts.index.astype(str),
                x=value_counts.values,
                orientation='h',
                marker=dict(color=colors),
                hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>',
                showlegend=False
            ), row=1, col=2)
            
            fig.update_layout(
                template='plotly_white',
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="center",
                    x=0.5
                ),
                margin=dict(l=40, r=40, t=60, b=40)
            )
            
            # Update bar chart axes
            fig.update_xaxes(title_text="Count", row=1, col=2)
            fig.update_yaxes(title_text=selected_field, row=1, col=2)
            
            return fig
        
        except Exception as e:
            logger.exception("Error creating pie chart")
            return go.Figure()
    
    
    # Show/hide pie chart section when transcript data is available
    @callback(
        Output('pie-chart-section', 'style'),
        Input('transcript-data', 'data')
    )
    def update_pie_chart_section_visibility(transcript_data):
        visible_style = {
            'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px',
            'marginBottom': '20px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'display': 'block'
        }
        hidden_style = {'display': 'none'}
        
        return visible_style if transcript_data else hidden_style
    
    
    # Update field plots selector dropdown
    @callback(
        Output('field-plots-selector', 'options'),
        Input('transcript-data', 'data')
    )
    def update_field_plots_selector(transcript_data):
        if not transcript_data:
            return []
        
        try:
            df = pd.read_json(io.StringIO(transcript_data), orient='split')
            
            # Find numeric fields that can be plotted
            excluded_cols = {'segment_id', 'index', 'start', 'end'}
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            plotable_fields = [col for col in numeric_cols if col.lower() not in excluded_cols]
            
            options = [{'label': col, 'value': col} for col in plotable_fields]
            return options
        
        except Exception as e:
            logger.exception("Error updating field plots selector")
            return []
    
    
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
            
            # Determine x-axis
            if 'segment_id' in df.columns:
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


    # Update sessions charts
    @callback(
        [Output('tccs-chart', 'figure'),
         Output('activation-engagement-chart', 'figure'),
         Output('cts-chart', 'figure'),
         Output('sessions-detailed-charts-section', 'style')],
        [Input('sessions-data', 'data')]
    )
    def update_sessions_charts(sessions_json):
        # Detailed charts style
        detailed_section_style = {'display': 'block', 'marginTop': '24px', 'padding': '0 12px'}
        hidden_style = {'display': 'none'}
        
        empty_figs = (go.Figure(), go.Figure(), go.Figure())
        
        if not sessions_json:
            return (*empty_figs, hidden_style)
        
        try:
            df = pd.read_json(io.StringIO(sessions_json), orient='split')
            
            if df.empty:
                return (*empty_figs, hidden_style)
            
            # Sort by session number
            if 'session_number' in df.columns:
                df = df.sort_values('session_number')
            
            x_data = df['session_number'] if 'session_number' in df.columns else df.index
            
            # Helper to create trace with rationale
            def create_trace(metric_name, label, color):
                if metric_name not in df.columns:
                    return None
                
                hover_texts = []
                
                for idx, row in df.iterrows():
                    val = row.get(metric_name)
                    
                    hover_text = f"<b>Session {row.get('session_number', idx+1)}</b><br>"
                    hover_text += f"{label}: {val}"
                    
                    hover_texts.append(hover_text)
                
                return go.Scatter(
                    x=x_data,
                    y=df[metric_name],
                    mode='lines+markers',
                    name=label,
                    line=dict(width=3, color=color),
                    marker=dict(size=8),
                    hovertemplate='%{text}<extra></extra>',
                    text=hover_texts
                )

            # 1. TCCS Chart (Challenging vs Supporting)
            fig_tccs = go.Figure()
            t1 = create_trace('challenging', 'Challenging (TCCS_SP)', '#dc2626') # Red
            t2 = create_trace('supporting', 'Supporting (TCCS_C)', '#2563eb')   # Blue
            if t1: fig_tccs.add_trace(t1)
            if t2: fig_tccs.add_trace(t2)
            
            fig_tccs.update_layout(
                xaxis_title='Session Number',
                yaxis_title='Score',
                hovermode='closest',
                template='plotly_white',
                margin=dict(l=40, r=40, t=30, b=40),
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
            )

            # 2. Activation vs Engagement Chart
            fig_ae = go.Figure()
            t1 = create_trace('activation', 'Activation', '#9333ea')   # Purple
            t2 = create_trace('engagement', 'Engagement', '#059669')   # Green
            if t1: fig_ae.add_trace(t1)
            if t2: fig_ae.add_trace(t2)
            
            fig_ae.update_layout(
                xaxis_title='Session Number',
                yaxis_title='Score',
                hovermode='closest',
                template='plotly_white',
                margin=dict(l=40, r=40, t=30, b=40),
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
            )

            # 3. CTS Chart
            fig_cts = go.Figure()
            cts_metrics = [
                ('cts_cognitions', 'Cognitions', '#ea580c'),   # Orange
                ('cts_behaviours', 'Behaviours', '#0891b2'),   # Cyan
                ('cts_discovery', 'Discovery', '#db2777'),     # Pink
                ('cts_methods', 'Methods', '#4f46e5')          # Indigo
            ]
            for metric, label, color in cts_metrics:
                t = create_trace(metric, label, color)
                if t: fig_cts.add_trace(t)
            
            fig_cts.update_layout(
                xaxis_title='Session Number',
                yaxis_title='Score',
                hovermode='closest',
                template='plotly_white',
                margin=dict(l=40, r=40, t=30, b=40),
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
            )
            
            return fig_tccs, fig_ae, fig_cts, detailed_section_style
            
        except Exception as e:
            logger.exception("Error updating sessions charts")
            return (*empty_figs, hidden_style)


    # Update rationale text boxes
    @callback(
        [Output('tccs-rationale', 'children'),
         Output('activation-rationale', 'children'),
         Output('cts-rationale', 'children')],
        [Input('tccs-chart', 'clickData'),
         Input('activation-engagement-chart', 'clickData'),
         Input('cts-chart', 'clickData'),
         Input('sessions-data', 'data')]
    )
    def update_rationale_boxes(tccs_click, ae_click, cts_click, sessions_json):
        if not sessions_json:
            return "No data", "No data", "No data"
            
        try:
            df = pd.read_json(io.StringIO(sessions_json), orient='split')
            if df.empty:
                return "No data", "No data", "No data"
            
            # Sort by session number
            if 'session_number' in df.columns:
                df = df.sort_values('session_number')
            
            # Helper to extract rationale text for a session index
            def get_rationale_text(session_idx, metrics):
                if session_idx >= len(df):
                    return "Session not found"
                
                row = df.iloc[session_idx]
                session_num = row.get('session_number', session_idx + 1)
                text_parts = [f"** Session {session_num} **"]
                
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
                            
                return "\n".join(text_parts)

            # Determine session index for each chart
            # Default to last session (-1) if no click
            
            # TCCS
            tccs_idx = -1
            if tccs_click and 'points' in tccs_click:
                # Use pointIndex from clickData
                tccs_idx = tccs_click['points'][0]['pointIndex']
            
            tccs_text = get_rationale_text(tccs_idx, [
                ('challenging', 'Challenging (TCCS_SP)'),
                ('supporting', 'Supporting (TCCS_C)')
            ])
            
            # Activation/Engagement
            ae_idx = -1
            if ae_click and 'points' in ae_click:
                ae_idx = ae_click['points'][0]['pointIndex']
            
            ae_text = get_rationale_text(ae_idx, [
                ('activation', 'Activation'),
                ('engagement', 'Engagement')
            ])
            
            # CTS
            cts_idx = -1
            if cts_click and 'points' in cts_click:
                cts_idx = cts_click['points'][0]['pointIndex']
            
            cts_text = get_rationale_text(cts_idx, [
                ('cts_cognitions', 'Cognitions'),
                ('cts_behaviours', 'Behaviours'),
                ('cts_discovery', 'Discovery'),
                ('cts_methods', 'Methods')
            ])
            
            return tccs_text, ae_text, cts_text
            
        except Exception as e:
            logger.exception("Error updating rationale boxes")
            return f"Error: {e}", f"Error: {e}", f"Error: {e}"


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

