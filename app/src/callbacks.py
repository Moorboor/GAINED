"""
Callbacks for GAINED application
"""
from dash import html, dcc, Input, Output, State, callback, no_update, clientside_callback
import pandas as pd
import io
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash.exceptions import PreventUpdate
from .data_loader import (
    get_patient_sessions, 
    load_session_from_disk,
    load_session_with_rationale,
    encode_audio_to_base64,
    process_audio_upload,
    process_transcript_upload,
    process_transcript_upload_with_rationale
)


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
                    'color': '#28a745' if '✅' in msg else '#dc3545' if '❌' in msg else '#ffc107'
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
                'color': '#6c757d', 'textAlign': 'center', 'padding': '20px'
            })
        
        audio_src = encode_audio_to_base64(audio_data)
        
        return html.Div([
            html.P(f"✅ Audio loaded: {audio_data.get('filename', 'Unknown')}", 
                   style={'color': '#28a745', 'marginBottom': '15px', 'fontWeight': 'bold'}),
            html.Audio(
                id='audio-element',
                src=audio_src,
                controls=True,
                autoPlay=False,
                preload='auto',
                style={'width': '100%', 'marginBottom': '15px'}
            ),
            html.P("📊 Waveform visualization below:", 
                   style={'color': '#6c757d', 'fontSize': '14px', 'marginBottom': '5px'}),
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
                
                # Simple row style - all segments stacked vertically
                segment_style = {
                    'padding': '12px',
                    'marginBottom': '8px',
                    'borderRadius': '8px',
                    'cursor': 'pointer',
                    'transition': 'all 0.2s ease',
                    'borderLeft': f'4px solid {"#1976d2" if is_therapist else "#2e7d32"}',
                    'backgroundColor': '#ffffff',
                    'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
                }
                
                # Hover effect via class
                segment_style_hover = {
                    **segment_style,
                    'backgroundColor': '#f5f5f5',
                }
                
                segment = html.Div([
                    html.Div([
                        html.Span(speaker, style={
                            'fontSize': '12px',
                            'fontWeight': 'bold',
                            'color': '#1976d2' if is_therapist else '#2e7d32',
                            'textTransform': 'uppercase',
                            'marginRight': '12px'
                        }),
                        html.Span(f"{start_time:.1f}s - {end_time:.1f}s", style={
                            'fontSize': '11px',
                            'color': '#999',
                        })
                    ], style={'marginBottom': '6px'}),
                    html.Div(text, style={
                        'fontSize': '14px',
                        'lineHeight': '1.5',
                        'color': '#333'
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
            print(f"Error displaying transcript: {e}")
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
            print(f"Error creating chart: {e}")
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
            print(f"Error updating field selector: {e}")
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
            
            # Create pie chart
            fig = go.Figure(data=[go.Pie(
                labels=value_counts.index.astype(str),
                values=value_counts.values,
                hole=0.4,  # Creates a donut chart
                textinfo='label+percent',
                textposition='outside',
                marker=dict(
                    colors=px.colors.qualitative.Set3,
                    line=dict(color='#FFFFFF', width=2)
                ),
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )])
            
            fig.update_layout(
                title=f'Distribution of {selected_field} in Patient Speech Turns',
                template='plotly_white',
                height=500,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.05
                ),
                annotations=[dict(
                    text=f'Total<br>Patient Turns<br>{len(field_data)}',
                    x=0.5, y=0.5,
                    font_size=16,
                    showarrow=False
                )]
            )
            
            return fig
        
        except Exception as e:
            print(f"Error creating pie chart: {e}")
            import traceback
            traceback.print_exc()
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
            print(f"Error updating field plots selector: {e}")
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
                          style={'textAlign': 'center', 'color': '#6c757d', 'padding': '40px'})
        
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
                    except Exception as e:
                        print(f"Error parsing rationale for {col_name}: {e}")
            
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
                
                # Create plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x_data,
                    y=df[field],
                    mode='lines+markers',
                    name=field,
                    line=dict(width=2),
                    marker=dict(size=6)
                ))
                
                fig.update_layout(
                    title=field,
                    xaxis_title=x_label,
                    yaxis_title='Value',
                    hovermode='x unified',
                    template='plotly_white',
                    height=350,
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                
                # Get rationale for this field if available
                rationale_text = None
                if field in rationale_dict:
                    rationale_df = rationale_dict[field]
                    
                    # If the rationale sheet has the same column name, use that column
                    if field in rationale_df.columns:
                        # Get non-null values from the rationale column
                        rationale_values = rationale_df[field].dropna().astype(str).tolist()
                        if rationale_values:
                            # Join with line breaks if multiple values
                            rationale_text = '\n'.join(rationale_values) if len(rationale_values) > 1 else rationale_values[0]
                    else:
                        # Try to find a text/description column in rationale
                        text_cols = [col for col in rationale_df.columns 
                                   if any(keyword in col.lower() for keyword in ['text', 'description', 'rationale', 'reason', 'explanation'])]
                        if text_cols:
                            # Combine all rationale text
                            rationale_values = rationale_df[text_cols[0]].dropna().astype(str).tolist()
                            if rationale_values:
                                rationale_text = '\n'.join(rationale_values) if len(rationale_values) > 1 else rationale_values[0]
                        elif len(rationale_df.columns) > 0:
                            # Use first column if no obvious text column
                            rationale_values = rationale_df.iloc[:, 0].dropna().astype(str).tolist()
                            if rationale_values:
                                rationale_text = '\n'.join(rationale_values) if len(rationale_values) > 1 else rationale_values[0]
                
                # Create component for this field
                field_component = html.Div([
                    dcc.Graph(figure=fig, id={'type': 'field-plot', 'field': field}),
                    html.Div([
                        html.H4("Rationale", style={'fontSize': '16px', 'fontWeight': 'bold', 'marginBottom': '10px', 'color': '#495057'}),
                        html.Div(
                            rationale_text if rationale_text else "No rationale available for this field",
                            style={
                                'padding': '15px',
                                'backgroundColor': '#f8f9fa',
                                'borderRadius': '5px',
                                'borderLeft': '4px solid #007bff',
                                'fontSize': '14px',
                                'lineHeight': '1.6',
                                'color': '#495057' if rationale_text else '#6c757d',
                                'fontStyle': 'italic' if not rationale_text else 'normal',
                                'whiteSpace': 'pre-line'  # Preserve line breaks
                            }
                        )
                    ], style={'marginTop': '15px', 'marginBottom': '30px'})
                ], style={
                    'marginBottom': '40px',
                    'padding': '20px',
                    'backgroundColor': '#ffffff',
                    'borderRadius': '8px',
                    'border': '1px solid #dee2e6',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                })
                
                plot_components.append(field_component)
            
            if not plot_components:
                return html.Div("No valid fields selected", 
                              style={'textAlign': 'center', 'color': '#6c757d', 'padding': '40px'})
            
            return plot_components
        
        except Exception as e:
            print(f"Error creating field plots: {e}")
            import traceback
            traceback.print_exc()
            return html.Div(f"Error displaying plots: {str(e)}", 
                          style={'textAlign': 'center', 'color': '#dc3545', 'padding': '40px'})
    
    
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
                    waveformDiv.innerHTML = '<p style="padding: 20px; text-align: center; color: #007bff;">⏳ Loading waveform...</p>';
                    
                    window.wavesurfer = WaveSurfer.create({
                        container: '#waveform',
                        waveColor: '#4a90e2',
                        progressColor: '#1e5a9e',
                        cursorColor: '#ff0000',
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
                                    seg.style.backgroundColor = '#e3f2fd';
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
                waveformDiv.innerHTML = '<p style="padding: 20px; text-align: center; color: #007bff;">⏳ Creating waveform with speaker regions...</p>';
                
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
                                seg.style.backgroundColor = '#e3f2fd';
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
                        seg.style.backgroundColor = '#e3f2fd';
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

