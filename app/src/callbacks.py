"""
Callbacks for GAINED application
"""
from dash import html, Input, Output, State, callback, no_update, clientside_callback
import pandas as pd
import io
import plotly.graph_objects as go
import dash
from .data_loader import (
    get_patient_sessions, 
    load_uploaded_files, 
    load_session_from_disk,
    encode_audio_to_base64
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
         Output('upload-status', 'children')],
        Input('upload-files', 'contents'),
        State('upload-files', 'filename')
    )
    def handle_file_upload(list_of_contents, list_of_filenames):
        audio_data, transcript_data, status_messages = load_uploaded_files(
            list_of_contents, list_of_filenames
        )
        
        if status_messages:
            status_div = html.Div([
                html.Div(msg, style={
                    'color': '#28a745' if '✅' in msg else '#dc3545' if '❌' in msg else '#ffc107'
                }) for msg in status_messages
            ])
            return audio_data, transcript_data, status_div
        
        return no_update, no_update, ""
    
    
    # Load existing session data
    @callback(
        [Output('audio-data', 'data', allow_duplicate=True),
         Output('transcript-data', 'data', allow_duplicate=True),
         Output('audio-section', 'style'),
         Output('transcript-section', 'style'),
         Output('chart-section', 'style')],
        Input('load-button', 'n_clicks'),
        [State('patient-dropdown', 'value'),
         State('session-dropdown', 'value')],
        prevent_initial_call=True
    )
    def load_session_data(n_clicks, patient_id, session_idx):
        from dash.exceptions import PreventUpdate
        
        if n_clicks == 0:
            raise PreventUpdate
        
        visible_style = {
            'padding': '20px', 'backgroundColor': '#ffffff', 'borderRadius': '10px',
            'marginBottom': '20px', 'boxShadow': '0 2px 4px rgba(0,0,0,0.1)', 'display': 'block'
        }
        
        audio_data, transcript_data = load_session_from_disk(patient_id, session_idx)
        
        if audio_data or transcript_data:
            return audio_data, transcript_data, visible_style, visible_style, visible_style
        
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
            
            # Create simple list of transcript segments
            segments = []
            for idx, row in df.iterrows():
                speaker = row.get('speaker', 'Unknown') if speaker_col else 'Unknown'
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
        State('transcript-data', 'data')
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

