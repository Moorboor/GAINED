"""
Data loading utilities for GAINED application
"""
import os
import base64
import io
import pandas as pd

# Paths
ABS_PATH = os.path.abspath("")
DATA_PATH = os.path.join(ABS_PATH, "app", "test_video_splits", "transcribe_request_NVIDIA")
AUDIO_PATH = os.path.join(ABS_PATH, "app", "test_video_splits")


def get_patient_sessions(patient_id):
    """Get list of sessions for a given patient ID"""
    if patient_id:
        patient_path = os.path.join(DATA_PATH, patient_id)
        if os.path.exists(patient_path):
            sessions = os.listdir(patient_path)
            sessions = [s for s in sessions if s.endswith(('.xlsx', '.csv'))]
            return [{'label': f'Session {i+1}: {s}', 'value': i} for i, s in enumerate(sessions)]
    return []


def load_uploaded_files(list_of_contents, list_of_filenames):
    """Process uploaded files and return audio data, transcript data, and status messages"""
    if not list_of_contents:
        return None, None, ""
    
    audio_data = None
    transcript_data = None
    status_messages = []
    
    for content, filename in zip(list_of_contents, list_of_filenames):
        content_type, content_string = content.split(',')
        
        # Check if it's an audio file
        if filename.endswith('.mp3') or filename.endswith('.wav'):
            audio_data = {
                'content': content_string,
                'filename': filename,
                'type': 'uploaded'
            }
            status_messages.append(f"✅ Audio loaded: {filename}")
        
        # Check if it's a transcript file
        elif filename.endswith('.xlsx') or filename.endswith('.csv'):
            decoded = base64.b64decode(content_string)
            try:
                if filename.endswith('.csv'):
                    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                else:
                    df = pd.read_excel(io.BytesIO(decoded), engine='openpyxl')
                
                transcript_data = df.to_json(date_format='iso', orient='split')
                status_messages.append(f"✅ Transcript loaded: {filename} ({len(df)} segments)")
            except Exception as e:
                status_messages.append(f"❌ Error loading {filename}: {str(e)}")
        else:
            status_messages.append(f"⚠️ Unsupported file: {filename}")
    
    return audio_data, transcript_data, status_messages


def load_session_from_disk(patient_id, session_idx):
    """Load audio and transcript from disk for existing session"""
    if patient_id is None or session_idx is None:
        return None, None
    
    # Load transcript
    patient_path = os.path.join(DATA_PATH, patient_id)
    sessions = [s for s in os.listdir(patient_path) if s.endswith(('.xlsx', '.csv'))]
    
    if session_idx >= len(sessions):
        return None, None
    
    transcript_file = os.path.join(patient_path, sessions[session_idx])
    
    try:
        if sessions[session_idx].endswith('.csv'):
            df = pd.read_csv(transcript_file)
        else:
            df = pd.read_excel(transcript_file, engine='openpyxl')
        
        transcript_data = df.to_json(date_format='iso', orient='split')
        
        # Load audio
        audio_files_path = os.path.join(AUDIO_PATH, patient_id)
        if os.path.exists(audio_files_path):
            audio_files = [f for f in os.listdir(audio_files_path) if f.endswith('.mp3')]
            if session_idx < len(audio_files):
                audio_file = os.path.join(audio_files_path, audio_files[session_idx])
                audio_data = {
                    'path': audio_file,
                    'filename': audio_files[session_idx],
                    'type': 'local'
                }
                return audio_data, transcript_data
        
        return None, transcript_data
        
    except Exception as e:
        print(f"Error loading session: {e}")
        return None, None


def encode_audio_to_base64(audio_data):
    """Convert audio data to base64 encoded data URI"""
    if not audio_data:
        return ""
    
    if audio_data.get('type') == 'uploaded':
        return f"data:audio/mp3;base64,{audio_data['content']}"
    else:
        # For local files, encode to base64
        audio_path = audio_data.get('path', '')
        try:
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
                audio_b64 = base64.b64encode(audio_bytes).decode()
                return f"data:audio/mp3;base64,{audio_b64}"
        except Exception as e:
            print(f"Error loading audio: {e}")
            return ""

