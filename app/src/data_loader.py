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


def _map_pred_to_speaker(value):
    """Convert classifier prediction to human readable speaker label."""
    therapist_tokens = {'therapist', 't', '1', '1.0', 'true'}
    patient_tokens = {'patient', 'p', '0', '0.0', 'false'}
    
    if pd.isna(value):
        return 'Unknown'
    
    try:
        numeric_value = float(value)
        # Explicit mapping requested: 1 -> therapist, 0 -> patient
        if numeric_value == 1:
            return 'Therapist'
        if numeric_value == 0:
            return 'Patient'
    except (TypeError, ValueError):
        pass
    
    value_str = str(value).strip().lower()
    
    if value_str in therapist_tokens:
        return 'Therapist'
    if value_str in patient_tokens:
        return 'Patient'
    
    # Fall back to boolean-ish interpretation
    try:
        numeric_value = float(value_str)
        return 'Therapist' if numeric_value >= 0.5 else 'Patient'
    except (TypeError, ValueError):
        return 'Unknown'


def _normalize_transcript_df(df):
    """
    Ensure transcripts always contain a speaker column so downstream UI
    can reliably style regions and labels.
    """
    if df is None or df.empty:
        return df
    
    normalized_df = df.copy()
    
    if 'speaker' in normalized_df.columns:
        normalized_df['speaker'] = normalized_df['speaker'].fillna('Unknown')
        return normalized_df
    
    if 'pred' in normalized_df.columns:
        normalized_df['speaker'] = normalized_df['pred'].apply(_map_pred_to_speaker)
        return normalized_df
    
    normalized_df['speaker'] = 'Unknown'
    return normalized_df


def process_audio_upload(content, filename):
    """Validate and serialize uploaded audio content."""
    if not content or not filename:
        return None, ""
    
    if not filename.lower().endswith(('.mp3', '.wav')):
        return None, f"⚠️ Unsupported audio file: {filename}"
    
    try:
        _, content_string = content.split(',', 1)
    except ValueError:
        return None, f"❌ Could not read audio file: {filename}"
    
    audio_data = {
        'content': content_string,
        'filename': filename,
        'type': 'uploaded'
    }
    return audio_data, f"✅ Audio loaded: {filename}"


def process_transcript_upload(content, filename):
    """Validate, parse, and serialize an uploaded transcript."""
    if not content or not filename:
        return None, ""
    
    if not filename.lower().endswith(('.xlsx', '.csv')):
        return None, f"⚠️ Unsupported transcript file: {filename}"
    
    try:
        _, content_string = content.split(',', 1)
        decoded = base64.b64decode(content_string)
        
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            df = pd.read_excel(io.BytesIO(decoded), engine='openpyxl')
        
        df = _normalize_transcript_df(df)
        transcript_json = df.to_json(date_format='iso', orient='split')
        return transcript_json, f"✅ Transcript loaded: {filename} ({len(df)} segments)"
    except Exception as e:
        return None, f"❌ Error loading {filename}: {str(e)}"


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
        
        df = _normalize_transcript_df(df)
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

