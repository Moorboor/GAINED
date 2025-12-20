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


def load_xlsx_with_sheets(file_path_or_buffer, is_buffer=False):
    """
    Load xlsx file and return main sheet data and rationale sheets.
    
    Returns:
        tuple: (main_df, rationale_dict) where:
            - main_df: DataFrame from the first sheet (main sheet)
            - rationale_dict: Dictionary mapping column names to their rationale DataFrames
    """
    try:
        if is_buffer:
            xl_file = pd.ExcelFile(file_path_or_buffer, engine='openpyxl')
        else:
            xl_file = pd.ExcelFile(file_path_or_buffer, engine='openpyxl')
        
        sheet_names = xl_file.sheet_names
        
        if not sheet_names:
            return None, {}
        
        # First sheet is the main sheet
        main_df = pd.read_excel(xl_file, sheet_name=sheet_names[0], engine='openpyxl')
        main_df = _normalize_transcript_df(main_df)
        
        # Look for rationale sheets
        rationale_dict = {}
        main_columns = set(main_df.columns)
        
        # Check for a sheet named "rationale" first
        rationale_sheet_name = None
        for sheet_name in sheet_names[1:]:
            if sheet_name.lower() == 'rationale':
                rationale_sheet_name = sheet_name
                break
        
        if rationale_sheet_name:
            # Load rationale sheet and match columns
            rationale_df = pd.read_excel(xl_file, sheet_name=rationale_sheet_name, engine='openpyxl')
            for col in main_columns:
                if col in rationale_df.columns:
                    # Extract just this column's rationale data
                    rationale_dict[col] = rationale_df[[col]].copy()
        else:
            # Check if sheet name matches a column name (case-insensitive)
            for sheet_name in sheet_names[1:]:  # Skip first sheet (main)
                sheet_lower = sheet_name.lower()
                matching_column = None
                
                for col in main_columns:
                    if col.lower() == sheet_lower:
                        matching_column = col
                        break
                
                if matching_column:
                    # Load rationale sheet
                    rationale_df = pd.read_excel(xl_file, sheet_name=sheet_name, engine='openpyxl')
                    rationale_dict[matching_column] = rationale_df
        
        return main_df, rationale_dict
    
    except Exception as e:
        print(f"Error loading xlsx with sheets: {e}")
        return None, {}


def load_session_with_rationale(patient_id, session_idx):
    """
    Load session data including main sheet and rationale sheets.
    
    Returns:
        tuple: (main_data_json, rationale_data_dict) where:
            - main_data_json: JSON string of main sheet data
            - rationale_data_dict: Dictionary mapping column names to rationale JSON strings
    """
    if patient_id is None or session_idx is None:
        return None, {}
    
    patient_path = os.path.join(DATA_PATH, patient_id)
    sessions = [s for s in os.listdir(patient_path) if s.endswith(('.xlsx', '.csv'))]
    
    if session_idx >= len(sessions):
        return None, {}
    
    transcript_file = os.path.join(patient_path, sessions[session_idx])
    
    try:
        if sessions[session_idx].endswith('.csv'):
            # CSV files don't have multiple sheets
            df = pd.read_csv(transcript_file)
            df = _normalize_transcript_df(df)
            main_data = df.to_json(date_format='iso', orient='split')
            return main_data, {}
        else:
            # XLSX file - load with sheets
            main_df, rationale_dict = load_xlsx_with_sheets(transcript_file, is_buffer=False)
            
            if main_df is None:
                return None, {}
            
            main_data = main_df.to_json(date_format='iso', orient='split')
            
            # Convert rationale DataFrames to JSON
            rationale_data_dict = {}
            for col_name, rationale_df in rationale_dict.items():
                rationale_data_dict[col_name] = rationale_df.to_json(date_format='iso', orient='split')
            
            return main_data, rationale_data_dict
    
    except Exception as e:
        print(f"Error loading session with rationale: {e}")
        return None, {}


def process_transcript_upload_with_rationale(content, filename):
    """
    Process uploaded transcript file and return main data and rationale data.
    
    Returns:
        tuple: (main_data_json, rationale_data_dict, status_message)
    """
    if not content or not filename:
        return None, {}, ""
    
    if not filename.lower().endswith(('.xlsx', '.csv')):
        return None, {}, f"⚠️ Unsupported transcript file: {filename}"
    
    try:
        _, content_string = content.split(',', 1)
        decoded = base64.b64decode(content_string)
        
        if filename.lower().endswith('.csv'):
            # CSV files don't have multiple sheets
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            df = _normalize_transcript_df(df)
            main_data = df.to_json(date_format='iso', orient='split')
            return main_data, {}, f"✅ Transcript loaded: {filename} ({len(df)} segments)"
        else:
            # XLSX file - load with sheets
            file_buffer = io.BytesIO(decoded)
            main_df, rationale_dict = load_xlsx_with_sheets(file_buffer, is_buffer=True)
            
            if main_df is None:
                return None, {}, f"❌ Error loading {filename}: Could not read main sheet"
            
            main_data = main_df.to_json(date_format='iso', orient='split')
            
            # Convert rationale DataFrames to JSON
            rationale_data_dict = {}
            for col_name, rationale_df in rationale_dict.items():
                rationale_data_dict[col_name] = rationale_df.to_json(date_format='iso', orient='split')
            
            rationale_count = len(rationale_dict)
            status_msg = f"✅ Transcript loaded: {filename} ({len(main_df)} segments"
            if rationale_count > 0:
                status_msg += f", {rationale_count} rationale sheet(s)"
            status_msg += ")"
            
            return main_data, rationale_data_dict, status_msg
    
    except Exception as e:
        return None, {}, f"❌ Error loading {filename}: {str(e)}"

