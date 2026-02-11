"""
Data loading utilities for GAINED application
"""
import logging
import os
import base64
import io

import pandas as pd

logger = logging.getLogger(__name__)

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
        
    except Exception:
        logger.exception("Error loading session")
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
        except Exception:
            logger.exception("Error loading audio")
            return ""


def load_xlsx_with_sheets(file_path_or_buffer):
    """
    Load xlsx file and return main sheet data and rationale sheets.
    
    Returns:
        tuple: (main_df, rationale_dict) where:
            - main_df: DataFrame from the first sheet (main sheet)
            - rationale_dict: Dictionary mapping column names to their rationale DataFrames.
              Also includes '_session_rationale' key with session-level rationale text
              (dict of metric_name -> rationale_text).
    """
    try:
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
            # Load rationale sheet
            rationale_df = pd.read_excel(xl_file, sheet_name=rationale_sheet_name, engine='openpyxl')
            
            # Preserve segment_id or index if present for matching
            key_cols = []
            if 'segment_id' in rationale_df.columns:
                key_cols.append('segment_id')
            if 'index' in rationale_df.columns:
                key_cols.append('index')
            
            # Track session-level rationale (columns ending in _rationale)
            session_rationale = {}
            
            for col in rationale_df.columns:
                col_str = str(col)
                
                # Check if this column matches a main data column (segment-level rationale)
                if col_str in main_columns:
                    cols_to_keep = key_cols + [col_str]
                    rationale_dict[col_str] = rationale_df[cols_to_keep].copy()
                # Check if column ends with _rationale (session-level rationale)
                elif col_str.lower().endswith('_rationale'):
                    # Extract metric name by removing _rationale suffix
                    metric_name = col_str.rsplit('_rationale', 1)[0]
                    val = rationale_df[col_str].iloc[0] if len(rationale_df) > 0 else None
                    if pd.notna(val) and str(val).strip():
                        session_rationale[metric_name] = str(val).strip()
            
            # Store session-level rationale under special key
            if session_rationale:
                rationale_dict['_session_rationale'] = session_rationale
                
            # Also check for 'data' column matching main columns
            if 'data' in rationale_df.columns:
                name_cols = [c for c in rationale_df.columns if any(kw in c.lower() for kw in ['name', 'field', 'column', 'metric'])]
                if name_cols:
                    for col in main_columns:
                        if col not in rationale_dict:
                            matching_rows = rationale_df[rationale_df[name_cols[0]].astype(str).str.lower() == col.lower()]
                            if not matching_rows.empty:
                                cols_to_keep = key_cols + ['data']
                                rationale_dict[col] = matching_rows[cols_to_keep].copy()
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
                    
                    # Preserve segment_id or index if present for matching
                    key_cols = []
                    if 'segment_id' in rationale_df.columns:
                        key_cols.append('segment_id')
                    if 'index' in rationale_df.columns:
                        key_cols.append('index')
                    
                    # Check if there's a 'data' column, use it if available
                    if 'data' in rationale_df.columns:
                        cols_to_keep = key_cols + ['data']
                        rationale_dict[matching_column] = rationale_df[cols_to_keep].copy()
                    else:
                        # Keep all columns including key columns
                        rationale_dict[matching_column] = rationale_df
        
        return main_df, rationale_dict
    
    except Exception:
        logger.exception("Error loading xlsx with sheets")
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
            main_df, rationale_dict = load_xlsx_with_sheets(transcript_file)
            
            if main_df is None:
                return None, {}
            
            main_data = main_df.to_json(date_format='iso', orient='split')
            
            # Convert rationale DataFrames to JSON
            rationale_data_dict = {}
            for col_name, rationale_df in rationale_dict.items():
                rationale_data_dict[col_name] = rationale_df.to_json(date_format='iso', orient='split')
            
            return main_data, rationale_data_dict
    
    except Exception:
        logger.exception("Error loading session with rationale")
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
            main_df, rationale_dict = load_xlsx_with_sheets(io.BytesIO(decoded))
            
            if main_df is None:
                return None, {}, f"❌ Error loading {filename}: Could not read main sheet"
            
            main_data = main_df.to_json(date_format='iso', orient='split')
            
            # Convert rationale data to JSON-serializable format
            rationale_data_dict = {}
            for col_name, rationale_val in rationale_dict.items():
                if col_name == '_session_rationale':
                    # Session-level rationale is already a plain dict
                    rationale_data_dict[col_name] = rationale_val
                elif isinstance(rationale_val, pd.DataFrame):
                    rationale_data_dict[col_name] = rationale_val.to_json(date_format='iso', orient='split')
                else:
                    rationale_data_dict[col_name] = rationale_val
            
            rationale_count = len(rationale_dict)
            status_msg = f"✅ Transcript loaded: {filename} ({len(main_df)} segments"
            if rationale_count > 0:
                status_msg += f", {rationale_count} rationale sheet(s)"
            status_msg += ")"
            
            return main_data, rationale_data_dict, status_msg
    
    except Exception as e:
        return None, {}, f"❌ Error loading {filename}: {str(e)}"


def extract_metadata_from_session(file_buffer, filename):
    """
    Extract metadata summary values and rationale from a session file.
    
    Structure based on user inputs and file inspection:
    - Metadata Sheet (2nd sheet or 'metadata'): contains metric values in columns.
      - activation_mean, engagement_mean, TCCS_SP, TCCS_C, CTS_Cognitions, etc.
    - Rationale Sheet (3rd sheet or 'rationale'): contains rationale text in columns.
      - activation_rationale, engagement_rationale, CTS_Cognitions_rationale, etc.
    
    Returns:
        dict: Dictionary containing metric values and rationales
    """
    # Mapping Config: Internal Name -> { 'metric_cols': [], 'rationale_cols': [] }
    # We look for the first matching column in the list.
    config = {
        'selfreflection': {
            'metric': ['selfreflection', 'self_reflection'], 
            'rationale': ['selfreflection_rationale', 'self_reflection_rationale']
        },
        'activation': {
            'metric': ['activation_mean', 'activation'], 
            'rationale': ['activation_rationale', 'activation_mean_rationale']
            # File inspection showed 'activation_mean' in metadata and 'activation_rationale' in rationale
        },
        'engagement': {
            'metric': ['engagement_mean', 'engagement'], 
            'rationale': ['engagement_rationale', 'engagement_mean_rationale']
        },
        'homework': {
            'metric': ['homework'], 
            'rationale': ['homework_rationale']
        },
        'challenging': {
            'metric': ['TCCS_SP', 'challenging'], 
            'rationale': ['TCCS_SP_rationale', 'challenging_rationale']
        },
        'supporting': {
            'metric': ['TCCS_C', 'supporting'], 
            'rationale': ['TCCS_C_rationale', 'supporting_rationale']
        },
        'cts_cognitions': {
            'metric': ['CTS_Cognitions', 'cognitions'], 
            'rationale': ['CTS_Cognitions_rationale', 'cognitions_rationale']
        },
        'cts_behaviours': {
            'metric': ['CTS_Behaviours', 'behaviours', 'behaviors'], 
            'rationale': ['CTS_Behaviours_rationale', 'behaviours_rationale']
        },
        'cts_discovery': {
            'metric': ['CTS_Discovery', 'discovery'], 
            'rationale': ['CTS_Discovery_rationale', 'discovery_rationale']
        },
        'cts_methods': {
            'metric': ['CTS_Methods', 'methods'], 
            'rationale': ['CTS_Methods_rationale', 'methods_rationale']
        }
    }
    
    result = {'filename': filename, 'session_idx': 0}
    # Initialize keys
    for key in config:
        result[key] = None
        result[f'{key}_rationale'] = None

    try:
        if filename.lower().endswith('.csv'):
            return result
        
        xl_file = pd.ExcelFile(file_buffer, engine='openpyxl')
        sheet_names = xl_file.sheet_names
        
        # 1. Identify Sheets
        meta_sheet = next((s for s in sheet_names if s.lower() == 'metadata'), None)
        if not meta_sheet and len(sheet_names) > 1:
            meta_sheet = sheet_names[1]
            
        rat_sheet = next((s for s in sheet_names if s.lower() == 'rationale'), None)
        if not rat_sheet and len(sheet_names) > 2:
            rat_sheet = sheet_names[2]
            
        if not meta_sheet:
            logger.warning(f"No metadata sheet found in {filename}")
            return result

        # 2. Extract Metrics from Metadata Sheet
        meta_df = pd.read_excel(xl_file, sheet_name=meta_sheet, engine='openpyxl')
        
        for key, conf in config.items():
            # Find matching column
            col_match = next((c for c in meta_df.columns if str(c) in conf['metric']), None)
            
            # If no exact match (case-sensitive from config), try case-insensitive
            if not col_match:
                col_match = next((c for c in meta_df.columns if str(c).lower() in [m.lower() for m in conf['metric']]), None)
            
            if col_match:
                val = meta_df[col_match].iloc[0] # Take first row
                try:
                    result[key] = float(val)
                except (ValueError, TypeError):
                    result[key] = val

        # 3. Extract Rationale from Rationale Sheet
        if rat_sheet:
            rat_df = pd.read_excel(xl_file, sheet_name=rat_sheet, engine='openpyxl')
            
            for key, conf in config.items():
                # Find matching column
                col_match = next((c for c in rat_df.columns if str(c) in conf['rationale']), None)
                
                if not col_match:
                    col_match = next((c for c in rat_df.columns if str(c).lower() in [r.lower() for r in conf['rationale']]), None)
                
                if col_match:
                    val = rat_df[col_match].iloc[0]
                    if pd.notna(val):
                        result[f'{key}_rationale'] = str(val)

        return result

    except Exception as e:
        logger.exception(f"Error extracting metadata from {filename}")
        return result


def process_sessions_upload(contents_list, filenames_list):
    """
    Process multiple session file uploads and extract metadata from each.
    
    Args:
        contents_list: List of base64 encoded file contents
        filenames_list: List of filenames
    
    Returns:
        tuple: (sessions_data_json, status_message)
            - sessions_data_json: JSON string of list with session metadata
            - status_message: Status message for UI
    """
    if not contents_list or not filenames_list:
        return None, ""
    
    # Ensure lists
    if not isinstance(contents_list, list):
        contents_list = [contents_list]
    if not isinstance(filenames_list, list):
        filenames_list = [filenames_list]
    
    sessions_data = []
    errors = []
    
    for idx, (content, filename) in enumerate(zip(contents_list, filenames_list)):
        if not filename.lower().endswith(('.xlsx', '.csv')):
            errors.append(f"⚠️ Skipped {filename}: unsupported format")
            continue
        
        try:
            _, content_string = content.split(',', 1)
            decoded = base64.b64decode(content_string)
            
            # Extract metadata
            metadata = extract_metadata_from_session(io.BytesIO(decoded), filename)
            metadata['session_number'] = idx + 1
            sessions_data.append(metadata)
            
        except Exception as e:
            errors.append(f"❌ Error with {filename}: {str(e)}")
    
    if not sessions_data:
        return None, " | ".join(errors) if errors else "No valid files uploaded"
    
    # Convert to JSON
    sessions_df = pd.DataFrame(sessions_data)
    sessions_json = sessions_df.to_json(date_format='iso', orient='split')
    
    status_parts = [f"✅ Loaded {len(sessions_data)} session(s)"]
    if errors:
        status_parts.extend(errors)
    
    return sessions_json, " | ".join(status_parts)

