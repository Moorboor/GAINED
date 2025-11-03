from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
import os


ABS_PATH = os.path.abspath("")
DATA_PATH = os.path.join(ABS_PATH, "data", "sampled_data", "transcribe_request_NVIDIA")
AUDIO_PATH = os.path.join(ABS_PATH, "data", "sampled_data")



class patientData: 
    
    def __init__(self, patient_id):
        self.patient_id = patient_id
        self.sessions = os.listdir(os.path.join(DATA_PATH, patient_id))

    def get_patient_session_data(self, session):
        df = pd.read_excel(os.path.join(DATA_PATH, self.patient_id, self.sessions[session]), engine='openpyxl')
        return df

    def get_patient_session_audio_files(self, session):
        audio_path = os.listdir(os.path.join(AUDIO_PATH, self.patient_id))[session]
        return os.path.join(AUDIO_PATH, self.patient_id, audio_path)



app = Dash()



patient_id = "013"
session = 1
patient = patientData(patient_id)
df = patient.get_patient_session_data(session=session)
audio_path = patient.get_patient_session_audio_files(session=session)

fig = px.line(df, x="segment_id", y="LLM_T", )

print(audio_path)
app.layout = html.Div(children=[
    html.H1(
        children="Hello world!"
        ),
    html.Div(
        children=f"Patient ID: {patient_id}, Session: {session}"
    ),
    dcc.Graph(
        id="example graph",
        figure=fig
    ),
    html.H1("Audio player"),
        html.Audio(audio_path, controls=True)
    ])


app.run(debug=True)
