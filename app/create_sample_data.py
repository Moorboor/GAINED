"""
Create sample transcript data for testing the GAINED app
"""
import pandas as pd
import numpy as np
import os

# Create sample transcript data
np.random.seed(42)

# Generate sample therapy session data
n_segments = 50
segment_ids = list(range(n_segments))

# Generate realistic timestamps (each segment ~5-15 seconds)
durations = np.random.uniform(5, 15, n_segments)
starts = np.cumsum(np.concatenate([[0], durations[:-1]]))
ends = starts + durations

# Sample therapy dialogue
therapist_prompts = [
    "How have you been feeling this week?",
    "Can you tell me more about that?",
    "What emotions come up when you think about that?",
    "How did that make you feel?",
    "Let's explore that thought further.",
    "That's an important realization.",
    "What would you like to work on today?",
    "How are you coping with that?",
    "What support do you need?",
    "You're making great progress.",
]

patient_responses = [
    "I've been feeling a bit overwhelmed lately.",
    "It's been challenging, but I'm trying my best.",
    "Sometimes I feel anxious about the future.",
    "I think I'm understanding myself better now.",
    "That's a good question, let me think...",
    "I hadn't considered it that way before.",
    "It makes me feel both excited and nervous.",
    "I'm starting to see some patterns in my behavior.",
    "Thank you for listening and understanding.",
    "I feel like I'm making progress, even if it's slow.",
]

# Generate alternating dialogue
texts = []
speakers = []
for i in range(n_segments):
    if i % 2 == 0:
        texts.append(np.random.choice(therapist_prompts))
        speakers.append("Therapist")
    else:
        texts.append(np.random.choice(patient_responses))
        speakers.append("Patient")

# Generate LLM_T metric (therapeutic alliance metric, 0-1)
# Simulate improving therapeutic alliance over session
base_trend = np.linspace(0.5, 0.85, n_segments)
noise = np.random.normal(0, 0.05, n_segments)
LLM_T = np.clip(base_trend + noise, 0, 1)

# Generate additional metrics
engagement = np.clip(LLM_T + np.random.normal(0, 0.1, n_segments), 0, 1)
emotional_depth = np.clip(0.3 + 0.5 * (np.array(segment_ids) / n_segments) + np.random.normal(0, 0.08, n_segments), 0, 1)

# Create DataFrame
df = pd.DataFrame({
    'segment_id': segment_ids,
    'start': starts,
    'end': ends,
    'speaker': speakers,
    'text': texts,
    'LLM_T': LLM_T,
    'engagement': engagement,
    'emotional_depth': emotional_depth
})

# Save to CSV and Excel
output_dir = "app/test_video_splits/transcribe_request_NVIDIA/012"
os.makedirs(output_dir, exist_ok=True)

csv_path = os.path.join(output_dir, "sample_session_transcript.csv")
df.to_csv(csv_path, index=False)
print(f"Created sample CSV: {csv_path}")

try:
    xlsx_path = os.path.join(output_dir, "sample_session_transcript.xlsx")
    df.to_excel(xlsx_path, index=False, engine='openpyxl')
    print(f"Created sample XLSX: {xlsx_path}")
except Exception as e:
    print(f"Could not create XLSX: {e}")

print(f"\nSample data preview:")
print(df.head(10))
print(f"\nTotal segments: {len(df)}")
print(f"Session duration: {df['end'].max():.1f} seconds ({df['end'].max()/60:.1f} minutes)")
print(f"\nMetrics summary:")
print(df[['LLM_T', 'engagement', 'emotional_depth']].describe())

