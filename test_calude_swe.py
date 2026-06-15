from datasets import load_dataset

# Load individual tables (each table is exposed as a named config)
conversations = load_dataset("SALT-NLP/SWE-chat", "conversations", split="train")
sessions = load_dataset("SALT-NLP/SWE-chat", "sessions", split="train")
commits = load_dataset("SALT-NLP/SWE-chat", "commits", split="train")

# Or load with pandas directly
import pandas as pd
df_conv = pd.read_parquet("conversations.parquet")
df_sessions = pd.read_parquet("sessions.parquet")

# Raw transcripts live under transcripts/{session_id}.jsonl
# session_logs.parquet holds the relative path + auxiliary text fields
df_session_logs = pd.read_parquet("session_logs.parquet")
with open(df_session_logs["transcript_path"].iloc[0]) as f:
    raw_transcript = f.read()