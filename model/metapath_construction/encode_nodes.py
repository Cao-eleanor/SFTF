"""
Encode comment timelines into global node IDs and build the metapath input table.
"""

import argparse
import pandas as pd
from pathlib import Path

SFTF_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = SFTF_ROOT / "model" / "data"

# Process the real or fake subset (CLI --label).
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--label", choices=["real", "fake"], default="real",
                    help="Process the real or fake subset (default: real)")
args = parser.parse_args()

# ==================== Configuration ====================
INPUT_FILE = DATA_DIR / "sentiment_data" / f"Deepseek_{args.label}_sentiment.xlsx"  # source Excel path
OUTPUT_FILE = DATA_DIR / "metapath_data" / f"Deepseek_{args.label}_id.xlsx"  # output Excel path
SHEET_NAME = 0  # worksheet index or name

# Sentiment trend mapping
TREND_MAP = {
    "positive": 0,
    "neutral": 1,
    "negative": 2
}

# ==================== Read data ====================
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

# Convert time columns to datetime; unparseable values become NaT.
df['L0_news_time'] = pd.to_datetime(df['L0_news_time'], errors='coerce')
df['L1_comments_time'] = pd.to_datetime(df['L1_comments_time'], errors='coerce')
df['L2_comments_time'] = pd.to_datetime(df['L2_comments_time'], errors='coerce')

# ==================== 1. Globally encode L1_comments_id ====================
def generate_global_ids(time_series):
    """
    Encode IDs for an entire sequence:
    - NaT (empty) -> None
    - same time -> same ID
    - new time -> ID increments (starts at 0, never resets)
    """
    ids = []
    current_id = -1
    prev_time = None
    for t in time_series:
        if pd.isna(t):
            ids.append(None)
        else:
            if t != prev_time:
                current_id += 1
                prev_time = t
            ids.append(current_id)
    return ids

df['L1_comments_id'] = generate_global_ids(df['L1_comments_time'])
df['L2_comments_id'] = generate_global_ids(df['L2_comments_time'])

# ==================== 2. Compute time difference (each group uses its own news time) ====================
def calc_seconds(row, col_name):
    """Return the seconds between col_name's time and L0_news_time, or None if either is empty."""
    t = row[col_name]
    base = row['L0_news_time']
    if pd.notna(t) and pd.notna(base):
        return (t - base).total_seconds()
    return None

df['L1_L0_s'] = df.apply(calc_seconds, axis=1, col_name='L1_comments_time')
df['L2_L0_s'] = df.apply(calc_seconds, axis=1, col_name='L2_comments_time')

# ==================== 3. Trend mapping ====================
df['L0_news_trend'] = df['L0_sentiment_trend'].map(TREND_MAP)
df['L1_comments_trend'] = df['L1_sentiment_trend'].map(TREND_MAP)
df['L2_comments_trend'] = df['L2_sentiment_trend'].map(TREND_MAP)

# ==================== 4. Build the output table ====================
output_columns = [
    'id',
    'L0_news_url',
    'id_news',                # renamed to L0_news_id below
    'L0_news_trend',
    'L1_comments_id',
    'L1_L0_s',
    'L1_comments_trend',
    'L2_comments_id',
    'L2_L0_s',
    'L2_comments_trend',
]

# If the id_news column is missing, fall back to another column (assumed to exist here).
if 'id_news' not in df.columns:
    # If absent, L0_news_url could serve as the news ID, though it changes the column's meaning.
    # Here we simply fall back to the URL; adjust according to the actual columns.
    df['id_news'] = df['L0_news_url']
    print("Warning: 'id_news' column not found, using 'L0_news_url' instead. Please check.")

output_df = df[output_columns].rename(columns={'id_news': 'L0_news_id'})

# Save
output_df.to_excel(OUTPUT_FILE, index=False)
print(f"Done! Global sequential encoding applied (empty skipped, duplicates copied); result saved to {OUTPUT_FILE}")
