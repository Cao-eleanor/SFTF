"""
Annotate reply-level comment sentiment via the DeepSeek API and append probability columns.
"""

import argparse
import os
import pandas as pd
import torch
import numpy as np
import json
import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SFTF_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = SFTF_ROOT / "model" / "data"
DATASET_DIR = SFTF_ROOT / "model" / "data"

# ==========================================
# 2. Basic configuration and paths
# ==========================================
# torch only sees the designated GPU here.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device} (system GPU {os.environ.get('CUDA_VISIBLE_DEVICES')})")

# Concurrency tuned for the API: DeepSeek recommends ~20-30 to balance speed and stability.
MAX_WORKERS = 2
BASE_URL = "https://api.deepseek.com"

# Read the API key from the environment; never hard-code it in source.
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    raise RuntimeError("Set the DEEPSEEK_API_KEY environment variable before running this script.")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Process the real or fake subset (CLI --label).
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--label", choices=["real", "fake"], default="real",
                    help="Process the real or fake subset (default: real)")
args = parser.parse_args()

# NOTE: this script annotates ``cleaned_reply_content``, which requires the
# original (non-de-identified) comment/reply text. The de-identified tables in
# model/data/raw_data/ do not contain that column, so supply your own text file.
INPUT_FILE = str(DATASET_DIR / "raw_data" / f"{args.label}_comments.xlsx")
OUTPUT_FILE = str(DATA_DIR / "sentiment_data" / f"Deepseek_{args.label}_2_comments_sentiment.xlsx")

# Periodic save: generate a separate backup Excel every 20000 rows.
SAVE_INTERVAL = 20000


# ==========================================
# 3. Core API call function
# ==========================================
def get_sentiment_worker(idx, text):
    if pd.isna(text) or str(text).strip() == "":
        return idx, {
            "neg_prob": 0.0, "neu_prob": 1.0, "pos_prob": 0.0,
            "sentiment": "neutral", "confidence": 1.0
        }, "empty text skipped"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system",
                     "content": "You are a sentiment analysis expert. Analyze the comment sentiment. "
                                "Return JSON with negative_prob, neutral_prob, positive_prob, sentiment, and confidence."},
                    {"role": "user", "content": f"Content: {text}"}
                ],
                response_format={'type': 'json_object'},
                temperature=0.1,
                max_tokens=150,
                timeout=25
            )

            res = json.loads(response.choices[0].message.content)
            data = {
                "neg_prob": res.get("negative_prob", 0.0),
                "neu_prob": res.get("neutral_prob", 0.0),
                "pos_prob": res.get("positive_prob", 0.0),
                "sentiment": res.get("sentiment", "neutral"),
                "confidence": res.get("confidence", 0.0)
            }
            # Flatten newlines in the preview text.
            preview = str(text).replace('\n', ' ').strip()
            return idx, data, preview
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(1.5)  # retry delay
                continue
            return idx, {"neg_prob": 0, "neu_prob": 0, "pos_prob": 0, "sentiment": "Error", "confidence": 0}, "request failed"


# ==========================================
# 4. Thread-pool scheduling and saving logic
# ==========================================
def run_extraction():
    # Load progress (resume logic).
    if os.path.exists(OUTPUT_FILE):
        print(f">>> Output file detected, loading existing progress...")
        df = pd.read_excel(OUTPUT_FILE)
    else:
        df = pd.read_excel(INPUT_FILE)

    col_name = 'cleaned_reply_content'
    suffix = ''
    cols = {
        'neg': f'negative_pro{suffix}', 'neu': f'neutral_pro{suffix}',
        'pos': f'positive_pro{suffix}', 'label': f'sentiment_trend{suffix}',
        'conf': f'confidence{suffix}'
    }

    # Initialize missing columns.
    for c in cols.values():
        if c not in df.columns:
            df[c] = None

    # Rows still to process (empty sentiment trend).
    indices = df[df[cols['label']].isna()].index
    total_tasks = len(indices)

    if total_tasks == 0:
        print(">>> All tasks already complete; no need to run again.")
        return

    print(f"\n>>> Task started | total pending: {total_tasks} | workers: {MAX_WORKERS}")
    start_time = time.time()
    completed_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_idx = {
            executor.submit(get_sentiment_worker, idx, df.at[idx, col_name]): idx
            for idx in indices
        }

        for future in as_completed(future_to_idx):
            idx, data, preview = future.result()

            # Fill results into the DataFrame.
            df.at[idx, cols['neg']] = data['neg_prob']
            df.at[idx, cols['neu']] = data['neu_prob']
            df.at[idx, cols['pos']] = data['pos_prob']
            df.at[idx, cols['label']] = data['sentiment']
            df.at[idx, cols['conf']] = data['confidence']

            completed_count += 1

            # Compute rate.
            elapsed = time.time() - start_time
            speed = completed_count / elapsed if elapsed > 0 else 0
            eta = (total_tasks - completed_count) / speed if speed > 0 else 0

            # Live progress line [completed/total].
            print(
                f"[{completed_count}/{total_tasks}] {data['sentiment']} | {speed:.1f}/s | {int(eta)}s left | {preview[:30]}")

            # Periodic save: write a separate progress file from time to time.
            if completed_count % SAVE_INTERVAL == 0:
                step_id = completed_count // SAVE_INTERVAL
                step_xlsx = OUTPUT_FILE.replace(".xlsx", f"_step_{step_id}.xlsx")
                df.to_excel(step_xlsx, index=False)
                # Also update the main output file.
                df.to_excel(OUTPUT_FILE, index=False)
                print(f"\n--- [Auto-save] progress backed up to: {step_xlsx} ---\n")

    # Final save after all tasks.
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"\n>>> All processing complete! Final result: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_extraction()
