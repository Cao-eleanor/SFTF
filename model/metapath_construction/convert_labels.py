"""
Convert sentiment-trend labels into numeric codes (positive=0, neutral=1, negative=2).
"""

import argparse
import pandas as pd
import os
from pathlib import Path

SFTF_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = SFTF_ROOT / "model" / "data"

# Process the real or fake subset (CLI --label).
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--label", choices=["real", "fake"], default="real",
                    help="Process the real or fake subset (default: real)")
args = parser.parse_args()

# ---------- Input file path ----------
input_path = DATA_DIR / "sentiment_data" / f"Deepseek_{args.label}_sentiment.xlsx"

# ---------- Output file path ----------
output_dir = os.path.dirname(input_path)
base_name = os.path.splitext(os.path.basename(input_path))[0]
output_path = os.path.join(output_dir, f"{base_name}_processed1.xlsx")

# ---------- Read Excel ----------
df = pd.read_excel(input_path)

# ---------- Replacement rules ----------
# NOTE: only cells exactly equal to these words are replaced (avoids substring damage).
replace_map = {
    'positive': 0,
    'neutral': 1,
    'negative': 2,
}

# ---------- Apply replacement across the whole table ----------
df_replaced = df.replace(replace_map)

# ---------- Save as a new file ----------
df_replaced.to_excel(output_path, index=False)

print("Replacement complete: sentiment labels converted to numeric codes.")
print(f"Processed file saved to: {output_path}")
