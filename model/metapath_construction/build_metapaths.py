"""
Build metapath lines (N/C1/C2 sentiment nodes plus time offsets) from the encoded id table.
"""

import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

SFTF_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = SFTF_ROOT / "model" / "data"

# Process the real or fake subset (CLI --label).
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--label", choices=["real", "fake"], default="real",
                    help="Process the real or fake subset (default: real)")
args = parser.parse_args()

# 1. Load data
file_path = DATA_DIR / "metapath_data" / f"Deepseek_{args.label}_id.xlsx"
df = pd.read_excel(file_path)

# 2. Define metapath generation logic
metapaths = []

for index, row in tqdm(df.iterrows(), total=len(df), desc="Building metapaths"):
    # Fetch base fields, safely cast to integers.
    l0_id = str(int(row['L0_news_id'])) if pd.notna(row['L0_news_id']) else "None"
    l0_trend = str(int(row['L0_news_trend'])) if pd.notna(row['L0_news_trend']) else "None"

    l1_id = str(int(row['L1_comments_id'])) if pd.notna(row['L1_comments_id']) else "None"
    l1_trend = str(int(row['L1_comments_trend'])) if pd.notna(row['L1_comments_trend']) else "None"
    l1_l0_s = str(int(row['L1_L0_s'])) if pd.notna(row['L1_L0_s']) else "0"

    node_n = f"N_{l0_id}_{l0_trend}"
    node_c1 = f"C1_{l1_id}_{l1_trend}"

    # Change: also require L2_comments_trend to be non-null; build a second-level comment node
    # only when all three fields are present.
    if (pd.notna(row['L2_comments_id']) and
        pd.notna(row['L2_comments_trend']) and
        pd.notna(row['L2_L0_s'])):
        l2_id = str(int(row['L2_comments_id']))
        l2_trend = str(int(row['L2_comments_trend']))
        l2_l0_s = str(int(row['L2_L0_s']))
        node_c2 = f"C2_{l2_id}_{l2_trend}"

        path = f"{node_n}——{l1_l0_s}——{node_c1}——{l2_l0_s}——{node_c2}"
    else:
        path = f"{node_n}——{l1_l0_s}——{node_c1}"

    metapaths.append(path)

# 3. Save to text file
output_path = DATA_DIR / "metapath_data" / f"Deepseek_{args.label}_metapaths.txt"
with open(output_path, 'w', encoding='utf-8') as f:
    for path in tqdm(metapaths, desc="Saving to TXT"):
        f.write(path + '\n')

print(f"\nProcessing complete! Generated {len(metapaths)} paths, saved to: {output_path}")
