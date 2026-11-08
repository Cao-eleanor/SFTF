"""
Merge per-level sentiment probabilities from the news-level file into the main sentiment table by news id.
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

# File paths
bert_path = DATA_DIR / "sentiment_data" / f"Deepseek_{args.label}_0_news_sentiment.xlsx"
fake_path = DATA_DIR / "sentiment_data" / f"Deepseek_{args.label}_sentiment.xlsx"
output_path = DATA_DIR / "sentiment_data" / f"Deepseek_{args.label}_sentiment_filled.xlsx"

df_bert = pd.read_excel(bert_path)
df_fake = pd.read_excel(fake_path)

# Select the columns needed from the news-level file, renaming id to id_news for the merge key.
bert_cols = {
    'id': 'id_news',  # rename to the merge key
    'positive_pro': 'L0_positive_pro',
    'neutral_pro': 'L0_neutral_pro',
    'negative_pro': 'L0_negative_pro',
    'sentiment_trend': 'L0_sentiment_trend'
}
# Keep only these columns and rename them.
df_bert_sub = df_bert[list(bert_cols.keys())].rename(columns=bert_cols)

# Left join: match the news-level data onto the main table.
merged = df_fake.merge(df_bert_sub, on='id_news', how='left')

# If the main table already has these columns, the merge would produce suffix conflicts,
# so drop them first so the new L0_* columns always win.
target_cols = ['L0_positive_pro', 'L0_neutral_pro', 'L0_negative_pro', 'L0_sentiment_trend']
for col in target_cols:
    if col in df_fake.columns:
        df_fake.drop(columns=col, inplace=True)

# Then merge.
merged = df_fake.merge(df_bert_sub, on='id_news', how='left')

# Save
merged.to_excel(output_path, index=False)
print(f"Matching complete, result saved to: {output_path}")
