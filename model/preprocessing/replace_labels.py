"""
Sentiment label -> numeric replacement.

- replace_sentiment_labels: map sentiment labels (positive/neutral/negative) to 0/1/2,
  across the whole table or for a single column
- replace_labels_in_excel: map a column in every sheet of an Excel file and write back
"""
from pathlib import Path

import pandas as pd

# Resolve paths relative to this file so the script runs from any location.
HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE.parent / "data"   # SFTF/model/data (raw / processed / split tables)
DATA_DIR = HERE.parent / "data"         # SFTF/model/data (sentiment / metapath / bert features)

# Sentiment label -> numeric code (positive=0, neutral=1, negative=2).
# The Chinese keys are legacy label strings still present in some older files.
SENTIMENT_LABEL_MAP = {
    "正向": 0, "正面": 0, "positive": 0,
    "中性": 1, "neutral": 1,
    "负向": 2, "负面": 2, "負面": 2, "negative": 2,
}


def replace_sentiment_labels(df, column=None):
    """Replace sentiment labels with numeric codes. If ``column`` is None, replace
    across the whole table (originally tihuan1.py); otherwise map only that column
    with exact matches, leaving unmatched values as <NA> (originally tihuan.py)."""
    if column is None:
        return df.replace(SENTIMENT_LABEL_MAP)
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found")

    s = df[column].astype(str).str.strip()
    s_lower = s.str.lower()
    mapped = s.map(SENTIMENT_LABEL_MAP).fillna(s_lower.map(SENTIMENT_LABEL_MAP))

    df = df.copy()
    df[column] = mapped.astype("Int64")
    return df


def replace_labels_in_excel(in_file, out_file, column='L2_sentiment_trend'):
    """(Originally tihuan.py.) Map sentiment labels to numbers for the given column
    in every sheet, and write back."""
    sheets = pd.read_excel(in_file, sheet_name=None)
    for name, df in sheets.items():
        if column in df.columns:
            sheets[name] = replace_sentiment_labels(df, column=column)

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
    print(f"Written to: {out_file}")


if __name__ == "__main__":
    # Example 1: whole-table replacement (originally tihuan1.py)
    in_file = DATA_DIR / "sentiment_data" / "Deepseek_real_sentiment.xlsx"
    out_file = DATA_DIR / "sentiment_data" / "Deepseek_real_sentiment_processed1.xlsx"
    df = pd.read_excel(in_file)
    df_replaced = replace_sentiment_labels(df)  # whole-table replacement
    df_replaced.to_excel(out_file, index=False)
    print("Sentiment labels replaced with numeric codes.")
    print(f"Saved to: {out_file}")
