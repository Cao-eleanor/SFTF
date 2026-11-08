"""
ID / URL encoding (downstream node encoding; NOT de-identification).

- encode_url_column: factorize a column into sequential integer codes
- encode_file: read an Excel file, encode a column, and save

Note: this uses ``pd.factorize`` (sequential integers) for downstream node
encoding. The HMAC-SHA256 de-identification of raw URLs/IDs is a separate,
out-of-repo step; this script does not perform or reproduce that step.
"""
import os
from pathlib import Path

import pandas as pd

# Resolve paths relative to this file so the script runs from any location.
HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE.parent / "data"   # SFTF/model/data (raw / processed / split tables)
DATA_DIR = HERE.parent / "data"         # SFTF/model/data (sentiment / metapath / bert features)


def encode_url_column(df, col='post_url', new_col='post_url_code', move_first=True):
    """Assign sequential integer codes (factorize, starting at 0) to a column
    in order of first appearance."""
    df = df.copy()
    df[new_col] = pd.factorize(df[col])[0]
    if move_first and new_col in df.columns:
        df = df[[new_col] + [c for c in df.columns if c != new_col]]
    return df


def encode_file(input_path, output_path=None, col='post_url', new_col='post_url_code'):
    """Read an Excel file, encode a column, and save."""
    df = pd.read_excel(input_path)
    df = encode_url_column(df, col=col, new_col=new_col)

    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(os.path.dirname(input_path), f"{base_name}_encoded1.xlsx")

    df.to_excel(output_path, index=False)
    print(f"Encoding done. Generated '{new_col}' from '{col}'.")
    print(f"{df[new_col].nunique()} unique URLs, code range 0 ~ {df[new_col].max()}")
    print(f"Saved to: {output_path}")
    return df


if __name__ == "__main__":
    encode_file(
        DATA_DIR / "sentiment_data" / "Deepseek_real_2_comments_sentiment.xlsx",
    )
