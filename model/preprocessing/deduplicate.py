"""
Row deduplication.

- deduplicate: drop duplicate rows, optionally keyed on a subset of columns
- deduplicate_file: read an Excel file, deduplicate, and save
"""
import os
from pathlib import Path

import pandas as pd

# Resolve paths relative to this file so the script runs from any location.
HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE.parent / "data"   # SFTF/model/data (raw / processed / split tables)
DATA_DIR = HERE.parent / "data"         # SFTF/model/data (sentiment / metapath / bert features)


def deduplicate(df, subset=None):
    """Drop duplicate rows. If ``subset`` is None, compare all columns;
    otherwise deduplicate on the given column subset."""
    if subset is not None:
        # keep only columns that actually exist, to avoid KeyError
        subset = [col for col in subset if col in df.columns]
        if not subset:
            raise ValueError("No dedup columns found in the table. Check column names.")
    return df.drop_duplicates(subset=subset)


def deduplicate_file(input_path, output_path=None, subset=None):
    """Read an Excel file, deduplicate, and save. Returns the deduplicated DataFrame."""
    df = pd.read_excel(input_path)
    df_dedup = deduplicate(df, subset=subset)

    if output_path is None:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        suffix = "_dedup_by_columns" if subset is not None else "_deduplicated"
        output_path = os.path.join(os.path.dirname(input_path), f"{base_name}{suffix}.xlsx")

    df_dedup.to_excel(output_path, index=False)
    print(f"Deduplication done. Rows: {len(df)} -> {len(df_dedup)}")
    if subset is not None:
        print(f"Dedup columns: {subset}")
    print(f"Saved to: {output_path}")
    return df_dedup


if __name__ == "__main__":
    # Example 1: deduplicate on all columns (originally quchong.py)
    deduplicate_file(
        DATA_DIR / "sentiment_data" / "Deepseek_fake_1_comments_sentiment.xlsx",
    )

    # Example 2: deduplicate on a column subset (originally quchong1.py)
    deduplicate_file(
        DATA_DIR / "sentiment_data" / "Deepseek_fake_2_comments_sentiment.xlsx",
        subset=[
            'id', 'post_url', 'post_time', 'comment_author_url', 'comment_time',
            'comment_likes', 'reply_time', 'reply_author_url'
        ],
    )
