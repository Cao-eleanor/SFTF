"""
Text cleaning and cell clearing.

- clean_text: clean a single comment text (strip emoji, @mentions, URLs, special chars)
- clean_comments_column: apply row-wise cleaning to a comment column, adding a new column
- clear_trend_cells: conditionally clear ``L2_comments_trend`` cells
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import emoji
from tqdm import tqdm

tqdm.pandas()

# Resolve paths relative to this file so the script runs from any location.
HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE.parent / "data"   # SFTF/model/data (all data lives here)


def clean_text(text):
    """Clean a single comment text (originally ``clean_news_content_v4``)."""
    if not isinstance(text, str) or pd.isna(text):
        return "no text content"  # must return a valid string or the model loader will fail

    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'//\s*@[\u4e00-\u9fa5a-zA-Z0-9_-]+\s*[::]?', '', text)
    text = re.sub(r'https?://[a-zA-Z0-9./?=&_-]+', '', text)
    text = text.replace('#', '')
    text = re.sub(r'[【】\[\]]', '', text)
    text = re.sub(r'@[\u4e00-\u9fa5a-zA-Z0-9_-]+', '', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：、“”（）]', ' ', text)
    text = re.sub(r'\s+', '', text).strip()
    if text == "":
        return "space or emoji-only content"

    return text


def clean_comments_column(df, target_col='comment_content', new_col='cleaned_comment_content'):
    """Clean the comment column row by row, adding a new column in place."""
    df[new_col] = df[target_col].progress_apply(clean_text)
    return df


def clear_trend_cells(file_path, output_path):
    """(Originally ``clean_trend``.) Clear ``L2_comments_trend`` for rows where
    ``L2_L0_s`` is empty (NaN/blank/whitespace) and ``L2_comments_trend`` is 1."""
    df = pd.read_excel(file_path, dtype=object)  # dtype=object avoids automatic type coercion
    empty_L2_L0 = (
        df['L2_L0_s'].isna() |
        (df['L2_L0_s'].astype(str).str.strip() == '')
    )
    trend_is_1 = df['L2_comments_trend'].astype(str).str.strip() == '1'
    mask = empty_L2_L0 & trend_is_1
    df.loc[mask, 'L2_comments_trend'] = np.nan
    df.to_excel(output_path, index=False)
    print(f"Done. Cleared {mask.sum()} rows. Saved to: {output_path}")
    return df


if __name__ == "__main__":
    # Example 1: clean comment text (originally clean_comments.py).
    # NOTE: this step requires the original (non-de-identified) comment text.
    # The de-identified tables under model/data/raw_data/ do not contain a
    # ``comment_content`` column, so supply your own original comment files here.
    input_dir = DATASET_DIR / "raw_data"
    output_dir = DATASET_DIR / "raw_data"
    files = ["fake_comments.xlsx", "real_comments.xlsx"]

    output_dir.mkdir(parents=True, exist_ok=True)

    for file_name in tqdm(files, desc="overall progress"):
        input_path = input_dir / file_name
        if not input_path.exists():
            continue
        print(f"\nLoading: {file_name}...")
        df = pd.read_excel(input_path)
        if 'comment_content' in df.columns:
            print(f"Cleaning {file_name} row by row (rows: {len(df)})...")
            clean_comments_column(df)
            output_path = output_dir / f"preprocessed_{file_name}"
            df.to_excel(output_path, index=False)
            print(f"Cleaning done. Output rows ({len(df)}) match input.")
