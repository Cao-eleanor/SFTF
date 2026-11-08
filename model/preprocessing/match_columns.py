"""
Column matching / merging.

- match_sentiment_columns: left-join annotation columns into a base table by id
"""
from pathlib import Path

import pandas as pd

# Resolve paths relative to this file so the script runs from any location.
HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE.parent / "data"   # SFTF/model/data (raw / processed / split tables)
DATA_DIR = HERE.parent / "data"         # SFTF/model/data (sentiment / metapath / bert features)


def match_sentiment_columns(base_df, extra_df, rename_map, join_on):
    """Left-join ``extra_df`` columns into ``base_df`` on ``join_on``, overwriting
    columns that already exist.

    rename_map: mapping from ``extra_df`` column names to the merged column names
                (including the renamed join key)
    join_on:    the join key (the renamed key name in ``rename_map``)
    """
    extra_sub = extra_df[list(rename_map.keys())].rename(columns=rename_map)

    # drop target columns (everything but the join key) from base first to avoid suffix conflicts
    target_cols = [v for v in rename_map.values() if v != join_on]
    base = base_df.copy()
    for col in target_cols:
        if col in base.columns:
            base = base.drop(columns=col)

    return base.merge(extra_sub, on=join_on, how='left')


if __name__ == "__main__":
    extra_path = DATA_DIR / "sentiment_data" / "Deepseek_real_0_news_sentiment.xlsx"
    base_path = DATA_DIR / "sentiment_data" / "Deepseek_real_sentiment.xlsx"
    output_path = DATA_DIR / "sentiment_data" / "Deepseek_real_sentiment_filled.xlsx"

    df_extra = pd.read_excel(extra_path)
    df_base = pd.read_excel(base_path)

    rename_map = {
        'id': 'id_news',
        'positive_pro': 'L0_positive_pro',
        'neutral_pro': 'L0_neutral_pro',
        'negative_pro': 'L0_negative_pro',
        'sentiment_trend': 'L0_sentiment_trend'
    }
    merged = match_sentiment_columns(df_base, df_extra, rename_map, join_on='id_news')
    merged.to_excel(output_path, index=False)
    print(f"Match done. Saved to: {output_path}")
