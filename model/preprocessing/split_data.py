"""
Data splitting.

- split_train_val_test: random 70/20/10 split, saved as train/val/test.csv ordered by id
- split_by_chunks: split an Excel file into fixed-size row chunks
"""
import os
from pathlib import Path

import pandas as pd

# Resolve paths relative to this file so the script runs from any location.
HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE.parent / "data"   # SFTF/model/data (all data lives here)


def split_train_val_test(excel_path, prefix, seed=42, output_dir=None, id_col="id"):
    """Read ``excel_path``, split randomly into 70/20/10, sort each set by ``id_col``
    ascending, and save as ``{prefix}_train/val/test.csv``."""
    if not os.path.isfile(excel_path):
        raise FileNotFoundError(f"File not found: {excel_path}")

    df = pd.read_excel(excel_path)  # read the first sheet by default
    n = len(df)
    if n == 0:
        raise ValueError(f"File is empty, no data: {excel_path}")
    if id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' not found, actual columns: {list(df.columns)}")

    # split sizes (sum to n)
    train_n = int(round(n * 0.7))
    val_n = int(round(n * 0.2))
    test_n = n - train_n - val_n  # remainder goes to test, keeping the total unchanged

    # shuffle
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    train_df = df_shuffled.iloc[:train_n].copy()
    val_df = df_shuffled.iloc[train_n:train_n + val_n].copy()
    test_df = df_shuffled.iloc[train_n + val_n:].copy()

    def sort_by_id(d):
        id_num = pd.to_numeric(d[id_col], errors="coerce")
        d = d.assign(_id_num=id_num, _id_str=d[id_col].astype(str))
        d = d.sort_values(by=["_id_num", "_id_str"], ascending=[True, True], na_position="last")
        return d.drop(columns=["_id_num", "_id_str"]).reset_index(drop=True)

    train_df = sort_by_id(train_df)
    val_df = sort_by_id(val_df)
    test_df = sort_by_id(test_df)

    if output_dir is None:
        output_dir = os.path.dirname(excel_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, f"{prefix}_train.csv")
    val_path = os.path.join(output_dir, f"{prefix}_val.csv")
    test_path = os.path.join(output_dir, f"{prefix}_test.csv")

    # save as utf-8-sig so Excel opens the files without mojibake
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    val_df.to_csv(val_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    print(f"\n[{prefix}] total rows: {n}")
    print(f"  train: {len(train_df)} -> {train_path}")
    print(f"  val  : {len(val_df)} -> {val_path}")
    print(f"  test : {len(test_df)} -> {test_path}")
    return train_df, val_df, test_df


def split_by_chunks(excel_path, chunk_size=20000, output_dir=None):
    """Split an Excel file into multiple files of ``chunk_size`` rows (originally chaifen.py)."""
    df = pd.read_excel(excel_path, engine='openpyxl')
    total_rows = len(df)

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(excel_path), "split_by_rows")
    os.makedirs(output_dir, exist_ok=True)

    num_chunks = (total_rows + chunk_size - 1) // chunk_size  # ceil
    paths = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, total_rows)
        chunk_df = df.iloc[start:end]

        output_file = os.path.join(output_dir, f"part_{i + 1:03d}.xlsx")
        chunk_df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"Saved: {output_file} (rows {start + 1}~{end}, {len(chunk_df)} rows)")
        paths.append(output_file)

    print("==================== splitting done ====================")
    return paths


if __name__ == "__main__":
    split_train_val_test(DATASET_DIR / "splits_data" / "fake_news.xlsx", prefix="fake", seed=99, id_col="id")
    split_train_val_test(DATASET_DIR / "splits_data" / "real_news.xlsx", prefix="real", seed=99, id_col="id")
