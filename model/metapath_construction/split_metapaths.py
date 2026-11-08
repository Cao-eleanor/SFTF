"""
Split a metapath text file into train/val/test by matching news IDs from the split CSVs.
"""

import os
import re
import pandas as pd
from pathlib import Path


def load_id_set(csv_path: str, id_col: str = "id") -> set:
    df = pd.read_csv(csv_path, dtype={id_col: str})
    if id_col not in df.columns:
        raise ValueError(f"Column not found in {csv_path}: {id_col}; actual columns: {list(df.columns)}")
    return set(df[id_col].dropna().astype(str).tolist())


def split_metapaths_by_ids(
    metapath_txt: str,
    train_ids: set,
    val_ids: set,
    test_ids: set,
    out_train_txt: str,
    out_val_txt: str,
    out_test_txt: str,
    unknown_txt: str = None,
    encoding: str = "utf-8",
):
    pat = re.compile(r"^\s*N_(\d+)_\d+")

    cnt = {"train": 0, "val": 0, "test": 0, "unknown": 0, "badline": 0, "total": 0}

    os.makedirs(os.path.dirname(out_train_txt) or ".", exist_ok=True)

    with open(metapath_txt, "r", encoding=encoding, errors="ignore") as fin, \
         open(out_train_txt, "w", encoding=encoding, newline="") as ftr, \
         open(out_val_txt, "w", encoding=encoding, newline="") as fva, \
         open(out_test_txt, "w", encoding=encoding, newline="") as fte:

        fuk = None
        if unknown_txt:
            fuk = open(unknown_txt, "w", encoding=encoding, newline="")

        for line in fin:
            cnt["total"] += 1
            s = line.rstrip("\n")
            if not s.strip():
                continue

            m = pat.match(s)
            if not m:
                cnt["badline"] += 1
                print(f"Unparseable line: {s}")
                if fuk:
                    fuk.write(s + "\n")
                continue

            news_id = m.group(1)
            news_id = str(news_id)

            if news_id in train_ids:
                ftr.write(s + "\n")
                cnt["train"] += 1
            elif news_id in val_ids:
                fva.write(s + "\n")
                cnt["val"] += 1
            elif news_id in test_ids:
                fte.write(s + "\n")
                cnt["test"] += 1
            else:
                cnt["unknown"] += 1
                if fuk:
                    fuk.write(s + "\n")

        if fuk:
            fuk.close()

    print(f"\nProcessing complete: {metapath_txt}")
    print(f"  total rows        : {cnt['total']}")
    print(f"  train hits        : {cnt['train']}")
    print(f"  val hits          : {cnt['val']}")
    print(f"  test hits         : {cnt['test']}")
    print(f"  unmatched unknown : {cnt['unknown']}")
    print(f"  unparseable bad   : {cnt['badline']}")


if __name__ == "__main__":
    SFTF_ROOT = Path(__file__).resolve().parents[2]
    data_root = SFTF_ROOT / "model" / "data"
    split_dir = SFTF_ROOT / "model" / "data" / "splits_data"
    meta_dir  = data_root / "metapath_data"

    fake_train_csv = os.path.join(split_dir, "fake_train.csv")
    fake_val_csv   = os.path.join(split_dir, "fake_val.csv")
    fake_test_csv  = os.path.join(split_dir, "fake_test.csv")
    fake_meta_txt  = os.path.join(meta_dir, "Deepseek_fake_metapaths.txt")

    fake_train_ids = load_id_set(fake_train_csv, id_col="id")
    fake_val_ids   = load_id_set(fake_val_csv,   id_col="id")
    fake_test_ids  = load_id_set(fake_test_csv,  id_col="id")

    split_metapaths_by_ids(
        metapath_txt=fake_meta_txt,
        train_ids=fake_train_ids,
        val_ids=fake_val_ids,
        test_ids=fake_test_ids,
        out_train_txt=os.path.join(meta_dir, "Deepseek_fake_train_metapaths.txt"),
        out_val_txt=os.path.join(meta_dir, "Deepseek_fake_val_metapaths.txt"),
        out_test_txt=os.path.join(meta_dir, "Deepseek_fake_test_metapaths.txt"),
        unknown_txt=os.path.join(meta_dir, "Deepseek_fake_unknown_metapaths.txt"),
    )

    # ===== Process real data =====
    real_train_csv = os.path.join(split_dir, "real_train.csv")
    real_val_csv   = os.path.join(split_dir, "real_val.csv")
    real_test_csv  = os.path.join(split_dir, "real_test.csv")
    real_meta_txt  = os.path.join(meta_dir, "Deepseek_real_metapaths.txt")

    if os.path.exists(real_meta_txt):
        real_train_ids = load_id_set(real_train_csv, id_col="id")
        real_val_ids   = load_id_set(real_val_csv,   id_col="id")
        real_test_ids  = load_id_set(real_test_csv,  id_col="id")

        split_metapaths_by_ids(
            metapath_txt=real_meta_txt,
            train_ids=real_train_ids,
            val_ids=real_val_ids,
            test_ids=real_test_ids,
            out_train_txt=os.path.join(meta_dir, "Deepseek_real_train_metapaths.txt"),
            out_val_txt=os.path.join(meta_dir, "Deepseek_real_val_metapaths.txt"),
            out_test_txt=os.path.join(meta_dir, "Deepseek_real_test_metapaths.txt"),
            unknown_txt=os.path.join(meta_dir, "Deepseek_real_unknown_metapaths.txt"),
        )
    else:
        print(f"\nHint: {real_meta_txt} not found. If the real-news metapath filename differs, please update real_meta_txt.")
