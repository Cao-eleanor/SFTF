# -*- coding: utf-8 -*-
"""
Extract mean-pooled BERT features for Twitter-COVID19 NEWS text only (no comments).

Mirrors PHEME's pheme_bert.py with identical configuration: same
SFTF/model/pretrained/bert-base-uncased model (AutoModel + mean pooling), same output
format of one (N, 768) float32 .npy per {real,fake} x {train,val,test}, in the
exact row order of the corresponding split CSV (so downstream can pair them
positionally with metapath files).

Input : Twitter-COVID19/splits_data/{real,fake}_{train,val,test}.csv  (column `text`)
Output: Twitter-COVID19/bert_feature/{real,fake}_{train,val,test}_bert.npy
"""
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

HERE = Path(__file__).resolve().parent            # .../external_evaluation/Twitter-COVID19
SFTF_ROOT = HERE.parents[1]                        # .../SFTF
MODEL_DIR = SFTF_ROOT / "model" / "pretrained" / "bert-base-uncased"
SPLITS_DIR = HERE / "splits_data"
OUT_DIR = HERE / "bert_feature"

TEXT_COL = "text"
MAX_LENGTH = 128          # 与 PHEME 相同
BATCH_SIZE = 32
LABELS = ("real", "fake")
SPLITS = ("train", "val", "test")


def mean_pool(last_hidden_state, attention_mask):
    """Mean-pool the last hidden state over real (non-padding) tokens."""
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device, batch_size):
    """Encode a list of texts into a (len(texts), 768) float32 array."""
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        vecs.append(mean_pool(out.last_hidden_state, attention_mask).cpu().numpy())
    if not vecs:
        return np.zeros((0, 768), dtype=np.float32)
    return np.concatenate(vecs, axis=0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device} | model = {MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True).to(device)
    model.eval()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for label in LABELS:
        for split in SPLITS:
            csv_path = SPLITS_DIR / f"{label}_{split}.csv"
            df = pd.read_csv(csv_path, dtype=str, encoding="utf-8-sig")
            texts = df[TEXT_COL].fillna("").astype(str).map(str.strip).tolist()

            arr = encode_texts(model, tokenizer, texts, device, BATCH_SIZE)
            out_path = OUT_DIR / f"{label}_{split}_bert.npy"
            np.save(out_path, arr.astype(np.float32))
            print(f"[OK] {label}_{split}: rows={len(texts)} -> {out_path.name} shape={arr.shape}")

    print("[DONE] Twitter-COVID19 BERT features written to", OUT_DIR)


if __name__ == "__main__":
    main()
