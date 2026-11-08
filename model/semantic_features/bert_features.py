"""
Continue-pretrain a BERT masked-LM model on TRAIN news text, then export
mean-pooled vectors for train/val/test for fake-news detection.

Design note (split-isolation):
    The MLM continue-pretraining step is applied to the TRAIN split only. The
    validation and test features are then extracted from that train-only model,
    so the val/test texts are never seen by the model during pretraining. This
    avoids cross-split leakage (the previous version pretrained on the val split
    and exported val features from the same model).

    The input tables must contain the original ``news_content`` text column,
    which is NOT present in the published de-identified data (that text is
    withheld for privacy). Researchers must supply their own original text.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from pathlib import Path

SFTF_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = SFTF_ROOT / "model" / "data"
DATASET_DIR = SFTF_ROOT / "model" / "data"

# =========================
# CONFIG: hardcode the parameters here
# =========================
CONFIG = {
    "model_dir": SFTF_ROOT / "model" / "pretrained" / "bert-base-chinese",
    "text_col": "news_content",

    # Input text tables for each split (require the original ``news_content``
    # column; absent from the de-identified data published in this repository).
    "splits": {
        "train": {
            "fake": DATASET_DIR / "splits_data" / "fake_train.csv",
            "real": DATASET_DIR / "splits_data" / "real_train.csv",
        },
        "val": {
            "fake": DATASET_DIR / "splits_data" / "fake_val.csv",
            "real": DATASET_DIR / "splits_data" / "real_val.csv",
        },
        "test": {
            "fake": DATASET_DIR / "splits_data" / "fake_test.csv",
            "real": DATASET_DIR / "splits_data" / "real_test.csv",
        },
    },

    "out_dir": SFTF_ROOT / "results" / "mlm_out",
    "vec_out_dir": DATA_DIR / "bert_feature",

    "seq_len": 64,
    "epochs": 2,
    "batch_size": 16,
    "lr": 5e-5,
    "mlm_prob": 0.15,
    "seed": 2,

    "real_windows": 1,
    "export_batch_size": 64,
    "num_workers": 0,
}


def read_texts(path: str, text_col: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path, engine="openpyxl")
    elif ext == ".csv":
        # Chinese text data often uses utf-8-sig or gbk; fall back automatically here.
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="gbk")
    else:
        raise ValueError(f"unsupported file type: {ext} (only xlsx/xls/csv are supported)")

    if text_col not in df.columns:
        raise ValueError(f"column not found: {text_col}; current columns: {list(df.columns)}")

    texts = df[text_col].dropna().astype(str).map(lambda x: x.strip()).tolist()
    texts = [t for t in texts if t]
    return texts



class CroppedMLMDataset(Dataset):
    def __init__(self, texts, tokenizer, seq_len=128, crop_strategy="random", seed=42):
        self.tok = tokenizer
        self.seq_len = seq_len
        self.crop_strategy = crop_strategy
        self.rng = random.Random(seed)

        self.encoded = [self.tok.encode(t, add_special_tokens=False) for t in texts]
        self.payload_len = max(1, self.seq_len - 2)

    def __len__(self):
        return len(self.encoded)

    def _crop(self, ids):
        if len(ids) <= self.payload_len:
            return ids
        if self.crop_strategy == "first":
            return ids[: self.payload_len]
        start = self.rng.randint(0, len(ids) - self.payload_len)
        return ids[start: start + self.payload_len]

    def __getitem__(self, idx):
        ids = self._crop(self.encoded[idx])
        input_ids = [self.tok.cls_token_id] + ids + [self.tok.sep_token_id]
        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class ConcatDS(Dataset):
    def __init__(self, a, b):
        self.a, self.b = a, b

    def __len__(self):
        return len(self.a) + len(self.b)

    def __getitem__(self, idx):
        if idx < len(self.a):
            return self.a[idx]
        return self.b[idx - len(self.a)]


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-6)
    return summed / counts


@torch.no_grad()
def export_vectors(
    encoder_dir: str,
    fake_texts,
    real_texts,
    out_fake: str,
    out_real: str,
    seq_len: int,
    real_windows: int,
    batch_size: int,
    num_workers: int,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(encoder_dir, local_files_only=True)
    enc_model = AutoModel.from_pretrained(encoder_dir, local_files_only=True).to(device)
    enc_model.eval()

    def embed_texts(texts, crop_strategy="first", windows=1, desc="Embedding"):
        all_vecs = []
        for w in range(windows):
            ds = CroppedMLMDataset(
                texts=texts,
                tokenizer=tokenizer,
                seq_len=seq_len,
                crop_strategy=crop_strategy,
                seed=1234 + w,
            )
            collate = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            dl = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate,
                num_workers=num_workers,
            )

            vecs = []
            pbar = tqdm(dl, total=len(dl), desc=f"{desc} (window {w+1}/{windows})", ncols=100)
            for batch in pbar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = enc_model(input_ids=input_ids, attention_mask=attention_mask)
                pooled = mean_pool(outputs.last_hidden_state, attention_mask)
                vecs.append(pooled.cpu().numpy())

            all_vecs.append(np.concatenate(vecs, axis=0))

        if windows == 1:
            return all_vecs[0]
        return np.mean(np.stack(all_vecs, axis=0), axis=0)

    print("[INFO] Export fake vectors ...")
    fake_vecs = embed_texts(fake_texts, crop_strategy="first", windows=1, desc="Fake vectors")

    print("[INFO] Export real vectors ...")
    real_vecs = embed_texts(real_texts, crop_strategy="first", windows=real_windows, desc="Real vectors")

    np.save(out_fake, fake_vecs)
    np.save(out_real, real_vecs)
    print(f"[OK] saved fake vectors: {out_fake} shape={fake_vecs.shape}")
    print(f"[OK] saved real vectors: {out_real} shape={real_vecs.shape}")


def main():
    cfg = CONFIG
    set_seed(cfg["seed"])
    os.makedirs(cfg["out_dir"], exist_ok=True)
    os.makedirs(cfg["vec_out_dir"], exist_ok=True)

    # 1. Load texts for every split.
    texts = {
        split: {
            label: read_texts(cfg["splits"][split][label], cfg["text_col"])
            for label in ("fake", "real")
        }
        for split in ("train", "val", "test")
    }

    if len(texts["train"]["fake"]) == 0 or len(texts["train"]["real"]) == 0:
        raise ValueError(
            f"empty train data: fake={len(texts['train']['fake'])} "
            f"real={len(texts['train']['real'])}"
        )

    for split in ("train", "val", "test"):
        print(f"[INFO] {split}: fake={len(texts[split]['fake'])} real={len(texts[split]['real'])}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_dir"], local_files_only=True)
    mlm_model = AutoModelForMaskedLM.from_pretrained(cfg["model_dir"], local_files_only=True)

    # 2. MLM continue-pretraining on the TRAIN split only (no val/test leakage).
    fake_ds = CroppedMLMDataset(
        texts["train"]["fake"], tokenizer, seq_len=cfg["seq_len"], crop_strategy="first", seed=cfg["seed"]
    )
    real_ds = CroppedMLMDataset(
        texts["train"]["real"], tokenizer, seq_len=cfg["seq_len"], crop_strategy="random", seed=cfg["seed"]
    )
    train_ds = ConcatDS(fake_ds, real_ds)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=cfg["mlm_prob"]
    )

    train_args = TrainingArguments(
        output_dir=cfg["out_dir"],
        overwrite_output_dir=True,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        learning_rate=cfg["lr"],
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=cfg["num_workers"],
    )

    trainer = Trainer(
        model=mlm_model,
        args=train_args,
        train_dataset=train_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # The Transformers Trainer shows a tqdm progress bar by default (no extra code needed).
    print("[INFO] Start MLM continue pretraining (train split only)...")
    trainer.train()

    trainer.save_model(cfg["out_dir"])
    tokenizer.save_pretrained(cfg["out_dir"])
    print(f"[OK] MLM model saved to: {cfg['out_dir']}")

    # 3. Export features for train/val/test from the train-only-pretrained model.
    for split in ("train", "val", "test"):
        print(f"[INFO] Export {split} vectors ...")
        export_vectors(
            encoder_dir=cfg["out_dir"],
            fake_texts=texts[split]["fake"],
            real_texts=texts[split]["real"],
            out_fake=cfg["vec_out_dir"] / f"fake_{split}_bert.npy",
            out_real=cfg["vec_out_dir"] / f"real_{split}_bert.npy",
            seq_len=cfg["seq_len"],
            real_windows=cfg["real_windows"],
            batch_size=cfg["export_batch_size"],
            num_workers=cfg["num_workers"],
        )

    print("[DONE]")


if __name__ == "__main__":
    main()
