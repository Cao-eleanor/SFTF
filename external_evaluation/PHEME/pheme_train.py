# -*- coding: utf-8 -*-
"""
PHEME SFTF fake-news detection — final training + evaluation.

Final configuration (best / frozen result):
  * best-checkpoint selection by validation AUC + early stopping (patience=12)
  * BERT perturbation disabled (drop / mask / noise = 0)
  * fake-class weighted BCE loss: pos_weight = 2.0
      — counters the precision >> recall imbalance caused by the real:fake ~ 1.67:1
        class ratio; larger weights over-push and degrade AUC.

Thresholds are selected ONLY on the validation set (balanced@ and max_f1@),
then fixed for test. Reports Accuracy / Precision / Recall / F1 + AUC.
"""

import os
import random
import json
import numpy as np
from pathlib import Path

# ==========================================
# 0. Reproducibility Settings
# ==========================================

SEED = 8

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

torch.set_num_threads(1)

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

import pandas as pd


def set_seed(seed=8):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


set_seed(SEED)

HERE = Path(__file__).resolve().parent          # .../external_evaluation/PHEME
PREFIX = "pheme"

# ==========================================
# 1. Configuration Parameters
# ==========================================

CONFIG = {
    'paths': {
        'meta_r': HERE / 'metapath_data' / f'{PREFIX}_real_metapaths.txt',
        'meta_f': HERE / 'metapath_data' / f'{PREFIX}_fake_metapaths.txt',
        'ids_r': HERE / 'metapath_data' / f'{PREFIX}_real_news_ids.csv',
        'ids_f': HERE / 'metapath_data' / f'{PREFIX}_fake_news_ids.csv',
    },

    'bert_paths': {
        'tr_r': HERE / 'bert_feature' / 'real_train_bert.npy',
        'va_r': HERE / 'bert_feature' / 'real_val_bert.npy',
        'te_r': HERE / 'bert_feature' / 'real_test_bert.npy',
        'tr_f': HERE / 'bert_feature' / 'fake_train_bert.npy',
        'va_f': HERE / 'bert_feature' / 'fake_val_bert.npy',
        'te_f': HERE / 'bert_feature' / 'fake_test_bert.npy',
    },

    'split_paths': {
        'tr_r': HERE / 'splits_data' / 'real_train.csv',
        'va_r': HERE / 'splits_data' / 'real_val.csv',
        'te_r': HERE / 'splits_data' / 'real_test.csv',
        'tr_f': HERE / 'splits_data' / 'fake_train.csv',
        'va_f': HERE / 'splits_data' / 'fake_val.csv',
        'te_f': HERE / 'splits_data' / 'fake_test.csv',
    },

    'num_channels': 36,
    'num_time_bins': 100,
    'max_time': 172800,

    'k_selected': 12,
    'embed_dim': 64,
    'metric_dim': 128,
    'bert_dim': 768,

    'batch_size': 32,
    'learning_rate': 0.0003,

    'epochs': 100,

    'seed': 8,

    'Beta_and_delta': 0.5,
    'delta': 2.0,
    'eta': 1.0,

    # BERT branch perturbation disabled.
    'bert_weight': 0.2,
    'bert_drop_sample_prob': 0.0,
    'bert_mask_prob': 0.0,
    'bert_noise_std': 0.0,

    # fake-class (label=1) weight in the BCE term — final value.
    'pos_weight': 2.0,

    # best-checkpoint by validation AUC + early stopping.
    'early_stop_patience': 12,

    'target_score': 0.94,

    'threshold_min': 0.20,
    'threshold_max': 0.70,
    'threshold_step': 0.005,

    'save_path': HERE / 'checkpoints' / f'{PREFIX}_best.pth',
    'threshold_save_path': HERE / 'checkpoints' / f'{PREFIX}_threshold.json',
}


# ==========================================
# 2. Data Processing
# ==========================================

def build_signal_matrix(lines):
    matrix = np.zeros(
        (CONFIG['num_channels'], CONFIG['num_time_bins']),
        dtype=np.float32
    )

    for line in lines:
        parts = line.split('——')

        try:
            n_sent = int(parts[0].split('_')[2])
            c1_sent = int(parts[2].split('_')[2])
            t_val = float(parts[1])

            t_idx = min(
                int((t_val / CONFIG['max_time']) * CONFIG['num_time_bins']),
                CONFIG['num_time_bins'] - 1
            )

            if len(parts) >= 5 and 'None' not in parts[4]:
                c2_sent = int(parts[4].split('_')[2])
                ch_idx = 9 + (n_sent * 9) + (c1_sent * 3) + c2_sent
            else:
                ch_idx = n_sent * 3 + c1_sent

            if ch_idx < CONFIG['num_channels']:
                matrix[ch_idx, t_idx] += 1

        except Exception:
            continue

    return np.log1p(matrix).astype(np.float32)


def parse_metapath_file(path):
    groups = {}
    order = []

    if not os.path.exists(path):
        return groups, order

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split('——')
            nid = parts[0].split('_')[1]

            if nid not in groups:
                groups[nid] = []
                order.append(nid)

            groups[nid].append(line)

    return groups, order


def load_split_bert(bert_path, label="Bert_News_pre_training"):
    if not os.path.exists(bert_path):
        raise FileNotFoundError(f"[{label}] BERT file not found: {bert_path}")

    x = np.load(bert_path).astype(np.float32)

    if x.ndim != 2:
        raise ValueError(f"[{label}] expected BERT_shape=(N,768), but got: {x.shape}")

    if x.shape[1] != CONFIG['bert_dim']:
        raise ValueError(
            f"[{label}] BERT_dim mismatch: expected {CONFIG['bert_dim']}, but got {x.shape[1]}"
        )

    return x


def load_meta_map(meta_key, ids_key):
    """Parse a full metapath file and the news-id -> metapath-nid mapping."""
    groups, _ = parse_metapath_file(CONFIG['paths'][meta_key])

    nid_df = pd.read_csv(CONFIG['paths'][ids_key], dtype=str)
    if 'original_news_id' in nid_df.columns:
        orig_to_nid = dict(zip(nid_df['original_news_id'], nid_df['news_id']))
    else:
        orig_to_nid = dict(zip(nid_df['news_id'], nid_df['news_id']))

    return groups, orig_to_nid


def build_freq_for_split(groups, orig_to_nid, split_csv, label_str):
    """Build (N, 36, 100) signal matrices in the split CSV's news order."""
    split_df = pd.read_csv(split_csv, dtype=str)
    mats = []
    for oid in split_df['news_id']:
        nid = orig_to_nid.get(oid)
        if nid is None:
            raise ValueError(f"[{label_str}] news_id {oid!r} not found in metapath mapping")
        mats.append(build_signal_matrix(groups.get(nid, [])))
    return np.array(mats, dtype=np.float32)


def load_dataset():
    groups_r, map_r = load_meta_map('meta_r', 'ids_r')
    groups_f, map_f = load_meta_map('meta_f', 'ids_f')

    def one_split(split_label, name):
        r_freq = build_freq_for_split(
            groups_r, map_r, CONFIG['split_paths'][f'{split_label}_r'], f"REAL {name}")
        r_bert = load_split_bert(CONFIG['bert_paths'][f'{split_label}_r'], f"REAL {name}")

        f_freq = build_freq_for_split(
            groups_f, map_f, CONFIG['split_paths'][f'{split_label}_f'], f"FAKE {name}")
        f_bert = load_split_bert(CONFIG['bert_paths'][f'{split_label}_f'], f"FAKE {name}")

        if r_freq.shape[0] != r_bert.shape[0]:
            raise ValueError(f"[{name}] REAL count mismatch: freq={r_freq.shape[0]} bert={r_bert.shape[0]}")
        if f_freq.shape[0] != f_bert.shape[0]:
            raise ValueError(f"[{name}] FAKE count mismatch: freq={f_freq.shape[0]} bert={f_bert.shape[0]}")

        xf = np.concatenate([r_freq, f_freq], axis=0).astype(np.float32)
        xb = np.concatenate([r_bert, f_bert], axis=0).astype(np.float32)
        y = np.array([0] * len(r_freq) + [1] * len(f_freq), dtype=np.float32)
        print(f"[DATA] {name}: real={len(r_freq)} fake={len(f_freq)} total={len(y)}")
        return xf, xb, y

    return (
        one_split('tr', 'TRAIN'),
        one_split('va', 'VAL'),
        one_split('te', 'TEST'),
    )


class HybridDataset(Dataset):
    def __init__(self, f, b, l):
        self.f = f
        self.b = b
        self.l = l

    def __len__(self):
        return len(self.l)

    def __getitem__(self, i):
        return (
            torch.tensor(self.f[i], dtype=torch.float32),
            torch.tensor(self.b[i], dtype=torch.float32),
            torch.tensor(self.l[i], dtype=torch.float32),
        )


# ==========================================
# 3. Model Architecture
# ==========================================

class MetaTrendAttentionEncoder(nn.Module):
    def __init__(self, num_channels, k_selected, embed_dim):
        super(MetaTrendAttentionEncoder, self).__init__()

        self.path_projection = nn.Linear(k_selected * 2, embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            batch_first=True
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x_signal):
        fft_res = torch.fft.rfft(x_signal, dim=-1)
        selected_fft = fft_res[:, :, :CONFIG['k_selected']]
        meta_trend_coeffs = torch.cat([selected_fft.real, selected_fft.imag], dim=-1)

        path_tokens = self.path_projection(meta_trend_coeffs)

        attn_output, _ = self.attention(
            path_tokens,
            path_tokens,
            path_tokens
        )

        h_t = self.norm(attn_output + path_tokens)

        return torch.mean(h_t, dim=1)


class ManifoldFakeNewsNet(nn.Module):
    def __init__(self):
        super(ManifoldFakeNewsNet, self).__init__()

        self.struct_encoder = MetaTrendAttentionEncoder(
            CONFIG['num_channels'],
            CONFIG['k_selected'],
            CONFIG['embed_dim']
        )

        self.bert_fc = nn.Sequential(
            nn.Linear(CONFIG['bert_dim'], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, CONFIG['embed_dim'])
        )

        self.phi = nn.Sequential(
            nn.Linear(CONFIG['embed_dim'] * 2, 256),
            nn.ReLU(),
            nn.Linear(256, CONFIG['metric_dim'])
        )

        self.center = nn.Parameter(torch.randn(1, CONFIG['metric_dim']))
        self.tau = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x_freq, x_bert):
        h_t = self.struct_encoder(x_freq)

        if self.training:
            p_drop = CONFIG.get('bert_drop_sample_prob', 0.0)

            if p_drop > 0:
                drop = (
                    torch.rand(x_bert.size(0), 1, device=x_bert.device) < p_drop
                ).float()
                x_bert = x_bert * (1.0 - drop)

            p_mask = CONFIG.get('bert_mask_prob', 0.0)

            if p_mask > 0:
                keep = (torch.rand_like(x_bert) > p_mask).float()
                x_bert = x_bert * keep

            x_bert = x_bert + torch.randn_like(x_bert) * CONFIG.get('bert_noise_std', 0.0)

        h_s = self.bert_fc(x_bert)

        h_s = h_s * CONFIG.get('bert_weight', 1.0)

        h_i = torch.cat([h_t, h_s], dim=1)
        phi_h = self.phi(h_i)

        d_i = torch.norm(
            phi_h - self.center,
            p=2,
            dim=1
        )

        y_hat = torch.sigmoid(CONFIG['eta'] * (d_i - self.tau))

        return y_hat, d_i


# ==========================================
# 4. Loss Function (fake-class weighted BCE)
# ==========================================

class ManifoldLoss(nn.Module):
    def __init__(self, beta=0.5, delta=2.0, pos_weight=1.0):
        super(ManifoldLoss, self).__init__()

        self.beta = beta
        self.delta = delta
        self.pos_weight = pos_weight
        self.bce = nn.BCELoss(reduction='none')

    def forward(self, y_hat, y_true, d_i):
        elementwise = self.bce(y_hat, y_true)
        w = torch.where(y_true > 0.5, self.pos_weight, 1.0)
        l_ce = (elementwise * w).mean()

        real_mask = (y_true == 0).float()
        fake_mask = (y_true == 1).float()

        dist_real = real_mask * torch.pow(d_i, 2)
        dist_fake = fake_mask * torch.pow(F.relu(self.delta - d_i), 2)

        l_dist = torch.mean(dist_real + dist_fake)

        return l_ce + self.beta * l_dist


# ==========================================
# 5. Evaluation & threshold selection
# ==========================================

def evaluate(model, loader, device, threshold=0.5):
    model.eval()

    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():
        for x_f, x_b, y in loader:
            out, _ = model(x_f.to(device), x_b.to(device))

            prob = out.detach().cpu().numpy()
            pred = (out > threshold).float().cpu().numpy()

            y_score.extend(prob.tolist())
            y_pred.extend(pred.tolist())
            y_true.extend(y.cpu().numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float('nan')

    return acc, prec, rec, f1, auc


def _threshold_grid():
    return np.arange(
        CONFIG['threshold_min'],
        CONFIG['threshold_max'] + CONFIG['threshold_step'],
        CONFIG['threshold_step']
    )


def select_balanced_threshold(model, loader, device):
    """Reference method: minimize a target_score / balance penalty on validation."""
    best_threshold = 0.5
    best_score = float("inf")
    best_metrics = None

    target = CONFIG['target_score']

    for th in _threshold_grid():
        acc, prec, rec, f1, auc = evaluate(model, loader, device, threshold=float(th))

        target_penalty = (
            abs(acc - target)
            + abs(prec - target)
            + abs(rec - target)
            + abs(f1 - target)
        )

        balance_penalty = (
            abs(prec - rec)
            + abs(prec - f1)
            + abs(rec - f1)
        )

        min_metric = min(acc, prec, rec, f1)
        low_metric_penalty = max(0.0, target - min_metric) * 2.0

        score = target_penalty + 1.2 * balance_penalty + low_metric_penalty

        if score < best_score:
            best_score = score
            best_threshold = float(th)
            best_metrics = (acc, prec, rec, f1, auc)

    return best_threshold, best_metrics


def select_max_f1_threshold(model, loader, device):
    """Pick the validation threshold maximizing F1."""
    best_threshold = 0.5
    best_f1 = -1.0
    best_metrics = None

    for th in _threshold_grid():
        acc, prec, rec, f1, auc = evaluate(model, loader, device, threshold=float(th))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(th)
            best_metrics = (acc, prec, rec, f1, auc)

    return best_threshold, best_metrics


def _fmt(name, m):
    acc, prec, rec, f1, auc = m
    return (f"{name}: Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1:.4f} AUC={auc:.4f}")


# ==========================================
# 6. Training (best checkpoint + early stopping)
# ==========================================

def main():
    set_seed(CONFIG['seed'])

    device = torch.device('cpu')

    print(f"[CONFIG] pos_weight (fake-class BCE weight) = {CONFIG['pos_weight']}")

    train_data, val_data, test_data = load_dataset()

    g = torch.Generator()
    g.manual_seed(CONFIG['seed'])

    train_loader = DataLoader(
        HybridDataset(*train_data),
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=0,
        generator=g
    )

    val_loader = DataLoader(
        HybridDataset(*val_data),
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        HybridDataset(*test_data),
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )

    set_seed(CONFIG['seed'])

    model = ManifoldFakeNewsNet().to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate']
    )

    criterion = ManifoldLoss(
        beta=CONFIG['Beta_and_delta'],
        delta=CONFIG['delta'],
        pos_weight=CONFIG['pos_weight']
    )

    save_path = CONFIG['save_path']
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    threshold_save_path = CONFIG['threshold_save_path']
    threshold_save_dir = os.path.dirname(threshold_save_path)
    if threshold_save_dir and not os.path.exists(threshold_save_dir):
        os.makedirs(threshold_save_dir, exist_ok=True)

    best_val_auc = -1.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0.0

        for x_f, x_b, labels in train_loader:
            x_f = x_f.to(device)
            x_b = x_b.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            y_hat, d_i = model(x_f, x_b)

            loss = criterion(y_hat, labels, d_i)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        val_acc, val_prec, val_rec, val_f1, val_auc = evaluate(
            model, val_loader, device, threshold=0.5
        )

        improved = val_auc > best_val_auc
        if improved:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            marker = " *"
        else:
            patience_counter += 1
            marker = ""

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Loss: {train_loss / max(1, len(train_loader)):.4f} | "
            f"Val Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | "
            f"F1: {val_f1:.4f} | AUC: {val_auc:.4f}{marker}"
        )

        if patience_counter >= CONFIG['early_stop_patience']:
            print(f">>> Early stopping at epoch {epoch + 1} "
                  f"(no val-AUC improvement for {patience_counter} epochs).")
            break

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    print(f"\n>>> Loaded best checkpoint from epoch {best_epoch} "
          f"(best val AUC = {best_val_auc:.4f}).")

    print("\n>>> Selecting thresholds on the validation set...")
    bal_th, bal_val = select_balanced_threshold(model, val_loader, device)
    f1_th, f1_val = select_max_f1_threshold(model, val_loader, device)

    print(f"[VAL] " + _fmt(f"balanced@{bal_th:.3f}", bal_val))
    print(f"[VAL] " + _fmt(f"max_f1@{f1_th:.3f}", f1_val))

    threshold_info = {
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "pos_weight": CONFIG['pos_weight'],
        "early_stop_patience": CONFIG['early_stop_patience'],
        "balanced_threshold": bal_th,
        "balanced_val": dict(zip(["accuracy", "precision", "recall", "f1", "auc"], bal_val)),
        "max_f1_threshold": f1_th,
        "max_f1_val": dict(zip(["accuracy", "precision", "recall", "f1", "auc"], f1_val)),
        "target_score": CONFIG['target_score'],
        "bert_weight": CONFIG['bert_weight'],
        "seed": CONFIG['seed'],
    }

    with open(threshold_save_path, "w", encoding="utf-8") as f:
        json.dump(threshold_info, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 52)
    print("Final Test Evaluation (thresholds fixed on validation)")
    print("=" * 52)

    bal_test = evaluate(model, test_loader, device, threshold=bal_th)
    f1_test = evaluate(model, test_loader, device, threshold=f1_th)

    print("[TEST] " + _fmt(f"balanced@{bal_th:.3f}", bal_test))
    print("[TEST] " + _fmt(f"max_f1@{f1_th:.3f}", f1_test))
    print("=" * 52)


if __name__ == '__main__':
    main()
