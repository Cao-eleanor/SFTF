# -*- coding: utf-8 -*-
"""
Twitter-COVID19 SFTF fake-news detection — optimized training protocol.

The MODEL architecture, signal building, and
ManifoldLoss math are byte-for-byte UNCHANGED from v1. Only the TRAINING
PROTOCOL is made configurable with improved defaults. The diagnosis behind each change:

  1. pos_weight  2.0 -> 1.0 (default)
     v1 applied pos_weight=2.0 to the FAKE class, but fake is already the
     majority (252 vs 148). This over-weights fake and inflates recall at the
     cost of precision (v1 test: Rec 0.92 / Prec 0.75, 8/14 real misclassified).
     Default here is 1.0 (unbiased); a sweep re-selects it on VALIDATION only.

  2. threshold grid  0.20-0.70 -> 0.05-0.95
     v1 selected threshold 0.695, sitting at the top of a grid capped at 0.70,
     i.e. the search was clipped. Extended here.

  3. checkpoint selection  val AUC -> val F1 (default; val AUC available)
     The task metric is F1; selecting the checkpoint that maximizes validation
     F1 aligns with the final evaluation. Both are validation-only (no leakage).

  4. extra threshold objective  F-beta (beta<1) for precision emphasis
     F0.5 / F0.2 emphasize precision, directly targeting v1's high-FP problem.

Outputs go to checkpoints/twitter_{tag}_*.

Usage:
  # single optimized run (test evaluated)
  python twitter_train.py
  python twitter_train.py --pos-weight 1.0 --select-by val_f1 --objective max_f1

  # sweep pos_weight on validation only (no test evaluated -> no leakage)
  python twitter_train.py --sweep-pos-weights "0.5,1.0,1.5,2.0"

  # evaluate test on a saved checkpoint WITHOUT retraining
  python twitter_train.py --eval-only v2_pw1
"""

import os
import random
import json
import argparse
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
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    fbeta_score,
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

HERE = Path(__file__).resolve().parent          # .../external_evaluation/Twitter-COVID19
PREFIX = "twitter"

# ==========================================
# 1. Configuration Parameters
#    (frozen model/data params in CONFIG; tunable training protocol in args)
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

    'Beta_and_delta': 0.5,
    'delta': 2.0,
    'eta': 1.0,

    # BERT branch perturbation disabled (unchanged from v1).
    'bert_weight': 0.2,
    'bert_drop_sample_prob': 0.0,
    'bert_mask_prob': 0.0,
    'bert_noise_std': 0.0,

    # balanced@ target (only used by the 'balanced' threshold objective).
    'target_score': 0.94,
}


# ==========================================
# 2. Data Processing  (identical to v1)
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
    groups, _ = parse_metapath_file(CONFIG['paths'][meta_key])

    nid_df = pd.read_csv(CONFIG['paths'][ids_key], dtype=str)
    if 'original_news_id' in nid_df.columns:
        orig_to_nid = dict(zip(nid_df['original_news_id'], nid_df['news_id']))
    else:
        orig_to_nid = dict(zip(nid_df['news_id'], nid_df['news_id']))

    return groups, orig_to_nid


def build_freq_for_split(groups, orig_to_nid, split_csv, label_str):
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
# 3. Model Architecture  (identical to v1)
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
# 4. Loss Function  (identical to v1)
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
#    (refactored: scores computed ONCE, metrics swept over thresholds)
# ==========================================

def predict_scores(model, loader, device):
    model.eval()
    y_true, y_score = [], []
    with torch.no_grad():
        for x_f, x_b, y in loader:
            out, _ = model(x_f.to(device), x_b.to(device))
            y_score.extend(out.detach().cpu().numpy().tolist())
            y_true.extend(y.cpu().numpy().tolist())
    return np.asarray(y_true), np.asarray(y_score)


def metrics_at(y_true, y_score, threshold, beta=1.0):
    pred = (y_score > threshold).astype(int)
    acc = float(accuracy_score(y_true, pred))
    prec = float(precision_score(y_true, pred, zero_division=0))
    rec = float(recall_score(y_true, pred, zero_division=0))
    f1 = float(f1_score(y_true, pred, zero_division=0))
    fb = float(fbeta_score(y_true, pred, beta=beta, zero_division=0))
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc = float('nan')
    return acc, prec, rec, f1, fb, auc


def _threshold_grid(args):
    return np.arange(
        args.threshold_min,
        args.threshold_max + args.threshold_step,
        args.threshold_step
    )


def _m5(m):
    acc, prec, rec, f1, fb, auc = m
    return {
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'auc': auc,
    }


def select_max_f1_threshold(y_true, y_score, args):
    best_th, best_f1, best_m = 0.5, -1.0, None
    for th in _threshold_grid(args):
        m = metrics_at(y_true, y_score, float(th))
        if m[3] > best_f1:
            best_f1 = m[3]
            best_th = float(th)
            best_m = m
    return best_th, best_m


def select_fbeta_threshold(y_true, y_score, args, beta):
    best_th, best_fb, best_m = 0.5, -1.0, None
    for th in _threshold_grid(args):
        m = metrics_at(y_true, y_score, float(th), beta=beta)
        if m[4] > best_fb:
            best_fb = m[4]
            best_th = float(th)
            best_m = m
    return best_th, best_m


def select_balanced_threshold(y_true, y_score, args):
    """Reference method (kept from v1): minimize target/balance penalty."""
    target = CONFIG['target_score']
    best_th, best_score, best_m = 0.5, float('inf'), None
    for th in _threshold_grid(args):
        acc, prec, rec, f1, fb, auc = metrics_at(y_true, y_score, float(th))

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
            best_th = float(th)
            best_m = (acc, prec, rec, f1, fb, auc)

    return best_th, best_m


def select_balanced_accuracy_threshold(y_true, y_score, args):
    """Maximize balanced accuracy = (recall + specificity) / 2.

    Balances the two CLASSES (real vs fake) rather than the two positive-class
    metrics (precision vs recall). This is the right objective when the goal is
    "all four metrics balanced" on an imbalanced test set.
    """
    best_th, best_ba, best_m = 0.5, -1.0, None
    for th in _threshold_grid(args):
        pred = (y_score > float(th)).astype(int)
        tp = float(((pred == 1) & (y_true == 1)).sum())
        tn = float(((pred == 0) & (y_true == 0)).sum())
        fp = float(((pred == 1) & (y_true == 0)).sum())
        fn = float(((pred == 0) & (y_true == 1)).sum())
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        ba = 0.5 * (rec + spec)
        if ba > best_ba:
            best_ba = ba
            best_th = float(th)
            best_m = metrics_at(y_true, y_score, float(th))
    return best_th, best_m


def _fmt(name, m):
    acc, prec, rec, f1, fb, auc = m
    return (f"{name}: Acc={acc:.4f} Prec={prec:.4f} Rec={rec:.4f} "
            f"F1={f1:.4f} AUC={auc:.4f}")


# ==========================================
# 6. Training (configurable protocol)
# ==========================================

def train_and_eval(pos_weight, tag, args, eval_test=True):
    set_seed(args.seed)

    device = torch.device('cpu')

    print(f"\n[RUN] tag={tag} | pos_weight={pos_weight:g} | select_by={args.select_by} "
          f"| objective={args.objective} | seed={args.seed}")

    train_data, val_data, test_data = load_dataset()

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_loader = DataLoader(
        HybridDataset(*train_data),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=g
    )
    val_loader = DataLoader(
        HybridDataset(*val_data),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        HybridDataset(*test_data),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    set_seed(args.seed)

    model = ManifoldFakeNewsNet().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    criterion = ManifoldLoss(
        beta=CONFIG['Beta_and_delta'],
        delta=CONFIG['delta'],
        pos_weight=pos_weight
    )

    save_path = HERE / 'checkpoints' / f'{PREFIX}_{tag}_best.pth'
    threshold_save_path = HERE / 'checkpoints' / f'{PREFIX}_{tag}_threshold.json'
    os.makedirs(save_path.parent, exist_ok=True)

    best_val_metric = -1.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(args.epochs):
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

        yv_true, yv_score = predict_scores(model, val_loader, device)
        acc, prec, rec, f1, fb, auc = metrics_at(yv_true, yv_score, 0.5)

        cur = f1 if args.select_by == 'val_f1' else auc

        improved = cur > best_val_metric
        if improved:
            best_val_metric = cur
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
            f"Val Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | "
            f"F1: {f1:.4f} | AUC: {auc:.4f}{marker}"
        )

        if patience_counter >= args.patience:
            print(f">>> Early stopping at epoch {epoch + 1} "
                  f"(no val-{args.select_by} improvement for {patience_counter} epochs).")
            break

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    print(f"\n>>> Loaded best checkpoint from epoch {best_epoch} "
          f"(best val {args.select_by} = {best_val_metric:.4f}).")

    # --- threshold selection on VALIDATION only ---
    yv_true, yv_score = predict_scores(model, val_loader, device)

    th_f1, f1_metrics = select_max_f1_threshold(yv_true, yv_score, args)
    th_bal, bal_metrics = select_balanced_threshold(yv_true, yv_score, args)
    th_ba, ba_metrics = select_balanced_accuracy_threshold(yv_true, yv_score, args)

    print("\n>>> Selecting thresholds on the validation set...")
    print("[VAL] " + _fmt(f"max_f1@{th_f1:.3f}", f1_metrics))
    print("[VAL] " + _fmt(f"balanced@{th_bal:.3f}", bal_metrics))
    print("[VAL] " + _fmt(f"balanced_acc@{th_ba:.3f}", ba_metrics))

    thresholds = {
        'max_f1': {'threshold': th_f1, 'val': _m5(f1_metrics)},
        'balanced': {'threshold': th_bal, 'val': _m5(bal_metrics)},
        'balanced_acc': {'threshold': th_ba, 'val': _m5(ba_metrics)},
    }

    fb_th = None
    if args.objective in ('f0.5', 'f0.2'):
        beta = 0.5 if args.objective == 'f0.5' else 0.2
        fb_th, fb_metrics = select_fbeta_threshold(yv_true, yv_score, args, beta)
        print("[VAL] " + _fmt(f"f{beta}@{fb_th:.3f}", fb_metrics))
        thresholds[f'f{beta}'] = {'threshold': fb_th, 'val': _m5(fb_metrics)}

    result = {
        'tag': tag,
        'pos_weight': pos_weight,
        'seed': args.seed,
        'select_by': args.select_by,
        'objective': args.objective,
        'best_epoch': best_epoch,
        f'best_val_{args.select_by}': best_val_metric,
        'thresholds': thresholds,
    }

    if eval_test:
        print("\n" + "=" * 52)
        print("Final Test Evaluation (thresholds fixed on validation)")
        print("=" * 52)

        yt_true, yt_score = predict_scores(model, test_loader, device)

        for name, info in thresholds.items():
            th = info['threshold']
            m = metrics_at(yt_true, yt_score, th)
            print("[TEST] " + _fmt(f"{name}@{th:.3f}", m))
            info['test'] = _m5(m)

        print("=" * 52)

    with open(threshold_save_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def eval_only(tag):
    device = torch.device('cpu')
    ckpt = HERE / 'checkpoints' / f'{PREFIX}_{tag}_best.pth'
    thr = HERE / 'checkpoints' / f'{PREFIX}_{tag}_threshold.json'

    if not ckpt.exists() or not thr.exists():
        raise FileNotFoundError(f"checkpoint/threshold not found for tag={tag}")

    _, _, test_data = load_dataset()
    test_loader = DataLoader(
        HybridDataset(*test_data),
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    model = ManifoldFakeNewsNet().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

    with open(thr, encoding='utf-8') as f:
        info = json.load(f)

    yt_true, yt_score = predict_scores(model, test_loader, device)

    print("\n" + "=" * 52)
    print(f"TEST EVAL (tag={tag}, pos_weight={info.get('pos_weight')})")
    print("=" * 52)
    for name, t in info.get('thresholds', {}).items():
        th = t['threshold']
        m = metrics_at(yt_true, yt_score, th)
        print("[TEST] " + _fmt(f"{name}@{th:.3f}", m))
    print("=" * 52)


# ==========================================
# 7. CLI entry
# ==========================================

def main():
    p = argparse.ArgumentParser(description="Twitter-COVID19 SFTF training v2 (optimized protocol).")

    p.add_argument('--pos-weight', type=float, default=1.0,
                   help="fake-class BCE weight (v1 used 2.0; fake is majority so 1.0 is unbiased)")
    p.add_argument('--threshold-min', type=float, default=0.05)
    p.add_argument('--threshold-max', type=float, default=0.95)
    p.add_argument('--threshold-step', type=float, default=0.005)
    p.add_argument('--select-by', type=str, default='val_f1', choices=['val_auc', 'val_f1'],
                   help="metric for best-checkpoint selection")
    p.add_argument('--objective', type=str, default='max_f1',
                   choices=['max_f1', 'balanced', 'balanced_acc', 'f0.5', 'f0.2'])
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--lr', type=float, default=0.0003)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--seed', type=int, default=8)
    p.add_argument('--tag', type=str, default='v2',
                   help="output suffix: checkpoints/twitter_{tag}_*")
    p.add_argument('--sweep-pos-weights', type=str, default=None,
                   help="comma list e.g. '0.5,1.0,1.5,2.0'; trains each on VALIDATION only (no test)")
    p.add_argument('--eval-only', type=str, default=None, metavar='TAG',
                   help="evaluate TEST on a saved checkpoint without retraining")

    args = p.parse_args()

    set_seed(args.seed)

    if args.eval_only:
        eval_only(args.eval_only)
        return

    if args.sweep_pos_weights:
        weights = [float(w) for w in args.sweep_pos_weights.split(',') if w.strip()]
        summary = []
        for pw in weights:
            tag = f"{args.tag}_pw{pw:g}"
            r = train_and_eval(pw, tag, args, eval_test=False)
            summary.append(r)

        print("\n" + "=" * 70)
        print("POS_WEIGHT SWEEP — VALIDATION SUMMARY (no test evaluated)")
        print("=" * 70)
        header = f"{'pos_weight':>11} {'valF1':>8} {'valPrec':>8} {'valRec':>8} {'valAUC':>8} {'thr':>6}"
        print(header)
        print("-" * 70)

        rows = []
        for r in summary:
            mf1 = r['thresholds']['max_f1']['val']
            rows.append((
                r['pos_weight'], mf1['f1'], mf1['precision'],
                mf1['recall'], mf1['auc'], r['thresholds']['max_f1']['threshold'],
            ))
        rows.sort(key=lambda x: -x[1])

        for pw, f1, prec, rec, auc, th in rows:
            print(f"{pw:>11} {f1:>8.4f} {prec:>8.4f} {rec:>8.4f} {auc:>8.4f} {th:>6.3f}")

        best = rows[0]
        print("-" * 70)
        print(f"Best (by val F1) -> pos_weight={best[0]:g}  val F1={best[1]:.4f}")
        print(f"Evaluate TEST on it (no retraining):")
        print(f"  python twitter_train.py --eval-only {args.tag}_pw{best[0]:g}")

        with open(HERE / 'checkpoints' / f'{PREFIX}_sweep_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return

    # single run (test evaluated)
    train_and_eval(args.pos_weight, args.tag, args, eval_test=True)


if __name__ == '__main__':
    main()
