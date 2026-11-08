"""
Run the observation-window M and temporal-decay lambda sensitivity experiment and export its metrics and heatmap.
"""

import os
import argparse
from pathlib import Path
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==========================================
# 1. Configuration Parameters
# ==========================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[1] / 'model' / 'data'  # SFTF/model/data

CONFIG = {
    'paths': {
        'tr_r': str(DATA_DIR / 'metapath_data' / 'Deepseek_real_train_metapaths.txt'),
        'va_r': str(DATA_DIR / 'metapath_data' / 'Deepseek_real_val_metapaths.txt'),
        'te_r': str(DATA_DIR / 'metapath_data' / 'Deepseek_real_test_metapaths.txt'),

        'tr_f': str(DATA_DIR / 'metapath_data' / 'Deepseek_fake_train_metapaths.txt'),
        'va_f': str(DATA_DIR / 'metapath_data' / 'Deepseek_fake_val_metapaths.txt'),
        'te_f': str(DATA_DIR / 'metapath_data' / 'Deepseek_fake_test_metapaths.txt')
    },
    'bert_paths': {
        'tr_r': str(DATA_DIR / 'bert_feature' / 'real_train_bert.npy'),
        'va_r': str(DATA_DIR / 'bert_feature' / 'real_val_bert.npy'),
        'te_r': str(DATA_DIR / 'bert_feature' / 'real_test_bert.npy'),

        'tr_f': str(DATA_DIR / 'bert_feature' / 'fake_train_bert.npy'),
        'va_f': str(DATA_DIR / 'bert_feature' / 'fake_val_bert.npy'),
        'te_f': str(DATA_DIR / 'bert_feature' / 'fake_test_bert.npy')
    },

    # ====== Structural signal settings ======
    'num_channels': 36,
    'num_time_bins': 100,

    # ====== Model settings ======
    'k_selected': 12,
    'embed_dim': 64,
    'metric_dim': 128,
    'bert_dim': 768,

    # ====== Training settings ======
    'batch_size': 32,
    'learning_rate': 0.0003,
    'epochs': 20,               # Sensitivity sweeps typically need fewer than 100 epochs.
    'seed': 8,

    # ====== Loss settings ======
    'Beta_and_delta': 0.5,
    'delta': 2.0,
    'eta': 1.0,

    # ====== BERT regularization ======
    'bert_weight': 0.2,
    'bert_drop_sample_prob': 0.3,
    'bert_mask_prob': 0.3,
    'bert_noise_std': 0.4,

    # ====== Early stop ======
    'early_stop_patience': 8,
    'early_stop_min_delta': 1e-4,

    # ====== Grid for Early Detection ======
    # M: observation window in hours
    'M_HOURS_LIST': [1, 2, 4, 8, 12, 24],
    # lambda_norm = lambda * M_seconds (dimensionless), controls exp(-lambda_norm * t/M)
    # Use lambda_norm to compare decay across different M values; the paper discusses lambda in [0.01, 1.0].
    'LAMBDA_NORM_LIST': [0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0],

    # repeated runs for robustness
    'R': 3,

    # output
    'OUT_CSV': str(SCRIPT_DIR / 'outputs' / 'early_detection_M_lambda_grid.csv'),
    'OUT_PNG': str(SCRIPT_DIR / 'outputs' / 'early_detection_sensitivity.png'),
}

# ==========================================
# 2. Reproducibility
# ==========================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(CONFIG['seed'])

# ==========================================
# 3. Robust metapath parsing (events-based)
# ==========================================
# Support N_4_1 followed by a timestamp and C1_4_2, with optional timestamp and C2 fields.
DELIM_RE = re.compile(r'\s*(?:—{1,}|-{2,}|–{1,}|－{1,})\s*')  # Accept different dash delimiters.

N_RE = re.compile(r'^N_(\d+)_(\d+)$')
C1_RE = re.compile(r'^C1_([^_]+)_([^_]+)$')
C2_RE = re.compile(r'^C2_([^_]+)_([^_]+)$')

def _safe_int(x):
    try:
        return int(x)
    except:
        return None

def _safe_float(x):
    try:
        return float(x)
    except:
        return None

def parse_metapath_file_to_events(path: str, label: str):
    """
    return:
      groups_events: dict[nid] -> list of (ch_idx, t_event_sec)
      order_nids: list[nid] by first appearance
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[{label}] metapath file not found: {path}")

    groups = {}
    order = []
    total_lines = 0
    parsed_lines = 0

    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            total_lines += 1

            parts = DELIM_RE.split(line)
            # expected:
            # 3 parts: [N_token, t1, C1_token]
            # 5 parts: [N_token, t1, C1_token, t2, C2_token]
            if len(parts) < 3:
                continue

            n_token = parts[0].strip()
            m = N_RE.match(n_token)
            if not m:
                continue
            nid = m.group(1)
            n_sent = _safe_int(m.group(2))
            if n_sent is None:
                continue

            if nid not in groups:
                groups[nid] = []
                order.append(nid)

            t1 = _safe_float(parts[1])
            c1_token = parts[2].strip()
            m1 = C1_RE.match(c1_token)
            if not m1:
                # The record may contain C1_None_None or malformed fields.
                continue
            c1_id, c1_sent_str = m1.group(1), m1.group(2)
            c1_sent = _safe_int(c1_sent_str)
            if c1_sent is None:
                # Retain the news ID for C1_None_None, but do not create an event.
                continue

            # Check for a second-level reply.
            has_c2 = (len(parts) >= 5)
            if has_c2:
                t2 = _safe_float(parts[3])
                c2_token = parts[4].strip()
                m2 = C2_RE.match(c2_token)
                if (t2 is not None) and m2:
                    c2_id, c2_sent_str = m2.group(1), m2.group(2)
                    c2_sent = _safe_int(c2_sent_str)
                    if c2_sent is not None:
                        # N-U-V: time use t2
                        ch_idx = 9 + (n_sent * 9) + (c1_sent * 3) + c2_sent
                        t_event = t2
                        if ch_idx < CONFIG['num_channels'] and t_event is not None:
                            groups[nid].append((ch_idx, float(t_event)))
                            parsed_lines += 1
                        continue

            # N-U: time use t1
            if t1 is None:
                continue
            ch_idx = n_sent * 3 + c1_sent
            t_event = t1
            if ch_idx < CONFIG['num_channels']:
                groups[nid].append((ch_idx, float(t_event)))
                parsed_lines += 1

    if len(order) == 0:
        raise ValueError(
            f"[{label}] Parsed 0 news IDs from metapath file.\n"
            f"  - Check delimiter (dash) encoding, file encoding, or data format.\n"
            f"  - file: {path}"
        )

    print(f"[{label}] total_lines={total_lines}, parsed_events_lines≈{parsed_lines}, unique_news={len(order)}")
    return groups, order

# ==========================================
# 4. Load BERT
# ==========================================
def load_split_bert(bert_path, label="BERT"):
    if not os.path.exists(bert_path):
        raise FileNotFoundError(f"[{label}] BERT file not found: {bert_path}")
    x = np.load(bert_path).astype(np.float32)
    if x.ndim != 2:
        raise ValueError(f"[{label}] Expected BERT_shape=(N,768), got: {x.shape}")
    if x.shape[1] != CONFIG['bert_dim']:
        raise ValueError(f"[{label}] BERT dimension mismatch: expected {CONFIG['bert_dim']}, got {x.shape[1]}")
    return x

# ==========================================
# 5. Build signal matrix with (M, lambda_norm)
# ==========================================
def build_signal_matrix_from_events(events, effective_max_time, lambda_decay, M_seconds):
    """
    events: list[(ch_idx, t_event_sec)]
    effective_max_time: maximum binning time, normally M_seconds
    lambda_decay: decay rate in inverse seconds
    M_seconds: observation window (sec)
    """
    C = CONFIG['num_channels']
    T = CONFIG['num_time_bins']
    mat = np.zeros((C, T), dtype=np.float32)

    if effective_max_time <= 0:
        effective_max_time = 1.0

    for ch_idx, t in events:
        if t is None:
            continue
        if t < 0:
            continue
        if t > M_seconds:
            continue

        t_idx = int((t / effective_max_time) * T)
        if t_idx >= T:
            t_idx = T - 1
        if t_idx < 0:
            t_idx = 0

        if lambda_decay > 0:
            w = float(np.exp(-lambda_decay * t))
        else:
            w = 1.0

        if 0 <= ch_idx < C:
            mat[ch_idx, t_idx] += w

    return np.log1p(mat).astype(np.float32)

def match_data_with_padding(groups_events, order_nids, bert_vectors, label_str="DATA",
                            M_seconds=86400, lambda_norm=0.0):
    """
    Preserve the BERT sample count whenever meta-path news IDs can be parsed.
    News without events in the first M hours receives an all-zero signal.
    If meta-path and BERT row counts differ:
        - n_meta < n_bert: pad signals with zeros to n_bert.
        - n_meta > n_bert: truncate signals to n_bert, treating BERT as the master index.
    """
    n_meta = len(order_nids)
    n_bert = bert_vectors.shape[0]

    if n_meta == 0:
        raise ValueError(f"[{label_str}] metapath parsed 0 items (likely parse failure).")

    # effective binning time
    effective_max_time = float(M_seconds)

    # lambda: 1/sec (normalized by window length)
    if M_seconds > 0:
        lambda_decay = float(lambda_norm) / float(M_seconds)
    else:
        lambda_decay = 0.0

    # build signals for all metapath nids
    X_freq_list = []
    for nid in order_nids:
        events = groups_events.get(nid, [])
        X_freq_list.append(build_signal_matrix_from_events(
            events=events,
            effective_max_time=effective_max_time,
            lambda_decay=lambda_decay,
            M_seconds=M_seconds
        ))
    X_freq = np.array(X_freq_list, dtype=np.float32)
    X_bert = bert_vectors.astype(np.float32)

    if X_freq.shape[0] < n_bert:
        pad_n = n_bert - X_freq.shape[0]
        pad = np.zeros((pad_n, CONFIG['num_channels'], CONFIG['num_time_bins']), dtype=np.float32)
        X_freq = np.concatenate([X_freq, pad], axis=0)
        print(f"[WARN {label_str}] metapath<{label_str} bert: pad_zero {pad_n} samples")

    if X_freq.shape[0] > n_bert:
        X_freq = X_freq[:n_bert]
        print(f"[WARN {label_str}] metapath>{label_str} bert: truncate to {n_bert} samples")

    return X_freq, X_bert

# ==========================================
# 6. Load raw splits once (events + bert)
# ==========================================
def load_raw_splits_once():
    raw = {}

    # metapath -> events
    raw['tr_r_groups'], raw['tr_r_order'] = parse_metapath_file_to_events(CONFIG['paths']['tr_r'], "TR_REAL")
    raw['va_r_groups'], raw['va_r_order'] = parse_metapath_file_to_events(CONFIG['paths']['va_r'], "VA_REAL")
    raw['te_r_groups'], raw['te_r_order'] = parse_metapath_file_to_events(CONFIG['paths']['te_r'], "TE_REAL")

    raw['tr_f_groups'], raw['tr_f_order'] = parse_metapath_file_to_events(CONFIG['paths']['tr_f'], "TR_FAKE")
    raw['va_f_groups'], raw['va_f_order'] = parse_metapath_file_to_events(CONFIG['paths']['va_f'], "VA_FAKE")
    raw['te_f_groups'], raw['te_f_order'] = parse_metapath_file_to_events(CONFIG['paths']['te_f'], "TE_FAKE")

    # bert
    raw['tr_r_bert'] = load_split_bert(CONFIG['bert_paths']['tr_r'], "TR_REAL_BERT")
    raw['va_r_bert'] = load_split_bert(CONFIG['bert_paths']['va_r'], "VA_REAL_BERT")
    raw['te_r_bert'] = load_split_bert(CONFIG['bert_paths']['te_r'], "TE_REAL_BERT")

    raw['tr_f_bert'] = load_split_bert(CONFIG['bert_paths']['tr_f'], "TR_FAKE_BERT")
    raw['va_f_bert'] = load_split_bert(CONFIG['bert_paths']['va_f'], "VA_FAKE_BERT")
    raw['te_f_bert'] = load_split_bert(CONFIG['bert_paths']['te_f'], "TE_FAKE_BERT")

    return raw

def build_dataset_for_M_lambda(raw, M_seconds: int, lambda_norm: float):
    """
    Return (train, val, test) consistent with the main training pipeline:
      each is (X_freq, X_bert, y)
    """
    tr_r_f, tr_r_b = match_data_with_padding(
        raw['tr_r_groups'], raw['tr_r_order'], raw['tr_r_bert'],
        label_str="TR_REAL", M_seconds=M_seconds, lambda_norm=lambda_norm
    )
    tr_f_f, tr_f_b = match_data_with_padding(
        raw['tr_f_groups'], raw['tr_f_order'], raw['tr_f_bert'],
        label_str="TR_FAKE", M_seconds=M_seconds, lambda_norm=lambda_norm
    )

    va_r_f, va_r_b = match_data_with_padding(
        raw['va_r_groups'], raw['va_r_order'], raw['va_r_bert'],
        label_str="VA_REAL", M_seconds=M_seconds, lambda_norm=lambda_norm
    )
    va_f_f, va_f_b = match_data_with_padding(
        raw['va_f_groups'], raw['va_f_order'], raw['va_f_bert'],
        label_str="VA_FAKE", M_seconds=M_seconds, lambda_norm=lambda_norm
    )

    te_r_f, te_r_b = match_data_with_padding(
        raw['te_r_groups'], raw['te_r_order'], raw['te_r_bert'],
        label_str="TE_REAL", M_seconds=M_seconds, lambda_norm=lambda_norm
    )
    te_f_f, te_f_b = match_data_with_padding(
        raw['te_f_groups'], raw['te_f_order'], raw['te_f_bert'],
        label_str="TE_FAKE", M_seconds=M_seconds, lambda_norm=lambda_norm
    )

    def merge(rf, rb, ff, fb):
        xf = np.concatenate([rf, ff], axis=0).astype(np.float32)
        xb = np.concatenate([rb, fb], axis=0).astype(np.float32)
        y = np.array([0] * len(rf) + [1] * len(ff), dtype=np.float32)
        return xf, xb, y

    return (
        merge(tr_r_f, tr_r_b, tr_f_f, tr_f_b),
        merge(va_r_f, va_r_b, va_f_f, va_f_b),
        merge(te_r_f, te_r_b, te_f_f, te_f_b),
    )

# ==========================================
# 7. Dataset wrapper
# ==========================================
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
# 8. Model Architecture (same as your main)
# ==========================================
class MetaTrendAttentionEncoder(nn.Module):
    def __init__(self, num_channels, k_selected, embed_dim):
        super().__init__()
        self.path_projection = nn.Linear(k_selected * 2, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x_signal):
        fft_res = torch.fft.rfft(x_signal, dim=-1)
        selected_fft = fft_res[:, :, :CONFIG['k_selected']]
        meta_trend_coeffs = torch.cat([selected_fft.real, selected_fft.imag], dim=-1)
        path_tokens = self.path_projection(meta_trend_coeffs)
        attn_output, _ = self.attention(path_tokens, path_tokens, path_tokens)
        h_t = self.norm(attn_output + path_tokens)
        return torch.mean(h_t, dim=1)

class ManifoldFakeNewsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.struct_encoder = MetaTrendAttentionEncoder(
            CONFIG['num_channels'], CONFIG['k_selected'], CONFIG['embed_dim']
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
                drop = (torch.rand(x_bert.size(0), 1, device=x_bert.device) < p_drop).float()
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
        d_i = torch.norm(phi_h - self.center, p=2, dim=1)
        y_hat = torch.sigmoid(CONFIG['eta'] * (d_i - self.tau))
        return y_hat, d_i

# ==========================================
# 9. Loss
# ==========================================
class ManifoldLoss(nn.Module):
    def __init__(self, beta=0.5, delta=2.0):
        super().__init__()
        self.beta = beta
        self.delta = delta
        self.bce = nn.BCELoss()

    def forward(self, y_hat, y_true, d_i):
        l_ce = self.bce(y_hat, y_true)
        real_mask = (y_true == 0).float()
        fake_mask = (y_true == 1).float()
        dist_real = real_mask * torch.pow(d_i, 2)
        dist_fake = fake_mask * torch.pow(F.relu(self.delta - d_i), 2)
        l_dist = torch.mean(dist_real + dist_fake)
        return l_ce + self.beta * l_dist

# ==========================================
# 10. Evaluation
# ==========================================
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for x_f, x_b, y in loader:
            out, _ = model(x_f.to(device), x_b.to(device))
            prob = out.detach().cpu().numpy()
            pred = (out > 0.5).float().cpu().numpy()
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

# ==========================================
# 11. Single run training for one (M, lambda_norm)
# ==========================================
def train_one_run(M_hours: int, lambda_norm: float, seed: int, device, raw, run_idx: int, total_runs: int):
    set_seed(seed)
    M_seconds = int(M_hours * 3600)

    train_data, val_data, _ = build_dataset_for_M_lambda(raw, M_seconds=M_seconds, lambda_norm=lambda_norm)

    if len(train_data[2]) == 0:
        return {"best_val_auc": float('nan'), "best_val_f1_at_best_auc": float('nan'), "best_epoch": 0}

    train_loader = DataLoader(HybridDataset(*train_data), batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(HybridDataset(*val_data), batch_size=CONFIG['batch_size'])

    model = ManifoldFakeNewsNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = ManifoldLoss(beta=CONFIG['Beta_and_delta'], delta=CONFIG['delta'])

    best_val_auc = -1.0
    best_val_f1_at_best_auc = -1.0
    best_epoch = 0
    no_improve = 0
    patience = CONFIG.get('early_stop_patience', 0)
    min_delta = CONFIG.get('early_stop_min_delta', 0.0)

    print(f"\n>>> [M={M_hours}h | lambda_norm={lambda_norm}] Run {run_idx+1}/{total_runs} (seed={seed})")
    print("-" * 70)

    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0.0
        for x_f, x_b, labels in train_loader:
            labels = labels.to(device)
            optimizer.zero_grad()
            y_hat, d_i = model(x_f.to(device), x_b.to(device))
            loss = criterion(y_hat, labels, d_i)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_acc, _, _, val_f1, val_auc = evaluate(model, val_loader, device)
        avg_loss = total_loss / max(1, len(train_loader))

        print(f"Epoch [{epoch+1:02d}/{CONFIG['epochs']}] "
              f"Loss={avg_loss:.4f} | Val_AUC={val_auc:.4f} | Val_F1={val_f1:.4f} | Val_Acc={val_acc:.4f}")

        improved = (not np.isnan(val_auc)) and (val_auc > best_val_auc + min_delta)
        if improved:
            best_val_auc = float(val_auc)
            best_val_f1_at_best_auc = float(val_f1)
            best_epoch = epoch + 1
            no_improve = 0
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1:02d}")
                break

    return {
        "best_val_auc": float(best_val_auc),
        "best_val_f1_at_best_auc": float(best_val_f1_at_best_auc),
        "best_epoch": int(best_epoch),
    }

# ==========================================
# 12. Main: grid search over (M, lambda_norm)
# ==========================================
def main_early_detection_grid():
    os.makedirs(os.path.dirname(CONFIG['OUT_CSV']), exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG['OUT_PNG']), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Device] {device}")

    raw = load_raw_splits_once()

    M_list = CONFIG['M_HOURS_LIST']
    L_list = CONFIG['LAMBDA_NORM_LIST']
    R = int(CONFIG['R'])
    base_seed = int(CONFIG['seed'])

    results = []

    for M_hours in tqdm(M_list, desc="M-hours loop"):
        for lambda_norm in tqdm(L_list, desc=f"lambda loop (M={M_hours}h)", leave=False):
            f1_list, auc_list, epoch_list = [], [], []

            for r in range(R):
                seed = base_seed + r
                out = train_one_run(
                    M_hours=M_hours,
                    lambda_norm=lambda_norm,
                    seed=seed,
                    device=device,
                    raw=raw,
                    run_idx=r,
                    total_runs=R
                )
                f1_list.append(out["best_val_f1_at_best_auc"])
                auc_list.append(out["best_val_auc"])
                epoch_list.append(out["best_epoch"])

            mean_f1 = float(np.nanmean(f1_list))
            std_f1 = float(np.nanstd(f1_list))
            mean_auc = float(np.nanmean(auc_list))
            std_auc = float(np.nanstd(auc_list))
            mean_ep = float(np.nanmean(epoch_list))

            print(f"\n[Summary] M={M_hours}h, lambda_norm={lambda_norm} "
                  f"| meanF1={mean_f1:.4f}±{std_f1:.4f} | meanAUC={mean_auc:.4f}±{std_auc:.4f}")
            print("=" * 80)

            results.append({
                "M_hours": M_hours,
                "lambda_norm": lambda_norm,
                "mean_val_f1_at_best_auc": mean_f1,
                "std_val_f1_at_best_auc": std_f1,
                "mean_best_val_auc": mean_auc,
                "std_best_val_auc": std_auc,
                "mean_best_epoch": mean_ep,
            })

    df = pd.DataFrame(results)
    df.to_csv(CONFIG['OUT_CSV'], index=False, encoding='utf-8-sig')
    print(f"\nSaved CSV: {CONFIG['OUT_CSV']}")

    plot_heatmap(df)

def plot_heatmap(df: pd.DataFrame):
    """
    TKDE-style heatmap:
    - perceptually uniform colormap (viridis/cividis)
    - annotate each cell with mean F1
    - highlight global best cell with a black rectangle + star
    - overlay per-M best lambda_norm trajectory (dots + line)
    - tighter colorbar range for better contrast
    """
    # ===== pivot to matrix [lambda_norm x M_hours] =====
    pivot = df.pivot(index="lambda_norm", columns="M_hours", values="mean_val_f1_at_best_auc")
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    lambdas = pivot.index.to_list()
    Ms = pivot.columns.to_list()
    Z = pivot.values.astype(float)

    # ===== robust color range (tighten) =====
    z_valid = Z[np.isfinite(Z)]
    if len(z_valid) == 0:
        raise ValueError("No valid values found in pivot table for plotting.")
    zmin, zmax = np.min(z_valid), np.max(z_valid)
    pad = (zmax - zmin) * 0.08 if (zmax > zmin) else 1e-4
    vmin = zmin - pad
    vmax = zmax + pad

    # ===== style =====
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    })

    fig, ax = plt.subplots(figsize=(10.0, 6.2))

    # TKDE-friendly colormap (choose one)
    cmap = plt.cm.viridis  # or plt.cm.cividis

    # ===== main heatmap =====
    im = ax.imshow(
        Z,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # axis labels
    ax.set_title("Early Detection Sensitivity: Interaction of $M \\times \\lambda$ (Validation F1)")
    ax.set_xlabel("Observation Window $M$ (hours)")
    ax.set_ylabel(r"Decay Strength")

    ax.set_xticks(np.arange(len(Ms)))
    ax.set_xticklabels([str(m) for m in Ms])

    ax.set_yticks(np.arange(len(lambdas)))
    ax.set_yticklabels([f"{x:g}" for x in lambdas])

    # ===== light gridlines for readability =====
    ax.set_xticks(np.arange(-.5, len(Ms), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(lambdas), 1), minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.8, alpha=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)

    # ===== annotate each cell (mean F1) =====
    # choose text color based on background
    threshold = (vmin + vmax) / 2.0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            val = Z[i, j]
            if not np.isfinite(val):
                continue
            txt_color = "black" if val > threshold else "white"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", color=txt_color, fontsize=9)

    # ===== colorbar =====
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Validation F1")
    cbar.ax.tick_params(labelsize=10)

    # ===== highlight global best =====
    best_idx = np.nanargmax(Z)
    by, bx = np.unravel_index(best_idx, Z.shape)

    # black rectangle around best cell
    rect = plt.Rectangle((bx - 0.5, by - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2.2)
    ax.add_patch(rect)

    # star marker + label
    ax.scatter([bx], [by], marker="*", s=220, color="white", edgecolor="black", linewidth=1.2, zorder=5)
    ax.text(bx + 0.15, by + 0.25, "Best", color="black", fontsize=11, weight="bold")

    # ===== overlay per-M best lambda trajectory =====
    # for each column (M), find row index of best lambda
    best_rows = np.nanargmax(Z, axis=0)  # shape: [len(Ms)]
    xs = np.arange(len(Ms))
    ys = best_rows

    ax.plot(xs, ys, linewidth=1.8, linestyle="-", color="white", alpha=0.9, zorder=4)
    ax.scatter(xs, ys, s=55, color="white", edgecolor="black", linewidth=0.9, zorder=5, label="Best $\\lambda$ for each $M$")

    ax.legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(CONFIG["OUT_PNG"], dpi=300)

    plt.show()

    print(f"Saved heatmap: {CONFIG['OUT_PNG']}")


def configure_paths(argv=None):
    """Resolve CLI input/output locations without changing experiment settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DATA_DIR,
                        help='Directory containing metapath_data/ and bert_feature/ files.')
    parser.add_argument('--output-dir', type=Path, default=SCRIPT_DIR / 'outputs',
                        help='Destination for the experiment CSV and PNG.')
    args = parser.parse_args(argv)
    data_dir = args.data_dir.expanduser().resolve()
    for key, value in CONFIG['paths'].items():
        CONFIG['paths'][key] = str(data_dir / 'metapath_data' / Path(value).name)
    for key, value in CONFIG['bert_paths'].items():
        CONFIG['bert_paths'][key] = str(data_dir / 'bert_feature' / Path(value).name)
    missing = [value for group in ('paths', 'bert_paths')
               for value in CONFIG[group].values() if not Path(value).is_file()]
    if missing:
        parser.error('Missing input files:\n  ' + '\n  '.join(missing))
    out_dir = args.output_dir.expanduser().resolve()
    CONFIG['OUT_CSV'] = str(out_dir / 'early_detection_M_lambda_grid.csv')
    CONFIG['OUT_PNG'] = str(out_dir / 'early_detection_sensitivity.png')


if __name__ == "__main__":
    configure_paths()
    main_early_detection_grid()
