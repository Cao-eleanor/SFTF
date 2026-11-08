"""
Sensitivity analysis of the loss weight beta and deviation margin delta: 
sweep the beta-by-delta grid and export the validation-F1 and separability-gap metrics (CSV) plus their line plots (PNG).
"""
import os
import re
import copy
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# =========================================================
# 1) CONFIG
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # SFTF root (contains model/data/)
OUT_DIR = SCRIPT_DIR / 'outputs'

CONFIG = {
    "paths": {
        "tr_r": str(PROJECT_ROOT / "model" / "data" / "metapath_data" / "Deepseek_real_train_metapaths.txt"),
        "va_r": str(PROJECT_ROOT / "model" / "data" / "metapath_data" / "Deepseek_real_val_metapaths.txt"),
        "te_r": str(PROJECT_ROOT / "model" / "data" / "metapath_data" / "Deepseek_real_test_metapaths.txt"),
        "tr_f": str(PROJECT_ROOT / "model" / "data" / "metapath_data" / "Deepseek_fake_train_metapaths.txt"),
        "va_f": str(PROJECT_ROOT / "model" / "data" / "metapath_data" / "Deepseek_fake_val_metapaths.txt"),
        "te_f": str(PROJECT_ROOT / "model" / "data" / "metapath_data" / "Deepseek_fake_test_metapaths.txt"),
    },
    "bert_paths": {
        "tr_r": str(PROJECT_ROOT / "model" / "data" / "bert_feature" / "real_train_bert.npy"),
        "va_r": str(PROJECT_ROOT / "model" / "data" / "bert_feature" / "real_val_bert.npy"),
        "te_r": str(PROJECT_ROOT / "model" / "data" / "bert_feature" / "real_test_bert.npy"),
        "tr_f": str(PROJECT_ROOT / "model" / "data" / "bert_feature" / "fake_train_bert.npy"),
        "va_f": str(PROJECT_ROOT / "model" / "data" / "bert_feature" / "fake_val_bert.npy"),
        "te_f": str(PROJECT_ROOT / "model" / "data" / "bert_feature" / "fake_test_bert.npy"),
    },

    # Structural signal
    "num_channels": 36,
    "num_time_bins": 100,

    # Model
    "k_selected": 12,
    "embed_dim": 64,
    "metric_dim": 128,
    "bert_dim": 768,

    # Training
    "batch_size": 32,
    "learning_rate": 3e-4,
    "epochs": 20,
    "seed": 8,

    # Loss baseline (overridden by grid)
    "Beta_and_delta": 0.5,
    "delta": 2.0,
    "eta": 1.0,

    # BERT regularization
    "bert_weight": 0.2,
    "bert_drop_sample_prob": 0.3,
    "bert_mask_prob": 0.3,
    "bert_noise_std": 0.4,

    # Early stop (based on Val F1)
    "early_stop_patience": 8,
    "early_stop_min_delta": 1e-4,
}

EXP2 = {
    # Fixed early-detection setting (use your best M, lambda_norm)
    "FIXED_M_HOURS": 2,
    "FIXED_LAMBDA_NORM": 1.0,

    # Delta grid
    "DELTA_LIST": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],

    # repeats per (Beta_and_delta, delta)
    "R": 3,

    # outputs
    "OUT_CSV": "beta_delta_F1.csv",

    # line plots (PNG only)
    "OUT_F1_PNG": "manifold_performance.png",
    "OUT_GAP_PNG": "manifold_separability.png",
}


# =========================================================
# 2) Reproducibility
# =========================================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# 3) Robust metapath parsing (events-based)
# =========================================================
DELIM_RE = re.compile(r"\s*(?:—{1,}|-{2,}|–{1,}|－{1,})\s*")
N_RE = re.compile(r"^N_(\d+)_(\d+)$")
C1_RE = re.compile(r"^C1_([^_]+)_([^_]+)$")
C2_RE = re.compile(r"^C2_([^_]+)_([^_]+)$")


def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def parse_metapath_file_to_events(path: str, label: str):
    """
    Return:
      groups_events: dict[nid] -> list of (ch_idx, t_event_sec)
      order_nids: list[nid] by first appearance
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[{label}] metapath file not found: {path}")

    groups = {}
    order = []
    total_lines = 0
    parsed_events = 0

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            total_lines += 1

            parts = DELIM_RE.split(line)
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
                continue
            c1_sent = _safe_int(m1.group(2))
            if c1_sent is None:
                continue

            # try C2
            has_c2 = len(parts) >= 5
            if has_c2:
                t2 = _safe_float(parts[3])
                c2_token = parts[4].strip()
                m2 = C2_RE.match(c2_token)
                if (t2 is not None) and m2:
                    c2_sent = _safe_int(m2.group(2))
                    if c2_sent is not None:
                        ch_idx = 9 + (n_sent * 9) + (c1_sent * 3) + c2_sent
                        if ch_idx < CONFIG["num_channels"]:
                            groups[nid].append((ch_idx, float(t2)))
                            parsed_events += 1
                        continue

            # fallback to C1
            if t1 is None:
                continue
            ch_idx = n_sent * 3 + c1_sent
            if ch_idx < CONFIG["num_channels"]:
                groups[nid].append((ch_idx, float(t1)))
                parsed_events += 1

    if len(order) == 0:
        raise ValueError(f"[{label}] Parsed 0 news IDs from metapath file. File: {path}")

    print(f"[{label}] total_lines={total_lines}, parsed_events≈{parsed_events}, unique_news={len(order)}")
    return groups, order


# =========================================================
# 4) Load BERT
# =========================================================
def load_split_bert(bert_path, label="BERT"):
    if not os.path.exists(bert_path):
        raise FileNotFoundError(f"[{label}] missing BERT file: {bert_path}")
    x = np.load(bert_path).astype(np.float32)
    if x.ndim != 2:
        raise ValueError(f"[{label}] expected (N,768), got {x.shape}")
    if x.shape[1] != CONFIG["bert_dim"]:
        raise ValueError(f"[{label}] bert_dim mismatch: expected {CONFIG['bert_dim']}, got {x.shape[1]}")
    return x


# =========================================================
# 5) Build signal matrix with (M_seconds, lambda_norm)
# =========================================================
def build_signal_matrix_from_events(events, effective_max_time, lambda_decay, M_seconds):
    C = CONFIG["num_channels"]
    T = CONFIG["num_time_bins"]
    mat = np.zeros((C, T), dtype=np.float32)

    if effective_max_time <= 0:
        effective_max_time = 1.0

    for ch_idx, t in events:
        if t is None or t < 0:
            continue
        if t > M_seconds:
            continue

        t_idx = int((t / effective_max_time) * T)
        t_idx = min(max(t_idx, 0), T - 1)

        w = float(np.exp(-lambda_decay * t)) if (lambda_decay > 0) else 1.0

        if 0 <= ch_idx < C:
            mat[ch_idx, t_idx] += w

    return np.log1p(mat).astype(np.float32)


def match_data_with_padding(groups_events, order_nids, bert_vectors, label_str="DATA", M_seconds=86400, lambda_norm=0.0):
    n_meta = len(order_nids)
    n_bert = bert_vectors.shape[0]
    if n_meta == 0:
        raise ValueError(f"[{label_str}] parsed 0 metapath items (parse failure).")

    effective_max_time = float(M_seconds)
    lambda_decay = float(lambda_norm) / float(M_seconds) if M_seconds > 0 else 0.0

    X_freq = []
    for nid in order_nids:
        events = groups_events.get(nid, [])
        X_freq.append(
            build_signal_matrix_from_events(
                events=events,
                effective_max_time=effective_max_time,
                lambda_decay=lambda_decay,
                M_seconds=M_seconds,
            )
        )
    X_freq = np.array(X_freq, dtype=np.float32)
    X_bert = bert_vectors.astype(np.float32)

    if X_freq.shape[0] < n_bert:
        pad_n = n_bert - X_freq.shape[0]
        pad = np.zeros((pad_n, CONFIG["num_channels"], CONFIG["num_time_bins"]), dtype=np.float32)
        X_freq = np.concatenate([X_freq, pad], axis=0)
        print(f"[WARN {label_str}] metapath<{label_str} bert: pad_zero {pad_n} samples")

    if X_freq.shape[0] > n_bert:
        X_freq = X_freq[:n_bert]
        print(f"[WARN {label_str}] metapath>{label_str} bert: truncate to {n_bert} samples")

    return X_freq, X_bert


# =========================================================
# 6) Load raw splits once
# =========================================================
def load_raw_splits_once():
    raw = {}

    raw["tr_r_groups"], raw["tr_r_order"] = parse_metapath_file_to_events(CONFIG["paths"]["tr_r"], "TR_REAL")
    raw["va_r_groups"], raw["va_r_order"] = parse_metapath_file_to_events(CONFIG["paths"]["va_r"], "VA_REAL")
    raw["te_r_groups"], raw["te_r_order"] = parse_metapath_file_to_events(CONFIG["paths"]["te_r"], "TE_REAL")

    raw["tr_f_groups"], raw["tr_f_order"] = parse_metapath_file_to_events(CONFIG["paths"]["tr_f"], "TR_FAKE")
    raw["va_f_groups"], raw["va_f_order"] = parse_metapath_file_to_events(CONFIG["paths"]["va_f"], "VA_FAKE")
    raw["te_f_groups"], raw["te_f_order"] = parse_metapath_file_to_events(CONFIG["paths"]["te_f"], "TE_FAKE")

    raw["tr_r_bert"] = load_split_bert(CONFIG["bert_paths"]["tr_r"], "TR_REAL_BERT")
    raw["va_r_bert"] = load_split_bert(CONFIG["bert_paths"]["va_r"], "VA_REAL_BERT")
    raw["te_r_bert"] = load_split_bert(CONFIG["bert_paths"]["te_r"], "TE_REAL_BERT")

    raw["tr_f_bert"] = load_split_bert(CONFIG["bert_paths"]["tr_f"], "TR_FAKE_BERT")
    raw["va_f_bert"] = load_split_bert(CONFIG["bert_paths"]["va_f"], "VA_FAKE_BERT")
    raw["te_f_bert"] = load_split_bert(CONFIG["bert_paths"]["te_f"], "TE_FAKE_BERT")

    return raw


def build_dataset_for_fixed_M_lambda(raw, M_seconds: int, lambda_norm: float):
    tr_r_f, tr_r_b = match_data_with_padding(
        raw["tr_r_groups"], raw["tr_r_order"], raw["tr_r_bert"],
        label_str="TR_REAL", M_seconds=M_seconds, lambda_norm=lambda_norm
    )
    tr_f_f, tr_f_b = match_data_with_padding(
        raw["tr_f_groups"], raw["tr_f_order"], raw["tr_f_bert"],
        label_str="TR_FAKE", M_seconds=M_seconds, lambda_norm=lambda_norm
    )

    va_r_f, va_r_b = match_data_with_padding(
        raw["va_r_groups"], raw["va_r_order"], raw["va_r_bert"],
        label_str="VA_REAL", M_seconds=M_seconds, lambda_norm=lambda_norm
    )
    va_f_f, va_f_b = match_data_with_padding(
        raw["va_f_groups"], raw["va_f_order"], raw["va_f_bert"],
        label_str="VA_FAKE", M_seconds=M_seconds, lambda_norm=lambda_norm
    )

    te_r_f, te_r_b = match_data_with_padding(
        raw["te_r_groups"], raw["te_r_order"], raw["te_r_bert"],
        label_str="TE_REAL", M_seconds=M_seconds, lambda_norm=lambda_norm
    )
    te_f_f, te_f_b = match_data_with_padding(
        raw["te_f_groups"], raw["te_f_order"], raw["te_f_bert"],
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


# =========================================================
# 7) Dataset
# =========================================================
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


# =========================================================
# 8) Model
# =========================================================
class MetaTrendAttentionEncoder(nn.Module):
    def __init__(self, num_channels, k_selected, embed_dim):
        super().__init__()
        self.path_projection = nn.Linear(k_selected * 2, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x_signal):
        fft_res = torch.fft.rfft(x_signal, dim=-1)
        selected_fft = fft_res[:, :, :CONFIG["k_selected"]]
        meta_trend_coeffs = torch.cat([selected_fft.real, selected_fft.imag], dim=-1)
        path_tokens = self.path_projection(meta_trend_coeffs)
        attn_output, _ = self.attention(path_tokens, path_tokens, path_tokens)
        h_t = self.norm(attn_output + path_tokens)
        return torch.mean(h_t, dim=1)


class ManifoldFakeNewsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.struct_encoder = MetaTrendAttentionEncoder(
            CONFIG["num_channels"], CONFIG["k_selected"], CONFIG["embed_dim"]
        )
        self.bert_fc = nn.Sequential(
            nn.Linear(CONFIG["bert_dim"], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, CONFIG["embed_dim"])
        )
        self.phi = nn.Sequential(
            nn.Linear(CONFIG["embed_dim"] * 2, 256),
            nn.ReLU(),
            nn.Linear(256, CONFIG["metric_dim"])
        )
        self.center = nn.Parameter(torch.randn(1, CONFIG["metric_dim"]))
        self.tau = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x_freq, x_bert):
        h_t = self.struct_encoder(x_freq)

        if self.training:
            p_drop = CONFIG.get("bert_drop_sample_prob", 0.0)
            if p_drop > 0:
                drop = (torch.rand(x_bert.size(0), 1, device=x_bert.device) < p_drop).float()
                x_bert = x_bert * (1.0 - drop)

            p_mask = CONFIG.get("bert_mask_prob", 0.0)
            if p_mask > 0:
                keep = (torch.rand_like(x_bert) > p_mask).float()
                x_bert = x_bert * keep

            x_bert = x_bert + torch.randn_like(x_bert) * CONFIG.get("bert_noise_std", 0.0)

        h_s = self.bert_fc(x_bert)
        h_s = h_s * CONFIG.get("bert_weight", 1.0)

        h_i = torch.cat([h_t, h_s], dim=1)
        phi_h = self.phi(h_i)

        d_i = torch.norm(phi_h - self.center, p=2, dim=1)
        y_hat = torch.sigmoid(CONFIG["eta"] * (d_i - self.tau))
        return y_hat, d_i


# =========================================================
# 9) Loss (Beta_and_delta, delta are variables in grid)
# =========================================================
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


# =========================================================
# 10) Evaluation (metrics + distances)
# =========================================================
def evaluate_with_distances(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    d_all = []
    with torch.no_grad():
        for x_f, x_b, y in loader:
            out, d_i = model(x_f.to(device), x_b.to(device))
            pred = (out > 0.5).float().cpu().numpy().tolist()
            y_pred.extend(pred)
            y_true.extend(y.cpu().numpy().tolist())
            d_all.extend(d_i.detach().cpu().numpy().tolist())

    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    d_all = np.array(d_all, dtype=float)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return acc, prec, rec, f1, y_true, d_all


def compute_manifold_stats(y_true: np.ndarray, d_all: np.ndarray, delta: float):
    real_mask = (y_true == 0)
    fake_mask = (y_true == 1)

    mean_d_real = float(np.nanmean(d_all[real_mask])) if np.any(real_mask) else float("nan")
    mean_d_fake = float(np.nanmean(d_all[fake_mask])) if np.any(fake_mask) else float("nan")
    gap = float(mean_d_fake - mean_d_real) if (np.isfinite(mean_d_real) and np.isfinite(mean_d_fake)) else float("nan")
    violation = float(np.nanmean((d_all[fake_mask] < float(delta)).astype(float))) if np.any(fake_mask) else float("nan")

    return mean_d_real, mean_d_fake, gap, violation


# =========================================================
# 11) Single run training (early-stop by Val F1)
# =========================================================
def train_one_run_beta_delta(beta: float, delta: float, seed: int, device, raw, M_hours: int, lambda_norm: float):
    set_seed(seed)

    M_seconds = int(M_hours * 3600)
    train_data, val_data, _ = build_dataset_for_fixed_M_lambda(raw, M_seconds=M_seconds, lambda_norm=lambda_norm)

    train_loader = DataLoader(HybridDataset(*train_data), batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(HybridDataset(*val_data), batch_size=CONFIG["batch_size"])

    model = ManifoldFakeNewsNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"])
    criterion = ManifoldLoss(beta=beta, delta=delta)

    best_val_f1 = -1.0
    best_val_acc_at_best_f1 = -1.0
    best_epoch = 0
    best_state = None

    no_improve = 0
    patience = CONFIG.get("early_stop_patience", 0)
    min_delta = CONFIG.get("early_stop_min_delta", 0.0)

    for epoch in range(CONFIG["epochs"]):
        model.train()
        for x_f, x_b, labels in train_loader:
            optimizer.zero_grad()
            y_hat, d_i = model(x_f.to(device), x_b.to(device))
            loss = criterion(y_hat, labels.to(device), d_i)
            loss.backward()
            optimizer.step()

        val_acc, _, _, val_f1, _, _ = evaluate_with_distances(model, val_loader, device)

        improved = (val_f1 > best_val_f1 + min_delta)
        if improved:
            best_val_f1 = float(val_f1)
            best_val_acc_at_best_f1 = float(val_acc)
            best_epoch = epoch + 1
            no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                break

    # Re-eval best checkpoint for manifold stats
    if best_state is not None:
        model.load_state_dict(best_state)

    val_acc, val_prec, val_rec, val_f1, y_true, d_all = evaluate_with_distances(model, val_loader, device)
    mean_d_real, mean_d_fake, gap, violation = compute_manifold_stats(y_true, d_all, delta=delta)

    return {
        "best_val_f1_earlystop": float(best_val_f1),
        "best_val_acc_at_best_f1": float(best_val_acc_at_best_f1),
        "best_epoch": int(best_epoch),

        "val_acc_best_model": float(val_acc),
        "val_prec_best_model": float(val_prec),
        "val_rec_best_model": float(val_rec),
        "val_f1_best_model": float(val_f1),

        "mean_d_real": float(mean_d_real),
        "mean_d_fake": float(mean_d_fake),
        "gap_dfake_dreal": float(gap),
        "violation_fake_lt_delta": float(violation),
    }


# =========================================================
# 12) Plot: line charts with equal-spaced Beta_and_delta positions
# =========================================================
def set_paper_style():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
    })


def plot_lines_equalspace(
    df: pd.DataFrame,
    y_col: str,
    y_std_col: str,
    title: str,
    y_label: str,
    out_png: str,
    max_xticks: int = 12,
    show: bool = True,
):
    set_paper_style()

    need = ["beta", "delta", y_col]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns for plotting: {miss}. Got: {list(df.columns)}")

    df = df.copy()
    df["beta"] = pd.to_numeric(df["beta"], errors="coerce")
    df["delta"] = pd.to_numeric(df["delta"], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    if y_std_col and y_std_col in df.columns:
        df[y_std_col] = pd.to_numeric(df[y_std_col], errors="coerce")

    df = df.dropna(subset=["beta", "delta", y_col]).copy()

    betas = np.array(sorted(df["beta"].unique()), dtype=float)
    deltas = np.array(sorted(df["delta"].unique()), dtype=float)

    # equal-spaced x positions (index), but tick labels show real beta values
    x_pos = np.arange(len(betas), dtype=int)
    beta_to_x = {b: i for i, b in enumerate(betas)}

    fig, ax = plt.subplots(figsize=(9.2, 6.2))

    for d in deltas:
        sub = df[df["delta"] == d].sort_values("beta")
        x = np.array([beta_to_x[b] for b in sub["beta"].to_numpy(dtype=float)], dtype=int)
        y = sub[y_col].to_numpy(dtype=float)

        ax.plot(x, y, marker="o", linewidth=2.2, markersize=5.5, label=fr"$\Delta={d:g}$")

        if y_std_col and (y_std_col in sub.columns) and sub[y_std_col].notna().any():
            s = sub[y_std_col].to_numpy(dtype=float)
            ax.errorbar(x, y, yerr=s, fmt="none", capsize=2, linewidth=1.0, alpha=0.55)

    ax.set_title(title)
    ax.set_xlabel(r"Loss Weight $\beta$ (equal-spaced)")
    ax.set_ylabel(y_label)

    # avoid crowded x ticks
    if len(betas) <= max_xticks:
        tick_idx = x_pos
    else:
        tick_idx = np.linspace(0, len(betas) - 1, max_xticks).round().astype(int)
        tick_idx = np.unique(tick_idx)

    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{betas[i]:g}" for i in tick_idx])

    ax.grid(True, linestyle=":", linewidth=0.9, alpha=0.35)
    ax.legend(loc="best", frameon=True, fancybox=True, framealpha=0.9, ncol=2)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"[Saved] {out_png}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# =========================================================
# 13) Grid search + Save CSV + Draw 2 line plots
# =========================================================
def make_beta_list(beta_min: float, beta_max: float, beta_step: float):
    if beta_step <= 0:
        raise ValueError("beta_step must be > 0")

    # inclusive range with float safety
    eps = beta_step * 1e-6
    betas = np.arange(beta_min, beta_max + eps, beta_step, dtype=float)

    # clean tiny negative zeros, and keep readable floats
    betas = np.where(np.isclose(betas, 0.0, atol=1e-12), 0.0, betas)
    betas = np.round(betas, 10)
    return betas.tolist()


def run_beta_delta_grid(
    out_dir: str,
    retrain: bool,
    plot_only: bool,
    fixed_M: int,
    fixed_lambda_norm: float,
    beta_min: float,
    beta_max: float,
    beta_step: float,
    R: int,
):
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, EXP2["OUT_CSV"])

    # plot-only
    if plot_only:
        if not os.path.exists(out_csv):
            raise FileNotFoundError(f"--plot_only requires existing CSV: {out_csv}")
        df = pd.read_csv(out_csv)
        draw_two_line_plots(df, out_dir=out_dir)
        return

    # if CSV exists and not retrain -> just plot
    if (not retrain) and os.path.exists(out_csv):
        print(f"[Info] Found existing CSV -> skip training: {out_csv}")
        df = pd.read_csv(out_csv)
        draw_two_line_plots(df, out_dir=out_dir)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    set_seed(CONFIG["seed"])
    raw = load_raw_splits_once()

    betas = make_beta_list(beta_min, beta_max, beta_step)
    deltas = EXP2["DELTA_LIST"]
    base_seed = int(CONFIG["seed"])

    results = []
    pbar = tqdm(total=len(betas) * len(deltas), desc="Grid β×Δ (Early-stop by Val F1)")

    for beta in betas:
        for delta in deltas:
            f1_list, acc_list = [], []
            mean_d_real_list, mean_d_fake_list = [], []
            gap_list, vio_list = [], []
            ep_list = []

            for r in range(int(R)):
                seed = base_seed + r
                out = train_one_run_beta_delta(
                    beta=float(beta),
                    delta=float(delta),
                    seed=int(seed),
                    device=device,
                    raw=raw,
                    M_hours=int(fixed_M),
                    lambda_norm=float(fixed_lambda_norm),
                )

                f1_list.append(out["best_val_f1_earlystop"])
                acc_list.append(out["best_val_acc_at_best_f1"])
                ep_list.append(out["best_epoch"])

                mean_d_real_list.append(out["mean_d_real"])
                mean_d_fake_list.append(out["mean_d_fake"])
                gap_list.append(out["gap_dfake_dreal"])
                vio_list.append(out["violation_fake_lt_delta"])

            results.append({
                "beta": float(beta),
                "delta": float(delta),

                # task
                "mean_val_f1": float(np.nanmean(f1_list)),
                "std_val_f1": float(np.nanstd(f1_list)),
                "mean_val_acc_at_best_f1": float(np.nanmean(acc_list)),
                "std_val_acc_at_best_f1": float(np.nanstd(acc_list)),

                # manifold structure
                "mean_d_real": float(np.nanmean(mean_d_real_list)),
                "std_d_real": float(np.nanstd(mean_d_real_list)),
                "mean_d_fake": float(np.nanmean(mean_d_fake_list)),
                "std_d_fake": float(np.nanstd(mean_d_fake_list)),
                "mean_gap_dfake_dreal": float(np.nanmean(gap_list)),
                "std_gap_dfake_dreal": float(np.nanstd(gap_list)),
                "mean_violation_fake_lt_delta": float(np.nanmean(vio_list)),
                "std_violation_fake_lt_delta": float(np.nanstd(vio_list)),

                # training dynamics
                "mean_best_epoch": float(np.nanmean(ep_list)),
                "std_best_epoch": float(np.nanstd(ep_list)),
            })

            pbar.update(1)

    pbar.close()

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[Saved CSV] {out_csv}")

    draw_two_line_plots(df, out_dir=out_dir)


def draw_two_line_plots(df: pd.DataFrame, out_dir: str):
    out_f1_png = os.path.join(out_dir, EXP2["OUT_F1_PNG"])
    out_gap_png = os.path.join(out_dir, EXP2["OUT_GAP_PNG"])

    plot_lines_equalspace(
        df=df,
        y_col="mean_val_f1",
        y_std_col="std_val_f1",
        title=r"Interaction Study ($\beta \times \Delta$): Mean Validation F1",
        y_label="Mean Validation F1",
        out_png=out_f1_png,
        max_xticks=14,
        show=True,
    )

    plot_lines_equalspace(
        df=df,
        y_col="mean_gap_dfake_dreal",
        y_std_col="std_gap_dfake_dreal",
        title=r"Manifold Separability ($\beta \times \Delta$): $\mathbb{E}[d_{fake}] - \mathbb{E}[d_{real}]$",
        y_label=r"Mean Gap: $\mathbb{E}[d_{fake}] - \mathbb{E}[d_{real}]$",
        out_png=out_gap_png,
        max_xticks=14,
        show=True,
    )


# =========================================================
# 14) Main
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default=str(OUT_DIR), help="output directory for CSV and figures")

    parser.add_argument("--retrain", action="store_true", help="force retrain even if CSV exists")
    parser.add_argument("--plot_only", action="store_true", help="only plot from existing CSV (no training)")

    parser.add_argument("--fixed_M", type=int, default=EXP2["FIXED_M_HOURS"], help="fixed observation window M (hours)")
    parser.add_argument("--fixed_lambda_norm", type=float, default=EXP2["FIXED_LAMBDA_NORM"], help="fixed lambda_norm")

    # continuous Beta_and_delta range
    parser.add_argument("--beta_min", type=float, default=0.0)
    parser.add_argument("--beta_max", type=float, default=5.0)
    parser.add_argument("--beta_step", type=float, default=0.5)

    parser.add_argument("--R", type=int, default=EXP2["R"], help="repeats per (Beta_and_delta, delta)")

    args = parser.parse_args()

    # allow override
    EXP2["FIXED_M_HOURS"] = int(args.fixed_M)
    EXP2["FIXED_LAMBDA_NORM"] = float(args.fixed_lambda_norm)

    run_beta_delta_grid(
        out_dir=str(args.out_dir),
        retrain=bool(args.retrain),
        plot_only=bool(args.plot_only),
        fixed_M=int(EXP2["FIXED_M_HOURS"]),
        fixed_lambda_norm=float(EXP2["FIXED_LAMBDA_NORM"]),
        beta_min=float(args.beta_min),
        beta_max=float(args.beta_max),
        beta_step=float(args.beta_step),
        R=int(args.R),
    )
