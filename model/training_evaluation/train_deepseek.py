"""
Train and evaluate the manifold fake-news detection model on DeepSeek sentiment metapaths and BERT features.
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

# Limit CPU threads under Windows + Anaconda + PyTorch to help reproducibility
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch

# Call this only once; do not use torch.set_num_interop_threads(1)
torch.set_num_threads(1)

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)


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

SFTF_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = SFTF_ROOT / "model" / "data"


# ==========================================
# 1. Configuration Parameters
# ==========================================

CONFIG = {
    'paths': {
        'tr_r': DATA_DIR / 'metapath_data' / 'Deepseek_real_train_metapaths.txt',
        'va_r': DATA_DIR / 'metapath_data' / 'Deepseek_real_val_metapaths.txt',
        'te_r': DATA_DIR / 'metapath_data' / 'Deepseek_real_test_metapaths.txt',

        'tr_f': DATA_DIR / 'metapath_data' / 'Deepseek_fake_train_metapaths.txt',
        'va_f': DATA_DIR / 'metapath_data' / 'Deepseek_fake_val_metapaths.txt',
        'te_f': DATA_DIR / 'metapath_data' / 'Deepseek_fake_test_metapaths.txt'
    },

    'bert_paths': {
        'tr_r': DATA_DIR / 'bert_feature' / 'real_train_bert.npy',
        'va_r': DATA_DIR / 'bert_feature' / 'real_val_bert.npy',
        'te_r': DATA_DIR / 'bert_feature' / 'real_test_bert.npy',

        'tr_f': DATA_DIR / 'bert_feature' / 'fake_train_bert.npy',
        'va_f': DATA_DIR / 'bert_feature' / 'fake_val_bert.npy',
        'te_f': DATA_DIR / 'bert_feature' / 'fake_test_bert.npy'
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

    'bert_weight': 0.2,
    'bert_drop_sample_prob': 0.2,
    'bert_mask_prob': 0.2,
    'bert_noise_std': 0.3,

    'target_score': 0.94,

    'threshold_min': 0.20,
    'threshold_max': 0.70,
    'threshold_step': 0.005,

    'save_path': SFTF_ROOT / 'model' / 'checkpoints' / 'SFTF_weibo25_best.pth',

    'threshold_save_path': SFTF_ROOT / 'model' / 'checkpoints' / 'SFTF_weibo25_best.json'
}

set_seed(CONFIG['seed'])


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


def get_nid_set(path):
    groups, _ = parse_metapath_file(path)
    return set(groups.keys())


def check_overlap(a, b, name_a, name_b):
    inter = a & b

    if len(inter) > 0:
        print(f"[Warning] {name_a} and {name_b} have overlapping samples; examples:", list(inter)[:5])


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


def match_data_direct(groups, order_nids, bert_vectors, label_str="DATA"):
    n_meta = len(order_nids)

    n_bert = bert_vectors.shape[0]

    if n_meta == 0:
        return (
            np.zeros(
                (0, CONFIG['num_channels'], CONFIG['num_time_bins']),
                dtype=np.float32
            ),
            np.zeros(
                (0, CONFIG['bert_dim']),
                dtype=np.float32
            )
        )

    if n_meta != n_bert:
        raise ValueError(
            f"[{label_str}] sample count mismatch: metapath samples={n_meta}, bert rows={n_bert}."
        )

    X_freq = [build_signal_matrix(groups[nid]) for nid in order_nids]
    X_freq = np.array(X_freq, dtype=np.float32)
    X_bert = bert_vectors.astype(np.float32)

    return X_freq, X_bert


def load_pre_split_dataset():
    tr_r = get_nid_set(CONFIG['paths']['tr_r'])
    va_r = get_nid_set(CONFIG['paths']['va_r'])
    te_r = get_nid_set(CONFIG['paths']['te_r'])

    check_overlap(tr_r, va_r, "train_real", "val_real")
    check_overlap(tr_r, te_r, "train_real", "test_real")
    check_overlap(va_r, te_r, "val_real", "test_real")

    tr_f = get_nid_set(CONFIG['paths']['tr_f'])
    va_f = get_nid_set(CONFIG['paths']['va_f'])
    te_f = get_nid_set(CONFIG['paths']['te_f'])

    check_overlap(tr_f, va_f, "train_fake", "val_fake")
    check_overlap(tr_f, te_f, "train_fake", "test_fake")
    check_overlap(va_f, te_f, "val_fake", "test_fake")

    tr_r_groups, tr_r_order = parse_metapath_file(CONFIG['paths']['tr_r'])
    tr_f_groups, tr_f_order = parse_metapath_file(CONFIG['paths']['tr_f'])
    va_r_groups, va_r_order = parse_metapath_file(CONFIG['paths']['va_r'])
    va_f_groups, va_f_order = parse_metapath_file(CONFIG['paths']['va_f'])
    te_r_groups, te_r_order = parse_metapath_file(CONFIG['paths']['te_r'])
    te_f_groups, te_f_order = parse_metapath_file(CONFIG['paths']['te_f'])

    tr_r_bert = load_split_bert(CONFIG['bert_paths']['tr_r'], "REAL TRAIN Bert_News_pre_training")
    tr_f_bert = load_split_bert(CONFIG['bert_paths']['tr_f'], "FAKE TRAIN Bert_News_pre_training")
    va_r_bert = load_split_bert(CONFIG['bert_paths']['va_r'], "REAL VAL Bert_News_pre_training")
    va_f_bert = load_split_bert(CONFIG['bert_paths']['va_f'], "FAKE VAL Bert_News_pre_training")
    te_r_bert = load_split_bert(CONFIG['bert_paths']['te_r'], "REAL TEST Bert_News_pre_training")
    te_f_bert = load_split_bert(CONFIG['bert_paths']['te_f'], "FAKE TEST Bert_News_pre_training")

    tr_r_f, tr_r_b = match_data_direct(tr_r_groups, tr_r_order, tr_r_bert, "REAL TRAIN")
    tr_f_f, tr_f_b = match_data_direct(tr_f_groups, tr_f_order, tr_f_bert, "FAKE TRAIN")
    va_r_f, va_r_b = match_data_direct(va_r_groups, va_r_order, va_r_bert, "REAL VAL")
    va_f_f, va_f_b = match_data_direct(va_f_groups, va_f_order, va_f_bert, "FAKE VAL")
    te_r_f, te_r_b = match_data_direct(te_r_groups, te_r_order, te_r_bert, "REAL TEST")
    te_f_f, te_f_b = match_data_direct(te_f_groups, te_f_order, te_f_bert, "FAKE TEST")

    def merge(rf, rb, ff, fb):
        xf = np.concatenate([rf, ff], axis=0).astype(np.float32)
        xb = np.concatenate([rb, fb], axis=0).astype(np.float32)
        y = np.array([0] * len(rf) + [1] * len(ff), dtype=np.float32)
        return xf, xb, y

    return (
        merge(tr_r_f, tr_r_b, tr_f_f, tr_f_b),
        merge(va_r_f, va_r_b, va_f_f, va_f_b),
        merge(te_r_f, te_r_b, te_f_f, te_f_b)
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
# 4. Loss Function
# ==========================================

class ManifoldLoss(nn.Module):
    def __init__(self, beta=0.5, delta=2.0):
        super(ManifoldLoss, self).__init__()

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
# 5. Training and Evaluation
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


def select_balanced_threshold(model, loader, device):

    best_threshold = 0.5
    best_score = float("inf")
    best_metrics = None

    thresholds = np.arange(
        CONFIG['threshold_min'],
        CONFIG['threshold_max'] + CONFIG['threshold_step'],
        CONFIG['threshold_step']
    )

    target = CONFIG['target_score']

    for th in thresholds:
        acc, prec, rec, f1, auc = evaluate(
            model,
            loader,
            device,
            threshold=float(th)
        )

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


def main():
    set_seed(CONFIG['seed'])

    device = torch.device('cpu')

    train_data, val_data, test_data = load_pre_split_dataset()

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
        delta=CONFIG['delta']
    )

    save_path = CONFIG['save_path']
    save_dir = os.path.dirname(save_path)

    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    threshold_save_path = CONFIG['threshold_save_path']
    threshold_save_dir = os.path.dirname(threshold_save_path)

    if threshold_save_dir and not os.path.exists(threshold_save_dir):
        os.makedirs(threshold_save_dir, exist_ok=True)


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
            model,
            val_loader,
            device,
            threshold=0.5
        )

        print(
            f"Epoch {epoch + 1:03d} | "
            f"Loss: {train_loss / max(1, len(train_loader)):.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Val Precision: {val_prec:.4f} | "
            f"Val Recall: {val_rec:.4f} | "
            f"Val F1: {val_f1:.4f} | "
        )

    torch.save(model.state_dict(), save_path)

    print(f"\n>>> Training finished after exactly {CONFIG['epochs']} epochs.")
    print(f">>> Final epoch model saved to: {save_path}")

    print("\n>>> Selecting balanced threshold on validation set...")

    best_threshold, val_metrics = select_balanced_threshold(
        model,
        val_loader,
        device
    )

    val_acc, val_prec, val_rec, val_f1, val_auc = val_metrics

    threshold_info = {
        "selected_threshold": best_threshold,
        "validation_accuracy": val_acc,
        "validation_precision": val_prec,
        "validation_recall": val_rec,
        "validation_f1": val_f1,
        "validation_auc": val_auc,
        "seed": CONFIG['seed'],
        "epochs": CONFIG['epochs']
    }

    with open(threshold_save_path, "w", encoding="utf-8") as f:
        json.dump(threshold_info, f, ensure_ascii=False, indent=2)


    print("\n" + "=" * 40)
    print("Final Test Evaluation")
    print("=" * 40)

    model.load_state_dict(
        torch.load(save_path, map_location=device, weights_only=True)
    )

    acc, prec, rec, f1, auc = evaluate(
        model,
        test_loader,
        device,
        threshold=best_threshold
    )

    print(
        f"Accuracy : {acc:.4f}\n"
        f"Precision: {prec:.4f}\n"
        f"Recall   : {rec:.4f}\n"
        f"F1-Score : {f1:.4f}\n"
        + "=" * 40
    )


if __name__ == '__main__':
    main()