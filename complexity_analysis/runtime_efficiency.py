"""Benchmark the runtime efficiency (FLOPs, latency, memory, throughput) of the manifold fake-news detector on the real dataset."""
import os
from pathlib import Path

# ==========================================
# 0. Reproducibility env variables
# Must be set before torch/CUDA initialization.
# ==========================================
SEED = 8
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import time
import random
import tracemalloc
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# ==========================================
# 0.1 Optional profiling libs
# ==========================================
try:
    from thop import profile as thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False


SFTF_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = SFTF_ROOT / 'model' / 'data'


# ==========================================
# 1. Configuration
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

    'model_ckpt': SFTF_ROOT / 'model' / 'checkpoints' / 'SFTF_weibo25_best.pth',

    'num_channels': 36,
    'num_time_bins': 100,
    'max_time': 172800,
    'k_selected': 12,
    'embed_dim': 64,
    'metric_dim': 128,
    'bert_dim': 768,

    'batch_size': 32,
    'learning_rate': 3e-4,
    'epochs': 100,
    'seed': SEED,

    'beta': 0.5,
    'delta': 2.0,
    'eta': 1.0,

    'bert_weight': 0.2,
    'bert_drop_sample_prob': 0.2,
    'bert_mask_prob': 0.2,
    'bert_noise_std': 0.3,

    'num_workers': 0,
}


# ==========================================
# 1.1 Reproducibility
# ==========================================
def set_seed(seed: int = 8):
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
    except TypeError:
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = CONFIG['seed'] + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


set_seed(CONFIG['seed'])

SEP = "=" * 72


def hr():
    print(SEP)


def section(title):
    print(f"\n{SEP}\n{title}\n{SEP}")


# ==========================================
# 2. Data Processing
# ==========================================
def build_signal_matrix(lines):
    matrix = np.zeros((CONFIG['num_channels'], CONFIG['num_time_bins']), dtype=np.float32)

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


def load_split_bert(bert_path, label="BERT"):
    if not os.path.exists(bert_path):
        raise FileNotFoundError(f"[{label}] file not found: {bert_path}")

    x = np.load(bert_path).astype(np.float32)

    if x.ndim != 2:
        raise ValueError(f"[{label}] expected shape (N, {CONFIG['bert_dim']}), got {x.shape}")

    if x.shape[1] != CONFIG['bert_dim']:
        raise ValueError(f"[{label}] bert_dim mismatch: expected {CONFIG['bert_dim']}, got {x.shape[1]}")

    return x


def match_data_direct(groups, order_nids, bert_vectors, label_str="DATA"):
    n_meta = len(order_nids)
    n_bert = bert_vectors.shape[0]

    if n_meta == 0:
        return (
            np.zeros((0, CONFIG['num_channels'], CONFIG['num_time_bins']), dtype=np.float32),
            np.zeros((0, CONFIG['bert_dim']), dtype=np.float32)
        )

    if n_meta != n_bert:
        raise ValueError(f"[{label_str}] sample count mismatch: metapath samples={n_meta}, BERT samples={n_bert}.")

    x_freq = [build_signal_matrix(groups[nid]) for nid in order_nids]
    x_freq = np.array(x_freq, dtype=np.float32)
    x_bert = bert_vectors.astype(np.float32)

    return x_freq, x_bert


def load_pre_split_dataset():
    tr_r_groups, tr_r_order = parse_metapath_file(CONFIG['paths']['tr_r'])
    tr_f_groups, tr_f_order = parse_metapath_file(CONFIG['paths']['tr_f'])
    va_r_groups, va_r_order = parse_metapath_file(CONFIG['paths']['va_r'])
    va_f_groups, va_f_order = parse_metapath_file(CONFIG['paths']['va_f'])
    te_r_groups, te_r_order = parse_metapath_file(CONFIG['paths']['te_r'])
    te_f_groups, te_f_order = parse_metapath_file(CONFIG['paths']['te_f'])

    tr_r_bert = load_split_bert(CONFIG['bert_paths']['tr_r'], "REAL TRAIN BERT")
    tr_f_bert = load_split_bert(CONFIG['bert_paths']['tr_f'], "FAKE TRAIN BERT")
    va_r_bert = load_split_bert(CONFIG['bert_paths']['va_r'], "REAL VAL BERT")
    va_f_bert = load_split_bert(CONFIG['bert_paths']['va_f'], "FAKE VAL BERT")
    te_r_bert = load_split_bert(CONFIG['bert_paths']['te_r'], "REAL TEST BERT")
    te_f_bert = load_split_bert(CONFIG['bert_paths']['te_f'], "FAKE TEST BERT")

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
# 3. Model
# ==========================================
class MetaTrendAttentionEncoder(nn.Module):
    def __init__(self, num_channels, k_selected, embed_dim):
        super().__init__()
        self.k_selected = k_selected
        self.path_projection = nn.Linear(k_selected * 2, embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x_signal):
        fft_res = torch.fft.rfft(x_signal, dim=-1)
        selected_fft = fft_res[:, :, :self.k_selected]
        meta_trend_coeffs = torch.cat([selected_fft.real, selected_fft.imag], dim=-1)
        path_tokens = self.path_projection(meta_trend_coeffs)
        attn_output, _ = self.attention(path_tokens, path_tokens, path_tokens)
        h_t = self.norm(attn_output + path_tokens)
        return torch.mean(h_t, dim=1)


class ManifoldFakeNewsNet(nn.Module):
    def __init__(self):
        super().__init__()

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
                drop = (torch.rand(x_bert.size(0), 1, device=x_bert.device) < p_drop).float()
                x_bert = x_bert * (1.0 - drop)

            p_mask = CONFIG.get('bert_mask_prob', 0.0)
            if p_mask > 0:
                keep = (torch.rand_like(x_bert) > p_mask).float()
                x_bert = x_bert * keep

            noise_std = CONFIG.get('bert_noise_std', 0.0)
            if noise_std > 0:
                x_bert = x_bert + torch.randn_like(x_bert) * noise_std

        h_s = self.bert_fc(x_bert)
        h_s = h_s * CONFIG.get('bert_weight', 1.0)

        h_i = torch.cat([h_t, h_s], dim=1)
        phi_h = self.phi(h_i)
        d_i = torch.norm(phi_h - self.center, p=2, dim=1)
        y_hat = torch.sigmoid(CONFIG['eta'] * (d_i - self.tau))

        return y_hat, d_i


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
# 4. Evaluation
# ==========================================
def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():
        for x_f, x_b, y in loader:
            x_f = x_f.to(device, non_blocking=True)
            x_b = x_b.to(device, non_blocking=True)

            out, _ = model(x_f, x_b)

            prob = out.detach().cpu().numpy()
            pred = (out > 0.5).float().cpu().numpy()

            y_score.extend(prob.tolist())
            y_pred.extend(pred.tolist())
            y_true.extend(y.numpy().tolist())

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
# 5. Efficiency Helpers
# ==========================================
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_cpu_memory_mb():
    proc = psutil.Process()
    return proc.memory_info().rss / 1024 ** 2


def benchmark_real_test_inference(model, loader, device, warmup_batches=2):
    model.eval()

    with torch.no_grad():
        warmup_count = 0
        for x_f, x_b, _ in loader:
            x_f = x_f.to(device, non_blocking=True)
            x_b = x_b.to(device, non_blocking=True)
            model(x_f, x_b)

            warmup_count += 1
            if warmup_count >= warmup_batches:
                break

    if device.type == 'cuda':
        torch.cuda.synchronize()

    total_samples = 0
    batch_latencies = []

    t0 = time.perf_counter()

    with torch.no_grad():
        for x_f, x_b, _ in loader:
            x_f = x_f.to(device, non_blocking=True)
            x_b = x_b.to(device, non_blocking=True)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            tb = time.perf_counter()
            model(x_f, x_b)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            batch_latencies.append((time.perf_counter() - tb) * 1000.0)
            total_samples += x_f.size(0)

    total_secs = time.perf_counter() - t0

    return {
        'total_samples': total_samples,
        'total_secs': total_secs,
        'mean_batch_ms': float(np.mean(batch_latencies)),
        'p95_batch_ms': float(np.percentile(batch_latencies, 95)),
        'per_sample_ms': total_secs * 1000.0 / total_samples,
        'throughput_sps': total_samples / total_secs
    }


def benchmark_real_test_memory(model, loader, device):
    model.eval()

    cpu_before = get_cpu_memory_mb()
    tracemalloc.start()

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for x_f, x_b, _ in loader:
            x_f = x_f.to(device, non_blocking=True)
            x_b = x_b.to(device, non_blocking=True)
            model(x_f, x_b)

    cpu_after = get_cpu_memory_mb()
    _, peak_trace = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result = {
        'cpu_rss_delta_mb': cpu_after - cpu_before,
        'peak_trace_mb': peak_trace / 1024 ** 2
    }

    if device.type == 'cuda':
        result['gpu_peak_mb'] = torch.cuda.max_memory_allocated(device) / 1024 ** 2

    return result


def benchmark_real_batchsize_sweep(model, test_data, device, batch_sizes=(1, 16, 32, 64, 128)):
    x_f, x_b, y = test_data
    results = []

    for bs in batch_sizes:
        loader = DataLoader(
            HybridDataset(x_f, x_b, y),
            batch_size=bs,
            shuffle=False,
            num_workers=CONFIG['num_workers'],
            pin_memory=(device.type == 'cuda'),
            worker_init_fn=seed_worker if CONFIG['num_workers'] > 0 else None
        )

        r = benchmark_real_test_inference(model, loader, device)
        r['batch_size'] = bs
        results.append(r)

    return results


def benchmark_real_training_one_epoch(model, train_loader, device):
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = ManifoldLoss(beta=CONFIG['beta'], delta=CONFIG['delta'])

    if device.type == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()

    total_loss = 0.0
    total_samples = 0
    n_batches = 0

    for x_f, x_b, y in train_loader:
        x_f = x_f.to(device, non_blocking=True)
        x_b = x_b.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        y_hat, d_i = model(x_f, x_b)
        loss = criterion(y_hat, y, d_i)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_samples += x_f.size(0)
        n_batches += 1

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0

    return {
        'epoch_secs': elapsed,
        'avg_loss': total_loss / max(1, n_batches),
        'samples_per_sec': total_samples / elapsed,
        'total_samples': total_samples
    }


def measure_flops(model, device):
    if not HAS_THOP:
        return None

    generator = torch.Generator(device='cpu')
    generator.manual_seed(CONFIG['seed'] + 2024)

    x_freq = torch.randn(
        1,
        CONFIG['num_channels'],
        CONFIG['num_time_bins'],
        generator=generator
    ).to(device)

    x_bert = torch.randn(
        1,
        CONFIG['bert_dim'],
        generator=generator
    ).to(device)

    try:
        flops, params = thop_profile(model, inputs=(x_freq, x_bert), verbose=False)
        return flops, params
    except Exception:
        return None


# ==========================================
# 6. Main
# ==========================================
def main():
    set_seed(CONFIG['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    section("Device")
    print(f"Seed: {CONFIG['seed']}")
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    section("Loading Real Dataset")
    train_data, val_data, test_data = load_pre_split_dataset()

    train_loader_generator = torch.Generator()
    train_loader_generator.manual_seed(CONFIG['seed'])

    train_loader = DataLoader(
        HybridDataset(*train_data),
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=(device.type == 'cuda'),
        worker_init_fn=seed_worker if CONFIG['num_workers'] > 0 else None,
        generator=train_loader_generator
    )

    val_loader = DataLoader(
        HybridDataset(*val_data),
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=(device.type == 'cuda'),
        worker_init_fn=seed_worker if CONFIG['num_workers'] > 0 else None
    )

    test_loader = DataLoader(
        HybridDataset(*test_data),
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=(device.type == 'cuda'),
        worker_init_fn=seed_worker if CONFIG['num_workers'] > 0 else None
    )

    print(f"Train samples: {len(train_data[2])}")
    print(f"Val samples  : {len(val_data[2])}")
    print(f"Test samples : {len(test_data[2])}")

    section("Building Model")
    set_seed(CONFIG['seed'])
    model = ManifoldFakeNewsNet().to(device)

    if not os.path.exists(CONFIG['model_ckpt']):
        raise FileNotFoundError(f"model weight file not found: {CONFIG['model_ckpt']}")

    model.load_state_dict(torch.load(CONFIG['model_ckpt'], map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded checkpoint: {CONFIG['model_ckpt']}")

    section("A. Parameter Count")
    total, trainable = count_parameters(model)
    print(f"Total parameters   : {total:,} ({total / 1e6:.3f} M)")
    print(f"Trainable params   : {trainable:,}")
    print(f"Model size (FP32)  : {total * 4 / 1024**2:.2f} MB")

    section("B. FLOPs (optional)")
    flops_result = measure_flops(model, device)

    if flops_result is None:
        print("thop not installed or FLOPs counting failed; skipping this item.")
        print("To count FLOPs, run: pip install thop")
    else:
        flops, params = flops_result
        print(f"FLOPs  : {flops / 1e6:.2f} M")
        print(f"Params : {params / 1e6:.3f} M")

    # C. Real Test Set Metrics output removed as requested.
    # To re-enable, uncomment the code below.
    #
    # section("C. Real Test Set Metrics")
    # acc, prec, rec, f1, auc = evaluate(model, test_loader, device)
    # print(f"Accuracy : {acc:.4f}")
    # print(f"Precision: {prec:.4f}")
    # print(f"Recall   : {rec:.4f}")
    # print(f"F1-Score : {f1:.4f}")
    # print(f"AUC      : {auc:.4f}" if not np.isnan(auc) else "AUC      : nan")

    section("D. Real Test Set Inference Time")
    infer_stats = benchmark_real_test_inference(model, test_loader, device)

    print(f"Total test samples      : {infer_stats['total_samples']}")
    print(f"Total inference time    : {infer_stats['total_secs']:.4f} s")
    print(f"Mean batch latency      : {infer_stats['mean_batch_ms']:.4f} ms")
    print(f"P95 batch latency       : {infer_stats['p95_batch_ms']:.4f} ms")
    print(f"Mean per-sample latency : {infer_stats['per_sample_ms']:.4f} ms")
    print(f"Throughput              : {infer_stats['throughput_sps']:.2f} samples/s")

    section("E. Real Test Set Memory")
    mem_stats = benchmark_real_test_memory(model, test_loader, device)

    print(f"CPU RSS delta : {mem_stats['cpu_rss_delta_mb']:.4f} MB")
    print(f"Peak traced   : {mem_stats['peak_trace_mb']:.4f} MB")

    if 'gpu_peak_mb' in mem_stats:
        print(f"GPU peak alloc: {mem_stats['gpu_peak_mb']:.4f} MB")

    section("F. Real Test Set Batch Size Sweep")
    sweep_results = benchmark_real_batchsize_sweep(
        model,
        test_data,
        device,
        batch_sizes=(1, 16, 32, 64, 128)
    )

    print(f"{'Batch':>8} {'Total(s)':>12} {'Per-sample(ms)':>16} {'Throughput(sps)':>18}")
    print("-" * 60)

    for r in sweep_results:
        print(
            f"{r['batch_size']:>8} "
            f"{r['total_secs']:>12.4f} "
            f"{r['per_sample_ms']:>16.4f} "
            f"{r['throughput_sps']:>18.2f}"
        )

    section("G. Real Training Throughput (1 Epoch Benchmark)")
    set_seed(CONFIG['seed'])

    train_loader_generator_benchmark = torch.Generator()
    train_loader_generator_benchmark.manual_seed(CONFIG['seed'])

    train_loader_for_benchmark = DataLoader(
        HybridDataset(*train_data),
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=(device.type == 'cuda'),
        worker_init_fn=seed_worker if CONFIG['num_workers'] > 0 else None,
        generator=train_loader_generator_benchmark
    )

    train_stats = benchmark_real_training_one_epoch(
        model,
        train_loader_for_benchmark,
        device
    )

    print(f"Samples in epoch : {train_stats['total_samples']}")
    print(f"Epoch time       : {train_stats['epoch_secs']:.4f} s")
    print(f"Avg loss         : {train_stats['avg_loss']:.4f}")
    print(f"Throughput       : {train_stats['samples_per_sec']:.2f} samples/s")

    hr()
    print("Real-dataset efficiency analysis complete.")
    hr()


if __name__ == '__main__':
    main()