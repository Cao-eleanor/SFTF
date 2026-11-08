"""
SFTF Model Efficiency Analysis (Enhanced + Reproducible, No Plotting)
====================================================================
This script measures:
  - Training time per epoch
  - Inference time per sample
  - Memory consumption
  - FLOPs estimation
  - Throughput
  - Parameter count
  - Scalability across batch sizes / dataset sizes
  - LaTeX-ready table rows for selected training epochs

Notes:
  - Randomness is controlled by a fixed seed.
  - Timing results may still have small fluctuations due to OS scheduling,
    CPU/GPU load, temperature, and background processes.
"""

import os

# ─────────────────────────────────────────────
# 0. Reproducibility environment variables
# Must be set before importing torch / CUDA initialization
# ─────────────────────────────────────────────
SEED = 8
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import time
import random
import tracemalloc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────────────────────────
# 1. Reproducibility
# ─────────────────────────────────────────────
def set_seed(seed: int = SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Strong deterministic mode
    # warn_only=True prevents crashing if some operations are not fully deterministic.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


set_seed(SEED)


# ─────────────────────────────────────────────
# 2. Try to import optional profiling libs
# ─────────────────────────────────────────────
try:
    from thop import profile as thop_profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


# ─────────────────────────────────────────────
# 3. Configuration
# ─────────────────────────────────────────────
CONFIG = {
    "seed": SEED,

    "num_channels":   36,
    "num_time_bins":  100,
    "k_selected":     12,
    "embed_dim":      64,
    "metric_dim":     128,
    "bert_dim":       768,
    "batch_size":     32,
    "learning_rate":  3e-4,
    "beta":           0.5,
    "delta":          2.0,
    "eta":            1.0,
    "bert_weight":    0.2,
    "bert_drop_sample_prob": 0.2,
    "bert_mask_prob": 0.2,
    "bert_noise_std": 0.3,

    # Synthetic benchmark settings
    "benchmark_train_samples": 960,
    "benchmark_train_epochs": 100,
    "benchmark_train_batch_size": 32,

    # DataLoader reproducibility
    "num_workers": 0,
}


# ─────────────────────────────────────────────
# 4. Model Definition
# ─────────────────────────────────────────────
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
        # x_signal: (B, P, M)
        fft_res = torch.fft.rfft(x_signal, dim=-1)          # (B, P, M//2+1), complex
        selected_fft = fft_res[:, :, :self.k_selected]      # (B, P, K)

        meta_trend_coeffs = torch.cat(
            [selected_fft.real, selected_fft.imag],
            dim=-1
        )                                                   # (B, P, 2K)

        path_tokens = self.path_projection(meta_trend_coeffs)  # (B, P, D)
        attn_output, _ = self.attention(path_tokens, path_tokens, path_tokens)
        h_t = self.norm(attn_output + path_tokens)

        return torch.mean(h_t, dim=1)                       # (B, D)


class ManifoldFakeNewsNet(nn.Module):
    def __init__(self):
        super().__init__()
        C = CONFIG

        self.struct_encoder = MetaTrendAttentionEncoder(
            C["num_channels"],
            C["k_selected"],
            C["embed_dim"]
        )

        self.bert_fc = nn.Sequential(
            nn.Linear(C["bert_dim"], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, C["embed_dim"])
        )

        self.phi = nn.Sequential(
            nn.Linear(C["embed_dim"] * 2, 256),
            nn.ReLU(),
            nn.Linear(256, C["metric_dim"])
        )

        self.center = nn.Parameter(torch.randn(1, C["metric_dim"]))
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

            noise_std = CONFIG.get("bert_noise_std", 0.0)
            if noise_std > 0:
                x_bert = x_bert + torch.randn_like(x_bert) * noise_std

        h_s = self.bert_fc(x_bert) * CONFIG["bert_weight"]

        h_i = torch.cat([h_t, h_s], dim=1)
        phi_h = self.phi(h_i)

        d_i = torch.norm(phi_h - self.center, p=2, dim=1)
        y_hat = torch.sigmoid(CONFIG["eta"] * (d_i - self.tau))

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

        l_dist = torch.mean(
            real_mask * d_i ** 2 +
            fake_mask * F.relu(self.delta - d_i) ** 2
        )

        return l_ce + self.beta * l_dist


# ─────────────────────────────────────────────
# 5. Synthetic Dataset
# ─────────────────────────────────────────────
class SyntheticDataset(Dataset):
    def __init__(self, n, seed=SEED):
        C = CONFIG

        # Local generator for reproducible synthetic dataset creation
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        self.x_freq = torch.randn(
            n,
            C["num_channels"],
            C["num_time_bins"],
            generator=generator
        )

        self.x_bert = torch.randn(
            n,
            C["bert_dim"],
            generator=generator
        )

        self.labels = torch.randint(
            0,
            2,
            (n,),
            generator=generator
        ).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.x_freq[i], self.x_bert[i], self.labels[i]


# ─────────────────────────────────────────────
# 6. Helper
# ─────────────────────────────────────────────
SEP = "=" * 72


def hr():
    print(SEP)


def section(title):
    print(f"\n{SEP}\n{title}\n{SEP}")


def seed_worker(worker_id):
    worker_seed = CONFIG["seed"] + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# ─────────────────────────────────────────────
# 7. Parameter Count
# ─────────────────────────────────────────────
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ─────────────────────────────────────────────
# 8. Memory helpers
# ─────────────────────────────────────────────
def get_gpu_memory_mb(device):
    if device.type == "cuda":
        return torch.cuda.memory_allocated(device) / 1024 ** 2
    return 0.0


def get_cpu_memory_mb():
    if HAS_PSUTIL:
        proc = psutil.Process()
        return proc.memory_info().rss / 1024 ** 2
    return float("nan")


# ─────────────────────────────────────────────
# 9. Inference Latency Benchmark
# ─────────────────────────────────────────────
def benchmark_inference(model, device, batch_size=32, warmup=5, repeats=50, seed=SEED):
    model.eval()
    C = CONFIG

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + batch_size)

    x_freq = torch.randn(
        batch_size,
        C["num_channels"],
        C["num_time_bins"],
        generator=generator
    ).to(device)

    x_bert = torch.randn(
        batch_size,
        C["bert_dim"],
        generator=generator
    ).to(device)

    with torch.no_grad():
        for _ in range(warmup):
            model(x_freq, x_bert)

    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []

    with torch.no_grad():
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(x_freq, x_bert)

            if device.type == "cuda":
                torch.cuda.synchronize()

            latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies = np.array(latencies)

    return {
        "batch_size": batch_size,
        "mean_batch_ms": float(np.mean(latencies)),
        "std_batch_ms": float(np.std(latencies)),
        "p50_batch_ms": float(np.percentile(latencies, 50)),
        "p95_batch_ms": float(np.percentile(latencies, 95)),
        "mean_per_sample_ms": float(np.mean(latencies)) / batch_size,
        "throughput_samples_per_sec": batch_size / (np.mean(latencies) / 1000.0),
    }


# ─────────────────────────────────────────────
# 10. Training Throughput Benchmark
# ─────────────────────────────────────────────
def benchmark_training(model, device, n_samples=1000, batch_size=32, epochs=100, seed=SEED):
    set_seed(seed)

    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"]
    )

    criterion = ManifoldLoss(
        CONFIG["beta"],
        CONFIG["delta"]
    )

    dataset = SyntheticDataset(n_samples, seed=seed)

    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker if CONFIG["num_workers"] > 0 else None,
        generator=loader_generator
    )

    results = []

    for epoch in range(epochs):
        if device.type == "cuda":
            torch.cuda.synchronize()

        t_epoch_start = time.perf_counter()
        epoch_loss = 0.0
        n_batches = 0

        for x_f, x_b, y in loader:
            x_f = x_f.to(device)
            x_b = x_b.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            y_hat, d_i = model(x_f, x_b)
            loss = criterion(y_hat, y, d_i)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if device.type == "cuda":
            torch.cuda.synchronize()

        epoch_secs = time.perf_counter() - t_epoch_start

        results.append({
            "epoch": epoch + 1,
            "epoch_secs": epoch_secs,
            "avg_loss": epoch_loss / max(1, n_batches),
            "samples_per_sec": n_samples / epoch_secs,
        })

    return results


def summarize_training_results(results):
    secs = np.array([r["epoch_secs"] for r in results], dtype=np.float64)
    sps = np.array([r["samples_per_sec"] for r in results], dtype=np.float64)
    loss = np.array([r["avg_loss"] for r in results], dtype=np.float64)

    return {
        "epochs": len(results),
        "time_mean": float(np.mean(secs)),
        "time_std": float(np.std(secs)),
        "time_min": float(np.min(secs)),
        "time_max": float(np.max(secs)),
        "sps_mean": float(np.mean(sps)),
        "sps_std": float(np.std(sps)),
        "sps_min": float(np.min(sps)),
        "sps_max": float(np.max(sps)),
        "loss_start": float(loss[0]),
        "loss_end": float(loss[-1]),
    }


def print_training_latex_table(results, selected_epochs=(20, 50, 100)):
    print("\nLaTeX table rows for training benchmark:")
    print("-" * 72)

    selected_epochs = set(selected_epochs)

    for r in results:
        if r["epoch"] in selected_epochs:
            print(
                f"Training Epoch {r['epoch']} & "
                f"{r['epoch_secs']:.2f} & "
                f"{r['samples_per_sec']:.0f} \\\\"
            )

    print("-" * 72)


# ─────────────────────────────────────────────
# 11. Scalability
# ─────────────────────────────────────────────
def benchmark_scalability(model, device, sizes=(500, 1000, 5000, 10000), seed=SEED):
    model.eval()
    C = CONFIG
    results = []

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 999)

    with torch.no_grad():
        for n in sizes:
            x_freq = torch.randn(
                n,
                C["num_channels"],
                C["num_time_bins"],
                generator=generator
            ).to(device)

            x_bert = torch.randn(
                n,
                C["bert_dim"],
                generator=generator
            ).to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            bs = 64

            for i in range(0, n, bs):
                model(x_freq[i:i + bs], x_bert[i:i + bs])

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - t0
            gpu_mb = get_gpu_memory_mb(device)

            results.append({
                "n_samples": n,
                "total_secs": elapsed,
                "throughput_sps": n / elapsed,
                "gpu_mem_mb": gpu_mb,
            })

    return results


# ─────────────────────────────────────────────
# 12. Memory footprint during inference
# ─────────────────────────────────────────────
def benchmark_memory(model, device, batch_size=32, seed=SEED):
    C = CONFIG

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1234)

    x_freq = torch.randn(
        batch_size,
        C["num_channels"],
        C["num_time_bins"],
        generator=generator
    ).to(device)

    x_bert = torch.randn(
        batch_size,
        C["bert_dim"],
        generator=generator
    ).to(device)

    tracemalloc.start()
    cpu_before = get_cpu_memory_mb()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.eval()

    with torch.no_grad():
        model(x_freq, x_bert)

    cpu_after = get_cpu_memory_mb()
    _, peak_trace = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    result = {
        "batch_size": batch_size,
        "cpu_rss_delta_mb": cpu_after - cpu_before,
        "peak_trace_mb": peak_trace / 1024 ** 2,
    }

    if device.type == "cuda":
        peak_gpu = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        result["gpu_peak_mb"] = peak_gpu

    return result


# ─────────────────────────────────────────────
# 13. Theoretical complexity
# ─────────────────────────────────────────────
def print_complexity_analysis():
    C = CONFIG

    P = C["num_channels"]
    M = C["num_time_bins"]
    K = C["k_selected"]
    D = C["embed_dim"]
    d_bert = C["bert_dim"]
    d_e = C["metric_dim"]

    section("Theoretical Complexity Analysis")

    print("Key dimensions:")
    print(f"  P (meta-paths)           = {P}")
    print(f"  M (time bins)            = {M}")
    print(f"  K (freq components)      = {K}")
    print(f"  D (embed dim)            = {D}")
    print(f"  d_bert                   = {d_bert}")
    print(f"  d_e (metric dim)         = {d_e}")

    print("\n── Time Complexity (per sample) ──")

    fft_ops = M * np.log2(M)
    proj_ops = P * K * 2 * D
    attn_ops = P * P * D
    bert_ops = d_bert * 256 + 256 * D
    phi_ops = D * 2 * 256 + 256 * d_e
    total_ops = fft_ops + proj_ops + attn_ops + bert_ops + phi_ops

    print(f"  FFT  (P×M·log M)         ≈ {P:.0f} × {fft_ops:.0f} = {P * fft_ops:,.0f}")
    print(f"  Path projection (P·2K·D) ≈ {proj_ops:,.0f}")
    print(f"  Multi-head attn (P²·D)   ≈ {attn_ops:,.0f}")
    print(f"  BERT-FC                  ≈ {bert_ops:,.0f}")
    print(f"  MLP-phi                  ≈ {phi_ops:,.0f}")
    print(f"  Total FLOPs (approx)     ≈ {total_ops:,.0f}")
    print("\n  Overall: O(P·M·logM + P²·D + d_bert·D)")
    print(f"  Bottleneck: BERT-FC ({bert_ops:,.0f}) then Attn ({attn_ops:,.0f})")

    print("\n── Space Complexity ──")

    signal_mem = P * M * 4
    fft_mem = P * (M // 2 + 1) * 8
    attn_mem = P * P * 4

    print(f"  Signal matrix  (P×M)     = {signal_mem / 1024:.1f} KB per sample")
    print(f"  FFT output     (P×M/2)   = {fft_mem / 1024:.1f} KB per sample")
    print(f"  Attention map  (P×P)     = {attn_mem} bytes per sample")
    print("\n  Overall: O(P·M) per sample → scales LINEARLY with dataset size")

    print("\n── Preprocessing (offline) ──")
    print("  LLM sentiment labeling: O(N × C_LLM)  — done ONCE, not at inference")
    print("  Signal matrix build:    O(N × P × M)  — done ONCE per dataset")
    print("  BERT embedding:         O(N × L × d)  — done ONCE, cached as .npy")
    print("  → Inference cost excludes all preprocessing overhead")


# ─────────────────────────────────────────────
# 14. FLOPs via thop
# ─────────────────────────────────────────────
def measure_flops(model, device):
    if not HAS_THOP:
        print("  [thop not installed] skipping FLOPs measurement.")
        print("  Install with: pip install thop")
        return None

    C = CONFIG

    generator = torch.Generator(device="cpu")
    generator.manual_seed(CONFIG["seed"] + 2024)

    x_freq = torch.randn(
        1,
        C["num_channels"],
        C["num_time_bins"],
        generator=generator
    ).to(device)

    x_bert = torch.randn(
        1,
        C["bert_dim"],
        generator=generator
    ).to(device)

    try:
        flops, params = thop_profile(
            model,
            inputs=(x_freq, x_bert),
            verbose=False
        )
        return flops, params
    except Exception as e:
        print(f"  [thop] failed: {e}")
        return None


# ─────────────────────────────────────────────
# 15. Main
# ─────────────────────────────────────────────
def main():
    set_seed(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nSeed  : {CONFIG['seed']}")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Re-seed immediately before model initialization
    set_seed(CONFIG["seed"])
    model = ManifoldFakeNewsNet().to(device)
    model.eval()

    # A. Parameter count
    section("A. Parameter Count")

    total, trainable = count_parameters(model)

    print(f"  Total parameters    : {total:,}  ({total / 1e6:.3f} M)")
    print(f"  Trainable params    : {trainable:,}")
    print(f"  Model size (fp32)   : {total * 4 / 1024**2:.2f} MB")

    # B. FLOPs
    section("B. FLOPs (single sample forward pass)")

    flops_result = measure_flops(model, device)

    if flops_result:
        flops, params = flops_result
        print(f"  FLOPs  : {flops / 1e6:.2f} M")
        print(f"  Params : {params / 1e6:.3f} M")

    # C. Inference latency
    section("C. Inference Latency")

    for bs in [1, 32, 128]:
        r = benchmark_inference(
            model,
            device,
            batch_size=bs,
            warmup=5,
            repeats=50,
            seed=CONFIG["seed"]
        )

        print(
            f"  batch={bs:3d} | "
            f"mean={r['mean_batch_ms']:.2f}ms | "
            f"p95={r['p95_batch_ms']:.2f}ms | "
            f"per-sample={r['mean_per_sample_ms']:.3f}ms | "
            f"throughput={r['throughput_samples_per_sec']:.0f} sps"
        )

    # D. Memory
    section("D. Memory Consumption (batch=32)")

    mem = benchmark_memory(
        model,
        device,
        batch_size=32,
        seed=CONFIG["seed"]
    )

    print(f"  CPU RSS delta      : {mem['cpu_rss_delta_mb']:.2f} MB")
    print(f"  Peak traced (CPU)  : {mem['peak_trace_mb']:.2f} MB")

    if "gpu_peak_mb" in mem:
        print(f"  GPU peak allocated : {mem['gpu_peak_mb']:.2f} MB")

    # E. Training throughput
    section("E. Training Throughput (synthetic data, multi-epoch)")

    # Re-initialize model before training benchmark for reproducible training curve
    set_seed(CONFIG["seed"])
    train_model = ManifoldFakeNewsNet().to(device)

    train_results = benchmark_training(
        train_model,
        device,
        n_samples=CONFIG["benchmark_train_samples"],
        batch_size=CONFIG["benchmark_train_batch_size"],
        epochs=CONFIG["benchmark_train_epochs"],
        seed=CONFIG["seed"]
    )

    for r in train_results:
        print(
            f"  Epoch {r['epoch']:03d} | "
            f"time={r['epoch_secs']:.2f}s | "
            f"loss={r['avg_loss']:.4f} | "
            f"throughput={r['samples_per_sec']:.0f} sps"
        )

    summary = summarize_training_results(train_results)

    section("E1. Training Throughput Summary")

    print(f"  Epochs benchmarked     : {summary['epochs']}")
    print(f"  Time mean ± std        : {summary['time_mean']:.4f} ± {summary['time_std']:.4f} s")
    print(f"  Time min / max         : {summary['time_min']:.4f} / {summary['time_max']:.4f} s")
    print(f"  Throughput mean ± std  : {summary['sps_mean']:.2f} ± {summary['sps_std']:.2f} sps")
    print(f"  Throughput min / max   : {summary['sps_min']:.2f} / {summary['sps_max']:.2f} sps")
    print(f"  Loss start → end       : {summary['loss_start']:.4f} → {summary['loss_end']:.4f}")

    print_training_latex_table(
        train_results,
        selected_epochs=(20, 50, 100)
    )

    # F. Scalability
    section("F. Scalability (inference, varying dataset size)")

    scale_results = benchmark_scalability(
        model,
        device,
        sizes=[100, 500, 1000, 5000, 10000],
        seed=CONFIG["seed"]
    )

    print(f"  {'N':>7}  {'time(s)':>10}  {'throughput(sps)':>16}  {'gpu_MB':>10}")
    print(f"  {'-' * 7}  {'-' * 10}  {'-' * 16}  {'-' * 10}")

    for r in scale_results:
        gpu_str = f"{r['gpu_mem_mb']:.1f}" if r["gpu_mem_mb"] > 0 else "N/A"

        print(
            f"  {r['n_samples']:>7}  "
            f"{r['total_secs']:>10.2f}  "
            f"{r['throughput_sps']:>16.0f}  "
            f"{gpu_str:>10}"
        )

    # G. Theoretical analysis
    print_complexity_analysis()

    # H. Summary
    section("G. Efficiency Advantage Summary")

    print("""
  ┌──────────────────────┬──────────────────────────────────────────────────┐
  │ Component            │ Optimization Notes                               │
  ├──────────────────────┼──────────────────────────────────────────────────┤
  │ Signal matrix build  │ Offline pre-processing, stored as float32 arrays │
  │ FFT (rfft)           │ O(M·logM), PyTorch FFT, low-cost path encoding   │
  │ Path projection      │ Single linear layer, weight sharing across paths │
  │ Multi-head attention │ P=36 tokens only; O(P²) << full graph attention  │
  │ BERT embedding       │ Pre-computed & cached (.npy), zero cost at infer │
  │ Manifold distance    │ Single L2 norm, low-dimensional metric space     │
  └──────────────────────┴──────────────────────────────────────────────────┘

  Key insight:
    The model complexity is O(P·M·logM + P²·D) per sample,
    where P, M, and D are all small constants.
    This scales linearly with the number of news posts (N),
    making it suitable for large-scale deployment.

  Reproducibility:
    The random seed is fixed for Python, NumPy, and PyTorch.
    Dataset generation, DataLoader shuffling, and model initialization
    are controlled by the same seed.
    Plot generation has been removed.
    """)

    hr()
    print("Enhanced efficiency analysis complete.")
    hr()


if __name__ == "__main__":
    main()