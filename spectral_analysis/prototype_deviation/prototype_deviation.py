"""
Compare reconstructed meta-trends with authentic training prototypes across sentiment meta-paths.
RQ3 spectral analysis. See README.md for inputs, outputs, and interpretation.
"""

from pathlib import Path
import argparse

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib

# =========================
# 0) Experiment configuration
# =========================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[2] / "model" / "data"  # SFTF/model/data

CONFIG = {
    "paths": {
        "tr_r": str(DATA_DIR / "metapath_data" / "Deepseek_real_train_metapaths.txt"),
        "te_r": str(DATA_DIR / "metapath_data" / "Deepseek_real_test_metapaths.txt"),
        "te_f": str(DATA_DIR / "metapath_data" / "Deepseek_fake_test_metapaths.txt"),
    },
    "num_channels": 36,
    "num_time_bins": 100,
    "max_time": 172800,
    "k_recon": 12,
    "active_power_threshold": 1e-6,
    "seed": 8,

    "normalize": "zscore",
    "distance_metric": "mse",

    "out_dir": str(SCRIPT_DIR / "outputs"),

    # Typography and style
    "font_family": "Times New Roman",
    "font_weight": "bold",
    "title_fontsize": 16,  # Title font size
    "label_fontsize": 14,  # Axis-label font size
    "tick_fontsize": 13,  # Tick-label font size
    "legend_fontsize": 12,  # Legend font size

    # Curve colors
    "colors": {
        "train_real": "#FF0000",  # Red: authentic training prototype
        "test_real": "#1f77b4",  # Blue: real test example
        "test_fake": "#ff7f0e",  # Orange: fake test example
    },

    "dpi": 600,
}

np.random.seed(CONFIG["seed"])


# =========================
# Global typography
# =========================
def setup_fonts():
    """Set global typography to bold Times New Roman."""
    rcParams['font.family'] = CONFIG["font_family"]
    rcParams['font.weight'] = CONFIG["font_weight"]
    rcParams['axes.labelweight'] = CONFIG["font_weight"]
    rcParams['axes.titleweight'] = CONFIG["font_weight"]
    rcParams['figure.titlesize'] = CONFIG["title_fontsize"]
    rcParams['figure.titleweight'] = CONFIG["font_weight"]
    rcParams['axes.labelsize'] = CONFIG["label_fontsize"]
    rcParams['axes.titlesize'] = CONFIG["title_fontsize"]
    rcParams['xtick.labelsize'] = CONFIG["tick_fontsize"]
    rcParams['ytick.labelsize'] = CONFIG["tick_fontsize"]
    rcParams['legend.fontsize'] = CONFIG["legend_fontsize"]
    rcParams['legend.title_fontsize'] = CONFIG["legend_fontsize"]
    rcParams['pdf.fonttype'] = 42  # Embed TrueType fonts when using the PDF backend.
    rcParams['ps.fonttype'] = 42  # Embed TrueType fonts when using the PostScript backend.

    # Math typography
    rcParams['mathtext.fontset'] = 'custom'
    rcParams['mathtext.it'] = f'{CONFIG["font_family"]}:italic'
    rcParams['mathtext.rm'] = CONFIG["font_family"]
    rcParams['mathtext.tt'] = CONFIG["font_family"]
    rcParams['mathtext.bf'] = f'{CONFIG["font_family"]}:bold'
    rcParams['mathtext.cal'] = CONFIG["font_family"]
    rcParams['mathtext.sf'] = CONFIG["font_family"]


# Apply font settings.
setup_fonts()


# =========================
# 1) Convert meta-paths into 36-by-M signals
# =========================
def build_signal_matrix(lines):
    matrix = np.zeros((CONFIG["num_channels"], CONFIG["num_time_bins"]), dtype=np.float32)
    for line in lines:
        parts = line.split("——")
        try:
            n_sent = int(parts[0].split("_")[2])  # 0/1/2
            c1_sent = int(parts[2].split("_")[2])
            t_val = float(parts[1])
            t_idx = min(int((t_val / CONFIG["max_time"]) * CONFIG["num_time_bins"]),
                        CONFIG["num_time_bins"] - 1)

            if len(parts) >= 5 and "None" not in parts[4]:
                c2_sent = int(parts[4].split("_")[2])
                ch_idx = 9 + (n_sent * 9) + (c1_sent * 3) + c2_sent
            else:
                ch_idx = n_sent * 3 + c1_sent

            if ch_idx < CONFIG["num_channels"]:
                matrix[ch_idx, t_idx] += 1
        except:
            continue
    return np.log1p(matrix).astype(np.float32)


def parse_metapath_file(path):
    groups = {}
    order = []
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("——")
            nid = parts[0].split("_")[1]  # N_4_1 -> "4"
            if nid not in groups:
                groups[nid] = []
                order.append(nid)
            groups[nid].append(line)
    return groups, order


def load_signals(metapath_path):
    groups, order = parse_metapath_file(metapath_path)
    X = np.stack([build_signal_matrix(groups[nid]) for nid in order], axis=0)  # (N,36,M)
    return X, order


# =========================
# 2) FFT + active(total_power)
# =========================
def rfft_and_total_power(X):
    """
    X: (N,C,M)
    returns:
      fft: (N,C,kmax+1) complex
      total_power: (N,C) float, sum_{k=1..kmax}|X[k]|^2 exclude DC
    """
    N, C, M = X.shape
    fft = np.fft.rfft(X, axis=-1)
    power = (fft.real ** 2 + fft.imag ** 2)
    kmax = M // 2
    total_power = power[:, :, 1:kmax + 1].sum(axis=-1)
    return fft, total_power.astype(np.float32)


# =========================
# 3) Reconstruct signals with sine and cosine components (k=1..K)
# =========================
def reconstruct_from_coeffs(coeffs_k1_to_K, M, normalize="zscore"):
    """
    coeffs_k1_to_K: (K,) complex for k=1..K
    x[n] = sum_k (a_k cos + b_k sin)
    a_k=(2/M)Re(X[k]), b_k=-(2/M)Im(X[k])
    """
    K = coeffs_k1_to_K.shape[0]
    k = np.arange(1, K + 1)[None, :]
    n = np.arange(0, M)[:, None]
    ang = 2 * np.pi * n * k / M

    A = (2.0 / M) * coeffs_k1_to_K.real
    B = -(2.0 / M) * coeffs_k1_to_K.imag

    x = np.cos(ang) @ A + np.sin(ang) @ B

    if normalize == "zscore":
        mu = x.mean()
        sd = x.std() + 1e-12
        x = (x - mu) / sd
    return x.astype(np.float32)


# =========================
# 4) Meta-path labels
# =========================
SENT_SUP = {0: "+", 1: "0", 2: "-"}


def channel_to_p_label(ch):
    if ch < 9:
        n = ch // 3
        u = ch % 3
        return rf"$P = N^{{{SENT_SUP[n]}}}U^{{{SENT_SUP[u]}}}$"
    else:
        idx = ch - 9
        n = idx // 9
        rem = idx % 9
        u = rem // 3
        v = rem % 3
        return rf"$P = N^{{{SENT_SUP[n]}}}U^{{{SENT_SUP[u]}}}V^{{{SENT_SUP[v]}}}$"


# =========================
# 5) Train-Real prototype
# =========================
def compute_train_real_prototypes(X_train_real):
    fft, total_power = rfft_and_total_power(X_train_real)
    N, C, klen = fft.shape
    M = X_train_real.shape[-1]
    K = min(CONFIG["k_recon"], klen - 1)
    thr = CONFIG["active_power_threshold"]

    proto_wave = [None] * C
    active_counts = [0] * C

    for ch in range(C):
        mask = total_power[:, ch] > thr
        active_counts[ch] = int(mask.sum())
        if active_counts[ch] == 0:
            continue
        mean_coeffs = fft[mask, ch, 1:K + 1].mean(axis=0)
        proto_wave[ch] = reconstruct_from_coeffs(mean_coeffs, M, normalize=CONFIG["normalize"])

    return proto_wave, active_counts


# =========================
# 6) Waveform distance
# =========================
def distance(x, y, metric="mse"):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if metric == "mse":
        return float(np.mean((x - y) ** 2))
    if metric == "l1":
        return float(np.mean(np.abs(x - y)))
    if metric == "neg_corr":
        xc = x - x.mean()
        yc = y - y.mean()
        denom = (np.linalg.norm(xc) * np.linalg.norm(yc) + 1e-12)
        corr = float(np.dot(xc, yc) / denom)
        return float(1.0 - corr)
    raise ValueError("Unknown metric: " + metric)


def select_test_examples_against_proto(X_test, test_nids, proto_wave, want="closest"):
    fft, total_power = rfft_and_total_power(X_test)
    N, C, klen = fft.shape
    M = X_test.shape[-1]
    K = min(CONFIG["k_recon"], klen - 1)
    thr = CONFIG["active_power_threshold"]
    metric = CONFIG["distance_metric"]

    pick_idx = [None] * C
    pick_wave = [None] * C

    for ch in range(C):
        proto = proto_wave[ch]
        if proto is None:
            continue

        candidates = np.where(total_power[:, ch] > thr)[0]
        if len(candidates) == 0:
            continue

        best_idx = None
        best_d = None
        best_wave = None

        for idx in candidates:
            coeffs = fft[idx, ch, 1:K + 1]
            wave = reconstruct_from_coeffs(coeffs, M, normalize=CONFIG["normalize"])
            d = distance(proto, wave, metric=metric)

            if best_d is None:
                best_d = d
                best_idx = int(idx)
                best_wave = wave
            else:
                if want == "closest":
                    if d < best_d:
                        best_d = d
                        best_idx = int(idx)
                        best_wave = wave
                else:  # farthest
                    if d > best_d:
                        best_d = d
                        best_idx = int(idx)
                        best_wave = wave

        pick_idx[ch] = best_idx
        pick_wave[ch] = best_wave

    return pick_idx, pick_wave


# =========================
# 8) Save a square figure for each valid channel
# =========================
def save_per_channel_square_plots(proto_wave,
                                  test_real_wave, test_fake_wave,
                                  test_real_idx, test_fake_idx,
                                  test_real_nids, test_fake_nids,
                                  out_dir):
    os.makedirs(out_dir, exist_ok=True)
    M = CONFIG["num_time_bins"]
    x = np.arange(M)

    for ch in range(CONFIG["num_channels"]):
        proto = proto_wave[ch]
        if proto is None:
            continue

        # Square figure
        plt.figure(figsize=(7.0, 7.0))

        # Authentic training prototype: red solid line
        plt.plot(x, proto, linewidth=2.8, color=CONFIG["colors"]["train_real"],
                 label="Train-Real Prototype")

        # Real test example: blue dashed line
        if test_real_wave[ch] is not None and test_real_idx[ch] is not None:
            idx = test_real_idx[ch]
            nid = test_real_nids[idx] if idx < len(test_real_nids) else "NA"
            plt.plot(x, test_real_wave[ch], linewidth=2.2, linestyle="--",
                     color=CONFIG["colors"]["test_real"],
                     label=f"Test-Real  N_id:{nid}")

        # Fake test example: orange dashed line
        if test_fake_wave[ch] is not None and test_fake_idx[ch] is not None:
            idx = test_fake_idx[ch]
            nid = test_fake_nids[idx] if idx < len(test_fake_nids) else "NA"
            plt.plot(x, test_fake_wave[ch], linewidth=2.2, linestyle="--",
                     color=CONFIG["colors"]["test_fake"],
                     label=f"Test-Fake  N_id:{nid}")

        # Set the title and axis labels.
        plt.title(channel_to_p_label(ch), fontsize=CONFIG["title_fontsize"],
                  fontweight=CONFIG["font_weight"])
        plt.xlabel("Time (binned since news release)", fontsize=CONFIG["label_fontsize"],
                   fontweight=CONFIG["font_weight"])
        ylabel = "Reconstructed (z-score)" if CONFIG["normalize"] == "zscore" else "Reconstructed signal"
        plt.ylabel(ylabel, fontsize=CONFIG["label_fontsize"],
                   fontweight=CONFIG["font_weight"])

        # Set tick typography.
        plt.tick_params(axis='both', which='major', labelsize=CONFIG["tick_fontsize"])

        plt.grid(True, alpha=0.3)

        # Configure the legend.
        legend = plt.legend(fontsize=CONFIG["legend_fontsize"], loc="upper right")
        if legend:
            plt.setp(legend.get_texts(), fontweight=CONFIG["font_weight"])

        plt.tight_layout()

        # Save PNG.
        base_path = os.path.join(out_dir, f"prototype_deviation_ch{ch:02d}")
        plt.savefig(base_path + '.png', dpi=CONFIG["dpi"], bbox_inches='tight', transparent=False)

        plt.close()


# =========================
# main
# =========================
def main():
    print("Loading signals...")
    Xtr_r, _ = load_signals(CONFIG["paths"]["tr_r"])
    Xte_r, te_r_nids = load_signals(CONFIG["paths"]["te_r"])
    Xte_f, te_f_nids = load_signals(CONFIG["paths"]["te_f"])

    print("Train real:", Xtr_r.shape)
    print("Test  real:", Xte_r.shape)
    print("Test  fake:", Xte_f.shape)

    # 1) Train-Real Prototype
    print("Computing Train-Real prototypes...")
    proto_wave, proto_active_counts = compute_train_real_prototypes(Xtr_r)

    # 2) Test Real: closest
    print("Selecting Test-Real closest per channel...")
    te_r_idx, te_r_wave = select_test_examples_against_proto(
        Xte_r, te_r_nids, proto_wave, want="closest"
    )

    # 3) Test Fake: farthest
    print("Selecting Test-Fake farthest per channel...")
    te_f_idx, te_f_wave = select_test_examples_against_proto(
        Xte_f, te_f_nids, proto_wave, want="farthest"
    )

    # 4) Save 36 plots
    out_dir = os.path.join(CONFIG["out_dir"], "channels")
    print(f"Saving 36 plots to: {out_dir}")

    save_per_channel_square_plots(
        proto_wave,
        te_r_wave, te_f_wave,
        te_r_idx, te_f_idx,
        te_r_nids, te_f_nids,
        out_dir
    )

    print("Done!")



def configure_paths(argv=None):
    """Resolve portable input/output paths without changing analysis settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Directory containing metapath_data/ meta-path files (default: SFTF/model/data).")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs",
                        help="Output directory (default: outputs/ beside this script).")
    args = parser.parse_args(argv)
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    inputs = {}
    inputs["tr_r"] = str(data_dir / "metapath_data" / "Deepseek_real_train_metapaths.txt")
    inputs["te_r"] = str(data_dir / "metapath_data" / "Deepseek_real_test_metapaths.txt")
    inputs["te_f"] = str(data_dir / "metapath_data" / "Deepseek_fake_test_metapaths.txt")
    missing = [path for path in inputs.values() if not Path(path).is_file()]
    if missing:
        parser.error("Missing input files:\n" + "\n".join(missing))
    CONFIG["paths"].update(inputs)
    CONFIG["out_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    configure_paths()
    main()
