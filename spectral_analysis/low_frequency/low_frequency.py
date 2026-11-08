"""
Plot low-frequency energy ratios across sentiment meta-paths for real and fake news.
RQ3 spectral analysis. See README.md for inputs, outputs, and interpretation.
"""

from pathlib import Path
import argparse

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib

# =========================================================
# 0) Experiment configuration
# =========================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[2] / "model" / "data"  # SFTF/model/data

CONFIG = {
    # Input meta-path files
    "real_path": str(DATA_DIR / "metapath_data" / "Deepseek_real_test_metapaths.txt"),
    "fake_path": str(DATA_DIR / "metapath_data" / "Deepseek_fake_test_metapaths.txt"),

    # Output files
    "out_dir": str(SCRIPT_DIR / "outputs"),
    "out_prefix": "low_frequency_energy",

    # Signal and frequency parameters
    "num_channels": 36,
    "num_time_bins": 100,
    "max_time": 172800,  # 2 days
    "k_low": 12,
    "include_dc": True,  # Include the zero-frequency (DC) bin.
    "use_tail_time": True,  # Use t2 for NUV paths; otherwise use t1.
    "do_minmax": True,  # Apply per-channel min-max scaling.

    "k_select": 12,
    # Map sentiment IDs to meta-path symbols.
    "sentiment_id_to_symbol": {0: "+", 1: "0", 2: "-"},

    # Typography and style
    "font_family": "Times New Roman",
    "font_weight": "bold",
    "title_fontsize": 20,  # Title font size
    "label_fontsize": 19,  # Axis-label font size
    "tick_fontsize": 16,  # Tick-label font size
    "legend_fontsize": 12,  # Legend font size
    "dpi": 600,
}


# =========================================================
# Global typography
# =========================================================
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


# =========================================================
# 1) Load meta-paths into an (N, 36, 100) tensor
# =========================================================
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


def load_signal_tensor_from_metapath_file(
        path: str,
        num_channels: int,
        num_time_bins: int,
        max_time: float,
        use_tail_time: bool = True,
        log1p: bool = True,
):
    if not os.path.exists(path):
        raise FileNotFoundError(f"metapath file not found: {path}")

    matrices = []
    nids = []
    nid_to_idx = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("——")
            if len(parts) < 3:
                continue

            n_tokens = parts[0].split("_")
            if len(n_tokens) < 3:
                continue

            nid = n_tokens[1]
            n_sent = _safe_int(n_tokens[2])
            if n_sent is None or not (0 <= n_sent <= 2):
                continue

            t1 = _safe_float(parts[1])
            if t1 is None:
                continue

            c1_part = parts[2]
            if "None" in c1_part:
                continue
            c1_tokens = c1_part.split("_")
            if len(c1_tokens) < 3:
                continue
            c1_sent = _safe_int(c1_tokens[2])
            if c1_sent is None or not (0 <= c1_sent <= 2):
                continue

            if nid not in nid_to_idx:
                nid_to_idx[nid] = len(matrices)
                nids.append(nid)
                matrices.append(np.zeros((num_channels, num_time_bins), dtype=np.float32))
            mat = matrices[nid_to_idx[nid]]

            has_c2 = (len(parts) >= 5 and ("None" not in parts[4]))
            if has_c2:
                t2 = _safe_float(parts[3])
                c2_tokens = parts[4].split("_")
                if t2 is None or len(c2_tokens) < 3:
                    continue
                c2_sent = _safe_int(c2_tokens[2])
                if c2_sent is None or not (0 <= c2_sent <= 2):
                    continue

                t_val = t2 if use_tail_time else t1
                ch_idx = 9 + (n_sent * 9) + (c1_sent * 3) + c2_sent
            else:
                t_val = t1
                ch_idx = (n_sent * 3) + c1_sent

            if not (0 <= ch_idx < num_channels):
                continue

            t_idx = int((t_val / max_time) * num_time_bins)
            t_idx = max(0, min(num_time_bins - 1, t_idx))
            mat[ch_idx, t_idx] += 1.0

    if len(matrices) == 0:
        return np.zeros((0, num_channels, num_time_bins), dtype=np.float32), []

    X = np.stack(matrices, axis=0).astype(np.float32)
    if log1p:
        X = np.log1p(X).astype(np.float32)
    return X, nids


# =========================================================
# 2) Low-frequency Energy Ratio
# =========================================================
def low_frequency_energy_ratio(X, k_low: int, include_dc: bool = True, eps: float = 1e-12):
    F = np.fft.rfft(X, axis=-1)
    power = (F.real ** 2 + F.imag ** 2).astype(np.float64)

    max_bins = power.shape[-1]
    k_low = int(k_low)
    k_low = max(1, min(k_low, max_bins))

    start = 0 if include_dc else 1
    if start >= k_low:
        start = 0

    low_energy = power[..., start:k_low].sum(axis=-1)
    total_energy = power.sum(axis=-1)

    ratio = np.divide(low_energy, total_energy + eps)
    return np.clip(ratio, 0.0, 1.0).astype(np.float32)


# =========================================================
# 3) Cliff's delta
# =========================================================
def rankdata_average_ties(a: np.ndarray):
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    sorted_a = a[order]

    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + (j + 1))
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def cliffs_delta(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan

    combined = np.concatenate([x, y], axis=0)
    ranks = rankdata_average_ties(combined)
    rx = ranks[:nx].sum()

    U = rx - nx * (nx + 1) / 2.0
    return float(2.0 * U / (nx * ny) - 1.0)


# =========================================================
# 4) Channel index to LaTeX label
# =========================================================
def channel_to_latex_label(ch_idx: int, id2sym: dict):
    if 0 <= ch_idx <= 8:
        n = ch_idx // 3
        u = ch_idx % 3
        return rf"$P=N^{{{id2sym[n]}}}U^{{{id2sym[u]}}}$"

    if 9 <= ch_idx <= 35:
        k = ch_idx - 9
        n = k // 9
        rem = k % 9
        u = rem // 3
        v = rem % 3
        return rf"$P=N^{{{id2sym[n]}}}U^{{{id2sym[u]}}}V^{{{id2sym[v]}}}$"

    return rf"$P=\mathrm{{ch}}_{ch_idx}$"


# =========================================================
# 5) Channel selection and normalization
# =========================================================
def minmax_scale_per_channel(M: np.ndarray, eps: float = 1e-12):
    mn = M.min(axis=0, keepdims=True)
    mx = M.max(axis=0, keepdims=True)
    return ((M - mn) / (mx - mn + eps)).astype(np.float32)


def select_topk_by_cliffs_delta(metric_real: np.ndarray, metric_fake: np.ndarray, k: int):
    C = metric_real.shape[1]
    deltas = np.zeros(C, dtype=np.float64)
    for c in range(C):
        deltas[c] = cliffs_delta(metric_real[:, c], metric_fake[:, c])
    order = np.argsort(np.abs(deltas))[::-1]
    return order[:k].tolist(), deltas


# =========================================================
# 6) Radar chart
# =========================================================
def plot_radar(labels, real_vals, fake_vals, title, out_path):
    K = len(labels)
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)

    real_plot = np.r_[real_vals, real_vals[0]]
    fake_plot = np.r_[fake_vals, fake_vals[0]]
    angles_plot = np.r_[angles, angles[0]]

    # Square figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)

    ax.plot(angles_plot, real_plot, linewidth=2.2, label="Real")
    ax.fill(angles_plot, real_plot, alpha=0.18)

    ax.plot(angles_plot, fake_plot, linewidth=2.2, linestyle="--", label="Fake")
    ax.fill(angles_plot, fake_plot, alpha=0.18)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Set the radial range before placing outer labels.
    data_max = float(max(np.max(real_plot), np.max(fake_plot)))
    rmax = data_max * 1.20 if data_max > 0 else 1.0
    rmax = max(rmax, 0.10)
    ax.set_ylim(0, rmax)

    ticks = np.linspace(0, rmax, 6)[1:]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:.2f}" for t in ticks], fontsize=CONFIG["tick_fontsize"], fontweight=CONFIG["font_weight"])

    ax.set_xticks(angles)
    ax.set_xticklabels([])

    label_r = rmax * 1.16  # Outer label radius

    for lab, theta in zip(labels, angles):
        ax.text(theta, label_r, lab,
                fontsize=CONFIG["label_fontsize"], ha="center", va="center",
                fontweight=CONFIG["font_weight"],
                rotation=0, clip_on=False)

    ax.set_title(title, fontsize=CONFIG["title_fontsize"], fontweight=CONFIG["font_weight"], y=1.12)
    legend = ax.legend(loc="upper left", bbox_to_anchor=(0.95, 1.05), frameon=True, fontsize=CONFIG["legend_fontsize"])
    if legend:
        plt.setp(legend.get_texts(), fontweight=CONFIG["font_weight"])



    # Adjust the layout.
    fig.subplots_adjust(left=0.06, right=0.78, top=0.88, bottom=0.06)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Save PNG.
    png_path = out_path.replace('.png', '') + '.png'
    fig.savefig(png_path, dpi=CONFIG["dpi"], bbox_inches="tight", pad_inches=0.2)
    print(f"Saved: {png_path}")


    plt.close(fig)


# =========================================================
# 7) Run the analysis
# =========================================================
def run():
    out_dir = CONFIG["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    Xr, nids_r = load_signal_tensor_from_metapath_file(
        CONFIG["real_path"],
        CONFIG["num_channels"],
        CONFIG["num_time_bins"],
        CONFIG["max_time"],
        use_tail_time=CONFIG["use_tail_time"],
        log1p=True,
    )
    Xf, nids_f = load_signal_tensor_from_metapath_file(
        CONFIG["fake_path"],
        CONFIG["num_channels"],
        CONFIG["num_time_bins"],
        CONFIG["max_time"],
        use_tail_time=CONFIG["use_tail_time"],
        log1p=True,
    )
    print(f"[Loaded] real: {Xr.shape}, fake: {Xf.shape}")

    Mr = low_frequency_energy_ratio(Xr, k_low=CONFIG["k_low"], include_dc=CONFIG["include_dc"])
    Mf = low_frequency_energy_ratio(Xf, k_low=CONFIG["k_low"], include_dc=CONFIG["include_dc"])

    if CONFIG["do_minmax"]:
        Mall = np.concatenate([Mr, Mf], axis=0)
        Mall_s = minmax_scale_per_channel(Mall)
        Mr = Mall_s[: len(Mr)]
        Mf = Mall_s[len(Mr):]

    selected, deltas = select_topk_by_cliffs_delta(Mr, Mf, k=CONFIG["k_select"])

    id2sym = CONFIG["sentiment_id_to_symbol"]
    labels = [channel_to_latex_label(i, id2sym) for i in selected]

    mean_r = Mr.mean(axis=0)
    mean_f = Mf.mean(axis=0)

    report = []
    for i in selected:
        report.append({
            "channel": int(i),
            "label": channel_to_latex_label(i, id2sym),
            "cliffs_delta": float(deltas[i]),
            "abs_delta": float(abs(deltas[i])),
            "mean_real": float(mean_r[i]),
            "mean_fake": float(mean_f[i]),
        })

    report_path = os.path.join(out_dir, f"{CONFIG['out_prefix']}_channels.json")
    with open(report_path, "w", encoding="utf-8") as fw:
        json.dump(report, fw, ensure_ascii=False, indent=2)
    print(f"[Saved] selection report -> {report_path}")

    out_png = os.path.join(out_dir, f"{CONFIG['out_prefix']}_radar.png")
    title = "Low-frequency Energy Ratio"
    plot_radar(labels, Mr[:, selected].mean(axis=0), Mf[:, selected].mean(axis=0), title, out_png)

    print("\nTop-K selected paths:")
    for r in report:
        print(f"  ch={r['channel']:02d}  {r['label']}  delta={r['cliffs_delta']:+.4f}  "
              f"meanR={r['mean_real']:.3f} meanF={r['mean_fake']:.3f}")



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
    inputs["real_path"] = str(data_dir / "metapath_data" / "Deepseek_real_test_metapaths.txt")
    inputs["fake_path"] = str(data_dir / "metapath_data" / "Deepseek_fake_test_metapaths.txt")
    missing = [path for path in inputs.values() if not Path(path).is_file()]
    if missing:
        parser.error("Missing input files:\n" + "\n".join(missing))
    CONFIG.update(inputs)
    CONFIG["out_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    configure_paths()
    run()
