"""
Compare low-, mid-, and high-frequency energy distributions across sentiment meta-paths.
RQ3 spectral analysis. See README.md for inputs, outputs, and interpretation.
"""

from pathlib import Path
import argparse

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import matplotlib

# -------------------------
# 0) Experiment configuration
# -------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[2] / "model" / "data"  # SFTF/model/data

CONFIG = {
    "paths": {
        "te_r": str(DATA_DIR / "metapath_data" / "Deepseek_real_test_metapaths.txt"),
        "te_f": str(DATA_DIR / "metapath_data" / "Deepseek_fake_test_metapaths.txt"),
    },
    "num_channels": 36,
    "num_time_bins": 100,
    "max_time": 172800,
    "out_dir": str(SCRIPT_DIR / "outputs"),

    "active_power_threshold": 1e-6,

    "min_active_samples_per_class": 20,

    "TOPK_PER_ROW": 4,

    # Typography and style
    "font_family": "Times New Roman",
    "font_weight": "bold",
    "title_fontsize": 14,
    "label_fontsize": 12,
    "tick_fontsize": 11,
    "legend_fontsize": 11,

    # Class colors
    "colors": {
        "real": "#1f77b4",  # Blue
        "fake": "#ff7f0e",  # Orange
    },

    "dpi": 600,
}



# -------------------------
# Global typography
# -------------------------
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


# -------------------------
# 1) Parse meta-path data
# -------------------------
def build_signal_matrix(lines):
    matrix = np.zeros((CONFIG["num_channels"], CONFIG["num_time_bins"]), dtype=np.float32)
    for line in lines:
        parts = line.split("——")
        try:
            n_sent = int(parts[0].split("_")[2])  # 0/1/2
            c1_sent = int(parts[2].split("_")[2])  # 0/1/2
            t_val = float(parts[1])
            t_idx = min(int((t_val / CONFIG["max_time"]) * CONFIG["num_time_bins"]),
                        CONFIG["num_time_bins"] - 1)

            if len(parts) >= 5 and "None" not in parts[4]:
                c2_sent = int(parts[4].split("_")[2])  # 0/1/2
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
            nid = parts[0].split("_")[1]
            if nid not in groups:
                groups[nid] = []
                order.append(nid)
            groups[nid].append(line)
    return groups, order


def load_signals(metapath_path):
    groups, order = parse_metapath_file(metapath_path)
    X = np.stack([build_signal_matrix(groups[nid]) for nid in order], axis=0)
    return X, order


# -------------------------
# 2) Frequency-band energy ratios and total power
# -------------------------
def compute_band_energy_ratios_and_total_power(X):
    N, C, M = X.shape
    kmax = M // 2

    low_end = int(0.1 * M)
    mid_end = int(0.3 * M)
    low_end = max(1, min(low_end, kmax))
    mid_end = max(low_end + 1, min(mid_end, kmax))

    fft = np.fft.rfft(X, axis=-1)
    power = (fft.real ** 2 + fft.imag ** 2)

    eps = 1e-12
    total_power = power[:, :, 1:kmax + 1].sum(axis=-1)  # exclude DC

    low = power[:, :, 1:low_end + 1].sum(axis=-1)
    mid = power[:, :, low_end + 1:mid_end + 1].sum(axis=-1)
    high = power[:, :, mid_end + 1:kmax + 1].sum(axis=-1)

    ratios = np.zeros((N, C, 3), dtype=np.float32)
    ratios[:, :, 0] = low / (total_power + eps)
    ratios[:, :, 1] = mid / (total_power + eps)
    ratios[:, :, 2] = high / (total_power + eps)

    band_info = {
        "low": (1, low_end),
        "mid": (low_end + 1, mid_end),
        "high": (mid_end + 1, kmax),
        "M": M
    }
    return ratios, total_power.astype(np.float32), band_info


# -------------------------
# 3) Channel index to meta-path label
# -------------------------
SENT_SUP = {0: "+", 1: "0", 2: "-"}


def channel_to_p_label(ch):
    """
    Return the meta-path label as a mathtext string.
    NU:  P = N^{a}U^{b}
    NUV: P = N^{a}U^{b}V^{c}
    """
    if ch < 9:
        n = ch // 3
        u = ch % 3
        sn, su = SENT_SUP[n], SENT_SUP[u]
        return rf"$P = N^{{{sn}}}U^{{{su}}}$"
    else:
        idx = ch - 9
        n = idx // 9
        rem = idx % 9
        u = rem // 3
        v = rem % 3
        sn, su, sv = SENT_SUP[n], SENT_SUP[u], SENT_SUP[v]
        return rf"$P = N^{{{sn}}}U^{{{su}}}V^{{{sv}}}$"


# -------------------------
# 4) Build the long-format table
# -------------------------
def make_long_df(ratios, total_power, label_name):
    N, C, _ = ratios.shape
    bands = ["low", "mid", "high"]
    thr = CONFIG["active_power_threshold"]

    rows = []
    for i in range(N):
        for ch in range(C):
            tp = float(total_power[i, ch])
            active = 1 if tp > thr else 0
            for bidx, b in enumerate(bands):
                rows.append({
                    "sample": i,
                    "label": label_name,
                    "channel": ch,
                    "p": channel_to_p_label(ch),
                    "band": b,
                    "ratio": float(ratios[i, ch, bidx]),
                    "total_power": tp,
                    "active": active
                })
    return pd.DataFrame(rows)


# -------------------------
# 5) Cliff's delta
# -------------------------
def cliffs_delta(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan

    all_vals = np.concatenate([x, y])
    ranks = pd.Series(all_vals).rank(method="average").to_numpy()
    r1 = ranks[:n1].sum()
    U1 = r1 - n1 * (n1 + 1) / 2.0
    delta = (2.0 * U1) / (n1 * n2) - 1.0
    return float(delta)


# -------------------------
# 6) Rank channels by maximum absolute Cliff's delta over active samples
# -------------------------
def summarize_channels_active(df_all, channel_range):
    bands = ["low", "mid", "high"]
    min_n = CONFIG["min_active_samples_per_class"]

    # Use active samples to estimate class differences.
    df_active = df_all[df_all["active"] == 1].copy()

    records = []
    for ch in channel_range:
        # Count unique active samples per class.
        n_real_active = df_active[(df_active["channel"] == ch) & (df_active["label"] == "real")]["sample"].nunique()
        n_fake_active = df_active[(df_active["channel"] == ch) & (df_active["label"] == "fake")]["sample"].nunique()

        for band in bands:
            d = df_active[(df_active["channel"] == ch) & (df_active["band"] == band)]
            real = d[d["label"] == "real"]["ratio"].values
            fake = d[d["label"] == "fake"]["ratio"].values

            # Skip effect-size estimation when either class has too few active samples.
            if n_real_active < min_n or n_fake_active < min_n:
                delta = np.nan
                abs_delta = np.nan
            else:
                delta = cliffs_delta(fake, real)
                abs_delta = abs(delta) if not np.isnan(delta) else np.nan

            records.append({
                "channel": ch,
                "band": band,
                "delta(fake-real)": delta,
                "abs_delta": abs_delta,
                "n_real_active": n_real_active,
                "n_fake_active": n_fake_active
            })

    stat_df = pd.DataFrame(records)

    # Keep the band with the largest absolute effect size for each channel.
    best = (stat_df.sort_values(["channel", "abs_delta"], ascending=[True, False])
            .groupby("channel", as_index=False)
            .first())

    # Sort by absolute effect size in descending order.
    best = best.sort_values("abs_delta", ascending=False).copy()
    return best, stat_df


# -------------------------
# 7) Plot individual channels
# -------------------------
def plot_violin_per_channel_active(df_all, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Set the Seaborn style.
    sns.set(style="whitegrid")

    # Define the class palette.
    palette = {
        "real": CONFIG["colors"]["real"],
        "fake": CONFIG["colors"]["fake"]
    }

    df = df_all[df_all["active"] == 1].copy()
    df["band"] = pd.Categorical(df["band"], categories=["low", "mid", "high"], ordered=True)
    df["label"] = pd.Categorical(df["label"], categories=["real", "fake"], ordered=True)

    for ch in range(CONFIG["num_channels"]):
        d = df[df["channel"] == ch]
        if d.empty:
            # Skip channels without active samples.
            continue

        # Create the figure.
        plt.figure(figsize=(6.8, 4.6))

        # Draw the violin plot.
        ax = sns.violinplot(
            data=d, x="band", y="ratio", hue="label",
            split=True, inner="quartile", cut=0, linewidth=1.5,
            palette=palette
        )

        # Set the title and axis labels.
        ax.set_title(channel_to_p_label(ch), fontsize=CONFIG["title_fontsize"], fontweight=CONFIG["font_weight"])
        ax.set_xlabel("Frequency Band", fontsize=CONFIG["label_fontsize"], fontweight=CONFIG["font_weight"])
        ax.set_ylabel("Energy Ratio", fontsize=CONFIG["label_fontsize"], fontweight=CONFIG["font_weight"])

        # Set the vertical range.
        ax.set_ylim(0, 1)

        # Set tick typography.
        ax.tick_params(axis='both', which='major', labelsize=CONFIG["tick_fontsize"])

        # Configure the legend.
        legend = ax.legend(title="", loc="upper right", fontsize=CONFIG["legend_fontsize"])
        if legend:
            plt.setp(legend.get_texts(), fontweight=CONFIG["font_weight"])

        # Configure the grid.
        ax.grid(True, linestyle='--', alpha=0.7)

        # Adjust the layout.
        plt.tight_layout()

        # Save PNG.
        png_path = os.path.join(out_dir, f"band_energy_ch{ch:02d}.png")
        plt.savefig(png_path, dpi=CONFIG["dpi"], bbox_inches='tight', transparent=False)


        plt.close()


# -------------------------
# 8) Plot the 2-by-4 overview
# -------------------------
def plot_top4_NU_and_top4_NUV_2x4_active(df_all, nu_best, nuv_best, out_path):
    # Set the Seaborn style.
    sns.set(style="whitegrid")

    # Define the class palette.
    palette = {
        "real": CONFIG["colors"]["real"],
        "fake": CONFIG["colors"]["fake"]
    }

    df = df_all[df_all["active"] == 1].copy()
    df["band"] = pd.Categorical(df["band"], categories=["low", "mid", "high"], ordered=True)
    df["label"] = pd.Categorical(df["label"], categories=["real", "fake"], ordered=True)

    topk = CONFIG["TOPK_PER_ROW"]
    nu_top = nu_best.head(topk).copy()
    nuv_top = nuv_best.head(topk).copy()

    nu_channels = nu_top["channel"].tolist()
    nuv_channels = nuv_top["channel"].tolist()

    # Create the figure.
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5), sharey=True)
    axes = axes.reshape(2, 4)

    def draw(ax, ch):
        d = df[df["channel"] == ch]
        if d.empty:
            return

        sns.violinplot(
            data=d, x="band", y="ratio", hue="label",
            split=True, inner="quartile", cut=0, linewidth=1.5,
            palette=palette, ax=ax
        )

        # Set the title.
        ax.set_title(channel_to_p_label(ch), fontsize=CONFIG["title_fontsize"], fontweight=CONFIG["font_weight"])

        # Set axis labels.
        ax.set_xlabel("", fontsize=CONFIG["label_fontsize"], fontweight=CONFIG["font_weight"])
        ax.set_ylabel("Energy Ratio", fontsize=CONFIG["label_fontsize"], fontweight=CONFIG["font_weight"])

        # Set the vertical range.
        ax.set_ylim(0, 1)

        # Set tick typography.
        ax.tick_params(axis='both', which='major', labelsize=CONFIG["tick_fontsize"])

        # Remove local legends; use a shared figure legend.
        if ax.get_legend() is not None:
            ax.get_legend().remove()

        # Configure the grid.
        ax.grid(True, linestyle='--', alpha=0.7)

    # First row: NU channels
    for j, ch in enumerate(nu_channels):
        draw(axes[0, j], ch)

    # Second row: NUV channels
    for j, ch in enumerate(nuv_channels):
        draw(axes[1, j], ch)

    # Add a shared legend.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True,
               fontsize=CONFIG["legend_fontsize"], prop={'weight': CONFIG["font_weight"]})

    # Adjust the layout.
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # Save PNG.
    png_path = out_path.replace('.png', '') + '.png'
    plt.savefig(png_path, dpi=CONFIG["dpi"], bbox_inches='tight', transparent=False)


    plt.close()


# -------------------------
# main
# -------------------------
def main():
    # Create the output directory.
    os.makedirs(CONFIG["out_dir"], exist_ok=True)

    # Load input signals.
    Xr, _ = load_signals(CONFIG["paths"]["te_r"])
    Xf, _ = load_signals(CONFIG["paths"]["te_f"])
    print("Loaded:", Xr.shape, Xf.shape)

    # Compute frequency-band energies.
    rr, tr_power, band_info = compute_band_energy_ratios_and_total_power(Xr)
    ff, tf_power, _ = compute_band_energy_ratios_and_total_power(Xf)
    print("Band definition:", band_info)
    print("Active threshold:", CONFIG["active_power_threshold"])

    # Build the data frames.
    dfr = make_long_df(rr, tr_power, "real")
    dff = make_long_df(ff, tf_power, "fake")
    df_all = pd.concat([dfr, dff], ignore_index=True)

    # Save the long-format table.
    csv_path = os.path.join(CONFIG["out_dir"], "band_energy_samples.csv")
    df_all.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved long table to: {csv_path}")

    # Rank NU and NUV channels by class differences.
    nu_best, nu_stat = summarize_channels_active(df_all, range(0, 9))
    nuv_best, nuv_stat = summarize_channels_active(df_all, range(9, 36))

    # Save channel rankings.
    nu_best_path = os.path.join(CONFIG["out_dir"], "band_energy_nu_ranking.csv")
    nu_best.to_csv(nu_best_path, index=False, encoding="utf-8-sig")
    print(f"Saved NU best channels to: {nu_best_path}")

    nuv_best_path = os.path.join(CONFIG["out_dir"], "band_energy_nuv_ranking.csv")
    nuv_best.to_csv(nuv_best_path, index=False, encoding="utf-8-sig")
    print(f"Saved NUV best channels to: {nuv_best_path}")

    # Plot the 2-by-4 overview.
    out_grid_base = os.path.join(CONFIG["out_dir"], "band_energy_distribution")
    plot_top4_NU_and_top4_NUV_2x4_active(df_all, nu_best, nuv_best, out_grid_base)
    print(f"Saved 2x4 violin plots to: {out_grid_base}.png")

    # Plot individual channels.
    out_single = os.path.join(CONFIG["out_dir"], "channels")
    plot_violin_per_channel_active(df_all, out_single)
    print(f"Saved individual violin plots to: {out_single}")

    print("Done. All outputs saved to:", CONFIG["out_dir"])



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
