"""
Plot F1 and representation separability across loss weights and deviation margins.
"""
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def plot_lines_equalspace_noerr(
    df: pd.DataFrame,
    y_col: str,
    title: str,
    y_label: str,
    out_png: str,
    beta_min: float = 0.0,
    beta_max: float = 4.0,
    beta_step: float = 0.5,
    show: bool = True,
    dpi: int = 600,
):
    set_paper_style()

    need = ["beta", "delta", y_col]
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns: {miss}; available columns: {list(df.columns)}")

    df = df.copy()
    df["beta"] = pd.to_numeric(df["beta"], errors="coerce")
    df["delta"] = pd.to_numeric(df["delta"], errors="coerce")
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df = df.dropna(subset=["beta", "delta", y_col]).copy()

    betas = np.arange(beta_min, beta_max + 1e-12, beta_step, dtype=float)
    betas = np.round(betas, 10)

    df["beta"] = np.round(df["beta"].to_numpy(dtype=float), 10)
    df = df[(df["beta"] >= betas.min() - 1e-12) & (df["beta"] <= betas.max() + 1e-12)].copy()

    deltas = np.array(sorted(df["delta"].unique()), dtype=float)

    x_pos = np.arange(len(betas), dtype=int)
    beta_to_x = {float(b): int(i) for i, b in enumerate(betas)}

    # Create a square canvas.
    fig, ax = plt.subplots(figsize=(6.0, 6.0))

    for d in deltas:
        sub = df[df["delta"] == d].copy()
        g = sub.groupby("beta", as_index=False)[y_col].mean()
        lookup = {float(b): float(v) for b, v in zip(g["beta"].to_numpy(), g[y_col].to_numpy())}

        y = np.array([lookup.get(float(b), np.nan) for b in betas], dtype=float)
        mask = ~np.isnan(y)
        if mask.sum() == 0:
            continue

        x = np.array([beta_to_x[float(b)] for b in betas], dtype=int)
        ax.plot(
            x[mask], y[mask],
            marker="o",
            linewidth=2.2,
            markersize=5.5,
            label=fr"$\Delta={d:g}$"
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(r"Loss Weight $\beta$", fontsize=15, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=15, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{b:g}" for b in betas])  # Set tick labels.

    # Use bold axis tick labels.
    ax.tick_params(axis="both", labelsize=12)
    plt.setp(ax.get_xticklabels(), fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")

    ax.grid(True, linestyle=":", linewidth=0.9, alpha=0.35)

    # Use bold legend text.
    leg = ax.legend(loc="best", frameon=True, fancybox=True, framealpha=0.9, ncol=2, fontsize=11)
    plt.setp(leg.get_texts(), fontweight="bold")

    # Use a square plotting area.
    ax.set_box_aspect(1)

    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)

    print(f"[Saved] {out_png}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'outputs')
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--input-csv', type=Path, default=Path(__file__).resolve().parent / 'outputs' / 'beta_delta_F1.csv')
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error('--dpi must be positive')
    if not args.show:
        plt.switch_backend('Agg')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.input_csv.is_file():
        parser.error(f'Missing input CSV: {args.input_csv}')
    csv_path = args.input_csv
    out_dir = str(args.output_dir)
    df = pd.read_csv(csv_path)

    plot_lines_equalspace_noerr(
        df=df,
        y_col="mean_val_f1",
        title=r"Performance Across Loss Weight $\beta$ and Threshold $\Delta$ Settings",
        y_label="Mean Validation F1 Score",
        out_png=os.path.join(out_dir, "manifold_performance.png"),
        beta_min=0.0,
        beta_max=5.0,
        beta_step=0.5,
        show=args.show,
        dpi=args.dpi,
    )

    plot_lines_equalspace_noerr(
        df=df,
        y_col="mean_gap_dfake_dreal",
        title=r"Distance Gap Between Fake and Real News Across $\beta$ and $\Delta$",
        y_label=r"Mean Distance Gap Between Fake and Real News",
        out_png=os.path.join(out_dir, "manifold_separability.png"),
        beta_min=0.0,
        beta_max=5.0,
        beta_step=0.5,
        show=args.show,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
