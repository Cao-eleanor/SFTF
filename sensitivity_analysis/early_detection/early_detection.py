"""
Plot the sensitivity of early-detection F1 to observation window M and temporal decay lambda from a metrics CSV.
"""
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle


def plot_heatmap_from_csv(
    csv_path=Path(__file__).resolve().parent / "outputs" / "early_detection_M_lambda_grid.csv",
    out_name="early_detection_sensitivity",
    out_dir=Path(__file__).resolve().parent / "outputs",
    dpi=600,
    annotate=True,
    highlight_best=False,
    show=False
):
    # Read the input CSV.
    df = pd.read_csv(csv_path)

    pivot = df.pivot(index="lambda_norm", columns="M_hours", values="mean_val_f1_at_best_auc")
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)

    lambdas = pivot.index.to_list()
    Ms = pivot.columns.to_list()
    Z = pivot.values.astype(float)

    z_valid = Z[np.isfinite(Z)]
    if len(z_valid) == 0:
        raise ValueError("No finite values in the CSV; cannot plot the heatmap.")

    # tighten contrast
    vmin = np.percentile(z_valid, 5)
    vmax = np.percentile(z_valid, 95)
    if vmax <= vmin:
        vmin = float(np.min(z_valid))
        vmax = float(np.max(z_valid))

    # ===== colormap =====
    fresh_colors = ["#f7fcf0", "#d9f0d3", "#a8ddb5", "#7bccc4", "#4eb3d3", "#2b8cbe"]
    cmap = LinearSegmentedColormap.from_list("fresh_mint_blue", fresh_colors, N=256)

    # Global style: bold Times New Roman.
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size": 20,
        "font.weight": "bold",

        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "axes.labelweight": "bold",

        "xtick.labelsize": 12,
        "ytick.labelsize": 12,

        "legend.fontsize": 11,
    })

    # Create the figure with the original dimensions.
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(
        Z, aspect="auto", origin="lower",
        cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="nearest"
    )

    # Title and axis labels.
    ax.set_title(r"Early Detection Sensitivity: Interaction of $M$ and $\lambda$ (F1 Score)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel(r"Observation Window $M$ (hours)", fontsize=13, fontweight="bold")
    ax.set_ylabel(r"Decay Strength $\lambda$", fontsize=13, fontweight="bold")

    # Tick labels.
    ax.set_xticks(np.arange(len(Ms)))
    ax.set_xticklabels([str(m) for m in Ms], fontweight="bold")

    ax.set_yticks(np.arange(len(lambdas)))
    ax.set_yticklabels([f"{x:g}" for x in lambdas], fontweight="bold")

    # Use bold tick labels.
    plt.setp(ax.get_xticklabels(), fontweight="bold")
    plt.setp(ax.get_yticklabels(), fontweight="bold")

    # Light grid lines.
    ax.set_xticks(np.arange(-.5, len(Ms), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(lambdas), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate finite cell values.
    if annotate:
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                val = Z[i, j]
                if not np.isfinite(val):
                    continue
                ax.text(
                    j, i, f"{val:.4f}",
                    ha="center", va="center",
                    color="black", fontsize=12)

    # Colorbar with bold label and tick text.
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Validation F1 Score", fontsize=12, fontweight="bold")
    cbar.ax.tick_params(labelsize=11)
    plt.setp(cbar.ax.get_yticklabels(), fontweight="bold")

    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, out_name + ".png")

    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.show() if show else plt.close(fig)

    print(f"Saved: {out_png}")



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'outputs')
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--input-csv', type=Path, default=Path(__file__).resolve().parent / 'outputs' / 'early_detection_M_lambda_grid.csv')
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error('--dpi must be positive')
    if not args.show:
        plt.switch_backend('Agg')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.input_csv.is_file():
        parser.error(f'Missing input CSV: {args.input_csv}')
    plot_heatmap_from_csv(csv_path=args.input_csv, out_dir=args.output_dir,
                          dpi=args.dpi, annotate=True, highlight_best=True, show=args.show)


if __name__ == '__main__':
    main()
