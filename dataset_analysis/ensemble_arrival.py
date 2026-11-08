"""
Plot pooled comment-arrival patterns for real and fake news ensembles.

"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from _cli import arguments

FIGURES_DIR = None
DPI = 600
SHOW = False


def load_sample_and_normalize(file_path, sample_n=500, plot_limit=100):
    """Load comments, sample posts, and compute pooled cumulative arrival ranks.

    Combine first-level comments and second-level replies when their columns
    are present. Drop rows with missing post URLs or invalid timestamps.
    Sample up to sample_n post URLs without replacement. For each post, measure
    elapsed hours from its earliest retained comment or reply. Keep elapsed
    times in [0, plot_limit], sort all selected observations together, and
    assign cumulative_ratio = (row rank starting at one) / total_count.

    Return the resulting DataFrame and its row count. Excel read failures
    raise an error. Repeated comment rows are not removed.
    """
    print(f'--- Processing file: {os.path.basename(file_path)} ---')
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        raise RuntimeError(f"Could not read workbook: {file_path}") from e

    # Extract and combine first-level comments and second-level replies.
    df_l1 = df[['post_url', 'comment_author', 'comment_time']].rename(
        columns={'comment_author': 'user', 'comment_time': 'time'}
    )
    if 'reply_author' in df.columns and 'reply_time' in df.columns:
        df_l2 = df[['post_url', 'reply_author', 'reply_time']].rename(
            columns={'reply_author': 'user', 'reply_time': 'time'}
        )
        df_all_raw = pd.concat([df_l1, df_l2], ignore_index=True)
    else:
        df_all_raw = df_l1

    df_all_raw['time'] = pd.to_datetime(df_all_raw['time'], errors='coerce')
    df_all_raw = df_all_raw.dropna(subset=['time', 'post_url'])

    # Sample post URLs without replacement.
    all_post_ids = df_all_raw['post_url'].unique()
    total_posts = len(all_post_ids)
    if total_posts <= sample_n:
        selected_ids = all_post_ids
    else:
        selected_ids = np.random.choice(all_post_ids, sample_n, replace=False)

    df_sample = df_all_raw[df_all_raw['post_url'].isin(selected_ids)].copy()

    # Align each post to its earliest retained comment or reply timestamp.
    df_sample['post_start_t'] = df_sample.groupby('post_url')['time'].transform('min')
    df_sample['hours_relative'] = (df_sample['time'] - df_sample['post_start_t']).dt.total_seconds() / 3600

    # Restrict the observation window and compute the pooled cumulative proportion.
    df_plot = df_sample[
        (df_sample['hours_relative'] >= 0) &
        (df_sample['hours_relative'] <= plot_limit)
    ].copy()

    df_plot = df_plot.sort_values('hours_relative').reset_index(drop=True)
    total_count = len(df_plot)
    if total_count > 0:
        df_plot['cumulative_ratio'] = (df_plot.index + 1) / total_count
    else:
        df_plot['cumulative_ratio'] = 0

    return df_plot, total_count

def safe_sample(df, max_points=50000):
    """Limit displayed observations without recalculating cumulative proportions.

    If more than max_points rows are available, sample with random_state=42
    and sort by relative time. Otherwise return the input unchanged.
    """
    if len(df) > max_points:
        return df.sample(n=max_points, random_state=42).sort_values('hours_relative')
    return df

def main():
    global FIGURES_DIR, DPI, SHOW
    args = arguments('Plot pooled ensemble arrivals.', sentiment=False)
    BASE_PATH = str(args.data_dir)
    FIGURES_DIR = str(args.output_dir)
    DPI, SHOW = args.dpi, args.show
    if not SHOW:
        plt.switch_backend("Agg")
    REAL_COMMENTS_PATH = os.path.join(BASE_PATH, "real_comments.xlsx")
    FAKE_COMMENTS_PATH = os.path.join(BASE_PATH, "fake_comments.xlsx")
    np.random.seed(2)
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['axes.unicode_minus'] = False
    df_real, real_total = load_sample_and_normalize(REAL_COMMENTS_PATH, sample_n=500, plot_limit=100)
    df_fake, fake_total = load_sample_and_normalize(FAKE_COMMENTS_PATH, sample_n=500, plot_limit=100)
    df_real_vis = safe_sample(df_real)
    df_fake_vis = safe_sample(df_fake)
    fig, ax = plt.subplots(figsize=(8, 8))
    if not df_real_vis.empty:
        ax.scatter(df_real_vis['hours_relative'], df_real_vis['cumulative_ratio'],
                    c='#0052CC',
                    s=2,
                    alpha=0.15,
                    label=f'Real News Ensemble (N={real_total})',
                    edgecolors='none')
    if not df_fake_vis.empty:
        ax.scatter(df_fake_vis['hours_relative'], df_fake_vis['cumulative_ratio'],
                    c='#FF6B00',
                    s=2,
                    alpha=0.15,
                    label=f'Fake News Ensemble (N={fake_total})',
                    edgecolors='none')
    ax.set_title("Aggregated Propagation Dynamics: Ensemble Arrival Patterns",
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Hours Relative to First Comment", fontsize=12, fontweight='bold')
    ax.set_ylabel("Normalized Cumulative Information Saturation", fontsize=12, fontweight='bold')
    ax.set_box_aspect(1)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle='--', alpha=0.3)
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
    legend = ax.legend(loc='lower right', frameon=True, fontsize=10, markerscale=5,
                       prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 10})
    plt.tight_layout()
    output_file_png = os.path.join(FIGURES_DIR, 'ensemble_arrival.png')
    plt.savefig(output_file_png, dpi=DPI, bbox_inches='tight')
    print(f'PNG saved: {output_file_png}')
    plt.show() if SHOW else plt.close()
    print(f'\nScatter plot generated in PNG format.')


if __name__ == "__main__":
    main()
