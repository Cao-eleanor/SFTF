"""

Plot user-arrival observations for one real-news and one fake-news case.

"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from _cli import arguments

FIGURES_DIR = None
DPI = 600
SHOW = False


def find_burst_start_time(df, buffer_hours=2):
    """Estimate the time origin from the peak hourly observation count.

    Count nonmissing user names per hour. Subtract buffer_hours from the
    start of the peak hour and clamp the result to the earliest observation.
    Return None for empty input, or the earliest timestamp if no hourly bins
    are available. This origin need not equal the first comment timestamp.
    """
    if df.empty:
        return None
    temp_df = df.set_index('time')
    # Count observations with nonmissing user names in hourly bins.
    hourly_counts = temp_df.resample('1h')['user'].count()
    if hourly_counts.empty:
        return df['time'].min()

    peak_time = hourly_counts.idxmax()
    start_time = peak_time - pd.Timedelta(hours=buffer_hours)

    real_min = df['time'].min()
    return max(start_time, real_min)

def load_lifecycle_data_merged(file_path, label_name):
    """Select the most active post and prepare its user-arrival coordinates.

    Combine both comment levels when reply columns exist, discard invalid
    timestamps and missing post URLs, and select the URL with the most rows.
    Align timestamps to the burst-based origin and keep [-1, 48] hours.
    Assign zero-based user_y ranks by each user name's first observation in
    that window. Return the plotted rows, or an empty DataFrame when no valid observations
    remain before case selection. Workbook read failures raise an error.
    Repeated rows are not deduplicated; user names are not platform user IDs.
    """
    print(f'--- Processing: {label_name} ({os.path.basename(file_path)}) ---')
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        raise RuntimeError(f"Could not read workbook: {file_path}") from e

    # Extract first-level comments.
    df_l1 = df[['post_url', 'comment_author', 'comment_time']].rename(
        columns={'comment_author': 'user', 'comment_time': 'time'}
    )

    # Extract second-level replies when their columns exist.
    if 'reply_author' in df.columns and 'reply_time' in df.columns:
        df_l2 = df[['post_url', 'reply_author', 'reply_time']].rename(
            columns={'reply_author': 'user', 'reply_time': 'time'}
        )
        df_all = pd.concat([df_l1, df_l2], ignore_index=True)
    else:
        df_all = df_l1

    # Parse timestamps and remove missing timestamps or post URLs.
    df_all['time'] = pd.to_datetime(df_all['time'], errors='coerce')
    df_all = df_all.dropna(subset=['time', 'post_url'])

    if df_all.empty:
        return pd.DataFrame()

    # Select the post with the largest number of retained comment/reply rows.
    top_post = df_all['post_url'].value_counts().idxmax()
    df_case = df_all[df_all['post_url'] == top_post].copy()
    df_case = df_case.sort_values('time')

    # Align to the estimated burst start.
    start_time = find_burst_start_time(df_case, buffer_hours=2)
    # Compute elapsed hours from the selected origin.
    df_case['hours_relative'] = (df_case['time'] - start_time).dt.total_seconds() / 3600

    # Restrict observations to the inclusive [-1, 48] hour window.
    df_plot = df_case[(df_case['hours_relative'] >= -1) & (df_case['hours_relative'] <= 48)].copy()

    # Rank user names by their earliest observation within the plotted window.
    user_first_seen = df_plot.groupby('user')['hours_relative'].min().sort_values()
    user_map = {u: i for i, u in enumerate(user_first_seen.index)}
    df_plot['user_y'] = df_plot['user'].map(user_map)

    return df_plot

def main():
    global FIGURES_DIR, DPI, SHOW
    args = arguments('Plot individual-case user arrivals.', sentiment=False)
    BASE_PATH = str(args.data_dir)
    FIGURES_DIR = str(args.output_dir)
    DPI, SHOW = args.dpi, args.show
    if not SHOW:
        plt.switch_backend("Agg")
    REAL_COMMENTS_PATH = os.path.join(BASE_PATH, "real_comments.xlsx")
    FAKE_COMMENTS_PATH = os.path.join(BASE_PATH, "fake_comments.xlsx")
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
    df_real = load_lifecycle_data_merged(REAL_COMMENTS_PATH, "Real News")
    df_fake = load_lifecycle_data_merged(FAKE_COMMENTS_PATH, "Fake News")
    fig, ax = plt.subplots(figsize=(8, 8))
    if not df_real.empty:
        ax.scatter(df_real['hours_relative'], df_real['user_y'],
                   c='#0052CC', s=8, alpha=0.5,
                   label=f'Real News (Total Actions: {len(df_real)})', edgecolors='none')
    if not df_fake.empty:
        ax.scatter(df_fake['hours_relative'], df_fake['user_y'],
                   c='#FF6B00', s=8, alpha=0.5,
                   label=f'Fake News (Total Actions: {len(df_fake)})', edgecolors='none')
    ax.set_title("Individual Case Study: User Arrival Patterns", fontsize=14, fontweight='bold', pad=15)
    # Historical label retained; the computed origin is burst-based, as above.
    ax.set_xlabel("Hours Relative to First Comment", fontsize=12, fontweight='bold')
    ax.set_ylabel("Cumulative Entry Sequence of Unique Users", fontsize=12, fontweight='bold')
    ax.set_box_aspect(1)
    ax.set_xlim(-1, 49)
    ax.grid(True, linestyle='--', alpha=0.3)
    legend = ax.legend(loc='lower right', frameon=True, fontsize=10, framealpha=0.9,
                       prop={'family': 'Times New Roman', 'weight': 'bold', 'size': 10})
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax.get_yticklabels():
        tick.set_fontweight('bold')
    plt.tight_layout()
    output_file_png = os.path.join(FIGURES_DIR, 'case_arrival.png')
    plt.savefig(output_file_png, dpi=DPI, bbox_inches='tight')
    print(f'PNG saved: {output_file_png}')
    plt.show() if SHOW else plt.close()
    print(f'\nScatter plot generated in PNG format.')


if __name__ == "__main__":
    main()
