"""
Plot aggregated comment sentiment over time for real and fake news.

"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from _cli import arguments

FIGURES_DIR = None
DPI = 600
SHOW = False
HAS_SNOWNLP = False
SnowNLP = None


def calculate_sentiment(text):
    """Return a sentiment score using SnowNLP or the original keyword fallback.

    Non-string input and SnowNLP errors return 0.5.
    The retained bare exception handler does not report failure counts and
    also catches interruptions raised during scoring (such as Ctrl+C).
    Consequently, a completed plot does not prove every score was computed. Without SnowNLP, negative
    keywords return 0.2, positive keywords return 0.8, and other text returns
    0.5. Negative keyword matches take precedence.
    """
    if not isinstance(text, str): return 0.5
    if HAS_SNOWNLP:
        try:
            return SnowNLP(text).sentiments
        # Historical fallback: catches all exceptions, including interruptions.
        # Failures become neutral scores without a warning or failure counter.
        except:
            return 0.5
    else:
        # Keyword fallback; preserve the original Chinese matching terms.
        if any(x in text for x in ['假', '谣', '骗']): return 0.2
        if any(x in text for x in ['真', '赞', '好']): return 0.8
        return 0.5

def get_top_n_links(df_comments, n=50):
    """Return the top n post URLs ranked by comment-table row count."""
    return df_comments.groupby('post_url').size().nlargest(n).index.tolist()

def collect_aggregated_data(news_path, comments_path, top_n=50):
    """Aggregate first-level and second-level sentiment observations.

    Select top_n posts by comment-table row count and join publication times
    by URL. Retain observations from 0 through 24 hours after publication,
    inclusive. Nonempty results have Time (hours), Sentiment, and Layer columns.
    Existing rows are processed without deduplication.
    """
    print(f'Aggregating data (Top {top_n}): {os.path.basename(news_path)}...')

    df_news = pd.read_excel(news_path)
    df_comments = pd.read_excel(comments_path)

    # Select the top N URLs by comment-table row count.
    target_links = get_top_n_links(df_comments, n=top_n)

    # Filter comments and news to the selected URLs.
    df_sub_comments = df_comments[df_comments['post_url'].isin(target_links)].copy()
    df_sub_news = df_news[df_news['post_url'].isin(target_links)][['post_url', 'post_time']]

    # Join comments to publication timestamps by URL.
    merged = pd.merge(df_sub_comments, df_sub_news, left_on='post_url', right_on='post_url', how='inner')

    data_points = []

    for _, row in merged.iterrows():
        pub_time = pd.to_datetime(row['post_time'], errors='coerce')
        if pd.isna(pub_time): continue

        # Process first-level comments.
        t1 = pd.to_datetime(row['comment_time'], errors='coerce')
        if pd.notna(t1):
            diff1 = (t1 - pub_time).total_seconds() / 3600
            if 0 <= diff1 <= 24:
                sent1 = calculate_sentiment(row['comment_content'])
                data_points.append({'Time': diff1, 'Sentiment': sent1, 'Layer': 'Level 1'})

        # Process second-level replies.
        if pd.notna(row['reply_content']):
            t2 = pd.to_datetime(row['reply_time'], errors='coerce')
            if pd.notna(t2):
                diff2 = (t2 - pub_time).total_seconds() / 3600
                if 0 <= diff2 <= 24:
                    sent2 = calculate_sentiment(row['reply_content'])
                    data_points.append({'Time': diff2, 'Sentiment': sent2, 'Layer': 'Level 2'})

    return pd.DataFrame(data_points)

def plot_beautiful_heatmap(df, title, filename_base, cmap_tone):
    """Plot joint time-sentiment density and sentiment distributions by layer.

    Use a filled two-dimensional KDE on the left and sentiment KDEs grouped by
    comment depth on the right. Skip empty input. Save PNG files under
    FIGURES_DIR; display the figure only with --show.
    """
    if df.empty:
        print('No data available; skipping the plot.')
        return

    # Create an 8-by-6-inch figure.
    fig, (ax_main, ax_side) = plt.subplots(1, 2, figsize=(8, 6), gridspec_kw={'width_ratios': [3, 1]})

    # Left panel: joint time-sentiment kernel density estimate.
    sns.kdeplot(
        data=df, x='Time', y='Sentiment',
        fill=True,
        cmap=cmap_tone,
        thresh=0.05,
        levels=20,
        ax=ax_main
    )

    ax_main.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax_main.set_xlim(0, 24)
    ax_main.set_ylim(0, 1)
    ax_main.set_xlabel("Time Since Publication (Hours)", fontsize=12, fontweight='bold')
    ax_main.set_ylabel("Sentiment Polarity (0=Neg, 1=Pos)", fontsize=12, fontweight='bold')

    # Use bold tick labels.
    for tick in ax_main.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax_main.get_yticklabels():
        tick.set_fontweight('bold')

    # Right panel: sentiment distributions grouped by comment depth.
    sns.kdeplot(data=df, y='Sentiment', hue='Layer', fill=True,
                ax=ax_side, palette='Set2', legend=False)
    ax_side.set_ylim(0, 1)
    ax_side.set_xlabel("Density", fontsize=12, fontweight='bold')
    ax_side.set_title("Distribution", fontsize=12, fontweight='bold')

    # Use bold tick labels.
    for tick in ax_side.get_xticklabels():
        tick.set_fontweight('bold')
    for tick in ax_side.get_yticklabels():
        tick.set_fontweight('bold')

    plt.tight_layout()

    # Save a PNG at the configured DPI (default: 600).
    save_path_png = os.path.join(FIGURES_DIR, f"{filename_base}.png")
    print(f'Saving PNG: {save_path_png}')
    plt.savefig(save_path_png, dpi=DPI, bbox_inches='tight')


    plt.show() if SHOW else plt.close()

def main():
    global FIGURES_DIR, DPI, SHOW
    args = arguments('Plot sentiment-time KDE distributions.', sentiment=True)
    BASE_PATH = str(args.data_dir)
    FIGURES_DIR = str(args.output_dir)
    DPI, SHOW = args.dpi, args.show
    if not SHOW:
        plt.switch_backend("Agg")
    REAL_COMMENTS_PATH = os.path.join(BASE_PATH, "real_comments.xlsx")
    FAKE_COMMENTS_PATH = os.path.join(BASE_PATH, "fake_comments.xlsx")
    global HAS_SNOWNLP, SnowNLP
    HAS_SNOWNLP = args.sentiment == "snownlp"
    if HAS_SNOWNLP:
        from snownlp import SnowNLP
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
    df_real = collect_aggregated_data(
        os.path.join(BASE_PATH, 'real_news.xlsx'),
        os.path.join(BASE_PATH, 'real_comments.xlsx'),
        top_n=50
    )
    plot_beautiful_heatmap(df_real, "Real News: Aggregated Sentiment Flow (Top 50)", "sentiment_real", "Blues")
    df_fake = collect_aggregated_data(
        os.path.join(BASE_PATH, 'fake_news.xlsx'),
        os.path.join(BASE_PATH, 'fake_comments.xlsx'),
        top_n=50
    )
    plot_beautiful_heatmap(df_fake, "Fake News: Aggregated Sentiment Flow (Top 50)", "sentiment_fake", "Reds")
    print('\nFinished. Figures have been saved in PNG format.')


if __name__ == "__main__":
    main()
