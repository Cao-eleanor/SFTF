"""
Plot three-dimensional comment lifecycle surfaces for real and fake news.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
import os
from _cli import arguments

FIGURES_DIR = None
DPI = 600
SHOW = False


def get_top_n_links(df_comments, n=50):
    """Return the top n post URLs ranked by comment-table row count."""
    return df_comments.groupby('post_url').size().nlargest(n).index.tolist()

def build_lifecycle_matrix(news_path, comments_path, top_n=50, max_hours=48, time_bins=24):
    """Aggregate comment timestamps into a normalized two-layer time matrix.

    Join comments to news by URL, select the top_n posts by comment-table
    row count, and count events in [0, max_hours) after publication.
    Normalize each layer by its own maximum bin count. Return a NumPy array
    with shape (2, time_bins). Existing rows are counted without deduplication.
    """
    print(f'Building matrix: {os.path.basename(news_path)}...')
    df_news = pd.read_excel(news_path)
    df_comments = pd.read_excel(comments_path)

    target_links = get_top_n_links(df_comments, n=top_n)
    merged = pd.merge(df_comments[df_comments['post_url'].isin(target_links)],
                      df_news[['post_url', 'post_time']],
                      left_on='post_url', right_on='post_url', how='inner')

    z_raw = np.zeros((2, time_bins))

    for _, row in merged.iterrows():
        pub_time = pd.to_datetime(row['post_time'], errors='coerce')
        if pd.isna(pub_time): continue

        # Level 1
        t1 = pd.to_datetime(row['comment_time'], errors='coerce')
        if pd.notna(t1):
            diff = (t1 - pub_time).total_seconds() / 3600
            if 0 <= diff < max_hours:
                bin_idx = int(diff / (max_hours / time_bins))
                if bin_idx < time_bins:
                    z_raw[0, bin_idx] += 1

        # Level 2
        if pd.notna(row['reply_content']):
            t2 = pd.to_datetime(row['reply_time'], errors='coerce')
            if pd.notna(t2):
                diff = (t2 - pub_time).total_seconds() / 3600
                if 0 <= diff < max_hours:
                    bin_idx = int(diff / (max_hours / time_bins))
                    if bin_idx < time_bins:
                        z_raw[1, bin_idx] += 1

    # Normalize each depth layer independently by its maximum bin count.
    z_norm = np.zeros_like(z_raw)
    if z_raw[0].max() > 0: z_norm[0] = z_raw[0] / z_raw[0].max()
    if z_raw[1].max() > 0: z_norm[1] = z_raw[1] / z_raw[1].max()

    return z_norm

def plot_optimized_3d(z_matrix, title, filename_base, cmap_style, max_hours=48, elev=35, azim=70):
    """Render a smoothed surface and save PNG versions.

    Interpolate the two observed depth layers onto 40 display positions and
    apply Gaussian smoothing with sigma=0.8. Intermediate depths are visual
    interpolation, not additional observed comment levels.
    """
    _, cols = z_matrix.shape

    # 1. Historical display grid (not aggregation-bin centers).
    # With 24 bins over 48 hours, this spacing is 48/23 hours; the actual
    # aggregation bins are two hours wide. Bin centers would be 1, 3, ..., 47.
    # Preserve the existing display coordinates for compatibility.
    x = np.linspace(0, max_hours, cols)
    y = np.array([1, 2])
    X_raw, Y_raw = np.meshgrid(x, y)

    # 2. Dense interpolation grid between the two depth layers.
    y_dense = np.linspace(1, 2, 40)
    X_dense, Y_dense = np.meshgrid(x, y_dense)

    points = np.column_stack((X_raw.flatten(), Y_raw.flatten()))
    values = z_matrix.flatten()
    Z_dense = griddata(points, values, (X_dense, Y_dense), method='linear')

    # 3. Gaussian smoothing.
    Z_final = gaussian_filter(Z_dense, sigma=0.8)

    # Figure configuration.
    fig = plt.figure(figsize=(8, 8))  # Square 8-by-8-inch figure.
    ax = fig.add_subplot(111, projection='3d')

    # Use transparent panes instead of the default gray background.
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Use light gray grid lines.
    grid_color = (0.9, 0.9, 0.9, 1)
    ax.xaxis._axinfo["grid"]['color'] = grid_color
    ax.yaxis._axinfo["grid"]['color'] = grid_color
    ax.zaxis._axinfo["grid"]['color'] = grid_color

    # Draw the interpolated surface.
    surf = ax.plot_surface(X_dense, Y_dense, Z_final, cmap=cmap_style,
                           edgecolor='none', alpha=0.95,
                           rstride=1, cstride=1, antialiased=True, shade=False)

    # Add a translucent projection below the surface.
    ax.contourf(X_dense, Y_dense, Z_final, zdir='z', offset=-0.1, cmap=cmap_style, alpha=0.4)

    # Axis labels and tick styling.
    tick_font_size = 12

    # X-axis.
    ax.set_xlabel('\nTime (Hours)', fontsize=14, labelpad=8, fontweight='bold')
    ax.tick_params(axis='x', labelsize=tick_font_size)
    ax.xaxis.label.set_fontweight('bold')  # Bold X-axis label.
    for tick in ax.get_xticklabels():
        tick.set_fontweight('bold')  # Bold X-axis tick labels.

    # Y-axis.
    ax.set_yticks([1, 2])
    ax.tick_params(axis='y', labelsize=12, pad=-1)
    ax.set_yticklabels(['Level 1', 'Level 2'],
                       verticalalignment='center',
                       horizontalalignment='right',
                       rotation=-67,
                       fontweight='bold')  # Bold Y-axis tick labels.
    ax.yaxis.label.set_fontweight('bold')  # Bold Y-axis label.

    # Adjust label spacing to reduce overlap.
    ax.set_ylabel('\nDepth Layer', fontsize=14, labelpad=8, fontweight='bold')

    # Z-axis.
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel('\nRelative Intensity',
                  fontsize=14,
                  labelpad=8,
                  rotation=93,
                  fontweight='bold')  # Bold Z-axis label.
    ax.zaxis.label.set_fontweight('bold')  # Bold Z-axis label.
    ax.tick_params(axis='z', labelsize=tick_font_size)
    ax.set_zlim(-0.1, 1.1)
    for tick in ax.get_zticklabels():
        tick.set_fontweight('bold')  # Bold Z-axis tick labels.

    # Set the viewing angles.
    ax.view_init(elev=elev, azim=azim)

    # Figure title.
    ax.set_title(title, fontsize=16, pad=1, fontweight='bold')

    # Colorbar styling.
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.01, fraction=0.035)
    cbar.ax.tick_params(labelsize=12)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight('bold')  # Bold colorbar tick labels.
    cbar.set_label('Normalized Activity Density', fontsize=13, labelpad=10, fontweight='bold')

    # Save the PNG figure.
    png_path = os.path.join(FIGURES_DIR, f"{filename_base}.png")
    print(f'Saving high-resolution PNG: {png_path}')
    plt.savefig(png_path, dpi=DPI, bbox_inches='tight')


    plt.show() if SHOW else plt.close()

def main():
    global FIGURES_DIR, DPI, SHOW
    args = arguments('Plot comment lifecycle surfaces.')
    BASE_PATH = str(args.data_dir)
    FIGURES_DIR = str(args.output_dir)
    DPI, SHOW = args.dpi, args.show
    if not SHOW:
        plt.switch_backend("Agg")
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['axes.unicode_minus'] = False
    z_real = build_lifecycle_matrix(
        os.path.join(BASE_PATH, 'real_news.xlsx'),
        os.path.join(BASE_PATH, 'real_comments.xlsx'),
        top_n=50
    )
    z_fake = build_lifecycle_matrix(
        os.path.join(BASE_PATH, 'fake_news.xlsx'),
        os.path.join(BASE_PATH, 'fake_comments.xlsx'),
        top_n=50
    )
    VIEW_ELEV = 35
    VIEW_AZIM = 70
    plot_optimized_3d(z_real,
                      "",
                      "lifecycle_real",
                      plt.cm.GnBu,
                      elev=VIEW_ELEV, azim=VIEW_AZIM)
    plot_optimized_3d(z_fake,
                      "",
                      "lifecycle_fake",
                      plt.cm.YlOrRd,
                      elev=VIEW_ELEV, azim=VIEW_AZIM)
    print('\n3D figures complete. PNG files have been generated.')


if __name__ == "__main__":
    main()
