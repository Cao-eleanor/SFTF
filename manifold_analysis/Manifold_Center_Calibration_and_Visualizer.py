"""
Plot the t-SNE embeddings and distance distributions of real and fake news 
relative to the real-news test-embedding center.
"""

import os
from _input_paths import default_inputs, configure_inputs
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from matplotlib import rcParams


# ==========================================
# 0. Font configuration
# ==========================================
def setup_fonts():
    """Configure bold Times New Roman typography globally."""
    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.weight'] = 'bold'
    rcParams['axes.labelweight'] = 'bold'
    rcParams['axes.titleweight'] = 'bold'
    rcParams['figure.titlesize'] = 16
    rcParams['figure.titleweight'] = 'bold'
    rcParams['axes.labelsize'] = 14
    rcParams['axes.titlesize'] = 16
    rcParams['xtick.labelsize'] = 12
    rcParams['ytick.labelsize'] = 12
    rcParams['legend.fontsize'] = 12
    rcParams['legend.title_fontsize'] = 12
    rcParams['pdf.fonttype'] = 42  # Embed TrueType fonts in PDF output.
    rcParams['ps.fonttype'] = 42  # Embed TrueType fonts in PostScript output.

    # Configure mathematical text fonts.
    rcParams['mathtext.fontset'] = 'custom'
    rcParams['mathtext.it'] = 'Times New Roman:italic'
    rcParams['mathtext.rm'] = 'Times New Roman'
    rcParams['mathtext.tt'] = 'Times New Roman'
    rcParams['mathtext.bf'] = 'Times New Roman:bold'
    rcParams['mathtext.cal'] = 'Times New Roman'
    rcParams['mathtext.sf'] = 'Times New Roman'


# Apply the font configuration.
setup_fonts()

# ==========================================
# 1. Basic configuration
# ==========================================
CONFIG = {
    **default_inputs(include_train=False),
    'num_channels': 36,
    'num_time_bins': 100,
    'max_time': 172800,
    'k_selected': 12,
    'embed_dim': 64,
    'metric_dim': 128,
    'bert_dim': 768,
    'batch_size': 32,
    'bert_weight': 0.2,
    'dpi': 600,  # Configure the export resolution in DPI.

    # Configure class colors.
    'colors': {
        'real': '#1f77b4',  # Blue.
        'fake': '#ff7f0e',  # Orange.
    }
}


# ==========================================
# 2. Data processing function definitions
# ==========================================
def build_signal_matrix(lines):
    matrix = np.zeros((CONFIG['num_channels'], CONFIG['num_time_bins']), dtype=np.float32)
    for line in lines:
        parts = line.split('——')
        try:
            n_sent = int(parts[0].split('_')[2])
            c1_sent = int(parts[2].split('_')[2])
            t_val = float(parts[1])
            t_idx = min(int((t_val / CONFIG['max_time']) * CONFIG['num_time_bins']), CONFIG['num_time_bins'] - 1)
            if len(parts) >= 5 and 'None' not in parts[4]:
                c2_sent = int(parts[4].split('_')[2])
                ch_idx = 9 + (n_sent * 9) + (c1_sent * 3) + c2_sent
            else:
                ch_idx = n_sent * 3 + c1_sent
            if ch_idx < CONFIG['num_channels']:
                matrix[ch_idx, t_idx] += 1
        except:
            continue
    return np.log1p(matrix).astype(np.float32)


def parse_metapath_file(path):
    groups, order = {}, []
    if not os.path.exists(path): return groups, order
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            nid = line.split('——')[0].split('_')[1]
            if nid not in groups:
                groups[nid] = []
                order.append(nid)
            groups[nid].append(line)
    return groups, order


def load_test_data():
    te_r_groups, te_r_order = parse_metapath_file(CONFIG['paths']['te_r'])
    te_f_groups, te_f_order = parse_metapath_file(CONFIG['paths']['te_f'])
    te_r_bert = np.load(CONFIG['bert_paths']['te_r']).astype(np.float32)
    te_f_bert = np.load(CONFIG['bert_paths']['te_f']).astype(np.float32)

    te_r_f = np.array([build_signal_matrix(te_r_groups[nid]) for nid in te_r_order])
    te_f_f = np.array([build_signal_matrix(te_f_groups[nid]) for nid in te_f_order])

    xf = np.concatenate([te_r_f, te_f_f], axis=0)
    xb = np.concatenate([te_r_bert, te_f_bert], axis=0)
    y = np.array([0] * len(te_r_f) + [1] * len(te_f_f))
    return xf, xb, y


class HybridDataset(Dataset):
    def __init__(self, f, b, l):
        self.f, self.b, self.l = f, b, l

    def __len__(self): return len(self.l)

    def __getitem__(self, i):
        return torch.tensor(self.f[i]), torch.tensor(self.b[i]), torch.tensor(self.l[i])


# ==========================================
# 3. Model architecture definitions
# ==========================================
class MetaTrendAttentionEncoder(nn.Module):
    def __init__(self, k_selected, embed_dim):
        super().__init__()
        self.path_projection = nn.Linear(k_selected * 2, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        selected_fft = torch.fft.rfft(x, dim=-1)[:, :, :CONFIG['k_selected']]
        meta_trend = torch.cat([selected_fft.real, selected_fft.imag], dim=-1)
        tokens = self.path_projection(meta_trend)
        attn, _ = self.attention(tokens, tokens, tokens)
        return torch.mean(self.norm(attn + tokens), dim=1)


class ManifoldFakeNewsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.struct_encoder = MetaTrendAttentionEncoder(CONFIG['k_selected'], CONFIG['embed_dim'])
        self.bert_fc = nn.Sequential(nn.Linear(CONFIG['bert_dim'], 256), nn.BatchNorm1d(256), nn.ReLU(),
                                     nn.Linear(256, CONFIG['embed_dim']))
        self.phi = nn.Sequential(nn.Linear(CONFIG['embed_dim'] * 2, 256), nn.ReLU(),
                                 nn.Linear(256, CONFIG['metric_dim']))
        self.center = nn.Parameter(torch.randn(1, CONFIG['metric_dim']))
        self.tau = nn.Parameter(torch.tensor([1.0]))

    def forward(self, x_f, x_b):
        h_t = self.struct_encoder(x_f)
        h_s = self.bert_fc(x_b) * CONFIG['bert_weight']
        phi_h = self.phi(torch.cat([h_t, h_s], dim=1))
        d_i = torch.norm(phi_h - self.center, p=2, dim=1)
        return phi_h, d_i


def main():
    configure_inputs(CONFIG)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 1. Load the model and data.
    model = ManifoldFakeNewsNet().to(device)
    model_path = CONFIG['model_ckpt']
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    xf, xb, y = load_test_data()
    loader = DataLoader(HybridDataset(xf, xb, y), batch_size=CONFIG['batch_size'])

    all_phi, all_labels = [], []
    with torch.no_grad():
        for f, b, l in loader:
            phi_h, _ = model(f.to(device), b.to(device))
            all_phi.append(phi_h.cpu().numpy())
            all_labels.append(l.numpy())

    phi_h = np.concatenate(all_phi)
    labels = np.concatenate(all_labels)

    center = model.center.detach().cpu().numpy()
    tau = model.tau.item()

    real_cluster_center = np.mean(phi_h[labels == 0], axis=0, keepdims=True)

    dists_to_real_center = np.linalg.norm(phi_h - real_cluster_center, axis=1)

    # 1. KDE plot
    plt.figure(figsize=(8, 8))  # Use a square figure.
    sns.kdeplot(dists_to_real_center[labels == 0], fill=True, color=CONFIG['colors']['real'], label="Real News",
                alpha=0.3, lw=2)
    sns.kdeplot(dists_to_real_center[labels == 1], fill=True, color=CONFIG['colors']['fake'], label="Fake News",
                alpha=0.2, lw=2)
    plt.axvline(tau, color='red', ls='--', lw=2, label=f'Threshold $\\tau$')

    plt.title("Manifold Distance Distribution", fontsize=16, fontweight='bold')
    plt.xlabel("Distance from Manifold Center", fontsize=14)
    plt.ylabel("Density", fontsize=14)

    # Configure the legend.
    legend = plt.legend(fontsize=12)
    if legend:
        plt.setp(legend.get_texts(), fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.4)

    # Save the PNG in the output directory beside this script.
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'manifold_distance_distribution.png'), dpi=CONFIG['dpi'], bbox_inches='tight')
    print('>>> KDE plot generated.')

    # --- 2. t-SNE dimensionality-reduction visualization ---
    print('>>> Generating the t-SNE visualization...')

    # Retain the existing center selection strategy.
    combined = np.vstack([phi_h, real_cluster_center])
    tsne = TSNE(n_components=2, perplexity=50, learning_rate='auto', init='pca', random_state=42)
    res_2d = tsne.fit_transform(combined)

    points_2d = res_2d[:-1]
    center_2d = res_2d[-1]

    # Create a square canvas.
    fig, ax = plt.subplots(figsize=(8, 8))

    # Real news: blue markers with alpha=0.6.
    ax.scatter(points_2d[labels == 0, 0], points_2d[labels == 0, 1],
               c=CONFIG['colors']['real'], alpha=0.6, s=30, label='Real News',
               edgecolors='darkblue', linewidths=0.5, zorder=2)

    # Fake news: orange markers with alpha=0.6.
    ax.scatter(points_2d[labels == 1, 0], points_2d[labels == 1, 1],
               c=CONFIG['colors']['fake'], alpha=0.6, s=30, label='Fake News',
               edgecolors='darkorange', linewidths=0.5, zorder=2)

    # Plot the manifold center as a gold star.
    ax.scatter(center_2d[0], center_2d[1], c='gold', marker='*', s=200,
               edgecolors='black', linewidths=1.5, label='Manifold Center $\mathbf{c}$', zorder=5)

    # Illustrative boundary circle in t-SNE space. t-SNE does not preserve
    # absolute distances, so draw the circle at the 95th-percentile of the 2-D
    # distances from the projected real-cluster center (schematic only).
    dists_2d_to_center = np.linalg.norm(points_2d - center_2d, axis=1)
    radius = np.percentile(dists_2d_to_center[labels == 0], 95)
    circle = plt.Circle((center_2d[0], center_2d[1]), radius,
                        color='red', fill=False, linestyle='--', alpha=0.4, lw=2,
                        label='Manifold Boundary', zorder=3)
    ax.add_patch(circle)

    ax.set_aspect('equal', adjustable='box')

    plt.title("t-SNE Visualization of Real and Fake News", fontsize=16, fontweight='bold')

    # Configure the legend position.
    legend = plt.legend(
        loc='lower left',
        bbox_to_anchor=(0.4, 0.00),
        fontsize=12,
        frameon=True,
        shadow=False
    )
    if legend:
        plt.setp(legend.get_texts(), fontweight='bold')

    plt.grid(linestyle=':', alpha=0.5)

    # Save the PNG in the output directory beside this script.
    plt.savefig(os.path.join(output_dir, 'manifold_embedding.png'), dpi=CONFIG['dpi'], bbox_inches='tight')
    plt.show()
    print('>>> t-SNE plot generated: PNG')


if __name__ == '__main__':
    main()