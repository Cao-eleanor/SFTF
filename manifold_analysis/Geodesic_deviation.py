"""
Plot the relationship between Euclidean distance to the manifold center and 
geodesic excess for real and fake news.
"""

import os
from _input_paths import default_inputs, configure_inputs
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scipy.sparse as sp
import scipy.sparse.csgraph as csgraph


# =========================
# Font configuration
# =========================
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

# =========================
# CONFIG (EDIT PATHS)
# =========================
CONFIG = {
    **default_inputs(include_train=True),

    "num_channels": 36,
    "num_time_bins": 100,
    "max_time": 172800,
    "k_selected": 12,

    "embed_dim": 64,
    "metric_dim": 128,
    "bert_dim": 768,
    "bert_weight": 0.2,

    "batch_size": 64,
    "seed": 8,

    # manifold graph
    "knn_k": 10,
    "mutual_prune_q": 0.90,
    "attach_k": 15,

    # core
    "core_quantile": 0.30,

    # bridge components to avoid inf
    "bridge_components": True,
    "inf_margin": 5.0,

    # output
    "out_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"),
    "dpi": 600,  # Configure the export resolution in DPI.
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


# =========================
# Data processing
# =========================
def build_signal_matrix(lines):
    mat = np.zeros((CONFIG["num_channels"], CONFIG["num_time_bins"]), dtype=np.float32)
    for line in lines:
        parts = line.split("——")
        try:
            n_sent = int(parts[0].split("_")[2])
            c1_sent = int(parts[2].split("_")[2])
            t_val = float(parts[1])
            t_idx = min(int((t_val / CONFIG["max_time"]) * CONFIG["num_time_bins"]),
                        CONFIG["num_time_bins"] - 1)

            if len(parts) >= 5 and ("None" not in parts[4]):
                c2_sent = int(parts[4].split("_")[2])
                ch_idx = 9 + (n_sent * 9) + (c1_sent * 3) + c2_sent
            else:
                ch_idx = n_sent * 3 + c1_sent

            if ch_idx < CONFIG["num_channels"]:
                mat[ch_idx, t_idx] += 1.0
        except Exception:
            continue
    return np.log1p(mat).astype(np.float32)


def parse_metapath_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"metapath file not found: {path}")
    groups, order = {}, []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("——")
            nid = parts[0].split("_")[1]
            if nid not in groups:
                groups[nid] = []
                order.append(nid)
            groups[nid].append(line)
    return groups, order


def load_bert(path, label="BERT"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[{label}] npy not found: {path}")
    x = np.load(path).astype(np.float32)
    if x.ndim != 2 or x.shape[1] != CONFIG["bert_dim"]:
        raise ValueError(f"[{label}] expected (N,{CONFIG['bert_dim']}), got {x.shape}")
    return x


def load_split(mp_path, bert_path, label):
    groups, order = parse_metapath_file(mp_path)
    bert = load_bert(bert_path, label=label)
    if len(order) != bert.shape[0]:
        raise ValueError(f"[{label}] mismatch: metapath={len(order)}, bert={bert.shape[0]}")
    Xf = np.array([build_signal_matrix(groups[nid]) for nid in order], dtype=np.float32)
    Xb = bert.astype(np.float32)
    return Xf, Xb


class HybridDataset(Dataset):
    def __init__(self, Xf, Xb):
        self.Xf = Xf
        self.Xb = Xb

    def __len__(self):
        return self.Xf.shape[0]

    def __getitem__(self, i):
        return (torch.tensor(self.Xf[i], dtype=torch.float32),
                torch.tensor(self.Xb[i], dtype=torch.float32))


# =========================
# Model (same as training)
# =========================
class MetaTrendAttentionEncoder(nn.Module):
    def __init__(self, k_selected, embed_dim):
        super().__init__()
        self.path_projection = nn.Linear(k_selected * 2, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x_signal):
        fft_res = torch.fft.rfft(x_signal, dim=-1)
        selected_fft = fft_res[:, :, :CONFIG["k_selected"]]
        meta = torch.cat([selected_fft.real, selected_fft.imag], dim=-1)
        tokens = self.path_projection(meta)
        out, _ = self.attention(tokens, tokens, tokens)
        out = self.norm(out + tokens)
        return torch.mean(out, dim=1)


class ManifoldFakeNewsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.struct_encoder = MetaTrendAttentionEncoder(CONFIG["k_selected"], CONFIG["embed_dim"])
        self.bert_fc = nn.Sequential(
            nn.Linear(CONFIG["bert_dim"], 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, CONFIG["embed_dim"])
        )
        self.phi = nn.Sequential(
            nn.Linear(CONFIG["embed_dim"] * 2, 256),
            nn.ReLU(),
            nn.Linear(256, CONFIG["metric_dim"])
        )
        self.center = nn.Parameter(torch.randn(1, CONFIG["metric_dim"]))
        self.tau = nn.Parameter(torch.tensor([1.0]))

    @torch.no_grad()
    def encode_metric(self, x_freq, x_bert):
        self.eval()
        h_t = self.struct_encoder(x_freq)
        h_s = self.bert_fc(x_bert) * CONFIG["bert_weight"]
        h = torch.cat([h_t, h_s], dim=1)
        return self.phi(h)


@torch.no_grad()
def infer_Z(model, Xf, Xb, device):
    ds = HybridDataset(Xf, Xb)
    loader = DataLoader(ds, batch_size=CONFIG["batch_size"], shuffle=False)
    out = []
    for xf, xb in loader:
        z = model.encode_metric(xf.to(device), xb.to(device))
        out.append(z.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


# =========================
# Manifold graph utils
# =========================
def build_mutual_knn_graph(Z, k=10, prune_q=0.90):
    N = Z.shape[0]
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(Z)
    knn_dist, knn_ind = nbrs.kneighbors(Z, return_distance=True)

    edges = {}
    for i in range(N):
        for t in range(1, k + 1):
            j = int(knn_ind[i, t])
            edges[(i, j)] = float(knn_dist[i, t])

    mutual = {(i, j): w for (i, j), w in edges.items() if (j, i) in edges}
    if len(mutual) == 0:
        raise RuntimeError("Mutual-kNN edges=0. Try larger k (e.g., 12~15).")

    w_all = np.array(list(mutual.values()), dtype=np.float32)
    thr = float(np.quantile(w_all, prune_q))

    keep = [(i, j, w) for (i, j), w in mutual.items() if w <= thr]
    rows = np.array([x[0] for x in keep], dtype=np.int32)
    cols = np.array([x[1] for x in keep], dtype=np.int32)
    data = np.array([x[2] for x in keep], dtype=np.float32)

    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    A = A + A.T
    A.eliminate_zeros()
    return nbrs, A


def bridge_components_to_core(A, Z, core_idx):
    n_components, labels = csgraph.connected_components(A, directed=False, return_labels=True)
    if n_components <= 1:
        return A

    core_labels = labels[core_idx]
    core_comp = np.bincount(core_labels).argmax()
    core_nodes = np.where(labels == core_comp)[0]

    nn_core = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(Z[core_nodes])

    add_rows, add_cols, add_data = [], [], []
    for comp in range(n_components):
        if comp == core_comp:
            continue
        comp_nodes = np.where(labels == comp)[0]
        if len(comp_nodes) == 0:
            continue
        dist, idx = nn_core.kneighbors(Z[comp_nodes], return_distance=True)
        dist = dist.reshape(-1)
        idx = idx.reshape(-1)
        m = int(np.argmin(dist))
        u = int(comp_nodes[m])
        v = int(core_nodes[idx[m]])
        w = float(dist[m])

        add_rows += [u, v]
        add_cols += [v, u]
        add_data += [w, w]

    if len(add_data) == 0:
        return A

    A_add = sp.csr_matrix((np.array(add_data, dtype=np.float32),
                           (np.array(add_rows, dtype=np.int32), np.array(add_cols, dtype=np.int32))),
                          shape=A.shape)
    A2 = A + A_add
    A2.eliminate_zeros()
    return A2


def multi_source_shortest_path_to_core(A, core_idx):
    """
    Multi-source Dijkstra via super-source (COO build, avoid CSR structure edits)
    """
    N = A.shape[0]
    super_id = N

    A = A.tocoo()
    rows = A.row.astype(np.int32)
    cols = A.col.astype(np.int32)
    data = A.data.astype(np.float32)

    core_idx = np.array(core_idx, dtype=np.int32)
    ss_rows = np.full_like(core_idx, super_id)
    ss_cols = core_idx
    ss_data = np.zeros_like(core_idx, dtype=np.float32)

    ss_rows2 = core_idx
    ss_cols2 = np.full_like(core_idx, super_id)
    ss_data2 = np.zeros_like(core_idx, dtype=np.float32)

    all_rows = np.concatenate([rows, ss_rows, ss_rows2], axis=0)
    all_cols = np.concatenate([cols, ss_cols, ss_cols2], axis=0)
    all_data = np.concatenate([data, ss_data, ss_data2], axis=0)

    A2 = sp.coo_matrix((all_data, (all_rows, all_cols)), shape=(N + 1, N + 1)).tocsr()
    dist = csgraph.dijkstra(A2, directed=False, indices=super_id)
    return np.asarray(dist[:N], dtype=np.float32)


def geodesic_to_core(sample_Z, nbrs_real, dist_to_core, attach_k=15, inf_fill=None):
    dist_u, idx_u = nbrs_real.kneighbors(sample_Z, n_neighbors=attach_k, return_distance=True)
    vals = dist_u + dist_to_core[idx_u]
    if inf_fill is not None:
        vals = np.where(np.isfinite(vals), vals, inf_fill)
    return np.min(vals, axis=1).astype(np.float32)


# =========================
# Plot (ONLY ONE FIGURE)
# =========================
def plot_hexbin_excess(dE_real, ex_real, dE_fake, ex_fake, tauE, out_png):
    # Create a square figure.
    plt.figure(figsize=(8, 6), dpi=CONFIG["dpi"])
    ax = plt.gca()

    hb = ax.hexbin(dE_real, ex_real, gridsize=55, mincnt=1, linewidths=0.0)
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label("Real density (hexbin count)", fontsize=14, fontweight='bold')

    # Configure colorbar tick label fonts.
    cb.ax.tick_params(labelsize=12)
    for label in cb.ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.scatter(dE_fake, ex_fake, s=16, alpha=0.40, label="Fake", linewidths=0)
    ax.axhline(0.0, linestyle='-.', color='gray', linewidth=1.2, label=r"Geometric-consistency baseline")
    ax.axvline(tauE, linestyle='--', color='crimson', linewidth=1.2, label=r"Normality boundary (Euclidean cutoff)")

    ax.set_xlabel(r"Distance to Manifold Center", fontsize=14, fontweight='bold')
    ax.set_ylabel(r"Manifold Detour", fontsize=14, fontweight='bold')
    ax.set_title("Geometry violation measured by geodesic excess", fontsize=16, fontweight='bold')

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.22)

    # Configure the legend.
    legend = ax.legend(frameon=False, loc="upper right", fontsize=12)
    if legend:
        plt.setp(legend.get_texts(), fontweight='bold')

    # Configure axis tick label fonts.
    ax.tick_params(axis='both', which='major', labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=CONFIG["dpi"], bbox_inches="tight")
    plt.close()


# =========================
# MAIN
# =========================
def main():
    configure_inputs(CONFIG)
    os.makedirs(CONFIG["out_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Device: {device}")

    # load data (ONLY train real + test real/fake)
    tr_r_f, tr_r_b = load_split(CONFIG["paths"]["tr_r"], CONFIG["bert_paths"]["tr_r"], "TRAIN_REAL")
    te_r_f, te_r_b = load_split(CONFIG["paths"]["te_r"], CONFIG["bert_paths"]["te_r"], "TEST_REAL")
    te_f_f, te_f_b = load_split(CONFIG["paths"]["te_f"], CONFIG["bert_paths"]["te_f"], "TEST_FAKE")

    # load model
    if not os.path.exists(CONFIG["model_ckpt"]):
        raise FileNotFoundError(f"checkpoint not found: {CONFIG['model_ckpt']}")

    model = ManifoldFakeNewsNet().to(device)
    sd = torch.load(CONFIG["model_ckpt"], map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval()

    c_hat = model.center.detach().cpu().numpy().reshape(-1)
    tauE = float(model.tau.detach().cpu().numpy().reshape(-1)[0])
    print(f">>> tauE (learned) = {tauE:.4f}")

    # infer embeddings
    Z_tr_r = infer_Z(model, tr_r_f, tr_r_b, device)
    Z_te_r = infer_Z(model, te_r_f, te_r_b, device)
    Z_te_f = infer_Z(model, te_f_f, te_f_b, device)

    # Euclidean deviation
    dE_tr_r = np.linalg.norm(Z_tr_r - c_hat[None, :], axis=1).astype(np.float32)
    dE_te_r = np.linalg.norm(Z_te_r - c_hat[None, :], axis=1).astype(np.float32)
    dE_te_f = np.linalg.norm(Z_te_f - c_hat[None, :], axis=1).astype(np.float32)

    # build manifold graph on TRAIN REAL
    nbrs_real, A = build_mutual_knn_graph(Z_tr_r, k=CONFIG["knn_k"], prune_q=CONFIG["mutual_prune_q"])

    # core set
    core_thr = float(np.quantile(dE_tr_r, CONFIG["core_quantile"]))
    core_idx = np.where(dE_tr_r <= core_thr)[0]
    print(f">>> core |C|={len(core_idx)} (quantile={CONFIG['core_quantile']}, thr={core_thr:.4f})")

    # bridge components to avoid inf
    if CONFIG["bridge_components"]:
        before = csgraph.connected_components(A, directed=False, return_labels=False)
        A = bridge_components_to_core(A, Z_tr_r, core_idx)
        after = csgraph.connected_components(A, directed=False, return_labels=False)
        print(f">>> components: before={before}, after={after}")

    # multi-source shortest path to core
    dist_to_core = multi_source_shortest_path_to_core(A, core_idx)

    # safety: replace inf
    finite = np.isfinite(dist_to_core)
    if not np.all(finite):
        max_finite = float(np.max(dist_to_core[finite])) if np.any(finite) else 0.0
        fill = max_finite + CONFIG["inf_margin"]
        dist_to_core[~finite] = fill
        print(f"!!! dist_to_core had inf -> filled with {fill:.4f}, unreachable={np.sum(~finite)}")

    # compute geodesic deviation to core
    max_finite = float(np.max(dist_to_core))
    inf_fill = max_finite + CONFIG["inf_margin"]

    dG_te_r = geodesic_to_core(Z_te_r, nbrs_real, dist_to_core, attach_k=CONFIG["attach_k"], inf_fill=inf_fill)
    dG_te_f = geodesic_to_core(Z_te_f, nbrs_real, dist_to_core, attach_k=CONFIG["attach_k"], inf_fill=inf_fill)

    # excess
    ex_te_r = (dG_te_r - dE_te_r).astype(np.float32)
    ex_te_f = (dG_te_f - dE_te_f).astype(np.float32)

    # plot ONLY ONE FIGURE
    out_png = os.path.join(CONFIG["out_dir"], "geodesic_deviation.png")
    plot_hexbin_excess(dE_te_r, ex_te_r, dE_te_f, ex_te_f, tauE, out_png)

    print(f">>> Saved ONE figure:\n  {out_png}")


if __name__ == "__main__":
    main()