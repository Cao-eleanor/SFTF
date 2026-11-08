"""
Plot Kaplan-Meier survival curves comparing time to sustained propagation 
deviation for real and fake news.
"""


import os
from _input_paths import default_inputs, configure_inputs
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib import rcParams


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
# CONFIG
# =========================
CONFIG = {
    **default_inputs(include_train=True),

    # must match training
    "num_channels": 36,
    "num_time_bins": 100,
    "max_time": 172800,
    "k_selected": 12,
    "embed_dim": 64,
    "metric_dim": 128,
    "bert_dim": 768,
    "bert_weight": 0.2,

    # survival settings
    "tau_quantile": 0.95,
    "persist": 3,

    # speed control
    "tau_train_cap": 4000,  # None means use all

    # incremental metric choice (keep as dz to match your current plot)
    "delta_metric": "dz",  # "dz" recommended; "dd" optional

    # inference
    "batch_size": 128,
    "seed": 8,

    # output
    "out_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"),
    "fig_prefix": "deviation_survival",
    # Use a square canvas.
    "fig_size": (8, 8),
    "dpi": 600,

    # Configure class colors.
    'colors': {
        'real': '#1f77b4',  # Blue.
        'fake': '#ff7f0e',  # Orange.
    }
}

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])


# =========================
# Data processing
# =========================
def parse_metapath_file(path):
    groups, order = {}, []
    if not os.path.exists(path):
        raise FileNotFoundError(f"metapath file not found: {path}")
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


def load_bert(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[{label}] npy not found: {path}")
    x = np.load(path).astype(np.float32)
    if x.ndim != 2 or x.shape[1] != CONFIG["bert_dim"]:
        raise ValueError(f"[{label}] expected (N,{CONFIG['bert_dim']}), got {x.shape}")
    return x


def build_log_signal_matrix(lines):
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


def load_split(mp_path, bert_path, label):
    groups, order = parse_metapath_file(mp_path)
    bert = load_bert(bert_path, label)
    if len(order) != bert.shape[0]:
        raise ValueError(f"[{label}] mismatch: metapath={len(order)}, bert={bert.shape[0]}")
    Xf = np.array([build_log_signal_matrix(groups[nid]) for nid in order], dtype=np.float32)
    Xb = bert.astype(np.float32)
    return Xf, Xb


# =========================
# Model
# =========================
class MetaTrendAttentionEncoder(nn.Module):
    def __init__(self, k_selected, embed_dim):
        super().__init__()
        self.path_projection = nn.Linear(k_selected * 2, embed_dim)
        # name MUST be "attention" to match ckpt
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x_signal):
        fft_res = torch.fft.rfft(x_signal, dim=-1)
        selected = fft_res[:, :, :CONFIG["k_selected"]]
        meta = torch.cat([selected.real, selected.imag], dim=-1)
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


# =========================
# Survival utilities
# =========================
def kaplan_meier(durations, events, max_time):
    durations = np.asarray(durations, dtype=np.int32)
    events = np.asarray(events, dtype=np.int32)

    n = len(durations)
    at_risk = n
    S = 1.0
    gsum = 0.0

    times = np.arange(0, max_time + 1, dtype=np.int32)
    surv = np.ones(max_time + 1, dtype=np.float64)
    lo = np.ones(max_time + 1, dtype=np.float64)
    hi = np.ones(max_time + 1, dtype=np.float64)

    for t in range(1, max_time + 1):
        d_t = int(np.sum((durations == t) & (events == 1)))
        c_t = int(np.sum((durations == t) & (events == 0)))

        if at_risk > 0 and d_t > 0:
            S *= (1.0 - d_t / at_risk)
            if at_risk - d_t > 0:
                gsum += d_t / (at_risk * (at_risk - d_t))

        surv[t] = S

        # log-log CI (computed but not plotted)
        if S <= 0.0:
            lo[t], hi[t] = 0.0, 0.0
        elif S >= 1.0:
            lo[t], hi[t] = 1.0, 1.0
        else:
            se_loglog = np.sqrt(gsum) / abs(np.log(S))
            z = 1.96
            a = np.log(-np.log(S))
            lo[t] = np.exp(-np.exp(a + z * se_loglog))
            hi[t] = np.exp(-np.exp(a - z * se_loglog))

        at_risk -= (d_t + c_t)
        at_risk = max(at_risk, 0)

        if t > 1 and surv[t] == surv[t - 1]:
            lo[t] = lo[t - 1]
            hi[t] = hi[t - 1]

    return times, surv, lo, hi


def logrank_test(dur1, evt1, dur2, evt2, max_time):
    dur1 = np.asarray(dur1, dtype=np.int32);
    evt1 = np.asarray(evt1, dtype=np.int32)
    dur2 = np.asarray(dur2, dtype=np.int32);
    evt2 = np.asarray(evt2, dtype=np.int32)

    O_E = 0.0
    V = 0.0

    for t in range(1, max_time + 1):
        n1 = int(np.sum(dur1 >= t))
        n2 = int(np.sum(dur2 >= t))
        n = n1 + n2
        if n <= 1:
            continue

        d1 = int(np.sum((dur1 == t) & (evt1 == 1)))
        d2 = int(np.sum((dur2 == t) & (evt2 == 1)))
        d = d1 + d2
        if d == 0:
            continue

        E1 = d * (n1 / n)
        Vt = (n1 * n2 * d * (n - d)) / ((n ** 2) * (n - 1))
        O_E += (d1 - E1)
        V += Vt

    if V <= 0:
        return 0.0, 1.0
    chi2_stat = (O_E ** 2) / V
    p = float(chi2.sf(chi2_stat, df=1))
    return float(chi2_stat), p


def cox_hr_binary(durations, events, group):
    durations = np.asarray(durations, dtype=np.int32)
    events = np.asarray(events, dtype=np.int32)
    x = np.asarray(group, dtype=np.int32)

    event_times = np.unique(durations[events == 1])
    if len(event_times) == 0:
        return 1.0, (1.0, 1.0)

    beta = 0.0
    for _ in range(30):
        U = 0.0
        H = 0.0
        exp_bx = np.exp(beta * x)

        for t in event_times:
            risk = (durations >= t)
            S0 = float(np.sum(exp_bx[risk]))
            S1 = float(np.sum(x[risk] * exp_bx[risk]))
            at_t = (durations == t) & (events == 1)
            d = int(np.sum(at_t))
            if d == 0:
                continue
            x_events = float(np.sum(x[at_t]))

            U += (x_events - d * (S1 / S0))
            H += -d * ((S1 / S0) - (S1 / S0) ** 2)

        if H == 0.0:
            break
        step = U / H
        beta_new = beta - step
        if abs(beta_new - beta) < 1e-7:
            beta = beta_new
            break
        beta = beta_new

    exp_bx = np.exp(beta * x)
    I = 0.0
    for t in event_times:
        risk = (durations >= t)
        S0 = float(np.sum(exp_bx[risk]))
        S1 = float(np.sum(x[risk] * exp_bx[risk]))
        at_t = (durations == t) & (events == 1)
        d = int(np.sum(at_t))
        if d == 0:
            continue
        I += d * ((S1 / S0) - (S1 / S0) ** 2)

    se = np.sqrt(1.0 / I) if I > 0 else 0.0
    hr = float(np.exp(beta))
    z = 1.96
    ci = (float(np.exp(beta - z * se)), float(np.exp(beta + z * se)))
    return hr, ci


# =========================
# Core: incremental-only Δz (Full model)
# =========================
@torch.no_grad()
def compute_hs(model, Xb, device):
    model.eval()
    Xb_t = torch.tensor(Xb, dtype=torch.float32, device=device)
    N = Xb_t.shape[0]
    hs = torch.empty((N, CONFIG["embed_dim"]), dtype=torch.float32, device=device)
    for s in range(0, N, CONFIG["batch_size"]):
        e = min(N, s + CONFIG["batch_size"])
        hs[s:e] = model.bert_fc(Xb_t[s:e]) * CONFIG["bert_weight"]
    return hs


def apply_mode(ht, hs, mode):
    # keep function for minimal change; only "full" will be used
    if mode == "full":
        return ht, hs
    if mode == "bonly":
        return torch.zeros_like(ht), hs
    if mode == "tonly":
        return ht, torch.zeros_like(hs)
    raise ValueError(f"unknown mode: {mode}")


@torch.no_grad()
def embed_at_k_mode(model, Xf, hs, k, mask_mat, device, mode):
    model.eval()
    c = model.center.detach().to(device)

    Xf_t = torch.tensor(Xf, dtype=torch.float32, device=device)
    N, _, T = Xf_t.shape
    mask = mask_mat[k - 1].view(1, 1, T)

    Z = np.zeros((N, CONFIG["metric_dim"]), dtype=np.float32)
    D = np.zeros(N, dtype=np.float32)

    for s in range(0, N, CONFIG["batch_size"]):
        e = min(N, s + CONFIG["batch_size"])
        xk = Xf_t[s:e] * mask
        ht = model.struct_encoder(xk)
        ht_m, hs_m = apply_mode(ht, hs[s:e], mode)
        hk = torch.cat([ht_m, hs_m], dim=1)
        zk = model.phi(hk)
        dk = torch.norm(zk - c, p=2, dim=1)

        Z[s:e] = zk.detach().cpu().numpy().astype(np.float32)
        D[s:e] = dk.detach().cpu().numpy().astype(np.float32)

    return Z, D


def delta_from_baseline(Zk, Dk, Z1, D1, metric):
    if metric == "dz":
        diff = (Zk.astype(np.float32) - Z1.astype(np.float32))
        return np.sqrt(np.sum(diff * diff, axis=1)).astype(np.float32)
    if metric == "dd":
        return (Dk.astype(np.float32) - D1.astype(np.float32)).astype(np.float32)
    raise ValueError(f"unknown delta_metric: {metric}")


@torch.no_grad()
def compute_tau_delta_schedule_mode(model, Xtr_f, Xtr_b, device, q, mode, delta_metric):
    if CONFIG["tau_train_cap"] is not None and Xtr_f.shape[0] > CONFIG["tau_train_cap"]:
        rng = np.random.RandomState(CONFIG["seed"])
        idx = rng.choice(Xtr_f.shape[0], size=CONFIG["tau_train_cap"], replace=False)
        Xf_use = Xtr_f[idx]
        Xb_use = Xtr_b[idx]
    else:
        Xf_use, Xb_use = Xtr_f, Xtr_b

    hs = compute_hs(model, Xb_use, device)
    T = CONFIG["num_time_bins"]
    mask_mat = torch.tril(torch.ones((T, T), dtype=torch.float32, device=device))

    Z1, D1 = embed_at_k_mode(model, Xf_use, hs, 1, mask_mat, device, mode)

    tau = np.zeros(T + 1, dtype=np.float32)
    tau[0] = 0.0
    tau[1] = 0.0

    for k in range(2, T + 1):
        Zk, Dk = embed_at_k_mode(model, Xf_use, hs, k, mask_mat, device, mode)
        delt = delta_from_baseline(Zk, Dk, Z1, D1, delta_metric)
        tau[k] = float(np.quantile(delt, q))

    return tau


@torch.no_grad()
def time_to_deviate_delta_mode(model, Xf, Xb, device, tau_delta, persist, mode, delta_metric):
    hs = compute_hs(model, Xb, device)
    T = CONFIG["num_time_bins"]
    mask_mat = torch.tril(torch.ones((T, T), dtype=torch.float32, device=device))

    Z1_all, D1_all = embed_at_k_mode(model, Xf, hs, 1, mask_mat, device, mode)

    N = Xf.shape[0]
    durations = np.full(N, T, dtype=np.int32)
    events = np.zeros(N, dtype=np.int32)
    alive = np.ones(N, dtype=bool)
    streak = np.zeros(N, dtype=np.int32)

    for k in range(2, T + 1):
        idx = np.where(alive)[0]
        if idx.size == 0:
            break

        Zk, Dk = embed_at_k_mode(model, Xf[idx], hs[idx], k, mask_mat, device, mode)
        delt = delta_from_baseline(Zk, Dk, Z1_all[idx], D1_all[idx], delta_metric)

        exceed = delt > tau_delta[k]
        streak[idx] = np.where(exceed, streak[idx] + 1, 0)

        newly = idx[streak[idx] >= persist]
        if newly.size > 0:
            t0 = k - persist + 1
            durations[newly] = t0
            events[newly] = 1
            alive[newly] = False

    return durations, events


# =========================
# Single-panel plot (Full only)
# =========================
def plot_single_panel_full(result, out_png, delta_metric):
    title = "Kaplan-Meier survival curves for time-to-deviate"
    ylab = r"Survival Probability (Within Normal Boundary)"

    fig = plt.figure(figsize=CONFIG["fig_size"], dpi=CONFIG["dpi"])
    ax = plt.gca()

    t = result["times"]
    # Use the configured class colors.
    ax.step(t, result["S_real"], where="post", linewidth=2.2, label="Real",
            color=CONFIG['colors']['real'])
    ax.step(t, result["S_fake"], where="post", linewidth=2.2, label="Fake",
            color=CONFIG['colors']['fake'])

    ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
    ax.set_xlabel("Time bin $k$", fontsize=14, fontweight='bold')
    ax.set_ylabel(ylab, fontsize=14, fontweight='bold')

    ax.grid(True, alpha=0.22)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Configure tick label sizes.
    ax.tick_params(axis='both', which='major', labelsize=12)
    # Use bold tick labels.
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    # Annotate the statistical comparison.
    txt = (
        f"log-rank p={result['p']:.2g}\n"
        f"HR={result['HR']:.2f}\n"
        f"CI {result['CI'][0]:.2f}-{result['CI'][1]:.2f}"
    )
    ax.text(
        0.55, 0.15, txt, transform=ax.transAxes, fontsize=11, fontweight='bold', linespacing=1.8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", alpha=0.85)
    )

    # Configure the legend.
    legend = ax.legend(frameon=False, loc="upper right", fontsize=12)
    if legend:
        plt.setp(legend.get_texts(), fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=CONFIG["dpi"], bbox_inches="tight")
    plt.close(fig)


# =========================
# MAIN
# =========================
def main():
    configure_inputs(CONFIG)
    os.makedirs(CONFIG["out_dir"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> Device: {device}")

    # load data
    print(">>> Loading TRAIN REAL ...")
    Xtr_f, Xtr_b = load_split(CONFIG["paths"]["tr_r"], CONFIG["bert_paths"]["tr_r"], "TRAIN_REAL")

    print(">>> Loading TEST REAL ...")
    Xr_f, Xr_b = load_split(CONFIG["paths"]["te_r"], CONFIG["bert_paths"]["te_r"], "TEST_REAL")

    print(">>> Loading TEST FAKE ...")
    Xf_f, Xf_b = load_split(CONFIG["paths"]["te_f"], CONFIG["bert_paths"]["te_f"], "TEST_FAKE")

    # load model
    if not os.path.exists(CONFIG["model_ckpt"]):
        raise FileNotFoundError(f"checkpoint not found: {CONFIG['model_ckpt']}")
    model = ManifoldFakeNewsNet().to(device)
    sd = torch.load(CONFIG["model_ckpt"], map_location=device, weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print(">>> Model loaded OK. (K_selected=12 fixed)")

    M = CONFIG["num_time_bins"]
    q = CONFIG["tau_quantile"]
    persist = CONFIG["persist"]
    delta_metric = CONFIG["delta_metric"]

    mode = "full"
    print(f"\n=== Mode: {mode} ===")
    print(f">>> Estimating tauΔ_{mode}(k) from TRAIN-REAL (q={q}, metric={delta_metric}) ...")
    tau_delta = compute_tau_delta_schedule_mode(model, Xtr_f, Xtr_b, device, q, mode, delta_metric)
    print(f">>> tauΔ_{mode}(2)={tau_delta[2]:.4f}, tauΔ_{mode}(M)={tau_delta[M]:.4f}")

    print(f">>> Computing time-to-deviate (persist={persist}) for TEST REAL ...")
    dur_r, evt_r = time_to_deviate_delta_mode(model, Xr_f, Xr_b, device, tau_delta, persist, mode, delta_metric)

    print(f">>> Computing time-to-deviate (persist={persist}) for TEST FAKE ...")
    dur_f, evt_f = time_to_deviate_delta_mode(model, Xf_f, Xf_b, device, tau_delta, persist, mode, delta_metric)

    print(f">>> REAL: N={len(dur_r)}, event_rate={evt_r.mean():.3f}")
    print(f">>> FAKE: N={len(dur_f)}, event_rate={evt_f.mean():.3f}")

    times, S_r, L_r, U_r = kaplan_meier(dur_r, evt_r, M)
    _, S_f, L_f, U_f = kaplan_meier(dur_f, evt_f, M)

    chi2_stat, p = logrank_test(dur_r, evt_r, dur_f, evt_f, M)
    group = np.concatenate([np.zeros_like(dur_r), np.ones_like(dur_f)], axis=0)
    durations = np.concatenate([dur_r, dur_f], axis=0)
    events = np.concatenate([evt_r, evt_f], axis=0)
    HR, CI = cox_hr_binary(durations, events, group)

    print(f">>> Log-rank p={p:.3g}, chi2={chi2_stat:.3f}")
    print(f">>> Cox HR(fake/real)={HR:.3f}, 95% CI=({CI[0]:.3f},{CI[1]:.3f})")

    result = {
        "times": times,
        "S_real": S_r, "L_real": L_r, "U_real": U_r,
        "S_fake": S_f, "L_fake": L_f, "U_fake": U_f,
        "p": p, "HR": HR, "CI": CI,
        "event_rate_real": float(evt_r.mean()),
        "event_rate_fake": float(evt_f.mean()),
    }

    out_png = os.path.join(CONFIG["out_dir"], f"{CONFIG['fig_prefix']}.png")
    plot_single_panel_full(result, out_png, delta_metric)

    print("\n" + "=" * 60)
    print("Summary (for paper text):")
    print(
        f"[full] p={p:.2g}, HR={HR:.2f} (CI {CI[0]:.2f}-{CI[1]:.2f}), "
        f"event_rate real={evt_r.mean():.3f}, fake={evt_f.mean():.3f}"
    )
    print("=" * 60)
    print(">>> Saved:")
    print(f"  {out_png}")


if __name__ == "__main__":
    main()
