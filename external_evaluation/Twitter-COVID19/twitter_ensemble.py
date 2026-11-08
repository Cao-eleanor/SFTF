# -*- coding: utf-8 -*-
"""
Twitter-COVID19 SFTF — multi-seed score-averaging ensemble.

Trains the SFTF model N times with different random seeds, collects the
(val, test) prediction scores from each run, AVERAGES the scores across seeds,
then selects the decision threshold on the AVERAGED validation scores and
reports test metrics on the AVERAGED test scores.

Why score-averaging (and not hard voting):
  - On a 40-sample test set, hard voting is coarse and unstable.
  - Averaging continuous probabilities reduces per-model variance and typically
    lifts AUC by 1-3 points, which is the ONLY lever left for pushing Accuracy
    past the single-model ceiling.

The model architecture / loss / signal building are reused verbatim from
twitter_train (identical model/loss to v1); only the random seed changes
between members. No MLM, no label changes, no re-splitting.

Usage:
  python twitter_ensemble.py --seeds "8,42,2026,777,1234" --pos-weight 2.0 \
      --select-by val_auc
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

import twitter_train as v2

HERE = v2.HERE
device = torch.device('cpu')


def train_one_seed(seed, pos_weight, args):
    """Train one model with `seed`; return (val_true, val_score, test_true, test_score).

    Training loop mirrors v2.train_and_eval exactly, except the best checkpoint is
    kept in memory (not written to disk) and scores are returned for ensembling.
    """
    v2.set_seed(seed)

    train_data, val_data, test_data = v2.load_dataset()

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        v2.HybridDataset(*train_data), batch_size=args.batch_size,
        shuffle=True, num_workers=0, generator=g)
    val_loader = DataLoader(
        v2.HybridDataset(*val_data), batch_size=args.batch_size,
        shuffle=False, num_workers=0)
    test_loader = DataLoader(
        v2.HybridDataset(*test_data), batch_size=args.batch_size,
        shuffle=False, num_workers=0)

    v2.set_seed(seed)

    model = v2.ManifoldFakeNewsNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = v2.ManifoldLoss(
        beta=v2.CONFIG['Beta_and_delta'],
        delta=v2.CONFIG['delta'],
        pos_weight=pos_weight)

    best_metric = -1.0
    best_epoch = 0
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0

        for x_f, x_b, labels in train_loader:
            x_f, x_b, labels = x_f.to(device), x_b.to(device), labels.to(device)
            optimizer.zero_grad()
            y_hat, d_i = model(x_f, x_b)
            loss = criterion(y_hat, labels, d_i)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        yv_true, yv_score = v2.predict_scores(model, val_loader, device)
        acc, prec, rec, f1, fb, auc = v2.metrics_at(yv_true, yv_score, 0.5)

        cur = f1 if args.select_by == 'val_f1' else auc

        if cur > best_metric:
            best_metric = cur
            best_epoch = epoch + 1
            patience_counter = 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            break

    model.load_state_dict(best_state)
    yv_true, yv_score = v2.predict_scores(model, val_loader, device)
    yt_true, yt_score = v2.predict_scores(model, test_loader, device)

    val_auc = v2.metrics_at(yv_true, yv_score, 0.5)[5]
    print(f"[seed {seed:>5}] best epoch {best_epoch:>3} | "
          f"best val {args.select_by}={best_metric:.4f} | val AUC={val_auc:.4f}")

    return yv_true, yv_score, yt_true, yt_score


def main():
    p = argparse.ArgumentParser(description="Multi-seed score-averaging ensemble.")

    p.add_argument('--seeds', type=str, default='8,42,2026,777,1234',
                   help="comma list of seeds to ensemble")
    p.add_argument('--pos-weight', type=float, default=2.0,
                   help="fake-class BCE weight (2.0 recovered the best single-model AUC)")
    p.add_argument('--select-by', type=str, default='val_auc', choices=['val_auc', 'val_f1'])
    p.add_argument('--threshold-min', type=float, default=0.05)
    p.add_argument('--threshold-max', type=float, default=0.95)
    p.add_argument('--threshold-step', type=float, default=0.005)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--lr', type=float, default=0.0003)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=100)

    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]

    print(f"\n[ENSEMBLE] {len(seeds)} seeds={seeds} | pos_weight={args.pos_weight:g} "
          f"| select_by={args.select_by}")

    val_true = None
    test_true = None
    val_scores = []
    test_scores = []

    for seed in seeds:
        yv_t, yv_s, yt_t, yt_s = train_one_seed(seed, args.pos_weight, args)

        if val_true is None:
            val_true, test_true = yv_t, yt_t
        assert (val_true == yv_t).all() and (test_true == yt_t).all(), \
            "label order mismatch across seeds (should be impossible — same split)"

        val_scores.append(yv_s)
        test_scores.append(yt_s)

    val_score = np.mean(val_scores, axis=0)
    test_score = np.mean(test_scores, axis=0)

    # --- per-member sanity: test AUC of each single model ---
    print("\nPer-member test AUC:")
    for seed, ts in zip(seeds, test_scores):
        print(f"  seed {seed:>5}: test AUC={v2.metrics_at(test_true, ts, 0.5)[5]:.4f}")
    print(f"  AVERAGED scores     : test AUC={v2.metrics_at(test_true, test_score, 0.5)[5]:.4f}")

    # --- threshold selection on AVERAGED validation only ---
    print("\n>>> Threshold selection on AVERAGED validation scores ...")
    th_f1, m_f1 = v2.select_max_f1_threshold(val_true, val_score, args)
    th_bal, m_bal = v2.select_balanced_threshold(val_true, val_score, args)
    th_ba, m_ba = v2.select_balanced_accuracy_threshold(val_true, val_score, args)

    for name, th, mv in [('max_f1', th_f1, m_f1),
                         ('balanced', th_bal, m_bal),
                         ('balanced_acc', th_ba, m_ba)]:
        print("[VAL] " + v2._fmt(f"{name}@{th:.3f}", mv))

    print("\n" + "=" * 60)
    print("ENSEMBLE TEST (thresholds fixed on averaged validation)")
    print("=" * 60)
    for name, th in [('max_f1', th_f1), ('balanced', th_bal), ('balanced_acc', th_ba)]:
        mt = v2.metrics_at(test_true, test_score, th)
        print("[TEST] " + v2._fmt(f"{name}@{th:.3f}", mt))
    print("=" * 60)


if __name__ == '__main__':
    main()
