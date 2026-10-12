"""Mitigations applied to RDST, the strong leaker (Table 2 used the classic transform).

  noise-s   Gaussian noise added to each shapelet's raw window, std = s x the channel's std over training data
  kanon-k   each shapelet replaced by the mean of its k nearest training windows (same length and
            dilation), each from a different account
After each change we write the new windows back into the fitted transform, refit the detector, and measure
  unique  share of released shapelets matching exactly one account (distance <= 1e-3)
  top1    among shapelets that singled out their source before the change, the share whose nearest
          account is still that source (nearest-neighbour linkage)
Distances are RDST's own minimum distances from each shapelet to each account.
Usage: DATA_NPZ=... python3 src/rdst_defenses.py --seed 0
"""
import argparse
import json
import os
import numpy as np
from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform
from common import load, det_metrics
from rdst_attack import raw_windows
from shapelet_attack import fit_detector


def set_windows(t, new_raw):
    """Write raw windows back into the fitted RDST, re-normalising the shapelets that use normalisation."""
    vals, starts, lens, dils, thr, norm, means, stds, cls = t.shapelets_
    vals, means, stds = vals.copy(), means.copy(), stds.copy()
    for i, w in enumerate(new_raw):
        L = int(lens[i])
        if norm[i]:
            mu, sd = w.mean(1), w.std(1)
            sd = np.where(sd > 1e-8, sd, 1.0)
            means[i], stds[i] = mu, sd
            vals[i][:, :L] = (w - mu[:, None]) / sd[:, None]
        else:
            vals[i][:, :L] = w
    t.shapelets_ = (vals, starts, lens, dils, thr, norm, means, stds, cls)


def nearest_windows(v, L, d, Xtr, k):
    """Raw windows of the k training accounts whose best-matching window is closest to v (one per account)."""
    T = Xtr.shape[2]
    idx = np.arange(T - (L - 1) * d)[:, None] + np.arange(L)[None] * d
    W = Xtr[:, :, idx]  # [n, C, P, L]
    dist = ((W - v[None, :, None, :]) ** 2).mean((1, 3))  # [n, P]
    bp = dist.argmin(1)
    acc = np.argsort(dist[np.arange(len(dist)), bp], kind="stable")[:k]
    return W[acc, :, bp[acc], :]  # [k, C, L]


def evaluate(t, X, y, tr, te, src_ok, src, seed):
    F = np.empty((len(y), 3 * len(t.shapelets_[2])))
    F[tr], F[te] = t.transform(X[tr]), t.transform(X[te])
    _, p = fit_detector(F[tr], y[tr], F[te], seed)
    D = F[:, 0::3].T  # RDST's own min distance from each shapelet to each account
    uniq = (D <= 1e-3).sum(1) == 1
    top1 = D.argmin(1) == src
    return {"detection": det_metrics(y[te], p), "unique": float(uniq.mean()),
            "top1": float(top1[src_ok].mean()) if src_ok.any() else 0.0,
            "n_identifiable_before": int(src_ok.sum())}, D


def main(a):
    d, X, y, tr, te = load(seed=a.seed)
    X = np.ascontiguousarray(X, dtype=np.float64)
    t = RandomDilatedShapeletTransform(max_shapelets=a.k, random_state=a.seed, n_jobs=4).fit(X[tr], y[tr])
    orig = [(w.copy(), L, dl) for w, L, dl in raw_windows(t)]
    saved = t.shapelets_
    src_ok, src = np.zeros(len(orig), bool), np.zeros(len(orig), int)
    res = {}
    res["as_trained"], D0 = evaluate(t, X, y, tr, te, src_ok, src, a.seed)
    src_ok, src = (D0 <= 1e-3).sum(1) == 1, D0.argmin(1)  # shapelets that single out their source
    res["as_trained"]["top1"] = 1.0
    res["as_trained"]["n_identifiable_before"] = int(src_ok.sum())
    print("as_trained", res["as_trained"], flush=True)
    ch_sd = X[tr].std((0, 2))  # per-channel std over training data
    rng = np.random.default_rng(a.seed)
    for s in [0.1, 0.25, 0.5, 1.0]:
        new = [(w + rng.normal(0, 1, w.shape) * s * ch_sd[:, None], L, dl) for w, L, dl in orig]
        t.shapelets_ = saved
        set_windows(t, [w for w, _, _ in new])
        res[f"noise-{s}"] = evaluate(t, X, y, tr, te, src_ok, src, a.seed)[0]
        print(f"noise-{s}", res[f"noise-{s}"], flush=True)
    nn = [nearest_windows(w, L, dl, X[tr], max(a.kanon)) for w, L, dl in orig]
    for k in a.kanon:
        new = [(nb[:k].mean(0), L, dl) for nb, (_, L, dl) in zip(nn, orig)]
        t.shapelets_ = saved
        set_windows(t, [w for w, _, _ in new])
        res[f"kanon-{k}"] = evaluate(t, X, y, tr, te, src_ok, src, a.seed)[0]
        print(f"kanon-{k}", res[f"kanon-{k}"], flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(f"results/rdst_defenses_seed{a.seed}.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=1000)
    p.add_argument("--kanon", type=int, nargs="+", default=[5, 20, 50])
    main(p.parse_args())
