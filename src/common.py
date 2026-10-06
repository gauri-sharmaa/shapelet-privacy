"""Shared utilities: data loading, splits, shapelet distances, metrics."""
import os
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split


def load(path=None, seed=0):
    path = path or os.environ.get("DATA_NPZ", "data/samld_weekly.npz")
    d = np.load(path, allow_pickle=True)
    X, y = d["X"], d["y"].astype(int)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.5, stratify=y, random_state=seed)
    return d, X, y, tr, te


def znorm(a, axis=-1, eps=1e-8):
    mu = a.mean(axis, keepdims=True)
    sd = a.std(axis, keepdims=True)
    return np.where(sd > eps, (a - mu) / np.maximum(sd, eps), 0.0)


def min_dists(shapelets, X, normalise=True, chunk=4096):
    """Min sliding Euclidean distance (length-normalised) from each shapelet to each series.

    shapelets: list of (channel, values) ; X: [n, C, T]. Returns D [n_shapelets, n].
    Windows are z-normalised when ``normalise`` (matching aeon's shapelet transform).
    """
    n = X.shape[0]
    D = np.empty((len(shapelets), n), dtype=np.float32)
    by_key = {}
    for k, (c, v) in enumerate(shapelets):
        by_key.setdefault((int(c), len(v)), []).append(k)
    for (c, L), ks in by_key.items():
        S = np.stack([np.asarray(shapelets[k][1], dtype=np.float32) for k in ks])  # [m, L]
        s2 = (S ** 2).sum(1)
        for st in range(0, n, chunk):
            W = sliding_window_view(X[st:st + chunk, c, :], L, axis=-1).astype(np.float32)  # [b, P, L]
            if normalise:
                W = znorm(W).astype(np.float32)
            w2 = (W ** 2).sum(-1)  # [b, P]
            dot = np.einsum("bpl,ml->mbp", W, S)
            d2 = w2[None] + s2[:, None, None] - 2 * dot
            D[ks, st:st + chunk] = np.sqrt(np.maximum(d2.min(-1), 0) / L)
    return D


def det_metrics(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
            "recall@1%fpr": float(np.interp(0.01, fpr, tpr))}


def mia_metrics(is_member, score):
    """score: higher = more likely member."""
    fpr, tpr, _ = roc_curve(is_member, score)
    return {"auc": roc_auc_score(is_member, score),
            "tpr@0.1%fpr": float(np.interp(0.001, fpr, tpr)),
            "tpr@1%fpr": float(np.interp(0.01, fpr, tpr))}


def reid_metrics(D, src, eps=1e-4):
    """D [m, n] distances from m released shapelets to n candidate accounts; src [m] true source idx.

    top1: true source is (one of) the nearest; unique: true source is the ONLY account within eps
    (i.e. the shapelet singles out one customer); rank: pessimistic rank of the source.
    """
    d_src = D[np.arange(len(src)), src]
    rank = (D <= d_src[:, None] + 1e-7).sum(1)  # pessimistic tie handling, 1 = best
    within = (D <= eps).sum(1)
    unique = (d_src <= eps) & (within == 1)
    return {"top1": float((rank == 1).mean()), "top10": float((rank <= 10).mean()),
            "unique_match": float(unique.mean()), "median_rank": float(np.median(rank)),
            "n_candidates": int(D.shape[1])}
