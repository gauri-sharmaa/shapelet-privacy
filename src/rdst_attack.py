"""Random Dilated Shapelet Transform (RDST, SOTA shapelet method) and an amount-reconstruction attack.

RDST stores, for every shapelet, its (possibly z-normalised) values AND the per-channel mean and
std used to normalise it. So the raw window -- here, exact log weekly amounts / counts across all
6 channels of one account -- is recoverable as values * std + mean. We then search the candidate
pool for accounts containing that exact dilated window (re-identification) and check how much of
the source account's real transaction history is revealed (attribute inference).
"""
import argparse
import json
import time
import numpy as np
from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform
from common import load, det_metrics, mia_metrics, znorm
from shapelet_attack import fit_detector


def raw_windows(t, strip_stats=False):
    vals, starts, lens, dils, thr, norm, means, stds, cls = t.shapelets_
    out = []
    for i in range(len(lens)):
        v = vals[i][:, :lens[i]].astype(np.float64)
        if norm[i] and not strip_stats:
            v = v * stds[i][:, None] + means[i][:, None]
        out.append((v, int(lens[i]), int(dils[i])))
    return out


def match(raw, X, eps=1e-3, normalise=False):
    """For each released raw window, distance to the best-matching dilated window of every series."""
    n, C, T = X.shape
    D = np.empty((len(raw), n), dtype=np.float32)
    for k, (v, L, d) in enumerate(raw):
        span = (L - 1) * d + 1
        idx = np.arange(T - span + 1)[:, None] + np.arange(L)[None] * d  # [P, L]
        W = X[:, :, idx]  # [n, C, P, L]
        if normalise:
            W = znorm(W)
        dist = np.sqrt(((W - v[None, :, None, :]) ** 2).mean((1, 3)))  # [n, P]
        D[k] = dist.min(1)
    return D


def main(args):
    d, X, y, tr, te = load(seed=args.seed)
    t0 = time.time()
    t = RandomDilatedShapeletTransform(max_shapelets=args.k, random_state=args.seed, n_jobs=4,
                                       proba_normalization=1.0 if args.norm_only else 0.8)
    F_tr = t.fit_transform(X[tr], y[tr])
    F_te = t.transform(X[te])
    _, p = fit_detector(F_tr, y[tr], F_te, args.seed)
    res = {"detection": det_metrics(y[te], p), "fit_s": time.time() - t0, "n_shapelets": len(t.shapelets_[2])}
    print("RDST detection", res, flush=True)

    raw = raw_windows(t, strip_stats=args.norm_only)
    pool = np.r_[tr, te]
    D = match(raw, X[pool], normalise=args.norm_only)
    eps = 1e-3
    n_match = (D <= eps).sum(1)
    member_match = (D[:, :len(tr)] <= eps).sum(1)
    # A released shapelet "singles out" a customer if exactly one account in the pool contains it.
    unique = n_match == 1
    nonzero = np.array([np.abs(v).sum() > 1e-9 for v, _, _ in raw])
    res["reid"] = {
        "frac_exact_in_some_member": float((member_match >= 1).mean()),
        "frac_unique_customer": float(unique.mean()),
        "frac_unique_among_nonzero": float(unique[nonzero].mean()),
        "n_distinct_customers_singled_out": int(len(set(D[unique].argmin(1).tolist()))),
        "n_singled_out_suspicious": int(y[pool[np.array(sorted(set(D[unique].argmin(1).tolist())), dtype=int)]].sum()),
        "frac_singled_out_are_members": float((D[unique].argmin(1) < len(tr)).mean()) if unique.any() else 0.0,
        "median_matches": float(np.median(n_match)),
        "n_candidates": int(len(pool)),
    }
    # Attribute inference: for singled-out customers, how many non-zero weekly cells (amounts/counts)
    # are revealed exactly, and the leaked amount scale.
    cells = [int((np.abs(v) > 1e-9).sum()) for (v, _, _), u in zip(raw, unique) if u]
    amt = [float(np.expm1(v[:2]).max()) for (v, _, _), u in zip(raw, unique) if u]
    res["attribute"] = {"median_nonzero_cells_revealed_per_unique_shapelet": float(np.median(cells)) if cells else 0,
                        "median_max_weekly_amount_revealed": float(np.median(amt)) if amt else 0}
    is_mem = np.r_[np.ones(len(tr)), np.zeros(len(te))]
    res["mia"] = mia_metrics(is_mem, -D.min(0))
    print(json.dumps(res, indent=1, default=float), flush=True)
    json.dump(res, open(f"results/rdst{'_normonly' if args.norm_only else ''}_attack_seed{args.seed}.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=1000)
    p.add_argument("--norm-only", action="store_true",
                   help="defense: all shapelets z-normalised and mean/std NOT released")
    main(p.parse_args())
