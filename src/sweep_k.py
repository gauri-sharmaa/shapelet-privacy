"""Leakage vs. number of released shapelets k (STC)."""
import json
import sys
import numpy as np
from common import load, min_dists, det_metrics, reid_metrics
from shapelet_attack import shapelet_transform, fit_detector

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
d, X, y, tr, te = load(seed=seed)
full = shapelet_transform(X[tr], y[tr], 1000, 8000, 3, 12, 4000, seed)
pool = np.r_[tr, te]
D_all = min_dists([(s[3], s[6]) for s in full], X[pool])
F_all = D_all.T
pos = {a: i for i, a in enumerate(pool)}
for k in [25, 50, 100, 200, 500, 1000]:
    sub = full[:k]
    src = tr[np.array([s[4] for s in sub])]
    _, p = fit_detector(F_all[:len(tr), :k], y[tr], F_all[len(tr):, :k], seed)
    D = D_all[:k]
    sl = np.array([pos[s] for s in src])
    reid = reid_metrics(D, sl)
    uniq = (D[np.arange(k), sl] <= 1e-4) & ((D <= 1e-4).sum(1) == 1)
    cust = set(src[uniq].tolist())
    res = {"k": k, "detection": det_metrics(y[te], p), "reid": reid, "n_unique_customers": len(cust),
           "n_unique_suspicious": int(sum(y[c] for c in cust))}
    print(res, flush=True)
    json.dump(res, open(f"results/sweep_k{k}.json", "w"), indent=2, default=float)
