"""Tuned learned shapelets for AMLworld's short 6-hour series: shorter lengths, more shapelets, longer training."""
import json
import os
import sys
import numpy as np
from common import load, min_dists, det_metrics, mia_metrics
from defenses import train_learned
from shapelet_attack import fit_detector

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
d, X, y, tr, te = load(seed=seed)
out = {}
for name, (lengths, per, ep) in {"len 3/5/8, 16 per, 60 ep": ((3, 5, 8), 16, 60),
                                 "len 3/5/8, 16 per, 150 ep": ((3, 5, 8), 16, 150)}.items():
    shp = train_learned(X, y, tr, seed, epochs=ep, per=per, lengths=lengths)
    F = min_dists(shp, X).T
    _, p = fit_detector(F[tr], y[tr], F[te], seed)
    D = min_dists(shp, X[np.r_[tr, te]])
    out[name] = {"detection": det_metrics(y[te], p), "n_shapelets": len(shp),
                 "nearest_account_dist_median": float(np.median(D.min(1))),
                 "frac_exact_copy(<1e-2)": float((D.min(1) < 1e-2).mean()),
                 "mia": mia_metrics(np.r_[np.ones(len(tr)), np.zeros(len(te))], -D.min(0))}
    print(name, out[name], flush=True)
json.dump(out, open(f"results/learned_tune_seed{seed}.json", "w"), indent=2)
