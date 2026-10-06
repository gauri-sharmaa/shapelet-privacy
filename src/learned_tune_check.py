"""Uniqueness check for the tuned learned shapelets on AMLworld: do near-copies single out one account?"""
import json
import numpy as np
from common import load, min_dists
from defenses import train_learned

d, X, y, tr, te = load(seed=0)
shp = train_learned(X, y, tr, 0, epochs=150, per=16, lengths=(3, 5, 8))
D = min_dists(shp, X)
near = (D < 1e-2)
n_near = near.sum(1)
close = n_near >= 1
res = {"n_shapelets": len(shp), "frac_with_near_copy": float(close.mean()),
       "frac_unique_near_copy": float((n_near == 1).mean()),
       "median_accounts_matched_when_close": float(np.median(n_near[close])) if close.any() else 0,
       "unique_sources_are_members": int(sum(D[k].argmin() in set(tr.tolist()) for k in np.where(n_near == 1)[0])),
       "len_of_close": [len(shp[k][1]) for k in np.where(close)[0]]}
print(res)
json.dump(res, open("results/learned_tune_check_seed0.json", "w"), indent=2)
