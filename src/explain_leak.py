"""Explanation-only leakage: the model never leaves the bank, investigators only see alert explanations.

For each alerted test account A, the explanation is the standard shapelet rationale: the top-m shapelets,
by detector importance, that occur in A (RDST occurrence count > 0), shown as raw windows (values * std +
mean, i.e. in the investigator's units). We check whether a shown pattern is in fact another customer's
exact history: its raw window matches exactly one account in the bank (eps = 1e-3), which is not A.
Alerts: the top-N test accounts by detector score, N = number of test positives (fixed alert budget).
Usage: DATA_NPZ=... python3 src/explain_leak.py --seed 0
"""
import argparse
import json
import os
import numpy as np
from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform
from common import load
from rdst_attack import raw_windows, match
from shapelet_attack import fit_detector


def main(a):
    d, X, y, tr, te = load(seed=a.seed)
    X = np.ascontiguousarray(X, dtype=np.float64)
    t = RandomDilatedShapeletTransform(max_shapelets=a.k, random_state=a.seed, n_jobs=4)
    F_tr = t.fit_transform(X[tr], y[tr])
    F_te = t.transform(X[te])
    clf, p = fit_detector(F_tr, y[tr], F_te, a.seed)
    imp = clf.feature_importances_.reshape(-1, 3).sum(1)
    present = F_te[:, 2::3] > 0  # [n_te, n_shapelets]: shapelet occurs in the account
    cls = np.asarray(t.shapelets_[8]).astype(int)

    n_alert = int(y[te].sum())
    alerts = np.argsort(-p)[:n_alert]  # positions within te
    dist = F_te[:, 0::3]
    top = np.argsort(-imp)[:a.pool_top]
    E = {"global": {}, "local": {}}
    for i in alerts:
        order = np.argsort(-(imp * present[i]))  # most important shapelets that occur in the account
        E["global"][i] = [j for j in order[:a.m] if present[i, j]]
        E["local"][i] = list(top[np.argsort(dist[i, top])[:a.m]])  # best-matching among the most important
    used = sorted({j for e in E.values() for v in e.values() for j in v})
    raw = raw_windows(t)
    pool = np.arange(len(y))
    D = match([raw[j] for j in used], X[pool])
    hit = D <= 1e-3
    uniq = hit.sum(1) == 1
    src = D.argmin(1)
    col = {j: k for k, j in enumerate(used)}

    def leaks(expl, i, m):
        """Other customers whose exact history appears among the first m patterns shown for alert i."""
        out = []
        for j in expl[i][:m]:
            k = col[j]
            if uniq[k] and src[k] != te[i]:
                out.append(int(src[k]))
        return out

    res = {"n_alerts": n_alert, "alert_precision": float(y[te][alerts].mean()), "m": a.m,
           "median_patterns_shown": float(np.median([len(v) for v in E["global"].values()]))}
    for name, expl in E.items():
        for m in (1, a.m):
            L = {i: leaks(expl, i, m) for i in alerts}
            who = sorted({s for v in L.values() for s in v})
            res[f"{name}_top{m}"] = {
                "frac_alerts_showing_other_customer": float(np.mean([len(v) > 0 for v in L.values()])),
                "n_other_customers_exposed": len(who),
                "n_exposed_flagged": int(y[who].sum()) if who else 0,
                "n_exposed_training_members": int(np.isin(who, tr).sum()) if who else 0,
                "label_shown_correct": float(np.mean([cls[j] == y[src[col[j]]] for i in alerts for j in expl[i][:m]
                                                      if uniq[col[j]] and src[col[j]] != te[i]])) if who else 0.0,
            }
    cells = [int((np.abs(raw[j][0]) > 1e-9).sum()) for j in used if uniq[col[j]]]
    res["median_nonzero_cells_per_leaking_pattern"] = float(np.median(cells)) if cells else 0.0
    print(json.dumps(res, indent=1), flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(f"results/explain_leak_seed{a.seed}.json", "w"), indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=1000)
    p.add_argument("--m", type=int, default=3)
    p.add_argument("--pool-top", type=int, default=50)
    main(p.parse_args())
