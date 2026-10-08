"""LiRA (Carlini et al. 2022) membership audit on Berka, for RDST and CatBoost.

Shadow models are trained on random halves of all accounts. For each account we collect a per-model
score when it was IN vs OUT of the shadow's training set, fit Gaussians, and score the target model's
observation with the likelihood ratio. Scores:
  RDST     s = -log(min distance from the account to any released shapelet + 1e-6)   (lower distance = more "in")
  CatBoost s = logit of the probability assigned to the true label
Reports AUC and TPR at 0.1% / 1% FPR for both, next to the per-explanation audit.
"""
import argparse
import json
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve
from catboost import CatBoostClassifier
from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform
from common import load
from rdst_attack import raw_windows, match


def rdst_score(X, y, idx_train, seed, k):
    t = RandomDilatedShapeletTransform(max_shapelets=k, random_state=seed, n_jobs=4).fit(X[idx_train], y[idx_train])
    D = match(raw_windows(t), X)
    hit = D <= 1e-3
    uniq_hits = hit[hit.sum(1) == 1].sum(0)
    return -np.log(D.min(0) + 1e-6), uniq_hits


def cb_score(Xf, y, idx_train, seed):
    m = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, verbose=0, thread_count=4,
                           random_seed=seed, auto_class_weights="Balanced").fit(Xf[idx_train], y[idx_train])
    p = np.clip(m.predict_proba(Xf)[:, 1], 1e-6, 1 - 1e-6)
    pt = np.where(y == 1, p, 1 - p)
    return np.log(pt) - np.log(1 - pt)


def lira(target, ins, outs):
    """Online LiRA with per-example Gaussians (global variance fallback for small samples)."""
    mu_in = np.array([np.mean(v) if len(v) else np.nan for v in ins])
    mu_out = np.array([np.mean(v) if len(v) else np.nan for v in outs])
    sd_in = np.nanstd(np.concatenate([np.array(v) - np.mean(v) for v in ins if len(v) > 1])) + 1e-6
    sd_out = np.nanstd(np.concatenate([np.array(v) - np.mean(v) for v in outs if len(v) > 1])) + 1e-6
    mu_in = np.where(np.isnan(mu_in), np.nanmean(mu_in), mu_in)
    mu_out = np.where(np.isnan(mu_out), np.nanmean(mu_out), mu_out)
    return norm.logpdf(target, mu_in, sd_in) - norm.logpdf(target, mu_out, sd_out)


def metrics(mem, s):
    fpr, tpr, _ = roc_curve(mem, s)
    at = lambda f: float(tpr[fpr <= f].max()) if (fpr <= f).any() else 0.0
    return {"auc": float(roc_auc_score(mem, s)), "tpr@0.1%fpr": at(0.001), "tpr@1%fpr": at(0.01)}


def main(a):
    d, X, y, tr, te = load(seed=0)
    n = len(y)
    Xf = np.c_[d["static"], X.reshape(n, -1)]
    mem = np.zeros(n)
    mem[tr] = 1
    rng = np.random.default_rng(123)
    ins = {"rdst": [[] for _ in range(n)], "cb": [[] for _ in range(n)]}
    outs = {"rdst": [[] for _ in range(n)], "cb": [[] for _ in range(n)]}
    for s in range(a.shadows):
        idx = rng.choice(n, size=n // 2, replace=False)
        inmask = np.zeros(n, bool)
        inmask[idx] = True
        r, _ = rdst_score(X, y, idx, 1000 + s, a.k)
        c = cb_score(Xf, y, idx, 1000 + s)
        for i in range(n):
            (ins if inmask[i] else outs)["rdst"][i].append(r[i])
            (ins if inmask[i] else outs)["cb"][i].append(c[i])
        print(f"shadow {s + 1}/{a.shadows}", flush=True)
    r_t, uniq_hits = rdst_score(X, y, tr, 0, a.k)
    c_t = cb_score(Xf, y, tr, 0)
    out = {"n_shadows": a.shadows, "k": a.k,
           "rdst_lira": metrics(mem, lira(r_t, ins["rdst"], outs["rdst"])),
           "rdst_aggregate": metrics(mem, r_t),
           "rdst_per_explanation": metrics(mem, 1e6 * uniq_hits + r_t),
           "catboost_lira": metrics(mem, lira(c_t, ins["cb"], outs["cb"])),
           "catboost_aggregate": metrics(mem, c_t)}
    print(json.dumps(out, indent=1), flush=True)
    json.dump(out, open("results/lira.json", "w"), indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shadows", type=int, default=32)
    p.add_argument("--k", type=int, default=300)
    main(p.parse_args())
