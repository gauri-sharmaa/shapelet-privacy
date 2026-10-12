"""LiRA (Carlini et al. 2022) on Berka for every model in the paper, with two scores per shapelet model.

  shapelets-only  s = -log(min distance from the account to any released shapelet)
                  (what leaks if only the shapelets are shared)
  full model      s = logit of the probability the full detector assigns to the true label
Models: RDST (1,000 shapelets, as in the main tables) + ExtraTrees; learned shapelets + ExtraTrees;
DP-SGD learned shapelets at epsilon = 1 with their own (DP-trained) linear head; CatBoost.
Also the per-explanation audit for RDST (accounts matched exactly by a shapelet that matches no other
account), with member / non-member counts. Shadow scores are cached per shadow, so the run is resumable.
Usage: DATA_NPZ=data/berka_monthly.npz python3 src/lira2.py
"""
import argparse
import json
import os
import numpy as np
import torch
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, roc_curve
from catboost import CatBoostClassifier
from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform
from common import load, min_dists
from rdst_attack import raw_windows, match
from shapelet_attack import fit_detector
from defenses import train_learned
from dp_shapelets import train as train_dp


def logit_true(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    pt = np.where(y == 1, p, 1 - p)
    return np.log(pt) - np.log(1 - pt)


def neglog(D):
    return -np.log(D.min(0) + 1e-6)


def run_rdst(X, y, idx, seed, k, audit=False):
    t = RandomDilatedShapeletTransform(max_shapelets=k, random_state=seed, n_jobs=4)
    F_in = t.fit_transform(X[idx], y[idx])
    F = t.transform(X)
    clf, _ = fit_detector(F_in, y[idx], F[:1], seed)
    shp_score = neglog(F[:, 0::3].T)  # RDST's own min distance from each shapelet to each account
    uniq_hits = None
    if audit:  # exact raw-window matching, only needed for the target model
        D = match(raw_windows(t), X)
        hit = D <= 1e-3
        uniq_hits = hit[hit.sum(1) == 1].sum(0)
    return shp_score, logit_true(clf.predict_proba(F)[:, 1], y), uniq_hits


def run_learned(X, y, idx, seed):
    shp = train_learned(X.astype(np.float32), y, idx, seed)
    F = min_dists(shp, X).T
    clf, _ = fit_detector(F[idx], y[idx], F[:1], seed)
    return neglog(F.T), logit_true(clf.predict_proba(F)[:, 1], y)


def run_dp(X, y, idx, seed, eps):
    lengths = (3, 5, 8) if X.shape[2] <= 46 else (4, 8, 12)
    shp, _, m = train_dp(X.astype(np.float32), y, idx, seed, eps, 30, 8, lengths, return_model=True)
    with torch.no_grad():
        p = torch.sigmoid(m.cpu()(torch.tensor(X, dtype=torch.float32))).numpy()
    return neglog(min_dists(shp, X)), logit_true(p, y)


def run_cb(Xf, y, idx, seed):
    m = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, verbose=0, thread_count=4,
                           random_seed=seed, auto_class_weights="Balanced").fit(Xf[idx], y[idx])
    return logit_true(m.predict_proba(Xf)[:, 1], y)


def lira(target, ins, outs):
    """Online LiRA with per-example means and a global variance."""
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


def all_scores(X, Xf, y, idx, seed, k, audit=False):
    r_s, r_f, uh = run_rdst(X, y, idx, seed, k, audit)
    l_s, l_f = run_learned(X, y, idx, seed)
    d_s, d_f = run_dp(X, y, idx, seed, 1.0)
    c_f = run_cb(Xf, y, idx, seed)
    return {"rdst_shapelets": r_s, "rdst_full": r_f, "learned_shapelets": l_s, "learned_full": l_f,
            "dp1_shapelets": d_s, "dp1_full": d_f, "catboost_full": c_f}, uh


def main(a):
    torch.set_num_threads(4)
    d, X, y, tr, te = load(seed=0)
    X = np.ascontiguousarray(X, dtype=np.float64)
    n = len(y)
    Xf = np.c_[d["static"], X.reshape(n, -1)]
    mem = np.zeros(n)
    mem[tr] = 1
    rng = np.random.default_rng(123)
    os.makedirs("results/lira2_cache", exist_ok=True)
    ins, outs = {}, {}
    for s in range(a.shadows):
        idx = rng.choice(n, size=n // 2, replace=False)
        inmask = np.zeros(n, bool)
        inmask[idx] = True
        cache = f"results/lira2_cache/shadow{s}_k{a.k}.npz"
        if os.path.exists(cache):
            sc = dict(np.load(cache))
        else:
            sc, _ = all_scores(X, Xf, y, idx, 1000 + s, a.k)
            np.savez(cache, **sc)
        for key, v in sc.items():
            ins.setdefault(key, [[] for _ in range(n)])
            outs.setdefault(key, [[] for _ in range(n)])
            for i in range(n):
                (ins if inmask[i] else outs)[key][i].append(v[i])
        print(f"shadow {s + 1}/{a.shadows}", flush=True)
    tgt, uh = all_scores(X, Xf, y, tr, 0, a.k, audit=True)
    out = {"n_shadows": a.shadows, "k": a.k, "n_members": int(mem.sum()), "n_nonmembers": int(n - mem.sum())}
    for key, v in tgt.items():
        out[key] = {"lira": metrics(mem, lira(v, ins[key], outs[key])), "aggregate": metrics(mem, v)}
    flag = uh > 0
    out["rdst_per_explanation"] = {**metrics(mem, 1e6 * uh + tgt["rdst_shapelets"]),
                                   "members_flagged": int(flag[mem == 1].sum()),
                                   "nonmembers_flagged": int(flag[mem == 0].sum())}
    print(json.dumps(out, indent=1), flush=True)
    json.dump(out, open("results/lira2.json", "w"), indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--shadows", type=int, default=32)
    p.add_argument("--k", type=int, default=1000)
    main(p.parse_args())
