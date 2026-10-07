"""Two reviewer-driven checks on RDST (SAML-D):

1. Tailored membership audit. Score each account by how many released shapelets match it and no other account
   (falls back to -min distance). Reports TPR at 0 / 0.1% / 1% FPR, to compare with the aggregate AUC.
2. Bank-scale candidate lists. Re-identification uniqueness when the adversary searches far more accounts:
   the 29,946 study accounts plus up to ~220k extra SAML-D accounts (all with >=5 transactions).
"""
import argparse
import json
import numpy as np
import pandas as pd
from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform
from sklearn.metrics import roc_curve, roc_auc_score
from common import load
from rdst_attack import raw_windows
from series import build, samld_tx


def match_chunked(raw, X, chunk=20000):
    n, C, T = X.shape
    D = np.empty((len(raw), n), dtype=np.float32)
    for k, (v, L, d) in enumerate(raw):
        span = (L - 1) * d + 1
        idx = np.arange(T - span + 1)[:, None] + np.arange(L)[None] * d
        v32 = v.astype(np.float32)
        for s in range(0, n, chunk):
            W = X[s:s + chunk][:, :, idx]
            D[k, s:s + chunk] = np.sqrt(((W - v32[None, :, None, :]) ** 2).mean((1, 3))).min(1)
    return D


def tpr_at(y, s, f):
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= f].max()) if (fpr <= f).any() else 0.0


def main(a):
    d, X, y, tr, te = load(seed=a.seed)
    t = RandomDilatedShapeletTransform(max_shapelets=1000, random_state=a.seed, n_jobs=4).fit(X[tr], y[tr])
    raw = raw_windows(t)
    pool = np.r_[tr, te]
    D = match_chunked(raw, X[pool])
    out = {"seed": a.seed}

    # 1. Tailored audit: exact containment (very high score) else -min distance.
    mind = D.min(0)
    hit = D <= 1e-3
    uniq = hit.sum(1) == 1                       # shapelets that match exactly one account
    n_unique_hits = hit[uniq].sum(0)             # per account: shapelets singling it out
    exact = n_unique_hits > 0
    score = 1e6 * n_unique_hits - mind
    mem = np.r_[np.ones(len(tr)), np.zeros(len(te))]
    out["audit"] = {"aggregate_auc": float(roc_auc_score(mem, -mind)),
                    "tpr@0fpr": tpr_at(mem, score, 0.0), "tpr@0.1%fpr": tpr_at(mem, score, 0.001),
                    "tpr@1%fpr": tpr_at(mem, score, 0.01),
                    "members_singled_out": int(exact[:len(tr)].sum()),
                    "nonmembers_singled_out": int(exact[len(tr):].sum())}
    print("audit", out["audit"], flush=True)

    # 2. Bank scale: extra SAML-D accounts not in the study.
    if a.scale:
        tx, nb = samld_tx()
        act = pd.concat([tx.snd, tx.rcv]).value_counts()
        extra = act[(act >= 5) & (~act.index.isin(d["acct"]))].index.to_numpy()
        rng = np.random.default_rng(a.seed)
        extra = rng.permutation(extra)[:a.max_extra]
        Xe = build(tx, extra, nb)
        uniq0 = (D <= 1e-3).sum(1) == 1
        sel = np.where(uniq0)[0]
        sel = rng.choice(sel, size=min(a.n_shp, len(sel)), replace=False)
        De = match_chunked([raw[i] for i in sel], Xe)
        base_hits = (D[sel] <= 1e-3).sum(1)
        out["scale"] = {}
        for n_extra in [0, 70000, len(extra)]:
            hits = base_hits + (De[:, :n_extra] <= 1e-3).sum(1)
            out["scale"][str(len(pool) + n_extra)] = float((hits == 1).mean())
        print("scale (share of initially-unique shapelets still unique)", out["scale"], flush=True)
    json.dump(out, open(f"results/rdst_extra_seed{a.seed}.json", "w"), indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scale", action="store_true")
    p.add_argument("--n-shp", type=int, default=150)
    p.add_argument("--max-extra", type=int, default=220000)
    main(p.parse_args())
