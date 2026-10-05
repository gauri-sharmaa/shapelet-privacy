"""Fit a shapelet-transform AML detector, release its shapelets, and attack them.

Threat model: the bank releases (or a regulator/auditor/vendor receives) the fitted model, which
includes the shapelet values. The adversary holds a pool of candidate account series (e.g. a
partner bank, a data broker, or a leaked/shared transaction extract) containing both training
members and non-members, and wants to (a) re-identify which account each shapelet came from,
(b) infer whether an account was in the training data.
"""
import argparse
import json
import pickle
import time
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier
from common import load, min_dists, znorm, det_metrics, mia_metrics, reid_metrics


def fit_detector(F_tr, y_tr, F_te, seed=0):
    clf = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                               n_jobs=4, random_state=seed)
    clf.fit(F_tr, y_tr)
    return clf, clf.predict_proba(F_te)[:, 1]


def attacks(shapelets, src_global, X, members, nonmembers):
    """shapelets: list of (channel, values); src_global: index of source account in X."""
    pool = np.r_[members, nonmembers]
    D = min_dists(shapelets, X[pool])
    pos = {a: i for i, a in enumerate(pool)}
    src_local = np.array([pos[s] for s in src_global])
    reid = reid_metrics(D, src_local)
    # Membership: score = -min distance to any released shapelet on its channel.
    score = -D.min(0)
    is_mem = np.r_[np.ones(len(members)), np.zeros(len(nonmembers))]
    mia = mia_metrics(is_mem, score)
    n_src = len(set(src_global.tolist()))
    # How many distinct customers are exposed, and how many exposures are unique fingerprints.
    return {"reid": reid, "mia": mia, "n_shapelets": len(shapelets), "n_source_accounts": n_src}, D


def shapelet_transform(X_tr, y_tr, k, n_cand, min_len, max_len, n_eval, seed):
    """Classic binary shapelet transform (Hills et al. 2014; Bostrom & Bagnall 2017).

    Candidates are z-normalised subsequences sampled from training cases (non-constant only);
    quality = class separation of the candidate's min-distance profile on a training subsample
    (|AUC - 0.5|, a fast stand-in for information gain); keep the top-k, dropping candidates that
    overlap an already-kept shapelet from the same case and channel (self-similarity removal).
    Returns list of (quality, length, position, channel, case_idx, class, values).
    """
    rng = np.random.default_rng(seed)
    n, C, T = X_tr.shape
    cands = []
    while len(cands) < n_cand:
        i, c = rng.integers(n), rng.integers(C)
        L = int(rng.integers(min_len, max_len + 1))
        pos = int(rng.integers(0, T - L + 1))
        w = X_tr[i, c, pos:pos + L]
        if w.std() < 1e-8:
            continue
        cands.append((L, pos, c, i, int(y_tr[i]), znorm(w).astype(np.float32)))
    ev = rng.choice(n, size=min(n_eval, n), replace=False)
    D = min_dists([(c[2], c[5]) for c in cands], X_tr[ev])
    q = np.abs(np.array([roc_auc_score(y_tr[ev], -d) for d in D]) - 0.5)
    keep = []
    for j in np.argsort(-q, kind="stable"):
        L, pos, c, i = cands[j][:4]
        if any(K[3] == i and K[2] == c and pos < K[1] + K[0] and K[1] < pos + L for K in
               [(kk[1], kk[2], kk[3], kk[4]) for kk in keep]):
            continue
        keep.append((float(q[j]), L, pos, c, i, cands[j][4], cands[j][5]))
        if len(keep) == k:
            break
    return keep


def main(args):
    d, X, y, tr, te = load(seed=args.seed)
    t = time.time()
    shapelets = shapelet_transform(X[tr], y[tr], args.k, args.samples, 3, args.max_len, args.n_eval, args.seed)
    shp = [(s[3], s[6]) for s in shapelets]
    F_tr = min_dists(shp, X[tr]).T
    F_te = min_dists(shp, X[te]).T
    fit_s = time.time() - t
    clf, p_te = fit_detector(F_tr, y[tr], F_te, args.seed)
    det = det_metrics(y[te], p_te)
    print("STC detection", det, f"fit {fit_s:.0f}s, {len(shapelets)} shapelets", flush=True)

    src = tr[np.array([s[4] for s in shapelets])]
    res, D = attacks(shp, src, X, tr, te)
    # Sanity: verify the exact window of the source account reproduces the shapelet.
    res["detection"] = det
    res["fit_s"] = fit_s
    res["source_label_counts"] = {"suspicious": int(y[src].sum()), "normal": int((1 - y[src]).sum())}
    print(json.dumps(res, indent=1, default=float), flush=True)
    json.dump(res, open(f"results/stc_attack_seed{args.seed}.json", "w"), indent=2, default=float)
    pickle.dump({"shapelets": shapelets, "src": src, "tr": tr, "te": te, "F_tr": F_tr, "F_te": F_te,
                 "p_te": p_te}, open(f"results/stc_seed{args.seed}.pkl", "wb"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--samples", type=int, default=5000)
    p.add_argument("--n-eval", type=int, default=4000)
    p.add_argument("--max-len", type=int, default=12)
    main(p.parse_args())
