"""Detection baselines: CatBoost on aggregates, MiniRocket. Also a confidence-based MIA reference."""
import json
import time
import numpy as np
from catboost import CatBoostClassifier
from aeon.transformations.collection.convolution_based import MiniRocket
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from common import load, det_metrics, mia_metrics


def conf_mia(p_tr, y_tr, p_te, y_te):
    # Standard confidence attack: probability assigned to the true label.
    s = np.r_[np.where(y_tr == 1, p_tr, 1 - p_tr), np.where(y_te == 1, p_te, 1 - p_te)]
    m = np.r_[np.ones(len(y_tr)), np.zeros(len(y_te))]
    return mia_metrics(m, s)


def main(seed=0):
    d, X, y, tr, te = load(seed=seed)
    S = d["static"]
    res = {}

    t = time.time()
    cb = CatBoostClassifier(iterations=800, depth=6, learning_rate=0.05, verbose=0,
                            thread_count=4, random_seed=seed, auto_class_weights="Balanced")
    Xf = np.c_[S, X.reshape(len(X), -1)]
    cb.fit(Xf[tr], y[tr])
    p_tr, p_te = cb.predict_proba(Xf[tr])[:, 1], cb.predict_proba(Xf[te])[:, 1]
    res["catboost"] = {**det_metrics(y[te], p_te), "mia_conf": conf_mia(p_tr, y[tr], p_te, y[te]),
                       "fit_s": time.time() - t}
    print("catboost", res["catboost"], flush=True)

    t = time.time()
    mr = MiniRocket(n_kernels=10000, random_state=seed, n_jobs=4)
    F_tr = mr.fit_transform(X[tr])
    F_te = mr.transform(X[te])
    sc = StandardScaler().fit(F_tr)
    lr = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")
    lr.fit(sc.transform(F_tr), y[tr])
    p_tr, p_te = lr.predict_proba(sc.transform(F_tr))[:, 1], lr.predict_proba(sc.transform(F_te))[:, 1]
    res["minirocket"] = {**det_metrics(y[te], p_te), "mia_conf": conf_mia(p_tr, y[tr], p_te, y[te]),
                         "fit_s": time.time() - t}
    print("minirocket", res["minirocket"], flush=True)
    json.dump(res, open(f"results/baselines_seed{seed}.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
