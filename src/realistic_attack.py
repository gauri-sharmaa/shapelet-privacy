"""Re-identification under a realistic adversary (SAML-D).

The adversary's data differs from the bank's:
  view      amounts perturbed (log-normal sigma) and/or only a fraction of transactions visible
  pool      only a fraction of all accounts are in the adversary's candidate list (source may be absent)
Decision rule: claim the nearest account only if the runner-up is at least `ratio` times farther
(d2 >= ratio * d1). Precision = correct claims / claims; recall = correct claims / released shapelets.

Models: classic shapelet transform (shapelets reloaded from results/stc_seed*.pkl, true sources known) and
RDST (refit with the same seed; a shapelet's true source = the unique account containing it exactly in the
bank's data, otherwise it has no identifiable source and any claim on it counts as wrong).
"""
import argparse
import json
import pickle
import numpy as np
from aeon.transformations.collection.shapelet_based import RandomDilatedShapeletTransform
from common import load, min_dists
from rdst_attack import raw_windows, match
import os
from series import build, samld_tx, amlworld_tx, berka_tx

VIEWS = {  # name: (keep_p, amt_noise)
    "exact": (1.0, 0.0),
    "amounts ±5%": (1.0, 0.05),
    "80% of txns, ±5%": (0.8, 0.05),
    "50% of txns, ±5%": (0.5, 0.05),
}
POOLS = [1.0, 0.5, 0.1]
RATIOS = [1.5, 2.0, 3.0]


def decide(D, src, y_all, pool_frac, ratio, rng):
    """D [m, n_all] distances to every account; src [m] true source index (-1 = none)."""
    n = D.shape[1]
    pool = np.sort(rng.choice(n, size=int(round(pool_frac * n)), replace=False)) if pool_frac < 1 else np.arange(n)
    Dp = D[:, pool]
    order = np.argsort(Dp, axis=1)[:, :2]
    d1 = Dp[np.arange(len(Dp)), order[:, 0]]
    d2 = Dp[np.arange(len(Dp)), order[:, 1]]
    claim = d2 >= ratio * d1 + 1e-9
    who = pool[order[:, 0]]
    correct = claim & (who == src) & (src >= 0)
    in_pool = np.isin(src, pool)
    cust = np.unique(src[correct])
    return {"claims": int(claim.sum()), "correct": int(correct.sum()),
            "precision": float(correct.sum() / max(1, claim.sum())), "recall": float(correct.mean()),
            "recall_given_in_pool": float(correct[in_pool].sum() / max(1, in_pool.sum())),
            "customers_exposed": int(len(cust)), "suspicious_exposed": int(y_all[cust].sum()) if len(cust) else 0}


def main(args):
    d, X, y, tr, te = load(seed=args.seed)
    dp = os.environ.get("DATA_NPZ", "")
    tx, nb = berka_tx() if "berka" in dp else amlworld_tx() if "amlworld" in dp else samld_tx()
    views = {name: (X if kp == 1 and nz == 0 else build(tx, d["acct"], nb, kp, nz, seed=100 + args.seed))
             for name, (kp, nz) in VIEWS.items()}
    rng_sel = np.random.default_rng(args.seed)
    out = {"seed": args.seed, "stc": {}, "rdst": {}, "rdst_normonly": {}}

    pk = pickle.load(open(f"results/stc_seed{args.seed}.pkl", "rb"))
    shp = [(s[3], np.asarray(s[6], dtype=np.float32)) for s in pk["shapelets"]]
    src = np.asarray(pk["src"])
    for vname, V in views.items():
        D = min_dists(shp, V)
        for pf in POOLS:
            for r in RATIOS:
                rng = np.random.default_rng(1000 + args.seed)
                out["stc"][f"{vname}|pool={pf}|ratio={r}"] = decide(D, src, y, pf, r, rng)
        print("stc", vname, out["stc"][f"{vname}|pool=1.0|ratio=2.0"], flush=True)

    for key, norm_only in [("rdst", False), ("rdst_normonly", True)]:
        t = RandomDilatedShapeletTransform(max_shapelets=1000, random_state=args.seed, n_jobs=4,
                                           proba_normalization=1.0 if norm_only else 0.8)
        t.fit(X[tr], y[tr])
        raw = raw_windows(t, strip_stats=norm_only)
        sel = rng_sel.choice(len(raw), size=min(args.n_rdst, len(raw)), replace=False)
        raw = [raw[i] for i in sel]
        D0 = match(raw, X, normalise=norm_only)
        exact_unique = (D0 <= 1e-3).sum(1) == 1
        src_r = np.where(exact_unique, D0.argmin(1), -1)
        out[key]["frac_identifiable_exact"] = float(exact_unique.mean())
        for vname, V in views.items():
            D = D0 if vname == "exact" else match(raw, V, normalise=norm_only)
            for pf in POOLS:
                for r in RATIOS:
                    rng = np.random.default_rng(1000 + args.seed)
                    out[key][f"{vname}|pool={pf}|ratio={r}"] = decide(D, src_r, y, pf, r, rng)
            print(key, vname, out[key][f"{vname}|pool=1.0|ratio=2.0"], flush=True)
        json.dump(out, open(f"results/realistic_seed{args.seed}.json", "w"), indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-rdst", type=int, default=300)
    main(p.parse_args())
