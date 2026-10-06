"""Do learned shapelets memorise customers when pushed (longer training, smaller training set)?

For each released learned shapelet, find the nearest account (all 29,946). If shapelets memorised training
customers, (a) the nearest account would be a training member far more often than 50%, and (b) it would be
clearly nearer than the runner-up (d2/d1 >= 2), i.e. a confident re-identification.
"""
import argparse
import json
import numpy as np
from common import load, min_dists, det_metrics, mia_metrics
from defenses import train_learned
from shapelet_attack import fit_detector

CONFIGS = {"30 epochs, full train": (30, None), "200 epochs, full train": (200, None),
           "200 epochs, 1k train": (200, 1000), "400 epochs, 300 train": (400, 300)}


def main(args):
    d, X, y, tr, te = load(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    out = {}
    for name, (ep, n_tr) in CONFIGS.items():
        sub = tr if n_tr is None else rng.choice(tr, size=n_tr, replace=False)
        shp = train_learned(X, y, sub, args.seed, epochs=ep)
        F = min_dists(shp, X).T
        _, p = fit_detector(F[sub], y[sub], F[te], args.seed)
        nonmem = np.setdiff1d(np.arange(len(y)), sub)
        D = F.T  # [m, n]
        order = np.argsort(D, 1)[:, :2]
        d1, d2 = D[np.arange(len(D)), order[:, 0]], D[np.arange(len(D)), order[:, 1]]
        is_mem = np.zeros(len(y), bool)
        is_mem[sub] = True
        near_mem = is_mem[order[:, 0]]
        conf = d2 >= 2 * d1 + 1e-9
        # Membership: compare members vs an equal-size random set of non-members.
        nm = rng.choice(nonmem, size=len(sub), replace=False)
        mia = mia_metrics(np.r_[np.ones(len(sub)), np.zeros(len(nm))], -np.r_[D[:, sub].min(0), D[:, nm].min(0)])
        out[name] = {"detection": det_metrics(y[te], p), "nearest_is_member": float(near_mem.mean()),
                     "member_base_rate": float(len(sub) / len(y)),
                     "confident_claims": int(conf.sum()), "confident_claims_on_members": int((conf & near_mem).sum()),
                     "median_d1": float(np.median(d1)), "min_d1": float(d1.min()), "mia": mia,
                     "n_shapelets": len(shp)}
        print(name, out[name], flush=True)
    json.dump(out, open(f"results/learned_probe_seed{args.seed}.json", "w"), indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
