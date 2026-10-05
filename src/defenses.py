"""Mitigations for shapelet leakage, evaluated on the privacy-utility plane.

Starting from the STC shapelets fit in shapelet_attack.py:
  none        released as-is (recomputed with our distance for a consistent comparison)
  noise-s     Gaussian noise (std s, in z-normalised units) added to every released shapelet
  kanon-k     each shapelet replaced by the mean of its k nearest training windows taken from k
              DISTINCT accounts (k-anonymous release: no shapelet is a single customer's history)
Plus, independently:
  learned     shapelets learned by gradient descent from random init (Grabocka et al. 2014 style)
"""
import argparse
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view
from common import load, min_dists, znorm, det_metrics, mia_metrics, reid_metrics
from shapelet_attack import fit_detector


def evaluate(shp, X, y, tr, te, src=None, seed=0):
    F = min_dists(shp, X).T  # [n, m] features for every account
    _, p = fit_detector(F[tr], y[tr], F[te], seed)
    out = {"detection": det_metrics(y[te], p)}
    pool = np.r_[tr, te]
    D = F[pool].T
    is_mem = np.r_[np.ones(len(tr)), np.zeros(len(te))]
    out["mia"] = mia_metrics(is_mem, -D.min(0))
    if src is not None:
        pos = {a: i for i, a in enumerate(pool)}
        out["reid"] = reid_metrics(D, np.array([pos[s] for s in src]))
    return out


def kanon(shp, X_tr, k):
    out = []
    for c, v in shp:
        L = len(v)
        W = znorm(sliding_window_view(X_tr[:, c, :], L, axis=-1))  # [n, P, L]
        d = np.sqrt(((W - v) ** 2).mean(-1))  # [n, P]
        best_p = d.argmin(1)
        best_d = d[np.arange(len(d)), best_p]
        acc = np.argsort(best_d, kind="stable")[:k]  # k distinct accounts
        out.append((c, znorm(W[acc, best_p[acc]].mean(0))))
    return out


class LearnedShapelets(nn.Module):
    def __init__(self, n_ch, lengths, per, alpha=-30.0):
        super().__init__()
        self.cfg = [(c, L) for c in range(n_ch) for L in lengths]
        self.S = nn.ParameterList([nn.Parameter(torch.randn(per, L) * 0.5) for _, L in self.cfg])
        self.alpha = alpha
        self.head = nn.Linear(len(self.cfg) * per, 1)

    def dists(self, x):
        feats = []
        for (c, L), S in zip(self.cfg, self.S):
            w = x[:, c].unfold(-1, L, 1)  # [b, P, L]
            mu, sd = w.mean(-1, keepdim=True), w.std(-1, unbiased=False, keepdim=True)
            w = torch.where(sd > 1e-8, (w - mu) / sd.clamp_min(1e-8), torch.zeros_like(w))
            d = ((w[:, None] - S[None, :, None]) ** 2).mean(-1).clamp_min(0).sqrt()  # [b, per, P]
            feats.append(-torch.logsumexp(self.alpha * -d, -1) / -self.alpha)  # soft-min
        return torch.cat(feats, 1)

    def forward(self, x):
        return self.head(self.dists(x)).squeeze(-1)


def train_learned(X, y, tr, seed, epochs=30, per=8, lengths=(4, 8, 12)):
    torch.manual_seed(seed)
    m = LearnedShapelets(X.shape[1], lengths, per)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    Xt, yt = torch.tensor(X[tr]), torch.tensor(y[tr], dtype=torch.float32)
    pw = torch.tensor((1 - y[tr]).sum() / y[tr].sum(), dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    for ep in range(epochs):
        perm = torch.randperm(len(tr))
        for i in range(0, len(tr), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            loss = lossf(m(Xt[b]), yt[b])
            loss.backward()
            opt.step()
    shp = [(c, S[j].detach().numpy().astype(np.float32)) for (c, L), S in zip(m.cfg, m.S) for j in range(per)]
    return shp


def main(args):
    d, X, y, tr, te = load(seed=args.seed)
    pk = pickle.load(open(f"results/stc_seed{args.seed}.pkl", "rb"))
    shp = [(s[3], np.asarray(s[6], dtype=np.float32)) for s in pk["shapelets"]]
    src = pk["src"]
    rng = np.random.default_rng(args.seed)
    res = {}
    res["none"] = evaluate(shp, X, y, tr, te, src, args.seed)
    print("none", res["none"], flush=True)
    for s in [0.1, 0.25, 0.5, 1.0]:
        noisy = [(c, (v + rng.normal(0, s, len(v))).astype(np.float32)) for c, v in shp]
        res[f"noise-{s}"] = evaluate(noisy, X, y, tr, te, src, args.seed)
        print(f"noise-{s}", res[f"noise-{s}"], flush=True)
    for k in [5, 20, 50]:
        ka = kanon(shp, X[tr], k)
        res[f"kanon-{k}"] = evaluate(ka, X, y, tr, te, src, args.seed)
        print(f"kanon-{k}", res[f"kanon-{k}"], flush=True)
    ls = train_learned(X, y, tr, args.seed)
    res["learned"] = evaluate(ls, X, y, tr, te, None, args.seed)
    # For learned shapelets there is no true source; report how close the nearest real account is.
    D = min_dists(ls, X[np.r_[tr, te]])
    res["learned"]["nearest_account_dist_median"] = float(np.median(D.min(1)))
    res["learned"]["frac_exact_copy(<1e-2)"] = float((D.min(1) < 1e-2).mean())
    print("learned", res["learned"], flush=True)
    json.dump(res, open(f"results/defenses_seed{args.seed}.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
