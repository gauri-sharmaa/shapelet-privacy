"""Learned shapelets trained with differential privacy (DP-SGD via Opacus).

Each shapelet bank is its own module so Opacus can compute per-sample gradients (functorch mode).
For each privacy budget epsilon (inf = non-private) we report detection PR-AUC (ExtraTrees on the learned
distance features, as for the other shapelet models), nearest-account distance of the released shapelets,
and the share of shapelets whose nearest account is a training member vs chance.
Usage: DATA_NPZ=... python3 src/dp_shapelets.py --seed 0 --dataset samld
"""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from common import load, min_dists, det_metrics
from shapelet_attack import fit_detector

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Bank(nn.Module):
    def __init__(self, c, L, per, alpha=-30.0):
        super().__init__()
        self.c, self.L, self.alpha = c, L, alpha
        self.S = nn.Parameter(torch.randn(per, L) * 0.5)

    def forward(self, x):
        w = x[:, self.c].unfold(-1, self.L, 1)
        mu, sd = w.mean(-1, keepdim=True), w.std(-1, unbiased=False, keepdim=True)
        w = torch.where(sd > 1e-8, (w - mu) / sd.clamp_min(1e-8), torch.zeros_like(w))
        d = ((w[:, None] - self.S[None, :, None]) ** 2).mean(-1).clamp_min(1e-12).sqrt()
        return -torch.logsumexp(self.alpha * -d, -1) / -self.alpha


class DPShapelets(nn.Module):
    def __init__(self, n_ch, lengths, per):
        super().__init__()
        self.banks = nn.ModuleList([Bank(c, L, per) for c in range(n_ch) for L in lengths])
        self.head = nn.Linear(len(self.banks) * per, 1)

    def forward(self, x):
        return self.head(torch.cat([b(x) for b in self.banks], 1)).squeeze(-1)


def train(X, y, tr, seed, eps, epochs, per, lengths, batch=256, delta=None, return_model=False):
    torch.manual_seed(seed)
    m = DPShapelets(X.shape[1], lengths, per).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    ds = torch.utils.data.TensorDataset(torch.tensor(X[tr]), torch.tensor(y[tr], dtype=torch.float32))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
    spent = float("inf")
    if np.isfinite(eps):
        pe = PrivacyEngine(accountant="rdp")
        m, opt, dl = pe.make_private_with_epsilon(module=m, optimizer=opt, data_loader=dl, target_epsilon=eps,
                                                  target_delta=delta or 1.0 / len(tr), epochs=epochs,
                                                  max_grad_norm=1.0, grad_sample_mode="functorch")
    pw = torch.tensor((1 - y[tr]).sum() / max(1, y[tr].sum()), dtype=torch.float32, device=DEV)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    for _ in range(epochs):
        for xb, yb in dl:
            if len(xb) == 0:
                continue
            opt.zero_grad()
            lossf(m(xb.to(DEV)), yb.to(DEV)).backward()
            opt.step()
    if np.isfinite(eps):
        spent = pe.get_epsilon(delta or 1.0 / len(tr))
    core = m._module if hasattr(m, "_module") else m
    shp = [(b.c, b.S[j].detach().cpu().numpy().astype(np.float32)) for b in core.banks for j in range(b.S.shape[0])]
    return (shp, spent, core) if return_model else (shp, spent)


def main(a):
    d, X, y, tr, te = load(seed=a.seed)
    lengths = (3, 5, 8) if X.shape[2] <= 46 else (4, 8, 12)
    out = {}
    for eps in [float("inf"), 8.0, 4.0, 1.0]:
        shp, spent = train(X, y, tr, a.seed, eps, a.epochs, a.per, lengths)
        F = min_dists(shp, X).T
        _, p = fit_detector(F[tr], y[tr], F[te], a.seed)
        D = F.T
        near = D.argmin(1)
        is_mem = np.zeros(len(y), bool)
        is_mem[tr] = True
        out[str(eps)] = {"epsilon_spent": spent, "detection": det_metrics(y[te], p),
                         "nearest_account_dist_median": float(np.median(D.min(1))),
                         "nearest_is_member": float(is_mem[near].mean()), "member_base_rate": float(is_mem.mean())}
        print(a.dataset, eps, out[str(eps)], flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump(out, open(f"results/dp_shapelets_seed{a.seed}.json", "w"), indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", default="samld")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--per", type=int, default=8)
    main(p.parse_args())
