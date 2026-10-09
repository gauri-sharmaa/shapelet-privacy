"""Prototype network (ProtoPNet-style, 1D) and a case-based (nearest-neighbour) explainer.

ProtoPNet explains a decision by "this part looks like that part of a training example": after training,
each latent prototype is *projected* onto the nearest latent patch of a training account of its class, and
the explanation shown is that account's input window. We release those windows and run the same
re-identification attack as for shapelets. We report accuracy before and after projection.

Case-based explainer: k-NN whose explanation is the nearest training account itself.
"""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier
from common import load, det_metrics
from rdst_attack import match

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RF = 9  # receptive field of two kernel-5 convolutions


class ProtoNet(nn.Module):
    def __init__(self, c, m_per_class=10, d=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(c, d, 5, padding=2), nn.ReLU(), nn.Conv1d(d, d, 5, padding=2), nn.ReLU())
        self.P = nn.Parameter(torch.rand(2 * m_per_class, d))
        self.cls = torch.arange(2 * m_per_class) // m_per_class  # prototype class (0 then 1)
        self.head = nn.Linear(2 * m_per_class, 1)

    def dists(self, x):
        z = self.enc(x).transpose(1, 2)  # [b, T, d]
        return ((z[:, None] - self.P[None, :, None]) ** 2).sum(-1)  # [b, m, T]

    def forward(self, x):
        dmin = self.dists(x).min(-1).values
        sim = torch.log((dmin + 1) / (dmin + 1e-4))
        return self.head(sim).squeeze(-1), dmin


def train(m, X, y, idx, epochs, lr=1e-3, head_only=False):
    params = m.head.parameters() if head_only else m.parameters()
    opt = torch.optim.Adam(params, lr=lr)
    Xt, yt = torch.tensor(X[idx]).to(DEV), torch.tensor(y[idx], dtype=torch.float32).to(DEV)
    pw = torch.tensor((1 - y[idx]).sum() / max(1, y[idx].sum()), dtype=torch.float32, device=DEV)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)
    cls = m.cls.to(DEV)
    for _ in range(epochs):
        perm = torch.randperm(len(idx), device=DEV)
        for i in range(0, len(idx), 256):
            b = perm[i:i + 256]
            opt.zero_grad()
            logit, dmin = m(Xt[b])
            loss = bce(logit, yt[b])
            if not head_only:  # cluster loss: each example close to some prototype of its own class
                own = (cls[None, :] == yt[b, None].long())
                loss = loss + 0.05 * torch.where(own, dmin, torch.full_like(dmin, 1e9)).min(1).values.mean()
            loss.backward()
            opt.step()


@torch.no_grad()
def predict(m, X, idx):
    out = []
    for i in range(0, len(idx), 1024):
        out.append(torch.sigmoid(m(torch.tensor(X[idx[i:i + 1024]]).to(DEV))[0]).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def project(m, X, y, tr):
    """Replace each prototype by its nearest latent patch among training accounts of its class."""
    src = []
    for j in range(m.P.shape[0]):
        c = int(m.cls[j])
        cand = tr[y[tr] == c]
        best = (np.inf, None, None, None)
        for i in range(0, len(cand), 1024):
            xb = torch.tensor(X[cand[i:i + 1024]]).to(DEV)
            z = m.enc(xb).transpose(1, 2)
            d = ((z - m.P[j]) ** 2).sum(-1)  # [b, T]
            v, flat = d.view(-1).min(0)
            if v.item() < best[0]:
                a, t = divmod(flat.item(), d.shape[1])
                best = (v.item(), cand[i + a], t, z[a, t].clone())
        m.P[j] = best[3]
        src.append((int(best[1]), int(best[2])))
    return src


def main(a):
    d, X, y, tr, te = load(seed=a.seed)
    torch.manual_seed(a.seed)
    m = ProtoNet(X.shape[1]).to(DEV)
    train(m, X, y, tr, a.epochs)
    before = det_metrics(y[te], predict(m, X, te))
    src = project(m, X, y, tr)
    train(m, X, y, tr, 5, head_only=True)  # standard last-layer step after projection
    after = det_metrics(y[te], predict(m, X, te))

    # Released explanations: the input window under each projected prototype's receptive field.
    T = X.shape[2]
    wins = []
    for acct, t in src:
        lo = min(max(0, t - RF // 2), T - RF)
        wins.append((X[acct, :, lo:lo + RF].astype(np.float64), RF, 1))
    pool = np.r_[tr, te]
    D = match(wins, X[pool])
    uniq = (D <= 1e-3).sum(1) == 1
    who = pool[D.argmin(1)]
    ok = uniq & (who == np.array([s[0] for s in src]))
    out = {"protonet": {"pr_auc_before_projection": before["pr_auc"], "pr_auc_after_projection": after["pr_auc"],
                        "n_prototypes": len(src), "frac_unique_customer": float(uniq.mean()),
                        "frac_correct_source": float(ok.mean()),
                        "n_customers_exposed": int(len(set(who[ok].tolist()))),
                        "n_exposed_positive": int(y[np.array(sorted(set(who[ok].tolist())), dtype=int)].sum()) if ok.any() else 0}}

    # Case-based explainer: k-NN on flattened series; the explanation is the nearest training account.
    knn = KNeighborsClassifier(n_neighbors=10, weights="distance").fit(X[tr].reshape(len(tr), -1), y[tr])
    p = knn.predict_proba(X[te].reshape(len(te), -1))[:, 1]
    out["knn_case_based"] = {"pr_auc": det_metrics(y[te], p)["pr_auc"],
                             "note": "each explanation is a full training account (identifiable by construction)"}
    print(json.dumps(out, indent=1), flush=True)
    os.makedirs("results", exist_ok=True)
    json.dump(out, open(f"results/protonet_seed{a.seed}.json", "w"), indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    main(p.parse_args())
