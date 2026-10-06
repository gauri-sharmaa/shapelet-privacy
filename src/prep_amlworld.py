"""AMLworld HI-Small -> per-account 6-hourly series (40 bins over Sep 1-10, 2022), same format as SAML-D.

Positives: accounts sending or receiving any laundering transaction. Normals sampled 3:1, matched on the
number of active bins (as for SAML-D).
"""
import argparse
import numpy as np
import pandas as pd
from series import build, amlworld_tx, CHANNELS


def main(ratio, seed, out):
    tx, n_bins = amlworld_tx()
    sus = tx[tx.laund == 1]
    sus_accts = pd.Index(pd.concat([sus.snd, sus.rcv]).unique())
    act = pd.concat([tx[["snd", "t"]].rename(columns={"snd": "a"}), tx[["rcv", "t"]].rename(columns={"rcv": "a"})])
    aw = act.groupby("a").t.nunique()
    is_sus = aw.index.isin(sus_accts)
    rng = np.random.default_rng(seed)
    want = (aw[is_sus].value_counts() * ratio).round().astype(int)
    pool = aw[~is_sus]
    normal = []
    for w, k in want.items():
        cand = pool[pool == w].index.to_numpy()
        normal.append(rng.choice(cand, size=min(k, len(cand)), replace=False))
    accts = np.concatenate([sus_accts.to_numpy(), np.concatenate(normal)])
    X = build(tx, accts, n_bins)
    stats = [X.mean(-1), X.std(-1), X.max(-1), (X > 0).mean(-1)]
    static = np.concatenate(stats, axis=1)
    y = np.isin(accts, sus_accts).astype(np.int8)
    np.savez_compressed(out, X=X, y=y, acct=accts.astype(str), ltype=np.where(y == 1, "Laundering", "Normal"),
                        static=static.astype(np.float32), channels=np.array(CHANNELS))
    act_n = ((X[:, 0] + X[:, 1]) > 0).sum(1)
    print(f"accounts={len(accts)} suspicious={y.sum()} bins={n_bins} active-bins median sus/normal "
          f"{np.median(act_n[y == 1])}/{np.median(act_n[y == 0])}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ratio", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/amlworld_6h.npz")
    a = p.parse_args()
    main(a.ratio, a.seed, a.out)
