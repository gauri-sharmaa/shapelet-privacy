"""Build per-account weekly multivariate series from SAML-D.

Output: data/samld_weekly.npz with
  X      float32 [n_accounts, n_channels, n_weeks]  (log1p-scaled)
  y      int8    [n_accounts]  1 if the account sent or received any laundering txn
  acct   int64   [n_accounts]
  ltype  str     [n_accounts]  most frequent suspicious typology (or "Normal")
  static float32 [n_accounts, n_static]  aggregate features for GBDT baseline
"""
import argparse
import numpy as np
import pandas as pd

CHANNELS = ["out_amt", "in_amt", "out_cnt", "in_cnt", "xborder_out_amt", "cash_amt"]


def main(ratio: float, seed: int, out: str):
    df = pd.read_csv("data/SAML-D.csv", usecols=[
        "Date", "Sender_account", "Receiver_account", "Amount",
        "Sender_bank_location", "Receiver_bank_location", "Payment_type",
        "Is_laundering", "Laundering_type"])
    df["week"] = ((pd.to_datetime(df.Date) - pd.Timestamp("2022-10-07")).dt.days // 7).astype(int)
    n_weeks = df.week.max() + 1

    # Account labels: involved in any laundering transaction on either side.
    sus = df[df.Is_laundering == 1]
    sus_accts = pd.Index(pd.concat([sus.Sender_account, sus.Receiver_account]).unique())
    typ = (pd.concat([sus[["Sender_account", "Laundering_type"]].rename(columns={"Sender_account": "a"}),
                      sus[["Receiver_account", "Laundering_type"]].rename(columns={"Receiver_account": "a"})])
           .groupby("a").Laundering_type.agg(lambda s: s.value_counts().index[0]))

    # Sample normal accounts matched on activity (number of active weeks) to the suspicious ones,
    # so detection cannot rely on "suspicious accounts are simply more active".
    aw = (pd.concat([df[["Sender_account", "week"]].rename(columns={"Sender_account": "a"}),
                     df[["Receiver_account", "week"]].rename(columns={"Receiver_account": "a"})])
          .groupby("a").week.nunique())
    is_sus = aw.index.isin(sus_accts)
    rng = np.random.default_rng(seed)
    want = (aw[is_sus].value_counts() * ratio).round().astype(int)
    pools = aw[~is_sus]
    normal = []
    for w, k in want.items():
        cand = pools[pools == w].index.to_numpy()
        normal.append(rng.choice(cand, size=min(k, len(cand)), replace=False))
    normal = np.concatenate(normal)
    accts = np.concatenate([sus_accts.to_numpy(), normal])
    idx = pd.Index(accts)

    X = np.zeros((len(accts), len(CHANNELS), n_weeks), dtype=np.float64)
    o = df[df.Sender_account.isin(idx)]
    i = df[df.Receiver_account.isin(idx)]
    oi, ii = idx.get_indexer(o.Sender_account), idx.get_indexer(i.Receiver_account)
    np.add.at(X, (oi, 0, o.week.to_numpy()), o.Amount.to_numpy())
    np.add.at(X, (ii, 1, i.week.to_numpy()), i.Amount.to_numpy())
    np.add.at(X, (oi, 2, o.week.to_numpy()), 1)
    np.add.at(X, (ii, 3, i.week.to_numpy()), 1)
    xb = (o.Sender_bank_location != o.Receiver_bank_location).to_numpy()
    np.add.at(X, (oi[xb], 4, o.week.to_numpy()[xb]), o.Amount.to_numpy()[xb])
    cash_o = o.Payment_type.str.contains("Cash").to_numpy()
    cash_i = i.Payment_type.str.contains("Cash").to_numpy()
    np.add.at(X, (oi[cash_o], 5, o.week.to_numpy()[cash_o]), o.Amount.to_numpy()[cash_o])
    np.add.at(X, (ii[cash_i], 5, i.week.to_numpy()[cash_i]), i.Amount.to_numpy()[cash_i])
    X = np.log1p(X).astype(np.float32)

    # Static aggregates (GBDT baseline, in the spirit of Tertychnyi et al. 2022).
    stats = [X.mean(-1), X.std(-1), X.max(-1), (X > 0).mean(-1)]
    pt_o = pd.crosstab(oi, o.Payment_type).reindex(range(len(accts)), fill_value=0)
    pt_i = pd.crosstab(ii, i.Payment_type).reindex(range(len(accts)), fill_value=0).add_prefix("in_")
    n_cp_o = o.groupby(oi).Receiver_account.nunique().reindex(range(len(accts)), fill_value=0)
    n_cp_i = i.groupby(ii).Sender_account.nunique().reindex(range(len(accts)), fill_value=0)
    static = np.concatenate(stats + [np.log1p(pt_o.to_numpy()), np.log1p(pt_i.to_numpy()),
                                     np.log1p(np.c_[n_cp_o.to_numpy(), n_cp_i.to_numpy()])], axis=1)

    y = np.isin(accts, sus_accts).astype(np.int8)
    ltype = np.array([typ.get(a, "Normal") for a in accts])
    np.savez_compressed(out, X=X, y=y, acct=accts, ltype=ltype, static=static.astype(np.float32),
                        channels=np.array(CHANNELS))
    print(f"accounts={len(accts)} suspicious={y.sum()} ({y.mean():.3%}) weeks={n_weeks} static={static.shape[1]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ratio", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/samld_weekly.npz")
    a = p.parse_args()
    main(a.ratio, a.seed, a.out)
