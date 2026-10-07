"""Berka (real, anonymised Czech bank, 1993-1998) -> per-account monthly series, same 6-channel format.

Berka has no laundering labels. Proxy task: does the account hold a loan? All loan-repayment transactions
(k_symbol UVER) are removed so the label cannot be read off directly. The privacy measurement does not
depend on the task; Berka tests whether *real* customers' histories are distinctive enough to be singled out.
"""
import numpy as np
import pandas as pd
from series import build, CHANNELS

tx = pd.read_csv("data/berka/trans.csv", sep=";", low_memory=False)
tx = tx[tx.k_symbol.fillna("").str.strip() != "UVER"]
date = pd.to_datetime(tx.date.astype(str), format="%y%m%d")
t = ((date.dt.year - 1993) * 12 + date.dt.month - 1).astype(int)
inc = tx.type == "PRIJEM"
acct = tx.account_id.astype(str)
frame = pd.DataFrame({
    "t": t,
    "snd": np.where(inc, "ext", acct),
    "rcv": np.where(inc, acct, "ext"),
    "amount": tx.amount.astype(float),
    "xb": tx.operation.isin(["PREVOD NA UCET", "PREVOD Z UCTU"]).to_numpy(),
    "cash": tx.operation.isin(["VYBER", "VKLAD"]).to_numpy(),
    "laund": 0,
})
accts = np.sort(tx.account_id.unique()).astype(str)
X = build(frame, accts, 72)
loans = set(pd.read_csv("data/berka/loan.csv", sep=";").account_id.astype(str))
y = np.isin(accts, list(loans)).astype(np.int8)
static = np.concatenate([X.mean(-1), X.std(-1), X.max(-1), (X > 0).mean(-1)], axis=1)
np.savez_compressed("data/berka_monthly.npz", X=X, y=y, acct=accts, ltype=np.where(y == 1, "Loan", "NoLoan"),
                    static=static.astype(np.float32), channels=np.array(CHANNELS))
print(f"accounts={len(accts)} positive(loan)={y.sum()} months=72 active-months median={np.median((X[:,0]+X[:,1]>0).sum(1))}")
