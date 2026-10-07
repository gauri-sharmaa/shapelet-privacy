"""Generic per-account multichannel series builder, shared by both datasets and by attacker views.

Input: a normalised transaction frame with columns
  t (int time bin), snd, rcv (account ids), amount (float), xb (bool: crosses border/bank), cash (bool), laund (0/1)
Channels: out_amt, in_amt, out_cnt, in_cnt, xborder_out_amt, cash_amt  (log1p-scaled)
"""
import os
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHANNELS = ["out_amt", "in_amt", "out_cnt", "in_cnt", "xborder_out_amt", "cash_amt"]


def build(tx, accts, n_bins, keep_p=1.0, amt_noise=0.0, round_to=0.0, seed=0):
    """Aggregate transactions into X [n_accts, 6, n_bins].

    keep_p     each transaction independently visible with this probability (partial view)
    amt_noise  multiplicative log-normal noise sigma on each visible amount (FX, fees, rounding)
    round_to   round each visible amount to this unit (0 = off)
    """
    rng = np.random.default_rng(seed)
    if keep_p < 1.0:
        tx = tx[rng.random(len(tx)) < keep_p]
    amt = tx.amount.to_numpy().astype(np.float64)
    if amt_noise > 0:
        amt = amt * np.exp(rng.normal(0, amt_noise, len(amt)))
    if round_to > 0:
        amt = np.round(amt / round_to) * round_to
    idx = pd.Index(accts)
    t = tx.t.to_numpy()
    X = np.zeros((len(accts), len(CHANNELS), n_bins), dtype=np.float64)
    oi, ii = idx.get_indexer(tx.snd), idx.get_indexer(tx.rcv)
    om, im = oi >= 0, ii >= 0
    np.add.at(X, (oi[om], 0, t[om]), amt[om])
    np.add.at(X, (ii[im], 1, t[im]), amt[im])
    np.add.at(X, (oi[om], 2, t[om]), 1)
    np.add.at(X, (ii[im], 3, t[im]), 1)
    xb, cash = tx.xb.to_numpy(), tx.cash.to_numpy()
    m = om & xb
    np.add.at(X, (oi[m], 4, t[m]), amt[m])
    m = om & cash
    np.add.at(X, (oi[m], 5, t[m]), amt[m])
    m = im & cash
    np.add.at(X, (ii[m], 5, t[m]), amt[m])
    return np.log1p(X).astype(np.float32)


def samld_tx(path=os.path.join(_ROOT, "data/SAML-D.csv")):
    df = pd.read_csv(path, usecols=["Date", "Sender_account", "Receiver_account", "Amount", "Sender_bank_location",
                                    "Receiver_bank_location", "Payment_type", "Is_laundering"])
    t = ((pd.to_datetime(df.Date) - pd.Timestamp("2022-10-07")).dt.days // 7).astype(int)
    return pd.DataFrame({"t": t, "snd": df.Sender_account, "rcv": df.Receiver_account, "amount": df.Amount,
                         "xb": df.Sender_bank_location != df.Receiver_bank_location,
                         "cash": df.Payment_type.str.contains("Cash"), "laund": df.Is_laundering}), int(t.max() + 1)


def amlworld_tx(path=os.path.join(_ROOT, "data/HI-Small_Trans.csv"), bin_hours=6, days=10):
    df = pd.read_csv(path)
    df.columns = ["ts", "from_bank", "from_acct", "to_bank", "to_acct", "amt_recv", "cur_recv", "amt_paid",
                  "cur_paid", "fmt", "laund"]
    ts = pd.to_datetime(df.ts, format="%Y/%m/%d %H:%M")
    t = ((ts - pd.Timestamp("2022-09-01")).dt.total_seconds() // (3600 * bin_hours)).astype(int)
    n_bins = days * 24 // bin_hours
    keep = t < n_bins
    # Account ids are only unique within a bank.
    snd = df.from_bank.astype(str) + ":" + df.from_acct.astype(str)
    rcv = df.to_bank.astype(str) + ":" + df.to_acct.astype(str)
    tx = pd.DataFrame({"t": t, "snd": snd, "rcv": rcv, "amount": df.amt_paid, "xb": df.from_bank != df.to_bank,
                       "cash": df.fmt == "Cash", "laund": df.laund})
    return tx[keep].reset_index(drop=True), n_bins


def berka_tx(path=os.path.join(_ROOT, "data/berka/trans.csv")):
    tx = pd.read_csv(path, sep=";", low_memory=False)
    tx = tx[tx.k_symbol.fillna("").str.strip() != "UVER"]
    date = pd.to_datetime(tx.date.astype(str), format="%y%m%d")
    t = ((date.dt.year - 1993) * 12 + date.dt.month - 1).astype(int)
    inc = tx.type == "PRIJEM"
    acct = tx.account_id.astype(str)
    return pd.DataFrame({"t": t, "snd": np.where(inc, "ext", acct), "rcv": np.where(inc, acct, "ext"),
                         "amount": tx.amount.astype(float),
                         "xb": tx.operation.isin(["PREVOD NA UCET", "PREVOD Z UCTU"]).to_numpy(),
                         "cash": tx.operation.isin(["VYBER", "VKLAD"]).to_numpy(), "laund": 0}), 72
