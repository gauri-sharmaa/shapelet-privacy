# shapelet-privacy

Privacy leakage of shapelet-based anti-money-laundering (AML) detectors, on SAML-D.
Target: ICAIF 2026 Workshop on Financial AI Security, Privacy and Safety (paper draft in `paper/draft.md`).

CPU-only; whole pipeline runs in ~2 h on 4 cores / 15 GB RAM.

## Setup

```bash
pip install pandas pyarrow numpy scikit-learn catboost lightgbm aeon matplotlib kaggle
pip install torch --index-url https://download.pytorch.org/whl/cpu
# Kaggle token in ~/.kaggle/access_token (or KAGGLE_API_TOKEN)
kaggle datasets download -d berkanoztas/synthetic-transaction-monitoring-dataset-aml -p data && unzip data/*.zip -d data
```

## Run

```bash
python3 src/prep_samld.py          # -> data/samld_weekly.npz (activity-matched accounts, weekly 6-channel series)
export PYTHONPATH=src
python3 src/baselines.py 0         # CatBoost, MiniRocket (+ confidence MIA)
python3 src/shapelet_attack.py --seed 0   # shapelet transform + re-ID + MIA
python3 src/defenses.py --seed 0   # noise, k-anonymous, learned shapelets
python3 src/rdst_attack.py --seed 0       # RDST + raw reconstruction attack
python3 src/rdst_attack.py --seed 0 --norm-only   # defense: normalised-only, stats withheld
python3 src/sweep_k.py 0           # leakage vs number of shapelets
python3 src/make_report.py         # -> results/RESULTS.md, results/privacy_utility.png
```

Or `./run_all.sh` then `./run_rdst_seeds.sh` (seeds 0–2).

## Files

- `src/prep_samld.py` — per-account weekly series (log1p out/in amount, out/in count, cross-border out amount, cash amount); positives = accounts in any laundering txn; normals sampled 3:1 matched on active weeks.
- `src/common.py` — splits, vectorised shapelet distances, detection/MIA/re-ID metrics.
- `src/shapelet_attack.py` — classic binary shapelet transform (own implementation; aeon's degenerates on 85%-zero series) and attacks.
- `src/rdst_attack.py` — aeon RDST; reconstructs raw windows from stored values × std + mean and matches them to accounts.
- `src/defenses.py` — Gaussian noise, k-anonymous shapelets, learned shapelets (PyTorch).
- `src/sweep_k.py`, `src/make_report.py`.
