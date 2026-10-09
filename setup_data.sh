#!/bin/bash
# Download all three datasets and build the account series. Needs a Kaggle token in ~/.kaggle/access_token.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd); cd $ROOT
die() { echo "ERROR: $*" >&2; exit 1; }
command -v kaggle >/dev/null || die "kaggle not found: activate the environment and pip install -r requirements.txt"
case "$(command -v kaggle)" in *.local/bin*) die "using the old kaggle in ~/.local; activate the venv (source ~/scratch/sp-venv/bin/activate)";; esac
[ -n "${KAGGLE_API_TOKEN:-}" ] || [ -f ~/.kaggle/access_token ] || [ -f ~/.kaggle/kaggle.json ] || die "no Kaggle token: export KAGGLE_API_TOKEN=..."
mkdir -p data/berka results logs runs/amlworld/results runs/amlworld/logs runs/berka/results runs/berka/logs
[ -f data/SAML-D.csv ] || { kaggle datasets download -d berkanoztas/synthetic-transaction-monitoring-dataset-aml -p data && unzip -o -q data/synthetic-transaction-monitoring-dataset-aml.zip -d data && rm data/synthetic-transaction-monitoring-dataset-aml.zip; } || die "SAML-D download failed"
[ -f data/HI-Small_Trans.csv ] || { curl -sSL -o data/hi_small.zip "https://huggingface.co/datasets/eexzzm/IBM-Transactions-for-Anti-Money-Laundering-HI-Small-Trans/resolve/main/HI-Small_Trans.csv.zip" && unzip -o -q data/hi_small.zip -d data && rm data/hi_small.zip; } || die "AMLworld download failed"
[ -f data/berka/trans.csv ] || { kaggle datasets download -d marceloventura/the-berka-dataset -p data/berka && unzip -o -q data/berka/*.zip -d data/berka && rm data/berka/*.zip; } || die "Berka download failed"
export PYTHONPATH=$ROOT/src
[ -f data/samld_weekly.npz ] || python3 src/prep_samld.py
[ -f data/amlworld_6h.npz ] || python3 src/prep_amlworld.py
[ -f data/berka_monthly.npz ] || python3 src/prep_berka.py
echo "data ready"
