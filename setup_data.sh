#!/bin/bash
# Download all three datasets and build the account series. Needs a Kaggle token in ~/.kaggle/access_token.
set -e
ROOT=$(cd "$(dirname "$0")" && pwd); cd $ROOT
mkdir -p data/berka results logs runs/amlworld/results runs/amlworld/logs runs/berka/results runs/berka/logs
[ -f data/SAML-D.csv ] || { kaggle datasets download -d berkanoztas/synthetic-transaction-monitoring-dataset-aml -p data && unzip -o -q data/synthetic-transaction-monitoring-dataset-aml.zip -d data && rm data/synthetic-transaction-monitoring-dataset-aml.zip; }
[ -f data/HI-Small_Trans.csv ] || { curl -sSL -o data/hi_small.zip "https://huggingface.co/datasets/eexzzm/IBM-Transactions-for-Anti-Money-Laundering-HI-Small-Trans/resolve/main/HI-Small_Trans.csv.zip" && unzip -o -q data/hi_small.zip -d data && rm data/hi_small.zip; }
[ -f data/berka/trans.csv ] || { kaggle datasets download -d marceloventura/the-berka-dataset -p data/berka && unzip -o -q data/berka/*.zip -d data/berka && rm data/berka/*.zip; }
export PYTHONPATH=$ROOT/src
[ -f data/samld_weekly.npz ] || python3 src/prep_samld.py
[ -f data/amlworld_6h.npz ] || python3 src/prep_amlworld.py
[ -f data/berka_monthly.npz ] || python3 src/prep_berka.py
echo "data ready"
