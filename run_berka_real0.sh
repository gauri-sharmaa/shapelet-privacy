#!/bin/bash
ROOT=$(cd "$(dirname "$0")" && pwd)
until [ -f $ROOT/runs/berka/logs/alldone ]; do sleep 20; done
export PYTHONPATH=$ROOT/src DATA_NPZ=$ROOT/data/berka_monthly.npz
cd $ROOT/runs/berka
[ -f results/realistic_seed0.json ] || python3 $ROOT/src/realistic_attack.py --seed 0 > logs/realistic_seed0.log 2>&1
echo done > logs/realdone
