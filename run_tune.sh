#!/bin/bash
ROOT=$(cd "$(dirname "$0")" && pwd)
until [ -f $ROOT/logs/resumedone ]; do sleep 20; done
export PYTHONPATH=$ROOT/src DATA_NPZ=$ROOT/data/amlworld_6h.npz TORCH_THREADS=4
cd $ROOT/runs/amlworld
[ -f results/learned_tune_seed0.json ] || python3 $ROOT/src/learned_tune.py 0 > logs/learned_tune_seed0.log 2>&1
echo done > $ROOT/logs/tunedone
