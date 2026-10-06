#!/bin/bash
ROOT=$(cd "$(dirname "$0")" && pwd)
until [ -f $ROOT/logs/realisticdone ]; do sleep 30; done
export PYTHONPATH=$ROOT/src DATA_NPZ=$ROOT/data/amlworld_6h.npz TORCH_THREADS=2
cd $ROOT/runs/amlworld && python3 $ROOT/src/defenses.py --seed 0 > logs/defenses_seed0.log 2>&1
echo done > logs/defdone
