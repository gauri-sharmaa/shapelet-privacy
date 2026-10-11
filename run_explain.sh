#!/bin/bash
# Explanation-only leakage on all datasets (resumable).
ROOT=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$ROOT/src
run() { out=$1; shift; [ -f "$out" ] && return; "$@"; }
for spec in "berka:runs/berka:berka_monthly" "samld:.:samld_weekly" "amlworld:runs/amlworld:amlworld_6h"; do
  IFS=: read name dir npz <<< "$spec"
  cd $ROOT/$dir && mkdir -p logs results
  for s in 0 1 2; do
    run results/explain_leak_seed$s.json bash -c "DATA_NPZ=$ROOT/data/$npz.npz python3 $ROOT/src/explain_leak.py --seed $s > logs/explain_seed$s.log 2>&1"
  done
done
echo done > $ROOT/logs/explain_done
