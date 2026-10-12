#!/bin/bash
# Reviewer-fix experiments: RDST mitigations on all datasets (resumable).
ROOT=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$ROOT/src
run() { out=$1; shift; [ -f "$out" ] && return; "$@"; }
for spec in "runs/berka:berka_monthly:0 1 2" ".:samld_weekly:0 1 2" "runs/amlworld:amlworld_6h:0"; do
  IFS=: read dir npz seeds <<< "$spec"
  cd $ROOT/$dir && mkdir -p logs results
  for s in $seeds; do
    run results/rdst_defenses_seed$s.json bash -c "DATA_NPZ=$ROOT/data/$npz.npz python3 $ROOT/src/rdst_defenses.py --seed $s > logs/rdst_defenses_seed$s.log 2>&1"
  done
done
echo done > $ROOT/logs/fix_done
