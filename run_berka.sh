#!/bin/bash
# Real-data replication on Berka (resumable).
ROOT=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$ROOT/src DATA_NPZ=$ROOT/data/berka_monthly.npz TORCH_THREADS=2
cd $ROOT/runs/berka
run() { out=$1; shift; [ -f "$out" ] && return; "$@"; }
for s in 0 1 2; do
  run results/baselines_seed$s.json bash -c "python3 $ROOT/src/baselines.py $s > logs/baselines_seed$s.log 2>&1"
  run results/stc_attack_seed$s.json bash -c "python3 $ROOT/src/shapelet_attack.py --seed $s --n-eval 2250 > logs/stc_seed$s.log 2>&1"
  run results/rdst_attack_seed$s.json bash -c "python3 $ROOT/src/rdst_attack.py --seed $s > logs/rdst_seed$s.log 2>&1"
  run results/rdst_normonly_attack_seed$s.json bash -c "python3 $ROOT/src/rdst_attack.py --seed $s --norm-only > logs/rdst_normonly_seed$s.log 2>&1"
  run results/defenses_seed$s.json bash -c "python3 $ROOT/src/defenses.py --seed $s > logs/defenses_seed$s.log 2>&1"
  run results/realistic_seed$s.json bash -c "python3 $ROOT/src/realistic_attack.py --seed $s --n-rdst 300 > logs/realistic_seed$s.log 2>&1"
done
echo done > logs/alldone
