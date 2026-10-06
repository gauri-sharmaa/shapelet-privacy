#!/bin/bash
# Resumable: each step skipped if its result file exists (container restarts kill background jobs).
ROOT=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$ROOT/src TORCH_THREADS=4
run() { out=$1; shift; [ -f "$out" ] && return; "$@"; }
cd $ROOT
run results/learned_probe_seed0.json bash -c "python3 src/learned_probe.py --seed 0 > logs/learned_probe_seed0.log 2>&1"
export DATA_NPZ=$ROOT/data/amlworld_6h.npz
cd $ROOT/runs/amlworld
run results/rdst_normonly_attack_seed0.json bash -c "python3 $ROOT/src/rdst_attack.py --seed 0 --norm-only > logs/rdst_normonly_seed0.log 2>&1"
run results/defenses_seed0.json bash -c "python3 $ROOT/src/defenses.py --seed 0 > logs/defenses_seed0.log 2>&1"
for s in 1 2; do
  run results/baselines_seed$s.json bash -c "python3 $ROOT/src/baselines.py $s > logs/baselines_seed$s.log 2>&1"
  run results/stc_attack_seed$s.json bash -c "python3 $ROOT/src/shapelet_attack.py --seed $s > logs/stc_seed$s.log 2>&1"
  run results/rdst_attack_seed$s.json bash -c "python3 $ROOT/src/rdst_attack.py --seed $s > logs/rdst_seed$s.log 2>&1"
  run results/rdst_normonly_attack_seed$s.json bash -c "python3 $ROOT/src/rdst_attack.py --seed $s --norm-only > logs/rdst_normonly_seed$s.log 2>&1"
done
python3 $ROOT/src/make_report.py > logs/report.log 2>&1
echo done > $ROOT/logs/resumedone
