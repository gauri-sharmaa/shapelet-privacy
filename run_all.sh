#!/bin/bash
# Full pipeline after data prep. Logs in logs/, results in results/.
export PYTHONPATH=src
python3 src/defenses.py --seed 0 > logs/defenses_seed0.log 2>&1
python3 src/rdst_attack.py --seed 0 > logs/rdst_seed0.log 2>&1
for s in 1 2; do
  python3 src/baselines.py $s > logs/baselines_seed$s.log 2>&1
  python3 src/shapelet_attack.py --seed $s > logs/stc_seed$s.log 2>&1
  python3 src/defenses.py --seed $s > logs/defenses_seed$s.log 2>&1
done
python3 src/sweep_k.py 0 > logs/sweep.log 2>&1
python3 src/make_report.py > logs/report.log 2>&1
echo ALLDONE > logs/alldone
