#!/bin/bash
# Same pipeline on AMLworld HI-Small (6-hour bins). Outputs in runs/amlworld/results.
ROOT=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$ROOT/src DATA_NPZ=$ROOT/data/amlworld_6h.npz
cd $ROOT/runs/amlworld
for s in 0 1 2; do
  python3 $ROOT/src/baselines.py $s > logs/baselines_seed$s.log 2>&1
  python3 $ROOT/src/shapelet_attack.py --seed $s > logs/stc_seed$s.log 2>&1
  python3 $ROOT/src/defenses.py --seed $s > logs/defenses_seed$s.log 2>&1
  python3 $ROOT/src/rdst_attack.py --seed $s > logs/rdst_seed$s.log 2>&1
  python3 $ROOT/src/rdst_attack.py --seed $s --norm-only > logs/rdst_normonly_seed$s.log 2>&1
done
python3 $ROOT/src/make_report.py > logs/report.log 2>&1
echo done > logs/alldone
