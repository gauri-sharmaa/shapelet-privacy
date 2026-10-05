#!/bin/bash
export PYTHONPATH=src
until [ -f logs/alldone ]; do sleep 30; done
for s in 0 1 2; do python3 src/rdst_attack.py --seed $s > logs/rdst_seed$s.log 2>&1; done
python3 src/rdst_attack.py --seed 0 --norm-only > logs/rdst_normonly_seed0.log 2>&1
python3 src/make_report.py > logs/report.log 2>&1
echo RDSTDONE > logs/rdstdone
