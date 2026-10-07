#!/bin/bash
export PYTHONPATH=src
[ -f results/rdst_extra_seed0.json ] || python3 src/rdst_extra.py --seed 0 --scale > logs/rdst_extra_seed0.log 2>&1
for s in 1 2; do [ -f results/rdst_extra_seed$s.json ] || python3 src/rdst_extra.py --seed $s > logs/rdst_extra_seed$s.log 2>&1; done
echo done > logs/extradone
