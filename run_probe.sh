#!/bin/bash
export PYTHONPATH=src
until [ -f logs/realisticdone ]; do sleep 30; done
for s in 0; do python3 src/learned_probe.py --seed $s > logs/learned_probe_seed$s.log 2>&1; done
echo done > logs/probedone
