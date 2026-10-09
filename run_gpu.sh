#!/bin/bash
# GPU experiments: DP-SGD learned shapelets and prototype networks on all three datasets (3 seeds).
# Resumable: finished runs are skipped. Run on a GPU node:  ./run_gpu.sh
ROOT=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$ROOT/src
python3 -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
for ds in samld amlworld berka; do
  case $ds in
    samld)    export DATA_NPZ=$ROOT/data/samld_weekly.npz; DIR=$ROOT ;;
    amlworld) export DATA_NPZ=$ROOT/data/amlworld_6h.npz;  DIR=$ROOT/runs/amlworld ;;
    berka)    export DATA_NPZ=$ROOT/data/berka_monthly.npz; DIR=$ROOT/runs/berka ;;
  esac
  mkdir -p $DIR/results $DIR/logs && cd $DIR
  for s in 0 1 2; do
    [ -f results/protonet_seed$s.json ] || { echo "[$ds seed $s] prototype network"; python3 $ROOT/src/protonet.py --seed $s > logs/protonet_seed$s.log 2>&1 || tail -5 logs/protonet_seed$s.log; }
    [ -f results/dp_shapelets_seed$s.json ] || { echo "[$ds seed $s] DP-SGD shapelets"; python3 $ROOT/src/dp_shapelets.py --seed $s --dataset $ds > logs/dp_shapelets_seed$s.log 2>&1 || tail -5 logs/dp_shapelets_seed$s.log; }
  done
done
cd $ROOT && echo "ALL DONE. Now push results:  git add -f results/*.json runs/*/results/*.json && git commit -m 'GPU results' && git push"
