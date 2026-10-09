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
# LiRA membership audit on Berka (CPU-bound; runs here so it is not lost to session restarts)
cd $ROOT/runs/berka && export DATA_NPZ=$ROOT/data/berka_monthly.npz
[ -f results/lira_final.json ] || { echo "[berka] LiRA audit (32 shadow models)"; python3 $ROOT/src/lira.py --shadows 32 --k 300 > logs/lira.log 2>&1 && cp results/lira.json results/lira_final.json || tail -5 logs/lira.log; }
cd $ROOT && echo "ALL DONE. Now push results:  git add -f results/*.json runs/*/results/*.json && git commit -m 'GPU results' && git push"
