#!/bin/bash
#SBATCH -J shapelet-lira
#SBATCH -A paceship-masbench
#SBATCH -q embers
#SBATCH -N1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH -t 6:00:00
#SBATCH -o logs/lira-%j.out
# CPU-only LiRA audit on Berka (free 'embers' tier; resumable if pre-empted: just sbatch again).
cd $SLURM_SUBMIT_DIR
module load anaconda3
source ~/scratch/sp-venv/bin/activate
cd runs/berka
PYTHONPATH=$SLURM_SUBMIT_DIR/src DATA_NPZ=$SLURM_SUBMIT_DIR/data/berka_monthly.npz python3 $SLURM_SUBMIT_DIR/src/lira.py --shadows 32 --k 300 && cp results/lira.json results/lira_final.json
cd $SLURM_SUBMIT_DIR
git add -f runs/berka/results/lira_final.json runs/berka/results/lira.json logs/lira-*.out
git commit -qm "LiRA results (job $SLURM_JOB_ID)" && git push || echo "Push failed: run  git push  manually"
