#!/bin/bash
#SBATCH -J shapelet-gpu
#SBATCH -A paceship-masbench
#SBATCH -q inferno
#SBATCH -N1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o logs/gpu-%j.out
# Submit from the repo folder:  sbatch gpu.sbatch     (runs on its own; safe to log out)
# Watch:   squeue -u $USER      and    tail -f logs/gpu-<jobid>.out
# Faster GPU (may queue longer): add  --gres=gpu:H100:1  (or A100 / L40s) on the sbatch command line.
cd $SLURM_SUBMIT_DIR
module load anaconda3
source ~/scratch/sp-venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv
./run_gpu.sh
git add -f results/*.json runs/*/results/*.json runs/*/logs/*.log logs/gpu-*.out 2>/dev/null
git commit -qm "GPU results (job $SLURM_JOB_ID)" && git push || echo "Push failed: push manually with  git push"
