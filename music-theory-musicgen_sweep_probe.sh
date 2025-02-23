#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=63G
#SBATCH -t 24:00:00
#SBATCH -J probe_experiment_lpd542bh
#SBATCH -e logs/probe/lpd542bh/probe_experiment_lpd542bh-%j.err
#SBATCH -o logs/probe/lpd542bh/probe_experiment_lpd542bh-%j.out

# Activate virtual environment
conda activate syntheory

# Run the script
python probe/main.py --sweep_id lpd542bh --wandb_project music-theory-musicgen