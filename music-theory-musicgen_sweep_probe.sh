#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=63G
#SBATCH -t 24:00:00
#SBATCH -J probe_experiment_g86kf1i4
#SBATCH -e logs/probe/g86kf1i4/probe_experiment_g86kf1i4-%j.err
#SBATCH -o logs/probe/g86kf1i4/probe_experiment_g86kf1i4-%j.out

# Activate virtual environment
conda activate syntheory

# Run the script
python probe/main.py --sweep_id g86kf1i4 --wandb_project music-theory-musicgen