#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=63G
#SBATCH -t 24:00:00
#SBATCH -J probe_experiment_5my59kcw
#SBATCH -e logs/probe/5my59kcw/probe_experiment_5my59kcw-%j.err
#SBATCH -o logs/probe/5my59kcw/probe_experiment_5my59kcw-%j.out

# Activate virtual environment
conda activate syntheory

# Run the script
python probe/main.py --sweep_id 5my59kcw --wandb_project music-theory-text-encoder