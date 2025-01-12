#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=63G
#SBATCH -t 24:00:00
#SBATCH -J probe_experiment_c8mw35tf
#SBATCH -e logs/probe/c8mw35tf/probe_experiment_c8mw35tf-%j.err
#SBATCH -o logs/probe/c8mw35tf/probe_experiment_c8mw35tf-%j.out

# Activate virtual environment
conda activate syntheory

# Run the script
python probe/main.py --sweep_id c8mw35tf --wandb_project music-theory-musicgen