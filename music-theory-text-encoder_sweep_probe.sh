#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=63G
#SBATCH -t 24:00:00
#SBATCH -J probe_experiment_blxz4tmz
#SBATCH -e logs/probe/blxz4tmz/probe_experiment_blxz4tmz-%j.err
#SBATCH -o logs/probe/blxz4tmz/probe_experiment_blxz4tmz-%j.out

# Activate virtual environment
conda activate syntheory

# Run the script
python probe/main.py --sweep_id blxz4tmz --wandb_project music-theory-text-encoder