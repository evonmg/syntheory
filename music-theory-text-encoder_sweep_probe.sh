#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=63G
#SBATCH -t 24:00:00
#SBATCH -J probe_experiment_kc4iz0xo
#SBATCH -e logs/probe/kc4iz0xo/probe_experiment_kc4iz0xo-%j.err
#SBATCH -o logs/probe/kc4iz0xo/probe_experiment_kc4iz0xo-%j.out

# Activate virtual environment
conda activate syntheory

# Run the script
python probe/main.py --sweep_id kc4iz0xo --wandb_project music-theory-text-encoder