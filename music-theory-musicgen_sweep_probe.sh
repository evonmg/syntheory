#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=63G
#SBATCH -t 24:00:00
#SBATCH -J probe_experiment_ou16kkga
#SBATCH -e logs/probe/ou16kkga/probe_experiment_ou16kkga-%j.err
#SBATCH -o logs/probe/ou16kkga/probe_experiment_ou16kkga-%j.out

# Activate virtual environment
conda activate syntheory

# Run the script
python probe/main.py --sweep_id ou16kkga --wandb_project music-theory-musicgen