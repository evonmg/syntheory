#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=63G
#SBATCH -t 24:00:00
#SBATCH -J probe_experiment_c4u2iwn5
#SBATCH -e logs/probe/c4u2iwn5/probe_experiment_c4u2iwn5-%j.err
#SBATCH -o logs/probe/c4u2iwn5/probe_experiment_c4u2iwn5-%j.out

# Activate virtual environment
conda activate syntheory

# Run the script
python probe/main.py --sweep_id c4u2iwn5 --wandb_project music-theory-musicgen