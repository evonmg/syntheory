#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=63G
#SBATCH -t 24:00:00
#SBATCH -J probe_experiment_gaut5jgp
#SBATCH -e logs/probe/gaut5jgp/probe_experiment_gaut5jgp-%j.err
#SBATCH -o logs/probe/gaut5jgp/probe_experiment_gaut5jgp-%j.out

# Activate virtual environment
conda activate syntheory

# Run the script
python probe/main.py --sweep_id gaut5jgp --wandb_project music-theory-text-encoder