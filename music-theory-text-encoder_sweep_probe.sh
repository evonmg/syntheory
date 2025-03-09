#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=63G
#SBATCH -t 24:00:00
#SBATCH -J probe_experiment_62iu5p4p
#SBATCH -e logs/probe/62iu5p4p/probe_experiment_62iu5p4p-%j.err
#SBATCH -o logs/probe/62iu5p4p/probe_experiment_62iu5p4p-%j.out

# Activate virtual environment
conda activate syntheory

# Run the script
python probe/main.py --sweep_id 62iu5p4p --wandb_project music-theory-text-encoder