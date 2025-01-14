#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=63G
#SBATCH -t 24:00:00
#SBATCH -J probe_experiment_3z52iavj
#SBATCH -e logs/probe/3z52iavj/probe_experiment_3z52iavj-%j.err
#SBATCH -o logs/probe/3z52iavj/probe_experiment_3z52iavj-%j.out

# Activate virtual environment
conda activate syntheory

# Run the script
python probe/main.py --sweep_id 3z52iavj --wandb_project music-theory-text-encoder