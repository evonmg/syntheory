#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=63G
#SBATCH -t 24:00:00
#SBATCH -J probe_experiment_fpaxcvrs
#SBATCH -e logs/probe/fpaxcvrs/probe_experiment_fpaxcvrs-%j.err
#SBATCH -o logs/probe/fpaxcvrs/probe_experiment_fpaxcvrs-%j.out

# Activate virtual environment
conda activate syntheory

# Run the script
python probe/main.py --sweep_id fpaxcvrs --wandb_project music-theory-text-encoder