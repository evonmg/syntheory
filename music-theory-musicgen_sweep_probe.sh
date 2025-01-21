#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=63G
#SBATCH -t 24:00:00
#SBATCH -J probe_experiment_6g9xrc7v
#SBATCH -e logs/probe/6g9xrc7v/probe_experiment_6g9xrc7v-%j.err
#SBATCH -o logs/probe/6g9xrc7v/probe_experiment_6g9xrc7v-%j.out

# Activate virtual environment
conda activate syntheory

# Run the script
python probe/main.py --sweep_id 6g9xrc7v --wandb_project music-theory-musicgen