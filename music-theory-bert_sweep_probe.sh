#!/bin/bash
#SBATCH -p gpu --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=63G
#SBATCH -t 24:00:00
#SBATCH -J probe_experiment_nfcpv971
#SBATCH -e logs/probe/nfcpv971/probe_experiment_nfcpv971-%j.err
#SBATCH -o logs/probe/nfcpv971/probe_experiment_nfcpv971-%j.out

# Activate virtual environment
conda activate syntheory

# Run the script
python probe/main.py --sweep_id nfcpv971 --wandb_project music-theory-bert