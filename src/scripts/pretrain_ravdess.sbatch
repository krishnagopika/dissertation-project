#!/bin/bash
#SBATCH --job-name=ravdess_pretrain
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=0-00:30:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv

set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a

mkdir -p logs

# Phase A: 8-class acoustic emotion classifier on RAVDESS cached embeddings.
# Saves best backbone weights (matching SumFusion/GatedFusion/CrossModalGating
# acoustic_proj layout) for Phase B transfer in train_fusion.py.
$VENV/bin/python3.12 src/training/pretrain_ravdess.py \
    --config src/configs/mini.yaml
