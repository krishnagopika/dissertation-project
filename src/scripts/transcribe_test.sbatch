#!/bin/bash
#SBATCH --job-name=meld_transcribe_test
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=0-01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/slurm_test_%j.out
#SBATCH --error=logs/slurm_test_%j.err

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv

# Load HF token from .env
set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a

mkdir -p logs

# Smoke test: 10 utterances from train only
$VENV/bin/python3.12 src/preprocessing/transcribe_all.py \
    --config src/configs/mini.yaml \
    --splits train \
    --max_samples 10
