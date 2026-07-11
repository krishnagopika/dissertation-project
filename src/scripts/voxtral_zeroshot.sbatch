#!/bin/bash
#SBATCH --job-name=voxtral_zeroshot
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv

# Load HF token from .env
set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a

mkdir -p logs

# Zero-shot Voxtral classification (no fine-tuning) — emotion + sentiment.
# Reports accuracy + weighted/macro F1 per task → results/mini/.
$VENV/bin/python3.12 src/evaluation/voxtral_zeroshot.py \
    --config src/configs/mini.yaml \
    --splits test