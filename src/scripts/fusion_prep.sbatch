#!/bin/bash
#SBATCH --job-name=fusion_prep
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=04:00:00
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

# Produce the three probability files for late fusion.
# Each runs as its own process so GPU memory is released between stages.

# 1. Zero-shot audio — full DEV (test predictions already exist).
$VENV/bin/python3.12 src/evaluation/voxtral_zeroshot.py \
    --config src/configs/mini.yaml --splits dev

# 2. Text probabilities — XLM-R mini_aug (dev + test).
$VENV/bin/python3.12 src/evaluation/extract_text_probs.py \
    --config src/configs/mini.yaml \
    --checkpoint_path /dcs/large/u5734759/checkpoints/mini_aug/best_model.pt

# 3. Acoustic probabilities — encoder fine-tune (dev + test).
$VENV/bin/python3.12 src/evaluation/extract_acoustic_probs.py \
    --config src/configs/mini.yaml \
    --checkpoint_path /dcs/large/u5734759/checkpoints/mini/encoder_finetune/best_encoder_full.pt
