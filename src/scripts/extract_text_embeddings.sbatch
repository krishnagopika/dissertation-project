#!/bin/bash
#SBATCH --job-name=extract_text_emb
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=0-01:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Extracts 768-dim XLM-RoBERTa CLS vectors for all three MELD splits
# using gold text from the dataset CSVs and the Phase 1 checkpoint.
# Output: /dcs/large/u5734759/data/meld_text_embeddings/{split}_text_embeddings.pt
#
# Run this AFTER Phase 1 (finetune.sbatch) completes.
# Run this BEFORE Phase 2 (train_fusion.sbatch).

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv

set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a

mkdir -p logs

$VENV/bin/python3.12 src/preprocessing/extract_text_embeddings.py \
    --config src/configs/mini.yaml
