#!/bin/bash
#SBATCH --job-name=context_asr
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv

set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
mkdir -p logs

# 1. Encode Voxtral ASR transcripts with XLM-R → meld_text_embeddings_asr
#    (same checkpoint as the gold cache → clean gold-vs-ASR comparison)
$VENV/bin/python3.12 src/preprocessing/extract_text_embeddings_asr.py \
    --config src/configs/mini.yaml \
    --out_dir /dcs/large/u5734759/data/meld_text_embeddings_asr

# 2. Train the bc-LSTM on ASR-text ⊕ acoustic (mini_asr.yaml points text at the ASR cache)
$VENV/bin/python3.12 src/training/train_context.py \
    --config src/configs/mini_asr.yaml \
    --hidden_dim 256 --num_layers 1 --epochs 30 --patience 5
