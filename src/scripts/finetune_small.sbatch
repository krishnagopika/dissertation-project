#!/bin/bash
#SBATCH --job-name=xlmr_finetune_small
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=0-06:00:00
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

# Phase 1: fine-tune XLM-RoBERTa on Voxtral-SMALL ASR transcripts.
# Only the XLM-R text encoder is loaded — Voxtral-Small is NOT loaded here.
# Uses gpu:1 (Voxtral-Small only needed for transcription, done by
# transcribe_small.sbatch, which produces meld_transcripts_small/).
#
# Prereq: transcribe_small.sbatch complete (Voxtral-Small transcripts exist)
# Output: /dcs/large/u5734759/checkpoints/small/best_model.pt
$VENV/bin/python3.12 src/training/finetune.py \
    --config src/configs/small.yaml
