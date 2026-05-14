#!/bin/bash
#SBATCH --job-name=meld_wer
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --account=wmlg
#SBATCH --time=0-00:30:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Compute WER between MELD gold transcripts and Voxtral ASR output.
# Prereq: transcribe.sbatch must have completed (meld_transcripts/ must exist)
# Output: results/mini/wer_analysis.json
# No GPU required — jiwer is CPU-only.

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv

set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a

mkdir -p logs

$VENV/bin/python3.12 src/evaluation/compute_wer.py \
    --config src/configs/mini.yaml
