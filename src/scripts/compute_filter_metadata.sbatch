#!/bin/bash
#SBATCH --job-name=meld_filter_meta
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=0-02:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Compute per-utterance Silero-VAD speech ratio + Voxtral/Whisper WER for
# every MELD utterance across all splits. One-off, ~30 min per split.
#
# Prereqs:
#   - transcribe.sbatch complete (Voxtral transcripts exist)
#   - Optionally, transcribe_whisper.sbatch complete for Whisper WER too

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv

set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a

mkdir -p logs

$VENV/bin/python3.12 src/preprocessing/compute_filter_metadata.py \
    --config src/configs/mini.yaml
