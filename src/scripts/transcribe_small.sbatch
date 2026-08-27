#!/bin/bash
#SBATCH --job-name=meld_transcribe_small
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --account=wmlg
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv

# Force fully-offline model loading on the compute node — Voxtral-Small
# weights (~48 GB) MUST be pre-downloaded on the login node with
# `huggingface-cli download mistralai/Voxtral-Small-24B-2507` before
# submitting this job. If weights are missing, the job will fail fast now
# instead of hanging silently for 48 h.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a

mkdir -p logs

# Pass 1 only: transcription with Voxtral-Small-24B
# Embeddings skipped — same Whisper encoder as Mini, reuse existing embeddings
# Output: /dcs/large/u5734759/data/meld_transcripts_small/{split}_transcripts.json
$VENV/bin/python3.12 src/preprocessing/transcribe_all.py \
    --config src/configs/small.yaml \
    --skip_embeddings \
    --splits train dev test
