#!/bin/bash
#SBATCH --job-name=combined_pretrain
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

# Joint 7-class acoustic training on RAVDESS + MELD (labels in MELD space).
# Validated on MELD dev, tested on MELD test. Saves a backbone that drops
# into MELD fusion's acoustic_proj (sum/gated/crossmodal).
#
# Prereqs:
#   - RAVDESS embeddings extracted (extract_ravdess_embeddings.sbatch)
#   - MELD acoustic embeddings extracted (transcribe.sbatch, Pass 2)
$VENV/bin/python3.12 src/training/pretrain_combined.py \
    --config src/configs/mini.yaml
