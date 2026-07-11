#!/bin/bash
#SBATCH --job-name=ravdess_extract
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=0-01:30:00
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

# Voxtral encoder over all RAVDESS speech clips → cached embeddings.pt
# Prereq: RAVDESS Audio_Speech_Actors_01-24.zip extracted under
#   /dcs/large/u5734759/data/ravdess/Actor_01 ... Actor_24
$VENV/bin/python3.12 src/preprocessing/extract_ravdess_embeddings.py \
    --config src/configs/mini.yaml
