#!/bin/bash
#SBATCH --job-name=pad_diag
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1 --cpus-per-task=8 --gres=gpu:1 --account=wmlg
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
source /etc/profile.d/modules.sh; source /etc/profile.d/conda.sh
export HF_HOME=/dcs/large/u5734759/hf_cache VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_ALLOW_INSECURE_SERIALIZATION=1
set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
$VENV/bin/python3.12 src/preprocessing/_pad_diag.py
