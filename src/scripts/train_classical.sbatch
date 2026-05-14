#!/bin/bash
#SBATCH --job-name=meld_classical
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=0-04:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Variant A: SVM + XGBoost on raw concatenated features
# Prereq: finetune.sbatch must have completed (checkpoints/mini/best_model.pt exists)
# Output: results/mini/classical_{svm,xgboost}_{emotion,sentiment}.json

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv

set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a

mkdir -p logs

$VENV/bin/python3.12 src/training/train_classical.py \
    --config src/configs/mini.yaml \
    --model both
