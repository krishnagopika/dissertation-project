#!/bin/bash
#SBATCH --job-name=train_context
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=01:00:00
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

# bc-LSTM dialogue-context over cached text+acoustic embeddings.
# Tiny model — trains in minutes. Focal loss, early stopping on dev WF1.
$VENV/bin/python3.12 src/training/train_context.py \
    --config src/configs/mini.yaml \
    --hidden_dim 512 --num_layers 2 --epochs 40 --patience 6
