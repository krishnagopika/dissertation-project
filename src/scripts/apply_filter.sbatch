#!/bin/bash
#SBATCH --job-name=meld_apply_filter
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --account=wmlg
#SBATCH --time=0-00:15:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Fast — no GPU, no audio decode. Just reads the pre-computed metadata JSONs
# and writes keep-list JSON + summary JSON + per-utterance CSV per policy.
# Prereq: compute_filter_metadata.sbatch already run.

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export VENV=/dcs/large/u5734759/venv

set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a

mkdir -p logs

$VENV/bin/python3.12 src/preprocessing/apply_filter.py \
    --config src/configs/mini.yaml
