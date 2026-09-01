#!/bin/bash
#SBATCH --job-name=eval_all
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=96G --gres=gpu:1
#SBATCH --account=wmlg --time=03:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
# Score every trained checkpoint on the FULL test set through ONE code path.
# Until now every number reported has been DEV, and each family scored test
# differently or not at all -- three paths means three chances for the metric,
# the subset or the label mapping to drift.
set -u
source /etc/profile.d/modules.sh; source /etc/profile.d/conda.sh
export HF_HOME=/dcs/large/u5734759/hf_cache VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
cd /dcs/pg25/u5734759/dissertation-project
mkdir -p logs results_new/test_all

# OUT is overridable so a re-scoring pass does not overwrite an earlier table.
# The previous pass covered 82 checkpoints; more grids have completed since, so
# a fresh run scores more and should be written somewhere new until compared:
#   sbatch --export=ALL,OUT=test_all_v2.json src/scripts/evaluate_all.sbatch
OUT="${OUT:-test_all.json}"

$VENV/bin/python3.12 src/evaluation/evaluate_all.py --out "$OUT"
