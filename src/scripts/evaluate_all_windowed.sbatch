#!/bin/bash
#SBATCH --job-name=eval_all_windowed
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=0-04:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Re-score every checkpoint on the FULL test set with the training-time
# context window re-applied.
#
# WHY THIS RE-RUN EXISTS
#   score_dialogue() built its DialogueDataset without a context window, while
#   training wraps it in WindowedDialogueDataset whenever --context_window >= 0.
#   The bc-LSTM checkpoints do not store the window, so it was silently lost and
#   every k0/k1/k2/k4 run was scored on whole dialogues -- a different model.
#   That is why only the `full` and `hidden_*` cells previously agreed with the
#   figures written at training time, and why 73 of 93 bc-LSTM rows in
#   test_all_v5.json disagree with results_new/bclstm/test_results_*.json.
#
# VALIDATION GATE
#   bc-LSTM is the only family with authoritative test metrics written at
#   training time. If this pass reproduces those, the fix is correct and the
#   ctxfusion/fusion numbers -- which exist on NO other test source -- can be
#   trusted. If it does not, nothing here goes in the dissertation.
#   Check with:  python3 src/evaluation/check_windowed_eval.py
#
# Writes results_new/test_all/test_all_v7_windowed_attnraw.json (refuses to
# clobber). v5 and v6 stay on disk for comparison.
#
# SECOND FIX IN THIS PASS: attnraw_ runs were scored on the default
# masked-mean cache instead of meld_pooled_raw/attention. Same width
# (1280), different vector, so it loaded silently and scored wrong.

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv

set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
cd /dcs/pg25/u5734759/dissertation-project
mkdir -p logs results_new/test_all

echo "=== $(date) | node $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

$VENV/bin/python3.12 -u -m src.evaluation.evaluate_all \
    --families fusion bclstm ctxfusion \
    --out test_all_v7_windowed_attnraw.json
rc=$?

echo "=== evaluate_all exit $rc ==="
if [ $rc -ne 0 ]; then exit $rc; fi

# Gate: compare against the training-time bc-LSTM figures before trusting it.
$VENV/bin/python3.12 -u src/evaluation/check_windowed_eval.py \
    --table test_all_v7_windowed_attnraw.json
