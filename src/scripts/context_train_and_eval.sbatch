#!/bin/bash
#SBATCH --job-name=ctx_train_eval
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# TRAIN two bc-LSTMs (with and without wer25 filtering of the training data),
# then EVALUATE each on BOTH test sets.
#
# Two bugs made this impossible before:
#   1. train_context.py wrote every run to a fixed context_bclstm/ directory, so
#      each run silently overwrote the last -- the filtered and unfiltered models
#      never coexisted. FIXED: checkpoint dir now includes the tag.
#   2. train_context.py always trains before scoring, so "evaluate model X on
#      test set Y" retrained X. FIXED: eval_context.py loads a checkpoint and
#      trains nothing.
#
# Result: ONE trained model per training condition, each scored on BOTH test
# sets. Cells sharing a test set are directly comparable.

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv

set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
mkdir -p logs

CKPT_ROOT=/dcs/large/u5734759/checkpoints/mini
fail=0

# ---------------------------------------------------------------- TRAIN ----
# tag drives the checkpoint dir: context_bclstm_trainfull / _trainwer25
echo "############ TRAIN 1/2: unfiltered training data (9989 utts) ############"
$VENV/bin/python3.12 src/training/train_context.py \
    --config src/configs/ctx_trainfull_testfull.yaml \
    --tag _trainfull
[ $? -ne 0 ] && { echo "!!! TRAIN trainfull FAILED"; fail=1; }

echo "############ TRAIN 2/2: wer25-filtered training data (3539 utts) ############"
$VENV/bin/python3.12 src/training/train_context.py \
    --config src/configs/ctx_trainwer25_testfull.yaml \
    --tag _trainwer25
[ $? -ne 0 ] && { echo "!!! TRAIN trainwer25 FAILED"; fail=1; }

# ----------------------------------------------------------------- EVAL ----
for train_cond in trainfull trainwer25; do
  CKPT="${CKPT_ROOT}/context_bclstm_${train_cond}/best_context.pt"
  if [ ! -f "$CKPT" ]; then
    echo "!!! SKIP evals for ${train_cond} — checkpoint missing: $CKPT"
    fail=1
    continue
  fi
  for test_cond in testfull testwer25; do
    echo "############ EVAL: ${train_cond} on ${test_cond} ############"
    $VENV/bin/python3.12 src/evaluation/eval_context.py \
        --config "src/configs/eval_${test_cond}.yaml" \
        --checkpoint "$CKPT" \
        --tag "_${train_cond}_on_${test_cond}"
    [ $? -ne 0 ] && { echo "!!! EVAL ${train_cond}/${test_cond} FAILED"; fail=1; }
  done
done

echo "=========================================================="
[ $fail -eq 0 ] && echo "=== TRAIN + EVAL COMPLETE ===" \
                || echo "=== FINISHED WITH FAILURES — see !!! lines ==="
exit $fail
