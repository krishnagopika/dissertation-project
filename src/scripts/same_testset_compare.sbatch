#!/bin/bash
#SBATCH --job-name=same_testset_cmp
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Same-test-set comparison.
#
# Problem this fixes: mini.yaml ships with filtering.enabled=true, so a model
# trained on filtered data and one trained on full data were being scored on
# DIFFERENT test sets (909 vs 2610 utterances) with different class balance.
# Those numbers cannot be compared -- a WF1 "gain" may just be a more-neutral
# eval set (filtered test keeps 41% of neutral but only 24% of joy).
#
# This runs every checkpoint against BOTH test sets so each column is
# apples-to-apples. evaluate_text_only.py runs XLM-R live on raw transcripts
# (no cached text embeddings), so any checkpoint is safe on any test set.
#
# Read the results as: compare DOWN a test-set column, never ACROSS columns.

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv

set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
mkdir -p logs

CKPT_ROOT=/dcs/large/u5734759/checkpoints

# name : checkpoint path
MODELS=(
  "mini:${CKPT_ROOT}/mini/best_model.pt"
  "mini_wer25_filter:${CKPT_ROOT}/mini_wer25_filter/best_model.pt"
  "mini_aug:${CKPT_ROOT}/mini_aug/best_model.pt"
)

# name : config (config decides which test set via filtering.enabled)
TESTSETS=(
  "testfull:src/configs/eval_testfull.yaml"
  "testwer25:src/configs/eval_testwer25.yaml"
)

fail=0
for m in "${MODELS[@]}"; do
  mname="${m%%:*}"; mpath="${m#*:}"
  if [ ! -f "$mpath" ]; then
    echo "!!! SKIP $mname — checkpoint missing: $mpath"
    fail=1
    continue
  fi
  for t in "${TESTSETS[@]}"; do
    tname="${t%%:*}"; tcfg="${t#*:}"
    echo "=========================================================="
    echo "=== MODEL=$mname  TESTSET=$tname"
    echo "=========================================================="
    $VENV/bin/python3.12 src/evaluation/evaluate_text_only.py \
        --config "$tcfg" \
        --checkpoint_path "$mpath" \
        --tag "_${mname}_on_${tname}"
    rc=$?
    [ $rc -ne 0 ] && { echo "!!! FAILED: $mname on $tname (exit $rc)"; fail=1; }
  done
done

echo "=========================================================="
if [ $fail -eq 0 ]; then
  echo "=== ALL EVALS COMPLETED ==="
else
  echo "=== FINISHED WITH FAILURES/SKIPS — see !!! lines above ==="
fi
exit $fail
