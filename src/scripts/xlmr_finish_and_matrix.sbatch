#!/bin/bash
#SBATCH --job-name=xlmr_matrix
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# 1. Re-run the two runs that died on a disk-quota error.
# 2. Cross-evaluate EVERY trained model on EVERY test condition.
#
#                          TEST
#                  gold     asr     asr_clean
#   TRAIN gold      .        .          .
#         asr       .        .          .
#         asr_clean .        .          .
#
# The diagonal is the matched condition. Off-diagonal cells measure transfer:
# train-gold/test-asr is the deployment cost of training on clean transcripts;
# train-asr/test-gold shows whether ASR-trained models lose anything when the
# transcript happens to be perfect.
#
# "asr_clean" as a TEST condition means ASR text on the text_clean subset
# (1,787 of 2,610), not a different signal -- the audio is identical
# throughout, and the acoustic branch is not involved in a text-only model.
#
# Checkpoints keep only best_model.pt: the retained per-epoch checkpoint
# doubled the footprint and exhausted the quota during the first grid.

set -u
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
cd /dcs/pg25/u5734759/dissertation-project
mkdir -p logs results_new/xml_roberta
fail=0

echo "##################### 1. RERUN FAILED #####################"
for LOSS in weighted focal; do
  RUN="xlmr_asr_cleaned_${LOSS}"
  if [ -f "/dcs/large/u5734759/checkpoints/$RUN/TRAINING_COMPLETE.json" ]; then
    echo "--- $RUN already complete, skipping"; continue
  fi
  echo "--- TRAIN $RUN"
  rm -f /dcs/large/u5734759/checkpoints/$RUN/checkpoint_epoch*.pt
  $VENV/bin/python3.12 src/training/finetune.py \
      --config src/configs/${RUN}.yaml \
      --text_source asr --loss $LOSS --patience 3 || fail=1
  # Drop the retained epoch checkpoint immediately: best_model.pt holds the
  # selected weights and the epoch file is 3.3 GB of quota we do not have.
  rm -f /dcs/large/u5734759/checkpoints/$RUN/checkpoint_epoch*.pt
done

echo ""
echo "##################### 2. CROSS-EVALUATION MATRIX #####################"
# TEST_SRC = the text the test set is scored with
# TEST_KEYS = "" for the full 2,610, or the clean keep-list for the subset
KEYS=/dcs/large/u5734759/data/meld_extracted/mini/filter_keys
for TRAIN_COND in gold asr asr_cleaned; do
  for LOSS in plain weighted focal; do
    RUN="xlmr_${TRAIN_COND}_${LOSS}"
    CKPT="/dcs/large/u5734759/checkpoints/$RUN/best_model.pt"
    [ -f "$CKPT" ] || { echo "--- SKIP $RUN (no checkpoint)"; continue; }

    for TEST_COND in gold asr asr_clean; do
      case $TEST_COND in
        gold)      SRC=gold; EXTRA="" ;;
        asr)       SRC=asr;  EXTRA="" ;;
        asr_clean) SRC=asr;  EXTRA="--filtered_keys_path $KEYS/test_keys_text_clean.json" ;;
      esac
      echo "--- EVAL train=$TRAIN_COND/$LOSS  test=$TEST_COND"
      $VENV/bin/python3.12 src/evaluation/evaluate_text_only.py \
          --config src/configs/${RUN}.yaml \
          --checkpoint_path "$CKPT" \
          --split test --text_source $SRC $EXTRA \
          --tag "_${LOSS}_TESTON_${TEST_COND}" || fail=1
    done
  done
done

echo ""
[ $fail -eq 0 ] && echo "=== MATRIX COMPLETE ===" || echo "=== FINISHED WITH FAILURES ==="
exit $fail
