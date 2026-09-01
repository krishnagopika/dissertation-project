#!/bin/bash
#SBATCH --job-name=ctx_same_testset
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# bc-LSTM 2x2: TRAIN condition x TEST condition.
#
# Why: the existing bclstm_unfiltered (0.6080 WF1) and bclstm_wer25_filter
# (0.6152 WF1) results appear to have been scored on DIFFERENT test sets
# (2610 vs 909 utterances), so the +0.7pt gap cannot be attributed to the
# training change. This separates the two variables.
#
# Note the existing results also disagree in direction: filtered wins on
# weighted F1 (0.6152 > 0.6080) but LOSES on macro F1 (0.3904 < 0.4335).
# That split is what a more neutral-heavy eval set produces, which is exactly
# what the wer25 filter does (keeps 41% of neutral, 24% of joy).
#
# Read DOWN a test column. The two cells sharing "testfull" are the honest
# answer to "did filtered training help?".

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv

set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
mkdir -p logs

CELLS=(
  "ctx_trainfull_testfull"
  "ctx_trainfull_testwer25"
  "ctx_trainwer25_testfull"
  "ctx_trainwer25_testwer25"
)

fail=0
for cell in "${CELLS[@]}"; do
  echo "=========================================================="
  echo "=== CELL: $cell"
  echo "=========================================================="
  $VENV/bin/python3.12 src/training/train_context.py \
      --config "src/configs/${cell}.yaml" \
      --tag "_${cell}"
  rc=$?
  [ $rc -ne 0 ] && { echo "!!! FAILED: $cell (exit $rc)"; fail=1; }
done

echo "=========================================================="
[ $fail -eq 0 ] && echo "=== ALL 4 CELLS COMPLETED ===" \
                || echo "=== FINISHED WITH FAILURES — see !!! lines ==="
exit $fail
