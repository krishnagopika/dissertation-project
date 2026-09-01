#!/bin/bash
#SBATCH --job-name=ctxfusion
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=64G --gres=gpu:1
#SBATCH --account=wmlg --time=06:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# CONTEXT THEN FUSION -- reversing the order the stacked model used.
#
#   acoustic ─► BiLSTM_a over dialogue ─┐
#                                        ├─► fuse ─► classify
#   text ─────► BiLSTM_t over dialogue ─┘
#
# The stacked model fused first and contextualised second, and did not work:
# `concat` mode kept the raw features alongside the fused vector (2560-d, MORE
# parameters than plain) and still lost to a plain bc-LSTM on gold. So the
# fused vector carries nothing the context model cannot already obtain.
# Compressing to 512-d against a per-utterance objective destroys exactly what
# the temporal model needed. Here compression happens last.
#
# THE 2x2, run for all three text conditions = 12 runs:
#
#   acoustic  text    what it isolates
#   --------  -----   --------------------------------
#   on        on      both contextualised
#   on        off     acoustic-only context
#   off       on      text-only context
#   off       off     no context -- the control (= per-utterance fusion)
#
# Prediction from existing results: text-only ~ both, acoustic-only ~ neither.
# The acoustic cache is BYTE-IDENTICAL across gold/asr/asr_cleaned, yet context
# only helped when the text was clean (gold +0.026 k0->k4, asr +0.005). If the
# benefit ran through the acoustic channel the gain would be equal in all three.
# That inference is not decisive -- plain bc-LSTM concatenates both channels, so
# "needs clean text to exploit EITHER channel" predicts the same table. This
# 2x2 separates the two explanations.
#
# PARAMETER FAIRNESS: the arms are NOT matched. acoustic_lstm is larger than
# text_lstm (1280 vs 768 input), so both-on is 6.31M, acoustic-only 4.34M,
# text-only 3.55M, neither 1.58M. A loss in a smaller arm may be capacity, not
# architecture -- every run records parameter_report() so this stays visible.

set -u
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
export HF_HOME=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
cd /dcs/pg25/u5734759/dissertation-project
mkdir -p logs results_new/ctxfusion
PY=$VENV/bin/python3.12
fail=0

run_one () {   # $1=cond $2=arm label $3=extra flags
  local cond=$1 arm=$2 flags=$3
  local tag="${cond}/${arm}"
  local marker="/dcs/large/u5734759/checkpoints/ctxfusion/${tag}/TRAINING_COMPLETE.json"
  if [ -f "$marker" ]; then echo "--- SKIP $tag (complete)"; return; fi
  echo ""
  echo "--- TRAIN $tag   ($flags)"
  $PY src/training/train_context_fusion.py \
      --config src/configs/ctxfusion_${cond}.yaml \
      --tag "$tag" --fusion concat \
      --lstm_hidden 256 --num_layers 1 \
      --epochs 30 --patience 5 --batch_size 16 \
      $flags || fail=1
}

for COND in gold asr asr_cleaned; do
  echo ""
  echo "############ CONDITION: $COND ############"
  run_one $COND both          "--use_acoustic_lstm --use_text_lstm"
  run_one $COND acoustic_only "--use_acoustic_lstm"
  run_one $COND text_only     "--use_text_lstm"
  run_one $COND neither       ""
  $PY src/evaluation/ctxfusion_report.py || true
done

echo ""
echo "############ SUMMARY ############"
$PY src/evaluation/ctxfusion_report.py || true

[ $fail -eq 0 ] && echo "=== CTXFUSION GRID COMPLETE ===" || echo "=== FINISHED WITH FAILURES ==="
exit $fail
