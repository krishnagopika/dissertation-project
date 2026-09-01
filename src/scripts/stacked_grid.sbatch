#!/bin/bash
#SBATCH --job-name=stacked_grid
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=96G --gres=gpu:1
#SBATCH --account=wmlg --time=06:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# FUSION + bc-LSTM: the stacked model.
#
#   acoustic (T,1280) ─┐ learned pooling
#                      ├─► fusion ─► 512-d ─► bc-LSTM over dialogue ─► prediction
#   text (768) ────────┘
#
# Two stages:
#   1. Cache fusion's LEARNED representation per utterance, using each
#      condition's best Phase 3 checkpoint.
#   2. Train bc-LSTM on those vectors: 3 conditions x 5 context widths.
#
#              K=0   K=1   K=2   K=4   full
#   gold        .     .     .     .     .
#   asr         .     .     .     .     .
#   asr_cleaned .     .     .     .     .
#
# Why this is the comparison that matters
# ---------------------------------------
# Plain bc-LSTM reads a raw 768+1280 concatenation with a FIXED masked-mean
# acoustic vector. Fusion LEARNS its pooling and its combination. So plain
# bc-LSTM vs fusion differs in both context AND representation, and neither
# number isolates either effect. Running bc-LSTM on fusion's own output makes
# CONTEXT the only difference from plain fusion, and REPRESENTATION the only
# difference from plain bc-LSTM.
#
# Source checkpoints (Phase 3 winners, all attention pooling):
#   gold         gold_sum         dev WF1 0.5652
#   asr          asr_concat       dev WF1 0.5021
#   asr_cleaned  asr_cleaned_sum  dev WF1 0.4935

set -u
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
export HF_HOME=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
cd /dcs/pg25/u5734759/dissertation-project
mkdir -p logs results_new/stacked
PY=$VENV/bin/python3.12
FUSED=/dcs/large/u5734759/data/meld_fused
CK=/dcs/large/u5734759/checkpoints/fusion/phase3_fusion
fail=0

echo "############ 1. EXTRACT FUSED REPRESENTATIONS ############"
extract () {   # $1=cond  $2=run name
  local cond=$1 run=$2
  if [ -f "$FUSED/$cond/test_fused.pt" ]; then
    echo "--- SKIP $cond (already extracted)"; return
  fi
  echo "--- EXTRACT $cond from $run"
  $PY src/preprocessing/extract_fused_features.py \
      --config src/configs/fusion_${cond}.yaml \
      --checkpoint $CK/${run}/best_model.pt \
      --out_dir $FUSED/$cond || fail=1
}
extract gold        gold_sum
extract asr         asr_concat
extract asr_cleaned asr_cleaned_sum

echo ""
echo "############ 2. STACKED GRID — 3 x 5 ############"
run_one () {   # $1=cond  $2=k label  $3=window value
  local cond=$1 klabel=$2 kval=$3
  local tag="stacked_${cond}/${klabel}"
  local marker="/dcs/large/u5734759/checkpoints/bclstm/${tag}/TRAINING_COMPLETE.json"
  if [ -f "$marker" ]; then echo "--- SKIP $tag (complete)"; return; fi
  echo ""
  echo "--- TRAIN $tag   (context_window=$kval, fused features)"
  $PY src/training/train_context.py \
      --config src/configs/ctx_${cond}.yaml \
      --fused_path $FUSED/$cond \
      --tag "$tag" --context_window $kval \
      --hidden_dim 256 --num_layers 1 \
      --epochs 30 --patience 5 --batch_size 16 || fail=1
}

for COND in gold asr asr_cleaned; do
  echo ""
  echo "############ CONDITION: $COND (stacked) ############"
  for K in 0 1 2 4; do run_one $COND "k${K}" $K; done
  run_one $COND "full" -1
  $PY src/evaluation/bclstm_report.py || true
done

echo ""
echo "############ SUMMARY ############"
$PY src/evaluation/bclstm_report.py || true

[ $fail -eq 0 ] && echo "=== STACKED GRID COMPLETE ===" || echo "=== FINISHED WITH FAILURES ==="
exit $fail
