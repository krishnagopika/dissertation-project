#!/bin/bash
#SBATCH --job-name=attnpool_ctx
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=96G --gres=gpu:1
#SBATCH --account=wmlg --time=06:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# THE GAP: fusion used ATTENTION pooling; both context families read the
# MASKED MEAN. Measured difference in the acoustic representation (test, gold):
#
#     masked mean   probe F1 0.3957   cosine gap 0.0138
#     attention     probe F1 0.4623   cosine gap 0.0779   <- 5.6x separation
#
# So every context result to date ran on the degraded acoustic input, while
# fusion did not. That makes "context helps less than fusion" partly an input
# artefact rather than an architectural fact -- which is what this grid tests.
#
# 45 runs, each pairing against an existing masked-mean cell:
#     bc-LSTM                 3 conditions x 5 widths        = 15
#     ctxfusion both          3 conditions x 5 widths        = 15
#     ctxfusion acoustic_only 3 conditions x 5 widths        = 15
#
# text_only and neither are NOT re-run: neither has an acoustic BiLSTM, so
# changing the acoustic pooling does not test what those arms exist to isolate.
# They stay as unchanged controls.

set -u
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
export HF_HOME=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
cd /dcs/pg25/u5734759/dissertation-project
mkdir -p logs
PY=$VENV/bin/python3.12
POOL=/dcs/large/u5734759/data/meld_pooled/attention
fail=0

echo "############ 0. dev_fused.pt (needed for early stopping) ############"
if [ ! -f "$POOL/dev_fused.pt" ]; then
  $PY src/preprocessing/extract_fused_features.py \
      --config src/configs/fusion_asr.yaml \
      --checkpoint /dcs/large/u5734759/checkpoints/fusion/phase1_pooling/acoustic_attention/best_model.pt \
      --out_dir $POOL --splits dev || { echo "dev extraction FAILED"; exit 1; }
else echo "  have dev_fused.pt"; fi

echo ""
echo "############ 1. bc-LSTM x attention pooling (15) ############"
for COND in gold asr asr_cleaned; do
  for K in 0 1 2 4 -1; do
    KL=$([ $K -eq -1 ] && echo full || echo k$K)
    TAG="attn_${COND}/${KL}"
    M="/dcs/large/u5734759/checkpoints/bclstm/${TAG}/TRAINING_COMPLETE.json"
    [ -f "$M" ] && { echo "--- SKIP $TAG"; continue; }
    echo ""; echo "--- TRAIN bclstm $TAG"
    $PY src/training/train_context.py \
        --config src/configs/ctx_${COND}.yaml \
        --fused_path $POOL --fused_mode acoustic \
        --tag "$TAG" --context_window $K \
        --hidden_dim 256 --num_layers 1 \
        --epochs 30 --patience 5 --batch_size 16 || fail=1
  done
done

echo ""
echo "############ 2. ContextThenFusion x attention pooling (30) ############"
for COND in gold asr asr_cleaned; do
  for ARM in both acoustic_only; do
    FLAGS=$([ "$ARM" = "both" ] && echo "--use_acoustic_lstm --use_text_lstm" \
                                || echo "--use_acoustic_lstm")
    for K in 0 1 2 4 -1; do
      KL=$([ $K -eq -1 ] && echo full || echo k$K)
      TAG="attn_${COND}/${ARM}_${KL}"
      M="/dcs/large/u5734759/checkpoints/ctxfusion/${TAG}/TRAINING_COMPLETE.json"
      [ -f "$M" ] && { echo "--- SKIP $TAG"; continue; }
      echo ""; echo "--- TRAIN ctxfusion $TAG"
      $PY src/training/train_context_fusion.py \
          --config src/configs/ctxfusion_${COND}.yaml \
          --fused_path $POOL --fused_mode acoustic \
          --tag "$TAG" --fusion concat --context_window $K \
          --lstm_hidden 256 --num_layers 1 \
          --epochs 30 --patience 5 --batch_size 16 \
          $FLAGS || fail=1
    done
  done
done

echo ""
echo "############ SUMMARY ############"
$PY src/evaluation/bclstm_report.py || true
$PY src/evaluation/ctxfusion_report.py || true
[ $fail -eq 0 ] && echo "=== ATTENTION-POOLED CONTEXT GRID COMPLETE ===" \
                || echo "=== FINISHED WITH FAILURES ==="
exit $fail
