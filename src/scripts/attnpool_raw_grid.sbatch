#!/bin/bash
#SBATCH --job-name=attn_raw
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=96G --gres=gpu:1
#SBATCH --account=wmlg --time=04:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# RAW attention-pooled acoustic, 1280-d, pre-projection.
#
# The first attention grid used SequenceFusion.represent(), which is
# POST-projection: pooler -> 1280 -> acoustic_proj -> 512 -> post -> 512. That
# bundles the pooler with a learned 1280->512 projection trained for fusion's
# objective, so a gain over the masked mean could come from either. The
# `learned masked_mean` control (cosine gap 0.0657 vs attention's 0.0779)
# suggests the projection was doing most of the work.
#
# This extracts the pooler output ALONE, so the comparison against
# {split}_embeddings_maskedmean.pt is like-for-like: same 1280 width, same
# frames, only the pooling rule differs.
#
# Feature vector stays text + acoustic: 768 + 1280 = 2048, identical in width
# to the original masked-mean runs.

set -u
source /etc/profile.d/modules.sh; source /etc/profile.d/conda.sh
export HF_HOME=/dcs/large/u5734759/hf_cache VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
cd /dcs/pg25/u5734759/dissertation-project
PY=$VENV/bin/python3.12
RAW=/dcs/large/u5734759/data/meld_pooled_raw/attention
P1=/dcs/large/u5734759/checkpoints/fusion/phase1_pooling
fail=0

echo "############ 0. EXTRACT raw 1280-d attention-pooled ############"
if [ ! -f "$RAW/test_fused.pt" ]; then
  $PY src/preprocessing/extract_pooled_acoustic.py \
      --config src/configs/fusion_asr.yaml \
      --checkpoint $P1/acoustic_attention/best_model.pt \
      --out_dir $RAW --splits train dev test || exit 1
else echo "  have it"; fi

echo ""
echo "############ 1. bc-LSTM, raw attention pooling (15) ############"
for COND in gold asr asr_cleaned; do
  for K in 0 1 2 4 -1; do
    KL=$([ $K -eq -1 ] && echo full || echo k$K)
    TAG="attnraw_${COND}/${KL}"
    [ -f "/dcs/large/u5734759/checkpoints/bclstm/${TAG}/TRAINING_COMPLETE.json" ] && \
      { echo "--- SKIP $TAG"; continue; }
    echo ""; echo "--- TRAIN bclstm $TAG"
    $PY src/training/train_context.py \
        --config src/configs/ctx_${COND}.yaml \
        --fused_path $RAW --fused_mode acoustic \
        --tag "$TAG" --context_window $K \
        --hidden_dim 256 --num_layers 1 \
        --epochs 30 --patience 5 --batch_size 16 || fail=1
  done
done

echo ""
echo "############ 2. ContextThenFusion, raw attention (30) ############"
for COND in gold asr asr_cleaned; do
  for ARM in both acoustic_only; do
    FLAGS=$([ "$ARM" = "both" ] && echo "--use_acoustic_lstm --use_text_lstm" \
                                || echo "--use_acoustic_lstm")
    for K in 0 1 2 4 -1; do
      KL=$([ $K -eq -1 ] && echo full || echo k$K)
      TAG="attnraw_${COND}/${ARM}_${KL}"
      [ -f "/dcs/large/u5734759/checkpoints/ctxfusion/${TAG}/TRAINING_COMPLETE.json" ] && \
        { echo "--- SKIP $TAG"; continue; }
      echo ""; echo "--- TRAIN ctxfusion $TAG"
      $PY src/training/train_context_fusion.py \
          --config src/configs/ctxfusion_${COND}.yaml \
          --fused_path $RAW --fused_mode acoustic \
          --tag "$TAG" --fusion concat --context_window $K \
          --lstm_hidden 256 --num_layers 1 \
          --epochs 30 --patience 5 --batch_size 16 \
          $FLAGS || fail=1
    done
  done
done

echo ""
$PY src/evaluation/bclstm_report.py || true
$PY src/evaluation/ctxfusion_report.py || true
[ $fail -eq 0 ] && echo "=== RAW ATTENTION GRID COMPLETE ===" || echo "=== FAILURES ==="
exit $fail
