#!/bin/bash
#SBATCH --job-name=repr_pool
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=96G --gres=gpu:1
#SBATCH --account=wmlg --time=03:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Does LEARNED pooling organise the acoustic space better than a masked mean?
#
# The `acoustic` stage in the representation analysis is the masked mean, and
# its UMAP shows no emotion structure whatsoever -- every class scattered
# through one blob. But Phase 1 measured attention pooling at 0.5044 dev WF1
# against masked mean's 0.4671, so the masked mean is the WEAKER
# representation and that plot understates the audio channel.
#
# These four runs are acoustic-ONLY (modality=acoustic, no text), so
# represent() returns the pooled acoustic vector alone. Same audio, same
# architecture, four poolers -- the only difference is how the (T,1280)
# sequence is collapsed.
#
#   attention        0.5044   learned Bahdanau additive
#   attentive_stats  0.4809   weighted mean AND std, 2560-d before projection
#   masked_mean      0.4671   the control
#   attention_fixed  0.4603   non-learned scores

set -u
source /etc/profile.d/modules.sh; source /etc/profile.d/conda.sh
export HF_HOME=/dcs/large/u5734759/hf_cache VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
cd /dcs/pg25/u5734759/dissertation-project
PY=$VENV/bin/python3.12
P1=/dcs/large/u5734759/checkpoints/fusion/phase1_pooling
OUT=/dcs/large/u5734759/data/meld_pooled
fail=0

for POOL in attention attentive_stats masked_mean attention_fixed; do
  if [ -f "$OUT/$POOL/test_fused.pt" ]; then echo "--- have $POOL"; continue; fi
  echo "--- extracting $POOL"
  # Phase 1 ran on the asr condition; the ACOUSTIC cache is identical across
  # conditions, and these models use no text, so the condition is irrelevant.
  $PY src/preprocessing/extract_fused_features.py \
      --config src/configs/fusion_asr.yaml \
      --checkpoint $P1/acoustic_${POOL}/best_model.pt \
      --out_dir $OUT/$POOL --splits train test || fail=1
done

echo ""
echo "############ ANALYSIS — poolers side by side ############"
$PY src/evaluation/represent_learned.py \
    --config src/configs/ctx_gold.yaml --condition gold \
    --splits test \
    --stages acoustic acoustic_attn acoustic_attnstats acoustic_mean_learned acoustic_attnfixed \
    --out_dir results_new/represent/pooling || fail=1

echo ""
find results_new/represent/pooling -name "*.png" | wc -l | xargs echo "figures:"
[ $fail -eq 0 ] && echo "=== POOLING COMPARISON COMPLETE ===" || echo "=== FAILURES ==="
exit $fail
