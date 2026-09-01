#!/bin/bash
#SBATCH --job-name=classical_repr
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=96G --gres=gpu:1
#SBATCH --account=wmlg --time=08:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# TWO analyses that share the same cached features.
#
# 1. CLASSICAL BASELINES -- 3 models x 3 feature sets x 3 conditions x 2 tasks.
#    logreg / linear SVM / XGBoost on exactly the vectors the neural models
#    consume. If logistic regression matches the fusion head, the head is not
#    earning its complexity and the representation is doing the work. Given
#    fusion beat text-only by 0.003 and the ctxfusion 2x2 showed no reliable
#    test benefit, that is a live possibility rather than a rhetorical one.
#
# 2. REPRESENTATION ANALYSIS -- probe / cosine / silhouette / PCA / UMAP /t-SNE
#    across the five pipeline stages, for all three text conditions.
#    Requires stage 4 and 5 caches, extracted here first.
#
# Stage-4 checkpoint choice: k4 UNIFORMLY, not each condition's own best.
# Defining the stage by one context width keeps the row from confounding
# representation with window size.

set -u
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
export HF_HOME=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=16
set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
cd /dcs/pg25/u5734759/dissertation-project
mkdir -p logs results_new/classical results_new/represent
PY=$VENV/bin/python3.12
fail=0

echo "############ 1. CLASSICAL BASELINES ############"
for COND in gold asr asr_cleaned; do
  echo ""
  echo "--- $COND"
  $PY src/training/train_classical.py \
      --config src/configs/ctx_${COND}.yaml --condition $COND \
      --features text acoustic both --models logreg svm xgboost \
      --test_split test --n_jobs 16 || fail=1
done

echo ""
echo "############ 2a. EXTRACT STAGE 4 + 5 REPRESENTATIONS ############"
CTX=/dcs/large/u5734759/data/meld_context
for COND in gold asr asr_cleaned; do
  # stage 4 -- plain bc-LSTM at k4
  if [ ! -f "$CTX/$COND/test_context.pt" ]; then
    echo "--- stage4 $COND (bc-LSTM k4)"
    $PY src/preprocessing/extract_context_features.py \
        --config src/configs/ctx_${COND}.yaml --model bclstm \
        --checkpoint /dcs/large/u5734759/checkpoints/bclstm/${COND}/k4/best_context.pt \
        --out_dir $CTX/$COND --splits train test || fail=1
  else echo "--- have stage4 $COND"; fi

  # stage 5 -- ContextThenFusion, both branches, full dialogue
  if [ ! -f "$CTX/ctxfusion_$COND/test_context.pt" ]; then
    echo "--- stage5 $COND (ContextThenFusion both)"
    $PY src/preprocessing/extract_context_features.py \
        --config src/configs/ctxfusion_${COND}.yaml --model ctxfusion \
        --checkpoint /dcs/large/u5734759/checkpoints/ctxfusion/${COND}/both/best_model.pt \
        --out_dir $CTX/ctxfusion_$COND --splits train test || fail=1
  else echo "--- have stage5 $COND"; fi
done

echo ""
echo "############ 2b. REPRESENTATION ANALYSIS ############"
for COND in gold asr asr_cleaned; do
  echo ""
  echo "--- $COND"
  $PY src/evaluation/represent_learned.py \
      --config src/configs/ctx_${COND}.yaml --condition $COND \
      --splits train test \
      --stages text acoustic fused context ctxfusion || fail=1
done

echo ""
echo "############ INVENTORY ############"
echo "figures:"
find results_new/represent -name "*.png" 2>/dev/null | wc -l
echo "json:"
find results_new/represent results_new/classical -name "*.json" 2>/dev/null | wc -l

[ $fail -eq 0 ] && echo "=== COMPLETE ===" || echo "=== FINISHED WITH FAILURES ==="
exit $fail
