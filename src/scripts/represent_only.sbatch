#!/bin/bash
#SBATCH --job-name=represent
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=96G --gres=gpu:1
#SBATCH --account=wmlg --time=04:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Representation analysis only -- split out from the classical baselines so the
# figures do not queue behind ~54 sklearn fits (the 2048-d XGBoost fits alone
# run several minutes each).
#
# Stage-4 checkpoint: k4 UNIFORMLY across conditions, not each condition's own
# best, so the stage is defined by ONE context width and the row does not
# confound representation quality with window size.
# Stage-5 checkpoint: the `both` arm at full dialogue, which won on gold.

set -u
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
export HF_HOME=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
cd /dcs/pg25/u5734759/dissertation-project
mkdir -p logs results_new/represent
PY=$VENV/bin/python3.12
CTX=/dcs/large/u5734759/data/meld_context
fail=0

echo "############ EXTRACT STAGE 4 + 5 ############"
for COND in gold asr asr_cleaned; do
  if [ ! -f "$CTX/$COND/test_context.pt" ]; then
    echo "--- stage4 $COND (bc-LSTM k4)"
    $PY src/preprocessing/extract_context_features.py \
        --config src/configs/ctx_${COND}.yaml --model bclstm \
        --checkpoint /dcs/large/u5734759/checkpoints/bclstm/${COND}/k4/best_context.pt \
        --out_dir $CTX/$COND --splits train test || fail=1
  else echo "--- have stage4 $COND"; fi

  if [ ! -f "$CTX/ctxfusion_$COND/test_context.pt" ]; then
    echo "--- stage5 $COND (ContextThenFusion both, full)"
    $PY src/preprocessing/extract_context_features.py \
        --config src/configs/ctxfusion_${COND}.yaml --model ctxfusion \
        --checkpoint /dcs/large/u5734759/checkpoints/ctxfusion/${COND}/both/best_model.pt \
        --out_dir $CTX/ctxfusion_$COND --splits train test || fail=1
  else echo "--- have stage5 $COND"; fi
done

echo ""
echo "############ ANALYSIS ############"
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
find results_new/represent -name "*.png" | wc -l | xargs echo "figures:"
find results_new/represent -name "*.png" | sort
[ $fail -eq 0 ] && echo "=== REPRESENT COMPLETE ===" || echo "=== FINISHED WITH FAILURES ==="
exit $fail
