#!/bin/bash
#SBATCH --job-name=repr_aonly
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=64G --gres=gpu:1
#SBATCH --account=wmlg --time=02:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
# Extract the acoustic-only-context representation, so "what does a dialogue
# model do to PROSODY" can be asked separately from "what does it do to the
# blend". bc-LSTM cannot answer that -- it emits one vector from concatenated
# input -- but ContextThenFusion's acoustic_only arm can.
set -u
source /etc/profile.d/modules.sh; source /etc/profile.d/conda.sh
export HF_HOME=/dcs/large/u5734759/hf_cache VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
cd /dcs/pg25/u5734759/dissertation-project
PY=$VENV/bin/python3.12
CTX=/dcs/large/u5734759/data/meld_context
for COND in gold asr asr_cleaned; do
  [ -f "$CTX/ctxfusion_aonly_$COND/test_context.pt" ] && { echo "have $COND"; continue; }
  echo "--- acoustic-only context, $COND"
  $PY src/preprocessing/extract_context_features.py \
      --config src/configs/ctxfusion_${COND}.yaml --model ctxfusion \
      --checkpoint /dcs/large/u5734759/checkpoints/ctxfusion/${COND}/acoustic_only/best_model.pt \
      --out_dir $CTX/ctxfusion_aonly_$COND --splits train test
done
echo "=== done ==="

echo ""
echo "############ MEASURE ############"
for COND in gold asr asr_cleaned; do
  $PY src/evaluation/represent_learned.py \
      --config src/configs/ctx_${COND}.yaml --condition $COND \
      --splits test --stages acoustic ctxfusion_acoustic \
      --out_dir results_new/represent/aonly_${COND} || true
done
