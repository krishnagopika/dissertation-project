#!/bin/bash
#SBATCH --job-name=bclstm_unfilt
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=01:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# bc-LSTM on the UNFILTERED MELD data, four context configurations:
#   whole dialogue | K=1 | K=2 | K=4
#
# Clean apples-to-apples with filter runs:
#   1. Backs up the current (filter-trained) Phase 1 XLM-R checkpoint.
#   2. Restores the aug-trained (unfiltered) XLM-R as mini/best_model.pt.
#   3. Re-extracts gold-text [CLS] embeddings using the unfiltered XLM-R —
#      overwrites meld_text_embeddings/ (previous filter cache is gone,
#      re-extract if you need filter bc-LSTM again).
#   4. Builds a temp config with filtering.enabled=false.
#   5. Runs bc-LSTM four times, one per context configuration.
#
# Outputs:
#   results/mini/test_results_bclstm_unfiltered.json      (whole dialogue)
#   results/mini/test_results_bclstm_unfiltered_win1.json
#   results/mini/test_results_bclstm_unfiltered_win2.json
#   results/mini/test_results_bclstm_unfiltered_win4.json

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv

set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
mkdir -p logs

CKPT_DIR=/dcs/large/u5734759/checkpoints/mini
FILTER_BACKUP=/dcs/large/u5734759/checkpoints/mini_wer25_filter
AUG_CKPT=/dcs/large/u5734759/checkpoints/mini_aug_v3/best_model.pt

# --- 1. Backup filter Phase 1 checkpoint (if present) ---
if [ -f "$CKPT_DIR/best_model.pt" ]; then
    echo "[1/5] Backing up current best_model.pt (filter) to $FILTER_BACKUP/"
    mkdir -p "$FILTER_BACKUP"
    mv "$CKPT_DIR/best_model.pt" "$FILTER_BACKUP/"
fi
rm -f "$CKPT_DIR"/checkpoint_epoch*.pt

# --- 2. Restore the aug (unfiltered) XLM-R ---
if [ ! -f "$AUG_CKPT" ]; then
    echo "ERROR: unfiltered aug XLM-R checkpoint not found at $AUG_CKPT"
    exit 2
fi
echo "[2/5] Copying unfiltered aug XLM-R into $CKPT_DIR/best_model.pt"
cp "$AUG_CKPT" "$CKPT_DIR/best_model.pt"

# --- 3. Build a temp config with filtering.enabled=false ---
TMPCFG="/tmp/mini_unfiltered_${SLURM_JOB_ID:-noslurm}.yaml"
cp src/configs/mini.yaml "$TMPCFG"
sed -i 's/^  enabled: true$/  enabled: false/' "$TMPCFG"
echo "[3/5] Temp config: $TMPCFG"
grep "^  enabled:" "$TMPCFG"

# --- 4. Re-extract text embeddings with the unfiltered XLM-R ---
echo "[4/5] Re-extracting text embeddings from unfiltered aug XLM-R..."
$VENV/bin/python3.12 src/preprocessing/extract_text_embeddings.py \
    --config "$TMPCFG" \
    --checkpoint_path "$CKPT_DIR/best_model.pt"

# --- 5. Run bc-LSTM × 4 (whole dialogue + K=1,2,4) ---
echo ""
echo "[5/5] Running bc-LSTM unfiltered × 4 configs..."

echo ""
echo "================================================"
echo "  bc-LSTM whole dialogue (no window cap)"
echo "================================================"
$VENV/bin/python3.12 src/training/train_context.py \
    --config "$TMPCFG" \
    --hidden_dim 512 --num_layers 2 --epochs 40 --patience 6 \
    --tag "_unfiltered"

for K in 1 2 4; do
    echo ""
    echo "================================================"
    echo "  bc-LSTM unfiltered, K=${K}"
    echo "================================================"
    $VENV/bin/python3.12 src/training/train_context.py \
        --config "$TMPCFG" \
        --hidden_dim 512 --num_layers 2 --epochs 40 --patience 6 \
        --tag "_unfiltered" \
        --context_window "${K}"
done

# --- Cleanup ---
rm -f "$TMPCFG"
echo ""
echo "All 4 unfiltered bc-LSTM configs complete."
ls -la results/mini/test_results_bclstm_unfiltered*.json
