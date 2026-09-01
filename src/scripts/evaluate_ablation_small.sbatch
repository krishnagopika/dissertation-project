#!/bin/bash
#SBATCH --job-name=meld_eval_ablation_small
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=0-04:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Full evaluation pipeline for one Phase 1 ablation, using the SMALL config.
# Identical to evaluate_ablation.sbatch but points every step at small.yaml so
# checkpoints, cached embeddings, and result JSONs land under the small tree.
#
# Usage:
#   sbatch src/scripts/evaluate_ablation_small.sbatch <ABLATION_TAG> <PHASE1_CKPT>
#
# Example:
#   sbatch src/scripts/evaluate_ablation_small.sbatch \
#       _small_baseline \
#       /dcs/large/u5734759/checkpoints/small/best_model.pt

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv

set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a

mkdir -p logs

ABLATION_TAG="${1:-}"
PHASE1_CKPT="${2:-}"

if [ -z "$ABLATION_TAG" ] || [ -z "$PHASE1_CKPT" ]; then
    echo "ERROR: missing positional arguments"
    echo "Usage:"
    echo "  sbatch src/scripts/evaluate_ablation_small.sbatch <ABLATION_TAG> <PHASE1_CKPT>"
    exit 2
fi

if [ ! -f "$PHASE1_CKPT" ]; then
    echo "ERROR: Phase 1 checkpoint not found: $PHASE1_CKPT"
    exit 2
fi

echo "============================================"
echo "Ablation tag:       $ABLATION_TAG"
echo "Phase 1 checkpoint: $PHASE1_CKPT"
echo "Using config:       src/configs/small.yaml"
echo "============================================"

# Stage the checkpoint at the canonical small location so fusion training and
# evaluate.py can find it via config["training"]["checkpoint_dir"] / best_model.pt.
mkdir -p /dcs/large/u5734759/checkpoints/small
cp "$PHASE1_CKPT" /dcs/large/u5734759/checkpoints/small/best_model.pt

# ---- 1. Re-extract gold-text CLS embeddings ----
echo ""
echo "[1/4] Re-extracting gold-text CLS embeddings with this ablation's XLM-R..."
$VENV/bin/python3.12 src/preprocessing/extract_text_embeddings.py \
    --config src/configs/small.yaml \
    --checkpoint_path "$PHASE1_CKPT"

# ---- 2. Re-train all 4 fusion variants on the new cached embeddings ----
echo ""
echo "[2/4] Re-training fusion variants on new embeddings..."
for FUSION_TYPE in concat sum gated crossmodal; do
    echo "    ----  fusion=$FUSION_TYPE  ----"
    rm -f /dcs/large/u5734759/checkpoints/small/fusion/${FUSION_TYPE}/checkpoint_epoch*.pt \
          /dcs/large/u5734759/checkpoints/small/fusion/${FUSION_TYPE}/best_model.pt
    $VENV/bin/python3.12 src/training/train_fusion.py \
        --config src/configs/small.yaml \
        --fusion_type "$FUSION_TYPE"
done

# ---- 3. Evaluate text-only + each fusion variant on test ----
echo ""
echo "[3/4] Evaluating text-only baseline..."
$VENV/bin/python3.12 src/evaluation/evaluate_text_only.py \
    --config src/configs/small.yaml \
    --checkpoint_path "$PHASE1_CKPT" \
    --tag "$ABLATION_TAG"

for FUSION_TYPE in concat sum gated crossmodal; do
    echo "    ----  eval fusion=$FUSION_TYPE  ----"
    $VENV/bin/python3.12 src/evaluation/evaluate.py \
        --config src/configs/small.yaml \
        --fusion_type "$FUSION_TYPE" \
        --tag "$ABLATION_TAG"
done

# ---- 4. Classical (SVM, XGBoost) on the new [CLS] embeddings ----
echo ""
echo "[4/4] Training + evaluating classical (SVM, XGBoost)..."
$VENV/bin/python3.12 src/training/train_classical.py \
    --config src/configs/small.yaml \
    --model both \
    --phase1_checkpoint "$PHASE1_CKPT" \
    --tag "$ABLATION_TAG"

echo ""
echo "Done. All output files for ablation '$ABLATION_TAG' are under results/small/."
