#!/bin/bash
#SBATCH --job-name=fusion_eval
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=0-02:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Evaluates the text-only baseline and all four fusion ablations on the MELD
# test set, saving per-run JSON results to results/mini/.

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv

set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a

mkdir -p logs

# --- Text-only baseline (Phase 1 checkpoint, no acoustic features) ---
echo "Evaluating text-only baseline..."
$VENV/bin/python3.12 src/evaluation/evaluate_text_only.py \
    --config src/configs/mini.yaml

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: text-only evaluation failed (exit ${EXIT_CODE})"
    exit $EXIT_CODE
fi

# --- Multimodal fusion ablations (Phase 2 checkpoints) ---
for FUSION_TYPE in concat sum gated crossmodal; do
    echo "============================================"
    echo "Evaluating fusion: ${FUSION_TYPE}"
    echo "============================================"

    $VENV/bin/python3.12 src/evaluation/evaluate.py \
        --config src/configs/mini.yaml \
        --fusion_type "${FUSION_TYPE}"

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: ${FUSION_TYPE} evaluation failed (exit ${EXIT_CODE})"
        exit $EXIT_CODE
    fi
done

echo "All evaluations complete."
