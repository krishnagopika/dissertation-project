#!/bin/bash
#SBATCH --job-name=meld_fusion_ablations
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --account=wmlg
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Trains all four fusion ablations back-to-back in a single job:
#   concat | sum | gated | crossmodal
# Each saves its best checkpoint to:
#   /dcs/large/u5734759/checkpoints/mini/fusion/<fusion_type>/best_model.pt

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv

set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a

mkdir -p logs

for FUSION_TYPE in concat sum gated crossmodal; do
    echo "============================================"
    echo "Starting Phase 2 fusion training: ${FUSION_TYPE}"
    echo "============================================"

    $VENV/bin/python3.12 src/training/train_fusion.py \
        --config src/configs/mini.yaml \
        --fusion_type "${FUSION_TYPE}"

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: ${FUSION_TYPE} training failed with exit code ${EXIT_CODE}"
        exit $EXIT_CODE
    fi

    echo "Completed: ${FUSION_TYPE}"
done

echo "All fusion ablations complete."
