#!/bin/bash
#SBATCH --job-name=train_context_filter
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv

set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
mkdir -p logs

# bc-LSTM on filter-restricted MELD dialogues.
# Uses filtering.keys_paths from mini.yaml (currently wer25). Filtered-out
# utterances remain in dialogues for BiLSTM context, but their labels are
# masked so only kept utterances contribute to loss + test metrics.
# Result: results/mini/test_results_bclstm_wer25_filter.json
#
# Prereqs (already done):
#   - filter metadata + keep-lists exist (compute_filter_metadata + apply_filter)
#   - filter Phase 1 XLM-R has been re-run (or use whatever is currently
#     cached in meld_text_embeddings/ — that's the filter-trained one now).
$VENV/bin/python3.12 src/training/train_context.py \
    --config src/configs/mini.yaml \
    --hidden_dim 512 --num_layers 2 --epochs 40 --patience 6 \
    --tag _wer25_filter
