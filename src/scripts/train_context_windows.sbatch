#!/bin/bash
#SBATCH --job-name=bclstm_windows
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=00:45:00
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

# bc-LSTM context-window ablations on filter subset (wer25):
#   K=1  → ±1 utterance  (3-utterance window)
#   K=2  → ±2 utterances (5-utterance window)
#   K=4  → ±4 utterances (9-utterance window)
#
# Filter is applied via mini.yaml's filtering.enabled=true. Windowed dataset
# unrolls each dialogue into per-utterance windows; only the centre utterance
# contributes to loss/metrics. Full-dialogue bc-LSTM (--context_window omitted)
# has already been run separately as the "whole dialogue" baseline.
#
# Outputs:
#   results/mini/test_results_bclstm_wer25_filter_win1.json
#   results/mini/test_results_bclstm_wer25_filter_win2.json
#   results/mini/test_results_bclstm_wer25_filter_win4.json

for K in 1 2 4; do
    echo ""
    echo "================================================"
    echo "  bc-LSTM context_window K=${K}"
    echo "================================================"
    $VENV/bin/python3.12 src/training/train_context.py \
        --config src/configs/mini.yaml \
        --hidden_dim 512 --num_layers 2 --epochs 40 --patience 6 \
        --tag "_wer25_filter" \
        --context_window "${K}"
done

echo ""
echo "All 3 window ablations complete."
ls -la results/mini/test_results_bclstm_wer25_filter_win*.json
