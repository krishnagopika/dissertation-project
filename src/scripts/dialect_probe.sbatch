#!/bin/bash
#SBATCH --job-name=dialect_probe
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Runs Voxtral Mini over the 110 sampled UK/Irish accent clips:
#   Pass 1 -- ASR  -> per-accent WER against the corpus gold transcripts
#   Pass 2 -- zero-shot emotion/sentiment -> manual annotation sheet
#
# gpu:1 and tp=1 on purpose: mini.yaml sets tensor_parallel_size=1, which is
# the configuration that has always worked here. Nothing about this job needs
# multi-GPU -- 110 clips is ~11 minutes of audio.

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
mkdir -p logs

$VENV/bin/python3.12 src/evaluation/dialect_probe.py \
    --config src/configs/mini.yaml \
    --probe_dir /dcs/large/u5734759/data/dialect_probe \
    --batch_size 16

rc=$?
[ $rc -eq 0 ] && echo "=== DIALECT PROBE COMPLETE ===" \
              || echo "=== DIALECT PROBE FAILED (exit $rc) ==="
exit $rc
