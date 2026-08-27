#!/bin/bash
#SBATCH --job-name=transcribe_1pass
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# SINGLE-PASS preprocessing: one Voxtral load produces both transcripts and
# acoustic frame sequences.
#
# The old path loaded Voxtral four times for a full run -- once per split under
# vllm, plus a fourth time under HF transformers to re-run an encoder forward
# vllm had already computed and thrown away. This builds the engine once and
# captures the encoder output with a forward hook.
#
# Outputs per split:
#   {split}_transcripts.json             (never overwritten -- keep-lists depend on it)
#   {split}_acoustic_seq.pt              fp16 (T, 1280), padding truncated
#   {split}_embeddings_maskedmean.pt     fp32 (1280,), the ADR-003 control
#
# The legacy {split}_embeddings.pt is left untouched.

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Required: llm.apply_model() ships the capture hook to the worker process,
# which vllm refuses under its default msgpack-only serialisation.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
mkdir -p logs

$VENV/bin/python3.12 src/preprocessing/transcribe_all.py \
    --config src/configs/mini.yaml \
    --splits ${SPLITS:-train dev test} \
    ${MAX_SAMPLES:+--max_samples $MAX_SAMPLES}

rc=$?
[ $rc -eq 0 ] && echo "=== SINGLE-PASS COMPLETE ===" \
              || echo "=== SINGLE-PASS FAILED (exit $rc) ==="
exit $rc
