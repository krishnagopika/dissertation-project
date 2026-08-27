#!/bin/bash
#SBATCH --job-name=dialect_full
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Transcribe the FULL english_dialects corpus (~17,877 clips, 11 accents) with
# Voxtral Mini, and classify each clip zero-shot.
#
# Pass 1 (ASR) is the point of this run: the corpus ships gold transcripts, so
# per-accent WER is measurable with no annotation. The 110-clip pilot gave
# Midlands 0.036 -> Irish 0.238; the full corpus turns that into a result with
# real sample sizes behind each accent.
#
# Pass 2 (zero-shot emotion) is kept because it is cheap (32 output tokens vs
# 256 for ASR) and the pilot returned only 9 non-neutral predictions out of 110
# -- scanning all 17,877 is the only way to accumulate enough non-neutral cases
# to populate every emotion category per accent.
#
# tp=1, gpu:1 -- mini.yaml's working configuration. Walltime 4h against a
# ~1.5h estimate extrapolated from the pilot's throughput.

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
    --probe_dir /dcs/large/u5734759/data/dialect_probe_full \
    --batch_size 32

rc=$?
[ $rc -eq 0 ] && echo "=== FULL DIALECT TRANSCRIPTION COMPLETE ===" \
              || echo "=== FULL DIALECT TRANSCRIPTION FAILED (exit $rc) ==="
exit $rc
