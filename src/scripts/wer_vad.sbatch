#!/bin/bash
#SBATCH --job-name=wer_vad
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Per-utterance WER + VAD metadata, then the report.
#
# MEASUREMENT ONLY -- nothing is filtered. compute_filter_metadata.py scores
# every utterance; apply_filter.py (not run here) is what turns thresholds into
# keep-lists. The report's "survival" table is informational: it shows what a
# threshold WOULD discard so the cost is visible before committing to one.
#
# Parameterised by $MODEL (mini|small) via --export, reading that model's own
# transcripts and writing to that model's own filter_metadata/ directory.
#
# GPU is for Silero-VAD; it runs on CPU too but is much slower over 13,708 clips.

set -u
MODEL="${MODEL:?set MODEL=mini or MODEL=small via --export}"

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
cd /dcs/pg25/u5734759/dissertation-project
mkdir -p logs

CFG=src/configs/extract_${MODEL}.yaml

echo "########## 1/2  metadata: VAD + WER per utterance ($MODEL) ##########"
$VENV/bin/python3.12 src/preprocessing/compute_filter_metadata.py --config $CFG
rc=$?
[ $rc -ne 0 ] && { echo "!!! metadata failed (exit $rc)"; exit $rc; }

echo; echo "########## 2/2  report ##########"
$VENV/bin/python3.12 src/evaluation/wer_vad_report.py \
    --config $CFG --splits train dev test
rc=$?

echo
if [ $rc -eq 0 ]; then
  echo "=== WER/VAD REPORT DONE ($MODEL) ==="
  echo "  results/extract_${MODEL}/wer_vad/report.md"
  ls -la results/extract_${MODEL}/wer_vad/ 2>/dev/null
else
  echo "=== REPORT FAILED (exit $rc) ==="
fi
exit $rc
