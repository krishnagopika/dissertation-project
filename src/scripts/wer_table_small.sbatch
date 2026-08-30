#!/bin/bash
#SBATCH --job-name=wer_small
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Per-utterance WER/VAD table + defect analysis + keep-lists for Voxtral-SMALL.
#
# This is the SAME two-stage path Mini went through, so Small's asr_cleaned
# condition is built by identical code and the two are comparable:
#
#   build_wer_vad_table.py  -> results_new/small/wer/{split}_wer_vad.csv
#   analyse_wer_vad.py      -> results_new/small/wer/{split}_defect_summary.json
#                              .../filter_keys/{split}_keys_text_clean.json
#
# Every path comes from extract_small.yaml, which writes only under
# meld_extracted/small/ and results_new/small/. Mini's transcripts, keep-lists
# and results are never touched, so the WER comparison between the two models
# is between two independently produced sets.
#
# GPU is for Silero-VAD over 13,708 clips; it runs on CPU but far slower.
#
#   sbatch src/scripts/wer_table_small.sbatch

set -u

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

CFG=src/configs/extract_small.yaml

echo "########## 1/2  per-utterance WER + VAD table (small) ##########"
$VENV/bin/python3.12 src/evaluation/build_wer_vad_table.py \
    --config $CFG --splits train dev test
rc=$?
[ $rc -ne 0 ] && { echo "!!! table failed (exit $rc)"; exit $rc; }

echo; echo "########## 2/2  defect analysis + keep-lists ##########"
$VENV/bin/python3.12 src/evaluation/analyse_wer_vad.py \
    --config $CFG --splits train dev test
rc=$?

echo
if [ $rc -eq 0 ]; then
  echo "=== SMALL WER/VAD DONE ==="
  ls -la results_new/small/wer/ 2>/dev/null
  echo "--- keep-lists ---"
  ls -la /dcs/large/u5734759/data/meld_extracted/small/filter_keys/ 2>/dev/null
else
  echo "=== FAILED (exit $rc) ==="
fi
exit $rc
