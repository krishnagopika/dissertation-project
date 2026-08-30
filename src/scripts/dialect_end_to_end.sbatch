#!/bin/bash
#SBATCH --job-name=dialect_e2e
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# End-to-end INFERENCE over accented English -- one job, whole pipeline.
#
#   .wav -> Voxtral-Mini encoder + LLM  -> frames (T,1280) + ASR transcript
#        -> XLM-R (per condition)        -> 768-d text
#        -> fusion x4 / bc-LSTM / ctxfusion -> emotion + sentiment
#
# NOTHING IS TRAINED. Every model is loaded from a finished checkpoint, put in
# .eval(), and run under torch.no_grad(). No optimiser is constructed anywhere
# in dialect_end_to_end.py -- grep it if in doubt.
#
# 10 models in total, both ASR conditions, no pre-selection of a "best" one:
# accented read speech is out-of-domain, so which variant generalises is the
# question rather than something to decide in advance.
#
# gpu:1 is enough -- Voxtral-MINI at tp=1. Mini is used deliberately: the MELD
# baselines were built with Mini, so using Small here would confound "accent
# shift" with "different acoustic model".
#
# Voxtral is loaded once, its outputs cached to acoustic_cache.pt, then freed
# before the classifiers load. Re-running reuses that cache and skips the
# expensive stage entirely.
#
#   sbatch src/scripts/dialect_end_to_end.sbatch

set -u

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Required for llm.apply_model() to ship the capture hook to the vLLM worker.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
cd /dcs/pg25/u5734759/dissertation-project
mkdir -p logs

# Overridable via --export=ALL,SEL=...,OUT=... so the same script runs the
# 5-clip smoke test and the full 100 without editing.
SEL="${SEL:-/dcs/large/u5734759/data/dialect_probe_100/selection.csv}"
OUT="${OUT:-/dcs/large/u5734759/data/dialect_probe_100}"

if [ ! -f "$SEL" ]; then
  echo "!!! selection not found: $SEL"
  echo "    build it with the stratified sampler first."
  exit 1
fi

echo "########## end-to-end dialect inference ##########"
echo "selection : $SEL  ($(($(wc -l < $SEL) - 1)) clips)"
echo "output    : $OUT"
echo

$VENV/bin/python3.12 src/evaluation/dialect_end_to_end.py \
    --config src/configs/extract_mini.yaml \
    --selection "$SEL" \
    --out_dir "$OUT" \
    --batch_size 8
rc=$?

echo
if [ $rc -eq 0 ]; then
  echo "=== DIALECT INFERENCE DONE ==="
  ls -la "$OUT"/dialect_predictions.csv "$OUT"/dialect_predictions_meta.json
  echo
  echo "Next: open dialect_predictions.csv, listen to each clip, and fill the"
  echo "blank human_emotion column. Metrics need that column."
else
  echo "=== FAILED (exit $rc) ==="
fi
exit $rc
