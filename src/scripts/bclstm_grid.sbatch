#!/bin/bash
#SBATCH --job-name=bclstm_grid
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# bc-LSTM context ablation: 3 text conditions x 5 context widths = 15 runs.
#
#              K=0   K=1   K=2   K=4   full
#   gold        .     .     .     .     .
#   asr         .     .     .     .     .
#   asr_cleaned .     .     .     .     .
#
# K is the number of neighbouring utterances visible on EACH side. "full" is
# the whole dialogue (--context_window -1), i.e. the classic bc-LSTM.
#
# What K=0 does and does not control for
# --------------------------------------
# K=0 runs the BiLSTM over a length-1 sequence, so it sees no neighbours -- but
# it KEEPS its own recurrent parameters and nonlinearity. It is therefore a
# control for CONTEXT WIDTH, not for "recurrent model vs plain MLP". Read the
# K=0 -> full trend as the value of context; do not read K=0 as a
# context-free baseline in the architectural sense.
#
# Cost note: windowed mode runs the BiLSTM over 2K+1 utterances to score ONE,
# so compute grows ~linearly in K. Full-dialogue mode scores every utterance in
# a single pass and is the cheapest of the five.
#
# All three conditions read the SAME acoustic cache and differ only in their
# text embeddings (each from that condition's own xlmr_*_plain checkpoint) and,
# for asr_cleaned, the text_clean keep-list.

set -u
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
export HF_HOME=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
cd /dcs/pg25/u5734759/dissertation-project
mkdir -p logs results_new/bclstm
PY=$VENV/bin/python3.12
fail=0

run_one () {   # $1=cond  $2=K label  $3=--context_window value
  local cond=$1 klabel=$2 kval=$3
  local tag="${cond}/${klabel}"
  local marker="/dcs/large/u5734759/checkpoints/bclstm/${tag}/TRAINING_COMPLETE.json"
  if [ -f "$marker" ]; then echo "--- SKIP $tag (complete)"; return; fi
  echo ""
  echo "--- TRAIN $tag   (context_window=$kval)"
  $PY src/training/train_context.py \
      --config src/configs/ctx_${cond}.yaml \
      --tag "$tag" --context_window $kval \
      --hidden_dim 256 --num_layers 1 \
      --epochs 30 --patience 5 --batch_size 16 || fail=1
}

for COND in gold asr asr_cleaned; do
  echo ""
  echo "############ CONDITION: $COND ############"
  for K in 0 1 2 4; do
    run_one $COND "k${K}" $K
  done
  run_one $COND "full" -1
  $PY src/evaluation/bclstm_report.py || true
done

echo ""
echo "############ SUMMARY ############"
$PY src/evaluation/bclstm_report.py || true

[ $fail -eq 0 ] && echo "=== bc-LSTM GRID COMPLETE ===" || echo "=== FINISHED WITH FAILURES ==="
exit $fail
