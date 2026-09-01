#!/bin/bash
#SBATCH --job-name=ctxfus_win
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=64G --gres=gpu:1
#SBATCH --account=wmlg --time=08:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ContextThenFusion x context width: 3 conditions x 4 windows x 3 arms = 36 runs.
#
# The full-dialogue column already exists (the original 12-cell 2x2 grid ran
# with context_window=-1), so only K=0,1,2,4 are new.
#
#            K=0   K=1   K=2   K=4   full
#   both      .     .     .     .     have
#   acoustic  .     .     .     .     have
#   text      .     .     .     .     have
#   neither   inert -- no BiLSTM to restrict --    have
#
# `neither` is excluded: with no temporal layer a window has nothing to
# restrict, so every windowed `neither` run would duplicate the full-dialogue
# control. The trainer warns if invoked that way.
#
# WHAT THIS ANSWERS
# Plain bc-LSTM peaked at k4 on gold and was WORSE at full dialogue, so distant
# utterances added noise. This grid asks whether prosodic and lexical context
# have DIFFERENT optimal widths -- if acoustic context keeps improving past the
# window where text context degrades, that is a mechanism, not a tuning detail.
#
# CAVEAT carried from the 2x2: arms are not parameter-matched (acoustic_only
# 4.34M vs text_only 3.55M, because the acoustic input is 1280-d vs 768-d), and
# these are single-seed dev numbers. The 2x2's dev ordering did NOT replicate on
# test -- asr/acoustic_only went from best on dev to worst on test -- so treat
# any ordering here as provisional until seeds are run.

set -u
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
export HF_HOME=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
cd /dcs/pg25/u5734759/dissertation-project
mkdir -p logs results_new/ctxfusion
PY=$VENV/bin/python3.12
fail=0

run_one () {   # $1=cond $2=arm $3=flags $4=Klabel $5=Kvalue
  local cond=$1 arm=$2 flags=$3 klabel=$4 kval=$5
  local tag="${cond}/${arm}_${klabel}"
  local marker="/dcs/large/u5734759/checkpoints/ctxfusion/${tag}/TRAINING_COMPLETE.json"
  if [ -f "$marker" ]; then echo "--- SKIP $tag"; return; fi
  echo ""
  echo "--- TRAIN $tag   (K=$kval)"
  $PY src/training/train_context_fusion.py \
      --config src/configs/ctxfusion_${cond}.yaml \
      --tag "$tag" --fusion concat \
      --lstm_hidden 256 --num_layers 1 --context_window $kval \
      --epochs 30 --patience 5 --batch_size 16 \
      $flags || fail=1
}

for COND in gold asr asr_cleaned; do
  echo ""
  echo "############ CONDITION: $COND ############"
  for K in 0 1 2 4; do
    run_one $COND both          "--use_acoustic_lstm --use_text_lstm" "k${K}" $K
    run_one $COND acoustic_only "--use_acoustic_lstm"                 "k${K}" $K
    run_one $COND text_only     "--use_text_lstm"                     "k${K}" $K
  done
  $PY src/evaluation/ctxfusion_report.py || true
done

echo ""
echo "############ SUMMARY ############"
$PY src/evaluation/ctxfusion_report.py || true
[ $fail -eq 0 ] && echo "=== CTXFUSION WINDOW GRID COMPLETE ===" || echo "=== FINISHED WITH FAILURES ==="
exit $fail
