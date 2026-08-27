#!/bin/bash
#SBATCH --job-name=xlmr_grid
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# XLM-R supervised fine-tuning: 3 data conditions x 3 loss variants = 9 runs.
# Every run starts from base FacebookAI/xlm-roberta-base.
#
#   DATA                                        LOSS
#   gold        gold text,  all 9,989           plain     unweighted CE
#   asr         ASR text,   all 9,989           weighted  CE + inverse-freq alpha
#   asr_cleaned ASR text,   6,729 text_clean    focal     FocalLoss(gamma=2) + alpha
#
# `gold` is the control for the bug fixed 2026-08-26: TranscriptDataset accepted
# transcripts_path and never opened it, so every prior run trained on GOLD while
# evaluate_text_only.py scored ASR. Train/test domain mismatch nobody chose.
#
# `plain` is the control for class weighting, which was previously applied
# unconditionally -- there was no unweighted baseline to compare against.
#
# Early stopping: patience 3 on dev weighted F1 (the selection metric, not loss).
# best_model.pt already holds the best epoch, so stopping early costs nothing.
#
# EVALUATION PROTOCOL: every run is scored on the SAME ASR test set, because
# that is the deployment condition -- gold transcripts do not exist at
# inference. The training regime is the only thing that varies. Gold-trained
# models are additionally scored on gold text as a CEILING diagnostic, not as
# their headline number.
#
# Results  -> results_new/xml_roberta/{gold,asr,asr_cleaned}/
# TB logs  -> logs/xlmr/{condition}/{loss}/tb

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
mkdir -p logs results_new/xml_roberta

declare -A TEXT=( [gold]=gold [asr]=asr [asr_cleaned]=asr )
PATIENCE=3
fail=0

for COND in gold asr asr_cleaned; do
  for LOSS in plain weighted focal; do
    RUN="xlmr_${COND}_${LOSS}"
    # Gate on the completion marker, NOT on best_model.pt. The latter is
    # written at the first improving epoch, so it exists for a run that died
    # at epoch 2 -- skipping on it would evaluate a half-trained model and
    # report it as a finished result.
    DONE="/dcs/large/u5734759/checkpoints/$RUN/TRAINING_COMPLETE.json"

    echo ""
    echo "############################################################"
    echo "# $RUN   text=${TEXT[$COND]}  loss=$LOSS  patience=$PATIENCE"
    echo "############################################################"

    if [ -f "$DONE" ]; then
      echo "--- $RUN already completed:"
      cat "$DONE" | sed 's/^/      /'
      echo "--- skipping training (delete the marker to force a rerun)."
    else
      $VENV/bin/python3.12 src/training/finetune.py \
          --config src/configs/${RUN}.yaml \
          --text_source ${TEXT[$COND]} \
          --loss $LOSS \
          --patience $PATIENCE
      if [ $? -ne 0 ]; then echo "!!! TRAIN $RUN FAILED"; fail=1; continue; fi
    fi

    # ---------------- PRIMARY: one fixed test condition -----------------
    # All three training regimes are scored on the SAME ASR test set.
    #
    #            TRAIN
    #      ┌──────┼──────────┐
    #    GOLD    ASR    ASR-CLEANED
    #      └──────┼──────────┘
    #             ↓
    #       same ASR TEST      <- the deployment condition
    #
    # At inference there are no gold transcripts -- only whatever Voxtral
    # produced. Scoring the gold-trained model on gold text would measure a
    # condition that does not exist in deployment and would flatter it.
    # Holding the test condition fixed makes the three arms directly
    # comparable, and the only thing that varies is the training regime.
    for SPLIT in train dev test; do
      echo "--- EVAL $RUN | $SPLIT | text=asr (FIXED) ---"
      $VENV/bin/python3.12 src/evaluation/evaluate_text_only.py \
          --config src/configs/${RUN}.yaml \
          --split $SPLIT \
          --text_source asr \
          --tag "_${LOSS}_${SPLIT}_asrtest" || fail=1
    done

    # ---------------- DIAGNOSTIC: gold-on-gold ---------------------------
    # Not the headline number. It answers a different question -- "how well
    # would this do if transcription were perfect?" -- so it is the ceiling,
    # and the gap to the ASR-test number above is the cost of ASR error.
    if [ "$COND" = "gold" ]; then
      echo "--- EVAL $RUN | test | text=gold (CEILING, diagnostic) ---"
      $VENV/bin/python3.12 src/evaluation/evaluate_text_only.py \
          --config src/configs/${RUN}.yaml --split test \
          --text_source gold --tag "_${LOSS}_test_goldceiling" || fail=1
    fi
  done
done

echo ""
echo "=========================================================="
[ $fail -eq 0 ] && echo "=== GRID COMPLETE (9 runs) ===" \
                || echo "=== FINISHED WITH FAILURES — see !!! lines ==="
for d in gold asr asr_cleaned; do
  echo "--- results_new/xml_roberta/$d ---"
  ls -1 results_new/xml_roberta/$d 2>/dev/null | sed 's/^/    /'
done
exit $fail
