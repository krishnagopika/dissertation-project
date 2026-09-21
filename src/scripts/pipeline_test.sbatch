#!/bin/bash
#SBATCH --job-name=pipeline_test
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=0-01:30:00
#SBATCH --output=logs/pipeline_test_%j.out
#SBATCH --error=logs/pipeline_test_%j.err

# Run EVERY pipeline stage on a minimal workload, to prove the code in this
# bundle executes end to end. This is not a reproduction of the results -- it
# answers "does each stage start, read its inputs, do work, and write output".
#
#   mkdir -p logs && sbatch setup/pipeline_test.sbatch
#
# NOTE the mkdir. Slurm opens --output/--error BEFORE the script body runs, so
# if logs/ does not exist the job is killed at launch and writes nothing at all
# -- no log, no error, and an empty queue. The mkdir inside the script cannot
# help, because it never gets to run.
#
# SAFETY: every stage writes under $SCRATCH (a fresh directory under the
# submission tree). No real cache, checkpoint or results file is touched.
# Stages that only read -- WER analysis, evaluation -- use the real caches.
#
# Expected wall time: 30-60 min, dominated by stage 1 (Voxtral loads a 9 GB
# model to transcribe four clips).

set -uo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
source /etc/profile.d/conda.sh   2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f .env ] || { echo "FATAL: no .env"; exit 1; }
set -a && source .env && set +a
mkdir -p logs

# vllm refuses to serialise the acoustic capture hook to its worker process
# without this. src/scripts/transcribe_single_pass.sbatch sets it too.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

PY="${VENV}/bin/python3.12"
[ -x "$PY" ] || { echo "FATAL: no interpreter at $PY"; exit 1; }

SCRATCH="$(pwd)/.pipeline_test"
rm -rf "$SCRATCH"; mkdir -p "$SCRATCH"/{transcripts,embeddings,results,ckpt,filter_keys}

echo "=============================================================="
echo " node    : $(hostname)"
echo " python  : $($PY --version 2>&1)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/ gpu     : /'
echo " scratch : $SCRATCH"
echo "=============================================================="

pass=0; fail=0
stage () {                      # stage <name> <command...>
  local name="$1"; shift
  echo; echo "--------------------------------------------------------------"
  echo "### $name"
  echo "--------------------------------------------------------------"
  local t0=$SECONDS
  if "$@"; then
    echo ">>> $name PASSED  [$((SECONDS-t0))s]"; pass=$((pass+1))
  else
    echo ">>> $name FAILED (exit $?)  [$((SECONDS-t0))s]"; fail=$((fail+1))
  fi
}

# --- a config whose OUTPUT paths point at scratch, inputs at the real data ---
SMOKE_CFG="$SCRATCH/smoke.yaml"
$PY - "$SMOKE_CFG" "$SCRATCH" <<'PYEOF'
import sys, re, pathlib
out, scratch = sys.argv[1], sys.argv[2]
src = pathlib.Path("src/configs/mini.yaml").read_text()
# redirect only the things a stage WRITES; leave meld_root and caches alone
src = re.sub(r'(transcripts_path:\s*)"[^"]*"', rf'\1"{scratch}/transcripts"', src)
src = re.sub(r'(\n\s*embeddings_path:\s*)"[^"]*meld_embeddings"', rf'\1"{scratch}/embeddings"', src)
src = re.sub(r'(output_dir:\s*)"[^"]*"', rf'\1"{scratch}/results"', src)
src = re.sub(r'(checkpoint_dir:\s*)"[^"]*"', rf'\1"{scratch}/ckpt"', src)
src = re.sub(r'(keys_dir:\s*)"[^"]*"', rf'\1"{scratch}/filter_keys"', src)
pathlib.Path(out).write_text(src)
print(f"  wrote {out} (outputs -> scratch, inputs unchanged)")
PYEOF

# =============================== STAGE 1 ====================================
# Voxtral: one pass yields transcript AND acoustic frames. Four clips only.
stage "1. transcribe + extract (4 clips)" \
  $PY src/preprocessing/transcribe_all.py \
      --config "$SMOKE_CFG" --splits test --max_samples 4

# =============================== STAGE 2 ====================================
# analyse_wer_vad reads the per-utterance table build_wer_vad_table.py writes.
# Seed scratch with the real table so the analysis runs on genuine input
# without recomputing WER over every clip.
mkdir -p "$SCRATCH/results/wer"
cp results_new/mini/wer/test_wer_vad.csv "$SCRATCH/results/wer/" 2>/dev/null \
  || echo "  (no cached WER table to seed from; stage 2 will report it)"

stage "2. WER / VAD analysis" \
  $PY src/evaluation/analyse_wer_vad.py --config "$SMOKE_CFG" --splits test

# =============================== STAGE 3 ====================================
# bc-LSTM on the cached features: one epoch, tagged into scratch.
stage "3. bc-LSTM context (1 epoch)" \
  $PY src/training/train_context.py \
      --config src/configs/ctx_asr.yaml \
      --tag "$SCRATCH/ckpt/bclstm_smoke" \
      --context_window 0 --epochs 1 --patience 1 \
      --hidden_dim 256 --num_layers 1 --batch_size 16

# =============================== STAGE 4 ====================================
# context-then-fusion: the other downstream family.
stage "4. context-then-fusion (1 epoch)" \
  $PY src/training/train_context_fusion.py \
      --config src/configs/ctxfusion_asr.yaml \
      --tag "$SCRATCH/ckpt/ctxfusion_smoke" \
      --fusion concat --epochs 1 --patience 1 \
      --lstm_hidden 256 --num_layers 1 --batch_size 16

# =============================== STAGE 5 ====================================
# Score checkpoints on the test split, then run the validation gate.
stage "5. evaluate_all (bclstm family)" \
  $PY -m src.evaluation.evaluate_all \
      --families bclstm --out "pipeline_test_$SLURM_JOB_ID.json"

stage "6. validation gate" \
  $PY src/evaluation/check_windowed_eval.py \
      --table "pipeline_test_$SLURM_JOB_ID.json"

# =============================== STAGE 7 ====================================
# The chain-level smoke test, for completeness.
stage "7. chain smoke test" $PY setup/smoke_test.py

echo
echo "=============================================================="
echo " STAGES PASSED : $pass"
echo " STAGES FAILED : $fail"
echo " scratch kept at $SCRATCH for inspection (delete when done)"
echo "=============================================================="
[ "$fail" -eq 0 ] && echo " PIPELINE TEST PASSED" || echo " PIPELINE TEST FAILED"
exit $fail
