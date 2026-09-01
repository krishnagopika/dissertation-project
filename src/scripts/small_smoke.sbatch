#!/bin/bash
#SBATCH --job-name=small_smoke
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --account=wmlg
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Can the acoustic hook coexist with Voxtral-Small's tp=2 forward pass?
#
# SETTLED (job 9708): Small LOADS. The old failures inside voxtral.py --
# AttributeError: downsample_factor, then TypeError: NoneType is not iterable
# -- were the missing consolidated.safetensors, exactly as ADR-006 predicted.
# build_llm() in transcribe_all_small.py now forces
# config_format/load_format="mistral" rather than relying on auto-detection,
# which would otherwise find Small's config.json and take the HF path.
#
# OPEN, and what this run tests: 9702 and 9708 both loaded, attached the hook,
# then FROZE on the first hooked audio forward -- tqdm elapsed stuck at 00:00,
# both Worker_TP procs at ~100% of a core, both GPUs at 100%, zero tokens.
# That is an NCCL spin-wait, not slow arithmetic.
#
# Hypothesis: Mini's hook does two blocking .cpu() copies INSIDE forward().
# At tp=1 that is merely slow. At tp=2 with NCCL_P2P_DISABLE=1 the collectives
# are staged through host memory -- the same path -- and a synchronising copy
# inside a collective-bearing forward wedges both ranks.
#
# So this uses acoustic_hook_small.py, whose hook does only asynchronous
# device-to-device clones and defers every host transfer, hash and truncation
# to drain_acoustic_hook, called between batches with no collective in flight.
# Mini's acoustic_hook.py is untouched -- proven over 13,708 clips at tp=1.
#
# Passes 24/24 -> the deferred-capture hypothesis is confirmed.
# Freezes again  -> the hook is exonerated and the tp=2 audio path is at fault.
#
# 24 dev clips, throwaway output.
# Do NOT run the full 13,708-clip extraction until this passes.
#
# gpu:2 because Small is 24B (~48 GB in bf16, more than one 46 GB L40S).

set -u
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
# NCCL settings -- established empirically, do NOT 'simplify' these:
#   9702  P2P off, customAR off -> model LOADED (only run that reached the hook)
#   9705  P2P on,  customAR on  -> hung before 'Loading safetensors'
#   9707  P2P on,  customAR off -> hung at the SAME point
# So NCCL_P2P_DISABLE=1 is the load-bearing one. `nvidia-smi topo -p2p r` reports
# OK between GPU0/GPU1, but P2P still deadlocks NCCL init here -- the capability
# report is not a guarantee (typically PCIe ACS / IOMMU). Trust the runs, not topo.
export NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0 VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
cd /dcs/pg25/u5734759/dissertation-project
mkdir -p logs

# Pick the snapshot that actually HAS the weights, rather than whichever sorts
# first. With one revision cached these agree; after a new revision is pulled
# they would not, and `head -1` could select a metadata-only snapshot.
HUB=/dcs/large/u5734759/hf_cache/hub/models--mistralai--Voxtral-Small-24B-2507
SNAP=$(find "$HUB/snapshots" -mindepth 1 -maxdepth 1 -type d \
         -exec test -e '{}/consolidated.safetensors' \; -print -quit 2>/dev/null)

echo "=== PRE-FLIGHT ==="
if [ -z "${SNAP:-}" ]; then
  echo "!!! No snapshot under $HUB/snapshots contains consolidated.safetensors."
  echo "!!! Available snapshots:"
  ls -1 "$HUB/snapshots" 2>/dev/null | sed 's/^/      /'
  echo "!!! Finish the download first. Aborting."
  exit 2
fi
echo "snapshot: $SNAP"
ls -la "$SNAP" | sed 's/^/  /'
echo "consolidated.safetensors present: $(stat -Lc %s "$SNAP/consolidated.safetensors" | awk '{printf "%.1f GB", $1/1e9}')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  /'

SMOKE=/dcs/large/u5734759/data/_smoke_small
rm -rf "$SMOKE" && mkdir -p "$SMOKE"
sed -e "s|transcripts_path:.*|transcripts_path: \"$SMOKE\"|" \
    -e "s|embeddings_path: \"/dcs/large/u5734759/data/meld_extracted/small/acoustic\"|embeddings_path: \"$SMOKE\"|" \
    src/configs/extract_small.yaml > src/configs/_smoke_small.yaml

# The sed above depends on exact formatting in extract_small.yaml. If that file
# is reformatted, the substitution silently misses and the smoke run would write
# into the REAL extraction directory. Verify rather than assume.
echo ""
echo "=== generated smoke config ==="
$VENV/bin/python3.12 - <<PY || exit 3
import sys, yaml
c = yaml.safe_load(open("src/configs/_smoke_small.yaml"))
d, m = c["data"], c["model"]
ok = True
for key in ("transcripts_path", "embeddings_path"):
    got = d[key]
    print(f"  {key:22s} {got}")
    if got != "$SMOKE":
        print(f"  !!! expected $SMOKE -- sed did not match; refusing to run")
        ok = False
print(f"  {'voxtral_id':22s} {m['voxtral_id']}")
print(f"  {'tensor_parallel_size':22s} {m['tensor_parallel_size']}")
# ravdess.embeddings_path must NOT have been rewritten by the embeddings_path sed
print(f"  {'ravdess (untouched)':22s} {c['ravdess']['embeddings_path']}")
sys.exit(0 if ok else 1)
PY

echo ""
echo "=== 24 dev clips through Voxtral-Small ==="
$VENV/bin/python3.12 src/preprocessing/transcribe_all_small.py \
    --config src/configs/_smoke_small.yaml \
    --splits dev \
    --max_samples 24 \
    --vllm_batch_size 8 &
SMOKE_PID=$!

# Liveness ping only -- says the process is alive, NOT that work is progressing.
# vLLM's own "Processed prompts" bar is the progress signal.
while kill -0 "$SMOKE_PID" 2>/dev/null; do
    sleep 30
    date +"  [%H:%M:%S] alive (see 'Processed prompts' for real progress)"
done

wait "$SMOKE_PID"
rc=$?

echo ""
echo "=== VERIFY ==="
if [ $rc -ne 0 ]; then
    echo "TRANSCRIPTION FAILED (exit $rc) — see the traceback above"
    exit $rc
fi

EXPECTED=24 $VENV/bin/python3.12 - <<PY
import json, os, sys, torch
from pathlib import Path

d = Path("$SMOKE")
expected = int(os.environ["EXPECTED"])

seq_file, tr_file = d / "dev_acoustic_seq.pt", d / "dev_transcripts.json"
for f in (seq_file, tr_file):
    if not f.exists():
        print(f"  FAIL: {f.name} was never written")
        sys.exit(1)

seq = torch.load(seq_file)
tr = json.load(open(tr_file))

# Guard before any indexing: an empty capture makes fr[0] and next(iter(...))
# raise, and the traceback would hide the actual failure.
if not seq:
    print(f"  FAIL: no acoustic sequences captured (transcripts: {len(tr)})")
    print("        the hook attached but produced nothing -- check hook errors above")
    sys.exit(1)

problems = []

# A hook miss is written as a (1, D) all-zero tensor. That is ONE failure mode;
# a clip missing from the dict entirely is another, and is invisible to it.
zero = [k for k, v in seq.items() if v.shape[0] == 1 and float(v.abs().sum()) == 0]
fr = sorted(v.shape[0] for v in seq.values())
dims = {v.shape[1] for v in seq.values()}

print(f"  clips {len(seq)}/{expected} | transcripts {len(tr)}/{expected} | zero-filled {len(zero)}")
print(f"  frames min={fr[0]} median={fr[len(fr)//2]} max={fr[-1]}  -> {fr[len(fr)//2]/50:.1f}s median")
print(f"  dim(s) {sorted(dims)} (expect [1280] -- same Whisper encoder as Mini)")

if len(seq) != expected:
    problems.append(f"expected {expected} sequences, got {len(seq)}")
if len(tr) != expected:
    problems.append(f"expected {expected} transcripts, got {len(tr)}")
if set(seq) - set(tr) or set(tr) - set(seq):
    problems.append("transcript keys and sequence keys do not match")
if zero:
    problems.append(f"{len(zero)} zero-filled (hook missed): {zero[:3]}")
if dims != {1280}:
    problems.append(f"unexpected acoustic width(s): {sorted(dims)}")
if fr[len(fr)//2] >= 1500:
    problems.append(f"median {fr[len(fr)//2]} frames -- padding was not stripped")
blank = [k for k, v in tr.items() if not str(v).strip()]
if blank:
    problems.append(f"{len(blank)} blank transcripts: {blank[:3]}")

print()
for k in list(tr)[:3]:
    print(f"  {k}: {tr[k][:70]!r}")
print()
if problems:
    print("  === CHECK ABOVE ===")
    for p in problems:
        print(f"    - {p}")
    sys.exit(1)
print("  === SMALL WORKS ===")
PY
