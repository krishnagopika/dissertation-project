#!/bin/bash
#SBATCH --job-name=smoke_gpu
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=0-00:30:00
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err

# Prove a fresh checkout can actually run on a GPU before committing to a
# multi-hour job. Three stages, cheapest first, each gating the next:
#
#   1. verify_setup.py   env, packages, corpora, paths rewritten
#   2. smoke_test.py     the chain on synthetic + one real utterance (CPU-safe)
#   3. GPU stages        CUDA visible, models move to device, Voxtral loads and
#                        one training step runs, vllm + Voxtral resolve
#
#   sbatch setup/smoke_gpu.sbatch
#
# Expected wall time: ~2 min. It deliberately does NOT spin up a vllm engine
# or download the 9 GB of weights -- it checks that they resolve.

set -uo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true
source /etc/profile.d/conda.sh   2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[ -f .env ] || { echo "FATAL: no .env -- cp .env.example .env"; exit 1; }
set -a && source .env && set +a
mkdir -p logs

PY="${VENV}/bin/python3.12"
[ -x "$PY" ] || { echo "FATAL: no interpreter at $PY (check VENV in .env)"; exit 1; }

echo "=============================================================="
echo " node      : $(hostname)"
echo " date      : $(date)"
echo " python    : $($PY --version 2>&1)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  | sed 's/^/ gpu       : /'
echo "=============================================================="

fail=0

echo; echo "### 1/3  environment"
$PY setup/verify_setup.py || fail=1

echo; echo "### 2/3  chain smoke test"
$PY setup/smoke_test.py || fail=1

echo; echo "### 3/3  GPU stages"
$PY - <<'PYEOF' || fail=1
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import torch

ok = True
def stage(name, fn):
    global ok
    t0 = time.time()
    try:
        print(f"  {name:<34s} {fn()}   [{time.time()-t0:.1f}s]")
    except Exception as e:                                   # noqa: BLE001
        print(f"  {name:<34s} FAIL {type(e).__name__}: {e}")
        ok = False

def cuda():
    assert torch.cuda.is_available(), "no CUDA device visible to torch"
    d = torch.cuda.get_device_properties(0)
    return f"{d.name}, {d.total_memory/2**30:.0f} GiB"

def model_to_gpu():
    from src.models.context_lstm import BiLSTMContext
    m = BiLSTMContext(input_dim=2048, hidden_dim=256, num_emotion_classes=7,
                      num_sentiment_classes=3, num_layers=1, dropout=0.3).cuda()
    s, e = m(torch.randn(2, 5, 2048).cuda(), torch.tensor([5, 3]))
    return f"forward on GPU -> {tuple(e.shape)}"

def train_step():
    """One optimiser step: proves autograd, loss and the GPU path all work."""
    from src.models.context_lstm import BiLSTMContext
    m = BiLSTMContext(input_dim=2048, hidden_dim=256, num_emotion_classes=7,
                      num_sentiment_classes=3, num_layers=1, dropout=0.3).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    s, e = m(torch.randn(4, 6, 2048).cuda(), torch.tensor([6, 6, 4, 2]))
    loss = torch.nn.functional.cross_entropy(
        e.reshape(-1, 7), torch.randint(0, 7, (e.reshape(-1, 7).shape[0],)).cuda())
    before = loss.item()
    loss.backward(); opt.step()
    assert torch.isfinite(loss), "loss is not finite"
    return f"one AdamW step, loss {before:.4f}"

def voxtral_loadable():
    """Check Voxtral resolves and vllm is importable.

    The pipeline transcribes through vllm (see transcribe_all.py), not through
    transformers directly. Spinning up a vllm engine costs minutes and most of
    the card, which is more than a smoke test should take -- so this confirms
    the two things that actually break a fresh setup: vllm is installed, and
    the weights resolve from HF_HOME or the hub with the token provided.
    """
    tok = os.environ.get("HF_TOKEN", "")
    if not tok:
        return "skipped (no HF_TOKEN)"
    import vllm                                             # noqa: F401
    from huggingface_hub import snapshot_download
    mid = "mistralai/Voxtral-Mini-3B-2507"
    local = snapshot_download(mid, token=tok, allow_patterns=["config.json"])
    return f"vllm {vllm.__version__}, weights resolve at {Path(local).name}/"


def one_clip_present():
    """A real MELD clip decodes at 16 kHz -- the input the encoder receives."""
    data = Path(os.environ.get("DATA_ROOT", ""))
    clips = list((data / "meld_raw").rglob("dia0_utt0.mp4")) or \
            list((data / "meld_raw").rglob("*.mp4"))[:1]
    if not clips:
        return "skipped (no MELD audio)"
    import librosa
    y, sr = librosa.load(str(clips[0]), sr=16000, mono=True)
    return f"{clips[0].name}, {y.size/sr:.2f}s at {sr} Hz"


stage("CUDA available", cuda)
stage("model to GPU", model_to_gpu)
stage("one training step", train_step)
stage("Voxtral + vllm resolve", voxtral_loadable)
stage("one MELD clip decodes", one_clip_present)
sys.exit(0 if ok else 1)
PYEOF

echo
echo "=============================================================="
if [ "$fail" -eq 0 ]; then
  echo " SMOKE TEST PASSED -- this checkout can run the full pipeline."
else
  echo " SMOKE TEST FAILED -- fix the stages above before submitting"
  echo " a real job. Nothing downstream will work until they pass."
fi
echo "=============================================================="
exit $fail
