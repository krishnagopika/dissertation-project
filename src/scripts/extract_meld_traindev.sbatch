#!/bin/bash
#SBATCH --job-name=meld_extract
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=06:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Single-pass extraction of MELD TRAIN + DEV into data/meld_extracted/.
# Test is deliberately excluded — held for a separate run.
#
#   train  9,989 utterances
#   dev    1,109 utterances
#   -----------------------
#          11,098 clips  ->  ~5 GB of fp16 sequences
#
# One Voxtral load produces transcripts AND acoustic sequences (CR-003).
# Nothing under meld_transcripts/ or meld_embeddings/ is touched.

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1   # apply_model ships the capture hook

set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
mkdir -p logs

$VENV/bin/python3.12 src/preprocessing/transcribe_all.py \
    --config src/configs/mini_extract.yaml \
    --splits train dev \
    --vllm_batch_size 32
rc=$?

echo "=== VERIFY ==="
$VENV/bin/python3.12 - <<'PY'
import json, torch
from pathlib import Path
root = Path("/dcs/large/u5734759/data/meld_extracted")
expected = {"train": 9989, "dev": 1109}
ok = True
for split, n_exp in expected.items():
    sp = root/"acoustic"/f"{split}_acoustic_seq.pt"
    tp = root/"transcripts"/f"{split}_transcripts.json"
    if not sp.exists():
        print(f"{split}: MISSING {sp}"); ok = False; continue
    seq = torch.load(sp); tr = json.load(open(tp))
    um = root/"acoustic"/f"{split}_unmatched.json"
    n_un = len(json.load(open(um))) if um.exists() else 0
    fr = sorted(v.shape[0] for v in seq.values())
    mb = sum(v.numel()*v.element_size() for v in seq.values())/1e6
    empty_tr = sum(1 for v in tr.values() if not v.strip())
    print(f"{split:6s} clips={len(seq):5d}/{n_exp}  transcripts={len(tr):5d} "
          f"(empty {empty_tr})  unmatched={n_un}  "
          f"frames med={fr[len(fr)//2]} ({fr[len(fr)//2]/50:.1f}s)  {mb/1000:.2f} GB")
    if len(seq) != n_exp or n_un: ok = False
print("=== EXTRACTION OK ===" if ok else "=== CHECK FAILURES ABOVE ===")
PY
exit $rc
