#!/bin/bash
#SBATCH --job-name=meld_extract
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --account=wmlg
#SBATCH --time=10:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Single-pass MELD extraction: transcripts + acoustic frame sequences from ONE
# Voxtral load (see docs/CHANGE_RECORD.md CR-003).
#
# Parameterised by $MODEL (mini|small) via --export. Each model writes ONLY to
# data/meld_extracted/$MODEL/, so the two can never overwrite each other, and
# nothing under meld_transcripts/ or meld_embeddings/ is touched.
#
# UNFILTERED: all 13,708 utterances across train/dev/test. No VAD, no WER gate.
# Filtering is a separate downstream step and is not invoked here.
#
#   train  9,989   dev  1,109   test  2,610   ->  ~6.1 GB of fp16 sequences
#
# --mem=96G is REQUIRED. Without it the job takes the partition default and is
# OOM-killed while serialising train's ~4 GB sequence dict (job 8086 died at
# exactly that point, leaving a truncated .pt). Peak is roughly 2x the largest
# split's cache: the capture buffer plus the dict being written.

set -u
MODEL="${MODEL:?set MODEL=mini or MODEL=small via --export}"

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1   # apply_model ships the capture hook

# Multi-GPU only: these got Voxtral-Small past an NCCL init hang that stalled
# two earlier jobs for 48h. Harmless at tp=1.
export NCCL_P2P_DISABLE=1
export NCCL_CUMEM_ENABLE=0
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1

set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
cd /dcs/pg25/u5734759/dissertation-project
mkdir -p logs

echo "########## EXTRACT: $MODEL | train dev test | UNFILTERED ##########"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
df -h /dcs/large | tail -1

$VENV/bin/python3.12 src/preprocessing/transcribe_all.py \
    --config src/configs/extract_${MODEL}.yaml \
    --splits train dev test \
    --vllm_batch_size 32
rc=$?

echo; echo "=== VERIFY ($MODEL) ==="
MODEL=$MODEL $VENV/bin/python3.12 - <<'PY'
import json, os, torch
from pathlib import Path
model = os.environ["MODEL"]
root = Path(f"/dcs/large/u5734759/data/meld_extracted/{model}")
expected = {"train": 9989, "dev": 1109, "test": 2610}
ok, total_gb = True, 0.0
for split, n_exp in expected.items():
    sp = root/"acoustic"/f"{split}_acoustic_seq.pt"
    tp = root/"transcripts"/f"{split}_transcripts.json"
    if not sp.exists():
        print(f"{split:6s} MISSING {sp}"); ok = False; continue
    seq = torch.load(sp); tr = json.load(open(tp))
    um = root/"acoustic"/f"{split}_unmatched.json"
    n_un = len(json.load(open(um))) if um.exists() else 0
    fr = sorted(v.shape[0] for v in seq.values())
    gb = sum(v.numel()*v.element_size() for v in seq.values())/1e9
    total_gb += gb
    empty = sum(1 for v in tr.values() if not v.strip())
    print(f"{split:6s} clips={len(seq):5d}/{n_exp}  transcripts={len(tr):5d} "
          f"(empty {empty})  unmatched={n_un}  "
          f"frames med={fr[len(fr)//2]} ({fr[len(fr)//2]/50:.1f}s) max={fr[-1]}  {gb:.2f} GB")
    if len(seq) != n_exp or n_un:
        ok = False
print(f"total {total_gb:.2f} GB")
print(f"=== {model.upper()} EXTRACTION OK ===" if ok else f"=== {model.upper()}: CHECK FAILURES ABOVE ===")
PY
exit $rc
