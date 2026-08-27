#!/bin/bash
#SBATCH --job-name=1pass_smoke
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=00:40:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
mkdir -p logs

# Throwaway output dir so the real caches are never touched.
SMOKE=/dcs/large/u5734759/data/_smoke_1pass
rm -rf $SMOKE && mkdir -p $SMOKE
sed -e "s|transcripts_path:.*|transcripts_path: \"$SMOKE\"|" \
    -e "s|embeddings_path:.*|embeddings_path: \"$SMOKE\"|" \
    src/configs/mini.yaml > src/configs/_smoke_1pass.yaml

$VENV/bin/python3.12 src/preprocessing/transcribe_all.py \
    --config src/configs/_smoke_1pass.yaml --splits dev --max_samples 24 \
    --vllm_batch_size 8
rc=$?

echo "=== VERIFY ==="
$VENV/bin/python3.12 - <<PY
import torch, json, sys
from pathlib import Path
sys.path.insert(0, '.')
d = Path("$SMOKE")
seq = torch.load(d/"dev_acoustic_seq.pt")
pool = torch.load(d/"dev_embeddings_maskedmean.pt")
tr = json.load(open(d/"dev_transcripts.json"))
zero = [k for k,v in seq.items() if v.shape[0]==1 and float(v.abs().sum())==0]
print(f"clips: {len(seq)} | transcripts: {len(tr)} | zero-filled (hook miss): {len(zero)}")
print(f"dim: {next(iter(seq.values())).shape[1]}  dtype: {next(iter(seq.values())).dtype}")
fr = sorted(v.shape[0] for v in seq.values())
print(f"frames: min={fr[0]} median={fr[len(fr)//2]} max={fr[-1]}  (50/sec -> {fr[len(fr)//2]/50:.1f}s median)")
mb = sum(v.numel()*v.element_size() for v in seq.values())/1e6
print(f"size: {mb:.1f} MB for {len(seq)} clips -> {mb/len(seq)*13708/1000:.1f} GB projected for all 13,708")
k = next(iter(seq))
print(f"masked-mean == mean of stored frames: {torch.allclose(pool[k], seq[k].float().mean(0), atol=1e-5)}")
print(f"sample ASR: {list(tr.values())[0][:70]!r}")
PY
exit $rc
