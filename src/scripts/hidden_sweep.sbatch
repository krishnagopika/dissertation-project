#!/bin/bash
#SBATCH --job-name=hidden_sweep
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=64G --gres=gpu:1
#SBATCH --account=wmlg --time=02:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Is hidden_dim=256 a bottleneck?
#
# Every input dimension reaches the LSTM gates (weight_ih is 1024 x 2048), so
# the model SEES all 2048. But the state carried ACROSS the dialogue is
# 2*256=512 numbers, and that was fixed by hand -- never swept -- while
# everything else was ablated. If 512 is too small to hold both modalities plus
# temporal context, every context result in this project is capacity-limited
# for a reason nobody tested.
#
# 6 sizes x 3 conditions = 18 runs, full dialogue, on the best-performing
# configuration. Parameter counts are recorded because a gain from a bigger
# state is a capacity result, not an architecture one.

set -u
source /etc/profile.d/modules.sh; source /etc/profile.d/conda.sh
export HF_HOME=/dcs/large/u5734759/hf_cache VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
cd /dcs/pg25/u5734759/dissertation-project
PY=$VENV/bin/python3.12
fail=0

for COND in gold asr asr_cleaned; do
  for H in 128 256 512 1024 1536 2048; do
    TAG="hidden_${COND}/h${H}"
    [ -f "/dcs/large/u5734759/checkpoints/bclstm/${TAG}/TRAINING_COMPLETE.json" ] && \
      { echo "--- SKIP $TAG"; continue; }
    echo ""; echo "--- TRAIN $TAG (state = 2 x $H)"
    $PY src/training/train_context.py \
        --config src/configs/ctx_${COND}.yaml \
        --tag "$TAG" --context_window -1 \
        --hidden_dim $H --num_layers 1 \
        --epochs 30 --patience 5 --batch_size 16 || fail=1
  done
done

echo ""
echo "############ RESULT ############"
$PY - <<'PYEOF'
import json, glob, os
print(f"  {'condition':<14s}" + "".join(f"{'h='+str(h):>10s}" for h in (128,256,512,1024,1536,2048)))
print("  " + "-"*74)
for c in ("gold","asr","asr_cleaned"):
    row=""
    for h in (128,256,512,1024,1536,2048):
        m=f"/dcs/large/u5734759/checkpoints/bclstm/hidden_{c}/h{h}/TRAINING_COMPLETE.json"
        row += f"{json.load(open(m))['best_dev_weighted_f1']:10.4f}" if os.path.exists(m) else f"{'·':>10s}"
    print(f"  {c:<14s}{row}")
print("\n  h=256 is what every context run in the project used.")
PYEOF
[ $fail -eq 0 ] && echo "=== HIDDEN SWEEP COMPLETE ===" || echo "=== FAILURES ==="
exit $fail
