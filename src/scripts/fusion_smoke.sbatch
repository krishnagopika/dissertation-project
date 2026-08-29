#!/bin/bash
#SBATCH --job-name=fusion_smoke
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=96G --gres=gpu:1
#SBATCH --account=wmlg --time=00:40:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# ONE run before the 25-run grid.
#
# Job 10040 marched through 13 runs and printed two "winners" while every
# single run died on `ImportError: cannot import name 'compute_metrics'`.
# The pickers fell back to defaults, so a totally broken grid looked healthy.
# Both pickers now abort instead -- but the real lesson is to verify one run
# end-to-end first, which costs 5 minutes.
#
# Exercises the parts most likely to break: a fusion run (both branches, the
# pooler, the text cache) and a text-only run (the isolation guard, which was
# vacuous until this session and now uses a separate unskipped probe loader).

set -u
source /etc/profile.d/modules.sh; source /etc/profile.d/conda.sh
export HF_HOME=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
cd /dcs/pg25/u5734759/dissertation-project
mkdir -p logs
PY=$VENV/bin/python3.12
rc=0

echo "=== A: fusion-concat, attention pooling, asr (both branches) ==="
$PY src/training/train_fusion_seq.py --config src/configs/fusion_asr.yaml \
    --modality both --pooling attention --fusion concat --loss weighted \
    --epochs 2 --patience 3 || rc=1

echo ""
echo "=== B: text-only, asr (exercises the isolation guard) ==="
$PY src/training/train_fusion_seq.py --config src/configs/fusion_asr.yaml \
    --modality text --pooling masked_mean --fusion concat --loss weighted \
    --epochs 2 --patience 3 || rc=1

echo ""
echo "=== VERIFY ==="
$PY - <<'PYEOF'
import json, glob, sys
ps = glob.glob("/dcs/large/u5734759/checkpoints/fusion_asr/*/TRAINING_COMPLETE.json")
if not ps:
    print("  FAIL: no TRAINING_COMPLETE.json written"); sys.exit(1)
ok = True
for p in sorted(ps):
    d = json.load(open(p))
    good = d.get("checkpoint_written") and d["best_dev_weighted_f1"] > 0.05
    print(f"  {d['run']:<44s} devWF1 {d['best_dev_weighted_f1']:.4f} "
          f"ep {d['epochs_run']} best_ep {d.get('best_epoch')} "
          f"ckpt {d.get('checkpoint_written')} n {d['train_size']} "
          f"dev {d['dev_size']} hash {d.get('dev_key_hash')}")
    ok = ok and good
print("\n  === FUSION PIPELINE OK ===" if ok else "\n  === CHECK ABOVE ===")
sys.exit(0 if ok else 1)
PYEOF
rc=$(( rc || $? ))

echo ""
echo "=== removing smoke markers so the real grid does not skip these ==="
rm -rf /dcs/large/u5734759/checkpoints/fusion_asr/fusion-concat_attention_weighted \
       /dcs/large/u5734759/checkpoints/fusion_asr/textonly_weighted
echo "  done"
exit $rc
