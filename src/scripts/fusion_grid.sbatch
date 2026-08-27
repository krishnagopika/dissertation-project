#!/bin/bash
#SBATCH --job-name=fusion_grid
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# 19 runs in three dependent phases.
#
#   PHASE 1  acoustic-only x 4 poolers                    4 runs
#            No text at all, so pooling is isolated with zero text confound.
#            Its winner sets the pooler for Phase 3.
#
#   PHASE 2  text-only x 3 text conditions                3 runs
#            Frozen [CLS] + the same head as fusion. NOT the same as the XLM-R
#            grid, which fine-tuned its encoder -- this is the FAIR unimodal
#            baseline, matched to fusion on everything except modality.
#
#   PHASE 3  fusion: 4 mechanisms x 3 text conditions    12 runs
#            Pooling fixed to Phase 1's winner.
#
# Every run: frozen Voxtral + frozen XLM-R caches, ~2M new parameters trained
# from scratch, early stopping on dev weighted F1 (patience 3), dev deliberately
# UNFILTERED so it is a fixed yardstick across all 19.
#
# --mem=96G because train_acoustic_seq.pt is 4 GB and DataLoader workers fork a
# copy. The XLM-R grid died on a quota error, so note these checkpoints are
# ~5 MB each, not 3.3 GB: no optimizer state, no per-epoch retention.

set -u
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
export HF_HOME=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
cd /dcs/pg25/u5734759/dissertation-project
mkdir -p logs
PY=$VENV/bin/python3.12
fail=0

run_one () {   # $1=config  $2=modality  $3=pooling  $4=fusion  $5=loss
  local cfg=$1 mod=$2 pool=$3 fus=$4 loss=$5
  local cond=$(basename $cfg .yaml | sed 's/^fusion_//')
  case $mod in
    acoustic) name="acousticonly_${pool}_${loss}" ;;
    text)     name="textonly_${loss}" ;;
    *)        name="fusion-${fus}_${pool}_${loss}" ;;
  esac
  local done_marker="/dcs/large/u5734759/checkpoints/fusion_${cond}/${name}/TRAINING_COMPLETE.json"
  if [ -f "$done_marker" ]; then echo "--- SKIP $cond/$name (complete)"; return; fi
  echo ""
  echo "--- TRAIN $cond / $name"
  $PY src/training/train_fusion_seq.py --config $cfg \
      --modality $mod --pooling $pool --fusion $fus --loss $loss \
      --patience 3 --epochs 30 || fail=1
}

echo "############ PHASE 1 — acoustic-only, 4 poolers ############"
# Text condition is irrelevant here (no text), so use fusion_asr arbitrarily.
for POOL in masked_mean attention_fixed attention attentive_stats; do
  run_one src/configs/fusion_asr.yaml acoustic $POOL concat weighted
done

echo ""
echo "############ PHASE 2 — text-only, 3 conditions ############"
for COND in gold asr asr_cleaned; do
  run_one src/configs/fusion_${COND}.yaml text masked_mean concat weighted
done

echo ""
echo "############ Pick Phase 1 winner ############"
BEST_POOL=$($PY - <<'PYEOF'
import json, glob, os
best, name = -1.0, "masked_mean"
for p in glob.glob("/dcs/large/u5734759/checkpoints/fusion_asr/acousticonly_*/TRAINING_COMPLETE.json"):
    d = json.load(open(p))
    if d["best_dev_weighted_f1"] > best:
        best, name = d["best_dev_weighted_f1"], d["pooling"]
print(name)
PYEOF
)
echo "  winner: $BEST_POOL"

echo ""
echo "############ PHASE 3 — fusion, 4 x 3 ############"
for COND in gold asr asr_cleaned; do
  for FUS in concat sum gated crossmodal; do
    run_one src/configs/fusion_${COND}.yaml both $BEST_POOL $FUS weighted
  done
done

echo ""
echo "############ SUMMARY ############"
$PY - <<'PYEOF'
import json, glob, os
rows=[]
for p in sorted(glob.glob("/dcs/large/u5734759/checkpoints/fusion_*/*/TRAINING_COMPLETE.json")):
    d=json.load(open(p)); cond=p.split("/")[-3].replace("fusion_","")
    rows.append((cond,d["run"],d["modality"],d["best_dev_weighted_f1"],
                 d["epochs_run"],d["train_size"],d["parameters"]["TOTAL"],d.get("dev_key_hash","")))
print(f"{'cond':<12s} {'run':<34s} {'dev WF1':>8s} {'ep':>3s} {'n':>6s} {'params':>10s} {'devhash':>13s}")
for r in sorted(rows, key=lambda x:-x[3]):
    print(f"{r[0]:<12s} {r[1]:<34s} {r[3]:8.4f} {r[4]:3d} {r[5]:6d} {r[6]:10,} {r[7]:>13s}")
hashes={r[7] for r in rows if r[7]}
print(f"\ndev key-set hashes: {hashes}  {'CONSISTENT' if len(hashes)<=1 else 'DIFFER -- runs NOT comparable'}")
PYEOF

[ $fail -eq 0 ] && echo "=== FUSION GRID COMPLETE ===" || echo "=== FINISHED WITH FAILURES ==="
exit $fail
