#!/bin/bash
#SBATCH --job-name=fusion_grid
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=20:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# 25 runs in four dependent phases.
#
#   PHASE 0  learning rate x batch size                   6 runs
#            3 LRs x 2 batch sizes on ONE representative config
#            (fusion-concat + attention, ASR). Its winner is held FIXED for
#            all 19 runs below, so no later comparison can be explained by
#            one arm having had a luckier optimiser setting.
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
# Defaults; Phase 0 overwrites them with the swept winner.
BEST_LR=1e-3
BEST_BS=32
PHASE=0

run_one () {   # $1=config $2=modality $3=pooling $4=fusion $5=loss [$6=lr $7=batch]
  local cfg=$1 mod=$2 pool=$3 fus=$4 loss=$5
  local lr=${6:-$BEST_LR} bs=${7:-$BEST_BS}
  local cond=$(basename $cfg .yaml | sed 's/^fusion_//')

  # TAG is the full path under checkpoints/fusion/. It is passed to the trainer
  # with --tag, which overrides its internal naming -- otherwise the trainer
  # derives run names itself and distinct configurations collide.
  local tag
  case $PHASE in
    0) tag="phase0_lr_batch/${cond}_lr${lr}_bs${bs}" ;;
    1) tag="phase1_pooling/acoustic_${pool}" ;;
    2) tag="phase2_textonly/${cond}" ;;
    3) tag="phase3_fusion/${cond}_${fus}" ;;
    *) tag="misc/${cond}_${mod}_${pool}_${fus}_${loss}" ;;
  esac

  local done_marker="/dcs/large/u5734759/checkpoints/fusion/${tag}/TRAINING_COMPLETE.json"
  if [ -f "$done_marker" ]; then echo "--- SKIP $tag (complete)"; return; fi
  echo ""
  echo "--- TRAIN $tag   (lr=$lr bs=$bs loss=$loss)"
  $PY src/training/train_fusion_seq.py --config $cfg \
      --modality $mod --pooling $pool --fusion $fus --loss $loss \
      --lr $lr --batch_size $bs --tag "$tag" \
      --patience 3 --epochs 30 || fail=1
}

echo "############ PHASE 0 — learning rate x batch size ############"
# Swept ONCE on a single representative configuration, not crossed with the
# other 19 runs: a full cross would be 19 x 6 = 114 runs, far beyond the
# walltime, and optimiser settings are not the object of study here. The
# winner is then held FIXED across every subsequent run so that differences
# between poolers, mechanisms and text conditions cannot be attributed to
# one of them having had a luckier learning rate.
#
# Representative config: fusion-concat + attention pooling on the ASR
# condition -- it exercises both branches, the learned pooler, and the text
# path, so a setting that works here is not tuned to a degenerate case.
PHASE=0
for LR in 1e-3 5e-4 2e-4; do
  for BS in 32 64; do
    run_one src/configs/fusion_asr.yaml both attention concat weighted $LR $BS
  done
done

echo ""
echo "############ Pick Phase 0 winner ############"
read BEST_LR BEST_BS <<< $($PY - <<'PYEOF'
import json, glob, re
best, lr, bs = -1.0, "1e-3", "32"
for p in glob.glob("/dcs/large/u5734759/checkpoints/fusion/phase0_lr_batch/*/TRAINING_COMPLETE.json"):
    d = json.load(open(p))
    # A run that never wrote a checkpoint is not a candidate, however its
    # best_dev_weighted_f1 happens to compare.
    if not d.get("checkpoint_written", True):
        continue
    if d["best_dev_weighted_f1"] > best:
        best = d["best_dev_weighted_f1"]
        m = re.search(r"_lr([0-9.e-]+)_bs(\d+)$", p.split("/")[-2])
        if m:
            lr, bs = m.group(1), m.group(2)
if best < 0:
    raise SystemExit("PHASE0_NO_RESULTS")
print(lr, bs)
PYEOF
)
if [ "$BEST_LR" = "PHASE0_NO_RESULTS" ] || [ -z "$BEST_LR" ]; then
  echo "!! PHASE 0 produced no completed runs. Every run failed."
  echo "!! Aborting rather than running 19 more on fallback defaults --"
  echo "!! that is what made an ImportError look like a working grid."
  tail -30 logs/slurm_${SLURM_JOB_ID}.err 2>/dev/null | sed 's/^/     /'
  exit 1
fi
echo "  winner: lr=$BEST_LR batch=$BEST_BS  (held fixed for all runs below)"

echo ""
PHASE=1
$PY src/evaluation/fusion_report.py || true
echo "############ PHASE 1 — acoustic-only, 4 poolers ############"
# Text condition is irrelevant here (no text), so use fusion_asr arbitrarily.
for POOL in masked_mean attention_fixed attention attentive_stats; do
  run_one src/configs/fusion_asr.yaml acoustic $POOL concat weighted
done

echo ""
PHASE=2
$PY src/evaluation/fusion_report.py || true
echo "############ PHASE 2 — text-only, 3 conditions ############"
for COND in gold asr asr_cleaned; do
  run_one src/configs/fusion_${COND}.yaml text masked_mean concat weighted
done

echo ""
echo "############ Pick Phase 1 winner ############"
BEST_POOL=$($PY - <<'PYEOF'
import json, glob, os
best, name = -1.0, "masked_mean"
for p in glob.glob("/dcs/large/u5734759/checkpoints/fusion/phase1_pooling/*/TRAINING_COMPLETE.json"):
    d = json.load(open(p))
    if not d.get("checkpoint_written", True):
        continue
    if d["best_dev_weighted_f1"] > best:
        best, name = d["best_dev_weighted_f1"], d["pooling"]
if best < 0:
    raise SystemExit("PHASE1_NO_RESULTS")
print(name)
PYEOF
)
if [ "$BEST_POOL" = "PHASE1_NO_RESULTS" ] || [ -z "$BEST_POOL" ]; then
  echo "!! PHASE 1 produced no completed runs. Aborting."
  exit 1
fi
echo "  winner: $BEST_POOL"

echo ""
PHASE=3
$PY src/evaluation/fusion_report.py || true
echo "############ PHASE 3 — fusion, 4 x 3 ############"
for COND in gold asr asr_cleaned; do
  for FUS in concat sum gated crossmodal; do
    run_one src/configs/fusion_${COND}.yaml both $BEST_POOL $FUS weighted
  done
done

echo ""
echo "############ SUMMARY ############"
$PY src/evaluation/fusion_report.py || true
echo ""
$PY - <<'PYEOF'
import json, glob, os
rows=[]
for p in sorted(glob.glob("/dcs/large/u5734759/checkpoints/fusion/*/*/TRAINING_COMPLETE.json")):
    d=json.load(open(p)); cond=p.split("/")[-3].replace("fusion_","")
    rows.append((cond,d["run"],d["modality"],d["best_dev_weighted_f1"],
                 d["epochs_run"],d["train_size"],d["parameters"]["TOTAL"],
                 d.get("dev_key_hash",""),d.get("lr","?"),d.get("batch_size","?"),
                 d.get("checkpoint_written", True)))
print(f"{'cond':<12s} {'run':<40s} {'dev WF1':>8s} {'ep':>3s} {'n':>6s} "
      f"{'lr':>7s} {'bs':>4s} {'params':>10s} {'devhash':>13s}")
for r in sorted(rows, key=lambda x:-x[3]):
    flag = "" if r[10] else "  <-- NO CHECKPOINT"
    print(f"{r[0]:<12s} {r[1]:<40s} {r[3]:8.4f} {r[4]:3d} {r[5]:6d} "
          f"{str(r[8]):>7s} {str(r[9]):>4s} {r[6]:10,} {r[7]:>13s}{flag}")
bad = [r[1] for r in rows if not r[10]]
if bad:
    print(f"\n!! {len(bad)} run(s) certified complete but never saved a checkpoint: {bad}")
hashes={r[7] for r in rows if r[7]}
print(f"\ndev key-set hashes: {hashes}  {'CONSISTENT' if len(hashes)<=1 else 'DIFFER -- runs NOT comparable'}")
PYEOF

[ $fail -eq 0 ] && echo "=== FUSION GRID COMPLETE ===" || echo "=== FINISHED WITH FAILURES ==="
exit $fail
