#!/bin/bash
#SBATCH --job-name=cleanup
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --account=wmlg
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Reclaim ~25 GB without deleting a single result.
#
#   1. Strip Adam optimizer state from finished checkpoints   ~16.3 GB
#      Every XLM-R checkpoint is 1 part weights : 2 parts Adam moments.
#      That state only lets you RESUME a run; these are all finished
#      best_model.pt files, loaded for inference and evaluation.
#      => after this they can be EVALUATED but not RESUMED.
#
#   2. Delete the duplicate Voxtral-Mini HF cache              ~8.8 GB
#      Cached twice under two HF_HOME layouts, same snapshot hash
#      3060fe34..., different inodes. Every job log resolves to hub/,
#      so the top-level copy is stale. whisper-large-v3 and
#      xlm-roberta-base exist ONLY at top level and are NOT touched.
#
#   3. Delete the single-pass smoke scratch dir                ~0.01 GB
#
# Voxtral-Small (46 GB) is deliberately KEPT.
#
# Each checkpoint runs in its own process: loading several 3.3 GB files in one
# process OOM-killed the login node. Each rewrite is verified by re-reading the
# new file and comparing a content digest of every weight tensor before the
# original is replaced (PM-001).

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
export VENV=/dcs/large/u5734759/venv
cd /dcs/pg25/u5734759/dissertation-project
mkdir -p logs

echo "########## BEFORE ##########"
quota -s -u u5734759 | tail -2
du -sh /dcs/large/u5734759/checkpoints /dcs/large/u5734759/hf_cache

CKPTS=$(find /dcs/large/u5734759/checkpoints -name "*.pt" -size +50M | sort)

echo; echo "########## 1a. DRY RUN ##########"
for f in $CKPTS; do
  $VENV/bin/python3.12 src/scripts/strip_optimizer_state.py "$f"
done

echo; echo "########## 1b. APPLY ##########"
for f in $CKPTS; do
  $VENV/bin/python3.12 src/scripts/strip_optimizer_state.py "$f" --apply || \
    echo "!!! stripping failed for $f — original left intact"
done

echo; echo "########## 2. duplicate Voxtral-Mini cache ##########"
LIVE=/dcs/large/u5734759/hf_cache/hub/models--mistralai--Voxtral-Mini-3B-2507
DUP=/dcs/large/u5734759/hf_cache/models--mistralai--Voxtral-Mini-3B-2507
# Refuse to delete unless the live copy is present and complete.
if [ -d "$LIVE/snapshots/3060fe34b35ba5d44202ce9ff3c097642914f8f3" ] && \
   [ "$(du -sm $LIVE | cut -f1)" -gt 8000 ]; then
    echo "live copy verified ($(du -sh $LIVE | cut -f1)) — removing stale duplicate"
    rm -rf "$DUP"
    rm -rf /dcs/large/u5734759/hf_cache/models--mistralai--Voxtral-Small-24B-2507  # 36K stub
else
    echo "!!! live copy NOT verified — duplicate left in place"
fi

echo; echo "########## 3. smoke scratch ##########"
rm -rf /dcs/large/u5734759/data/_smoke_1pass
rm -f  /dcs/pg25/u5734759/dissertation-project/src/configs/_smoke_1pass.yaml

echo; echo "########## AFTER ##########"
quota -s -u u5734759 | tail -2
du -sh /dcs/large/u5734759/checkpoints /dcs/large/u5734759/hf_cache
echo
echo "sanity: Voxtral-Mini still loadable from hub/?"
ls /dcs/large/u5734759/hf_cache/hub/models--mistralai--Voxtral-Mini-3B-2507/snapshots/*/ | head -3
echo "sanity: whisper + xlm-roberta untouched?"
du -sh /dcs/large/u5734759/hf_cache/models--openai--whisper-large-v3 \
       /dcs/large/u5734759/hf_cache/models--FacebookAI--xlm-roberta-base
echo "=== CLEANUP DONE ==="
