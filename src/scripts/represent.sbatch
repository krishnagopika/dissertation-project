#!/bin/bash
#SBATCH --job-name=represent
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --account=wmlg
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# RQ1 -- what emotion/sentiment information is in each modality BEFORE fusion?
#
# Operates on cached inputs only, so it needs no trained fusion model and no GPU.
#
#   linear probes     fit on TRAIN, scored per split. Primary result: probe F1
#                     measures RECOVERABLE information and is the only measure
#                     here comparable across representations of different width.
#   cosine silhouette metric="cosine", not sklearn's Euclidean default: at 768
#                     and 1280 dims Euclidean distances concentrate and drive
#                     the score to ~0 regardless of real structure.
#   PCA / UMAP        fitted on TRAIN, applied to each split, so the panels
#                     share a coordinate system and train-vs-test displacement
#                     is visible.
#   t-SNE             fitted on all splits JOINTLY -- sklearn's TSNE has no
#                     transform(), so fit-on-train is impossible. Read for local
#                     structure only; its cluster sizes and between-cluster
#                     distances are not meaningful.

set -u
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh
export VENV=/dcs/large/u5734759/venv
cd /dcs/pg25/u5734759/dissertation-project
mkdir -p logs

# Run once per TEXT condition; the acoustic side is identical throughout
# (same audio, same masked-mean pooling), so differences are text-attributable.
$VENV/bin/python3.12 src/evaluation/represent_inputs.py \
    --config src/configs/extract_mini.yaml \
    --text_dir /dcs/large/u5734759/data/meld_text_embeddings \
    --splits train test --tag _gold
rc1=$?

$VENV/bin/python3.12 src/evaluation/represent_inputs.py \
    --config src/configs/extract_mini.yaml \
    --text_dir /dcs/large/u5734759/data/meld_text_embeddings_asr \
    --splits train test --tag _asr
rc2=$?

echo ""
echo "=== FIGURES ==="
find results_new/mini/represent_inputs_* -name "*.png" 2>/dev/null | sort | sed 's/^/  /'
echo ""
echo "=== PROBE RESULTS ==="
$VENV/bin/python3.12 - <<'PYEOF'
import json, glob
for f in sorted(glob.glob("results_new/mini/represent_inputs_*/input_representations.json")):
    d = json.load(open(f)); cond = f.split("represent_inputs_")[1].split("/")[0]
    print(f"\n--- {cond} ---")
    print(f"  {'split':<6s} {'modality':<9s} {'emo WF1':>8s} {'emo macro':>10s} {'sent WF1':>9s} {'sil emo':>8s} {'sil sent':>9s}")
    for sp in ("train","test"):
        if sp not in d: continue
        for mod in ("text","acoustic"):
            m = d[sp][mod]
            print(f"  {sp:<6s} {mod:<9s} {m['probe_emotion']['weighted_f1']:8.4f} "
                  f"{m['probe_emotion']['macro_f1']:10.4f} {m['probe_sentiment']['weighted_f1']:9.4f} "
                  f"{str(m['silhouette_emotion']['overall']):>8s} {str(m['silhouette_sentiment']['overall']):>9s}")
PYEOF

[ $rc1 -eq 0 ] && [ $rc2 -eq 0 ] && echo "=== REPRESENTATION ANALYSIS COMPLETE ===" \
                                 || echo "=== FINISHED WITH FAILURES ==="
