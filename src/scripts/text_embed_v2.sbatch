#!/bin/bash
#SBATCH --job-name=text_embed_v2
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=48G --gres=gpu:1
#SBATCH --account=wmlg --time=02:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Regenerate the frozen text embeddings the fusion grid consumes.
#
# WHY: the existing caches are stale and from the WRONG encoders.
#   meld_text_embeddings      Jul 30
#   meld_text_embeddings_asr  Jun 18
# Both predate the Aug 26 XLM-R grid, so they came from checkpoints trained
# through the TranscriptDataset bug (transcripts_path ignored -> trained on
# gold, evaluated on ASR). Fusion built on them would inherit that.
#
# One cache per condition, each from THAT condition's own best checkpoint.
# `plain` wins every condition on dev weighted F1, so the loss is held constant
# across conditions and the comparison is not confounded by loss choice:
#   gold        plain 0.5733   (weighted 0.5565, focal 0.5617)
#   asr         plain 0.4594   (weighted 0.4409, focal 0.4445)
#   asr_cleaned plain 0.4769   (focal 0.4366; weighted failed on quota)
#
# gold uses the CSV Utterance column; asr and asr_cleaned use the SAME Voxtral
# transcripts -- they differ by which utterances the keep-list admits, not by
# text. They still need separate caches because the ENCODER differs.

set -u
source /etc/profile.d/modules.sh; source /etc/profile.d/conda.sh
export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
cd /dcs/pg25/u5734759/dissertation-project
mkdir -p logs
PY=$VENV/bin/python3.12
CK=/dcs/large/u5734759/checkpoints
OUT=/dcs/large/u5734759/data/meld_text_emb_v2
fail=0

echo "=== gold (CSV Utterance) <- xlmr_gold_plain ==="
$PY src/preprocessing/extract_text_embeddings.py \
    --config src/configs/xlmr_gold_plain.yaml \
    --checkpoint_path $CK/xlmr_gold_plain/best_model.pt \
    --out_dir $OUT/gold || fail=1

echo ""
echo "=== asr (Voxtral transcripts) <- xlmr_asr_plain ==="
$PY src/preprocessing/extract_text_embeddings_asr.py \
    --config src/configs/xlmr_asr_plain.yaml \
    --checkpoint_path $CK/xlmr_asr_plain/best_model.pt \
    --out_dir $OUT/asr || fail=1

echo ""
echo "=== asr_cleaned (same transcripts, cleaned-subset encoder) <- xlmr_asr_cleaned_plain ==="
$PY src/preprocessing/extract_text_embeddings_asr.py \
    --config src/configs/xlmr_asr_cleaned_plain.yaml \
    --checkpoint_path $CK/xlmr_asr_cleaned_plain/best_model.pt \
    --out_dir $OUT/asr_cleaned || fail=1

echo ""
echo "=== VERIFY ==="
$PY - <<'PYEOF'
import torch, pathlib
root = pathlib.Path("/dcs/large/u5734759/data/meld_text_emb_v2")
ok = True
for cond in ("gold", "asr", "asr_cleaned"):
    for split, want in (("train", 9989), ("dev", 1109), ("test", 2610)):
        f = root / cond / f"{split}_text_embeddings.pt"
        if not f.exists():
            print(f"  MISSING {f}"); ok = False; continue
        d = torch.load(f, map_location="cpu", weights_only=True)
        v = next(iter(d.values()))
        flag = "" if len(d) == want else f"  <-- expected {want}"
        print(f"  {cond:12s} {split:6s} {len(d):5d} keys, dim {tuple(v.shape)}{flag}")
        if len(d) != want or v.shape[-1] != 768:
            ok = False
# asr and asr_cleaned share text but differ in encoder -> vectors MUST differ
a = torch.load(root/"asr"/"dev_text_embeddings.pt", map_location="cpu", weights_only=True)
c = torch.load(root/"asr_cleaned"/"dev_text_embeddings.pt", map_location="cpu", weights_only=True)
k = next(iter(a))
same = torch.allclose(a[k], c[k])
print(f"\n  asr vs asr_cleaned identical vectors: {same}  (must be False -- different encoders)")
print("\n  === TEXT EMBEDDINGS V2 OK ===" if ok and not same else "\n  === CHECK ABOVE ===")
PYEOF
exit $fail
