#!/bin/bash
# Pull Voxtral-Small extraction outputs off Modal and verify them.
#
# Two copies by design: the Modal volume keeps one (free -- 65 GB against a
# 1 TiB allowance) and this brings a second to Warwick, where the rest of the
# pipeline reads from. Neither is a backup of the other until both are
# verified, which is what the checks below are for.
#
# Writes to meld_extracted/small/, mirroring meld_extracted/mini/ exactly, so
# every downstream config can switch model by changing one path.
set -u
export PYTHONPATH=/dcs/large/u5734759/modal_env
export PATH=/dcs/large/u5734759/modal_env/bin:$PATH
export MODAL_PROFILE=krishnagopika1701
cd /dcs/pg25/u5734759/dissertation-project

DEST=/dcs/large/u5734759/data/meld_extracted/small
mkdir -p $DEST/acoustic $DEST/transcripts

for split in dev test train; do
  for f in ${split}_acoustic_seq.pt ${split}_embeddings_maskedmean.pt; do
    if [ ! -f "$DEST/acoustic/$f" ]; then
      echo "--- fetching $f"
      modal volume get meld-extracted-small /$f $DEST/acoustic/$f 2>&1 | tail -2
    else
      echo "--- have $f"
    fi
  done
  f=${split}_transcripts.json
  if [ ! -f "$DEST/transcripts/$f" ]; then
    echo "--- fetching $f"
    modal volume get meld-extracted-small /$f $DEST/transcripts/$f 2>&1 | tail -2
  else
    echo "--- have $f"
  fi
done

echo ""
echo "=== VERIFY ==="
/dcs/large/u5734759/venv/bin/python3.12 - <<'PY'
import json, torch, pathlib
D = pathlib.Path("/dcs/large/u5734759/data/meld_extracted/small")
M = pathlib.Path("/dcs/large/u5734759/data/meld_extracted/mini")
WANT = {"train": 9989, "dev": 1109, "test": 2610}
ok = True
for split, n_csv in WANT.items():
    seq_f = D / "acoustic" / f"{split}_acoustic_seq.pt"
    tr_f = D / "transcripts" / f"{split}_transcripts.json"
    if not seq_f.exists():
        print(f"  {split:6s} NOT FETCHED"); ok = False; continue
    seq = torch.load(seq_f, map_location="cpu", weights_only=True)
    tr = json.load(open(tr_f)) if tr_f.exists() else {}
    dims = {v.shape[-1] for v in seq.values()}
    fr = sorted(v.shape[0] for v in seq.values())
    # MELD's dirs hold more .mp4 than the CSV labels (137 extra in test), so
    # coverage is measured against the CSV keys, not against the file count.
    cover = len(seq) >= n_csv
    print(f"  {split:6s} {len(seq):5d} seq | {len(tr):5d} tr | dim {sorted(dims)} | "
          f"median {fr[len(fr)//2]} frames | covers CSV({n_csv}): {cover}")
    if dims != {1280} or not cover:
        ok = False
    # Same clip must give a DIFFERENT vector than Mini -- the encoders were
    # fine-tuned separately (0/6 sampled tensors matched), so identical values
    # would mean we fetched the wrong cache.
    mini_f = M / "acoustic" / f"{split}_acoustic_seq.pt"
    if mini_f.exists():
        mini = torch.load(mini_f, map_location="cpu", weights_only=True)
        shared = set(seq) & set(mini)
        if shared:
            k = sorted(shared)[0]
            same = torch.allclose(seq[k].float(), mini[k].float()[:seq[k].shape[0]])
            print(f"         vs mini on {k}: identical={same} (must be False)")
            if same:
                ok = False
print("\n  === SMALL OUTPUTS VERIFIED ===" if ok else "\n  === CHECK ABOVE ===")
PY
du -sh $DEST 2>/dev/null | sed 's/^/  size: /'
