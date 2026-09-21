#!/bin/bash
# Fetch the three corpora and lay out $DATA_ROOT.
#
#   bash setup/download_data.sh            # all three
#   bash setup/download_data.sh meld       # just one
#
# Only MELD is needed for the main results. RAVDESS is used for a single
# transfer ablation (Appendix), and the dialect corpus for the out-of-domain
# evaluation. Total ~15 GB downloaded, ~30 GB on disk after extraction.
#
# Nothing here is redistributed: each corpus is fetched from its own
# publisher under its own licence, which you should read before use.
set -uo pipefail
cd "$(dirname "$0")/.."

[ -f .env ] || { echo "ERROR: no .env -- run: cp .env.example .env"; exit 1; }
set -a && source .env && set +a
[ -n "${DATA_ROOT:-}" ] || { echo "ERROR: DATA_ROOT unset in .env"; exit 1; }

WHICH="${1:-all}"
mkdir -p "$DATA_ROOT"

have() { command -v "$1" >/dev/null 2>&1; }
for t in curl tar; do have $t || { echo "ERROR: $t not found"; exit 1; }; done

# --------------------------------------------------------------------- MELD
# 9,989 train / 1,109 dev / 2,610 test utterances from *Friends*, with 7
# emotion and 3 sentiment labels. Licence: research use (see the repo).
if [ "$WHICH" = "all" ] || [ "$WHICH" = "meld" ]; then
  if [ -f "$DATA_ROOT/meld_raw/test_sent_emo.csv" ]; then
    echo "[meld]    already present, skipping"
  else
    echo "[meld]    downloading (~11 GB extracted)"
    mkdir -p "$DATA_ROOT/meld_raw"
    curl -L --fail --retry 3 -o "$DATA_ROOT/MELD.Raw.tar.gz" \
      "https://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz" || {
        echo "  download failed. The canonical source is"
        echo "    https://affective-meld.github.io/"
        echo "  Fetch MELD.Raw.tar.gz by hand into \$DATA_ROOT and re-run."
        exit 1; }
    tar -xzf "$DATA_ROOT/MELD.Raw.tar.gz" -C "$DATA_ROOT/meld_raw" --strip-components=1
    # the split archives are nested one level deeper
    for f in "$DATA_ROOT"/meld_raw/*.tar.gz; do
      [ -e "$f" ] && tar -xzf "$f" -C "$DATA_ROOT/meld_raw"
    done
    rm -f "$DATA_ROOT/MELD.Raw.tar.gz"
    echo "[meld]    done"
  fi
fi

# ------------------------------------------------------------------ RAVDESS
# 24 actors, class-balanced acted speech. Licence: CC BY-NC-SA 4.0.
if [ "$WHICH" = "all" ] || [ "$WHICH" = "ravdess" ]; then
  if [ -d "$DATA_ROOT/ravdess/Actor_01" ]; then
    echo "[ravdess] already present, skipping"
  else
    echo "[ravdess] downloading (~566 MB)"
    mkdir -p "$DATA_ROOT/ravdess"
    curl -L --fail --retry 3 -o "$DATA_ROOT/ravdess.zip" \
      "https://zenodo.org/records/1188976/files/Audio_Speech_Actors_01-24.zip?download=1" || {
        echo "  download failed. Source: https://zenodo.org/records/1188976"; exit 1; }
    have unzip && unzip -q "$DATA_ROOT/ravdess.zip" -d "$DATA_ROOT/ravdess" \
                || python3 -c "import zipfile,sys;zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
                     "$DATA_ROOT/ravdess.zip" "$DATA_ROOT/ravdess"
    rm -f "$DATA_ROOT/ravdess.zip"
    echo "[ravdess] done"
  fi
fi

# ------------------------------------------------------------------ DIALECT
# Open-source multi-speaker corpora of English accents in the British Isles
# (Demirsahin et al., LREC 2020), OpenSLR 83. Licence: CC BY-SA 4.0.
#
# The evaluation uses a 100-utterance sample drawn from this corpus and
# hand-annotated for emotion; that annotation sheet ships in the repository at
# data/dialect_annotation_sheet.csv, since it was produced for this work.
if [ "$WHICH" = "all" ] || [ "$WHICH" = "dialect" ]; then
  if [ -d "$DATA_ROOT/dialect_probe_full/audio" ]; then
    echo "[dialect] already present, skipping"
  else
    echo "[dialect] downloading OpenSLR 83 (~3.5 GB across accents)"
    mkdir -p "$DATA_ROOT/dialect_probe_full/audio"
    BASE="https://www.openslr.org/resources/83"
    for a in irish_english_male midlands_english_female midlands_english_male \
             northern_english_female northern_english_male scottish_english_female \
             scottish_english_male southern_english_female southern_english_male \
             welsh_english_female welsh_english_male; do
      echo "  $a"
      curl -L --fail --retry 3 -o "$DATA_ROOT/${a}.zip" "$BASE/${a}.zip" || {
        echo "  !! $a failed -- continuing"; continue; }
      have unzip && unzip -qo "$DATA_ROOT/${a}.zip" -d "$DATA_ROOT/dialect_probe_full/audio/$a" \
                 || python3 -c "import zipfile,sys;zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
                      "$DATA_ROOT/${a}.zip" "$DATA_ROOT/dialect_probe_full/audio/$a"
      rm -f "$DATA_ROOT/${a}.zip"
    done
    cp data/dialect_annotation_sheet.csv "$DATA_ROOT/dialect_probe_full/" 2>/dev/null || true
    echo "[dialect] done -- build the 100-clip sample with:"
    echo "          python3 src/preprocessing/sample_dialects.py --config src/configs/mini.yaml"
  fi
fi

echo
echo "Layout under $DATA_ROOT:"
for d in meld_raw ravdess dialect_probe_full; do
  printf "  %-22s %s\n" "$d" "$(du -sh "$DATA_ROOT/$d" 2>/dev/null | cut -f1 || echo absent)"
done
echo
echo "Next:  python3 setup/verify_setup.py"
