#!/bin/bash
# Rewrite the author's absolute paths to the ones in your .env.
#
#   bash setup/configure_paths.sh          # show what would change
#   bash setup/configure_paths.sh --apply  # actually change it
#
# WHY THIS EXISTS
#   126 files in this repository (73 .sbatch, 32 .yaml, 21 .py) were written
#   against the author's Warwick HPC layout and carry absolute paths such as
#   /dcs/large/u5734759/data. Rather than pretend otherwise, this script
#   rewrites those four roots to yours in one pass. It is idempotent: running
#   it twice changes nothing the second time.
#
#   A backup of every modified file is written alongside it as <file>.orig on
#   the first --apply, so the rewrite is reversible with setup/restore_paths.sh.
set -uo pipefail
cd "$(dirname "$0")/.."

# --- the author's roots, as they appear in the committed source -------------
OLD_PROJECT=/dcs/pg25/u5734759/dissertation-project
OLD_DATA=/dcs/large/u5734759/data
OLD_CKPT=/dcs/large/u5734759/checkpoints
OLD_HF=/dcs/large/u5734759/hf_cache
OLD_VENV=/dcs/large/u5734759/venv
# three one-off locations that sit beside the roots above rather than under them
OLD_MODALENV=/dcs/large/u5734759/modal_env
OLD_MODELS=/dcs/large/u5734759/models
OLD_DUPJSON=/dcs/large/u5734759/meld_duplicate_audio.json

if [ ! -f .env ]; then
  echo "ERROR: no .env found. Copy the template and edit it first:"
  echo "         cp .env.example .env"
  exit 1
fi
set -a && source .env && set +a

missing=0
for v in PROJECT_ROOT DATA_ROOT CKPT_ROOT HF_HOME VENV; do
  val="${!v:-}"
  if [ -z "$val" ] || [[ "$val" == /absolute/path/* ]]; then
    echo "ERROR: \$$v is unset or still the placeholder in .env"; missing=1
  fi
done
[ "$missing" -eq 1 ] && exit 1

# modal_env/ and models/ lived beside data/ and checkpoints/, so derive their
# new home from DATA_ROOT's parent rather than adding two more .env variables.
SCRATCH_ROOT="$(dirname "$DATA_ROOT")"

APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

FILES=$(grep -rlE "/dcs/(large|pg25)/u5734759" \
          --include="*.py" --include="*.yaml" --include="*.yml" \
          --include="*.sbatch" --include="*.sh" src/ 2>/dev/null)

if [ -z "$FILES" ]; then
  echo "Nothing to rewrite -- paths already point somewhere else."
  exit 0
fi

n=$(echo "$FILES" | wc -l)
echo "Rewriting $n files"
echo "  $OLD_PROJECT  ->  $PROJECT_ROOT"
echo "  $OLD_DATA     ->  $DATA_ROOT"
echo "  $OLD_CKPT     ->  $CKPT_ROOT"
echo "  $OLD_HF       ->  $HF_HOME"
echo "  $OLD_VENV     ->  $VENV"
echo

if [ "$APPLY" -eq 0 ]; then
  echo "DRY RUN -- nothing written. Re-run with --apply to make the change."
  echo "Files that would be modified:"
  echo "$FILES" | sed 's/^/  /'
  exit 0
fi

BACKUP_DIR=".configure_paths_backup"
for f in $FILES; do
  mkdir -p "$BACKUP_DIR/$(dirname "$f")"
  [ -f "$BACKUP_DIR/$f" ] || cp "$f" "$BACKUP_DIR/$f"
  # DATA/CKPT/HF/VENV before PROJECT: the first four are more specific, and
  # rewriting the shorter prefix first would leave the longer ones mangled.
  sed -i \
    -e "s#${OLD_DATA}#${DATA_ROOT}#g" \
    -e "s#${OLD_CKPT}#${CKPT_ROOT}#g" \
    -e "s#${OLD_HF}#${HF_HOME}#g" \
    -e "s#${OLD_VENV}#${VENV}#g" \
    -e "s#${OLD_MODALENV}#${SCRATCH_ROOT}/modal_env#g" \
    -e "s#${OLD_MODELS}#${SCRATCH_ROOT}/models#g" \
    -e "s#${OLD_DUPJSON}#${DATA_ROOT}/meld_duplicate_audio.json#g" \
    -e "s#${OLD_PROJECT}#${PROJECT_ROOT}#g" \
    "$f"
done

left=$(grep -rlE "/dcs/(large|pg25)/u5734759" \
         --include="*.py" --include="*.yaml" --include="*.yml" \
         --include="*.sbatch" --include="*.sh" src/ 2>/dev/null | wc -l)
echo "Done. $n files rewritten; $left still contain an author path."
echo "Originals backed up under $BACKUP_DIR/ (gitignored)."
[ "$left" -gt 0 ] && {
  echo "Remaining (inspect by hand -- likely a path root not covered above):"
  grep -rlE "/dcs/(large|pg25)/u5734759" --include="*.py" --include="*.yaml" \
       --include="*.sbatch" --include="*.sh" src/ | sed 's/^/  /'
}
echo
echo "Next:  python3 setup/verify_setup.py"
exit 0
