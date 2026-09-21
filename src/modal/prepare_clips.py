#!/usr/bin/env python3.12
"""Build the MELD subset the web app's corpus picker offers.

Why a subset, and why re-encoded
--------------------------------
MELD is 11 GB of .mp4 across 13,708 clips. A picker needs a browsable handful,
and a browser needs audio it can play without a video decoder. This selects a
class-balanced sample, transcodes each clip to 16 kHz mono wav -- the exact
rate the pipeline consumes, so the app plays back what the model hears rather
than something adjacent to it -- and writes a manifest carrying the gold
labels and gold transcript.

The gold transcript is the point. WER needs a reference, and a live demo that
can only score uploads has nothing to compare against; a corpus clip arrives
with MELD's `Utterance` column attached, so the app can show ASR error and the
`wer25` quality gate for real.

Output layout, uploaded wholesale to the `emotion-clips` volume::

    manifest.json        one record per clip: key, split, labels, gold text
    audio/<key>.wav      16 kHz mono, seconds long, ~100 KB each

Selection
---------
Class-balanced over MELD's 7 emotions rather than proportional, because the
corpus is 47% neutral and a proportional sample would offer a picker that is
half neutral. Drawn from dev and test only: the app must never demo on clips
the encoder was fine-tuned on, or every number it shows is optimistic.

Usage
-----
    python3.12 src/modal/prepare_clips.py --out /tmp/modal_clips
    modal volume put emotion-clips /tmp/modal_clips /
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.text_normalisation import repair_encoding

TARGET_SR = 16_000

#: MELD ships each split's media under a differently-named directory.
SPLIT_DIR = {
    "dev": "dev_splits_complete",
    "test": "output_repeated_splits_test",
}

EMOTIONS = ("neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger")


def transcode(src: Path, dst: Path) -> float:
    """Transcode one clip to 16 kHz mono wav.

    Args:
        src: MELD ``.mp4``.
        dst: Destination ``.wav``.

    Returns:
        Duration in seconds, or 0.0 if ffmpeg could not decode the clip.
        MELD ships at least one file with no moov atom, so a failure here is
        expected occasionally and must not abort the build.
    """
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src),
         "-ac", "1", "-ar", str(TARGET_SR), str(dst)],
        capture_output=True, check=False,
    )
    if result.returncode != 0 or not dst.exists():
        dst.unlink(missing_ok=True)
        return 0.0

    # Size rather than a second ffprobe call: a 16-bit mono wav is exactly
    # 2 bytes per sample after a 44-byte header.
    return round((dst.stat().st_size - 44) / (2 * TARGET_SR), 2)


def select(meld_root: Path, per_emotion: int) -> List[Dict]:
    """Choose a class-balanced sample of dev and test utterances.

    Args:
        meld_root: Directory holding the MELD CSVs and media directories.
        per_emotion: Clips to take per emotion, per the combined dev+test pool.

    Returns:
        Records with the CSV fields needed downstream, in selection order.

    Raises:
        FileNotFoundError: If a split CSV is missing.
    """
    frames = []
    for split, subdir in SPLIT_DIR.items():
        csv = meld_root / f"{split}_sent_emo.csv"
        if not csv.exists():
            raise FileNotFoundError(f"MELD CSV not found: {csv}")
        df = pd.read_csv(csv)
        df.columns = (df.columns.str.strip().str.lower()
                      .str.replace(" ", "_", regex=False))
        df["split"] = split
        df["subdir"] = subdir
        frames.append(df)

    pool = pd.concat(frames, ignore_index=True)

    chosen = []
    for emotion in EMOTIONS:
        rows = pool[pool["emotion"] == emotion]
        # Deterministic: the same demo corpus every time it is rebuilt, so a
        # screenshot in the dissertation still matches the deployed app.
        rows = rows.sample(n=min(per_emotion, len(rows)), random_state=42)
        chosen.extend(rows.to_dict("records"))

    return chosen


def main() -> None:
    """Build the clip corpus and its manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, required=True,
                        help="Staging directory; upload this whole tree.")
    parser.add_argument("--meld-root", type=str,
                        default=os.environ.get(
                            "MELD_ROOT", "/dcs/large/u5734759/data/meld_raw"),
                        help="Directory with the MELD CSVs and media.")
    parser.add_argument("--per-emotion", type=int, default=10,
                        help="Clips per emotion class.")
    parser.add_argument("--volume", type=str, default="emotion-clips",
                        help="Modal volume name, matching serve_emotion.py.")
    args = parser.parse_args()

    meld_root = Path(args.meld_root)
    out = Path(args.out)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict] = []
    skipped = 0

    for row in select(meld_root, args.per_emotion):
        key = f"dia{int(row['dialogue_id'])}_utt{int(row['utterance_id'])}"
        src = meld_root / row["subdir"] / f"{key}.mp4"
        if not src.exists():
            skipped += 1
            continue

        dst = audio_dir / f"{key}.wav"
        duration = transcode(src, dst)
        if duration == 0.0:
            print(f"  skip {key}: ffmpeg could not decode it")
            skipped += 1
            continue

        manifest.append({
            "key": key,
            "file": f"audio/{key}.wav",
            "split": row["split"],
            "emotion": row["emotion"],
            "sentiment": row["sentiment"],
            # Repaired here rather than at request time: ~27% of MELD's
            # utterances carry cp1252->UTF-8 mojibake, and an unrepaired
            # reference scores every contraction as a substitution.
            "utterance": repair_encoding(str(row["utterance"])),
            "speaker": row.get("speaker"),
            "season": int(row["season"]) if pd.notna(row.get("season")) else None,
            "episode": int(row["episode"]) if pd.notna(row.get("episode")) else None,
            "duration_sec": duration,
        })

    with open(out / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)

    total_mb = sum(
        (audio_dir / Path(c["file"]).name).stat().st_size for c in manifest) / 1e6
    by_emotion = {e: sum(1 for c in manifest if c["emotion"] == e)
                  for e in EMOTIONS}

    print(f"\n{len(manifest)} clips, {total_mb:.1f} MB ({skipped} skipped)")
    print(f"  per emotion: {by_emotion}")
    print(f"\nUpload:\n  modal volume put {args.volume} {out} /")


if __name__ == "__main__":
    main()
