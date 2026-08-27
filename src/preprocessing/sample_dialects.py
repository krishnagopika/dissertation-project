#!/usr/bin/env python3.12
"""Sample a balanced set of UK/Irish accent clips for the dialect-robustness probe.

Pulls from ylacombe/english_dialects (CSTR/Google UK-Irish read-speech corpus),
which has 11 accent x gender configs. Each row carries `text` (gold transcript),
`audio`, `speaker_id` and `line_id` -- but NO emotion label. Gold text makes
per-accent WER directly measurable; emotion has to be human-annotated, which is
what the output CSV's blank columns are for.

Two deliberate implementation choices:

1. Streaming, not download. The full corpus is 8.98 GB; a 10-per-accent sample
   is ~110 clips. Streaming reads only the rows we keep.

2. decode=False + soundfile, not the datasets Audio decoder. torchcodec cannot
   load on this cluster (libnvrtc.so.13 missing), so the default decode path
   raises. Casting to decode=False hands back raw encoded bytes, which soundfile
   reads without touching torchcodec.

Usage:
    python3.12 src/preprocessing/sample_dialects.py --per_accent 10
"""
from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
from pathlib import Path

import soundfile as sf


# All 11 configs. Accent and gender are split out so results can be grouped
# either way -- accent is the thesis variable, gender is a confound to check.
CONFIGS: list[tuple[str, str, str]] = [
    ("irish_male",       "Irish",     "male"),
    ("midlands_female",  "Midlands",  "female"),
    ("midlands_male",    "Midlands",  "male"),
    ("northern_female",  "Northern",  "female"),
    ("northern_male",    "Northern",  "male"),
    ("scottish_female",  "Scottish",  "female"),
    ("scottish_male",    "Scottish",  "male"),
    ("southern_female",  "Southern",  "female"),
    ("southern_male",    "Southern",  "male"),
    ("welsh_female",     "Welsh",     "female"),
    ("welsh_male",       "Welsh",     "male"),
]

TARGET_SR = 16_000   # Voxtral/Whisper expect 16 kHz mono


def build_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    return logging.getLogger("sample_dialects")


def sample_config(
    config: str,
    accent: str,
    gender: str,
    per_accent: int,
    audio_dir: Path,
    seed: int,
    logger: logging.Logger,
) -> list[dict]:
    """Stream one accent config and keep the first `per_accent` usable clips.

    Returns a list of metadata dicts, one per saved clip.
    """
    from datasets import Audio, load_dataset

    ds = load_dataset(
        "ylacombe/english_dialects", config, split="train", streaming=True
    )
    # Raw bytes instead of decoded arrays -- see module docstring.
    ds = ds.cast_column("audio", Audio(decode=False))
    # shuffle_buffer keeps this from returning 10 clips by the same speaker
    # reading consecutive prompts, which would make the sample unrepresentative.
    ds = ds.shuffle(seed=seed, buffer_size=500)

    rows: list[dict] = []
    for row in ds:
        if len(rows) >= per_accent:
            break

        raw = row["audio"].get("bytes")
        if not raw:
            continue

        try:
            audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        except Exception as exc:                       # noqa: BLE001
            logger.warning("%s | %s | unreadable audio (%s)", config, row["line_id"], exc)
            continue

        if audio.ndim > 1:                              # stereo -> mono
            audio = audio.mean(axis=1)

        if sr != TARGET_SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)

        duration = len(audio) / TARGET_SR
        # Sub-second clips carry almost no prosody; >30 s bloats the LLM prompt.
        if duration < 1.0 or duration > 30.0:
            continue

        # line_id is the PROMPT id, not a clip id -- several speakers in the same
        # config read the same prompt, so line_id alone collides and the second
        # write silently overwrites the first (110 rows -> 108 files). speaker_id
        # is what makes it unique.
        key = f"{config}__{row['speaker_id']}__{row['line_id']}"
        out_path = audio_dir / f"{key}.wav"
        sf.write(out_path, audio, TARGET_SR, subtype="PCM_16")

        rows.append({
            "key": key,
            "config": config,
            "accent": accent,
            "gender": gender,
            "speaker_id": row["speaker_id"],
            "line_id": row["line_id"],
            "duration_s": round(duration, 2),
            "gold_text": row["text"],
            "audio_path": str(out_path),
        })

    logger.info("%-18s | %s/%s | kept %d clips", config, accent, gender, len(rows))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per_accent", type=int, default=10,
                        help="clips to sample per accent/gender config (11 configs)")
    parser.add_argument("--out_dir", type=Path,
                        default=Path("/dcs/large/u5734759/data/dialect_probe"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = build_logger()

    audio_dir = args.out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for config, accent, gender in CONFIGS:
        try:
            all_rows.extend(sample_config(
                config, accent, gender, args.per_accent, audio_dir, args.seed, logger,
            ))
        except Exception as exc:                        # noqa: BLE001
            logger.error("%s | FAILED: %s", config, exc)

    if not all_rows:
        logger.error("No clips sampled — aborting without writing manifest.")
        sys.exit(1)

    manifest = args.out_dir / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    total_dur = sum(r["duration_s"] for r in all_rows)
    logger.info("=" * 60)
    logger.info("Sampled %d clips across %d configs (%.1f min audio)",
                len(all_rows), len(CONFIGS), total_dur / 60)
    logger.info("Manifest → %s", manifest)
    logger.info("Audio    → %s", audio_dir)


if __name__ == "__main__":
    main()
