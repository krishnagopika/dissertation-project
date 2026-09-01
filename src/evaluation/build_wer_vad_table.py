#!/usr/bin/env python3.12
"""Build one CSV per split with every per-utterance field, for manual analysis.

Self-contained: computes WER, CER, VAD speech ratio and true audio duration
directly from the audio and the transcripts. Does not depend on
compute_filter_metadata.py having run.

Nothing is filtered, thresholded, or summarised. One row per utterance, every
column raw, so the file can be sorted and pivoted however you like.

Columns
-------
key                utterance key, "diaD_uttU"
dialogue_id        MELD Dialogue_ID
utterance_id       MELD Utterance_ID
sr_no              MELD "Sr No."
speaker            who said it
season, episode    source episode
start_time         MELD StartTime  (HH:MM:SS,mmm)
end_time           MELD EndTime
csv_duration_sec   EndTime - StartTime, from the MELD annotation
audio_duration_sec true decoded length of the .mp4 (may differ from the above)
gold_text          MELD reference transcript
asr_text           Voxtral output
gold_words         word count of gold
asr_words          word count of ASR
word_ratio         asr_words / gold_words
wer_exact          WER with NO normalisation — verbatim, case- and
                   punctuation-sensitive. The reproducible floor.
cer_exact          CER, same verbatim policy
wer_normalised     WER after consistent normalisation: lowercase, apostrophes
                   deleted, other punctuation replaced by a SPACE (so 'KL-5'
                   becomes 'kl 5', not 'kl5'), whitespace collapsed
cer_normalised     CER, same normalised policy
sub_norm           substitutions, normalised policy
del_norm           deletions
ins_norm           insertions
hits_norm          correct words
ref_words_norm     reference length = sub + del + hits
                   -> corpus WER = sum(sub+del+ins) / sum(ref_words_norm)
speech_ratio       Silero-VAD fraction of samples flagged as speech, 0-1
emotion            MELD emotion label
sentiment          MELD sentiment label
audio_present      whether the .mp4 was found and decoded
note               why a row has blanks, when it does

BOTH WER policies are reported, and the SAME transform is applied to the
reference and the hypothesis in each. See src/evaluation/text_normalisation.py
for why the old compute_wer.py policy was wrong: RemovePunctuation DELETES
rather than separates, so 'KL-5' collapsed to 'kl5' and an ASR output of
'KL 5' scored WER 2.0 on a one-word reference.

Usage:
    python3.12 src/evaluation/build_wer_vad_table.py \\
        --config src/configs/extract_mini.yaml --splits train dev test
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.text_normalisation import (
    EXACT, EXACT_CHARS, NORMALISED, NORMALISED_CHARS, repair_encoding,
)
from src.utils import load_config, setup_logging

TARGET_SR: int = 16_000

_SPLIT_CSV: Dict[str, str] = {
    "train": "train_sent_emo.csv",
    "dev": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv",
}

_FIELDS = [
    "key", "dialogue_id", "utterance_id", "sr_no",
    "speaker", "season", "episode",
    "start_time", "end_time", "csv_duration_sec", "audio_duration_sec",
    "gold_text", "asr_text",
    "gold_words", "asr_words", "word_ratio",
    "wer_exact", "cer_exact",
    "wer_normalised", "cer_normalised",
    "sub_norm", "del_norm", "ins_norm", "hits_norm", "ref_words_norm",
    "speech_ratio",
    "emotion", "sentiment",
    "audio_present", "note",
]


def parse_srt_time(value: str) -> Optional[float]:
    """Parse MELD's 'HH:MM:SS,mmm' timestamp into seconds.

    Args:
        value: Timestamp string from StartTime / EndTime.

    Returns:
        Seconds as float, or None if unparseable.
    """
    try:
        hms, _, ms = str(value).strip().partition(",")
        h, m, s = (int(x) for x in hms.split(":"))
        return h * 3600 + m * 60 + s + (int(ms) / 1000.0 if ms else 0.0)
    except Exception:                                          # noqa: BLE001
        return None


def decode_audio(path: Path) -> Optional[np.ndarray]:
    """Decode an .mp4 to a mono 16 kHz float32 waveform via ffmpeg.

    ffmpeg rather than torchaudio: mp4 works without torchcodec/libnvrtc, which
    are unreliable on these nodes.

    Args:
        path: Path to the audio file.

    Returns:
        1-D float32 array, or None if ffmpeg failed.
    """
    try:
        out = subprocess.run(
            ["ffmpeg", "-i", str(path), "-ar", str(TARGET_SR), "-ac", "1",
             "-f", "f32le", "pipe:1", "-loglevel", "error"],
            capture_output=True, check=True, timeout=120,
        ).stdout
        return np.frombuffer(out, dtype=np.float32).copy()
    except Exception:                                          # noqa: BLE001
        return None


def speech_ratio(waveform: np.ndarray, vad_model, get_ts) -> Optional[float]:
    """Fraction of samples Silero-VAD flags as speech.

    Args:
        waveform: 1-D float32 waveform at 16 kHz.
        vad_model: Loaded Silero model.
        get_ts: silero_vad.get_speech_timestamps.

    Returns:
        Ratio in [0, 1], or None if VAD could not run.
    """
    if waveform is None or waveform.size == 0:
        return None
    try:
        t = torch.from_numpy(waveform)
        stamps = get_ts(t, vad_model, sampling_rate=TARGET_SR)
        speech = sum(s["end"] - s["start"] for s in stamps)
        return round(speech / len(waveform), 4)
    except Exception:                                          # noqa: BLE001
        return None


def safe_rate(reference: str, hypothesis: str, transform) -> Optional[float]:
    """WER or CER under the given transform, applied to BOTH sides.

    Returns None (not 0.0) when the reference normalises away to nothing, so an
    empty gold line cannot contribute a spurious perfect score. Deliberately NOT
    the `except: return 1.0` pattern used in compute_wer.py -- scoring a failure
    as 100% turns a code error into a plausible data point and biases the mean
    upward (POSTMORTEMS.md PM-002).
    """
    import jiwer

    try:
        norm = transform(reference)
        if not norm or not norm[0]:
            return None
        return round(
            jiwer.wer(reference, hypothesis,
                      reference_transform=transform, hypothesis_transform=transform),
            6,
        )
    except Exception:                                          # noqa: BLE001
        return None


def edit_counts(reference: str, hypothesis: str, transform) -> dict:
    """Per-row substitution/deletion/insertion/hit counts and reference length.

    Summing `sub + del + ins` and dividing by summed `ref_words` across rows
    reproduces the CORPUS WER, which is the robust aggregate. Averaging the
    per-row `wer` column does not -- a one-word reference against a 176-word
    hallucination contributes 176.0 to that mean.
    """
    import jiwer

    try:
        norm = transform(reference)
        if not norm or not norm[0]:
            return {}
        o = jiwer.process_words(reference, hypothesis,
                                reference_transform=transform,
                                hypothesis_transform=transform)
        return {
            "sub_norm": o.substitutions,
            "del_norm": o.deletions,
            "ins_norm": o.insertions,
            "hits_norm": o.hits,
            "ref_words_norm": o.substitutions + o.deletions + o.hits,
        }
    except Exception:                                          # noqa: BLE001
        return {}


def build_split(
    split: str,
    meld_root: Path,
    transcripts_dir: Path,
    out_dir: Path,
    vad_model,
    get_ts,
    logger,
    limit: Optional[int] = None,
) -> Tuple[int, int]:
    """Write one CSV for a split. Returns (rows written, rows missing audio)."""
    import pandas as pd

    tr_path = transcripts_dir / f"{split}_transcripts.json"
    if not tr_path.exists():
        logger.error("%s | no transcripts at %s — skipping", split, tr_path)
        return 0, 0
    with open(tr_path, encoding="utf-8") as f:
        transcripts: Dict[str, str] = json.load(f)

    df = pd.read_csv(meld_root / _SPLIT_CSV[split])
    if limit:
        df = df.head(limit)
    audio_dir = meld_root / split

    rows: List[dict] = []
    missing = 0
    for i, r in enumerate(df.itertuples()):
        dia = int(getattr(r, "Dialogue_ID"))
        utt = int(getattr(r, "Utterance_ID"))
        key = f"dia{dia}_utt{utt}"

        # Repair cp1252->UTF-8 mojibake before ANY scoring. ~27% of MELD
        # utterances carry C1 control bytes where an apostrophe belongs; left
        # alone they score as substitutions against correct ASR.
        gold = repair_encoding(str(getattr(r, "Utterance", "") or "")).strip()
        # Repaired on BOTH sides so the preprocessing is symmetric by
        # construction rather than by the fact that Voxtral happens to emit
        # clean UTF-8. Verified a no-op: 0 of 13,708 ASR strings contain any
        # C1 control byte. Costs nothing; removes an asymmetry a reader would
        # otherwise have to re-verify.
        asr = repair_encoding(str(transcripts.get(key, "") or "")).strip()

        st = parse_srt_time(getattr(r, "StartTime", ""))
        et = parse_srt_time(getattr(r, "EndTime", ""))
        csv_dur = round(et - st, 3) if (st is not None and et is not None) else ""

        wav = decode_audio(audio_dir / f"{key}.mp4")
        present = wav is not None and wav.size > 0
        if not present:
            missing += 1

        notes = []
        if not present:
            notes.append("audio missing or undecodable")
        if not gold:
            notes.append("empty gold")
        if not asr:
            notes.append("empty ASR")

        gw, aw = len(gold.split()), len(asr.split())
        rows.append({
            "key": key,
            "dialogue_id": dia,
            "utterance_id": utt,
            "sr_no": getattr(r, "_1", ""),          # "Sr No." — not a valid identifier
            "speaker": getattr(r, "Speaker", ""),
            "season": getattr(r, "Season", ""),
            "episode": getattr(r, "Episode", ""),
            "start_time": getattr(r, "StartTime", ""),
            "end_time": getattr(r, "EndTime", ""),
            "csv_duration_sec": csv_dur,
            "audio_duration_sec": round(len(wav) / TARGET_SR, 3) if present else "",
            "gold_text": gold,
            "asr_text": asr,
            "gold_words": gw,
            "asr_words": aw,
            "word_ratio": round(aw / gw, 3) if gw else "",
            "wer_exact": safe_rate(gold, asr, EXACT) if gold else "",
            "cer_exact": safe_rate(gold, asr, EXACT_CHARS) if gold else "",
            "wer_normalised": safe_rate(gold, asr, NORMALISED) if gold else "",
            "cer_normalised": safe_rate(gold, asr, NORMALISED_CHARS) if gold else "",
            **{k: "" for k in ("sub_norm", "del_norm", "ins_norm",
                               "hits_norm", "ref_words_norm")},
            **(edit_counts(gold, asr, NORMALISED) if gold else {}),
            "speech_ratio": speech_ratio(wav, vad_model, get_ts) if present else "",
            "emotion": str(getattr(r, "Emotion", "")).strip().lower(),
            "sentiment": str(getattr(r, "Sentiment", "")).strip().lower(),
            "audio_present": int(present),
            "note": "; ".join(notes),
        })

        if (i + 1) % 500 == 0:
            logger.info("  %s | %d / %d", split, i + 1, len(df))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}_wer_vad.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        w.writerows(rows)
    logger.info("%s | %d rows (%d missing audio) → %s",
                split, len(rows), missing, out_path)
    return len(rows), missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"],
                    choices=["train", "dev", "test"])
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N utterances per split — smoke test.")
    ap.add_argument("--no_vad", action="store_true",
                    help="Skip Silero-VAD; leaves speech_ratio blank but runs much faster.")
    args = ap.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config["training"]["log_dir"], "build_wer_vad_table")

    meld_root = Path(config["data"]["meld_root"])
    transcripts_dir = Path(config["data"]["transcripts_path"])
    out_dir = Path(config["evaluation"]["output_dir"]) / "wer"

    vad_model = get_ts = None
    if not args.no_vad:
        from silero_vad import get_speech_timestamps, load_silero_vad
        vad_model, get_ts = load_silero_vad(), get_speech_timestamps
        logger.info("Silero-VAD loaded")

    logger.info("Transcripts: %s", transcripts_dir)
    logger.info("Output:      %s", out_dir)

    total = miss = 0
    for split in args.splits:
        n, m = build_split(split, meld_root, transcripts_dir, out_dir,
                           vad_model, get_ts, logger, args.limit)
        total += n
        miss += m

    # Combined file, with a split column, for pivoting across all three at once.
    combined = out_dir / "all_splits_wer_vad.csv"
    wrote = False
    with open(combined, "w", newline="", encoding="utf-8") as out:
        w = csv.DictWriter(out, fieldnames=["split"] + _FIELDS)
        w.writeheader()
        for split in args.splits:
            p = out_dir / f"{split}_wer_vad.csv"
            if not p.exists():
                continue
            with open(p, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    w.writerow({"split": split, **row})
                    wrote = True
    logger.info("%d rows total (%d missing audio)%s",
                total, miss, f" | combined → {combined}" if wrote else "")


if __name__ == "__main__":
    main()
