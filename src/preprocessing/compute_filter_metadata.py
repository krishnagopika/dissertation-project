"""
compute_filter_metadata.py — Per-utterance VAD + WER metadata for MELD
=======================================================================
For every MELD utterance across every split, computes:

  * Silero-VAD  speech_ratio  ∈ [0, 1]      (fraction of audio flagged as speech)
  * Voxtral   WER  vs the gold MELD `utterance` text
  * Whisper   WER  vs the gold MELD `utterance` text  (optional; run if cached
                                                       Whisper transcripts exist)
  * gold_words, pred_words, duration_sec

This is a one-off, expensive-ish pass (~15-30 min per split on 1 GPU).
Downstream ``apply_filter.py`` reads this metadata and produces filtered
key-lists at arbitrary thresholds without re-decoding audio.

Outputs
-------
  {filtering.metadata_dir}/{split}_filter_metadata.json
    [{"key": "dia0_utt0",
      "gold_words": 5,  "pred_words": 5,
      "vox_wer": 0.05,  "whisper_wer": 0.02,
      "speech_ratio": 0.87,  "duration_sec": 2.4}, ...]

Usage
-----
  python3.12 src/preprocessing/compute_filter_metadata.py --config src/configs/mini.yaml

Slurm: see src/scripts/compute_filter_metadata.sbatch
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import jiwer
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import load_config, set_seed, setup_logging


_TARGET_SR: int = 16_000
_SPLIT_CSV: Dict[str, str] = {
    "train": "train_sent_emo.csv",
    "dev":   "dev_sent_emo.csv",
    "test":  "test_sent_emo.csv",
}


# ---------------------------------------------------------------------------
# Audio helpers (reused pattern from transcribe_all.py)
# ---------------------------------------------------------------------------

def load_audio_mono_16k(
    audio_path: Path,
    max_duration_sec: float,
) -> Optional[torch.Tensor]:
    """Load audio via ffmpeg → 1-D float32 tensor at 16 kHz, mono, truncated.

    Args:
        audio_path: Path to .mp4 file.
        max_duration_sec: Maximum audio duration in seconds.

    Returns:
        1-D float32 tensor, or None if the file failed to decode.
    """
    if not audio_path.exists():
        return None
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-i", str(audio_path),
                "-ar", str(_TARGET_SR),
                "-ac", "1",
                "-t", str(max_duration_sec),
                "-f", "f32le",
                "pipe:1",
                "-loglevel", "error",
            ],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return None
    return torch.from_numpy(
        np.frombuffer(result.stdout, dtype=np.float32).copy()
    )


# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------

def load_vad_model() -> object:
    """Load the Silero-VAD model (pip: silero-vad).

    Returns:
        The Silero-VAD model instance (callable, but we use the utility API).
    """
    from silero_vad import load_silero_vad
    return load_silero_vad()


def compute_speech_ratio(
    waveform: torch.Tensor,
    model,
    threshold: float = 0.5,
) -> float:
    """Run Silero-VAD and return the fraction of samples flagged as speech.

    Args:
        waveform: 1-D float32 waveform at 16 kHz.
        model: Loaded Silero-VAD model.
        threshold: Speech probability threshold used by
            ``get_speech_timestamps`` (default 0.5, Silero's recommended value).

    Returns:
        speech_ratio ∈ [0, 1]. 0.0 if VAD fails or waveform is too short.
    """
    from silero_vad import get_speech_timestamps

    if waveform.numel() < _TARGET_SR // 10:  # < 100 ms → treat as no speech
        return 0.0

    try:
        segments = get_speech_timestamps(
            waveform,
            model,
            sampling_rate=_TARGET_SR,
            threshold=threshold,
        )
    except Exception:
        return 0.0

    if not segments:
        return 0.0

    speech_samples = sum(seg["end"] - seg["start"] for seg in segments)
    return float(speech_samples) / float(waveform.numel())


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------

def _normalize(s: object) -> str:
    """Lower-case, collapse whitespace, strip. Handles NaN/None gracefully."""
    if not isinstance(s, str):
        return ""
    return " ".join(s.strip().lower().split())


def compute_wer(gold: str, pred: str) -> float:
    """Word Error Rate of ``pred`` against ``gold`` reference.

    Args:
        gold: Reference (gold-standard) text.
        pred: Hypothesis (ASR-produced) text.

    Returns:
        WER as a float in [0, ∞). Returns 1.0 when the gold text is empty
        (nothing to score against — treated as maximally bad).
    """
    if not gold:
        return 1.0
    try:
        return float(jiwer.wer(gold, pred))
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Per-split driver
# ---------------------------------------------------------------------------

def process_split(
    split: str,
    meld_root: Path,
    transcripts_dir: Path,
    metadata_dir: Path,
    max_duration_sec: float,
    vad_model,
    logger: logging.Logger,
) -> List[Dict]:
    """Compute VAD + WER metadata for every utterance in the split.

    Args:
        split: 'train' | 'dev' | 'test'.
        meld_root: MELD root containing CSVs and audio subdirs.
        transcripts_dir: Directory with cached ASR JSON files.
        metadata_dir: Directory to write metadata JSON.
        max_duration_sec: Audio truncation length.
        vad_model: Loaded Silero-VAD model.
        logger: Logger.

    Returns:
        List of per-utterance metadata dicts.
    """
    csv_path = meld_root / _SPLIT_CSV[split]
    df = pd.read_csv(csv_path)
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    )
    df = df.dropna(subset=["utterance"]).reset_index(drop=True)

    # Load cached ASR transcripts
    with open(transcripts_dir / f"{split}_transcripts.json", "r", encoding="utf-8") as f:
        vox: Dict[str, str] = json.load(f)

    whisper_path = transcripts_dir / f"{split}_transcripts_whisper.json"
    if whisper_path.exists():
        with open(whisper_path, "r", encoding="utf-8") as f:
            wsp: Dict[str, str] = json.load(f)
    else:
        wsp = {}
        logger.info("No Whisper transcripts found at %s — skipping.", whisper_path)

    audio_dir = meld_root / split
    records: List[Dict] = []
    n_total = len(df)

    for i, row in df.iterrows():
        dia = int(row["dialogue_id"])
        utt = int(row["utterance_id"])
        key = f"dia{dia}_utt{utt}"
        audio_path = audio_dir / f"{key}.mp4"

        gold = _normalize(row["utterance"])
        vox_pred = _normalize(vox.get(key, ""))
        whisper_pred = _normalize(wsp.get(key, "")) if wsp else ""

        waveform = load_audio_mono_16k(audio_path, max_duration_sec)
        if waveform is None:
            speech_ratio = 0.0
            duration_sec = 0.0
        else:
            duration_sec = waveform.numel() / float(_TARGET_SR)
            speech_ratio = compute_speech_ratio(waveform, vad_model)

        vox_wer = compute_wer(gold, vox_pred)
        whisper_wer = compute_wer(gold, whisper_pred) if wsp else None

        records.append({
            "key":           key,
            "gold_words":    len(gold.split()) if gold else 0,
            "vox_words":     len(vox_pred.split()) if vox_pred else 0,
            "whisper_words": len(whisper_pred.split()) if whisper_pred else 0,
            "vox_wer":       vox_wer,
            "whisper_wer":   whisper_wer,
            "speech_ratio":  speech_ratio,
            "duration_sec":  duration_sec,
        })

        if (i + 1) % 500 == 0:
            logger.info("  %s: %d / %d done", split, i + 1, n_total)

    metadata_dir.mkdir(parents=True, exist_ok=True)
    out_path = metadata_dir / f"{split}_filter_metadata.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    logger.info(
        "%s | wrote %d records → %s", split, len(records), out_path,
    )
    return records


# ---------------------------------------------------------------------------
# Summary reporting
# ---------------------------------------------------------------------------

def report_thresholds(
    records: List[Dict],
    split: str,
    logger: logging.Logger,
) -> None:
    """Log survival rate at a range of WER thresholds and VAD floors."""
    total = len(records)
    if total == 0:
        return
    logger.info("--- %s survival rates ---", split)
    for th in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50):
        cnt = sum(1 for r in records if r["vox_wer"] <= th)
        cnt_vad = sum(
            1 for r in records
            if r["vox_wer"] <= th and r["speech_ratio"] >= 0.20
        )
        pct = 100.0 * cnt / total
        pct_vad = 100.0 * cnt_vad / total
        logger.info(
            "  WER ≤ %5.1f%%: %4d (%5.1f%%) | +VAD≥0.20: %4d (%5.1f%%)",
            th * 100, cnt, pct, cnt_vad, pct_vad,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-utterance VAD (Silero) and WER (Voxtral, Whisper) "
            "metadata for MELD. One record per utterance, one JSON per split."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config yaml (mini.yaml or small.yaml)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "test"],
        choices=["train", "dev", "test"],
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["data"]["seed"])

    if "filtering" not in config:
        raise KeyError(
            "Config missing 'filtering' block. Add filtering.metadata_dir + "
            "filtering.audio_max_duration in mini.yaml."
        )

    log_dir = config["training"]["log_dir"]
    logger = setup_logging(log_dir, "compute_filter_metadata")
    logger.info("Config: %s | splits=%s", args.config, args.splits)

    meld_root = Path(config["data"]["meld_root"])
    transcripts_dir = Path(config["data"]["transcripts_path"])
    metadata_dir = Path(config["filtering"]["metadata_dir"])
    max_duration = float(config["filtering"].get(
        "audio_max_duration",
        config["data"]["max_audio_duration"],
    ))

    logger.info("Loading Silero-VAD model...")
    vad_model = load_vad_model()

    for split in args.splits:
        records = process_split(
            split=split,
            meld_root=meld_root,
            transcripts_dir=transcripts_dir,
            metadata_dir=metadata_dir,
            max_duration_sec=max_duration,
            vad_model=vad_model,
            logger=logger,
        )
        report_thresholds(records, split, logger)

    logger.info("Done.")


if __name__ == "__main__":
    main()
