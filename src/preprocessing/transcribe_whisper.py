"""
transcribe_whisper.py — Standalone Whisper-large-v3 ASR baseline
=================================================================
Transcribes every MELD utterance with standalone Whisper-large-v3 and
caches the text, so its ASR quality (WER etc.) can be compared against
Voxtral's transcripts.

Whisper-large-v3 IS the audio encoder inside Voxtral-Mini-3B; the
difference here is the decoder — Whisper's own seq2seq decoder rather
than Voxtral's 3B LLM. This isolates how much the LLM decoder helps (or
hurts, via over-generation) the transcription.

Output (one file per split, same key scheme as Voxtral):
  {transcripts_path}/{split}_transcripts_whisper.json
    → dict mapping "dia{D}_utt{U}" → transcribed string

Resumption: a split's output file is skipped if it already exists, so an
interrupted Slurm job can be safely re-submitted.

Usage
-----
  python3.12 src/preprocessing/transcribe_whisper.py \\
      --config src/configs/mini.yaml
  python3.12 src/preprocessing/transcribe_whisper.py \\
      --config src/configs/mini.yaml --splits test --batch_size 16

Slurm: see src/scripts/transcribe_whisper.sbatch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Reuse the audio indexing / loading utilities that already handle MELD's
# mp4 files via ffmpeg (no torchaudio / torchcodec needed on GPU nodes).
from src.preprocessing.transcribe_all import (
    TARGET_SR,
    build_utterance_index,
    extract_tar_if_needed,
    load_audio_mono_16k,
)
from src.utils import get_device, load_config, set_seed, setup_logging


def transcribe_split(
    split: str,
    records: List[Tuple[str, Path]],
    transcripts_dir: Path,
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    device: torch.device,
    max_duration: float,
    batch_size: int,
    max_new_tokens: int,
    logger,
) -> None:
    """Transcribe one MELD split with Whisper and cache the result.

    Args:
        split: One of 'train', 'dev', 'test'.
        records: List of (key, audio_path) pairs.
        transcripts_dir: Directory to write {split}_transcripts_whisper.json.
        processor: Whisper processor (feature extractor + tokenizer).
        model: Whisper model in eval mode on device.
        device: Target device.
        max_duration: Maximum audio duration in seconds to keep.
        batch_size: Number of clips per forward pass.
        max_new_tokens: Max tokens to generate per clip.
        logger: Logger instance.
    """
    out_path = transcripts_dir / f"{split}_transcripts_whisper.json"
    if out_path.exists():
        logger.info("Transcripts for '%s' already exist — skipping.", split)
        return

    logger.info("'%s' | %d utterances | batch_size=%d", split, len(records), batch_size)
    transcripts: Dict[str, str] = {}
    missing = 0

    for start in range(0, len(records), batch_size):
        batch = records[start: start + batch_size]
        waveforms: List[torch.Tensor] = []
        valid_keys: List[str] = []

        for key, audio_path in batch:
            if not audio_path.exists():
                logger.warning("Missing audio: %s — storing empty string", key)
                transcripts[key] = ""
                missing += 1
                continue
            try:
                wav = load_audio_mono_16k(audio_path, max_duration)
            except Exception as exc:  # noqa: BLE001 — log and skip bad files
                logger.warning("ffmpeg failed on %s (%s) — empty string", key, exc)
                transcripts[key] = ""
                missing += 1
                continue
            waveforms.append(wav.numpy())
            valid_keys.append(key)

        if not valid_keys:
            continue

        inputs = processor(
            waveforms,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(device, dtype=model.dtype)

        with torch.no_grad():
            generated = model.generate(
                input_features,
                language="en",
                task="transcribe",
                max_new_tokens=max_new_tokens,
            )
        texts = processor.batch_decode(generated, skip_special_tokens=True)

        for key, text in zip(valid_keys, texts):
            transcripts[key] = text.strip()

        done = start + len(batch)
        if done % (batch_size * 20) == 0 or done >= len(records):
            logger.info("  [%s] %d / %d done", split, done, len(records))

    transcripts_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, ensure_ascii=False, indent=2)
    logger.info(
        "Saved %d transcripts (%d missing audio) to %s",
        len(transcripts), missing, out_path,
    )


def main() -> None:
    """Transcribe requested MELD splits with Whisper-large-v3."""
    parser = argparse.ArgumentParser()
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
        help="Which splits to transcribe (default: all three)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Clips per forward pass",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Max tokens to generate per clip (matches Voxtral run)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config["data"]["seed"]))

    logger = setup_logging(config["training"]["log_dir"], "transcribe_whisper")
    device = get_device()
    logger.info("Device: %s | Splits: %s", device, args.splits)

    meld_root = Path(config["data"]["meld_root"])
    transcripts_dir = Path(config["data"]["transcripts_path"])
    max_duration = float(config["data"]["max_audio_duration"])
    model_id = config["model"]["whisper_id"]

    logger.info("Loading %s ...", model_id)
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    for split in args.splits:
        extract_tar_if_needed(meld_root, split, logger)
        records = build_utterance_index(meld_root, split)
        logger.info("Split '%s': %d utterances indexed", split, len(records))
        transcribe_split(
            split=split,
            records=records,
            transcripts_dir=transcripts_dir,
            processor=processor,
            model=model,
            device=device,
            max_duration=max_duration,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            logger=logger,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
