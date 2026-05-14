"""
transcribe_all.py — Preprocessing: transcription + acoustic embedding extraction
=================================================================================
Runs in two sequential passes over every MELD utterance:

  Pass 1 — vllm (fast batched transcription)
      Voxtral is loaded via vllm for efficient text generation.
      Output: data/meld_transcripts/{split}_transcripts.json
              dict keyed by "dia{d}_utt{u}" → transcript string

  Pass 2 — HuggingFace transformers (acoustic embedding extraction)
      Voxtral is loaded via HF transformers with a forward hook on the
      audio-adapter output layer to pull acoustic embeddings.
      Output: data/meld_embeddings/{split}_embeddings.pt
              dict keyed by "dia{d}_utt{u}" → float32 Tensor (acoustic_dim,)

Both artefacts are written once and never recomputed — downstream training
scripts load them directly. Voxtral is never loaded during training.

Resumption: each output file is skipped if it already exists on disk, so
interrupted Slurm jobs can be safely re-submitted.

Usage
-----
  python3.12 src/preprocessing/transcribe_all.py --config src/configs/mini.yaml
  python3.12 src/preprocessing/transcribe_all.py --config src/configs/mini.yaml \\
      --splits train dev   # process a subset of splits

Slurm: see src/scripts/transcribe.sbatch
"""

from __future__ import annotations

import argparse
import base64
import gc
import json
import logging
import sys
import tarfile
from pathlib import Path
from typing import Dict, List, Tuple

import subprocess
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project-local imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.voxtral import VoxtralWrapper
from src.utils import get_device, load_config, set_seed, setup_logging


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SR: int = 16_000
CSV_MAP: Dict[str, str] = {
    "train": "train_sent_emo.csv",
    "dev":   "dev_sent_emo.csv",
    "test":  "test_sent_emo.csv",
}


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def extract_tar_if_needed(
    meld_root: Path,
    split: str,
    logger: logging.Logger,
) -> None:
    """Extract {split}.tar.gz into meld_root/{split}/ if not already done.

    Args:
        meld_root: Root directory of MELD data.
        split: One of 'train', 'dev', 'test'.
        logger: Logger instance.
    """
    audio_dir = meld_root / split
    tar_path = meld_root / f"{split}.tar.gz"

    if audio_dir.exists() and any(audio_dir.glob("*.mp4")):
        logger.info("Audio dir already extracted: %s", audio_dir)
        return

    if not tar_path.exists():
        raise FileNotFoundError(
            f"Neither audio dir nor tar.gz found for split '{split}'. "
            f"Expected {tar_path} or {audio_dir} with .mp4 files."
        )

    logger.info("Extracting %s …", tar_path)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(meld_root)
    logger.info("Extraction complete → %s", audio_dir)


def load_audio_mono_16k(
    audio_path: Path,
    max_duration_sec: float,
) -> torch.Tensor:
    """Load audio file via ffmpeg, returning a mono 16 kHz float32 tensor.

    Uses ffmpeg subprocess (raw PCM output) instead of torchaudio so that
    mp4 files work without torchcodec / libnvrtc. ffmpeg is always available
    on the HPC nodes regardless of CUDA library state.

    Args:
        audio_path: Path to audio file (.mp4, .wav, etc.).
        max_duration_sec: Maximum duration to keep in seconds.

    Returns:
        1-D float32 tensor of shape (num_samples,) at 16 kHz.

    Raises:
        subprocess.CalledProcessError: If ffmpeg fails on the file.
    """
    result = subprocess.run(
        [
            "ffmpeg",
            "-i", str(audio_path),
            "-ar", str(TARGET_SR),  # resample to 16 kHz
            "-ac", "1",             # mono
            "-t", str(max_duration_sec),  # truncate to max duration
            "-f", "f32le",          # raw float32 little-endian PCM
            "pipe:1",
            "-loglevel", "error",
        ],
        capture_output=True,
        check=True,
    )
    waveform = torch.from_numpy(
        np.frombuffer(result.stdout, dtype=np.float32).copy()
    )
    return waveform


def audio_to_wav_base64(audio_path: Path) -> str:
    """Convert any audio file to 16 kHz mono WAV and return base64-encoded bytes.

    Uses ffmpeg to pipe audio through stdout — no temp files written.
    vllm requires WAV format; MP4 is not accepted directly.

    Args:
        audio_path: Path to the source audio file (.mp4, .wav, etc.).

    Returns:
        Base64-encoded string of the WAV bytes.

    Raises:
        subprocess.CalledProcessError: If ffmpeg exits non-zero.
    """
    result = subprocess.run(
        [
            "ffmpeg",
            "-i", str(audio_path),
            "-ar", "16000",   # resample to 16 kHz
            "-ac", "1",       # mono
            "-f", "wav",
            "pipe:1",         # write WAV to stdout
            "-y",             # overwrite (unused, but silences warnings)
            "-loglevel", "error",  # suppress progress output
        ],
        capture_output=True,
        check=True,
    )
    return base64.b64encode(result.stdout).decode("utf-8")


def build_utterance_index(
    meld_root: Path,
    split: str,
) -> List[Tuple[str, Path]]:
    """Build ordered list of (key, audio_path) pairs for a split.

    Args:
        meld_root: Root directory of MELD data.
        split: One of 'train', 'dev', 'test'.

    Returns:
        List of (key, audio_path) sorted by dialogue then utterance ID.
    """
    import pandas as pd

    csv_path = meld_root / CSV_MAP[split]
    if not csv_path.exists():
        raise FileNotFoundError(f"MELD CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    )

    audio_dir = meld_root / split
    records: List[Tuple[str, Path]] = []
    for _, row in df.iterrows():
        dia = int(row["dialogue_id"])
        utt = int(row["utterance_id"])
        key = f"dia{dia}_utt{utt}"
        # MELD stores audio as .mp4 files
        audio_path = audio_dir / f"{key}.mp4"
        records.append((key, audio_path))

    return records


# ---------------------------------------------------------------------------
# Pass 1 — vllm transcription
# ---------------------------------------------------------------------------

def transcribe_split_vllm(
    split: str,
    records: List[Tuple[str, Path]],
    transcripts_dir: Path,
    model_id: str,
    max_tokens: int,
    batch_size: int,
    tensor_parallel_size: int,
    logger: logging.Logger,
) -> None:
    """Transcribe all utterances in a split using vllm.

    Saves results to {transcripts_dir}/{split}_transcripts.json.
    Skips if the output file already exists.

    Args:
        split: Dataset split name.
        records: List of (key, audio_path) pairs.
        transcripts_dir: Directory to write transcript JSON files.
        model_id: HuggingFace model identifier for Voxtral.
        max_tokens: Maximum tokens to generate per utterance.
        batch_size: Number of utterances to process per vllm batch.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        logger: Logger instance.
    """
    from vllm import LLM, SamplingParams

    out_path = transcripts_dir / f"{split}_transcripts.json"
    if out_path.exists():
        logger.info(
            "Transcripts for '%s' already exist — skipping Pass 1.", split
        )
        return

    logger.info(
        "Pass 1 | '%s' | %d utterances | model: %s | tp=%d",
        split, len(records), model_id, tensor_parallel_size,
    )

    llm = LLM(
        model=model_id,
        tokenizer_mode="mistral",
        max_model_len=8192,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True,  # skip Inductor/CUDA graph compilation — avoids multi-hour startup
    )
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
    )

    transcripts: Dict[str, str] = {}
    missing = 0

    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start: batch_start + batch_size]
        messages_batch = []
        valid_keys = []

        for key, audio_path in batch:
            if not audio_path.exists():
                logger.warning("Missing audio: %s — storing empty string", key)
                transcripts[key] = ""
                missing += 1
                continue

            try:
                # Convert to 16 kHz mono WAV in-memory via ffmpeg.
                # vllm rejects mp4 — WAV is the only reliably accepted format.
                audio_b64 = audio_to_wav_base64(audio_path)
                mime = "audio/wav"
                messages_batch.append([
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "audio_url",
                                "audio_url": {
                                    "url": f"data:{mime};base64,{audio_b64}"
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Output only the verbatim spoken words from this audio. "
                                    "Plain text only. No timestamps, no speaker labels, "
                                    "no formatting. If unclear, output your best guess. "
                                    "Do not say you did not understand."
                                ),
                            },
                        ],
                    },
                ])
                valid_keys.append(key)
            except Exception as exc:
                logger.error("Error preparing %s: %s", key, exc)
                transcripts[key] = ""
                missing += 1

        if not messages_batch:
            continue

        try:
            outputs = llm.chat(messages_batch, sampling_params=sampling_params)
            for key, output in zip(valid_keys, outputs):
                transcripts[key] = output.outputs[0].text.strip()
        except Exception as exc:
            logger.error(
                "vllm batch error (batch starting %d): %s", batch_start, exc
            )
            for key in valid_keys:
                transcripts[key] = ""

        done = min(batch_start + batch_size, len(records))
        logger.info("  Pass 1 | %d / %d done", done, len(records))

    transcripts_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, ensure_ascii=False, indent=2)

    logger.info(
        "Pass 1 complete | '%s' | saved %d transcripts (%d missing) → %s",
        split, len(transcripts), missing, out_path,
    )

    # Unload vllm and free GPU memory before Pass 2
    del llm
    torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# Pass 2 — HF transformers embedding extraction
# ---------------------------------------------------------------------------

def extract_embeddings_split_hf(
    split: str,
    records: List[Tuple[str, Path]],
    embeddings_dir: Path,
    wrapper: VoxtralWrapper,
    max_duration_sec: float,
    acoustic_dim: int,
    logger: logging.Logger,
) -> None:
    """Extract acoustic embeddings for all utterances using HF VoxtralWrapper.

    Saves results to {embeddings_dir}/{split}_embeddings.pt.
    Skips if the output file already exists.

    Args:
        split: Dataset split name.
        records: List of (key, audio_path) pairs.
        embeddings_dir: Directory to write embedding .pt files.
        wrapper: Loaded VoxtralWrapper instance.
        max_duration_sec: Maximum audio duration to process.
        acoustic_dim: Expected embedding dimension (from config).
        logger: Logger instance.
    """
    out_path = embeddings_dir / f"{split}_embeddings.pt"
    if out_path.exists():
        logger.info(
            "Embeddings for '%s' already exist — skipping Pass 2.", split
        )
        return

    logger.info(
        "Pass 2 | '%s' | %d utterances | extracting acoustic embeddings",
        split, len(records),
    )

    embeddings: Dict[str, torch.Tensor] = {}
    missing = 0

    for i, (key, audio_path) in enumerate(records):
        if not audio_path.exists():
            embeddings[key] = torch.zeros(acoustic_dim, dtype=torch.float32)
            missing += 1
            continue

        try:
            waveform = load_audio_mono_16k(audio_path, max_duration_sec)
            emb = wrapper.extract_acoustic_embeddings(
                waveform, sample_rate=TARGET_SR
            )  # (1, D) float32 on CPU
            embeddings[key] = emb.squeeze(0)
        except Exception as exc:
            logger.error("Error extracting embeddings for %s: %s", key, exc)
            embeddings[key] = torch.zeros(acoustic_dim, dtype=torch.float32)
            missing += 1

        if (i + 1) % 100 == 0:
            logger.info("  Pass 2 | %d / %d done", i + 1, len(records))

    embeddings_dir.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, str(out_path))

    logger.info(
        "Pass 2 complete | '%s' | saved %d embeddings (%d missing) → %s",
        split, len(embeddings), missing, out_path,
    )

    torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess MELD: Pass 1 = vllm transcription, "
            "Pass 2 = HF acoustic embedding extraction."
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
        help="Which splits to process (default: all three)",
    )
    parser.add_argument(
        "--skip_transcription",
        action="store_true",
        help="Skip Pass 1 (vllm transcription) — useful if transcripts exist",
    )
    parser.add_argument(
        "--skip_embeddings",
        action="store_true",
        help="Skip Pass 2 (HF embedding extraction) — useful if embeddings exist",
    )
    parser.add_argument(
        "--vllm_batch_size",
        type=int,
        default=32,
        help="Number of utterances per vllm batch (default: 32)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit each split to this many utterances — for quick smoke tests.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["data"]["seed"])

    log_dir = config["training"]["log_dir"]
    logger = setup_logging(log_dir, "transcribe_all")
    logger.info("Config: %s", args.config)
    logger.info("Splits: %s", args.splits)

    meld_root            = Path(config["data"]["meld_root"])
    transcripts_dir      = Path(config["data"]["transcripts_path"])
    embeddings_dir       = Path(config["data"]["embeddings_path"])
    max_duration         = float(config["data"]["max_audio_duration"])
    acoustic_dim         = int(config["model"]["acoustic_dim"])
    model_id             = config["model"]["voxtral_id"]
    tensor_parallel_size = int(config["model"].get("tensor_parallel_size", 1))

    # Extract tar.gz archives for each split if needed
    for split in args.splits:
        extract_tar_if_needed(meld_root, split, logger)

    # Build utterance index for each split
    split_records = {
        split: build_utterance_index(meld_root, split)
        for split in args.splits
    }
    # Optionally limit to a small subset for smoke testing
    if args.max_samples is not None:
        split_records = {
            split: records[: args.max_samples]
            for split, records in split_records.items()
        }
        logger.info("--max_samples=%d: truncating each split for smoke test", args.max_samples)
    for split, records in split_records.items():
        logger.info("Split '%s': %d utterances indexed", split, len(records))

    # ------------------------------------------------------------------
    # Pass 1 — vllm transcription
    # ------------------------------------------------------------------
    if not args.skip_transcription:
        logger.info("=== Pass 1: vllm transcription ===")
        for split in args.splits:
            transcribe_split_vllm(
                split=split,
                records=split_records[split],
                transcripts_dir=transcripts_dir,
                model_id=model_id,
                max_tokens=200,
                batch_size=args.vllm_batch_size,
                tensor_parallel_size=tensor_parallel_size,
                logger=logger,
            )
    else:
        logger.info("Pass 1 skipped (--skip_transcription flag set).")

    # ------------------------------------------------------------------
    # Pass 2 — HF transformers embedding extraction
    # ------------------------------------------------------------------
    if not args.skip_embeddings:
        logger.info("=== Pass 2: HF acoustic embedding extraction ===")
        logger.info("Loading Voxtral via HF transformers: %s", model_id)

        wrapper = VoxtralWrapper(
            model_name_or_path=model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        # Warn if acoustic_dim doesn't match what the model reports
        actual_dim = wrapper.get_acoustic_hidden_dim()
        if actual_dim != acoustic_dim:
            logger.warning(
                "acoustic_dim mismatch: model=%d, config=%d. "
                "Using model value (%d). Update your config.",
                actual_dim, acoustic_dim, actual_dim,
            )
            acoustic_dim = actual_dim

        for split in args.splits:
            extract_embeddings_split_hf(
                split=split,
                records=split_records[split],
                embeddings_dir=embeddings_dir,
                wrapper=wrapper,
                max_duration_sec=max_duration,
                acoustic_dim=acoustic_dim,
                logger=logger,
            )

        del wrapper
        torch.cuda.empty_cache()
        gc.collect()
    else:
        logger.info("Pass 2 skipped (--skip_embeddings flag set).")

    logger.info("All done.")


if __name__ == "__main__":
    main()
