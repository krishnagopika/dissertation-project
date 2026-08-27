"""
transcribe_all.py — Preprocessing: transcription + acoustic frame sequences
=================================================================================
ONE Voxtral load produces both artefacts for every MELD utterance.

Voxtral's Whisper encoder already runs on each clip during transcription — vllm
needs it to build the audio tokens the LLM attends to — and then discards the
output once the adapter has projected it into text-embedding space. A forward
hook on that encoder captures it in flight, so the acoustic branch costs nothing
beyond the transcription pass it rides along with.

The previous implementation loaded Voxtral FOUR times for a full run: once per
split under vllm, plus a fourth time under HF transformers to recompute an
encoder forward vllm had already performed. See docs/DECISIONS.md ADR-002.

Outputs per split
-----------------
  {transcripts_path}/{split}_transcripts.json
      dict "dia{d}_utt{u}" → transcript string. NEVER overwritten: the WER
      filter keep-lists were computed against the existing file, so
      regenerating it would silently invalidate every *_filtered_keys_*.json.

  {embeddings_path}/{split}_acoustic_seq.pt
      dict "dia{d}_utt{u}" → fp16 Tensor (T, 1280), variable length, vllm's
      30 s zero-padding removed. The primary acoustic artefact; pooling happens
      in the trainable head (ADR-003), because a learned pooling cannot be
      baked into a cache written before training starts.

  {embeddings_path}/{split}_embeddings_maskedmean.pt
      dict "dia{d}_utt{u}" → float32 Tensor (1280,), mean over real frames.
      The ADR-003 control. Deliberately a DISTINCT filename so the legacy
      {split}_embeddings.pt (unmasked mean over 1500 padded frames) is left
      untouched and existing results stay reproducible.

Voxtral is never loaded during training — downstream scripts read these files.

Resumption: a split is skipped when its transcripts AND sequences both exist,
so interrupted Slurm jobs can be safely re-submitted.

Usage
-----
  python3.12 src/preprocessing/transcribe_all.py --config src/configs/mini.yaml
  python3.12 src/preprocessing/transcribe_all.py --config src/configs/mini.yaml \\
      --splits train dev          # subset of splits
  python3.12 src/preprocessing/transcribe_all.py --config src/configs/mini.yaml \\
      --legacy_two_pass           # old behaviour, writes {split}_embeddings.pt

Requires VLLM_ALLOW_INSECURE_SERIALIZATION=1 — llm.apply_model() ships the
capture hook to the worker process, which vllm refuses under its default
msgpack-only serialisation.

Slurm: see src/scripts/transcribe_single_pass.sbatch
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
# Single pass — transcription + acoustic sequences from ONE model load
# ---------------------------------------------------------------------------

TRANSCRIBE_INSTRUCTION: str = (
    "Output only the verbatim spoken words from this audio. "
    "Plain text only. No timestamps, no speaker labels, "
    "no formatting. If unclear, output your best guess. "
    "Do not say you did not understand."
)


def wav_b64_to_array(audio_b64: str) -> np.ndarray:
    """Decode base64 WAV bytes to the float32 waveform vllm will see.

    The fingerprint that pairs a sequence back to its utterance key is computed
    over this array, so it must be the *decoded* waveform rather than the
    encoded bytes -- the hook hashes what vllm's audio loader produces, not what
    we sent over the wire.

    Args:
        audio_b64: Base64-encoded 16 kHz mono WAV from :func:`audio_to_wav_base64`.

    Returns:
        Contiguous float32 array of shape ``(num_samples,)``.
    """
    import io

    import soundfile as sf

    data, _ = sf.read(io.BytesIO(base64.b64decode(audio_b64)), dtype="float32")
    if data.ndim > 1:                    # defensive: ffmpeg already gives mono
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32)


def is_complete_tensor_file(path: Path) -> bool:
    """True if ``path`` is a fully-written torch save file.

    A .pt is a zip archive, so a truncated one fails the central-directory
    check without deserialising anything — cheap enough to call on a 4 GB file.

    This exists because ``path.exists()`` is not a validity test. Job 8086 was
    killed mid-``torch.save`` and left a 1.8 GB fragment of a 4.0 GB file; the
    resume logic accepted it, which would have fed a corrupt cache to every
    downstream job. See POSTMORTEMS.md PM-009.

    Args:
        path: Candidate .pt file.

    Returns:
        True if the file exists and its archive structure is intact.
    """
    import zipfile

    if not path.exists():
        return False
    try:
        with zipfile.ZipFile(path) as z:
            return z.namelist() != []
    except (zipfile.BadZipFile, OSError):
        return False


def save_atomic(obj: object, path: Path, logger: logging.Logger) -> None:
    """torch.save to a temp file in the same directory, then atomically rename.

    A reader can then only ever observe the complete file or no file at all —
    never a half-written one. os.replace is atomic within a filesystem, and the
    temp file is a sibling so that holds.

    Args:
        obj: Object to serialise.
        path: Final destination.
        logger: Logger instance.
    """
    import os

    tmp = path.with_suffix(path.suffix + ".partial")
    try:
        torch.save(obj, str(tmp))
        os.replace(tmp, path)
    except BaseException:
        # Includes KeyboardInterrupt / SIGTERM-driven exits: leave no fragment
        # that a later resume could mistake for a finished artefact.
        tmp.unlink(missing_ok=True)
        logger.error("Failed writing %s — partial file removed", path.name)
        raise


def build_llm(model_id: str, tensor_parallel_size: int, logger: logging.Logger):
    """Construct the vllm engine ONCE for the whole preprocessing run.

    The previous implementation built and tore down an engine per split, so a
    full run paid three engine startups plus a fourth HF load. This is called
    once and shared.

    Args:
        model_id: HuggingFace model identifier or local path for Voxtral.
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        logger: Logger instance.

    Returns:
        A loaded ``vllm.LLM`` instance.
    """
    from vllm import LLM

    logger.info("Loading Voxtral under vllm: %s (tp=%d)",
                model_id, tensor_parallel_size)
    return LLM(
        model=model_id,
        tokenizer_mode="mistral",
        max_model_len=8192,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True,  # skip Inductor/CUDA graph compilation — avoids multi-hour startup
    )


def transcribe_and_extract_split(
    llm,
    split: str,
    records: List[Tuple[str, Path]],
    transcripts_dir: Path,
    embeddings_dir: Path,
    max_tokens: int,
    batch_size: int,
    acoustic_dim: int,
    logger: logging.Logger,
) -> None:
    """Transcribe a split and capture its acoustic sequences in ONE pass.

    Voxtral's Whisper encoder already runs on every clip during transcription --
    vllm needs it to build the audio tokens the LLM attends to, then discards
    the result once the adapter has projected it. A forward hook (see
    :mod:`src.preprocessing.acoustic_hook`) intercepts that tensor, so the
    second HF model load the old Pass 2 required is no longer needed.

    Writes three artefacts:

    ``{split}_transcripts.json``
        Unchanged format. **Never overwritten** -- the WER filter keep-lists
        were computed against the existing file, so regenerating transcripts
        would silently invalidate every ``*_filtered_keys_*.json`` on disk.

    ``{split}_acoustic_seq.pt``
        ``{key: fp16 Tensor(T, 1280)}`` -- variable length, padding truncated.
        The new primary artefact; pooling happens in the trainable head.

    ``{split}_embeddings_maskedmean.pt``
        ``{key: fp32 Tensor(1280,)}`` -- masked mean over real frames, i.e. the
        ADR-003 control. Written under a DISTINCT name so the legacy
        ``{split}_embeddings.pt`` (unmasked mean over 1500 padded frames) is
        left untouched and existing results stay reproducible.

    Args:
        llm: Loaded vllm engine, shared across splits.
        split: Dataset split name.
        records: List of (key, audio_path) pairs.
        transcripts_dir: Directory to write transcript JSON files.
        embeddings_dir: Directory to write acoustic artefacts.
        max_tokens: Maximum tokens to generate per utterance.
        batch_size: Number of utterances to submit per vllm batch.
        acoustic_dim: Frame dimension, for zero-fill on failure.
        logger: Logger instance.
    """
    from vllm import SamplingParams

    from src.preprocessing.acoustic_hook import (
        drain_acoustic_hook,
        drain_hook_errors,
        merge_worker_captures,
        sequence_stats,
        waveform_fingerprint,
    )

    transcripts_path = transcripts_dir / f"{split}_transcripts.json"
    sequences_path = embeddings_dir / f"{split}_acoustic_seq.pt"
    pooled_path = embeddings_dir / f"{split}_embeddings_maskedmean.pt"

    if transcripts_path.exists() and is_complete_tensor_file(sequences_path):
        logger.info("'%s' | transcripts and sequences both exist — skipping.", split)
        return
    if sequences_path.exists():
        # Existence is not validity. A job killed mid-torch.save leaves a
        # truncated .pt that an exists() check happily accepts, so a resume
        # would skip the split and every downstream job would load a corrupt
        # cache. Redo the split instead.
        logger.warning("'%s' | %s exists but is incomplete — re-extracting.",
                       split, sequences_path.name)

    logger.info("Single pass | '%s' | %d utterances | batch=%d",
                split, len(records), batch_size)

    sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)

    transcripts: Dict[str, str] = {}
    fingerprints: Dict[str, str] = {}
    captured: Dict[str, torch.Tensor] = {}
    missing_audio = 0

    for batch_start in range(0, len(records), batch_size):
        batch = records[batch_start: batch_start + batch_size]
        messages_batch: List[list] = []
        valid_keys: List[str] = []

        for key, audio_path in batch:
            if not audio_path.exists():
                logger.warning("Missing audio: %s — storing empty string", key)
                transcripts[key] = ""
                missing_audio += 1
                continue

            try:
                # Convert to 16 kHz mono WAV in-memory via ffmpeg.
                # vllm rejects mp4 — WAV is the only reliably accepted format.
                audio_b64 = audio_to_wav_base64(audio_path)
                fingerprints[key] = waveform_fingerprint(wav_b64_to_array(audio_b64))
                messages_batch.append([
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "audio_url",
                                "audio_url": {
                                    "url": f"data:audio/wav;base64,{audio_b64}"
                                },
                            },
                            {"type": "text", "text": TRANSCRIBE_INSTRUCTION},
                        ],
                    },
                ])
                valid_keys.append(key)
            except Exception as exc:                          # noqa: BLE001
                logger.error("Error preparing %s: %s", key, exc)
                transcripts[key] = ""
                missing_audio += 1

        if not messages_batch:
            continue

        try:
            outputs = llm.chat(messages_batch, sampling_params=sampling_params)
            for key, output in zip(valid_keys, outputs):
                transcripts[key] = output.outputs[0].text.strip()
        except Exception as exc:                              # noqa: BLE001
            logger.error("vllm batch error (batch starting %d): %s",
                         batch_start, exc)
            for key in valid_keys:
                transcripts[key] = ""

        # Drain per batch so the worker-side buffer stays bounded regardless of
        # split size -- sequences are ~500 KB per clip, unlike pooled vectors.
        captured.update(merge_worker_captures(llm.apply_model(drain_acoustic_hook)))

        # A hook cannot raise without aborting the engine step, so it banks
        # exceptions instead. Surface them immediately rather than inferring
        # the cause later from a pile of unmatched keys.
        for worker_errs in llm.apply_model(drain_hook_errors):
            for msg in worker_errs[:5]:
                logger.error("acoustic hook: %s", msg)

        done = min(batch_start + batch_size, len(records))
        n, mb, mean_frames = sequence_stats(captured)
        logger.info("  '%s' | %d / %d transcribed | %d seqs, %.0f MB, %.0f frames avg",
                    split, done, len(records), n, mb, mean_frames)

    # ---- Pair sequences back to utterance keys ----
    sequences: Dict[str, torch.Tensor] = {}
    pooled: Dict[str, torch.Tensor] = {}
    unmatched: List[str] = []

    for key, _ in records:
        fp = fingerprints.get(key)
        seq = captured.get(fp) if fp is not None else None
        if seq is None:
            sequences[key] = torch.zeros((1, acoustic_dim), dtype=torch.float16)
            pooled[key] = torch.zeros(acoustic_dim, dtype=torch.float32)
            if fp is not None:                 # audio existed but hook missed it
                unmatched.append(key)
        else:
            sequences[key] = seq
            # Masked mean == plain mean here: padding was already truncated at
            # capture, so every stored frame is real.
            pooled[key] = seq.float().mean(dim=0)

    if unmatched:
        # Loud, and recorded on disk: a silent zero vector looks like a valid
        # embedding to every downstream trainer.
        logger.error(
            "'%s' | %d clip(s) transcribed but NOT captured by the hook — "
            "zero-filled. See %s_unmatched.json", split, len(unmatched), split,
        )
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        with open(embeddings_dir / f"{split}_unmatched.json", "w",
                  encoding="utf-8") as f:
            json.dump(unmatched, f, indent=2)

    transcripts_dir.mkdir(parents=True, exist_ok=True)
    if transcripts_path.exists():
        logger.info("'%s' | transcripts already on disk — preserving %s "
                    "(filter keep-lists were derived from it)",
                    split, transcripts_path)
    else:
        with open(transcripts_path, "w", encoding="utf-8") as f:
            json.dump(transcripts, f, ensure_ascii=False, indent=2)
        logger.info("'%s' | %d transcripts → %s",
                    split, len(transcripts), transcripts_path)

    embeddings_dir.mkdir(parents=True, exist_ok=True)
    # Pooled FIRST: it is small, and writing the big file last means a kill
    # during the big write cannot leave the pair inconsistent in the direction
    # the resume check cares about (sequences present but pooled missing).
    save_atomic(pooled, pooled_path, logger)
    # Release the driver-side capture before the large write: `captured` holds
    # the same ~4 GB of arrays that `sequences` references, and job 8086 was
    # OOM-killed at exactly this point.
    captured.clear()
    gc.collect()
    save_atomic(sequences, sequences_path, logger)

    n, mb, mean_frames = sequence_stats(sequences)
    logger.info(
        "'%s' | %d sequences (%.0f MB, %.0f frames avg) → %s",
        split, n, mb, mean_frames, sequences_path,
    )
    logger.info("'%s' | %d masked-mean vectors → %s (%d missing audio, %d unmatched)",
                split, len(pooled), pooled_path, missing_audio, len(unmatched))

    # Three splits share one process; without this the peak is the sum.
    sequences.clear()
    pooled.clear()
    fingerprints.clear()
    gc.collect()


# ---------------------------------------------------------------------------
# Legacy Pass 1 — vllm transcription only (kept for --legacy_two_pass)
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
    parser.add_argument(
        "--legacy_two_pass",
        action="store_true",
        help=(
            "Use the old two-pass path: vllm for transcription, then a SECOND "
            "load under HF transformers for embeddings. Kept only to reproduce "
            "pre-existing {split}_embeddings.pt files."
        ),
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
    # Single pass (default) — one vllm load, both artefacts
    # ------------------------------------------------------------------
    if not args.legacy_two_pass:
        from src.preprocessing.acoustic_hook import (
            install_acoustic_hook,
            remove_acoustic_hook,
        )

        logger.info("=== Single pass: vllm transcription + acoustic capture ===")
        llm = build_llm(model_id, tensor_parallel_size, logger)

        # apply_model ships the callable to the worker process, which vllm
        # refuses unless insecure serialisation is enabled. Fail loudly here
        # rather than at the first apply_model call, deep into a long run.
        import os
        if os.environ.get("VLLM_ALLOW_INSECURE_SERIALIZATION") != "1":
            raise RuntimeError(
                "VLLM_ALLOW_INSECURE_SERIALIZATION=1 must be set for "
                "llm.apply_model() to ship the capture hook to the worker. "
                "Set it in the sbatch script, or pass --legacy_two_pass."
            )

        logger.info("Hook: %s", llm.apply_model(install_acoustic_hook))

        for split in args.splits:
            transcribe_and_extract_split(
                llm=llm,
                split=split,
                records=split_records[split],
                transcripts_dir=transcripts_dir,
                embeddings_dir=embeddings_dir,
                max_tokens=200,
                batch_size=args.vllm_batch_size,
                acoustic_dim=acoustic_dim,
                logger=logger,
            )

        llm.apply_model(remove_acoustic_hook)
        del llm
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("All done (single pass).")
        return

    # ------------------------------------------------------------------
    # Legacy Pass 1 — vllm transcription
    # ------------------------------------------------------------------
    logger.info("=== LEGACY two-pass mode (--legacy_two_pass) ===")
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
