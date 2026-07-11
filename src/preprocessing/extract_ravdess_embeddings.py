"""
extract_ravdess_embeddings.py — Pre-cache RAVDESS Voxtral acoustic embeddings
=============================================================================
Runs Voxtral's frozen audio encoder over every RAVDESS speech clip and saves
a single dict mapping ``filename_stem`` (e.g. ``"03-01-06-01-02-01-12"``) to a
1280-dim float32 tensor.

This is a one-off preprocessing step; downstream training (Phase A — RAVDESS
acoustic backbone) reads only the cached embeddings, never the raw audio.

Pipeline
--------
  RAVDESS Actor_XX/*.wav
        ↓
  load via ffmpeg → 16 kHz mono float32 (5 s padded/truncated)
        ↓
  VoxtralWrapper.extract_acoustic_embeddings   (frozen)
        ↓
  {<filename_stem>: Tensor(1280,)}  →  embeddings.pt

Output
------
  ``<ravdess.embeddings_path>`` (a single .pt file).

Usage
-----
  python3.12 src/preprocessing/extract_ravdess_embeddings.py \\
      --config src/configs/mini.yaml

Slurm: see src/scripts/extract_ravdess_embeddings.sbatch
"""

from __future__ import annotations

import argparse
import gc
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.voxtral import VoxtralWrapper
from src.utils import get_device, load_config, set_seed, setup_logging


_TARGET_SR: int = 16_000


def load_audio_mono_16k(
    audio_path: Path,
    max_duration_sec: float,
) -> torch.Tensor:
    """Load audio file via ffmpeg as mono 16 kHz float32 tensor.

    Mirrors the implementation in transcribe_all.py — uses ffmpeg subprocess
    rather than torchaudio so GPU nodes without torchcodec/libnvrtc still work.

    Args:
        audio_path: Path to .wav file.
        max_duration_sec: Maximum duration in seconds (truncation).

    Returns:
        1-D float32 waveform tensor at 16 kHz.
    """
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
    waveform = torch.from_numpy(
        np.frombuffer(result.stdout, dtype=np.float32).copy()
    )
    return waveform


def collect_ravdess_files(root: Path, logger: logging.Logger) -> List[Path]:
    """Return all RAVDESS speech .wav files under root/Actor_*/.

    Args:
        root: Root directory containing Actor_01/ ... Actor_24/.
        logger: Logger instance.

    Returns:
        Sorted list of .wav file paths.

    Raises:
        FileNotFoundError: If no .wav files found.
    """
    files = sorted(root.glob("Actor_*/*.wav"))
    if not files:
        raise FileNotFoundError(
            f"No RAVDESS .wav files found under {root}/Actor_*/. "
            "Ensure the RAVDESS audio-speech archive is extracted there."
        )
    logger.info("Indexed %d RAVDESS .wav files under %s", len(files), root)
    return files


def extract_embeddings(
    files: List[Path],
    wrapper: VoxtralWrapper,
    max_duration_sec: float,
    acoustic_dim: int,
    logger: logging.Logger,
) -> Dict[str, torch.Tensor]:
    """Run Voxtral encoder on every file and collect embeddings.

    Args:
        files: List of .wav file paths.
        wrapper: Loaded VoxtralWrapper instance.
        max_duration_sec: Maximum audio duration to process per clip.
        acoustic_dim: Expected output embedding dim.
        logger: Logger instance.

    Returns:
        Dict mapping filename stem → embedding tensor on CPU.
    """
    embeddings: Dict[str, torch.Tensor] = {}
    n_failed = 0

    for i, path in enumerate(files):
        try:
            waveform = load_audio_mono_16k(path, max_duration_sec)
            emb = wrapper.extract_acoustic_embeddings(
                waveform, sample_rate=_TARGET_SR
            )  # (1, D)
            embeddings[path.stem] = emb.squeeze(0).cpu().float()
        except Exception as exc:
            logger.error("Failed on %s: %s", path, exc)
            embeddings[path.stem] = torch.zeros(
                acoustic_dim, dtype=torch.float32
            )
            n_failed += 1

        if (i + 1) % 100 == 0:
            logger.info("  %d / %d done", i + 1, len(files))

    logger.info(
        "Embedding extraction complete | %d total | %d failed",
        len(embeddings), n_failed,
    )
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract and cache Voxtral acoustic embeddings for all "
            "RAVDESS speech clips."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config yaml (mini.yaml or small.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["data"]["seed"])

    log_dir = config["training"]["log_dir"]
    logger = setup_logging(log_dir, "extract_ravdess_embeddings")
    logger.info("Config: %s", args.config)

    if "ravdess" not in config:
        raise KeyError(
            "Config missing 'ravdess' block. Add paths and hidden_dim — see "
            "mini.yaml for the expected schema."
        )

    root = Path(config["ravdess"]["data_root"])
    out_path = Path(config["ravdess"]["embeddings_path"])
    max_duration = float(config["ravdess"]["max_duration_sec"])
    acoustic_dim = int(config["model"]["acoustic_dim"])

    if out_path.exists():
        logger.info(
            "Embeddings already exist at %s — skipping. "
            "Delete the file to regenerate.", out_path,
        )
        return

    files = collect_ravdess_files(root, logger)

    logger.info(
        "Loading Voxtral via HF transformers: %s",
        config["model"]["voxtral_id"],
    )
    wrapper = VoxtralWrapper(
        model_name_or_path=config["model"]["voxtral_id"],
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    actual_dim = wrapper.get_acoustic_hidden_dim()
    if actual_dim != acoustic_dim:
        logger.warning(
            "acoustic_dim mismatch: model=%d, config=%d. Using model value.",
            actual_dim, acoustic_dim,
        )
        acoustic_dim = actual_dim

    embeddings = extract_embeddings(
        files=files,
        wrapper=wrapper,
        max_duration_sec=max_duration,
        acoustic_dim=acoustic_dim,
        logger=logger,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, str(out_path))
    logger.info("Saved %d embeddings → %s", len(embeddings), out_path)

    del wrapper
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
