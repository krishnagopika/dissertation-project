"""
transcribe_all.py — Step 0: Preprocessing
==========================================
Runs Voxtral on every MELD utterance to produce two cached artefacts per split:

  data/meld_transcripts/{split}_transcripts.json
      dict keyed by "dia{d}_utt{u}" → transcript string

  data/meld_embeddings/{split}_embeddings.pt
      dict keyed by "dia{d}_utt{u}" → float32 Tensor of shape (acoustic_dim,)

Both artefacts are written once and never recomputed — downstream training
scripts load them directly (Voxtral is never loaded during training).

Resumption: if an output file already exists it is skipped, so interrupted
jobs can be re-submitted safely.

Usage
-----
  python3.12 src/preprocessing/transcribe_all.py --config src/configs/mini.yaml

Slurm: see src/scripts/transcribe.sbatch
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tarfile
from pathlib import Path
from typing import Dict

import torch
import torchaudio

# ---------------------------------------------------------------------------
# Project-local imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.voxtral import VoxtralWrapper


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load YAML config and return as dictionary."""
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(log_dir: str, script_name: str) -> logging.Logger:
    """Set up logging to both file and console."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    fh = logging.FileHandler(Path(log_dir) / f"{script_name}.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------

def extract_tar_if_needed(
    meld_root: Path,
    split: str,
    logger: logging.Logger,
) -> None:
    """Extract {split}.tar.gz into {meld_root}/{split}/ if not already done."""
    audio_dir = meld_root / split
    tar_path = meld_root / f"{split}.tar.gz"

    if audio_dir.exists() and any(audio_dir.glob("*.wav")):
        logger.info("Audio dir already extracted: %s", audio_dir)
        return

    if not tar_path.exists():
        raise FileNotFoundError(
            f"Neither audio dir nor tar.gz found for split '{split}'. "
            f"Expected {tar_path} or {audio_dir}."
        )

    logger.info("Extracting %s → %s ...", tar_path, meld_root)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(meld_root)
    logger.info("Extraction complete.")


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

TARGET_SR: int = 16_000


def load_audio(audio_path: Path, max_duration_sec: float) -> torch.Tensor:
    """Load wav, convert to mono, resample to 16 kHz, truncate/pad.

    Returns a 1-D float32 tensor of shape (num_samples,).
    """
    waveform, sr = torchaudio.load(str(audio_path))  # (C, T)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=TARGET_SR
        )
        waveform = resampler(waveform)

    waveform = waveform.squeeze(0)  # (T,)

    max_samples = int(max_duration_sec * TARGET_SR)
    if waveform.shape[0] > max_samples:
        waveform = waveform[:max_samples]

    return waveform.float()


def process_split(
    split: str,
    meld_root: Path,
    transcripts_dir: Path,
    embeddings_dir: Path,
    wrapper: VoxtralWrapper,
    max_duration_sec: float,
    logger: logging.Logger,
) -> None:
    """Transcribe and extract embeddings for all utterances in a split.

    Outputs
    -------
    transcripts_dir/{split}_transcripts.json
    embeddings_dir/{split}_embeddings.pt
    """
    import pandas as pd

    transcript_out = transcripts_dir / f"{split}_transcripts.json"
    embedding_out = embeddings_dir / f"{split}_embeddings.pt"

    if transcript_out.exists() and embedding_out.exists():
        logger.info(
            "Split '%s' already processed — skipping. "
            "Delete outputs to rerun.",
            split,
        )
        return

    csv_map = {
        "train": "train_sent_emo.csv",
        "dev": "dev_sent_emo.csv",
        "test": "test_sent_emo.csv",
    }
    csv_path = meld_root / csv_map[split]
    if not csv_path.exists():
        raise FileNotFoundError(f"MELD CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    )

    audio_dir = meld_root / split
    transcripts: Dict[str, str] = {}
    embeddings: Dict[str, torch.Tensor] = {}

    total = len(df)
    logger.info("Processing split '%s' — %d utterances", split, total)

    for i, row in df.iterrows():
        dia = int(row["dialogue_id"])
        utt = int(row["utterance_id"])
        key = f"dia{dia}_utt{utt}"

        audio_path = audio_dir / f"{key}.wav"
        if not audio_path.exists():
            logger.warning("Missing audio: %s — storing empty string/zeros", key)
            transcripts[key] = ""
            embeddings[key] = torch.zeros(
                wrapper.get_acoustic_hidden_dim(), dtype=torch.float32
            )
            continue

        try:
            waveform = load_audio(audio_path, max_duration_sec)

            # Transcription
            transcript = wrapper.transcribe(waveform, sample_rate=TARGET_SR)
            if isinstance(transcript, list):
                transcript = transcript[0]
            transcripts[key] = transcript

            # Acoustic embedding — shape (1, D) → squeeze to (D,)
            emb = wrapper.extract_acoustic_embeddings(
                waveform, sample_rate=TARGET_SR
            )  # (1, D) float32 cpu
            embeddings[key] = emb.squeeze(0)

        except Exception as exc:
            logger.error("Error processing %s: %s", key, exc)
            transcripts[key] = ""
            embeddings[key] = torch.zeros(
                wrapper.get_acoustic_hidden_dim(), dtype=torch.float32
            )

        if (i + 1) % 100 == 0:
            logger.info("  %d / %d done", i + 1, total)

    # --- Write outputs ---
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    with open(transcript_out, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, ensure_ascii=False, indent=2)
    logger.info("Transcripts saved → %s", transcript_out)

    torch.save(embeddings, str(embedding_out))
    logger.info("Embeddings saved → %s", embedding_out)

    # Free GPU memory between splits
    torch.cuda.empty_cache()
    import gc
    gc.collect()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe MELD audio and extract acoustic embeddings."
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
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["data"]["seed"])

    log_dir = config["training"]["log_dir"]
    logger = setup_logging(log_dir, "transcribe_all")
    logger.info("Config: %s", args.config)
    logger.info("Splits: %s", args.splits)

    meld_root = Path(config["data"]["meld_root"])
    transcripts_dir = Path(config["data"]["transcripts_path"])
    embeddings_dir = Path(config["data"]["embeddings_path"])
    max_duration_sec = float(config["data"]["max_audio_duration"])

    # Extract tar.gz archives if needed
    for split in args.splits:
        extract_tar_if_needed(meld_root, split, logger)

    # Load Voxtral — frozen inference only
    logger.info("Loading Voxtral: %s", config["model"]["voxtral_id"])
    wrapper = VoxtralWrapper(
        model_name_or_path=config["model"]["voxtral_id"],
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Verify the adapter output dim matches config
    actual_dim = wrapper.get_acoustic_hidden_dim()
    config_dim = config["model"]["acoustic_dim"]
    if actual_dim != config_dim:
        logger.warning(
            "acoustic_dim mismatch: model reports %d, config says %d. "
            "Embeddings will be %d-dim. Update acoustic_dim in your config.",
            actual_dim,
            config_dim,
            actual_dim,
        )

    for split in args.splits:
        process_split(
            split=split,
            meld_root=meld_root,
            transcripts_dir=transcripts_dir,
            embeddings_dir=embeddings_dir,
            wrapper=wrapper,
            max_duration_sec=max_duration_sec,
            logger=logger,
        )

    logger.info("All splits done.")

    # Final cleanup
    del wrapper
    torch.cuda.empty_cache()
    import gc
    gc.collect()


if __name__ == "__main__":
    main()
