"""
finetune.py — Phase 1: ASR-aware XLM-RoBERTa Fine-tuning
==========================================================
Fine-tunes XLM-RoBERTa on Voxtral's ASR transcripts of MELD so that the text
encoder becomes robust to ASR errors introduced during preprocessing.

Reads pre-cached transcripts from data/meld_transcripts/.
Voxtral is NOT loaded during this phase.

Pipeline
--------
  transcripts + emotion/sentiment labels
        ↓
  XLMRobertaClassifier (joint sentiment + emotion heads)
        ↓
  Checkpoints saved to config["training"]["checkpoint_dir"]

Usage
-----
  python3.12 src/training/finetune.py --config src/configs/mini.yaml

Slurm: see src/scripts/finetune.sbatch
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import (
    EMOTION_NAMES,
    SENTIMENT_NAMES,
    compute_emotion_metrics,
    compute_sentiment_metrics,
    log_metrics,
)
from src.models.xlmr import XLMRobertaClassifier


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
    """Set all random seeds for full reproducibility."""
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
# Dataset
# ---------------------------------------------------------------------------

EMOTION2IDX: Dict[str, int] = {name: i for i, name in enumerate(EMOTION_NAMES)}
SENTIMENT2IDX: Dict[str, int] = {
    "negative": 0, "neutral": 1, "positive": 2
}

_SPLIT_CSV = {
    "train": "train_sent_emo.csv",
    "dev": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv",
}


class TranscriptDataset(Dataset):
    """Dataset that serves ASR transcripts paired with MELD emotion labels.

    Args:
        meld_root: Path to MELD root directory (contains CSV files).
        transcripts_path: Path to directory with {split}_transcripts.json.
        split: One of 'train', 'dev', 'test'.
        tokenizer: HuggingFace tokenizer for XLM-RoBERTa.
        max_length: Maximum tokenised sequence length.
    """

    def __init__(
        self,
        meld_root: str,
        transcripts_path: str,
        split: str,
        tokenizer,
        max_length: int = 128,
    ) -> None:
        assert split in ("train", "dev", "test"), (
            f"split must be train/dev/test, got {split}"
        )
        self.split = split
        self.tokenizer = tokenizer
        self.max_length = max_length

        csv_path = Path(meld_root) / _SPLIT_CSV[split]
        if not csv_path.exists():
            raise FileNotFoundError(f"MELD CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )
        df = df.dropna(subset=["emotion", "sentiment"]).reset_index(drop=True)
        df["emotion"] = df["emotion"].str.strip().str.lower()
        df["sentiment"] = df["sentiment"].str.strip().str.lower()

        transcript_file = (
            Path(transcripts_path) / f"{split}_transcripts.json"
        )
        if not transcript_file.exists():
            raise FileNotFoundError(
                f"Transcripts not found: {transcript_file}. "
                "Run preprocessing/transcribe_all.py first."
            )
        with open(transcript_file, "r", encoding="utf-8") as f:
            transcripts: Dict[str, str] = json.load(f)

        self.samples: List[Tuple[str, int, int]] = []
        for _, row in df.iterrows():
            key = f"dia{int(row['dialogue_id'])}_utt{int(row['utterance_id'])}"
            text = transcripts.get(key, "")
            emotion_idx = EMOTION2IDX.get(row["emotion"], 0)
            sentiment_idx = SENTIMENT2IDX.get(row["sentiment"], 1)
            self.samples.append((text, emotion_idx, sentiment_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        text, emotion_label, sentiment_label = self.samples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "emotion_label": torch.tensor(emotion_label, dtype=torch.long),
            "sentiment_label": torch.tensor(sentiment_label, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metric: float,
    config: dict,
    checkpoint_dir: str,
    is_best: bool = False,
) -> None:
    """Save model checkpoint."""
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metric": metric,
        "config": config,
    }
    filename = f"checkpoint_epoch{epoch:03d}_f1{metric:.4f}.pt"
    path = Path(checkpoint_dir) / filename
    torch.save(state, path)
    if is_best:
        torch.save(state, Path(checkpoint_dir) / "best_model.pt")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[int, float]:
    """Load checkpoint; return (start_epoch, best_metric)."""
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    return state["epoch"] + 1, state["metric"]


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: XLMRobertaClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    emotion_criterion: nn.Module,
    sentiment_criterion: nn.Module,
    logger: logging.Logger,
    epoch: int,
) -> float:
    """Run one training epoch; return mean loss."""
    model.train()
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        emotion_labels = batch["emotion_label"].to(device)
        sentiment_labels = batch["sentiment_label"].to(device)

        optimizer.zero_grad()

        sentiment_logits, emotion_logits = model(input_ids, attention_mask)

        loss_emotion = emotion_criterion(emotion_logits, emotion_labels)
        loss_sentiment = sentiment_criterion(sentiment_logits, sentiment_labels)
        loss = loss_emotion + loss_sentiment

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: XLMRobertaClassifier,
    loader: DataLoader,
    device: torch.device,
    emotion_criterion: nn.Module,
    sentiment_criterion: nn.Module,
) -> Tuple[float, Dict, Dict]:
    """Evaluate model; return (mean_loss, emotion_metrics, sentiment_metrics)."""
    model.eval()
    total_loss = 0.0
    all_emotion_preds: List[int] = []
    all_emotion_labels: List[int] = []
    all_sentiment_preds: List[int] = []
    all_sentiment_labels: List[int] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        emotion_labels = batch["emotion_label"].to(device)
        sentiment_labels = batch["sentiment_label"].to(device)

        sentiment_logits, emotion_logits = model(input_ids, attention_mask)

        loss = emotion_criterion(
            emotion_logits, emotion_labels
        ) + sentiment_criterion(sentiment_logits, sentiment_labels)
        total_loss += loss.item()

        all_emotion_preds.extend(
            emotion_logits.argmax(dim=-1).cpu().tolist()
        )
        all_emotion_labels.extend(emotion_labels.cpu().tolist())
        all_sentiment_preds.extend(
            sentiment_logits.argmax(dim=-1).cpu().tolist()
        )
        all_sentiment_labels.extend(sentiment_labels.cpu().tolist())

    emotion_metrics = compute_emotion_metrics(
        all_emotion_preds, all_emotion_labels
    )
    sentiment_metrics = compute_sentiment_metrics(
        all_sentiment_preds, all_sentiment_labels
    )
    return total_loss / len(loader), emotion_metrics, sentiment_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1: fine-tune XLM-RoBERTa on MELD ASR transcripts."
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
    device = get_device()

    log_dir = config["training"]["log_dir"]
    logger = setup_logging(log_dir, "finetune")
    logger.info("Config: %s | Device: %s", args.config, device)

    checkpoint_dir = config["training"]["checkpoint_dir"]

    # ---- Tokenizer & datasets ----
    xlmr_id = config["model"]["xlmr_id"]
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(xlmr_id)

    max_length = config["data"]["max_text_length"]
    num_workers = config["data"]["num_workers"]
    batch_size = config["training"]["batch_size"]

    train_ds = TranscriptDataset(
        meld_root=config["data"]["meld_root"],
        transcripts_path=config["data"]["transcripts_path"],
        split="train",
        tokenizer=tokenizer,
        max_length=max_length,
    )
    dev_ds = TranscriptDataset(
        meld_root=config["data"]["meld_root"],
        transcripts_path=config["data"]["transcripts_path"],
        split="dev",
        tokenizer=tokenizer,
        max_length=max_length,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # ---- Model ----
    model = XLMRobertaClassifier(
        model_name_or_path=xlmr_id,
        num_sentiment_classes=config["model"]["num_sentiment_classes"],
        num_emotion_classes=config["model"]["num_classes"],
        dropout_prob=config["model"]["dropout"],
    )
    model = model.to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        logger.info("Using %d GPUs via DataParallel", torch.cuda.device_count())

    # ---- Optimiser & scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["phase1_lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    total_steps = (
        len(train_loader) * config["training"]["epochs_phase1"]
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=total_steps,
    )

    emotion_criterion = nn.CrossEntropyLoss()
    sentiment_criterion = nn.CrossEntropyLoss()

    # ---- Resume from checkpoint if available ----
    start_epoch = 0
    best_metric = 0.0
    ckpt_dir = Path(checkpoint_dir)
    if ckpt_dir.exists():
        checkpoints = sorted(ckpt_dir.glob("checkpoint_*.pt"))
        if checkpoints:
            latest = checkpoints[-1]
            logger.info("Resuming from checkpoint: %s", latest)
            # Unwrap DataParallel for loading
            m = model.module if hasattr(model, "module") else model
            start_epoch, best_metric = load_checkpoint(
                str(latest), m, optimizer
            )

    # ---- Training loop ----
    total_epochs = config["training"]["epochs_phase1"]
    for epoch in range(start_epoch, total_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            device, emotion_criterion, sentiment_criterion, logger, epoch,
        )
        val_loss, emotion_metrics, sentiment_metrics = evaluate(
            model, dev_loader, device, emotion_criterion, sentiment_criterion
        )

        wf1 = emotion_metrics["weighted_f1"]
        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | "
            "Emotion WF1: %.4f",
            epoch + 1, total_epochs, train_loss, val_loss, wf1,
        )
        log_metrics(emotion_metrics, "dev", "emotion", logger)
        log_metrics(sentiment_metrics, "dev", "sentiment", logger)

        is_best = wf1 > best_metric
        if is_best:
            best_metric = wf1

        m = model.module if hasattr(model, "module") else model
        save_checkpoint(
            m, optimizer, epoch, wf1, config,
            checkpoint_dir, is_best=is_best,
        )

    logger.info("Phase 1 complete. Best emotion WF1: %.4f", best_metric)


if __name__ == "__main__":
    main()
