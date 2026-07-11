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

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
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
from src.utils import get_device, load_config, set_seed, setup_logging


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
        paraphrases_path: Optional[str] = None,
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

        self.samples: List[Tuple[str, int, int]] = []
        for _, row in df.iterrows():
            text = str(row.get("utterance", "")).strip()
            emotion_idx = EMOTION2IDX.get(row["emotion"], 0)
            sentiment_idx = SENTIMENT2IDX.get(row["sentiment"], 1)
            self.samples.append((text, emotion_idx, sentiment_idx))

        self.num_original = len(self.samples)
        self.num_paraphrases = 0

        if paraphrases_path is not None and split == "train":
            p = Path(paraphrases_path)
            if not p.exists():
                raise FileNotFoundError(
                    f"Paraphrases enabled but file not found at {p}. "
                    f"Run preprocessing/augment_transcripts.py first."
                )
            with open(p, "r", encoding="utf-8") as f:
                aug = json.load(f).get("paraphrases", [])
            for item in aug:
                text = str(item.get("text", "")).strip()
                if not text:
                    continue
                emotion_idx = EMOTION2IDX.get(item["emotion"], 0)
                sentiment_idx = SENTIMENT2IDX.get(item["sentiment"], 1)
                self.samples.append((text, emotion_idx, sentiment_idx))
            self.num_paraphrases = len(self.samples) - self.num_original

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
# Imbalance handling
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class alpha weights.

    Computes ``- alpha_t * (1 - p_t) ** gamma * log(p_t)`` where ``p_t`` is the
    softmax probability of the ground-truth class. Reduces to weighted
    cross-entropy when ``gamma == 0``.

    Args:
        weight: Optional per-class alpha tensor of shape ``(num_classes,)``.
            If None, no alpha weighting is applied.
        gamma: Focusing parameter. Higher values down-weight easy examples
            more aggressively. Standard choice is 2.0 (Lin et al., 2017).
        reduction: One of ``'mean'``, ``'sum'``, ``'none'``.
    """

    def __init__(
        self,
        weight: Optional[Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        assert reduction in ("mean", "sum", "none"), (
            f"reduction must be mean/sum/none, got {reduction}"
        )
        self.register_buffer(
            "weight", weight if weight is not None else torch.empty(0)
        )
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        focal_factor = (1.0 - pt).pow(self.gamma)

        if self.weight.numel() > 0:
            alpha_t = self.weight.gather(0, target)
            loss = -alpha_t * focal_factor * log_pt
        else:
            loss = -focal_factor * log_pt

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_class_weighted_sampler(
    samples: List[Tuple[str, int, int]],
    num_classes: int,
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that oversamples minority emotion classes.

    Per-sample weight is ``1 / count[emotion_label]``, so each class is drawn
    with equal expected frequency regardless of its raw count.

    Args:
        samples: Dataset samples as (text, emotion_idx, sentiment_idx) tuples.
        num_classes: Number of emotion classes.

    Returns:
        WeightedRandomSampler over the same number of samples as the dataset,
        with replacement.
    """
    counts = torch.zeros(num_classes)
    for _, emo, _ in samples:
        counts[emo] += 1
    counts = counts.clamp(min=1)
    sample_weights = torch.tensor(
        [1.0 / counts[emo].item() for _, emo, _ in samples],
        dtype=torch.double,
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(samples),
        replacement=True,
    )


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
    """Save model checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer state to save (for resuming).
        epoch: Current epoch number.
        metric: Validation weighted F1 at this epoch.
        config: Full config dictionary.
        checkpoint_dir: Directory to save checkpoints.
        is_best: If True, also save as best_model.pt.
    """
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

    # Keep only the latest epoch checkpoint to avoid filling disk
    for old_ckpt in sorted(Path(checkpoint_dir).glob("checkpoint_epoch*.pt"))[:-1]:
        old_ckpt.unlink()


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[int, float]:
    """Load checkpoint; return (start_epoch, best_metric).

    Args:
        checkpoint_path: Path to checkpoint file.
        model: Model to restore weights into.
        optimizer: Optimizer to restore state into.

    Returns:
        Tuple of (start_epoch, best_metric).
    """
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
    """Run one training epoch; return mean loss.

    Args:
        model: XLMRobertaClassifier to train.
        loader: Training DataLoader.
        optimizer: AdamW optimizer.
        scheduler: Linear warmup scheduler.
        device: Target device.
        emotion_criterion: Weighted CrossEntropyLoss for emotion.
        sentiment_criterion: Weighted CrossEntropyLoss for sentiment.
        logger: Logger instance.
        epoch: Current epoch index (0-based).

    Returns:
        Mean training loss for this epoch.
    """
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
    """Evaluate model; return (mean_loss, emotion_metrics, sentiment_metrics).

    Args:
        model: XLMRobertaClassifier.
        loader: Validation DataLoader.
        device: Target device.
        emotion_criterion: Loss function for emotion.
        sentiment_criterion: Loss function for sentiment.

    Returns:
        Tuple of (mean_loss, emotion_metrics_dict, sentiment_metrics_dict).
    """
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
    tokenizer = AutoTokenizer.from_pretrained(xlmr_id)

    max_length = config["data"]["max_text_length"]
    num_workers = config["data"]["num_workers"]
    batch_size = config["training"]["batch_size"]

    use_paraphrases = bool(
        config["training"].get("use_paraphrases", False)
    )
    paraphrases_path: Optional[str] = None
    if use_paraphrases:
        paraphrases_path = config.get("augmentation", {}).get(
            "paraphrases_path"
        )
        if paraphrases_path is None:
            raise KeyError(
                "training.use_paraphrases=true but "
                "augmentation.paraphrases_path is not set."
            )

    train_ds = TranscriptDataset(
        meld_root=config["data"]["meld_root"],
        transcripts_path=config["data"]["transcripts_path"],
        split="train",
        tokenizer=tokenizer,
        max_length=max_length,
        paraphrases_path=paraphrases_path,
    )
    dev_ds = TranscriptDataset(
        meld_root=config["data"]["meld_root"],
        transcripts_path=config["data"]["transcripts_path"],
        split="dev",
        tokenizer=tokenizer,
        max_length=max_length,
    )
    logger.info(
        "Train dataset: %d originals + %d paraphrases = %d total",
        train_ds.num_original, train_ds.num_paraphrases, len(train_ds),
    )

    use_sampler = bool(
        config["training"].get("use_weighted_sampler", False)
    )
    if use_sampler:
        sampler = build_class_weighted_sampler(
            train_ds.samples, config["model"]["num_classes"]
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
        )
        logger.info("Using WeightedRandomSampler on emotion classes")
    else:
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

    # ---- Loss: class weights + optional focal loss ----
    # When the sampler is active each class is drawn uniformly already, so
    # adding inverse-frequency weights to the loss double-corrects and hurts
    # the majority classes. Skip alpha weighting in that case.
    emotion_labels_all = [s[1] for s in train_ds.samples]
    sentiment_labels_all = [s[2] for s in train_ds.samples]

    def compute_class_weights(labels: list, num_classes: int) -> Tensor:
        counts = torch.zeros(num_classes)
        for lbl in labels:
            counts[lbl] += 1
        counts = counts.clamp(min=1)
        weights = len(labels) / (num_classes * counts)
        return weights.to(device)

    if use_sampler:
        emotion_weights = None
        sentiment_weights = None
        logger.info(
            "Sampler active — using uniform alpha in loss (no class weights)"
        )
    else:
        emotion_weights = compute_class_weights(
            emotion_labels_all, config["model"]["num_classes"]
        )
        sentiment_weights = compute_class_weights(
            sentiment_labels_all,
            config["model"]["num_sentiment_classes"],
        )
        logger.info(
            "Emotion class weights: %s",
            [f"{w:.3f}" for w in emotion_weights.cpu().tolist()],
        )

    use_focal = bool(config["training"].get("use_focal_loss", False))
    if use_focal:
        gamma = float(config["training"].get("focal_gamma", 2.0))
        emotion_criterion = FocalLoss(weight=emotion_weights, gamma=gamma)
        sentiment_criterion = FocalLoss(weight=sentiment_weights, gamma=gamma)
        emotion_criterion = emotion_criterion.to(device)
        sentiment_criterion = sentiment_criterion.to(device)
        logger.info("Using FocalLoss(gamma=%.2f) on both heads", gamma)
    else:
        emotion_criterion = nn.CrossEntropyLoss(weight=emotion_weights)
        sentiment_criterion = nn.CrossEntropyLoss(weight=sentiment_weights)
        logger.info("Using CrossEntropyLoss on both heads")

    # ---- Resume from checkpoint if available ----
    start_epoch = 0
    best_metric = 0.0
    ckpt_dir = Path(checkpoint_dir)
    if ckpt_dir.exists():
        checkpoints = sorted(ckpt_dir.glob("checkpoint_*.pt"))
        if checkpoints:
            latest = checkpoints[-1]
            logger.info("Resuming from checkpoint: %s", latest)
            m = model.module if hasattr(model, "module") else model
            start_epoch, best_metric = load_checkpoint(
                str(latest), m, optimizer
            )

    # ---- Training loop ----
    total_epochs = config["training"]["epochs_phase1"]
    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(start_epoch, total_epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            device, emotion_criterion, sentiment_criterion, logger, epoch,
        )
        val_loss, emotion_metrics, sentiment_metrics = evaluate(
            model, dev_loader, device, emotion_criterion, sentiment_criterion
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

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

    # ---- Loss curve plot ----
    if train_losses:
        plot_dir = Path(config["evaluation"]["output_dir"])
        plot_dir.mkdir(parents=True, exist_ok=True)
        epochs_range = range(start_epoch + 1, start_epoch + len(train_losses) + 1)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs_range, train_losses, marker="o", label="Train Loss")
        ax.plot(epochs_range, val_losses, marker="o", label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Phase 1 — XLM-RoBERTa Fine-tuning Loss")
        ax.legend()
        ax.grid(True)
        plot_path = plot_dir / "phase1_loss_curve.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Loss curve saved to %s", plot_path)


if __name__ == "__main__":
    main()
