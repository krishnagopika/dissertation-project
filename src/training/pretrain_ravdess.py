"""
pretrain_ravdess.py — Phase A: RAVDESS Acoustic Backbone Pre-training
======================================================================
Trains an 8-class acoustic emotion classifier on cached Voxtral embeddings
of RAVDESS clips. The trained ``acoustic_proj`` sub-module is then
transferable into MELD fusion models (SumFusion, GatedFusion, CrossModalGating)
in Phase B — see train_fusion.py.

Why 8 classes (and not 7)?
--------------------------
RAVDESS has 8 emotions including ``calm``, which MELD lacks. Per supervisor
guidance we keep all RAVDESS data and use a task-specific 8-class head during
Phase A. When transferring to MELD we replace the head with a 7-class one;
the shared backbone weights carry the cross-emotion acoustic features.

Pipeline
--------
  ravdess_embeddings/embeddings.pt   (dict: stem → 1280-d Tensor)
        ↓
  shuffle and split (seed=42) into train / val / test
        ↓
  AcousticEmotionClassifier (1280 → 512 → 8)  ←  CrossEntropy + class weights
        ↓
  best checkpoint saved with two state-dicts:
    1. full_state_dict — backbone + head (for resuming)
    2. backbone_state_dict — only acoustic_proj.* keys (for fusion transfer)

Usage
-----
  python3.12 src/training/pretrain_ravdess.py --config src/configs/mini.yaml

Slurm: see src/scripts/pretrain_ravdess.sbatch
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.ravdess import EMOTION2IDX as RAVDESS_EMOTION2IDX
from src.data.ravdess import IDX2EMOTION as RAVDESS_IDX2EMOTION
from src.data.ravdess import _parse_emotion_from_filename
from src.models.acoustic_classifier import AcousticEmotionClassifier
from src.evaluation.metrics import compute_emotion_metrics
from src.utils import get_device, load_config, set_seed, setup_logging


# ---------------------------------------------------------------------------
# Dataset over cached embeddings
# ---------------------------------------------------------------------------

class CachedRavdessDataset(Dataset):
    """Dataset that serves (cached_embedding, emotion_label) pairs.

    Args:
        records: List of (stem, embedding, label) tuples.
    """

    def __init__(
        self, records: List[Tuple[str, Tensor, int]],
    ) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        stem, emb, label = self.records[idx]
        return {
            "embedding": emb.float(),
            "label": torch.tensor(label, dtype=torch.long),
            "stem": stem,
        }


def build_records(
    embeddings_path: Path,
    logger: logging.Logger,
) -> List[Tuple[str, Tensor, int]]:
    """Load cached embeddings and pair them with RAVDESS emotion labels.

    Args:
        embeddings_path: Path to the cached .pt embeddings dict.
        logger: Logger instance.

    Returns:
        List of (stem, embedding_tensor, emotion_idx) tuples for files whose
        filename parses to a known emotion.

    Raises:
        FileNotFoundError: If the embeddings file does not exist.
    """
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"RAVDESS embeddings not found: {embeddings_path}. "
            "Run preprocessing/extract_ravdess_embeddings.py first."
        )

    embeddings: Dict[str, Tensor] = torch.load(
        str(embeddings_path), map_location="cpu"
    )
    logger.info("Loaded %d cached RAVDESS embeddings", len(embeddings))

    records: List[Tuple[str, Tensor, int]] = []
    skipped = 0
    for stem, emb in embeddings.items():
        # _parse_emotion_from_filename expects a Path-like with a stem field
        label = _parse_emotion_from_filename(Path(stem + ".wav"))
        if label is None:
            skipped += 1
            continue
        records.append((stem, emb, label))

    logger.info(
        "Parsed %d valid records (skipped %d unparseable filenames)",
        len(records), skipped,
    )
    return records


def split_records(
    records: List[Tuple[str, Tensor, int]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[
    List[Tuple[str, Tensor, int]],
    List[Tuple[str, Tensor, int]],
    List[Tuple[str, Tensor, int]],
]:
    """Shuffle and partition records into train/val/test by fixed seed.

    Args:
        records: Full list of records.
        train_ratio: Fraction allocated to training.
        val_ratio: Fraction allocated to validation.
        seed: RNG seed for reproducibility.

    Returns:
        Tuple of (train, val, test) record lists.
    """
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = shuffled[:n_train]
    val = shuffled[n_train: n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def compute_class_weights(
    labels: List[int],
    num_classes: int,
    device: torch.device,
) -> Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss.

    Args:
        labels: List of integer class indices.
        num_classes: Total number of classes.
        device: Device to place the weight tensor on.

    Returns:
        Tensor of shape ``(num_classes,)`` on ``device``.
    """
    counts = torch.zeros(num_classes)
    for lbl in labels:
        counts[lbl] += 1
    counts = counts.clamp(min=1)
    weights = len(labels) / (num_classes * counts)
    return weights.to(device)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: AcousticEmotionClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float,
) -> float:
    """Run one training epoch over RAVDESS cached embeddings.

    Args:
        model: AcousticEmotionClassifier.
        loader: Training DataLoader.
        optimizer: AdamW optimizer.
        criterion: Loss function.
        device: Target device.
        max_grad_norm: Gradient clipping norm.

    Returns:
        Mean training loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    for batch in loader:
        emb = batch["embedding"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(emb)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: AcousticEmotionClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Dict]:
    """Evaluate model on a data split.

    Args:
        model: AcousticEmotionClassifier.
        loader: Evaluation DataLoader.
        criterion: Loss function.
        device: Target device.

    Returns:
        Tuple of (mean_loss, emotion_metrics_dict).
    """
    model.eval()
    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []
    for batch in loader:
        emb = batch["embedding"].to(device)
        labels = batch["label"].to(device)
        logits = model(emb)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    class_names = [
        RAVDESS_IDX2EMOTION[i]
        for i in sorted(RAVDESS_IDX2EMOTION.keys())
    ]
    metrics = compute_emotion_metrics(
        all_preds, all_labels, class_names=class_names
    )
    return total_loss / len(loader), metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Phase A: pretrain the acoustic backbone on RAVDESS "
            "(8-class emotion classification over cached Voxtral embeddings)."
        )
    )
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["data"]["seed"])
    device = get_device()

    if "ravdess" not in config:
        raise KeyError(
            "Config missing 'ravdess' block — see mini.yaml for the schema."
        )

    log_dir = config["training"]["log_dir"]
    logger = setup_logging(log_dir, "pretrain_ravdess")
    logger.info("Config: %s | Device: %s", args.config, device)

    embeddings_path = Path(config["ravdess"]["embeddings_path"])
    records = build_records(embeddings_path, logger)
    if not records:
        raise RuntimeError(
            "No valid RAVDESS records to train on — check the cached "
            f"embeddings at {embeddings_path}."
        )

    train, val, test = split_records(
        records,
        train_ratio=float(config["ravdess"]["train_ratio"]),
        val_ratio=float(config["ravdess"]["val_ratio"]),
        seed=int(config["data"]["seed"]),
    )
    logger.info(
        "Splits: train=%d  val=%d  test=%d", len(train), len(val), len(test),
    )

    batch_size = int(config["ravdess"]["batch_size"])
    num_workers = int(config["data"]["num_workers"])

    train_loader = DataLoader(
        CachedRavdessDataset(train),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        CachedRavdessDataset(val),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        CachedRavdessDataset(test),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # ---- Model ----
    model = AcousticEmotionClassifier(
        acoustic_dim=int(config["model"]["acoustic_dim"]),
        hidden_dim=int(config["ravdess"]["hidden_dim"]),
        num_classes=int(config["ravdess"]["num_classes"]),
        dropout_prob=float(config["model"]["dropout"]),
    ).to(device)
    logger.info(
        "Model: AcousticEmotionClassifier(acoustic_dim=%d, hidden=%d, "
        "num_classes=%d)",
        model.acoustic_dim, model.hidden_dim, model.num_classes,
    )

    # ---- Loss + optim ----
    train_labels = [r[2] for r in train]
    class_weights = compute_class_weights(
        train_labels, int(config["ravdess"]["num_classes"]), device,
    )
    logger.info(
        "Class weights: %s",
        [f"{w:.3f}" for w in class_weights.cpu().tolist()],
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["ravdess"]["lr"]),
        weight_decay=float(config["ravdess"]["weight_decay"]),
    )

    # ---- Training loop ----
    epochs = int(config["ravdess"]["epochs"])
    max_grad_norm = float(config["training"].get("max_grad_norm", 1.0))
    best_val_wf1 = 0.0
    backbone_ckpt_path = Path(config["ravdess"]["backbone_checkpoint"])
    backbone_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, max_grad_norm,
        )
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, device,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        wf1 = val_metrics["weighted_f1"]
        macro_f1 = val_metrics["macro_f1"]
        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | "
            "Val WF1: %.4f | Macro F1: %.4f",
            epoch + 1, epochs, train_loss, val_loss, wf1, macro_f1,
        )
        for name, score in val_metrics["per_class_f1"].items():
            logger.info("  %-10s F1: %.4f", name, score)

        if wf1 > best_val_wf1:
            best_val_wf1 = wf1
            state = {
                "epoch": epoch,
                "val_weighted_f1": wf1,
                "full_state_dict": model.state_dict(),
                "backbone_state_dict": model.backbone_state_dict(),
                "config": config["ravdess"],
            }
            torch.save(state, str(backbone_ckpt_path))
            logger.info(
                "New best val WF1=%.4f — saved → %s", wf1, backbone_ckpt_path,
            )

    # ---- Final test evaluation ----
    logger.info("Loading best checkpoint for test evaluation...")
    state = torch.load(str(backbone_ckpt_path), map_location="cpu")
    model.load_state_dict(state["full_state_dict"])
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    logger.info(
        "TEST | Loss: %.4f | WF1: %.4f | Macro F1: %.4f",
        test_loss, test_metrics["weighted_f1"], test_metrics["macro_f1"],
    )
    for name, score in test_metrics["per_class_f1"].items():
        logger.info("  %-10s F1: %.4f", name, score)

    # ---- Save metrics JSON ----
    output_dir = Path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "ravdess_pretrain_results.json"
    results = {
        "config": args.config,
        "best_val_weighted_f1": best_val_wf1,
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_per_class_f1": test_metrics["per_class_f1"],
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "backbone_checkpoint": str(backbone_ckpt_path),
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved → %s", results_path)

    # ---- Loss curve plot ----
    if train_losses:
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs_range = range(1, len(train_losses) + 1)
        ax.plot(epochs_range, train_losses, marker="o", label="Train Loss")
        ax.plot(epochs_range, val_losses, marker="o", label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Phase A — RAVDESS Acoustic Backbone Pretrain")
        ax.legend()
        ax.grid(True)
        plot_path = output_dir / "ravdess_pretrain_loss_curve.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Loss curve saved → %s", plot_path)


if __name__ == "__main__":
    main()
