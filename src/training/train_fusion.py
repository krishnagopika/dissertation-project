"""
train_fusion.py — Phase 2: Multimodal Fusion Training
======================================================
Trains the chosen fusion architecture using pre-cached embeddings only.
Neither XLM-RoBERTa nor Voxtral is loaded at runtime — both modalities
are read from pre-extracted .pt files:

  data/meld_text_embeddings/{split}_text_embeddings.pt   (768-dim, gold text)
  data/meld_embeddings/{split}_embeddings.pt             (1280-dim, Voxtral)

Supported fusion types (--fusion_type):
  concat      — Concatenation MLP (baseline)
  sum         — Element-wise sum (ablation)
  gated       — Symmetric gated fusion
  crossmodal  — Asymmetric cross-modal gating

Prerequisites
-------------
  1. Phase 1 complete  → checkpoints/mini/best_model.pt
  2. Text embeddings extracted → run extract_text_embeddings.py

Usage
-----
  python3.12 src/training/train_fusion.py \\
      --config src/configs/mini.yaml \\
      --fusion_type concat

Slurm: see src/scripts/train_fusion.sbatch
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
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import (
    EMOTION_NAMES,
    compute_emotion_metrics,
    compute_sentiment_metrics,
    log_metrics,
)
from src.models.fusion import CrossModalGating, FusionModel, GatedFusion, SumFusion
from src.utils import get_device, load_config, set_seed, setup_logging

_FUSION_CLASSES = {
    "concat":     FusionModel,
    "sum":        SumFusion,
    "gated":      GatedFusion,
    "crossmodal": CrossModalGating,
}

EMOTION2IDX: Dict[str, int] = {name: i for i, name in enumerate(EMOTION_NAMES)}
SENTIMENT2IDX: Dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}

_SPLIT_CSV: Dict[str, str] = {
    "train": "train_sent_emo.csv",
    "dev":   "dev_sent_emo.csv",
    "test":  "test_sent_emo.csv",
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FusionDataset(Dataset):
    """Dataset serving pre-cached text and acoustic embeddings.

    Loads both modalities from pre-extracted .pt files — no model inference
    occurs during data loading.

    Each sample contains:
    - text_embedding      (pre-computed XLM-RoBERTa CLS vector, 768-dim)
    - acoustic_embedding  (pre-extracted Voxtral encoder output, 1280-dim)
    - emotion_label       (int, 0-6)
    - sentiment_label     (int, 0-2)

    Args:
        meld_root: Path to MELD root directory (contains CSV files).
        text_embeddings_path: Directory containing {split}_text_embeddings.pt.
        embeddings_path: Directory containing {split}_embeddings.pt.
        split: One of 'train', 'dev', 'test'.
        text_dim: Expected text embedding dimension.
        acoustic_dim: Expected acoustic embedding dimension.
    """

    def __init__(
        self,
        meld_root: str,
        text_embeddings_path: str,
        embeddings_path: str,
        split: str,
        text_dim: int = 768,
        acoustic_dim: int = 1280,
        filtered_keys_path: Optional[str] = None,
    ) -> None:
        assert split in ("train", "dev", "test"), (
            f"split must be train/dev/test, got {split}"
        )
        self.split = split
        self.text_dim = text_dim
        self.acoustic_dim = acoustic_dim

        csv_path = Path(meld_root) / _SPLIT_CSV[split]
        if not csv_path.exists():
            raise FileNotFoundError(f"MELD CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        df.columns = (
            df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
        )
        df = df.dropna(subset=["emotion", "sentiment"]).reset_index(drop=True)
        df["emotion"]   = df["emotion"].str.strip().str.lower()
        df["sentiment"] = df["sentiment"].str.strip().str.lower()

        # Optional VAD+WER keep-list from apply_filter.py
        keep_set: Optional[set] = None
        self.num_before_filter = len(df)
        if filtered_keys_path is not None:
            p = Path(filtered_keys_path)
            if not p.exists():
                raise FileNotFoundError(
                    f"filtered_keys_path set but file not found at {p}. "
                    "Run preprocessing/compute_filter_metadata.py then "
                    "preprocessing/apply_filter.py first."
                )
            with open(p, "r", encoding="utf-8") as f:
                keep_set = set(json.load(f)["keys"])

        text_emb_file = Path(text_embeddings_path) / f"{split}_text_embeddings.pt"
        if not text_emb_file.exists():
            raise FileNotFoundError(
                f"Text embeddings not found: {text_emb_file}. "
                "Run src/preprocessing/extract_text_embeddings.py first."
            )
        text_embeddings: Dict[str, Tensor] = torch.load(
            str(text_emb_file), map_location="cpu"
        )

        acoustic_emb_file = Path(embeddings_path) / f"{split}_embeddings.pt"
        if not acoustic_emb_file.exists():
            raise FileNotFoundError(
                f"Acoustic embeddings not found: {acoustic_emb_file}. "
                "Run src/preprocessing/transcribe_all.py first."
            )
        acoustic_embeddings: Dict[str, Tensor] = torch.load(
            str(acoustic_emb_file), map_location="cpu"
        )

        self.samples: List[Tuple[Tensor, Tensor, int, int]] = []
        for _, row in df.iterrows():
            key = f"dia{int(row['dialogue_id'])}_utt{int(row['utterance_id'])}"
            if keep_set is not None and key not in keep_set:
                continue
            text_emb = text_embeddings.get(
                key, torch.zeros(text_dim, dtype=torch.float32)
            )
            acoustic_emb = acoustic_embeddings.get(
                key, torch.zeros(acoustic_dim, dtype=torch.float32)
            )
            emotion_idx   = EMOTION2IDX.get(row["emotion"], 0)
            sentiment_idx = SENTIMENT2IDX.get(row["sentiment"], 1)
            self.samples.append((text_emb, acoustic_emb, emotion_idx, sentiment_idx))

        self.num_after_filter = len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        text_emb, acoustic_emb, emotion_label, sentiment_label = self.samples[idx]
        return {
            "text_embedding":     text_emb.float(),
            "acoustic_embedding": acoustic_emb.float(),
            "emotion_label":      torch.tensor(emotion_label, dtype=torch.long),
            "sentiment_label":    torch.tensor(sentiment_label, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    fusion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metric: float,
    config: dict,
    checkpoint_dir: str,
    is_best: bool = False,
) -> None:
    """Save fusion model checkpoint.

    Args:
        fusion: FusionModel to save.
        optimizer: Optimizer state for resumption.
        epoch: Current epoch number.
        metric: Validation weighted F1.
        config: Full config dictionary.
        checkpoint_dir: Directory to write checkpoints.
        is_best: If True, also write as best_model.pt.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    state = {
        "epoch":                epoch,
        "fusion_state_dict":    fusion.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metric":               metric,
        "config":               config,
    }
    filename = f"checkpoint_epoch{epoch:03d}_f1{metric:.4f}.pt"
    torch.save(state, Path(checkpoint_dir) / filename)
    if is_best:
        torch.save(state, Path(checkpoint_dir) / "best_model.pt")

    # Keep only the latest epoch checkpoint to avoid filling disk
    for old_ckpt in sorted(Path(checkpoint_dir).glob("checkpoint_epoch*.pt"))[:-1]:
        old_ckpt.unlink()


def load_checkpoint(
    checkpoint_path: str,
    fusion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[int, float]:
    """Load checkpoint and restore fusion and optimizer state.

    Args:
        checkpoint_path: Path to checkpoint .pt file.
        fusion: FusionModel to restore.
        optimizer: Optimizer to restore.

    Returns:
        Tuple of (start_epoch, best_metric).
    """
    state = torch.load(checkpoint_path, map_location="cpu")
    fusion.load_state_dict(state["fusion_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    return state["epoch"] + 1, state["metric"]


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    fusion: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    emotion_criterion: nn.Module,
    sentiment_criterion: nn.Module,
    max_grad_norm: float,
) -> float:
    """Run one Phase 2 training epoch over pre-cached embeddings.

    Args:
        fusion: FusionModel (trainable).
        loader: Training DataLoader.
        optimizer: AdamW optimizer.
        scheduler: Linear warmup scheduler.
        device: Target device.
        emotion_criterion: Weighted CrossEntropyLoss for emotion.
        sentiment_criterion: Weighted CrossEntropyLoss for sentiment.
        max_grad_norm: Gradient clipping norm.

    Returns:
        Mean training loss for this epoch.
    """
    fusion.train()
    total_loss = 0.0

    for batch in loader:
        text_emb         = batch["text_embedding"].to(device)
        acoustic_emb     = batch["acoustic_embedding"].to(device)
        emotion_labels   = batch["emotion_label"].to(device)
        sentiment_labels = batch["sentiment_label"].to(device)

        optimizer.zero_grad()

        sentiment_logits, emotion_logits = fusion(text_emb, acoustic_emb)

        loss = (
            emotion_criterion(emotion_logits, emotion_labels)
            + sentiment_criterion(sentiment_logits, sentiment_labels)
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fusion.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    fusion: nn.Module,
    loader: DataLoader,
    device: torch.device,
    emotion_criterion: nn.Module,
    sentiment_criterion: nn.Module,
) -> Tuple[float, Dict, Dict]:
    """Evaluate fusion model on a data split.

    Args:
        fusion: FusionModel.
        loader: DataLoader for the evaluation split.
        device: Target device.
        emotion_criterion: Loss function for emotion.
        sentiment_criterion: Loss function for sentiment.

    Returns:
        Tuple of (mean_loss, emotion_metrics_dict, sentiment_metrics_dict).
    """
    fusion.eval()
    total_loss = 0.0
    all_emotion_preds:    List[int] = []
    all_emotion_labels:   List[int] = []
    all_sentiment_preds:  List[int] = []
    all_sentiment_labels: List[int] = []

    for batch in loader:
        text_emb         = batch["text_embedding"].to(device)
        acoustic_emb     = batch["acoustic_embedding"].to(device)
        emotion_labels   = batch["emotion_label"].to(device)
        sentiment_labels = batch["sentiment_label"].to(device)

        sentiment_logits, emotion_logits = fusion(text_emb, acoustic_emb)

        loss = (
            emotion_criterion(emotion_logits, emotion_labels)
            + sentiment_criterion(sentiment_logits, sentiment_labels)
        )
        total_loss += loss.item()

        all_emotion_preds.extend(emotion_logits.argmax(dim=-1).cpu().tolist())
        all_emotion_labels.extend(emotion_labels.cpu().tolist())
        all_sentiment_preds.extend(sentiment_logits.argmax(dim=-1).cpu().tolist())
        all_sentiment_labels.extend(sentiment_labels.cpu().tolist())

    emotion_metrics   = compute_emotion_metrics(all_emotion_preds, all_emotion_labels)
    sentiment_metrics = compute_sentiment_metrics(all_sentiment_preds, all_sentiment_labels)
    return total_loss / len(loader), emotion_metrics, sentiment_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: train multimodal fusion on cached embeddings."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--fusion_type",
        type=str,
        default="concat",
        choices=list(_FUSION_CLASSES.keys()),
    )
    args = parser.parse_args()

    config      = load_config(args.config)
    fusion_type = args.fusion_type
    set_seed(config["data"]["seed"])
    device      = get_device()

    log_dir = config["training"]["log_dir"]
    logger  = setup_logging(log_dir, f"train_fusion_{fusion_type}")
    logger.info(
        "Config: %s | Fusion type: %s | Device: %s",
        args.config, fusion_type, device,
    )

    checkpoint_dir = (
        Path(config["training"]["checkpoint_dir"]) / "fusion" / fusion_type
    )

    text_dim     = config["model"]["text_dim"]
    acoustic_dim = config["model"]["acoustic_dim"]
    num_workers  = config["data"]["num_workers"]
    batch_size   = config["training"]["batch_size"]

    filt_cfg = config.get("filtering", {})
    use_filter = bool(filt_cfg.get("enabled", False))
    train_filter_keys: Optional[str] = None
    dev_filter_keys: Optional[str] = None
    if use_filter:
        keys_paths = filt_cfg.get("keys_paths", {})
        train_filter_keys = keys_paths.get("train")
        dev_filter_keys = keys_paths.get("dev")
        if train_filter_keys is None or dev_filter_keys is None:
            raise KeyError(
                "filtering.enabled=true but filtering.keys_paths.{train,dev} "
                "not fully set in the config."
            )

    train_ds = FusionDataset(
        meld_root            = config["data"]["meld_root"],
        text_embeddings_path = config["data"]["text_embeddings_path"],
        embeddings_path      = config["data"]["embeddings_path"],
        split                = "train",
        text_dim             = text_dim,
        acoustic_dim         = acoustic_dim,
        filtered_keys_path   = train_filter_keys,
    )
    dev_ds = FusionDataset(
        meld_root            = config["data"]["meld_root"],
        text_embeddings_path = config["data"]["text_embeddings_path"],
        embeddings_path      = config["data"]["embeddings_path"],
        split                = "dev",
        text_dim             = text_dim,
        acoustic_dim         = acoustic_dim,
        filtered_keys_path   = dev_filter_keys,
    )
    logger.info("Train: %d | Dev: %d samples", len(train_ds), len(dev_ds))
    if use_filter:
        logger.info(
            "Filter enabled | train: %d/%d kept | dev: %d/%d kept",
            train_ds.num_after_filter, train_ds.num_before_filter,
            dev_ds.num_after_filter, dev_ds.num_before_filter,
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )

    # ---- Fusion model ----
    FusionClass = _FUSION_CLASSES[fusion_type]
    logger.info("Instantiating fusion model: %s", FusionClass.__name__)
    fusion = FusionClass(
        acoustic_dim          = acoustic_dim,
        text_dim              = text_dim,
        hidden_dim            = config["model"]["fusion_hidden"],
        num_sentiment_classes = config["model"]["num_sentiment_classes"],
        num_emotion_classes   = config["model"]["num_classes"],
        dropout_prob          = config["model"]["dropout"],
    )

    # ---- Phase B: optionally warm-start acoustic_proj from RAVDESS pretrain ----
    if bool(config.get("use_ravdess_pretrain", False)):
        if not hasattr(fusion, "acoustic_proj"):
            logger.warning(
                "use_ravdess_pretrain=true but %s has no acoustic_proj — "
                "FusionModel (concat) feeds the acoustic embedding into a "
                "joint MLP, so there is no transferable submodule. "
                "Skipping warm-start.",
                FusionClass.__name__,
            )
        else:
            backbone_ckpt = Path(config["ravdess"]["backbone_checkpoint"])
            if not backbone_ckpt.exists():
                raise FileNotFoundError(
                    f"RAVDESS backbone checkpoint not found: {backbone_ckpt}. "
                    "Run pretrain_ravdess.sbatch first."
                )
            state = torch.load(str(backbone_ckpt), map_location="cpu")
            backbone_state = state["backbone_state_dict"]
            # backbone_state keys are 'acoustic_proj.0.weight' etc. — strip
            # the prefix and load into fusion.acoustic_proj directly.
            stripped = {
                k[len("acoustic_proj."):]: v
                for k, v in backbone_state.items()
                if k.startswith("acoustic_proj.")
            }
            missing, unexpected = fusion.acoustic_proj.load_state_dict(
                stripped, strict=True,
            )
            logger.info(
                "Warm-started acoustic_proj from %s (val WF1 on RAVDESS: %.4f)",
                backbone_ckpt, state.get("val_weighted_f1", float("nan")),
            )

    fusion = fusion.to(device)

    if torch.cuda.device_count() > 1:
        fusion = torch.nn.DataParallel(fusion)
        logger.info("Using %d GPUs via DataParallel", torch.cuda.device_count())

    fusion_module = fusion.module if hasattr(fusion, "module") else fusion
    trainable = sum(p.numel() for p in fusion.parameters() if p.requires_grad)
    logger.info("Trainable fusion parameters: %d", trainable)

    # ---- Optimiser & scheduler ----
    optimizer = torch.optim.AdamW(
        fusion.parameters(),
        lr           = config["training"]["phase2_lr"],
        weight_decay = config["training"]["weight_decay"],
    )
    total_steps = len(train_loader) * config["training"]["epochs_phase2"]
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = config["training"]["warmup_steps"],
        num_training_steps = total_steps,
    )

    # ---- Class weights to handle MELD imbalance ----
    emotion_labels_all   = [s[2] for s in train_ds.samples]
    sentiment_labels_all = [s[3] for s in train_ds.samples]

    def compute_class_weights(labels: list, num_classes: int) -> Tensor:
        counts = torch.zeros(num_classes)
        for lbl in labels:
            counts[lbl] += 1
        counts = counts.clamp(min=1)
        weights = len(labels) / (num_classes * counts)
        return weights.to(device)

    emotion_weights   = compute_class_weights(
        emotion_labels_all, config["model"]["num_classes"]
    )
    sentiment_weights = compute_class_weights(
        sentiment_labels_all, config["model"]["num_sentiment_classes"]
    )
    logger.info(
        "Emotion class weights: %s",
        [f"{w:.3f}" for w in emotion_weights.cpu().tolist()],
    )

    emotion_criterion   = nn.CrossEntropyLoss(weight=emotion_weights)
    sentiment_criterion = nn.CrossEntropyLoss(weight=sentiment_weights)

    # ---- Resume from checkpoint if available ----
    start_epoch = 0
    best_metric = 0.0
    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch*.pt"))
        if checkpoints:
            latest = checkpoints[-1]
            logger.info("Resuming from checkpoint: %s", latest)
            start_epoch, best_metric = load_checkpoint(
                str(latest), fusion_module, optimizer
            )

    # ---- Training loop ----
    total_epochs  = config["training"]["epochs_phase2"]
    max_grad_norm = config["training"].get("max_grad_norm", 1.0)
    train_losses: List[float] = []
    val_losses:   List[float] = []

    for epoch in range(start_epoch, total_epochs):
        train_loss = train_one_epoch(
            fusion, train_loader, optimizer, scheduler,
            device, emotion_criterion, sentiment_criterion, max_grad_norm,
        )
        val_loss, emotion_metrics, sentiment_metrics = evaluate(
            fusion, dev_loader, device, emotion_criterion, sentiment_criterion,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        wf1 = emotion_metrics["weighted_f1"]
        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | Emotion WF1: %.4f",
            epoch + 1, total_epochs, train_loss, val_loss, wf1,
        )
        log_metrics(emotion_metrics,   "dev", "emotion",   logger)
        log_metrics(sentiment_metrics, "dev", "sentiment", logger)

        is_best = wf1 > best_metric
        if is_best:
            best_metric = wf1

        save_checkpoint(
            fusion_module, optimizer, epoch, wf1, config,
            str(checkpoint_dir), is_best=is_best,
        )

    logger.info("Phase 2 complete. Best emotion WF1: %.4f", best_metric)

    # ---- Loss curve ----
    if train_losses:
        plot_dir = Path(config["evaluation"]["output_dir"])
        plot_dir.mkdir(parents=True, exist_ok=True)
        epochs_range = range(start_epoch + 1, start_epoch + len(train_losses) + 1)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs_range, train_losses, marker="o", label="Train Loss")
        ax.plot(epochs_range, val_losses,   marker="o", label="Val Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"Phase 2 — Fusion Training Loss ({fusion_type})")
        ax.legend()
        ax.grid(True)
        plot_path = plot_dir / f"phase2_loss_curve_{fusion_type}.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Loss curve saved to %s", plot_path)


if __name__ == "__main__":
    main()
