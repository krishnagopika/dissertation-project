"""
pretrain_combined.py — Joint acoustic training on RAVDESS + MELD
==================================================================
Trains a single 7-class acoustic emotion classifier on the union of:

  * RAVDESS speech clips     (~1440, labels mapped into MELD's space)
  * MELD train acoustic embeddings (~9989, native MELD labels)

All labels live in MELD's 7-class taxonomy. RAVDESS ``calm`` is mapped to
MELD ``neutral`` per supervisor guidance (no data dropped). The model is
early-stopped on the **MELD dev set** so selection is aligned with the
target domain.

Two artefacts are saved at the best epoch:

  1. Full state dict (backbone + 7-class head)
     → use for acoustic-only inference on MELD test as an ablation row.
  2. Backbone-only state dict matching ``SumFusion.acoustic_proj`` keys
     → drop directly into MELD fusion variants in train_fusion.py.

Pipeline
--------
  RAVDESS cached embeddings + MELD train embeddings  (in MELD label space)
        ↓
  AcousticEmotionClassifier (1280 → 512 → 7)
        ↓
  best checkpoint (selected on MELD dev WF1)
        ↓
  test on MELD test → results/mini/combined_acoustic_results.json

Usage
-----
  python3.12 src/training/pretrain_combined.py --config src/configs/mini.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.ravdess import _parse_emotion_from_filename
from src.evaluation.metrics import EMOTION_NAMES, compute_emotion_metrics
from src.models.acoustic_classifier import AcousticEmotionClassifier
from src.utils import get_device, load_config, set_seed, setup_logging


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

# MELD's 7-class indices (from src/evaluation/metrics.py):
#   0 neutral, 1 surprise, 2 fear, 3 sadness, 4 joy, 5 disgust, 6 anger
#
# RAVDESS's 8-class indices (from src/data/ravdess.py):
#   0 neutral, 1 calm, 2 happy, 3 sad, 4 angry, 5 fearful, 6 disgust, 7 surprised
RAVDESS_TO_MELD: Dict[int, int] = {
    0: 0,   # neutral  → neutral
    1: 0,   # calm     → neutral   (supervisor: do not drop)
    2: 4,   # happy    → joy
    3: 3,   # sad      → sadness
    4: 6,   # angry    → anger
    5: 2,   # fearful  → fear
    6: 5,   # disgust  → disgust
    7: 1,   # surprised → surprise
}

MELD_EMOTION2IDX: Dict[str, int] = {n: i for i, n in enumerate(EMOTION_NAMES)}

_SPLIT_CSV: Dict[str, str] = {
    "train": "train_sent_emo.csv",
    "dev":   "dev_sent_emo.csv",
    "test":  "test_sent_emo.csv",
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AcousticOnlyDataset(Dataset):
    """Serves (acoustic_embedding, meld_label) pairs from a record list.

    Args:
        records: List of (key, embedding_tensor, meld_label) tuples.
    """

    def __init__(
        self, records: List[Tuple[str, Tensor, int]],
    ) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        key, emb, label = self.records[idx]
        return {
            "embedding": emb.float(),
            "label": torch.tensor(label, dtype=torch.long),
            "key": key,
        }


def load_ravdess_records(
    embeddings_path: Path,
    logger: logging.Logger,
) -> List[Tuple[str, Tensor, int]]:
    """Load RAVDESS cached embeddings, map labels into MELD space.

    Args:
        embeddings_path: Path to RAVDESS embeddings.pt.
        logger: Logger instance.

    Returns:
        List of (stem, embedding, meld_label) tuples.
    """
    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"RAVDESS embeddings not found: {embeddings_path}. "
            "Run preprocessing/extract_ravdess_embeddings.py first."
        )
    embeddings: Dict[str, Tensor] = torch.load(
        str(embeddings_path), map_location="cpu"
    )
    records: List[Tuple[str, Tensor, int]] = []
    skipped = 0
    for stem, emb in embeddings.items():
        ravdess_idx = _parse_emotion_from_filename(Path(stem + ".wav"))
        if ravdess_idx is None:
            skipped += 1
            continue
        meld_idx = RAVDESS_TO_MELD[ravdess_idx]
        records.append((stem, emb, meld_idx))
    logger.info(
        "RAVDESS: %d records loaded (skipped %d), labels mapped to MELD space",
        len(records), skipped,
    )
    return records


def load_meld_records(
    meld_root: Path,
    embeddings_path: Path,
    split: str,
    acoustic_dim: int,
    logger: logging.Logger,
) -> List[Tuple[str, Tensor, int]]:
    """Load MELD cached acoustic embeddings paired with emotion labels.

    Args:
        meld_root: Path to MELD root with CSV files.
        embeddings_path: Path to dir with {split}_embeddings.pt.
        split: One of 'train', 'dev', 'test'.
        acoustic_dim: Expected embedding dim (for fallback zero tensor).
        logger: Logger instance.

    Returns:
        List of (key, embedding, meld_label) tuples.
    """
    csv_path = meld_root / _SPLIT_CSV[split]
    if not csv_path.exists():
        raise FileNotFoundError(f"MELD CSV not found: {csv_path}")

    emb_file = embeddings_path / f"{split}_embeddings.pt"
    if not emb_file.exists():
        raise FileNotFoundError(
            f"MELD acoustic embeddings not found: {emb_file}. "
            "Run preprocessing/transcribe_all.py first."
        )
    acoustic: Dict[str, Tensor] = torch.load(
        str(emb_file), map_location="cpu",
    )

    df = pd.read_csv(csv_path)
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    )
    df = df.dropna(subset=["emotion"]).reset_index(drop=True)
    df["emotion"] = df["emotion"].str.strip().str.lower()

    records: List[Tuple[str, Tensor, int]] = []
    for _, row in df.iterrows():
        key = f"dia{int(row['dialogue_id'])}_utt{int(row['utterance_id'])}"
        emb = acoustic.get(key, torch.zeros(acoustic_dim, dtype=torch.float32))
        label = MELD_EMOTION2IDX.get(row["emotion"], 0)
        records.append((key, emb, label))

    logger.info("MELD %s: %d records loaded", split, len(records))
    return records


# ---------------------------------------------------------------------------
# Loss helper
# ---------------------------------------------------------------------------

def compute_class_weights(
    labels: List[int],
    num_classes: int,
    device: torch.device,
) -> Tensor:
    """Inverse-frequency class weights."""
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
    """Run one training epoch over combined audio."""
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
    """Evaluate model, return (mean_loss, MELD-space metrics)."""
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
    metrics = compute_emotion_metrics(all_preds, all_labels)
    return total_loss / len(loader), metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combined acoustic training on RAVDESS + MELD, all labels mapped "
            "into MELD's 7-class emotion space. Validated on MELD dev."
        )
    )
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["data"]["seed"])
    device = get_device()

    if "combined" not in config or "ravdess" not in config:
        raise KeyError(
            "Config missing 'combined' or 'ravdess' block — see mini.yaml."
        )

    log_dir = config["training"]["log_dir"]
    logger = setup_logging(log_dir, "pretrain_combined")
    logger.info("Config: %s | Device: %s", args.config, device)
    logger.info("RAVDESS→MELD label map: %s", RAVDESS_TO_MELD)

    acoustic_dim = int(config["model"]["acoustic_dim"])
    meld_root = Path(config["data"]["meld_root"])
    meld_emb = Path(config["data"]["embeddings_path"])
    ravdess_emb = Path(config["ravdess"]["embeddings_path"])

    # ---- Load all splits ----
    ravdess = load_ravdess_records(ravdess_emb, logger)
    meld_train = load_meld_records(meld_root, meld_emb, "train", acoustic_dim, logger)
    meld_dev = load_meld_records(meld_root, meld_emb, "dev", acoustic_dim, logger)
    meld_test = load_meld_records(meld_root, meld_emb, "test", acoustic_dim, logger)

    train_records = ravdess + meld_train  # all of RAVDESS counts as train
    logger.info(
        "Combined train: %d (RAVDESS=%d + MELD_train=%d) | val (MELD dev): %d | test (MELD test): %d",
        len(train_records), len(ravdess), len(meld_train),
        len(meld_dev), len(meld_test),
    )

    batch_size = int(config["combined"]["batch_size"])
    num_workers = int(config["data"]["num_workers"])

    train_loader = DataLoader(
        AcousticOnlyDataset(train_records),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        AcousticOnlyDataset(meld_dev),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        AcousticOnlyDataset(meld_test),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    # ---- Model ----
    num_classes = int(config["model"]["num_classes"])  # 7 for MELD
    model = AcousticEmotionClassifier(
        acoustic_dim=acoustic_dim,
        hidden_dim=int(config["combined"]["hidden_dim"]),
        num_classes=num_classes,
        dropout_prob=float(config["model"]["dropout"]),
    ).to(device)
    logger.info(
        "Model: AcousticEmotionClassifier(acoustic_dim=%d, hidden=%d, num_classes=%d)",
        model.acoustic_dim, model.hidden_dim, model.num_classes,
    )

    # ---- Loss + optim ----
    train_labels = [r[2] for r in train_records]
    class_weights = compute_class_weights(train_labels, num_classes, device)
    logger.info(
        "Class weights (MELD space): %s",
        [f"{w:.3f}" for w in class_weights.cpu().tolist()],
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["combined"]["lr"]),
        weight_decay=float(config["combined"]["weight_decay"]),
    )

    # ---- Training loop ----
    epochs = int(config["combined"]["epochs"])
    max_grad_norm = float(config["training"].get("max_grad_norm", 1.0))
    backbone_path = Path(config["combined"]["backbone_checkpoint"])
    full_path = Path(config["combined"]["full_checkpoint"])
    backbone_path.parent.mkdir(parents=True, exist_ok=True)

    best_val_wf1 = 0.0
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
            "Epoch %d/%d | Train Loss: %.4f | MELD-dev Loss: %.4f | "
            "WF1: %.4f | Macro F1: %.4f",
            epoch + 1, epochs, train_loss, val_loss, wf1, macro_f1,
        )
        for name, score in val_metrics["per_class_f1"].items():
            logger.info("  %-10s F1: %.4f", name, score)

        if wf1 > best_val_wf1:
            best_val_wf1 = wf1
            backbone_state = model.backbone_state_dict()
            full_state = {
                "epoch": epoch,
                "val_weighted_f1": wf1,
                "full_state_dict": model.state_dict(),
                "backbone_state_dict": backbone_state,
                "config": config["combined"],
                "label_mapping": RAVDESS_TO_MELD,
            }
            torch.save(full_state, str(full_path))
            # backbone-only file mirrors the schema used by pretrain_ravdess.py
            torch.save(
                {
                    "epoch": epoch,
                    "val_weighted_f1": wf1,
                    "backbone_state_dict": backbone_state,
                    "config": config["combined"],
                },
                str(backbone_path),
            )
            logger.info(
                "New best MELD-dev WF1=%.4f — saved → %s (full), %s (backbone)",
                wf1, full_path, backbone_path,
            )

    # ---- Final MELD test evaluation (acoustic-only) ----
    logger.info("Loading best checkpoint for MELD test evaluation...")
    state = torch.load(str(full_path), map_location="cpu")
    model.load_state_dict(state["full_state_dict"])
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    logger.info(
        "MELD TEST (acoustic-only) | Loss: %.4f | WF1: %.4f | Macro F1: %.4f",
        test_loss, test_metrics["weighted_f1"], test_metrics["macro_f1"],
    )
    for name, score in test_metrics["per_class_f1"].items():
        logger.info("  %-10s F1: %.4f", name, score)

    # ---- Save metrics JSON ----
    output_dir = Path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "combined_acoustic_results.json"
    results = {
        "config": args.config,
        "model": "acoustic_only_combined",
        "label_mapping": RAVDESS_TO_MELD,
        "best_meld_dev_weighted_f1": best_val_wf1,
        "test_samples_meld": len(meld_test),
        "meld_test": {
            "weighted_f1": test_metrics["weighted_f1"],
            "macro_f1": test_metrics["macro_f1"],
            "per_class_f1": test_metrics["per_class_f1"],
        },
        "n_train_ravdess": len(ravdess),
        "n_train_meld": len(meld_train),
        "n_val_meld_dev": len(meld_dev),
        "n_test_meld_test": len(meld_test),
        "backbone_checkpoint": str(backbone_path),
        "full_checkpoint": str(full_path),
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved → %s", results_path)

    # ---- Loss curve plot ----
    if train_losses:
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs_range = range(1, len(train_losses) + 1)
        ax.plot(epochs_range, train_losses, marker="o", label="Train Loss")
        ax.plot(epochs_range, val_losses, marker="o", label="MELD-dev Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Combined Acoustic Training (RAVDESS + MELD)")
        ax.legend()
        ax.grid(True)
        plot_path = output_dir / "combined_acoustic_loss_curve.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Loss curve saved → %s", plot_path)


if __name__ == "__main__":
    main()
