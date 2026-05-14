"""
evaluate.py — Final Evaluation on MELD Test Set
================================================
Loads the Phase 2 fusion checkpoint and evaluates on the MELD test set
using pre-cached text and acoustic embeddings (no XLM-RoBERTa at runtime).

Reports the following metrics for both emotion (7-class) and sentiment (3-class):
  - Accuracy
  - Weighted Precision / Recall / F1
  - Macro F1
  - AUC (one-vs-rest, macro-averaged)
  - Per-class F1

Saves:
  results/{mini|small}/test_results_{fusion_type}.json
  results/{mini|small}/confusion_emotion_{fusion_type}.png
  results/{mini|small}/confusion_sentiment_{fusion_type}.png

Usage
-----
  python3.12 src/evaluation/evaluate.py \\
      --config src/configs/mini.yaml \\
      --fusion_type concat

Slurm: see src/scripts/evaluate.sbatch
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import EMOTION_NAMES, SENTIMENT_NAMES, log_metrics
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

class EvalDataset(Dataset):
    """Dataset serving pre-cached text and acoustic embeddings for evaluation.

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
            text_emb = text_embeddings.get(
                key, torch.zeros(text_dim, dtype=torch.float32)
            )
            acoustic_emb = acoustic_embeddings.get(
                key, torch.zeros(acoustic_dim, dtype=torch.float32)
            )
            emotion_idx   = EMOTION2IDX.get(row["emotion"], 0)
            sentiment_idx = SENTIMENT2IDX.get(row["sentiment"], 1)
            self.samples.append((text_emb, acoustic_emb, emotion_idx, sentiment_idx))

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
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    fusion: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int], np.ndarray, List[int], List[int], np.ndarray]:
    """Run inference and collect predictions and probabilities.

    Args:
        fusion: Trained fusion model.
        loader: DataLoader for the evaluation split.
        device: Target device.

    Returns:
        Tuple of (emotion_preds, emotion_labels, emotion_probs,
                  sentiment_preds, sentiment_labels, sentiment_probs).
    """
    fusion.eval()

    emotion_preds:     List[int] = []
    emotion_labels:    List[int] = []
    emotion_probs_l:   List[np.ndarray] = []
    sentiment_preds:   List[int] = []
    sentiment_labels:  List[int] = []
    sentiment_probs_l: List[np.ndarray] = []

    for batch in loader:
        text_emb     = batch["text_embedding"].to(device)
        acoustic_emb = batch["acoustic_embedding"].to(device)
        e_labels     = batch["emotion_label"].to(device)
        s_labels     = batch["sentiment_label"].to(device)

        sentiment_logits, emotion_logits = fusion(text_emb, acoustic_emb)

        e_probs = F.softmax(emotion_logits, dim=-1).cpu().numpy()
        s_probs = F.softmax(sentiment_logits, dim=-1).cpu().numpy()

        emotion_preds.extend(emotion_logits.argmax(dim=-1).cpu().tolist())
        emotion_labels.extend(e_labels.cpu().tolist())
        emotion_probs_l.append(e_probs)

        sentiment_preds.extend(sentiment_logits.argmax(dim=-1).cpu().tolist())
        sentiment_labels.extend(s_labels.cpu().tolist())
        sentiment_probs_l.append(s_probs)

    return (
        emotion_preds,
        emotion_labels,
        np.concatenate(emotion_probs_l, axis=0),
        sentiment_preds,
        sentiment_labels,
        np.concatenate(sentiment_probs_l, axis=0),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_all_metrics(
    preds: List[int],
    labels: List[int],
    probs: np.ndarray,
    class_names: List[str],
) -> Dict:
    """Compute the full suite of classification metrics.

    Args:
        preds: Predicted class indices.
        labels: Ground-truth class indices.
        probs: Softmax probabilities of shape (N, num_classes).
        class_names: List of class name strings.

    Returns:
        Dictionary with accuracy, weighted precision/recall/F1, macro F1,
        AUC, per_class_f1, and full classification report.
    """
    num_classes = len(class_names)

    accuracy           = accuracy_score(labels, preds)
    weighted_precision = precision_score(labels, preds, average="weighted", zero_division=0)
    weighted_recall    = recall_score(labels, preds, average="weighted", zero_division=0)
    weighted_f1        = f1_score(labels, preds, average="weighted", zero_division=0)
    macro_f1           = f1_score(labels, preds, average="macro",    zero_division=0)

    try:
        auc = roc_auc_score(
            labels, probs, multi_class="ovr", average="macro",
            labels=list(range(num_classes)),
        )
    except ValueError:
        auc = float("nan")

    report = classification_report(
        labels, preds, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    per_class_f1 = {
        name: report[name]["f1-score"]
        for name in class_names if name in report
    }

    return {
        "accuracy":           accuracy,
        "weighted_precision": weighted_precision,
        "weighted_recall":    weighted_recall,
        "weighted_f1":        weighted_f1,
        "macro_f1":           macro_f1,
        "auc":                auc,
        "per_class_f1":       per_class_f1,
        "report":             report,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    preds: List[int],
    labels: List[int],
    class_names: List[str],
    title: str,
    save_path: Path,
) -> None:
    """Plot and save a row-normalised confusion matrix.

    Args:
        preds: Predicted class indices.
        labels: Ground-truth class indices.
        class_names: Class name strings for axis labels.
        title: Plot title.
        save_path: Path to save the PNG.
    """
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(
        figsize=(max(6, len(class_names)), max(5, len(class_names)))
    )
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            colour = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(
                j, i, f"{cm_norm[i, j]:.2f}",
                ha="center", va="center", color=colour, fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def log_full_metrics(
    metrics: Dict,
    task: str,
    logger: logging.Logger,
) -> None:
    """Log the full metrics suite for a task.

    Args:
        metrics: Output of compute_all_metrics.
        task: 'emotion' or 'sentiment'.
        logger: Logger instance.
    """
    logger.info("TEST | %s | Accuracy:           %.4f", task, metrics["accuracy"])
    logger.info("TEST | %s | Weighted Precision: %.4f", task, metrics["weighted_precision"])
    logger.info("TEST | %s | Weighted Recall:    %.4f", task, metrics["weighted_recall"])
    logger.info("TEST | %s | Weighted F1:        %.4f", task, metrics["weighted_f1"])
    logger.info("TEST | %s | Macro F1:           %.4f", task, metrics["macro_f1"])
    logger.info("TEST | %s | AUC (OvR macro):    %.4f", task, metrics["auc"])
    logger.info("TEST | %s | Per-class F1:", task)
    for name, score in metrics["per_class_f1"].items():
        logger.info("  %-12s F1: %.4f", name, score)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Phase 2 fusion model on MELD test set."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--fusion_type",
        type=str,
        default="concat",
        choices=list(_FUSION_CLASSES.keys()),
    )
    parser.add_argument(
        "--phase2_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to Phase 2 best_model.pt. "
            "Defaults to checkpoints/mini/fusion/{fusion_type}/best_model.pt"
        ),
    )
    args = parser.parse_args()

    config      = load_config(args.config)
    fusion_type = args.fusion_type
    set_seed(config["data"]["seed"])
    device = get_device()

    log_dir = config["training"]["log_dir"]
    logger  = setup_logging(log_dir, f"evaluate_{fusion_type}")
    logger.info(
        "Config: %s | Fusion: %s | Device: %s",
        args.config, fusion_type, device,
    )

    output_dir = Path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Test dataset ----
    test_ds = EvalDataset(
        meld_root            = config["data"]["meld_root"],
        text_embeddings_path = config["data"]["text_embeddings_path"],
        embeddings_path      = config["data"]["embeddings_path"],
        split                = "test",
        text_dim             = config["model"]["text_dim"],
        acoustic_dim         = config["model"]["acoustic_dim"],
    )
    logger.info("Test set: %d samples", len(test_ds))

    test_loader = DataLoader(
        test_ds,
        batch_size  = config["training"]["batch_size"],
        shuffle     = False,
        num_workers = config["data"]["num_workers"],
        pin_memory  = True,
        drop_last   = False,
    )

    # ---- Load fusion checkpoint ----
    ckpt_path = args.phase2_checkpoint or (
        Path(config["training"]["checkpoint_dir"])
        / "fusion" / fusion_type / "best_model.pt"
    )
    if not Path(str(ckpt_path)).exists():
        raise FileNotFoundError(f"Phase 2 checkpoint not found: {ckpt_path}")

    state = torch.load(str(ckpt_path), map_location="cpu")

    FusionClass = _FUSION_CLASSES[fusion_type]
    fusion = FusionClass(
        acoustic_dim          = config["model"]["acoustic_dim"],
        text_dim              = config["model"]["text_dim"],
        hidden_dim            = config["model"]["fusion_hidden"],
        num_sentiment_classes = config["model"]["num_sentiment_classes"],
        num_emotion_classes   = config["model"]["num_classes"],
        dropout_prob          = config["model"]["dropout"],
    )
    fusion.load_state_dict(state["fusion_state_dict"])
    fusion = fusion.to(device)
    logger.info("Loaded fusion model: %s from %s", FusionClass.__name__, ckpt_path)

    # ---- Inference ----
    logger.info("Running inference on test set...")
    (
        emotion_preds, emotion_labels, emotion_probs,
        sentiment_preds, sentiment_labels, sentiment_probs,
    ) = run_inference(fusion, test_loader, device)

    # ---- Metrics ----
    emotion_metrics   = compute_all_metrics(
        emotion_preds, emotion_labels, emotion_probs, EMOTION_NAMES
    )
    sentiment_metrics = compute_all_metrics(
        sentiment_preds, sentiment_labels, sentiment_probs, SENTIMENT_NAMES
    )

    log_full_metrics(emotion_metrics,   "emotion",   logger)
    log_full_metrics(sentiment_metrics, "sentiment", logger)

    # ---- Confusion matrices ----
    plot_confusion_matrix(
        emotion_preds, emotion_labels, EMOTION_NAMES,
        title     = f"Emotion — {FusionClass.__name__} (test)",
        save_path = output_dir / f"confusion_emotion_{fusion_type}.png",
    )
    plot_confusion_matrix(
        sentiment_preds, sentiment_labels, SENTIMENT_NAMES,
        title     = f"Sentiment — {FusionClass.__name__} (test)",
        save_path = output_dir / f"confusion_sentiment_{fusion_type}.png",
    )

    # ---- Save JSON results ----
    results = {
        "config":       args.config,
        "fusion_type":  fusion_type,
        "checkpoint":   str(ckpt_path),
        "test_samples": len(test_ds),
        "emotion":      {k: v for k, v in emotion_metrics.items() if k != "report"},
        "sentiment":    {k: v for k, v in sentiment_metrics.items() if k != "report"},
    }
    results_path = output_dir / f"test_results_{fusion_type}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
