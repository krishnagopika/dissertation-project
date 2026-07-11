"""
metrics.py — Evaluation Metrics
================================
Primary metric is weighted F1. Never use accuracy alone due to MELD class
imbalance (neutral dominates at ~47% of utterances).

All functions accept plain Python lists or numpy arrays.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

# Canonical label names — must match MELD_EMOTIONS in CLAUDE.md
EMOTION_NAMES: List[str] = [
    "neutral",   # 0
    "surprise",  # 1
    "fear",      # 2
    "sadness",   # 3
    "joy",       # 4
    "disgust",   # 5
    "anger",     # 6
]

SENTIMENT_NAMES: List[str] = [
    "negative",  # 0
    "neutral",   # 1
    "positive",  # 2
]


def compute_emotion_metrics(
    preds: List[int],
    labels: List[int],
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Compute weighted F1 and per-class F1 for the 7-class emotion task.

    Args:
        preds: List of predicted class indices.
        labels: List of ground-truth class indices.
        class_names: Optional list of class name strings. Defaults to
            EMOTION_NAMES.

    Returns:
        Dictionary with keys:
            weighted_f1    — scalar float
            per_class_f1   — dict mapping class name → f1 score
            macro_f1       — scalar float
            report         — full sklearn classification report dict
    """
    if class_names is None:
        class_names = EMOTION_NAMES

    # Fixed label set (0..K-1) so metrics are stable even when a class is
    # absent from preds/labels (e.g. small eval subsets) — otherwise sklearn
    # raises "Number of classes does not match size of target_names".
    label_ids = list(range(len(class_names)))

    weighted_f1 = f1_score(
        labels, preds, labels=label_ids, average="weighted", zero_division=0
    )
    macro_f1 = f1_score(
        labels, preds, labels=label_ids, average="macro", zero_division=0
    )

    report = classification_report(
        labels,
        preds,
        labels=label_ids,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class_f1 = {
        name: report[name]["f1-score"]
        for name in class_names
        if name in report
    }

    return {
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "report": report,
    }


def compute_sentiment_metrics(
    preds: List[int],
    labels: List[int],
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Compute weighted F1 and per-class F1 for the 3-class sentiment task.

    Args:
        preds: List of predicted class indices.
        labels: List of ground-truth class indices.
        class_names: Optional list of class name strings. Defaults to
            SENTIMENT_NAMES.

    Returns:
        Same structure as compute_emotion_metrics.
    """
    if class_names is None:
        class_names = SENTIMENT_NAMES

    # Fixed label set (0..K-1) so metrics are stable even when a class is
    # absent from preds/labels (e.g. small eval subsets) — otherwise sklearn
    # raises "Number of classes does not match size of target_names".
    label_ids = list(range(len(class_names)))

    weighted_f1 = f1_score(
        labels, preds, labels=label_ids, average="weighted", zero_division=0
    )
    macro_f1 = f1_score(
        labels, preds, labels=label_ids, average="macro", zero_division=0
    )

    report = classification_report(
        labels,
        preds,
        labels=label_ids,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    per_class_f1 = {
        name: report[name]["f1-score"]
        for name in class_names
        if name in report
    }

    return {
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "report": report,
    }


def compute_confusion_matrix(
    preds: List[int],
    labels: List[int],
    num_classes: int,
) -> np.ndarray:
    """Return confusion matrix as a (num_classes, num_classes) numpy array.

    Args:
        preds: Predicted class indices.
        labels: Ground-truth class indices.
        num_classes: Total number of classes.

    Returns:
        Confusion matrix where entry [i, j] is the count of samples with
        true label i predicted as j.
    """
    return confusion_matrix(labels, preds, labels=list(range(num_classes)))


def log_metrics(
    metrics: Dict,
    split: str,
    task: str,
    logger,
) -> None:
    """Log weighted F1 and per-class F1 via the provided logger.

    Args:
        metrics: Output of compute_emotion_metrics or compute_sentiment_metrics.
        split: One of 'train', 'dev', 'test'.
        task: 'emotion' or 'sentiment'.
        logger: A logging.Logger instance.
    """
    logger.info(
        "%s | %s | Weighted F1: %.4f | Macro F1: %.4f",
        split.upper(),
        task,
        metrics["weighted_f1"],
        metrics["macro_f1"],
    )
    for name, score in metrics["per_class_f1"].items():
        logger.info("  %-12s F1: %.4f", name, score)
