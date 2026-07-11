"""
train_classical.py — Variant A: Classical Classifiers on Raw Fused Features
============================================================================
Trains SVM and XGBoost classifiers directly on the concatenation of:
  - Pre-cached acoustic embeddings from Voxtral Whisper encoder (1280-dim)
  - XLM-RoBERTa [CLS] text representations (768-dim)

No neural fusion layer is used. This serves as a strong classical baseline
to compare against the neural fusion models (Phase 2).

Inputs (all pre-cached):
  data/meld_embeddings/{split}_embeddings.pt
  data/meld_transcripts/{split}_transcripts.json
  checkpoints/mini/best_model.pt  (Phase 1 XLM-R checkpoint — encoder frozen)

Output:
  results/mini/classical_{model}_{task}.json  — WF1 + per-class F1

Usage
-----
  python3.12 src/training/train_classical.py --config src/configs/mini.yaml
  python3.12 src/training/train_classical.py --config src/configs/mini.yaml \\
      --model svm        # svm | xgboost | both (default: both)
      --phase1_checkpoint checkpoints/mini/best_model.pt

Slurm: see src/scripts/train_classical.sbatch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

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
# Label mappings
# ---------------------------------------------------------------------------

EMOTION2IDX: Dict[str, int] = {name: i for i, name in enumerate(EMOTION_NAMES)}
SENTIMENT2IDX: Dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}

_SPLIT_CSV: Dict[str, str] = {
    "train": "train_sent_emo.csv",
    "dev":   "dev_sent_emo.csv",
    "test":  "test_sent_emo.csv",
}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    split: str,
    config: dict,
    tokenizer,
    xlmr: XLMRobertaClassifier,
    device: torch.device,
    logger,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract concatenated acoustic + text features for a split.

    Args:
        split: One of 'train', 'dev', 'test'.
        config: Loaded config dictionary.
        tokenizer: XLM-RoBERTa tokenizer.
        xlmr: Frozen XLM-RoBERTa classifier (encoder only used).
        device: Target device.
        logger: Logger instance.

    Returns:
        Tuple of (features, emotion_labels, sentiment_labels) as numpy arrays.
        features shape: (N, acoustic_dim + text_dim)
    """
    meld_root       = Path(config["data"]["meld_root"])
    transcripts_dir = Path(config["data"]["transcripts_path"])
    embeddings_dir  = Path(config["data"]["embeddings_path"])
    acoustic_dim    = config["model"]["acoustic_dim"]
    text_dim        = config["model"]["text_dim"]
    max_length      = config["data"]["max_text_length"]
    batch_size      = config["training"]["batch_size"]

    # ---- Load CSV ----
    csv_path = meld_root / _SPLIT_CSV[split]
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    df = df.dropna(subset=["emotion", "sentiment"]).reset_index(drop=True)
    df["emotion"]   = df["emotion"].str.strip().str.lower()
    df["sentiment"] = df["sentiment"].str.strip().str.lower()

    # ---- Load transcripts ----
    transcript_file = transcripts_dir / f"{split}_transcripts.json"
    if not transcript_file.exists():
        raise FileNotFoundError(f"Transcripts not found: {transcript_file}")
    with open(transcript_file, "r", encoding="utf-8") as f:
        transcripts: Dict[str, str] = json.load(f)

    # ---- Load acoustic embeddings ----
    embedding_file = embeddings_dir / f"{split}_embeddings.pt"
    if not embedding_file.exists():
        raise FileNotFoundError(f"Embeddings not found: {embedding_file}")
    acoustic_embeddings: Dict[str, torch.Tensor] = torch.load(
        str(embedding_file), map_location="cpu"
    )

    # ---- Build sample lists ----
    texts:           List[str]   = []
    acoustics:       List[torch.Tensor] = []
    emotion_labels:  List[int]   = []
    sentiment_labels: List[int]  = []

    for _, row in df.iterrows():
        key = f"dia{int(row['dialogue_id'])}_utt{int(row['utterance_id'])}"
        texts.append(transcripts.get(key, ""))
        acoustics.append(
            acoustic_embeddings.get(key, torch.zeros(acoustic_dim))
        )
        emotion_labels.append(EMOTION2IDX.get(row["emotion"], 0))
        sentiment_labels.append(SENTIMENT2IDX.get(row["sentiment"], 1))

    # ---- Extract XLM-R text representations in batches ----
    logger.info("Extracting XLM-R text representations for '%s' (%d samples)...", split, len(texts))
    xlmr.eval()
    text_reprs: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: i + batch_size]
        encoding = tokenizer(
            batch_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            text_repr = xlmr.get_text_representation(input_ids, attention_mask)
        text_reprs.append(text_repr.cpu().numpy())

        if (i // batch_size + 1) % 10 == 0:
            logger.info("  %d / %d batches done", i // batch_size + 1, (len(texts) + batch_size - 1) // batch_size)

    text_matrix     = np.vstack(text_reprs)                          # (N, text_dim)
    acoustic_matrix = torch.stack(acoustics).numpy().astype(np.float32)  # (N, acoustic_dim)

    # ---- Concatenate ----
    features = np.concatenate([text_matrix, acoustic_matrix], axis=1)  # (N, text_dim + acoustic_dim)
    logger.info(
        "Feature matrix for '%s': shape=%s, dtype=%s",
        split, features.shape, features.dtype,
    )

    return (
        features,
        np.array(emotion_labels,   dtype=np.int32),
        np.array(sentiment_labels, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Train + evaluate one classifier
# ---------------------------------------------------------------------------

def train_and_evaluate(
    clf,
    clf_name: str,
    task: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    output_dir: Path,
    logger,
    tag: str = "",
) -> Dict:
    """Fit classifier and evaluate on dev set.

    Args:
        clf: Sklearn-compatible classifier instance.
        clf_name: Name string for logging/saving (e.g. 'svm', 'xgboost').
        task: 'emotion' or 'sentiment'.
        X_train: Training features.
        y_train: Training labels.
        X_dev: Dev features.
        y_dev: Dev labels.
        output_dir: Directory to save results JSON.
        logger: Logger instance.

    Returns:
        Metrics dictionary.
    """
    logger.info("Training %s for %s (%d samples)...", clf_name, task, len(X_train))
    clf.fit(X_train, y_train)

    preds = clf.predict(X_dev)

    if task == "emotion":
        metrics = compute_emotion_metrics(preds.tolist(), y_dev.tolist())
    else:
        metrics = compute_sentiment_metrics(preds.tolist(), y_dev.tolist())

    log_metrics(metrics, "dev", task, logger)
    logger.info(
        "%s | %s | dev WF1: %.4f",
        clf_name.upper(), task, metrics["weighted_f1"],
    )

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"classical_{clf_name}_{task}{tag}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Results saved to %s", out_path)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train classical classifiers (SVM/XGBoost) on raw fused features."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config yaml (mini.yaml or small.yaml)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="both",
        choices=["svm", "xgboost", "both"],
        help="Which classifier(s) to train (default: both)",
    )
    parser.add_argument(
        "--phase1_checkpoint",
        type=str,
        default=None,
        help="Path to Phase 1 best_model.pt (defaults to config checkpoint_dir/best_model.pt)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help=(
            "Suffix added to output filenames "
            "(classical_{model}_{task}<tag>.json). Use to keep ablation runs "
            "distinct."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["data"]["seed"])
    device = get_device()

    log_dir = config["training"]["log_dir"]
    logger  = setup_logging(log_dir, "train_classical")
    logger.info("Config: %s | Model: %s | Device: %s", args.config, args.model, device)

    output_dir = Path(config["evaluation"]["output_dir"])

    # ---- Load XLM-RoBERTa (Phase 1 checkpoint, encoder frozen) ----
    xlmr_id   = config["model"]["xlmr_id"]
    tokenizer = AutoTokenizer.from_pretrained(xlmr_id)

    xlmr = XLMRobertaClassifier(
        model_name_or_path    = xlmr_id,
        num_sentiment_classes = config["model"]["num_sentiment_classes"],
        num_emotion_classes   = config["model"]["num_classes"],
        dropout_prob          = config["model"]["dropout"],
    )

    phase1_ckpt = args.phase1_checkpoint or (
        Path(config["training"]["checkpoint_dir"]) / "best_model.pt"
    )
    if Path(str(phase1_ckpt)).exists():
        state = torch.load(str(phase1_ckpt), map_location="cpu")
        xlmr.load_state_dict(state["model_state_dict"])
        logger.info("Loaded Phase 1 XLM-R checkpoint: %s", phase1_ckpt)
    else:
        logger.warning(
            "Phase 1 checkpoint not found at %s — using pretrained weights.", phase1_ckpt
        )

    for param in xlmr.parameters():
        param.requires_grad = False
    xlmr = xlmr.to(device)

    # ---- Extract features ----
    X_train, y_train_emo, y_train_sent = extract_features(
        "train", config, tokenizer, xlmr, device, logger
    )
    X_dev, y_dev_emo, y_dev_sent = extract_features(
        "dev", config, tokenizer, xlmr, device, logger
    )

    logger.info(
        "Features ready — train: %s, dev: %s",
        X_train.shape, X_dev.shape,
    )

    # ---- Build classifiers ----
    classifiers = {}

    if args.model in ("svm", "both"):
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        # SVM needs feature scaling — StandardScaler + RBF kernel SVC
        classifiers["svm"] = Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                class_weight="balanced",  # handles MELD class imbalance
                random_state=config["data"]["seed"],
            )),
        ])

    if args.model in ("xgboost", "both"):
        from xgboost import XGBClassifier

        classifiers["xgboost"] = XGBClassifier(
            n_estimators    = 500,
            max_depth       = 6,
            learning_rate   = 0.05,
            subsample       = 0.8,
            colsample_bytree= 0.8,
            use_label_encoder=False,
            eval_metric     = "mlogloss",
            random_state    = config["data"]["seed"],
            n_jobs          = -1,
        )

    # ---- Train and evaluate ----
    all_results = {}

    for clf_name, clf in classifiers.items():
        # Emotion
        import copy
        emotion_results = train_and_evaluate(
            clf        = copy.deepcopy(clf),
            clf_name   = clf_name,
            task       = "emotion",
            X_train    = X_train,
            y_train    = y_train_emo,
            X_dev      = X_dev,
            y_dev      = y_dev_emo,
            output_dir = output_dir,
            logger     = logger,
            tag        = args.tag,
        )

        # Sentiment
        sentiment_results = train_and_evaluate(
            clf        = copy.deepcopy(clf),
            clf_name   = clf_name,
            task       = "sentiment",
            X_train    = X_train,
            y_train    = y_train_sent,
            X_dev      = X_dev,
            y_dev      = y_dev_sent,
            output_dir = output_dir,
            logger     = logger,
            tag        = args.tag,
        )

        all_results[clf_name] = {
            "emotion":   emotion_results,
            "sentiment": sentiment_results,
        }

    # ---- Summary ----
    logger.info("=" * 60)
    logger.info("SUMMARY — Classical classifiers on raw fused features")
    logger.info("=" * 60)
    for clf_name, results in all_results.items():
        logger.info(
            "%s | Emotion WF1: %.4f | Sentiment WF1: %.4f",
            clf_name.upper(),
            results["emotion"]["weighted_f1"],
            results["sentiment"]["weighted_f1"],
        )


if __name__ == "__main__":
    main()
