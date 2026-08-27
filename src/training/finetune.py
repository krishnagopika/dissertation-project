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
    """Dataset serving either ASR transcripts or MELD gold text, with labels.

    BUG HISTORY: this class accepted `transcripts_path` and then never used it,
    reading `row["utterance"]` -- MELD's GOLD text -- regardless. Every
    text-only and fusion result produced before 2026-08-26 therefore trained on
    gold text while the class name and docstring claimed ASR transcripts. That
    silently made WER-filtering the training data a no-op by construction: a
    branch that never sees a transcription error cannot benefit from removing
    utterances with transcription errors. See POSTMORTEMS.md PM-010.

    `text_source` now makes the choice explicit and required at the call site.

    Args:
        meld_root: Path to MELD root directory (contains CSV files).
        transcripts_path: Directory with {split}_transcripts.json.
        split: One of 'train', 'dev', 'test'.
        tokenizer: HuggingFace tokenizer for XLM-RoBERTa.
        max_length: Maximum tokenised sequence length.
        paraphrases_path: Optional augmentation file (train only).
        filtered_keys_path: Optional keep-list; drops rows not in its "keys".
        text_source: "asr" reads {split}_transcripts.json -- the realistic
            end-to-end setting. "gold" reads the MELD CSV, which is the
            historical behaviour and assumes perfect transcription.
    """

    def __init__(
        self,
        meld_root: str,
        transcripts_path: str,
        split: str,
        tokenizer,
        max_length: int = 128,
        paraphrases_path: Optional[str] = None,
        filtered_keys_path: Optional[str] = None,
        text_source: str = "asr",
    ) -> None:
        assert split in ("train", "dev", "test"), (
            f"split must be train/dev/test, got {split}"
        )
        assert text_source in ("asr", "gold"), (
            f"text_source must be 'asr' or 'gold', got {text_source!r}"
        )
        self.text_source = text_source
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

        # Optional VAD+WER keep-list from apply_filter.py — drops MELD rows
        # whose utterance key isn't in the policy's keys list.
        keep_set: Optional[set] = None
        self.num_before_filter = len(df)
        if filtered_keys_path is not None:
            p = Path(filtered_keys_path)
            if not p.exists():
                raise FileNotFoundError(
                    f"filtered_keys_path is set but file not found at {p}. "
                    "Run preprocessing/compute_filter_metadata.py then "
                    "preprocessing/apply_filter.py first."
                )
            with open(p, "r", encoding="utf-8") as f:
                keep_set = set(json.load(f)["keys"])

        # Load ASR transcripts when asked for. Previously this file was never
        # opened, which is the bug described in the class docstring.
        transcripts: Dict[str, str] = {}
        if text_source == "asr":
            tp = Path(transcripts_path) / f"{split}_transcripts.json"
            if not tp.exists():
                raise FileNotFoundError(
                    f"text_source='asr' but no transcripts at {tp}. "
                    "Run preprocessing/transcribe_all.py first."
                )
            with open(tp, "r", encoding="utf-8") as f:
                transcripts = json.load(f)

        self.n_empty_text = 0
        self.samples: List[Tuple[str, int, int]] = []
        for _, row in df.iterrows():
            key = (f"dia{int(row['dialogue_id'])}"
                   f"_utt{int(row['utterance_id'])}")
            if keep_set is not None and key not in keep_set:
                continue
            if text_source == "asr":
                text = str(transcripts.get(key, "") or "").strip()
            else:
                text = str(row.get("utterance", "") or "").strip()
            if not text:
                # Kept, not dropped: an empty transcript is a real outcome of
                # the ASR pipeline and the label is still valid. Counted so the
                # run log states how many the model saw.
                self.n_empty_text += 1
            emotion_idx = EMOTION2IDX.get(row["emotion"], 0)
            sentiment_idx = SENTIMENT2IDX.get(row["sentiment"], 1)
            self.samples.append((text, emotion_idx, sentiment_idx))

        self.num_original = len(self.samples)
        self.num_paraphrases = 0
        self.num_after_filter = len(self.samples)

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

def save_checkpoint(  # noqa: PLR0913
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metric: float,
    config: dict,
    checkpoint_dir: str,
    is_best: bool = False,
    scheduler=None,
    best_metric_so_far: float = 0.0,
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
        scheduler: LR scheduler, so a deliberate resume restores the schedule.
        best_metric_so_far: Best dev metric seen across ALL epochs so far --
            distinct from `metric`, which is this epoch's score.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        # Saved so a deliberate --resume can restore the LR schedule. Without
        # it a resumed run rebuilds the scheduler from step 0, re-running
        # warmup mid-training while the optimizer carries on from where it was.
        "scheduler_state_dict": (scheduler.state_dict()
                                 if scheduler is not None else None),
        "metric": metric,
        # best_metric_so_far, NOT this epoch's metric. `metric` above is what
        # THIS epoch scored; on resume you need the best seen so far, or a
        # later worse epoch is mistaken for an improvement and overwrites
        # best_model.pt with an inferior model.
        "best_metric_so_far": best_metric_so_far,
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
    parser.add_argument(
        "--loss", type=str, default="weighted",
        choices=["plain", "weighted", "focal"],
        help=("'plain' = unweighted CE (baseline control); 'weighted' = CE with "
              "inverse-frequency class alpha (previous default); 'focal' = "
              "FocalLoss(gamma from config) with the same alpha."),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help=("Resume from the latest checkpoint. OFF by default: resume "
              "restores best_metric from the LATEST epoch rather than the best "
              "one, and rebuilds the LR scheduler from scratch. Both silently "
              "change results, so never use it for a controlled comparison."),
    )
    parser.add_argument(
        "--patience", type=int, default=3,
        help=("Stop when dev weighted F1 has not improved for this many "
              "epochs. 0 disables early stopping."),
    )
    parser.add_argument(
        "--text_source", type=str, default="asr", choices=["asr", "gold"],
        help=("'asr' trains on {split}_transcripts.json (realistic end-to-end); "
              "'gold' trains on the MELD CSV utterance column (assumes perfect "
              "transcription -- the historical, undocumented behaviour)."),
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

    filt_cfg = config.get("filtering", {})
    use_filter = bool(filt_cfg.get("enabled", False))
    train_filter_keys: Optional[str] = None
    dev_filter_keys: Optional[str] = None
    if use_filter:
        keys_paths = filt_cfg.get("keys_paths", {})
        train_filter_keys = keys_paths.get("train")
        if train_filter_keys is None:
            raise KeyError(
                "filtering.enabled=true but filtering.keys_paths.train "
                "is not set in the config."
            )
        # filter_dev decides whether the SELECTION set is filtered too.
        #
        # false (default): train on filtered data, select on the FULL dev set.
        #   Question answered: "does removing bad ASR examples from training
        #   improve performance on the general ASR distribution?" -- the filter
        #   is the treatment and dev is a fixed yardstick.
        #
        # true: train and select on filtered data.
        #   Question answered: "does the model do better on the cleaner
        #   distribution?" -- but the filter now changes both the training data
        #   AND the early-stopping criterion, so the two effects cannot be
        #   separated, and dev is no longer comparable across runs.
        filter_dev = bool(filt_cfg.get("filter_dev", False))
        dev_filter_keys = keys_paths.get("dev") if filter_dev else None
        logger.info(
            "Filtering: train=FILTERED, dev=%s",
            "FILTERED" if filter_dev else "FULL (unfiltered — fixed yardstick)",
        )

    train_ds = TranscriptDataset(
        meld_root=config["data"]["meld_root"],
        transcripts_path=config["data"]["transcripts_path"],
        split="train",
        tokenizer=tokenizer,
        max_length=max_length,
        paraphrases_path=paraphrases_path,
        filtered_keys_path=train_filter_keys,
        text_source=args.text_source,
    )
    dev_ds = TranscriptDataset(
        meld_root=config["data"]["meld_root"],
        transcripts_path=config["data"]["transcripts_path"],
        split="dev",
        tokenizer=tokenizer,
        max_length=max_length,
        filtered_keys_path=dev_filter_keys,
        text_source=args.text_source,
    )
    if use_filter:
        logger.info(
            "Filter enabled | train: %d/%d kept | dev: %d/%d kept",
            train_ds.num_after_filter, train_ds.num_before_filter,
            dev_ds.num_after_filter, dev_ds.num_before_filter,
        )
    logger.info(
        "Train dataset: %d originals + %d paraphrases = %d total",
        train_ds.num_original, train_ds.num_paraphrases, len(train_ds),
    )
    logger.info("TEXT SOURCE: %s | empty text: train %d, dev %d",
                args.text_source, train_ds.n_empty_text, dev_ds.n_empty_text)

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

    # The A/B/C loss ablation isolates alpha (A->B) and gamma (B->C). A
    # WeightedRandomSampler re-balances the batches themselves, and the code
    # below then drops alpha to avoid double-correcting -- so with the sampler
    # on, B and C would BOTH silently lose their alpha and the ladder would
    # collapse to (CE+sampler, CE+sampler, focal+sampler). Refuse rather than
    # rely on the config being right.
    if use_sampler and args.loss in ("weighted", "focal"):
        raise ValueError(
            f"use_weighted_sampler=true is incompatible with --loss {args.loss}. "
            "The sampler suppresses class alpha, so 'weighted' would be "
            "identical to 'plain' and 'focal' would lose its alpha term -- the "
            "A/B/C ablation would silently measure nothing. Set "
            "training.use_weighted_sampler: false."
        )

    # `plain` deliberately passes weight=None: an UNWEIGHTED baseline. Without
    # it there is no control showing what class weighting actually buys, since
    # weights were previously applied unconditionally.
    if args.loss == "plain":
        emotion_weights = None
        sentiment_weights = None
        logger.info("loss=plain — unweighted CrossEntropy (no class alpha)")
    elif use_sampler:
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

    # CLI wins over config so one config file serves all three loss variants.
    use_focal = (args.loss == "focal")
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
        logger.info("Using CrossEntropyLoss on both heads (loss=%s)", args.loss)

    # ---- Resume (OFF by default) --------------------------------------
    # Automatic resume is unsafe here, for two reasons that are easy to miss:
    #
    #  1. best_metric is restored from the LATEST checkpoint, not the best one.
    #     Epochs 0.60 / 0.65 / 0.63 leave checkpoint_epoch003 (0.63) on disk
    #     alongside best_model.pt (0.65). Resuming sets best_metric=0.63, so a
    #     later 0.64 epoch counts as "best" and OVERWRITES the genuine 0.65.
    #     The run then silently reports a worse model than it actually found.
    #
    #  2. The scheduler is not part of the checkpoint -- only model and
    #     optimizer state are saved. A resumed run restores the optimizer but
    #     builds a FRESH scheduler, so the learning-rate schedule no longer
    #     matches the step count. Warmup re-runs mid-training.
    #
    # For a controlled ablation neither is acceptable: both change the result
    # without any error. Resume must be asked for explicitly.
    start_epoch = 0
    best_metric = 0.0
    ckpt_dir = Path(checkpoint_dir)
    if args.resume:
        if ckpt_dir.exists():
            checkpoints = sorted(ckpt_dir.glob("checkpoint_*.pt"))
            if checkpoints:
                latest = checkpoints[-1]
                logger.warning(
                    "--resume: loading %s. best_metric is taken from THIS "
                    "checkpoint, not from best_model.pt, and the LR schedule "
                    "restarts. Do not use for controlled comparisons.", latest,
                )
                m = model.module if hasattr(model, "module") else model
                start_epoch, best_metric = load_checkpoint(
                    str(latest), m, optimizer
                )
    elif ckpt_dir.exists() and any(ckpt_dir.glob("checkpoint_*.pt")):
        logger.info(
            "Found checkpoints in %s but --resume was not passed — starting "
            "from base %s as intended.", ckpt_dir, config["model"]["xlmr_id"],
        )

    # ---- Training loop ----
    total_epochs = config["training"]["epochs_phase1"]

    # Guard against the silent no-op: a stale checkpoint from a previous run
    # can push start_epoch past total_epochs, making range(...) empty and the
    # job "complete" instantly without training anything.
    if start_epoch >= total_epochs:
        logger.error(
            "start_epoch=%d ≥ epochs_phase1=%d — the training loop would "
            "run zero iterations. This usually means a stale checkpoint from "
            "a previous run is still in %s. Move it aside (or delete it) and "
            "resubmit.",
            start_epoch, total_epochs, ckpt_dir,
        )
        sys.exit(2)
    train_losses: List[float] = []
    val_losses: List[float] = []

    # TensorBoard. One directory per run, derived from log_dir, so the three
    # runs appear as separate curves rather than overwriting each other --
    # the same per-run-path discipline PM-001 is about.
    from torch.utils.tensorboard import SummaryWriter
    tb_dir = Path(config["training"]["log_dir"]) / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    logger.info("TensorBoard: %s", tb_dir)
    writer.add_text("run/config", args.config, 0)
    writer.add_text("run/text_source", args.text_source, 0)
    writer.add_text("run/loss", args.loss, 0)
    writer.add_text("run/patience", str(args.patience), 0)
    writer.add_text("run/train_size", str(len(train_ds)), 0)

    epochs_since_best = 0

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

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/dev", val_loss, epoch)
        writer.add_scalar("emotion/dev_weighted_f1", wf1, epoch)
        writer.add_scalar("emotion/dev_macro_f1",
                          emotion_metrics.get("macro_f1", 0.0), epoch)
        writer.add_scalar("sentiment/dev_weighted_f1",
                          sentiment_metrics.get("weighted_f1", 0.0), epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        # Per-class F1: the minority classes are where the imbalance bites, and
        # a weighted average hides them (dev has 22 disgust, 40 fear).
        for cls, f1 in (emotion_metrics.get("per_class_f1") or {}).items():
            writer.add_scalar(f"emotion_f1_per_class/{cls}", f1, epoch)

        is_best = wf1 > best_metric
        if is_best:
            best_metric = wf1
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        m = model.module if hasattr(model, "module") else model
        save_checkpoint(
            m, optimizer, epoch, wf1, config,
            checkpoint_dir, is_best=is_best,
            scheduler=scheduler, best_metric_so_far=best_metric,
        )

        # Early stopping on dev weighted F1 -- the selection metric, not loss.
        # best_model.pt already holds the best epoch, so stopping early costs
        # nothing but wasted epochs. patience<=0 disables.
        writer.add_scalar("train/epochs_since_best", epochs_since_best, epoch)
        if args.patience > 0 and epochs_since_best >= args.patience:
            logger.info(
                "EARLY STOP at epoch %d — dev weighted F1 has not improved on "
                "%.4f for %d epoch(s) (patience=%d).",
                epoch + 1, best_metric, epochs_since_best, args.patience,
            )
            break

    writer.add_scalar("emotion/best_dev_weighted_f1", best_metric, 0)
    writer.flush()
    writer.close()
    # Completion marker. best_model.pt appears at the FIRST improving epoch, so
    # its presence proves a run STARTED, not that it finished. A job killed at
    # epoch 2 of 10 leaves one behind, and any "skip if checkpoint exists" logic
    # would then treat a half-trained model as a finished result. This file is
    # written only here, after the loop exits normally.
    import json as _json
    (Path(checkpoint_dir) / "TRAINING_COMPLETE.json").write_text(_json.dumps({
        "run": Path(checkpoint_dir).name,
        "config": args.config,
        "text_source": args.text_source,
        "loss": args.loss,
        "patience": args.patience,
        "epochs_run": epoch + 1,
        "epochs_configured": total_epochs,
        "early_stopped": (epoch + 1) < total_epochs,
        "best_dev_weighted_f1": best_metric,
        "train_size": len(train_ds),
        "dev_size": len(dev_ds),
    }, indent=2), encoding="utf-8")
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
