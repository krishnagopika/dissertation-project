"""
finetune_encoder.py — Fine-tune Voxtral's Whisper encoder for MELD
===================================================================
Trains :class:`~src.models.voxtral_encoder_classifier.VoxtralEncoderClassifier`
end-to-end on MELD audio: the audio encoder (last N layers) + a learned
attention-pool + emotion/sentiment heads. The 3B LLM is never loaded.

Mel features are computed on the fly from the raw .mp4 audio (cheap), so no
embedding cache is needed — this is the point of the experiment: the encoder
representations adapt to emotion, which frozen cached embeddings cannot.

Imbalance handling (tests the original theory):
  * Focal loss (gamma from config) — down-weights easy/majority examples.
  * ``--balanced`` — train on an equal-samples-per-emotion-class subset.
  * No WeightedRandomSampler (deliberately, per the experiment design).

Overfitting control (635M encoder on limited data):
  * Validation loss logged every epoch next to train loss.
  * Early stopping on dev weighted-F1 with ``--patience``.
  * Train/val loss curve saved to results dir.

Usage
-----
  python3.12 src/training/finetune_encoder.py --config src/configs/mini.yaml
  python3.12 src/training/finetune_encoder.py --config src/configs/mini.yaml --balanced
  python3.12 src/training/finetune_encoder.py --config src/configs/mini.yaml \\
      --unfreeze_last_n 2 --patience 3 --max_samples 200    # quick smoke test

Slurm: see src/scripts/finetune_encoder.sbatch
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoProcessor, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import (
    EMOTION_NAMES,
    SENTIMENT_NAMES,
    compute_emotion_metrics,
    compute_sentiment_metrics,
    log_metrics,
)
from src.models.voxtral_encoder_classifier import VoxtralEncoderClassifier
from src.preprocessing.transcribe_all import (
    CSV_MAP,
    build_utterance_index,
    load_audio_mono_16k,
)
from src.training.finetune import FocalLoss
from src.utils import get_device, load_config, set_seed, setup_logging

EMOTION2IDX: Dict[str, int] = {name: i for i, name in enumerate(EMOTION_NAMES)}
SENTIMENT2IDX: Dict[str, int] = {name: i for i, name in enumerate(SENTIMENT_NAMES)}

TARGET_SR: int = 16_000


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MeldAudioDataset(Dataset):
    """Serves raw MELD waveforms + emotion/sentiment labels (on-the-fly decode).

    Args:
        meld_root: MELD root directory containing the split CSVs and audio.
        split: One of 'train', 'dev', 'test'.
        max_duration_sec: Max audio duration to decode per clip.
    """

    def __init__(self, meld_root: str, split: str, max_duration_sec: float) -> None:
        assert split in ("train", "dev", "test")
        self.split = split
        self.max_duration_sec = max_duration_sec

        meld_root = Path(meld_root)
        df = pd.read_csv(meld_root / CSV_MAP[split])
        df.columns = (
            df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
        )
        df = df.dropna(subset=["emotion", "sentiment"]).reset_index(drop=True)
        df["emotion"] = df["emotion"].str.strip().str.lower()
        df["sentiment"] = df["sentiment"].str.strip().str.lower()

        key_to_path = dict(build_utterance_index(meld_root, split))

        # samples: (key, audio_path, emotion_idx, sentiment_idx)
        self.samples: List[Tuple[str, Path, int, int]] = []
        for _, row in df.iterrows():
            key = f"dia{int(row['dialogue_id'])}_utt{int(row['utterance_id'])}"
            path = key_to_path.get(key)
            if path is None or not path.exists():
                continue
            self.samples.append((
                key,
                path,
                EMOTION2IDX.get(row["emotion"], 0),
                SENTIMENT2IDX.get(row["sentiment"], 1),
            ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        key, path, emotion, sentiment = self.samples[idx]
        try:
            waveform = load_audio_mono_16k(path, self.max_duration_sec)
        except Exception:
            waveform = torch.zeros(TARGET_SR, dtype=torch.float32)  # 1s of silence
        return {
            "waveform": waveform,
            "emotion": emotion,
            "sentiment": sentiment,
            "key": key,
        }


def collate(batch: List[Dict], feature_extractor) -> Tuple[Tensor, Tensor, Tensor]:
    """Batch waveforms into log-mel ``input_features`` + label tensors.

    Args:
        batch: List of dataset items.
        feature_extractor: Voxtral/Whisper feature extractor.

    Returns:
        Tuple of (input_features (B, n_mels, 3000), emotion (B,), sentiment (B,)).
    """
    audios = [b["waveform"].float().numpy() for b in batch]
    fe = feature_extractor(
        audios,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )
    emotions = torch.tensor([b["emotion"] for b in batch], dtype=torch.long)
    sentiments = torch.tensor([b["sentiment"] for b in batch], dtype=torch.long)
    return fe["input_features"], emotions, sentiments


def build_balanced_subset(dataset: MeldAudioDataset, seed: int) -> Subset:
    """Return an equal-samples-per-emotion-class Subset of the dataset.

    Caps every emotion class to the size of the smallest class (no replacement),
    so the training distribution is uniform over emotions.

    Args:
        dataset: Full training dataset.
        seed: RNG seed for reproducible sampling.

    Returns:
        A torch Subset with balanced emotion classes.
    """
    by_class: Dict[int, List[int]] = {}
    for i, (_, _, emo, _) in enumerate(dataset.samples):
        by_class.setdefault(emo, []).append(i)

    min_count = min(len(idxs) for idxs in by_class.values())
    rng = random.Random(seed)
    chosen: List[int] = []
    for emo, idxs in sorted(by_class.items()):
        picks = idxs[:]
        rng.shuffle(picks)
        chosen.extend(picks[:min_count])
    rng.shuffle(chosen)
    return Subset(dataset, chosen)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    emotion_criterion: nn.Module,
    sentiment_criterion: nn.Module,
    max_grad_norm: float,
) -> float:
    """Run one fine-tuning epoch. Returns mean training loss."""
    model.train()
    total_loss = 0.0
    for input_features, emotions, sentiments in loader:
        input_features = input_features.to(device)
        emotions = emotions.to(device)
        sentiments = sentiments.to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            sentiment_logits, emotion_logits = model(input_features)
            loss = (
                emotion_criterion(emotion_logits, emotions)
                + sentiment_criterion(sentiment_logits, sentiments)
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), max_grad_norm
        )
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    emotion_criterion: nn.Module,
    sentiment_criterion: nn.Module,
) -> Tuple[float, Dict, Dict]:
    """Evaluate on a split.

    Returns:
        Tuple of (mean_val_loss, emotion_metrics, sentiment_metrics).
    """
    model.eval()
    total_loss = 0.0
    e_preds: List[int] = []
    e_labels: List[int] = []
    s_preds: List[int] = []
    s_labels: List[int] = []
    for input_features, emotions, sentiments in loader:
        input_features = input_features.to(device)
        emotions_d = emotions.to(device)
        sentiments_d = sentiments.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            sentiment_logits, emotion_logits = model(input_features)
            loss = (
                emotion_criterion(emotion_logits, emotions_d)
                + sentiment_criterion(sentiment_logits, sentiments_d)
            )
        total_loss += loss.item()
        e_preds.extend(emotion_logits.float().argmax(dim=-1).cpu().tolist())
        e_labels.extend(emotions.tolist())
        s_preds.extend(sentiment_logits.float().argmax(dim=-1).cpu().tolist())
        s_labels.extend(sentiments.tolist())
    return (
        total_loss / max(len(loader), 1),
        compute_emotion_metrics(e_preds, e_labels),
        compute_sentiment_metrics(s_preds, s_labels),
    )


def save_loss_curve(
    train_losses: List[float],
    val_losses: List[float],
    output_dir: Path,
    tag: str,
) -> None:
    """Save a train/val loss curve PNG (for overfitting diagnosis)."""
    if not train_losses:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs_range = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs_range, train_losses, marker="o", label="Train Loss")
    ax.plot(epochs_range, val_losses, marker="o", label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Encoder fine-tune loss ({tag})")
    ax.legend()
    ax.grid(True)
    path = output_dir / f"encoder_finetune_loss_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune Voxtral's Whisper encoder for MELD emotion/sentiment."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--balanced", action="store_true",
                        help="Train on an equal-samples-per-emotion-class subset.")
    parser.add_argument("--unfreeze_last_n", type=int, default=2,
                        help="Number of final encoder layers to fine-tune (default 2).")
    parser.add_argument("--patience", type=int, default=3,
                        help="Early-stopping patience on dev weighted-F1 (default 3).")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Cap train/dev size — for smoke tests.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["data"]["seed"])
    device = get_device()

    logger = setup_logging(config["training"]["log_dir"], "finetune_encoder")
    logger.info("Config: %s | balanced=%s | unfreeze_last_n=%d | patience=%d | device=%s",
                args.config, args.balanced, args.unfreeze_last_n, args.patience, device)

    meld_root = config["data"]["meld_root"]
    max_dur = float(config["data"]["max_audio_duration"])
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["data"]["num_workers"])
    epochs = int(config["training"]["epochs_phase2"])
    seed = int(config["data"]["seed"])

    # ---- Data ----
    train_ds = MeldAudioDataset(meld_root, "train", max_dur)
    dev_ds = MeldAudioDataset(meld_root, "dev", max_dur)
    logger.info("Train: %d | Dev: %d (pre-subset)", len(train_ds), len(dev_ds))

    train_view: Dataset = train_ds
    if args.balanced:
        train_view = build_balanced_subset(train_ds, seed)
        logger.info("Balanced subset → %d train samples", len(train_view))
    if args.max_samples is not None:
        train_view = Subset(train_view, list(range(min(args.max_samples, len(train_view)))))
        dev_ds = Subset(dev_ds, list(range(min(args.max_samples, len(dev_ds)))))
        logger.info("--max_samples=%d (smoke test)", args.max_samples)

    processor = AutoProcessor.from_pretrained(config["model"]["voxtral_id"])
    collate_fn = partial(collate, feature_extractor=processor.feature_extractor)

    train_loader = DataLoader(
        train_view, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=False, collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False, collate_fn=collate_fn,
    )

    # ---- Model ----
    model = VoxtralEncoderClassifier(
        voxtral_id=config["model"]["voxtral_id"],
        num_emotion_classes=config["model"]["num_classes"],
        num_sentiment_classes=config["model"]["num_sentiment_classes"],
        acoustic_dim=config["model"]["acoustic_dim"],
        hidden_dim=config["model"]["fusion_hidden"],
        dropout_prob=config["model"]["dropout"],
        unfreeze_last_n=args.unfreeze_last_n,
    ).to(device)
    logger.info("Trainable parameters: %d", model.trainable_parameters())

    # ---- Optimiser: low LR for the encoder, higher for the new head/pool ----
    encoder_lr = float(config["training"].get("phase3_lr", 1e-5))
    head_lr = float(config["training"].get("phase2_lr", 1e-4))
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = (
        list(model.attn_pool.parameters())
        + list(model.proj.parameters())
        + list(model.sentiment_head.parameters())
        + list(model.emotion_head.parameters())
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": head_params, "lr": head_lr},
        ],
        weight_decay=float(config["training"]["weight_decay"]),
    )
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config["training"]["warmup_steps"]),
        num_training_steps=max(total_steps, 1),
    )

    # ---- Loss: focal, no sampler (per experiment design) ----
    gamma = float(config["training"].get("focal_gamma", 2.0))
    emotion_criterion = FocalLoss(gamma=gamma).to(device)
    sentiment_criterion = FocalLoss(gamma=gamma).to(device)
    logger.info("Loss: FocalLoss(gamma=%.1f), no weighted sampler", gamma)

    # ---- Train with early stopping on dev weighted-F1 ----
    checkpoint_dir = Path(config["training"]["checkpoint_dir"]) / "encoder_finetune"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(config["evaluation"]["output_dir"])
    max_grad_norm = float(config["training"].get("max_grad_norm", 1.0))
    tag = "balanced" if args.balanced else "full"

    best_wf1 = 0.0
    epochs_no_improve = 0
    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            emotion_criterion, sentiment_criterion, max_grad_norm,
        )
        val_loss, emotion_metrics, sentiment_metrics = evaluate(
            model, dev_loader, device, emotion_criterion, sentiment_criterion,
        )
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        wf1 = emotion_metrics["weighted_f1"]

        logger.info(
            "Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f | "
            "Emotion WF1: %.4f | Emotion Macro F1: %.4f",
            epoch + 1, epochs, train_loss, val_loss, wf1, emotion_metrics["macro_f1"],
        )
        log_metrics(emotion_metrics, "dev", "emotion", logger)
        log_metrics(sentiment_metrics, "dev", "sentiment", logger)

        if wf1 > best_wf1:
            best_wf1 = wf1
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "emotion_weighted_f1": wf1,
                    "emotion_macro_f1": emotion_metrics["macro_f1"],
                    "config": config,
                },
                checkpoint_dir / f"best_encoder_{tag}.pt",
            )
            logger.info("  ↑ new best (WF1 %.4f) saved", wf1)
        else:
            epochs_no_improve += 1
            logger.info("  no improvement (%d/%d) — best WF1 %.4f",
                        epochs_no_improve, args.patience, best_wf1)
            if epochs_no_improve >= args.patience:
                logger.info("Early stopping at epoch %d (patience %d reached).",
                            epoch + 1, args.patience)
                break

    save_loss_curve(train_losses, val_losses, results_dir, tag)
    logger.info("Encoder fine-tune complete. Best dev emotion WF1: %.4f", best_wf1)


if __name__ == "__main__":
    main()
