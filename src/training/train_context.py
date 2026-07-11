"""
train_context.py — Train bc-LSTM (dialogue-context) on cached embeddings
=========================================================================
Trains :class:`~src.models.context_lstm.BiLSTMContext` over MELD conversations.
Each training example is a **whole dialogue**: a sequence of per-utterance
feature vectors (cached text [768] + acoustic [1280] = 2048), ordered by
utterance index. A BiLSTM contextualises each utterance with its neighbours,
then per-utterance heads predict emotion + sentiment.

Tiny model on cached features — trains in minutes, no large model loaded.

Imbalance handling: focal loss (gamma from config), padding-masked. Early
stopping on dev weighted-F1.

Usage
-----
  python3.12 src/training/train_context.py --config src/configs/mini.yaml
  python3.12 src/training/train_context.py --config src/configs/mini.yaml \\
      --hidden_dim 256 --epochs 30 --patience 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import (
    EMOTION_NAMES,
    SENTIMENT_NAMES,
    compute_emotion_metrics,
    compute_sentiment_metrics,
    log_metrics,
)
from src.models.context_lstm import BiLSTMContext
from src.training.finetune import FocalLoss
from src.utils import get_device, load_config, set_seed, setup_logging

EMOTION2IDX: Dict[str, int] = {n: i for i, n in enumerate(EMOTION_NAMES)}
SENTIMENT2IDX: Dict[str, int] = {n: i for i, n in enumerate(SENTIMENT_NAMES)}
PAD_LABEL = -100

_SPLIT_CSV = {"train": "train_sent_emo.csv", "dev": "dev_sent_emo.csv", "test": "test_sent_emo.csv"}


# ---------------------------------------------------------------------------
# Dataset — one example per dialogue
# ---------------------------------------------------------------------------

class DialogueDataset(Dataset):
    """Serves whole dialogues as sequences of cached utterance features.

    Args:
        meld_root: MELD root (CSV files).
        text_embeddings_path: Dir with {split}_text_embeddings.pt.
        embeddings_path: Dir with {split}_embeddings.pt (acoustic).
        split: 'train' | 'dev' | 'test'.
        text_dim: Text embedding dim.
        acoustic_dim: Acoustic embedding dim.
    """

    def __init__(self, meld_root, text_embeddings_path, embeddings_path, split,
                 text_dim=768, acoustic_dim=1280) -> None:
        self.text_dim, self.acoustic_dim = text_dim, acoustic_dim

        text_emb: Dict[str, Tensor] = torch.load(
            str(Path(text_embeddings_path) / f"{split}_text_embeddings.pt"), map_location="cpu")
        acou_emb: Dict[str, Tensor] = torch.load(
            str(Path(embeddings_path) / f"{split}_embeddings.pt"), map_location="cpu")

        df = pd.read_csv(Path(meld_root) / _SPLIT_CSV[split])
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
        df = df.dropna(subset=["emotion", "sentiment"]).reset_index(drop=True)
        df["emotion"] = df["emotion"].str.strip().str.lower()
        df["sentiment"] = df["sentiment"].str.strip().str.lower()
        df["dialogue_id"] = df["dialogue_id"].astype(int)
        df["utterance_id"] = df["utterance_id"].astype(int)

        self.dialogues: List[Dict] = []
        for dia_id, grp in df.groupby("dialogue_id"):
            grp = grp.sort_values("utterance_id")
            feats, emos, sents = [], [], []
            for _, row in grp.iterrows():
                key = f"dia{dia_id}_utt{int(row['utterance_id'])}"
                t = text_emb.get(key, torch.zeros(text_dim))
                a = acou_emb.get(key, torch.zeros(acoustic_dim))
                feats.append(torch.cat([t.float(), a.float()]))
                emos.append(EMOTION2IDX.get(row["emotion"], 0))
                sents.append(SENTIMENT2IDX.get(row["sentiment"], 1))
            self.dialogues.append({
                "features": torch.stack(feats),                       # (T, 2048)
                "emotions": torch.tensor(emos, dtype=torch.long),     # (T,)
                "sentiments": torch.tensor(sents, dtype=torch.long),  # (T,)
            })

    def __len__(self) -> int:
        return len(self.dialogues)

    def __getitem__(self, idx: int) -> Dict:
        return self.dialogues[idx]


def collate(batch: List[Dict]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pad a batch of dialogues. Labels padded with PAD_LABEL (-100)."""
    lengths = torch.tensor([b["features"].size(0) for b in batch], dtype=torch.long)
    tmax = int(lengths.max())
    feat_dim = batch[0]["features"].size(1)
    B = len(batch)
    feats = torch.zeros(B, tmax, feat_dim)
    emos = torch.full((B, tmax), PAD_LABEL, dtype=torch.long)
    sents = torch.full((B, tmax), PAD_LABEL, dtype=torch.long)
    for i, b in enumerate(batch):
        t = b["features"].size(0)
        feats[i, :t] = b["features"]
        emos[i, :t] = b["emotions"]
        sents[i, :t] = b["sentiments"]
    return feats, emos, sents, lengths


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def masked_loss(logits: Tensor, labels: Tensor, criterion: nn.Module) -> Tensor:
    """Focal loss over non-padding utterances only."""
    C = logits.size(-1)
    flat_logits = logits.reshape(-1, C)
    flat_labels = labels.reshape(-1)
    valid = flat_labels != PAD_LABEL
    if valid.sum() == 0:
        return logits.sum() * 0.0
    return criterion(flat_logits[valid], flat_labels[valid])


def train_one_epoch(model, loader, optimizer, device, e_crit, s_crit, max_grad_norm) -> float:
    model.train()
    total = 0.0
    for feats, emos, sents, lengths in loader:
        feats, emos, sents = feats.to(device), emos.to(device), sents.to(device)
        optimizer.zero_grad()
        s_logits, e_logits = model(feats, lengths)
        loss = masked_loss(e_logits, emos, e_crit) + masked_loss(s_logits, sents, s_crit)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, e_crit, s_crit) -> Tuple[float, Dict, Dict]:
    model.eval()
    total = 0.0
    ep, el, spr, sl = [], [], [], []
    for feats, emos, sents, lengths in loader:
        feats, emos_d, sents_d = feats.to(device), emos.to(device), sents.to(device)
        s_logits, e_logits = model(feats, lengths)
        total += (masked_loss(e_logits, emos_d, e_crit)
                  + masked_loss(s_logits, sents_d, s_crit)).item()
        e_flat = e_logits.argmax(-1).reshape(-1).cpu()
        s_flat = s_logits.argmax(-1).reshape(-1).cpu()
        emo_flat = emos.reshape(-1); sen_flat = sents.reshape(-1)
        ve = emo_flat != PAD_LABEL
        ep.extend(e_flat[ve].tolist()); el.extend(emo_flat[ve].tolist())
        spr.extend(s_flat[ve].tolist()); sl.extend(sen_flat[ve].tolist())
    return (total / max(len(loader), 1),
            compute_emotion_metrics(ep, el),
            compute_sentiment_metrics(spr, sl))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train bc-LSTM dialogue-context model on cached embeddings.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16, help="dialogues per batch")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["data"]["seed"])
    device = get_device()
    logger = setup_logging(config["training"]["log_dir"], "train_context")
    logger.info("Config: %s | hidden=%d layers=%d | device=%s",
                args.config, args.hidden_dim, args.num_layers, device)

    text_dim = config["model"]["text_dim"]
    acoustic_dim = config["model"]["acoustic_dim"]

    def make_ds(split):
        return DialogueDataset(
            config["data"]["meld_root"], config["data"]["text_embeddings_path"],
            config["data"]["embeddings_path"], split, text_dim, acoustic_dim)

    train_ds, dev_ds = make_ds("train"), make_ds("dev")
    logger.info("Train dialogues: %d | Dev dialogues: %d", len(train_ds), len(dev_ds))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = BiLSTMContext(
        input_dim=text_dim + acoustic_dim, hidden_dim=args.hidden_dim,
        num_emotion_classes=config["model"]["num_classes"],
        num_sentiment_classes=config["model"]["num_sentiment_classes"],
        num_layers=args.num_layers, dropout=config["model"]["dropout"],
    ).to(device)
    logger.info("Trainable parameters: %d", model.trainable_parameters())

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["phase2_lr"]),
        weight_decay=float(config["training"]["weight_decay"]))
    gamma = float(config["training"].get("focal_gamma", 2.0))
    e_crit = FocalLoss(gamma=gamma).to(device)
    s_crit = FocalLoss(gamma=gamma).to(device)
    max_grad_norm = float(config["training"].get("max_grad_norm", 1.0))

    ckpt_dir = Path(config["training"]["checkpoint_dir"]) / "context_bclstm"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_wf1, no_improve = 0.0, 0

    for epoch in range(args.epochs):
        tr = train_one_epoch(model, train_loader, optimizer, device, e_crit, s_crit, max_grad_norm)
        vl, em, sm = evaluate(model, dev_loader, device, e_crit, s_crit)
        wf1 = em["weighted_f1"]
        logger.info("Epoch %d/%d | Train %.4f | Val %.4f | Emo WF1 %.4f | Emo Macro %.4f",
                    epoch + 1, args.epochs, tr, vl, wf1, em["macro_f1"])
        log_metrics(em, "dev", "emotion", logger)
        log_metrics(sm, "dev", "sentiment", logger)
        if wf1 > best_wf1:
            best_wf1, no_improve = wf1, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "emotion_weighted_f1": wf1, "emotion_macro_f1": em["macro_f1"],
                        "config": config}, ckpt_dir / "best_context.pt")
            logger.info("  ↑ new best (WF1 %.4f) saved", wf1)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info("Early stopping at epoch %d.", epoch + 1)
                break

    logger.info("bc-LSTM complete. Best dev emotion WF1: %.4f", best_wf1)

    # ---- Final TEST evaluation of the best checkpoint (comparable to other test numbers) ----
    best = torch.load(ckpt_dir / "best_context.pt", map_location=device)
    model.load_state_dict(best["model_state_dict"])
    test_loader = DataLoader(make_ds("test"), batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate)
    _, em_t, sm_t = evaluate(model, test_loader, device, e_crit, s_crit)
    logger.info("=== TEST (best checkpoint, epoch %d) ===", best["epoch"] + 1)
    log_metrics(em_t, "test", "emotion", logger)
    log_metrics(sm_t, "test", "sentiment", logger)


if __name__ == "__main__":
    main()
