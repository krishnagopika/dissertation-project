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
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
                 text_dim=768, acoustic_dim=1280,
                 filtered_keys_path: Optional[str] = None) -> None:
        self.text_dim, self.acoustic_dim = text_dim, acoustic_dim

        text_emb: Dict[str, Tensor] = torch.load(
            str(Path(text_embeddings_path) / f"{split}_text_embeddings.pt"), map_location="cpu")
        acou_emb: Dict[str, Tensor] = torch.load(
            str(Path(embeddings_path) / f"{split}_embeddings.pt"), map_location="cpu")

        # Optional keep-list from VAD+WER filter. Filtered utterances stay in
        # the dialogue (so the BiLSTM keeps its context) but their labels are
        # masked to PAD_LABEL so they contribute to neither loss nor metrics.
        keep_set: Optional[Set[str]] = None
        if filtered_keys_path is not None:
            p = Path(filtered_keys_path)
            if not p.exists():
                raise FileNotFoundError(
                    f"filtered_keys_path set but file not found at {p}."
                )
            with open(p, "r", encoding="utf-8") as f:
                keep_set = set(json.load(f)["keys"])
        self.num_total_utts = 0
        self.num_kept_utts = 0

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
                self.num_total_utts += 1
                if keep_set is not None and key not in keep_set:
                    # Filtered-out — keep in dialogue for context, mask labels.
                    emos.append(PAD_LABEL)
                    sents.append(PAD_LABEL)
                else:
                    emos.append(EMOTION2IDX.get(row["emotion"], 0))
                    sents.append(SENTIMENT2IDX.get(row["sentiment"], 1))
                    self.num_kept_utts += 1
            self.dialogues.append({
                "features": torch.stack(feats),                       # (T, 2048)
                "emotions": torch.tensor(emos, dtype=torch.long),     # (T,)
                "sentiments": torch.tensor(sents, dtype=torch.long),  # (T,)
            })

    def __len__(self) -> int:
        return len(self.dialogues)

    def __getitem__(self, idx: int) -> Dict:
        return self.dialogues[idx]


class WindowedDialogueDataset(Dataset):
    """One training example per utterance, with a fixed ±K neighbour window.

    For each utterance ``i`` in each dialogue we extract features from indices
    ``[max(0, i-K), min(T-1, i+K)]`` (up to ``2K+1`` utterances, shorter near
    dialogue edges). All labels in the window are set to ``PAD_LABEL`` except
    the centre, so loss and metrics score only the centre utterance while the
    BiLSTM sees its ±K neighbours as context.

    Args:
        base: A :class:`DialogueDataset` whose dialogues are already loaded.
        context_window: K, the number of neighbours on each side (>=0).
    """

    def __init__(self, base: "DialogueDataset", context_window: int) -> None:
        assert context_window >= 0, "context_window must be >= 0"
        self.context_window = context_window
        self.samples: List[Dict] = []
        for dia in base.dialogues:
            feats = dia["features"]
            emos = dia["emotions"]
            sents = dia["sentiments"]
            T = feats.size(0)
            for i in range(T):
                start = max(0, i - context_window)
                end = min(T, i + context_window + 1)
                center_idx = i - start
                W = end - start
                w_e = torch.full((W,), PAD_LABEL, dtype=torch.long)
                w_s = torch.full((W,), PAD_LABEL, dtype=torch.long)
                w_e[center_idx] = emos[i]
                w_s[center_idx] = sents[i]
                self.samples.append({
                    "features": feats[start:end],
                    "emotions": w_e,
                    "sentiments": w_s,
                })
        # Reuse the count trackers so main() log lines still make sense
        self.num_total_utts = base.num_total_utts
        self.num_kept_utts = base.num_kept_utts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


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
    parser.add_argument("--tag", type=str, default="",
                        help="Suffix for the results JSON, e.g. '_wer25_filter'.")
    parser.add_argument("--context_window", type=int, default=-1,
                        help=("If ≥ 0, restrict each utterance's context to ±K "
                              "neighbours (K=0 → no context, K=1 → ±1 utterance, "
                              "etc.). Default −1 = whole dialogue."))
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["data"]["seed"])
    device = get_device()
    logger = setup_logging(config["training"]["log_dir"], "train_context")
    logger.info("Config: %s | hidden=%d layers=%d | device=%s | tag=%r",
                args.config, args.hidden_dim, args.num_layers, device, args.tag)

    text_dim = config["model"]["text_dim"]
    acoustic_dim = config["model"]["acoustic_dim"]

    # Filter keep-lists (per-split) come from filtering.keys_paths when
    # filtering.enabled=true. Filtered utterances remain in each dialogue for
    # BiLSTM context but their labels are masked, so they do not contribute to
    # loss or metrics — same convention as the other datasets.
    filt = config.get("filtering", {})
    filter_on = bool(filt.get("enabled", False))
    filter_keys = filt.get("keys_paths", {}) if filter_on else {}

    def make_ds(split):
        return DialogueDataset(
            config["data"]["meld_root"], config["data"]["text_embeddings_path"],
            config["data"]["embeddings_path"], split, text_dim, acoustic_dim,
            filtered_keys_path=filter_keys.get(split))

    train_ds, dev_ds = make_ds("train"), make_ds("dev")
    logger.info("Train dialogues: %d | Dev dialogues: %d", len(train_ds), len(dev_ds))
    if filter_on:
        logger.info(
            "Filter enabled | train utterances: %d/%d kept | dev utterances: %d/%d kept",
            train_ds.num_kept_utts, train_ds.num_total_utts,
            dev_ds.num_kept_utts, dev_ds.num_total_utts,
        )

    # Windowed mode — replace each dialogue with per-utterance ±K windows.
    # Auto-append e.g. "_win1" to the tag so results file names are distinct.
    tag = args.tag
    if args.context_window >= 0:
        K = args.context_window
        tag = f"{tag}_win{K}" if tag else f"_win{K}"
        logger.info("Context window K=%d (window size %d) — using WindowedDialogueDataset", K, 2 * K + 1)
        train_ds = WindowedDialogueDataset(train_ds, K)
        dev_ds   = WindowedDialogueDataset(dev_ds, K)
        logger.info("Windowed | train examples: %d | dev examples: %d",
                    len(train_ds), len(dev_ds))
    args.tag = tag
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

    # NOTE: the tag MUST be part of the checkpoint directory. Previously this was
    # a fixed "context_bclstm" path while --tag only renamed the results JSON, so
    # every run silently overwrote the previous run's weights -- the unfiltered
    # and wer25 models never coexisted on disk, making them impossible to compare
    # or re-evaluate afterwards. Distinct tag => distinct directory.
    ckpt_dir = Path(config["training"]["checkpoint_dir"]) / f"context_bclstm{tag}"
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
    test_ds = make_ds("test")
    if filter_on:
        logger.info("Filter enabled | test utterances: %d/%d kept",
                    test_ds.num_kept_utts, test_ds.num_total_utts)
    if args.context_window >= 0:
        test_ds = WindowedDialogueDataset(test_ds, args.context_window)
        logger.info("Windowed test examples: %d", len(test_ds))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate)
    _, em_t, sm_t = evaluate(model, test_loader, device, e_crit, s_crit)
    logger.info("=== TEST (best checkpoint, epoch %d) ===", best["epoch"] + 1)
    log_metrics(em_t, "test", "emotion", logger)
    log_metrics(sm_t, "test", "sentiment", logger)

    # ---- Save test results as JSON (same format as evaluate.py) ----
    output_dir = Path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    def _drop_report(d: Dict) -> Dict:
        return {k: v for k, v in d.items() if k != "report"}
    results = {
        "config": args.config,
        "model":  "bclstm",
        "tag":    args.tag,
        "checkpoint": str(ckpt_dir / "best_context.pt"),
        "best_epoch": best["epoch"] + 1,
        "filter_enabled": filter_on,
        "n_train_dialogues": len(train_ds),
        "n_dev_dialogues": len(dev_ds),
        "n_test_dialogues": len(test_ds),
        "n_test_utterances_scored": test_ds.num_kept_utts if filter_on else test_ds.num_total_utts,
        "emotion":   _drop_report(em_t),
        "sentiment": _drop_report(sm_t),
    }
    results_path = output_dir / f"test_results_bclstm{args.tag}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
