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

#: Refuse to build a dataset if more than this fraction of utterances lacks an
#: embedding. Matches src/training/fusion_data.py so the two models are held to
#: the same data-integrity standard.
_MAX_MISS_RATE = 0.02

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
                 filtered_keys_path: Optional[str] = None,
                 fused_path: Optional[str] = None,
                 fused_mode: str = "replace") -> None:
        """
        fused_path: If given, load ``{split}_fused.pt`` from here and use that
            single learned vector per utterance INSTEAD of concatenating the
            text and acoustic caches. This is the stacked model: fusion learns
            the representation, bc-LSTM adds dialogue context, and context
            becomes the only difference between the two models.
        """
        self.text_dim, self.acoustic_dim = text_dim, acoustic_dim
        self.fused_path = fused_path
        self.fused_mode = fused_mode

        # weights_only=True: these caches are plain tensor dicts, and the
        # default flipped in torch >= 2.6 -- pin it so a cluster upgrade
        # cannot silently change load behaviour.
        if fused_path is not None:
            fused_file = Path(fused_path) / f"{split}_fused.pt"
            if not fused_file.exists():
                raise FileNotFoundError(
                    f"{fused_file} not found. Run "
                    "src/preprocessing/extract_fused_features.py first.")
            fused_emb: Dict[str, Tensor] = torch.load(
                str(fused_file), map_location="cpu", weights_only=True)
            fused_dim = int(next(iter(fused_emb.values())).shape[-1])
            if fused_mode == "acoustic":
                # Text stays real; only the ACOUSTIC half is swapped for the
                # pooled vector. The context models otherwise read the masked
                # mean, whose acoustic cosine gap is 0.0138 against attention
                # pooling's 0.0779 -- a 5.6x difference in class separation
                # that every context result to date was handicapped by.
                self.feature_dim = text_dim + fused_dim
            elif fused_mode == "concat":
                # Keep the raw caches too: the learned vector is APPENDED, not
                # substituted, so no information is thrown away.
                self.feature_dim = fused_dim + text_dim + acoustic_dim
            else:
                self.feature_dim = fused_dim
                text_emb, acou_emb = None, None
        else:
            fused_emb = None
            self.feature_dim = text_dim + acoustic_dim

        text_file = Path(text_embeddings_path) / f"{split}_text_embeddings.pt"
        acou_file = Path(embeddings_path) / f"{split}_embeddings_maskedmean.pt"
        if fused_emb is None or fused_mode in ("concat", "acoustic"):
            for f, hint in ((text_file, "extract_text_embeddings*.py"),
                            (acou_file, "transcribe_all.py")):
                if not f.exists():
                    raise FileNotFoundError(f"{f} not found. Run {hint} first.")
        if fused_emb is None or fused_mode in ("concat", "acoustic"):
            text_emb = torch.load(
                str(text_file), map_location="cpu", weights_only=True)
        # Masked mean over REAL frames, not the legacy {split}_embeddings.pt,
        # which averaged over vllm's 30 s zero-padding and so diluted a short
        # utterance by up to ~15x.
            acou_emb = torch.load(
                str(acou_file), map_location="cpu", weights_only=True)

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
        missing_text: List[str] = []
        missing_acoustic: List[str] = []

        df = pd.read_csv(Path(meld_root) / _SPLIT_CSV[split])
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
        df = df.dropna(subset=["emotion", "sentiment"]).reset_index(drop=True)
        df["emotion"] = df["emotion"].str.strip().str.lower()
        df["sentiment"] = df["sentiment"].str.strip().str.lower()
        df["dialogue_id"] = df["dialogue_id"].astype(int)
        df["utterance_id"] = df["utterance_id"].astype(int)

        self.dialogues: List[Dict] = []
        for dia_id, grp in df.groupby("dialogue_id", sort=True):
            grp = grp.sort_values("utterance_id")
            feats, emos, sents, keys = [], [], [], []
            for _, row in grp.iterrows():
                key = f"dia{dia_id}_utt{int(row['utterance_id'])}"
                keys.append(key)
                self.num_total_utts += 1
                if fused_emb is not None:
                    v = fused_emb.get(key)
                    if v is None:
                        missing_text.append(key)
                        feats.append(torch.zeros(self.feature_dim))
                        emos.append(PAD_LABEL); sents.append(PAD_LABEL)
                        continue
                    if fused_mode == "replace" and v.shape[-1] != self.feature_dim:
                        raise ValueError(
                            f"{key}: fused vector is {v.shape[-1]}-d, expected "
                            f"{self.feature_dim}.")
                    if fused_mode == "acoustic":
                        t2 = text_emb.get(key)
                        if t2 is None:
                            missing_text.append(key)
                            feats.append(torch.zeros(self.feature_dim))
                            emos.append(PAD_LABEL); sents.append(PAD_LABEL)
                            continue
                        feats.append(torch.cat([t2.float(), v.float()]))
                    elif fused_mode == "concat":
                        t2, a2 = text_emb.get(key), acou_emb.get(key)
                        if t2 is None or a2 is None:
                            (missing_text if t2 is None else missing_acoustic).append(key)
                            feats.append(torch.zeros(self.feature_dim))
                            emos.append(PAD_LABEL); sents.append(PAD_LABEL)
                            continue
                        feats.append(torch.cat([v.float(), t2.float(), a2.float()]))
                    else:
                        feats.append(v.float())
                    if keep_set is not None and key not in keep_set:
                        emos.append(PAD_LABEL); sents.append(PAD_LABEL)
                    else:
                        emo, sen = row["emotion"], row["sentiment"]
                        if emo not in EMOTION2IDX:
                            raise ValueError(f"{key}: unrecognised emotion {emo!r}.")
                        if sen not in SENTIMENT2IDX:
                            raise ValueError(f"{key}: unrecognised sentiment {sen!r}.")
                        emos.append(EMOTION2IDX[emo]); sents.append(SENTIMENT2IDX[sen])
                        self.num_kept_utts += 1
                    continue

                t = text_emb.get(key)
                a = acou_emb.get(key)

                # A missing embedding used to become torch.zeros(...), which is
                # indistinguishable from a real vector and was counted nowhere.
                # Worse in a dialogue model than in a per-utterance one: a zero
                # utterance sits INSIDE the sequence, so the BiLSTM propagates
                # it into its neighbours' hidden states and one failed
                # extraction degrades context for the whole conversation.
                #
                # The utterance cannot simply be dropped -- that would break the
                # sequence the model exists to read -- so it is kept as context
                # but its labels are masked, and counted SEPARATELY from
                # filtering so the two never get confused.
                if t is None or a is None:
                    (missing_text if t is None else missing_acoustic).append(key)
                    feats.append(torch.zeros(text_dim + acoustic_dim))
                    emos.append(PAD_LABEL)
                    sents.append(PAD_LABEL)
                    continue

                if t.shape[-1] != text_dim:
                    raise ValueError(
                        f"{key}: text embedding is {t.shape[-1]}-d, expected "
                        f"{text_dim}. Wrong cache, or the config disagrees.")
                if a.shape[-1] != acoustic_dim:
                    raise ValueError(
                        f"{key}: acoustic embedding is {a.shape[-1]}-d, "
                        f"expected {acoustic_dim}.")

                feats.append(torch.cat([t.float(), a.float()]))

                if keep_set is not None and key not in keep_set:
                    # Filtered-out — keep in dialogue for context, mask labels.
                    emos.append(PAD_LABEL)
                    sents.append(PAD_LABEL)
                else:
                    # Unknown labels raise. Mapping them to a default silently
                    # converts a data error into a majority-class example.
                    emo, sen = row["emotion"], row["sentiment"]
                    if emo not in EMOTION2IDX:
                        raise ValueError(
                            f"{key}: unrecognised emotion {emo!r}. "
                            f"Expected one of {sorted(EMOTION2IDX)}.")
                    if sen not in SENTIMENT2IDX:
                        raise ValueError(
                            f"{key}: unrecognised sentiment {sen!r}. "
                            f"Expected one of {sorted(SENTIMENT2IDX)}.")
                    emos.append(EMOTION2IDX[emo])
                    sents.append(SENTIMENT2IDX[sen])
                    self.num_kept_utts += 1
            self.dialogues.append({
                "features": torch.stack(feats),                       # (T, 2048)
                "emotions": torch.tensor(emos, dtype=torch.long),     # (T,)
                "sentiments": torch.tensor(sents, dtype=torch.long),  # (T,)
                # Utterance keys, so the dev key SET can be hashed for the
                # comparability check. Not consumed by collate.
                "keys": keys,
            })

        self.missing_text = missing_text
        self.missing_acoustic = missing_acoustic
        n_missing = len(missing_text) + len(missing_acoustic)
        self.miss_rate = n_missing / max(1, self.num_total_utts)
        if n_missing:
            print(f"  {split}: {n_missing} utterance(s) lack an embedding "
                  f"({100 * self.miss_rate:.2f}%) -- kept as context, labels "
                  f"masked. text={len(missing_text)} acoustic="
                  f"{len(missing_acoustic)}; first: "
                  f"{(missing_text + missing_acoustic)[:3]}")
        # A handful is a data defect to report; a large fraction means the
        # wrong cache directory is configured, which should stop the run.
        if self.miss_rate > _MAX_MISS_RATE:
            raise ValueError(
                f"{split}: {100 * self.miss_rate:.1f}% of utterances lack an "
                f"embedding (limit {100 * _MAX_MISS_RATE:.0f}%). This usually "
                "means the wrong cache directory is configured, or extraction "
                "did not complete. Refusing to train on a partial dataset.")

    def utterance_keys(self) -> List[str]:
        """Every utterance key this dataset serves, dialogue order.

        Uniform with WindowedDialogueDataset so the comparability hash does not
        need to know which of the two it was handed.
        """
        return [k for d in self.dialogues for k in d["keys"]]

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
                    # Only the CENTRE utterance is scored, so the centre key is
                    # what identifies this example for the comparability hash.
                    "keys": [dia["keys"][i]] if dia.get("keys") else [],
                })
        # These describe the BASE dataset, not the windowed one. They coincide
        # with the scored-utterance count only because each window scores
        # exactly one centre -- record the dialogue count separately so
        # n_*_dialogues in the results JSON is not silently a window count.
        self.num_total_utts = base.num_total_utts
        self.num_kept_utts = base.num_kept_utts
        self.num_base_dialogues = len(base.dialogues)
        self.feature_dim = base.feature_dim

    def utterance_keys(self) -> List[str]:
        """Centre keys of every window -- one per scored utterance."""
        return [k for s_ in self.samples for k in s_["keys"]]

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
        # Every utterance in this batch is padding or filtered out. Return a
        # zero that is still CONNECTED TO THE GRAPH -- a bare 0.0 would detach
        # the batch and break backward(). This is not a no-op despite looking
        # like one.
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
    total = torch.zeros((), device=device)
    ep, el, spr, sl = [], [], [], []
    for feats, emos, sents, lengths in loader:
        feats, emos_d, sents_d = feats.to(device), emos.to(device), sents.to(device)
        s_logits, e_logits = model(feats, lengths)
        # Accumulate on device; .item() here forces a host sync per batch.
        total = total + (masked_loss(e_logits, emos_d, e_crit)
                         + masked_loss(s_logits, sents_d, s_crit)).detach()
        e_flat = e_logits.argmax(-1).reshape(-1).cpu()
        s_flat = s_logits.argmax(-1).reshape(-1).cpu()
        emo_flat = emos.reshape(-1); sen_flat = sents.reshape(-1)
        # Each task is masked by ITS OWN labels. Using the emotion mask for
        # sentiment happens to work only while the two are always masked
        # together; the moment anything masks them independently (a
        # sentiment-specific filter, a missing sentiment label) the sentiment
        # predictions and labels misalign and sentiment F1 becomes meaningless
        # with no error raised.
        ve = emo_flat != PAD_LABEL
        vs = sen_flat != PAD_LABEL
        ep.extend(e_flat[ve].tolist()); el.extend(emo_flat[ve].tolist())
        spr.extend(s_flat[vs].tolist()); sl.extend(sen_flat[vs].tolist())
    return (float(total) / max(len(loader), 1),
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
    parser.add_argument(
        "--fused_path", type=str, default=None,
        help=("Directory with {split}_fused.pt from extract_fused_features.py. "
              "When given, the BiLSTM consumes fusion's LEARNED 512-d vector "
              "instead of the raw 768+1280 concatenation -- the stacked model, "
              "in which context is the only difference from plain fusion."))
    parser.add_argument(
        "--fused_mode", choices=["replace", "concat", "acoustic"],
        default="replace",
        help=("acoustic: keep the real text embedding and swap ONLY the "
              "acoustic half for the given pooled vector -- for feeding an "
              "attention-pooled acoustic representation to the context models, "
              "which otherwise read the masked mean (cosine gap 0.0138 vs "
              "0.0779, a 5.6x difference in class separation). "
              "replace: use ONLY fusion's learned vector (512-d). "
              "concat: use [fused | text | acoustic] (2560-d) so the BiLSTM "
              "keeps everything the raw features carry AND the learned "
              "representation. `replace` lost to plain features on gold "
              "(-0.019) and asr_cleaned (-0.009) -- compressing 2048 -> 512 "
              "through a per-utterance objective discards information the "
              "context model wanted. `concat` cannot lose that information."))
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
            filtered_keys_path=filter_keys.get(split),
            fused_path=args.fused_path, fused_mode=args.fused_mode)

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
        # A tag containing "/" is an explicit path the caller chose (e.g.
        # "gold/k1"); appending _win{K} to it would bury the window twice and
        # break the directory tree. Only auto-name when the caller did not.
        if "/" not in tag:
            tag = f"{tag}_win{K}" if tag else f"_win{K}"
        logger.info("Context window K=%d (window size %d) — using WindowedDialogueDataset", K, 2 * K + 1)
        train_ds = WindowedDialogueDataset(train_ds, K)
        dev_ds   = WindowedDialogueDataset(dev_ds, K)
        logger.info("Windowed | train examples: %d | dev examples: %d",
                    len(train_ds), len(dev_ds))
    args.tag = tag
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        # Pinned shuffle order, independent of however much global RNG anything
        # else consumes, so two runs shuffle identically.
        generator=torch.Generator().manual_seed(config["data"]["seed"]),
        # num_workers left at 0 deliberately: features are already resident in
        # RAM, so workers would fork a copy of the whole cache for no gain.
    )
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # From the dataset, not text_dim + acoustic_dim: in fused mode the feature
    # is a single learned vector whose width comes from the fusion model's
    # hidden_dim, and hardcoding the sum would build a model of the wrong shape.
    feat_dim = getattr(train_ds, "feature_dim", text_dim + acoustic_dim)
    logger.info("BiLSTM input_dim=%d (%s)", feat_dim,
                "fused" if args.fused_path else "text+acoustic concat")
    model = BiLSTMContext(
        input_dim=feat_dim, hidden_dim=args.hidden_dim,
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
    # An explicit path tag nests directly; otherwise keep the historical
    # "context_bclstm<tag>" name so old invocations still land where they did.
    ckpt_dir = (Path(config["training"]["checkpoint_dir"]) / tag.lstrip("/")
                if "/" in tag
                else Path(config["training"]["checkpoint_dir"]) / f"context_bclstm{tag}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Scheduler patience must sit below early-stopping patience with room for
    # the reduced LR to demonstrate an effect, or the reduction fires one epoch
    # before the run dies and is purely decorative.
    sched_patience = max(0, args.patience - 3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=sched_patience)
    logger.info("LR scheduler patience=%d vs early-stop patience=%d",
                sched_patience, args.patience)

    # Hash the dev key SET (sorted -- we are hashing membership, not order) so
    # runs launched at different times are verifiably comparable. A differing
    # hash means the runs were early-stopped against different dev sets.
    import hashlib
    dev_keys = sorted(dev_ds.utterance_keys())
    dev_hash = (hashlib.sha1("".join(dev_keys).encode()).hexdigest()[:12]
                if dev_keys else "n/a")
    logger.info("dev dialogues: %d | dev key hash: %s", len(dev_ds), dev_hash)

    # best_epoch stays None until a checkpoint is actually written. best_wf1
    # starting at 0.0 cannot distinguish "never improved" from "improved to
    # exactly 0.0", and the test block below LOADS best_context.pt -- without
    # this guard a degenerate run raises FileNotFoundError after training
    # fully, with no explanation.
    best_wf1, no_improve, best_epoch = 0.0, 0, None

    for epoch in range(args.epochs):
        tr = train_one_epoch(model, train_loader, optimizer, device, e_crit, s_crit, max_grad_norm)
        vl, em, sm = evaluate(model, dev_loader, device, e_crit, s_crit)
        wf1 = em["weighted_f1"]
        logger.info("Epoch %d/%d | Train %.4f | Val %.4f | Emo WF1 %.4f | Emo Macro %.4f",
                    epoch + 1, args.epochs, tr, vl, wf1, em["macro_f1"])
        log_metrics(em, "dev", "emotion", logger)
        log_metrics(sm, "dev", "sentiment", logger)
        sched.step(wf1)
        if wf1 > best_wf1:
            best_wf1, no_improve, best_epoch = wf1, 0, epoch
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

    if best_epoch is None:
        raise RuntimeError(
            "No checkpoint was ever written: dev weighted F1 never exceeded "
            f"{best_wf1:.4f} in {args.epochs} epoch(s). There is nothing to "
            "evaluate on test. This is a failed run, not a completed one.")

    # ---- Final TEST evaluation of the best checkpoint (comparable to other test numbers) ----
    best = torch.load(ckpt_dir / "best_context.pt", map_location=device,
                      weights_only=False)
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
        # In windowed mode len(ds) is a WINDOW count, not a dialogue count --
        # report both under honest names rather than one under a wrong one.
        "n_train_examples": len(train_ds),
        "n_dev_examples": len(dev_ds),
        "n_test_examples": len(test_ds),
        "n_test_dialogues": getattr(test_ds, "num_base_dialogues",
                                    len(test_ds)),
        "windowed": args.context_window >= 0,
        "context_window": args.context_window if args.context_window >= 0 else None,
        "fused_path": args.fused_path,
        "input_dim": feat_dim,
        "stacked": args.fused_path is not None,
        "fused_mode": args.fused_mode if args.fused_path else None,
        "n_test_utterances_scored": test_ds.num_kept_utts if filter_on else test_ds.num_total_utts,
        "dev_key_hash": dev_hash,
        "seed": config["data"]["seed"],
        "best_dev_weighted_f1": best_wf1,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "batch_size": args.batch_size,
        "trainable_parameters": model.trainable_parameters(),
        # bc-LSTM uses FocalLoss WITHOUT class alpha, while train_fusion_seq.py
        # offers weighted/focal that both carry inverse-frequency alpha. That is
        # an uncontrolled difference between two models compared in the same
        # results table -- recorded here so it is visible rather than assumed.
        "loss": f"focal(gamma={gamma}, alpha=None)",
        "emotion":   _drop_report(em_t),
        "sentiment": _drop_report(sm_t),
    }
    flat = args.tag.strip("/").replace("/", "_") or "default"
    results_path = output_dir / f"test_results_bclstm_{flat}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    # The sklearn text report is stripped from the JSON (it is long and not
    # machine-readable) but is genuinely useful for the writeup, so keep it.
    for name, mt in (("emotion", em_t), ("sentiment", sm_t)):
        rep = mt.get("report")
        if rep:
            (output_dir / f"test_report_bclstm_{flat}_{name}.txt").write_text(
                str(rep), encoding="utf-8")

    # Completion marker. best_model.pt existing does NOT mean a run finished --
    # the same reason train_fusion_seq.py writes one.
    with open(ckpt_dir / "TRAINING_COMPLETE.json", "w", encoding="utf-8") as f:
        json.dump({"model": "bclstm", "tag": args.tag,
                   "best_dev_weighted_f1": best_wf1,
                   "best_epoch": best_epoch,
                   "checkpoint_written": best_epoch is not None,
                   "epochs_run": epoch + 1,
                   "epochs_configured": args.epochs,
                   "early_stopped": no_improve >= args.patience,
                   "dev_key_hash": dev_hash,
                   "seed": config["data"]["seed"],
                   "filter_enabled": filter_on,
                   "context_window": args.context_window,
                   "test_emotion_weighted_f1": em_t.get("weighted_f1"),
                   "trainable_parameters": model.trainable_parameters()},
                  f, indent=2)
    logger.info("Wrote %s", ckpt_dir / "TRAINING_COMPLETE.json")


if __name__ == "__main__":
    main()
