"""
extract_text_probs.py — XLM-R per-class probabilities for late fusion
======================================================================
Runs the (already fine-tuned) Phase-1 XLM-RoBERTa text model in inference mode
and saves per-utterance softmax probabilities for emotion (7) and sentiment (3),
keyed by "dia{d}_utt{u}", for dev + test. No training — eval only.

Output (consumed by src/evaluation/late_fusion.py):
  results/<run>/xlmr_text_probs.json
  { "dev":  { "dia0_utt0": {"emotion_probs": [...7], "sentiment_probs": [...3]}, ... },
    "test": { ... } }

Usage
-----
  python3.12 src/evaluation/extract_text_probs.py --config src/configs/mini.yaml \\
      --checkpoint_path /dcs/large/u5734759/checkpoints/mini_aug/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import EMOTION_NAMES, SENTIMENT_NAMES
from src.models.xlmr import XLMRobertaClassifier
from src.utils import get_device, load_config, set_seed, setup_logging

EMOTION2IDX: Dict[str, int] = {n: i for i, n in enumerate(EMOTION_NAMES)}
SENTIMENT2IDX: Dict[str, int] = {n: i for i, n in enumerate(SENTIMENT_NAMES)}


class KeyedTextDataset(Dataset):
    """Tokenised ASR transcripts + labels, keeping the utterance key.

    Args:
        meld_root: MELD root directory (CSV files).
        transcripts_path: Directory with {split}_transcripts.json.
        split: 'train' | 'dev' | 'test'.
        tokenizer: XLM-R tokenizer.
        max_length: Max token length.
    """

    def __init__(self, meld_root, transcripts_path, split, tokenizer, max_length=128) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

        df = pd.read_csv(Path(meld_root) / f"{split}_sent_emo.csv")
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
        df = df.dropna(subset=["emotion", "sentiment"]).reset_index(drop=True)
        df["emotion"] = df["emotion"].str.strip().str.lower()
        df["sentiment"] = df["sentiment"].str.strip().str.lower()

        with open(Path(transcripts_path) / f"{split}_transcripts.json", "r", encoding="utf-8") as f:
            transcripts: Dict[str, str] = json.load(f)

        self.samples: List[Tuple[str, str]] = []
        for _, row in df.iterrows():
            key = f"dia{int(row['dialogue_id'])}_utt{int(row['utterance_id'])}"
            self.samples.append((key, transcripts.get(key, "")))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        key, text = self.samples[idx]
        enc = self.tokenizer(
            text, max_length=self.max_length, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "key": key,
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }


@torch.no_grad()
def extract_split(model, loader, device) -> Dict[str, Dict]:
    """Return {key: {emotion_probs, sentiment_probs}} for one split."""
    model.eval()
    out: Dict[str, Dict] = {}
    for batch in loader:
        keys = batch["key"]
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        sentiment_logits, emotion_logits = model(input_ids, attention_mask)
        e_probs = F.softmax(emotion_logits, dim=-1).cpu().numpy()
        s_probs = F.softmax(sentiment_logits, dim=-1).cpu().numpy()
        for i, key in enumerate(keys):
            out[key] = {
                "emotion_probs": e_probs[i].tolist(),
                "sentiment_probs": s_probs[i].tolist(),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract XLM-R per-class probabilities (eval only).")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the chosen XLM-R best_model.pt (e.g. mini_aug).")
    parser.add_argument("--splits", nargs="+", default=["dev", "test"],
                        choices=["train", "dev", "test"])
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["data"]["seed"])
    device = get_device()
    logger = setup_logging(config["training"]["log_dir"], "extract_text_probs")

    ckpt_path = Path(args.checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    xlmr_id = config["model"]["xlmr_id"]
    tokenizer = AutoTokenizer.from_pretrained(xlmr_id)
    model = XLMRobertaClassifier(
        model_name_or_path=xlmr_id,
        num_sentiment_classes=config["model"]["num_sentiment_classes"],
        num_emotion_classes=config["model"]["num_classes"],
        dropout_prob=config["model"]["dropout"],
    )
    state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    logger.info("Loaded XLM-R checkpoint: %s (saved metric=%.4f)",
                ckpt_path, state.get("metric", float("nan")))

    result: Dict[str, Dict] = {}
    for split in args.splits:
        ds = KeyedTextDataset(
            meld_root=config["data"]["meld_root"],
            transcripts_path=config["data"]["transcripts_path"],
            split=split,
            tokenizer=tokenizer,
            max_length=config["data"]["max_text_length"],
        )
        loader = DataLoader(
            ds, batch_size=config["training"]["batch_size"], shuffle=False,
            num_workers=config["data"]["num_workers"], pin_memory=True,
        )
        result[split] = extract_split(model, loader, device)
        logger.info("Split '%s': %d utterances", split, len(result[split]))

    out_path = Path(config["evaluation"]["output_dir"]) / "xlmr_text_probs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    logger.info("Saved text probabilities → %s", out_path)


if __name__ == "__main__":
    main()
