"""
extract_text_embeddings_asr.py — XLM-R CLS embeddings from ASR transcripts
===========================================================================
Identical to extract_text_embeddings.py EXCEPT the text source: instead of the
gold CSV utterances, it encodes Voxtral's **ASR transcripts**
({transcripts_path}/{split}_transcripts.json). This is the realistic setting —
on unseen UK/AU audio you won't have gold transcripts, you'll have ASR output
(with accent-induced errors). Encoding ASR text makes the in-domain pipeline
match the dialect-eval scenario, so the two are comparable.

Same XLM-R checkpoint as the gold extractor → the only difference is gold vs ASR
text, so the gold-vs-ASR bc-LSTM comparison is clean.

Output (separate dir so it doesn't clobber the gold cache):
  {out_dir}/{split}_text_embeddings.pt   → dict "dia{D}_utt{U}" → Tensor(768,)

Usage
-----
  python3.12 src/preprocessing/extract_text_embeddings_asr.py \\
      --config src/configs/mini.yaml \\
      --out_dir /dcs/large/u5734759/data/meld_text_embeddings_asr
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.xlmr import XLMRobertaClassifier
from src.utils import get_device, load_config, set_seed, setup_logging

_SPLIT_CSV: Dict[str, str] = {
    "train": "train_sent_emo.csv",
    "dev":   "dev_sent_emo.csv",
    "test":  "test_sent_emo.csv",
}


class AsrTextDataset(Dataset):
    """Tokenised ASR transcripts keyed by utterance (keys come from the CSV).

    Args:
        meld_root: MELD root (for the split CSV → canonical key set).
        transcripts_path: Dir with {split}_transcripts.json (Voxtral ASR).
        split: 'train' | 'dev' | 'test'.
        tokenizer: XLM-R tokenizer.
        max_length: Max token length.
    """

    def __init__(self, meld_root, transcripts_path, split, tokenizer, max_length=128) -> None:
        df = pd.read_csv(Path(meld_root) / _SPLIT_CSV[split])
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
        df = df.dropna(subset=["dialogue_id", "utterance_id"]).reset_index(drop=True)

        with open(Path(transcripts_path) / f"{split}_transcripts.json", "r", encoding="utf-8") as f:
            transcripts: Dict[str, str] = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: List[Tuple[str, str]] = []
        n_missing = 0
        for _, row in df.iterrows():
            key = f"dia{int(row['dialogue_id'])}_utt{int(row['utterance_id'])}"
            text = str(transcripts.get(key, "")).strip()
            if text == "":
                n_missing += 1
            self.samples.append((key, text))
        self.n_missing = n_missing

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
def extract_split(model, loader, device) -> Dict[str, Tensor]:
    model.eval()
    embeddings: Dict[str, Tensor] = {}
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        cls = model.get_text_representation(input_ids, attention_mask)
        for key, vec in zip(batch["key"], cls.cpu()):
            embeddings[key] = vec
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache XLM-R CLS embeddings from Voxtral ASR transcripts."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="XLM-R checkpoint (default: <checkpoint_dir>/best_model.pt — "
                             "match whatever produced the gold cache for a clean comparison).")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output dir (default: <text_embeddings_path>_asr).")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["data"]["seed"])
    device = get_device()
    logger = setup_logging(config["training"]["log_dir"], "extract_text_embeddings_asr")

    xlmr_id = config["model"]["xlmr_id"]
    tokenizer = AutoTokenizer.from_pretrained(xlmr_id)

    ckpt_path = Path(args.checkpoint_path) if args.checkpoint_path else (
        Path(config["training"]["checkpoint_dir"]) / "best_model.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"XLM-R checkpoint not found: {ckpt_path}")

    model = XLMRobertaClassifier(
        model_name_or_path=xlmr_id,
        num_sentiment_classes=config["model"]["num_sentiment_classes"],
        num_emotion_classes=config["model"]["num_classes"],
        dropout_prob=config["model"]["dropout"],
    )
    state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    logger.info("Loaded XLM-R checkpoint: %s", ckpt_path)

    out_dir = Path(args.out_dir) if args.out_dir else Path(
        str(config["data"]["text_embeddings_path"]) + "_asr")
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "dev", "test"):
        ds = AsrTextDataset(
            meld_root=config["data"]["meld_root"],
            transcripts_path=config["data"]["transcripts_path"],
            split=split,
            tokenizer=tokenizer,
            max_length=config["data"]["max_text_length"],
        )
        loader = DataLoader(
            ds, batch_size=config["training"]["batch_size"], shuffle=False,
            num_workers=config["data"]["num_workers"], pin_memory=True)
        emb = extract_split(model, loader, device)
        torch.save(emb, str(out_dir / f"{split}_text_embeddings.pt"))
        logger.info("Split '%s': %d embeddings (%d empty ASR transcripts) → %s",
                    split, len(emb), ds.n_missing, out_dir / f"{split}_text_embeddings.pt")

    logger.info("ASR text embedding extraction complete → %s", out_dir)


if __name__ == "__main__":
    main()
