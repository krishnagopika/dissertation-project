"""
extract_text_embeddings.py — Pre-cache XLM-RoBERTa Text Embeddings
===================================================================
Loads the Phase 1 fine-tuned XLM-RoBERTa checkpoint and extracts the
768-dim [CLS] representation for every MELD utterance using the gold
text from the dataset CSVs (not ASR transcripts).

Saves one file per split:
  {text_embeddings_path}/{split}_text_embeddings.pt
    → dict mapping "dia{D}_utt{U}" → Tensor(768,)

Run this once after Phase 1 completes. Phase 2 fusion training then
loads these cached vectors directly — no XLM-RoBERTa at runtime.

Usage
-----
  python3.12 src/preprocessing/extract_text_embeddings.py \\
      --config src/configs/mini.yaml

Slurm: see src/scripts/extract_text_embeddings.sbatch
"""

from __future__ import annotations

import argparse
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


class GoldTextDataset(Dataset):
    """Minimal dataset returning tokenised gold utterances with their keys.

    Args:
        meld_root: Path to MELD root directory.
        split: One of 'train', 'dev', 'test'.
        tokenizer: HuggingFace tokenizer for XLM-RoBERTa.
        max_length: Maximum tokenised sequence length.
    """

    def __init__(
        self,
        meld_root: str,
        split: str,
        tokenizer,
        max_length: int = 128,
    ) -> None:
        assert split in ("train", "dev", "test"), (
            f"split must be train/dev/test, got {split}"
        )
        csv_path = Path(meld_root) / _SPLIT_CSV[split]
        if not csv_path.exists():
            raise FileNotFoundError(f"MELD CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        df.columns = (
            df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
        )
        df = df.dropna(subset=["utterance", "dialogue_id", "utterance_id"]).reset_index(
            drop=True
        )

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: List[Tuple[str, str]] = []  # (key, utterance)
        for _, row in df.iterrows():
            key = f"dia{int(row['dialogue_id'])}_utt{int(row['utterance_id'])}"
            text = str(row.get("utterance", "")).strip()
            self.samples.append((key, text))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        key, text = self.samples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "key":            key,
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


@torch.no_grad()
def extract_split(
    model: XLMRobertaClassifier,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Tensor]:
    """Run inference and collect {key: cls_vector} for a split.

    Args:
        model: Fine-tuned XLMRobertaClassifier (frozen).
        loader: DataLoader for the split.
        device: Target device.

    Returns:
        Dictionary mapping utterance key to 768-dim CLS tensor (on CPU).
    """
    model.eval()
    embeddings: Dict[str, Tensor] = {}

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        keys: List[str] = batch["key"]

        cls_vectors = model.get_text_representation(input_ids, attention_mask)
        # cls_vectors: (B, 768)

        for key, vec in zip(keys, cls_vectors.cpu()):
            embeddings[key] = vec

    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and cache XLM-RoBERTa CLS embeddings from gold MELD text."
    )
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["data"]["seed"])
    device = get_device()

    logger = setup_logging(config["training"]["log_dir"], "extract_text_embeddings")
    logger.info("Config: %s | Device: %s", args.config, device)

    xlmr_id   = config["model"]["xlmr_id"]
    tokenizer = AutoTokenizer.from_pretrained(xlmr_id)

    # Load Phase 1 checkpoint
    ckpt_path = Path(config["training"]["checkpoint_dir"]) / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Phase 1 checkpoint not found: {ckpt_path}. "
            "Run Phase 1 training first."
        )

    model = XLMRobertaClassifier(
        model_name_or_path    = xlmr_id,
        num_sentiment_classes = config["model"]["num_sentiment_classes"],
        num_emotion_classes   = config["model"]["num_classes"],
        dropout_prob          = config["model"]["dropout"],
    )
    state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    logger.info("Loaded Phase 1 checkpoint: %s", ckpt_path)

    output_dir = Path(config["data"]["text_embeddings_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size  = config["training"]["batch_size"]
    num_workers = config["data"]["num_workers"]
    max_length  = config["data"]["max_text_length"]

    for split in ("train", "dev", "test"):
        logger.info("Processing split: %s", split)

        ds = GoldTextDataset(
            meld_root  = config["data"]["meld_root"],
            split      = split,
            tokenizer  = tokenizer,
            max_length = max_length,
        )
        loader = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = False,
            num_workers = num_workers,
            pin_memory  = True,
            drop_last   = False,
        )

        embeddings = extract_split(model, loader, device)

        out_path = output_dir / f"{split}_text_embeddings.pt"
        torch.save(embeddings, str(out_path))
        logger.info(
            "Saved %d embeddings to %s", len(embeddings), out_path
        )

    logger.info("Text embedding extraction complete.")


if __name__ == "__main__":
    main()
