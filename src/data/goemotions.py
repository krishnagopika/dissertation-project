"""
GoEmotions Dataset Loader
=========================
GoEmotions — 58k Reddit comments annotated with 28 fine-grained emotion
categories (including neutral).

Data is loaded directly from HuggingFace Datasets using the
``"go_emotions"`` identifier with the ``"simplified"`` configuration, which
maps the original 27 emotion labels to 28 classes (the extra class is
``"neutral"``).

28 emotion classes (alphabetical order, index 0–27)
----------------------------------------------------
    admiration, amusement, anger, annoyance, approval, caring, confusion,
    curiosity, desire, disappointment, disapproval, disgust, embarrassment,
    excitement, fear, gratitude, grief, joy, love, nervousness, neutral,
    optimism, pride, realisation, relief, remorse, sadness, surprise

Labels are multi-hot encoded because a single comment can carry multiple
emotions simultaneously.

Reference
---------
Demszky et al., 2020. GoEmotions: A Dataset of Fine-Grained Emotions.
ACL 2020. https://arxiv.org/abs/2005.00547
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Label definitions
# ---------------------------------------------------------------------------

# 28 emotion labels in the order used by the HuggingFace ``simplified`` config
EMOTIONS: List[str] = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "neutral",
    "optimism",
    "pride",
    "realisation",
    "relief",
    "remorse",
    "sadness",
    "surprise",
]

NUM_CLASSES: int = len(EMOTIONS)  # 28
EMOTION2IDX: Dict[str, int] = {e: i for i, e in enumerate(EMOTIONS)}
IDX2EMOTION: Dict[int, str] = {i: e for i, e in enumerate(EMOTIONS)}

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class GoEmotionsDataset(Dataset):
    """PyTorch Dataset for GoEmotions (text-only, multi-label).

    Parameters
    ----------
    split:
        HuggingFace dataset split: ``"train"``, ``"validation"`` or
        ``"test"``.
    tokenizer_name:
        HuggingFace tokenizer identifier. Defaults to
        ``"FacebookAI/xlm-roberta-base"``.
    max_length:
        Maximum token sequence length for the tokenizer. Defaults to 128.
    cache_dir:
        Optional path to cache the HuggingFace dataset and tokenizer locally
        (useful on compute clusters without internet access after first download).
    """

    def __init__(
        self,
        split: str = "train",
        tokenizer_name: str = "FacebookAI/xlm-roberta-base",
        max_length: int = 128,
        cache_dir: Optional[str] = None,
    ) -> None:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        valid_splits = {"train", "validation", "test"}
        if split not in valid_splits:
            raise ValueError(
                f"split must be one of {valid_splits}, got '{split}'"
            )

        self.split = split
        self.max_length = max_length

        # Load dataset from HuggingFace Hub
        self.hf_dataset = load_dataset(
            "go_emotions",
            "simplified",
            split=split,
            cache_dir=cache_dir,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cache_dir,
        )

        # Pre-tokenize the entire split for efficiency
        self._input_ids: List[Tensor] = []
        self._attention_masks: List[Tensor] = []
        self._labels: List[Tensor] = []

        self._preprocess()

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _preprocess(self) -> None:
        """Tokenise all texts and encode multi-hot label vectors."""
        texts: List[str] = self.hf_dataset["text"]
        raw_labels: List[List[int]] = self.hf_dataset["labels"]

        encoding = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        self._input_ids = encoding["input_ids"]          # (N, L)
        self._attention_masks = encoding["attention_mask"]  # (N, L)

        # Build multi-hot label matrix
        n = len(texts)
        label_matrix = torch.zeros(n, NUM_CLASSES, dtype=torch.float32)
        for i, label_list in enumerate(raw_labels):
            for lbl in label_list:
                if 0 <= lbl < NUM_CLASSES:
                    label_matrix[i, lbl] = 1.0

        self._labels = label_matrix  # (N, 28)

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: int) -> Dict:
        return {
            "input_ids": self._input_ids[idx],          # (L,) long
            "attention_mask": self._attention_masks[idx],  # (L,) long
            "labels": self._labels[idx],                 # (28,) float32
            "text": self.hf_dataset["text"][idx],
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def collate_goemotions(batch: List[Dict]) -> Dict:
    """Collate GoEmotions samples into batched tensors.

    Since GoEmotions samples are pre-padded to ``max_length`` during
    construction, this function simply stacks tensors without dynamic
    padding.

    Parameters
    ----------
    batch:
        List of sample dicts from :class:`GoEmotionsDataset`.

    Returns
    -------
    dict with keys:
        - ``input_ids``:      ``(B, L)`` long tensor
        - ``attention_mask``: ``(B, L)`` long tensor
        - ``labels``:         ``(B, 28)`` float32 tensor (multi-hot)
        - ``texts``:          list of str
    """
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack(
            [item["attention_mask"] for item in batch]
        ),
        "labels": torch.stack([item["labels"] for item in batch]),
        "texts": [item["text"] for item in batch],
    }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def get_goemotions_dataloaders(
    batch_size: int = 32,
    num_workers: int = 4,
    tokenizer_name: str = "FacebookAI/xlm-roberta-base",
    max_length: int = 128,
    cache_dir: Optional[str] = None,
) -> Tuple:
    """Build train / validation / test DataLoaders for GoEmotions.

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader

    datasets = {
        split: GoEmotionsDataset(
            split=split,
            tokenizer_name=tokenizer_name,
            max_length=max_length,
            cache_dir=cache_dir,
        )
        for split in ("train", "validation", "test")
    }

    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collate_goemotions,
            pin_memory=True,
        )
        for split, ds in datasets.items()
    }

    return loaders["train"], loaders["validation"], loaders["test"]
