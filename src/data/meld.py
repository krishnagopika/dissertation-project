"""
MELD Dataset Loader
===================
Multimodal EmotionLines Dataset (MELD) — ~13,000 utterances from the Friends TV show.

Annotations:
  - 7 emotion classes: neutral, surprise, fear, sadness, joy, disgust, anger
  - 3 sentiment classes: neutral, positive, negative

Expected directory layout::

    <root>/
        train_sent_emo.csv
        dev_sent_emo.csv
        test_sent_emo.csv
        train/
            dia0_utt0.wav
            dia0_utt1.wav
            ...
        dev/
            dia0_utt0.wav
            ...
        test/
            dia0_utt0.wav
            ...

Reference
---------
Poria et al., 2019. MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations.
https://arxiv.org/abs/1810.02508
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torchaudio
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------

EMOTION2IDX: Dict[str, int] = {
    "neutral": 0,
    "surprise": 1,
    "fear": 2,
    "sadness": 3,
    "joy": 4,
    "disgust": 5,
    "anger": 6,
}

IDX2EMOTION: Dict[int, str] = {v: k for k, v in EMOTION2IDX.items()}

SENTIMENT2IDX: Dict[str, int] = {
    "neutral": 0,
    "positive": 1,
    "negative": 2,
}

IDX2SENTIMENT: Dict[int, str] = {v: k for k, v in SENTIMENT2IDX.items()}

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

TARGET_SAMPLE_RATE: int = 16_000


class MELDDataset(Dataset):
    """PyTorch Dataset for the MELD corpus.

    Parameters
    ----------
    root_dir:
        Path to the MELD root directory containing the CSV files and audio
        subdirectories (``train/``, ``dev/``, ``test/``).
    split:
        One of ``"train"``, ``"dev"`` or ``"test"``.
    max_duration_sec:
        Maximum audio duration in seconds. Waveforms longer than this are
        truncated; shorter ones are zero-padded to this length. Set to
        ``None`` to disable padding/truncation (use :func:`collate_meld`
        for dynamic padding in that case).
    target_sample_rate:
        Sample rate to resample all audio to. Defaults to 16 kHz.
    return_text:
        If ``True``, also return the raw utterance text from the CSV.
    """

    _SPLIT_CSV: Dict[str, str] = {
        "train": "train_sent_emo.csv",
        "dev": "dev_sent_emo.csv",
        "test": "test_sent_emo.csv",
    }

    def __init__(
        self,
        root_dir: str | os.PathLike,
        split: str = "train",
        max_duration_sec: Optional[float] = 10.0,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        return_text: bool = True,
    ) -> None:
        if split not in self._SPLIT_CSV:
            raise ValueError(
                f"split must be one of {list(self._SPLIT_CSV.keys())}, got '{split}'"
            )

        self.root_dir = Path(root_dir)
        self.split = split
        self.max_duration_sec = max_duration_sec
        self.target_sample_rate = target_sample_rate
        self.return_text = return_text

        # Maximum number of samples after padding
        self._max_samples: Optional[int] = (
            int(max_duration_sec * target_sample_rate)
            if max_duration_sec is not None
            else None
        )

        csv_path = self.root_dir / self._SPLIT_CSV[split]
        if not csv_path.exists():
            raise FileNotFoundError(f"MELD CSV not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        # Normalise column names to lowercase with underscores
        self.df.columns = (
            self.df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
        )

        # Validate required columns exist
        required = {"dialogue_id", "utterance_id", "emotion", "sentiment"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"MELD CSV is missing columns: {missing}")

        # Drop rows where audio or labels are NaN
        self.df = self.df.dropna(subset=["emotion", "sentiment"]).reset_index(
            drop=True
        )

        # Normalise label strings to lowercase
        self.df["emotion"] = self.df["emotion"].str.strip().str.lower()
        self.df["sentiment"] = self.df["sentiment"].str.strip().str.lower()

        self._audio_dir = self.root_dir / split
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_resampler(self, orig_sr: int) -> torchaudio.transforms.Resample:
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=self.target_sample_rate
            )
        return self._resamplers[orig_sr]

    def _load_audio(self, dialogue_id: int, utterance_id: int) -> Tensor:
        """Load, resample, convert to mono, and optionally pad/truncate audio.

        Returns a 1-D float32 tensor of shape ``(num_samples,)``.
        """
        filename = f"dia{dialogue_id}_utt{utterance_id}.wav"
        audio_path = self._audio_dir / filename

        if not audio_path.exists():
            # Return silence if the file is missing (graceful degradation)
            num_samples = self._max_samples or self.target_sample_rate
            return torch.zeros(num_samples, dtype=torch.float32)

        waveform, sample_rate = torchaudio.load(str(audio_path))  # (C, T)

        # Convert to mono by averaging channels
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # (1, T)

        # Resample if necessary
        if sample_rate != self.target_sample_rate:
            resampler = self._get_resampler(sample_rate)
            waveform = resampler(waveform)

        waveform = waveform.squeeze(0)  # (T,)

        # Pad or truncate
        if self._max_samples is not None:
            if waveform.shape[0] < self._max_samples:
                pad_len = self._max_samples - waveform.shape[0]
                waveform = torch.nn.functional.pad(waveform, (0, pad_len))
            else:
                waveform = waveform[: self._max_samples]

        return waveform.float()

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]

        dialogue_id = int(row["dialogue_id"])
        utterance_id = int(row["utterance_id"])

        waveform = self._load_audio(dialogue_id, utterance_id)

        emotion_label = EMOTION2IDX.get(row["emotion"], 0)
        sentiment_label = SENTIMENT2IDX.get(row["sentiment"], 0)

        sample: Dict = {
            "waveform": waveform,                              # (T,) float32
            "emotion_label": torch.tensor(emotion_label, dtype=torch.long),
            "sentiment_label": torch.tensor(sentiment_label, dtype=torch.long),
            "dialogue_id": dialogue_id,
            "utterance_id": utterance_id,
        }

        if self.return_text:
            utterance_col = "utterance" if "utterance" in self.df.columns else None
            if utterance_col:
                sample["text"] = str(row[utterance_col])

        return sample


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def collate_meld(batch: List[Dict]) -> Dict:
    """Collate a list of MELD samples into a batched dictionary.

    Waveforms are padded to the length of the longest waveform in the batch
    so that they can be stacked into a 2-D tensor of shape ``(B, T_max)``.

    Parameters
    ----------
    batch:
        List of sample dictionaries returned by :class:`MELDDataset`.

    Returns
    -------
    dict with keys:
        - ``waveforms``:       ``(B, T_max)`` float32 tensor
        - ``attention_mask``:  ``(B, T_max)`` bool tensor (True = real sample)
        - ``emotion_labels``:  ``(B,)`` long tensor
        - ``sentiment_labels``:``(B,)`` long tensor
        - ``dialogue_ids``:    list of ints
        - ``utterance_ids``:   list of ints
        - ``texts``:           list of str (optional, only if present)
    """
    waveforms: List[Tensor] = [item["waveform"] for item in batch]
    lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)
    max_len = int(lengths.max().item())

    padded = torch.zeros(len(batch), max_len, dtype=torch.float32)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (wav, length) in enumerate(zip(waveforms, lengths)):
        padded[i, : length.item()] = wav
        attention_mask[i, : length.item()] = True

    collated: Dict = {
        "waveforms": padded,
        "attention_mask": attention_mask,
        "emotion_labels": torch.stack(
            [item["emotion_label"] for item in batch]
        ),
        "sentiment_labels": torch.stack(
            [item["sentiment_label"] for item in batch]
        ),
        "dialogue_ids": [item["dialogue_id"] for item in batch],
        "utterance_ids": [item["utterance_id"] for item in batch],
    }

    if "text" in batch[0]:
        collated["texts"] = [item["text"] for item in batch]

    return collated


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def get_meld_dataloaders(
    root_dir: str | os.PathLike,
    batch_size: int = 16,
    num_workers: int = 4,
    max_duration_sec: float = 10.0,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> Tuple:
    """Build train / dev / test DataLoaders for MELD.

    Returns
    -------
    (train_loader, dev_loader, test_loader)
    """
    from torch.utils.data import DataLoader

    datasets = {
        split: MELDDataset(
            root_dir=root_dir,
            split=split,
            max_duration_sec=max_duration_sec,
            target_sample_rate=target_sample_rate,
        )
        for split in ("train", "dev", "test")
    }

    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collate_meld,
            pin_memory=True,
        )
        for split, ds in datasets.items()
    }

    return loaders["train"], loaders["dev"], loaders["test"]
