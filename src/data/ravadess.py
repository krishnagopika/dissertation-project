"""
RAVDESS Dataset Loader
======================
Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).
Contains acted emotional speech recordings from 24 professional actors.

Filename convention
-------------------
Fields are separated by hyphens (1-based indexing):

    Modality  - Voice channel - Emotion - Intensity - Statement - Repetition - Actor

Example: ``03-01-06-01-02-01-12.wav``
    * Field 1 (modality):    03 = audio-only speech
    * Field 2 (channel):     01 = speech
    * Field 3 (emotion):     06 = fearful
    * Field 4 (intensity):   01 = normal
    * Field 5 (statement):   02 = second statement
    * Field 6 (repetition):  01 = first repetition
    * Field 7 (actor):       12 = Actor 12

Emotion codes (field 3)
-----------------------
    01 = neutral
    02 = calm
    03 = happy
    04 = sad
    05 = angry
    06 = fearful
    07 = disgust
    08 = surprised

Expected directory layout::

    <root>/
        Actor_01/
            03-01-01-01-01-01-01.wav
            ...
        Actor_02/
            ...
        ...
        Actor_24/
            ...

Reference
---------
Livingstone & Russo, 2018. The Ryerson Audio-Visual Database of Emotional
Speech and Song (RAVDESS). PLOS ONE.
https://doi.org/10.1371/journal.pone.0196391
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------

EMOTION2IDX: Dict[str, int] = {
    "neutral": 0,
    "calm": 1,
    "happy": 2,
    "sad": 3,
    "angry": 4,
    "fearful": 5,
    "disgust": 6,
    "surprised": 7,
}

IDX2EMOTION: Dict[int, str] = {v: k for k, v in EMOTION2IDX.items()}

# RAVDESS uses 1-based emotion codes
_RAVDESS_CODE2EMOTION: Dict[int, str] = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}

TARGET_SAMPLE_RATE: int = 16_000


def _parse_emotion_from_filename(path: Path) -> Optional[int]:
    """Extract the RAVDESS emotion index (0-based) from a filename.

    Parameters
    ----------
    path:
        Path to the ``.wav`` file. The stem must follow the RAVDESS naming
        convention (hyphen-separated fields).

    Returns
    -------
    int or None:
        0-based emotion class index, or ``None`` if parsing fails.
    """
    try:
        parts = path.stem.split("-")
        emotion_code = int(parts[2])  # 3rd field, 1-based
        emotion_str = _RAVDESS_CODE2EMOTION.get(emotion_code)
        if emotion_str is None:
            return None
        return EMOTION2IDX[emotion_str]
    except (IndexError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class RAVDESSDataset(Dataset):
    """PyTorch Dataset for RAVDESS.

    Parameters
    ----------
    file_paths:
        List of paths to ``.wav`` files that belong to this split.
    max_duration_sec:
        Maximum audio duration. Clips are padded or truncated to this length.
        Pass ``None`` to skip fixed-length handling.
    target_sample_rate:
        Target sample rate in Hz. Defaults to 16 kHz.
    """

    def __init__(
        self,
        file_paths: List[Path],
        max_duration_sec: Optional[float] = 5.0,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> None:
        self.file_paths = file_paths
        self.max_duration_sec = max_duration_sec
        self.target_sample_rate = target_sample_rate

        self._max_samples: Optional[int] = (
            int(max_duration_sec * target_sample_rate)
            if max_duration_sec is not None
            else None
        )

        # Pre-parse labels to filter out files with unrecognised names
        self.records: List[Tuple[Path, int]] = []
        for fp in file_paths:
            label = _parse_emotion_from_filename(fp)
            if label is not None:
                self.records.append((fp, label))

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

    def _load_audio(self, path: Path) -> Tensor:
        """Load, resample, convert to mono, and optionally pad/truncate.

        Returns a 1-D float32 tensor of shape ``(num_samples,)``.
        """
        if not path.exists():
            num_samples = self._max_samples or self.target_sample_rate
            return torch.zeros(num_samples, dtype=torch.float32)

        waveform, sample_rate = torchaudio.load(str(path))  # (C, T)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sample_rate != self.target_sample_rate:
            resampler = self._get_resampler(sample_rate)
            waveform = resampler(waveform)

        waveform = waveform.squeeze(0)  # (T,)

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
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        path, emotion_label = self.records[idx]
        waveform = self._load_audio(path)

        return {
            "waveform": waveform,
            "emotion_label": torch.tensor(emotion_label, dtype=torch.long),
            "file_path": str(path),
        }


# ---------------------------------------------------------------------------
# Split factory
# ---------------------------------------------------------------------------


def get_splits(
    root_dir: str | os.PathLike,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    max_duration_sec: Optional[float] = 5.0,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> Tuple[RAVDESSDataset, RAVDESSDataset, RAVDESSDataset]:
    """Discover all RAVDESS audio files and split them into train/val/test.

    Splitting is performed at the **file** level with a fixed random seed so
    results are reproducible. Actor identity is *not* used as a hard boundary
    by default; pass stratified actor splits manually if needed for
    speaker-independent evaluation.

    Parameters
    ----------
    root_dir:
        Root directory containing ``Actor_01/`` … ``Actor_24/`` subdirectories.
    train_ratio:
        Fraction of files allocated to the training set.
    val_ratio:
        Fraction of files allocated to the validation set.
        The remainder goes to the test set.
    seed:
        Random seed for reproducibility.
    max_duration_sec:
        Passed to :class:`RAVDESSDataset`.
    target_sample_rate:
        Passed to :class:`RAVDESSDataset`.

    Returns
    -------
    (train_dataset, val_dataset, test_dataset)
    """
    root = Path(root_dir)

    # Collect all .wav files in Actor_*/ subdirs
    all_files: List[Path] = sorted(root.glob("Actor_*/*.wav"))
    if not all_files:
        raise FileNotFoundError(
            f"No .wav files found under {root}/Actor_*/. "
            "Please check the root_dir path."
        )

    rng = random.Random(seed)
    shuffled = list(all_files)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_files = shuffled[:n_train]
    val_files = shuffled[n_train : n_train + n_val]
    test_files = shuffled[n_train + n_val :]

    train_ds = RAVDESSDataset(train_files, max_duration_sec, target_sample_rate)
    val_ds = RAVDESSDataset(val_files, max_duration_sec, target_sample_rate)
    test_ds = RAVDESSDataset(test_files, max_duration_sec, target_sample_rate)

    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def collate_ravdess(batch: List[Dict]) -> Dict:
    """Collate RAVDESS samples with dynamic padding.

    Parameters
    ----------
    batch:
        List of sample dicts returned by :class:`RAVDESSDataset`.

    Returns
    -------
    dict with keys:
        - ``waveforms``:      ``(B, T_max)`` float32 tensor
        - ``attention_mask``: ``(B, T_max)`` bool tensor
        - ``emotion_labels``: ``(B,)`` long tensor
        - ``file_paths``:     list of str
    """
    waveforms: List[Tensor] = [item["waveform"] for item in batch]
    lengths = torch.tensor([w.shape[0] for w in waveforms], dtype=torch.long)
    max_len = int(lengths.max().item())

    padded = torch.zeros(len(batch), max_len, dtype=torch.float32)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, (wav, length) in enumerate(zip(waveforms, lengths)):
        padded[i, : length.item()] = wav
        attention_mask[i, : length.item()] = True

    return {
        "waveforms": padded,
        "attention_mask": attention_mask,
        "emotion_labels": torch.stack(
            [item["emotion_label"] for item in batch]
        ),
        "file_paths": [item["file_path"] for item in batch],
    }


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def get_ravdess_dataloaders(
    root_dir: str | os.PathLike,
    batch_size: int = 16,
    num_workers: int = 4,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    max_duration_sec: float = 5.0,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> Tuple:
    """Build train / val / test DataLoaders for RAVDESS.

    Returns
    -------
    (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader

    train_ds, val_ds, test_ds = get_splits(
        root_dir=root_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        max_duration_sec=max_duration_sec,
        target_sample_rate=target_sample_rate,
    )

    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_ravdess,
        pin_memory=True,
    )

    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader
