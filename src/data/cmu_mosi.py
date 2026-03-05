"""
CMU-MOSI Dataset Loader
=======================
CMU Multimodal Opinion Sentiment Intensity (CMU-MOSI) — opinion video segments
with continuous sentiment scores in [-3, +3].

Expected directory layout (primary path — labels.csv mode)::

    <root>/
        labels.csv          # columns: video_id, segment_id, label, split
        audio/
            <video_id>_<segment_id>.wav
            ...

Fallback path (cPickle mode)::

    <root>/
        CMU_MOSI_Opinion_Labels.cPickle
        audio/
            <video_id>_<segment_id>.wav
            ...

The cPickle file is expected to contain a dictionary with structure::

    {
        "train": {<video_id>: [label_0, label_1, ...]},
        "valid": {<video_id>: [label_0, label_1, ...]},
        "test":  {<video_id>: [label_0, label_1, ...]},
    }

or the raw SDK format produced by CMU-MultimodalSDK.

Binning convention
------------------
    label < -1.0  => negative  (class 0)
    -1.0 <= label <= 1.0 => neutral   (class 1)
    label > 1.0   => positive  (class 2)

Reference
---------
Zadeh et al., 2016. MOSI: Multimodal Sentiment Intensity Analysis of Videos.
https://arxiv.org/abs/1606.06259
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Label mappings
# ---------------------------------------------------------------------------

SENTIMENT2IDX: Dict[str, int] = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

IDX2SENTIMENT: Dict[int, str] = {v: k for k, v in SENTIMENT2IDX.items()}

TARGET_SAMPLE_RATE: int = 16_000


def bin_sentiment(score: float) -> int:
    """Convert a continuous MOSI score into a 3-class sentiment label.

    Parameters
    ----------
    score:
        Raw annotation value in [-3, +3].

    Returns
    -------
    int:
        0 = negative, 1 = neutral, 2 = positive.
    """
    if score < -1.0:
        return SENTIMENT2IDX["negative"]
    elif score > 1.0:
        return SENTIMENT2IDX["positive"]
    else:
        return SENTIMENT2IDX["neutral"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CMUMOSIDataset(Dataset):
    """PyTorch Dataset for CMU-MOSI.

    Parameters
    ----------
    root_dir:
        Root directory of the dataset. Must contain either ``labels.csv``
        or ``CMU_MOSI_Opinion_Labels.cPickle``, and an ``audio/`` subdirectory.
    split:
        One of ``"train"``, ``"valid"`` / ``"dev"`` or ``"test"``.
        The string ``"dev"`` is aliased to ``"valid"`` automatically.
    max_duration_sec:
        Maximum audio duration. Waveforms are padded or truncated to this
        length in seconds. Pass ``None`` to skip fixed-length padding.
    target_sample_rate:
        Target sample rate in Hz. Defaults to 16 kHz.
    return_raw_score:
        If ``True``, include the original continuous score in the sample dict.
    """

    _PICKLE_FILENAME = "CMU_MOSI_Opinion_Labels.cPickle"
    _CSV_FILENAME = "labels.csv"

    def __init__(
        self,
        root_dir: str | os.PathLike,
        split: str = "train",
        max_duration_sec: Optional[float] = 10.0,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        return_raw_score: bool = False,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = "valid" if split in ("dev", "valid") else split
        self.max_duration_sec = max_duration_sec
        self.target_sample_rate = target_sample_rate
        self.return_raw_score = return_raw_score

        self._max_samples: Optional[int] = (
            int(max_duration_sec * target_sample_rate)
            if max_duration_sec is not None
            else None
        )

        self.audio_dir = self.root_dir / "audio"

        # Attempt to load from labels.csv first, then cPickle
        csv_path = self.root_dir / self._CSV_FILENAME
        pickle_path = self.root_dir / self._PICKLE_FILENAME

        if csv_path.exists():
            self.records = self._load_from_csv(csv_path)
        elif pickle_path.exists():
            self.records = self._load_from_pickle(pickle_path)
        else:
            raise FileNotFoundError(
                f"No label file found in {self.root_dir}. "
                f"Expected '{self._CSV_FILENAME}' or '{self._PICKLE_FILENAME}'."
            )

        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def _load_from_csv(self, path: Path) -> List[Dict]:
        """Load records from a labels.csv file.

        Expected columns: video_id, segment_id, label, split.
        """
        df = pd.read_csv(path)
        df.columns = (
            df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
        )
        required = {"video_id", "segment_id", "label", "split"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"labels.csv is missing columns: {missing}")

        df = df[df["split"].str.lower() == self.split].reset_index(drop=True)
        records = [
            {
                "video_id": str(row["video_id"]),
                "segment_id": str(row["segment_id"]),
                "score": float(row["label"]),
            }
            for _, row in df.iterrows()
        ]
        return records

    def _load_from_pickle(self, path: Path) -> List[Dict]:
        """Load records from the CMU-MultimodalSDK cPickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        records: List[Dict] = []

        # Handle two common dict layouts from CMU-MultimodalSDK
        split_data = None
        if isinstance(data, dict):
            # Layout 1: {split: {video_id: [scores]}}
            if self.split in data:
                split_data = data[self.split]
            # Layout 2: {video_id: {"train": [...], ...}}
            else:
                # Flatten by looking inside each video entry
                for video_id, vid_dict in data.items():
                    if not isinstance(vid_dict, dict):
                        continue
                    split_scores = vid_dict.get(self.split, [])
                    for seg_idx, score in enumerate(split_scores):
                        if score is not None:
                            records.append(
                                {
                                    "video_id": str(video_id),
                                    "segment_id": str(seg_idx),
                                    "score": float(score),
                                }
                            )
                return records

        if split_data is None:
            raise ValueError(
                f"Cannot find split '{self.split}' in the cPickle file. "
                f"Available keys: {list(data.keys())}"
            )

        for video_id, scores in split_data.items():
            for seg_idx, score in enumerate(scores):
                if score is not None:
                    records.append(
                        {
                            "video_id": str(video_id),
                            "segment_id": str(seg_idx),
                            "score": float(score),
                        }
                    )
        return records

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_resampler(self, orig_sr: int) -> torchaudio.transforms.Resample:
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=self.target_sample_rate
            )
        return self._resamplers[orig_sr]

    def _load_audio(self, video_id: str, segment_id: str) -> Tensor:
        """Load, resample, convert to mono, and optionally pad/truncate.

        Returns a 1-D float32 tensor of shape ``(num_samples,)``.
        """
        filename = f"{video_id}_{segment_id}.wav"
        audio_path = self.audio_dir / filename

        if not audio_path.exists():
            num_samples = self._max_samples or self.target_sample_rate
            return torch.zeros(num_samples, dtype=torch.float32)

        waveform, sample_rate = torchaudio.load(str(audio_path))  # (C, T)

        # Mono downmix
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample
        if sample_rate != self.target_sample_rate:
            resampler = self._get_resampler(sample_rate)
            waveform = resampler(waveform)

        waveform = waveform.squeeze(0)  # (T,)

        # Pad / truncate
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
        record = self.records[idx]
        video_id = record["video_id"]
        segment_id = record["segment_id"]
        score = record["score"]

        waveform = self._load_audio(video_id, segment_id)
        sentiment_label = bin_sentiment(score)

        sample: Dict = {
            "waveform": waveform,
            "sentiment_label": torch.tensor(sentiment_label, dtype=torch.long),
            "video_id": video_id,
            "segment_id": segment_id,
        }

        if self.return_raw_score:
            sample["raw_score"] = torch.tensor(score, dtype=torch.float32)

        return sample


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------


def collate_cmu_mosi(batch: List[Dict]) -> Dict:
    """Collate CMU-MOSI samples with dynamic padding.

    Parameters
    ----------
    batch:
        List of sample dicts returned by :class:`CMUMOSIDataset`.

    Returns
    -------
    dict with keys:
        - ``waveforms``:        ``(B, T_max)`` float32 tensor
        - ``attention_mask``:   ``(B, T_max)`` bool tensor
        - ``sentiment_labels``: ``(B,)`` long tensor
        - ``video_ids``:        list of str
        - ``segment_ids``:      list of str
        - ``raw_scores``:       ``(B,)`` float32 tensor (optional)
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
        "sentiment_labels": torch.stack(
            [item["sentiment_label"] for item in batch]
        ),
        "video_ids": [item["video_id"] for item in batch],
        "segment_ids": [item["segment_id"] for item in batch],
    }

    if "raw_score" in batch[0]:
        collated["raw_scores"] = torch.stack(
            [item["raw_score"] for item in batch]
        )

    return collated


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def get_cmu_mosi_dataloaders(
    root_dir: str | os.PathLike,
    batch_size: int = 16,
    num_workers: int = 4,
    max_duration_sec: float = 10.0,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> Tuple:
    """Return train / valid / test DataLoaders for CMU-MOSI.

    Returns
    -------
    (train_loader, valid_loader, test_loader)
    """
    from torch.utils.data import DataLoader

    datasets = {
        split: CMUMOSIDataset(
            root_dir=root_dir,
            split=split,
            max_duration_sec=max_duration_sec,
            target_sample_rate=target_sample_rate,
        )
        for split in ("train", "valid", "test")
    }

    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=collate_cmu_mosi,
            pin_memory=True,
        )
        for split, ds in datasets.items()
    }

    return loaders["train"], loaders["valid"], loaders["test"]
