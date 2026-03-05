"""
Dialect Robustness Evaluation Datasets
=======================================
Two datasets used to evaluate the pipeline's robustness to regional English
dialect variation:

1. :class:`CommonVoiceAUDataset`
   Loads Australian-accented speech from the Mozilla Common Voice 17.0
   dataset via HuggingFace Datasets, filtering by ``accent == "australia"``.

2. :class:`UKDialectsDataset`
   Loads British regional-dialect speech from a local directory tree
   organised by dialect region::

       <root>/
           southern/
               *.wav
           midlands/
               *.wav
           northern/
               *.wav
           welsh/
               *.wav
           scottish/
               *.wav

Both datasets resample audio to 16 kHz mono and expose a unified sample
dictionary so a single :func:`collate_dialect` function works for both.

Notes
-----
CommonVoice requires accepting the dataset's licence on HuggingFace Hub and
passing a valid ``token`` (a HuggingFace access token) for authenticated
download. Pass ``token=None`` to skip authentication (works if the dataset
is already cached locally).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

TARGET_SAMPLE_RATE: int = 16_000

# Recognised UK dialect sub-directories
UK_DIALECT_DIRS: List[str] = ["southern", "midlands", "northern", "welsh", "scottish"]

DIALECT2IDX: Dict[str, int] = {d: i for i, d in enumerate(UK_DIALECT_DIRS)}
IDX2DIALECT: Dict[int, str] = {i: d for d, i in DIALECT2IDX.items()}


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _load_and_resample(
    path: str | os.PathLike,
    target_sr: int,
    max_samples: Optional[int],
    resamplers: Dict[int, torchaudio.transforms.Resample],
) -> Tensor:
    """Load a waveform, convert to mono, resample, and optionally pad/truncate.

    Parameters
    ----------
    path:
        Path to the audio file.
    target_sr:
        Target sample rate.
    max_samples:
        If not ``None``, pad or truncate to this many samples.
    resamplers:
        Shared dict of cached :class:`torchaudio.transforms.Resample` objects.

    Returns
    -------
    Tensor of shape ``(num_samples,)``, float32.
    """
    path = Path(path)
    if not path.exists():
        fallback = max_samples if max_samples else target_sr
        return torch.zeros(fallback, dtype=torch.float32)

    waveform, sample_rate = torchaudio.load(str(path))  # (C, T)

    # Mono downmix
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample
    if sample_rate != target_sr:
        if sample_rate not in resamplers:
            resamplers[sample_rate] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=target_sr
            )
        waveform = resamplers[sample_rate](waveform)

    waveform = waveform.squeeze(0)  # (T,)

    if max_samples is not None:
        if waveform.shape[0] < max_samples:
            pad_len = max_samples - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        else:
            waveform = waveform[:max_samples]

    return waveform.float()


# ---------------------------------------------------------------------------
# 1. Common Voice Australian English
# ---------------------------------------------------------------------------


class CommonVoiceAUDataset(Dataset):
    """Australian-accent speech from Mozilla Common Voice 17.0.

    Audio is streamed or downloaded via HuggingFace Datasets. Only rows
    where the ``accent`` field matches ``"australia"`` (case-insensitive) are
    retained.

    Parameters
    ----------
    split:
        Dataset split to load. Typical values: ``"train"``, ``"validation"``,
        ``"test"``, ``"other"``.
    token:
        HuggingFace access token required to download Common Voice. Pass
        ``None`` if the dataset is already cached.
    max_duration_sec:
        Maximum audio duration in seconds. Longer clips are truncated; shorter
        ones are zero-padded. Pass ``None`` to disable fixed-length padding.
    target_sample_rate:
        Target sample rate (Hz). Defaults to 16 kHz.
    cache_dir:
        Optional local directory for HuggingFace cache.
    streaming:
        If ``True``, use HuggingFace streaming mode (avoids downloading the
        full dataset to disk). Note: streaming disables random-access
        indexing; use :meth:`__iter__` instead of :meth:`__getitem__` in
        that case, or set ``streaming=False`` for DataLoader compatibility.
    """

    _DATASET_ID = "mozilla-foundation/common_voice_17_0"
    _LANGUAGE_CODE = "en"

    def __init__(
        self,
        split: str = "test",
        token: Optional[str] = None,
        max_duration_sec: Optional[float] = 15.0,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
        cache_dir: Optional[str] = None,
        streaming: bool = False,
    ) -> None:
        from datasets import load_dataset

        self.split = split
        self.target_sample_rate = target_sample_rate
        self.max_duration_sec = max_duration_sec
        self._max_samples: Optional[int] = (
            int(max_duration_sec * target_sample_rate)
            if max_duration_sec is not None
            else None
        )
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

        hf_dataset = load_dataset(
            self._DATASET_ID,
            self._LANGUAGE_CODE,
            split=split,
            token=token,
            cache_dir=cache_dir,
            streaming=streaming,
            trust_remote_code=True,
        )

        if streaming:
            # In streaming mode we cannot filter eagerly; store a reference
            # and filter lazily in __iter__. The dataset is not indexable.
            self._streaming = True
            self._hf_dataset = hf_dataset
            self._records: Optional[List[Dict]] = None
        else:
            self._streaming = False
            # Filter to Australian accent eagerly
            filtered = hf_dataset.filter(
                lambda ex: (ex.get("accent") or "").lower().strip() == "australia"
            )
            self._records = list(filtered)

    # ------------------------------------------------------------------
    # Dataset API  (non-streaming path only)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        if self._streaming:
            raise TypeError(
                "CommonVoiceAUDataset is in streaming mode; __len__ is unavailable."
            )
        return len(self._records)

    def __getitem__(self, idx: int) -> Dict:
        if self._streaming:
            raise TypeError(
                "CommonVoiceAUDataset is in streaming mode; use __iter__ instead."
            )
        record = self._records[idx]
        return self._record_to_sample(record)

    def __iter__(self):
        """Iterate over Australian-accent samples (works in both modes)."""
        if self._streaming:
            for record in self._hf_dataset:
                if (record.get("accent") or "").lower().strip() == "australia":
                    yield self._record_to_sample(record)
        else:
            for record in self._records:
                yield self._record_to_sample(record)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_to_sample(self, record: Dict) -> Dict:
        """Convert a HuggingFace Common Voice record to a sample dict."""
        # HuggingFace Common Voice stores audio as a dict with 'array' and
        # 'sampling_rate' keys when decoded, or as a path when not decoded.
        audio_field = record.get("audio", {})

        if isinstance(audio_field, dict) and "array" in audio_field:
            # Decoded audio
            import numpy as np

            array = audio_field["array"]
            orig_sr = audio_field["sampling_rate"]
            waveform = torch.from_numpy(np.array(array, dtype="float32")).unsqueeze(0)

            if orig_sr != self.target_sample_rate:
                if orig_sr not in self._resamplers:
                    self._resamplers[orig_sr] = torchaudio.transforms.Resample(
                        orig_freq=orig_sr, new_freq=self.target_sample_rate
                    )
                waveform = self._resamplers[orig_sr](waveform)

            waveform = waveform.squeeze(0)

            if self._max_samples is not None:
                if waveform.shape[0] < self._max_samples:
                    waveform = torch.nn.functional.pad(
                        waveform, (0, self._max_samples - waveform.shape[0])
                    )
                else:
                    waveform = waveform[: self._max_samples]

        elif isinstance(audio_field, dict) and "path" in audio_field:
            # Path-based loading
            waveform = _load_and_resample(
                audio_field["path"],
                self.target_sample_rate,
                self._max_samples,
                self._resamplers,
            )
        else:
            fallback = self._max_samples or self.target_sample_rate
            waveform = torch.zeros(fallback, dtype=torch.float32)

        return {
            "waveform": waveform.float(),
            "transcript": record.get("sentence", ""),
            "accent": record.get("accent", "australia"),
            "dataset": "common_voice_au",
        }


# ---------------------------------------------------------------------------
# 2. UK Dialects Dataset (local files)
# ---------------------------------------------------------------------------


class UKDialectsDataset(Dataset):
    """British regional dialect speech from a local directory tree.

    Expected layout::

        <root>/
            southern/   *.wav
            midlands/   *.wav
            northern/   *.wav
            welsh/      *.wav
            scottish/   *.wav

    Parameters
    ----------
    root_dir:
        Root directory containing the dialect sub-directories.
    dialects:
        List of dialect names (sub-directory names) to include. Defaults to
        all five recognised dialects.
    max_duration_sec:
        Maximum audio duration. Clips are padded or truncated to this length.
    target_sample_rate:
        Target sample rate (Hz).
    """

    def __init__(
        self,
        root_dir: str | os.PathLike,
        dialects: Optional[List[str]] = None,
        max_duration_sec: Optional[float] = 15.0,
        target_sample_rate: int = TARGET_SAMPLE_RATE,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.dialects = dialects if dialects is not None else UK_DIALECT_DIRS
        self.target_sample_rate = target_sample_rate
        self.max_duration_sec = max_duration_sec
        self._max_samples: Optional[int] = (
            int(max_duration_sec * target_sample_rate)
            if max_duration_sec is not None
            else None
        )
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

        # Discover all .wav files
        self.records: List[Tuple[Path, str]] = []  # (path, dialect_name)
        for dialect in self.dialects:
            dialect_dir = self.root_dir / dialect
            if not dialect_dir.is_dir():
                continue
            for wav_file in sorted(dialect_dir.glob("*.wav")):
                self.records.append((wav_file, dialect))

        if not self.records:
            raise FileNotFoundError(
                f"No .wav files found in any of {self.dialects} under {self.root_dir}."
            )

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        path, dialect_name = self.records[idx]
        waveform = _load_and_resample(
            path,
            self.target_sample_rate,
            self._max_samples,
            self._resamplers,
        )
        dialect_label = DIALECT2IDX.get(dialect_name, -1)
        return {
            "waveform": waveform,
            "dialect": dialect_name,
            "dialect_label": torch.tensor(dialect_label, dtype=torch.long),
            "file_path": str(path),
            "dataset": "uk_dialects",
        }


# ---------------------------------------------------------------------------
# Shared collate function
# ---------------------------------------------------------------------------


def collate_dialect(batch: List[Dict]) -> Dict:
    """Collate dialect evaluation samples with dynamic waveform padding.

    Compatible with both :class:`CommonVoiceAUDataset` and
    :class:`UKDialectsDataset`.

    Parameters
    ----------
    batch:
        List of sample dicts.

    Returns
    -------
    dict with keys:
        - ``waveforms``:      ``(B, T_max)`` float32 tensor
        - ``attention_mask``: ``(B, T_max)`` bool tensor
        - ``dialects``:       list of str  (present if ``"dialect"`` in batch)
        - ``dialect_labels``: ``(B,)`` long tensor (if present)
        - ``transcripts``:    list of str  (if present)
        - ``file_paths``:     list of str  (if present)
        - ``datasets``:       list of str
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
        "datasets": [item.get("dataset", "") for item in batch],
    }

    if "dialect" in batch[0]:
        collated["dialects"] = [item["dialect"] for item in batch]
        collated["dialect_labels"] = torch.stack(
            [item["dialect_label"] for item in batch]
        )

    if "transcript" in batch[0]:
        collated["transcripts"] = [item.get("transcript", "") for item in batch]

    if "file_path" in batch[0]:
        collated["file_paths"] = [item.get("file_path", "") for item in batch]

    return collated
