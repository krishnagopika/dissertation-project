"""Dataset and collation for fusion training on cached sequences.

Replaces the pooled-vector dataset in train_fusion.py. Three things change:

1. **Acoustic input is a SEQUENCE**, ``(T, 1280)``, not a pooled vector. Pooling
   moved into the trainable head (docs/DECISIONS.md ADR-002/003), so the cache
   holds frames and the model decides how to collapse them.

2. **Missing data fails loudly.** The previous implementation substituted a zero
   vector for any key absent from the embedding file:

       text_emb = text_embeddings.get(key, torch.zeros(text_dim))

   A zero vector is indistinguishable from a real embedding to the model, so a
   failed extraction became a silent training example with a valid label. This
   module counts every miss, reports them, and refuses to build a dataset whose
   miss rate exceeds a threshold.

3. **Unknown labels raise.** Previously ``EMOTION2IDX.get(row["emotion"], 0)``
   mapped any unrecognised string to *neutral*, and the sentiment equivalent
   mapped to *neutral* as well. A typo or an unexpected label silently became a
   majority-class training example.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.evaluation.metrics import EMOTION_NAMES, SENTIMENT_NAMES

EMOTION2IDX: Dict[str, int] = {n: i for i, n in enumerate(EMOTION_NAMES)}
SENTIMENT2IDX: Dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}

_SPLIT_CSV: Dict[str, str] = {
    "train": "train_sent_emo.csv",
    "dev": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv",
}

#: Refuse to build a dataset if more than this fraction of utterances is missing
#: an embedding. A handful of misses is a data defect to report; a large
#: fraction means the wrong file is being read, which should stop the run.
_MAX_MISS_RATE = 0.02


class FusionSequenceDataset(Dataset):
    """Serves (acoustic sequence, text embedding, labels) per utterance.

    Args:
        meld_root: MELD root directory containing the split CSVs.
        text_embeddings_path: Directory with ``{split}_text_embeddings.pt``.
        acoustic_seq_path: Directory with ``{split}_acoustic_seq.pt``.
        split: One of 'train', 'dev', 'test'.
        text_dim: Expected text embedding width; validated, not assumed.
        acoustic_dim: Expected acoustic frame width; validated, not assumed.
        filtered_keys_path: Optional keep-list JSON with a ``keys`` field.
        max_frames: Optional cap on sequence length. ``None`` keeps full length.
        logger: Logger for the data-integrity report.

    Raises:
        ValueError: On an unknown split, an unrecognised label, a dimension
            mismatch, or a miss rate above ``_MAX_MISS_RATE``.
        FileNotFoundError: If a required artefact is absent.
    """

    def __init__(
        self,
        meld_root: str,
        text_embeddings_path: str,
        acoustic_seq_path: str,
        split: str,
        text_dim: int = 768,
        acoustic_dim: int = 1280,
        filtered_keys_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        # Not an assert: assertions are stripped under python -O, so input
        # validation written that way silently disappears in optimised runs.
        if split not in _SPLIT_CSV:
            raise ValueError(f"split must be train/dev/test, got {split!r}")

        self.split = split
        self.max_frames = max_frames
        log = logger or logging.getLogger(__name__)

        csv_path = Path(meld_root) / _SPLIT_CSV[split]
        if not csv_path.exists():
            raise FileNotFoundError(f"MELD CSV not found: {csv_path}")
        df = pd.read_csv(csv_path)
        df.columns = (df.columns.str.strip().str.lower()
                      .str.replace(" ", "_", regex=False))
        df = df.dropna(subset=["emotion", "sentiment"]).reset_index(drop=True)
        df["emotion"] = df["emotion"].str.strip().str.lower()
        df["sentiment"] = df["sentiment"].str.strip().str.lower()

        text_file = Path(text_embeddings_path) / f"{split}_text_embeddings.pt"
        seq_file = Path(acoustic_seq_path) / f"{split}_acoustic_seq.pt"
        for f, hint in ((text_file, "extract_text_embeddings_asr.py"),
                        (seq_file, "transcribe_all.py")):
            if not f.exists():
                raise FileNotFoundError(f"{f} not found. Run {hint} first.")

        text_emb: Dict[str, Tensor] = torch.load(str(text_file), map_location="cpu")
        acoustic_seq: Dict[str, Tensor] = torch.load(str(seq_file), map_location="cpu")

        keep_set: Optional[set] = None
        if filtered_keys_path is not None:
            p = Path(filtered_keys_path)
            if not p.exists():
                raise FileNotFoundError(f"filtered_keys_path not found: {p}")
            with open(p, encoding="utf-8") as f:
                keep_set = set(json.load(f)["keys"])

        self.num_before_filter = len(df)
        self.samples: List[Tuple[str, Tensor, Tensor, int, int]] = []
        missing_text: List[str] = []
        missing_acoustic: List[str] = []
        zero_acoustic: List[str] = []
        n_considered = 0

        for _, row in df.iterrows():
            key = f"dia{int(row['dialogue_id'])}_utt{int(row['utterance_id'])}"
            if keep_set is not None and key not in keep_set:
                continue
            n_considered += 1

            # Unknown labels raise. Mapping them to a default silently converts
            # a data error into a majority-class training example.
            emo = row["emotion"]
            sen = row["sentiment"]
            if emo not in EMOTION2IDX:
                raise ValueError(
                    f"{key}: unrecognised emotion {emo!r}. "
                    f"Expected one of {sorted(EMOTION2IDX)}."
                )
            if sen not in SENTIMENT2IDX:
                raise ValueError(
                    f"{key}: unrecognised sentiment {sen!r}. "
                    f"Expected one of {sorted(SENTIMENT2IDX)}."
                )

            t = text_emb.get(key)
            a = acoustic_seq.get(key)
            if t is None:
                missing_text.append(key)
                continue
            if a is None:
                missing_acoustic.append(key)
                continue

            # An all-zero sequence is what the extractor writes when its hook
            # missed the clip. It is not a valid representation, and it is
            # invisible downstream, so exclude it rather than train on it.
            if a.shape[0] <= 1 and float(a.abs().sum()) == 0.0:
                zero_acoustic.append(key)
                continue

            if t.shape[-1] != text_dim:
                raise ValueError(
                    f"{key}: text embedding is {t.shape[-1]}-d, expected "
                    f"{text_dim}. Wrong file, or the config disagrees with it."
                )
            if a.shape[-1] != acoustic_dim:
                raise ValueError(
                    f"{key}: acoustic frames are {a.shape[-1]}-d, expected "
                    f"{acoustic_dim}."
                )

            self.samples.append(
                (key, t.float(), a, EMOTION2IDX[emo], SENTIMENT2IDX[sen])
            )

        self.num_after_filter = len(self.samples)
        self.missing_text = missing_text
        self.missing_acoustic = missing_acoustic
        self.zero_acoustic = zero_acoustic

        n_missing = len(missing_text) + len(missing_acoustic) + len(zero_acoustic)
        miss_rate = n_missing / max(1, n_considered)
        log.info(
            "%s | %d usable / %d considered | missing text %d, missing acoustic "
            "%d, zero-filled acoustic %d (%.2f%%)",
            split, len(self.samples), n_considered, len(missing_text),
            len(missing_acoustic), len(zero_acoustic), 100 * miss_rate,
        )
        for name, keys in (("text", missing_text), ("acoustic", missing_acoustic),
                           ("zero-filled", zero_acoustic)):
            if keys:
                log.warning("%s | first missing %s keys: %s",
                            split, name, keys[:5])

        if miss_rate > _MAX_MISS_RATE:
            raise ValueError(
                f"{split}: {100*miss_rate:.1f}% of utterances lack a usable "
                f"embedding (limit {100*_MAX_MISS_RATE:.0f}%). This usually "
                "means the wrong cache directory is configured, or extraction "
                "did not complete. Refusing to train on a partial dataset."
            )

        frames = [s[2].shape[0] for s in self.samples]
        if frames:
            frames_sorted = sorted(frames)
            log.info("%s | frames per clip: min %d, median %d, max %d",
                     split, frames_sorted[0],
                     frames_sorted[len(frames_sorted) // 2], frames_sorted[-1])

    def label_counts(self) -> Counter:
        """Emotion label distribution, for class weighting and reporting."""
        return Counter(EMOTION_NAMES[s[3]] for s in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        key, text, acoustic, emo, sen = self.samples[idx]
        if self.max_frames is not None and acoustic.shape[0] > self.max_frames:
            acoustic = acoustic[: self.max_frames]
        return {
            "key": key,
            "text": text,
            "acoustic": acoustic,
            "emotion_label": torch.tensor(emo, dtype=torch.long),
            "sentiment_label": torch.tensor(sen, dtype=torch.long),
        }


def collate_sequences(batch: List[Dict], skip_acoustic: bool = False) -> Dict:
    """Pad variable-length acoustic sequences and build the frame mask.

    The pooling modules require a mask so that padding is excluded from the
    weighted sum; a fully-padded row would otherwise pool pure silence. Every
    row here has at least one real frame because the dataset excludes
    zero-length and all-zero sequences.

    Args:
        batch: Items from :class:`FusionSequenceDataset`.

    Returns:
        Dict with ``acoustic`` ``(B, T_max, D)``, ``acoustic_mask``
        ``(B, T_max)`` bool (True = real frame), ``text`` ``(B, D_t)``,
        both label tensors, and the list of keys for provenance.
    """
    if skip_acoustic:
        # A text-only run would otherwise pad every sequence to the batch
        # maximum, move ~1500x1280 floats per sample to the GPU, and discard
        # them inside the model. Emit a placeholder instead.
        return {
            "keys": [i["key"] for i in batch],
            "text": torch.stack([i["text"] for i in batch]),
            "acoustic": torch.zeros(len(batch), 1, 1, dtype=torch.float32),
            "acoustic_mask": torch.ones(len(batch), 1, dtype=torch.bool),
            "emotion_label": torch.stack([i["emotion_label"] for i in batch]),
            "sentiment_label": torch.stack([i["sentiment_label"] for i in batch]),
        }

    lengths = [item["acoustic"].shape[0] for item in batch]
    t_max = max(lengths)
    dim = batch[0]["acoustic"].shape[1]

    acoustic = torch.zeros(len(batch), t_max, dim, dtype=torch.float32)
    mask = torch.zeros(len(batch), t_max, dtype=torch.bool)
    for i, item in enumerate(batch):
        n = lengths[i]
        acoustic[i, :n] = item["acoustic"].float()
        mask[i, :n] = True

    return {
        "keys": [item["key"] for item in batch],
        "text": torch.stack([item["text"] for item in batch]),
        "acoustic": acoustic,
        "acoustic_mask": mask,
        "emotion_label": torch.stack([item["emotion_label"] for item in batch]),
        "sentiment_label": torch.stack([item["sentiment_label"] for item in batch]),
    }
