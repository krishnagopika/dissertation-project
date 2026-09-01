"""
context_lstm.py — bc-LSTM: bidirectional contextual LSTM over a dialogue
=========================================================================
Classic MELD context model (Poria et al., 2019 "bc-LSTM"). Instead of
classifying utterances in isolation, it runs a BiLSTM over the *sequence of
utterances in a conversation*, so each utterance's prediction is informed by
its neighbours (past and future). On MELD this is the single biggest lever —
emotion is heavily context-dependent ("Oh, great." is joy or anger depending
on the preceding turns).

Operates on **pre-cached per-utterance feature vectors** (e.g. concatenated
text + acoustic embeddings), so it is tiny and trains in minutes — no large
model is loaded.

Architecture
------------
::

    utterance features  (B, T, input_dim)   one vector per utterance, padded
            │  pack_padded_sequence (lengths)
    BiLSTM (num_layers, bidirectional)       each utterance sees its neighbours
            │  (B, T, 2*hidden_dim)
    Dropout
            │
    sentiment_head (B, T, 3)   emotion_head (B, T, 7)   predicted per utterance
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMContext(nn.Module):
    """Bidirectional contextual LSTM over a dialogue's utterance sequence.

    Args:
        input_dim: Per-utterance feature dimension (e.g. 768+1280=2048).
        hidden_dim: LSTM hidden size (per direction).
        num_emotion_classes: Number of emotion classes.
        num_sentiment_classes: Number of sentiment classes.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout applied between LSTM layers and before the heads.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 256,
        num_emotion_classes: int = 7,
        num_sentiment_classes: int = 3,
        num_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.sentiment_head = nn.Linear(hidden_dim * 2, num_sentiment_classes)
        self.emotion_head = nn.Linear(hidden_dim * 2, num_emotion_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for head in (self.sentiment_head, self.emotion_head):
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, features: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """Contextualise utterances and predict per-utterance logits.

        Args:
            features: Padded utterance features, shape ``(B, T, input_dim)``.
            lengths: Real (unpadded) dialogue lengths, shape ``(B,)``.

        Returns:
            Tuple of ``(sentiment_logits, emotion_logits)``, each ``(B, T, C)``.
        """
        packed = pack_padded_sequence(
            features, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(
            out_packed, batch_first=True, total_length=features.size(1)
        )
        out = self.dropout(out)
        return self.sentiment_head(out), self.emotion_head(out)

    def represent(self, features: Tensor, lengths: Tensor) -> Tensor:
        """Contextualised per-utterance states, before the heads.

        This is the vector the classification heads consume: the BiLSTM output
        after dropout. Exposed so the representation analysis can measure what
        dialogue context does to class structure, alongside the raw inputs and
        the fusion model's own representation.

        Note this is the CONTEXTUALISED state -- utterance i's vector already
        contains information from its neighbours. That is the point: comparing
        it against the pre-context input is what isolates the effect of
        context on linear separability.

        Args:
            features: Padded utterance features, ``(B, T, input_dim)``.
            lengths: Real (unpadded) dialogue lengths, ``(B,)``.

        Returns:
            ``(B, T, 2 * hidden_dim)``. Padded positions are present but
            meaningless -- mask with ``lengths`` before use.
        """
        packed = pack_padded_sequence(
            features, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out_packed, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(
            out_packed, batch_first=True, total_length=features.size(1)
        )
        return self.dropout(out)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
