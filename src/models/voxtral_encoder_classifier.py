"""
VoxtralEncoderClassifier — fine-tunable Whisper encoder + classification heads
==============================================================================
Wraps **Voxtral's audio encoder** (``model.audio_tower``, a Whisper-large-v3
encoder, 1280-dim @ 50 Hz) with a learned attention-pool and emotion/sentiment
heads, so the encoder itself can be **fine-tuned** for MELD emotion recognition.

Unlike the Phase-2 fusion path (which trains a tiny head on *frozen, cached*
mean-pooled embeddings), this module keeps the encoder in the training loop so
its representations adapt to emotion. The 3B language model is **not** loaded —
only the ~635M audio encoder — so backprop never touches the LLM.

Architecture
------------
::

    input_features (B, 128, 3000)   # log-mel spectrogram (Whisper format)
            │
    Voxtral audio_tower  (Whisper encoder; last N layers trainable)
            │  last_hidden_state (B, T≈1500, 1280)
    AttentionPool  (learned query over frames → replaces lossy mean-pool)
            │  (B, 1280)
    proj: Linear→LayerNorm→GELU→Dropout  (B, hidden_dim)
            │
    sentiment_head (B, 3)   emotion_head (B, 7)

Memory note
-----------
``VoxtralForConditionalGeneration.from_pretrained`` loads the whole checkpoint;
we immediately keep only ``audio_tower`` and drop the rest so the LLM is freed
before training starts.
"""

from __future__ import annotations

import gc
import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

_ACOUSTIC_DIM: int = 1280   # Whisper-large-v3 encoder hidden size
_HIDDEN_DIM: int = 512
_NUM_EMOTION: int = 7
_NUM_SENTIMENT: int = 3


class AttentionPool(nn.Module):
    """Single-query attention pooling over a sequence of frames.

    Learns a query vector that scores each frame; the output is the
    attention-weighted sum. Replaces mean-pooling, which weights every frame
    (including silence/padding) equally and washes out short emotional bursts.

    Args:
        dim: Feature dimension of the input frames.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * (dim ** -0.5))
        self.scale = dim ** -0.5

    def forward(self, frames: Tensor) -> Tensor:
        """Pool a frame sequence to a single vector.

        Args:
            frames: Tensor of shape ``(batch, seq_len, dim)``.

        Returns:
            Pooled tensor of shape ``(batch, dim)``.
        """
        scores = (frames @ self.query) * self.scale          # (B, T)
        attn = F.softmax(scores, dim=1).unsqueeze(-1)          # (B, T, 1)
        return (attn * frames).sum(dim=1)                      # (B, dim)


class VoxtralEncoderClassifier(nn.Module):
    """Fine-tunable Voxtral audio encoder with emotion/sentiment heads.

    Args:
        voxtral_id: HuggingFace ID or path for the Voxtral model.
        num_emotion_classes: Number of emotion output classes.
        num_sentiment_classes: Number of sentiment output classes.
        acoustic_dim: Encoder hidden dimension (1280 for Whisper-large-v3).
        hidden_dim: Hidden dimension of the projection before the heads.
        dropout_prob: Dropout probability in the projection.
        unfreeze_last_n: Number of final encoder layers to keep trainable.
            ``0`` freezes the whole encoder (head-only training); a small value
            (e.g. 2) adapts the top of the encoder without overfitting.
        cache_dir: Optional HuggingFace cache directory.
    """

    def __init__(
        self,
        voxtral_id: str = "mistralai/Voxtral-Mini-3B-2507",
        num_emotion_classes: int = _NUM_EMOTION,
        num_sentiment_classes: int = _NUM_SENTIMENT,
        acoustic_dim: int = _ACOUSTIC_DIM,
        hidden_dim: int = _HIDDEN_DIM,
        dropout_prob: float = 0.1,
        unfreeze_last_n: int = 2,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()

        from transformers import VoxtralForConditionalGeneration

        # Load full model (bf16 to save RAM), keep only the audio encoder.
        logger.info("Loading Voxtral to extract audio_tower: %s", voxtral_id)
        full = VoxtralForConditionalGeneration.from_pretrained(
            voxtral_id,
            dtype=torch.bfloat16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
        )
        # Keep the encoder in fp32 for stable fine-tuning; drop the LLM.
        self.encoder = full.audio_tower.float()
        del full
        gc.collect()
        torch.cuda.empty_cache()

        # Freeze encoder, then unfreeze the last N transformer layers.
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.unfreeze_last_n = int(unfreeze_last_n)
        if self.unfreeze_last_n > 0:
            layers = self._encoder_layers()
            for layer in layers[-self.unfreeze_last_n:]:
                for p in layer.parameters():
                    p.requires_grad = True
            logger.info(
                "Unfroze last %d of %d encoder layers",
                self.unfreeze_last_n, len(layers),
            )

        self.attn_pool = AttentionPool(acoustic_dim)
        self.proj = nn.Sequential(
            nn.Linear(acoustic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
        )
        self.sentiment_head = nn.Linear(hidden_dim, num_sentiment_classes)
        self.emotion_head = nn.Linear(hidden_dim, num_emotion_classes)

        self._init_new_weights()

    def _encoder_layers(self) -> nn.ModuleList:
        """Return the encoder's transformer layer list (handles API variants)."""
        enc = self.encoder
        for attr in ("layers",):
            if hasattr(enc, attr):
                return getattr(enc, attr)
        # Some implementations nest layers under `.encoder`/`.model`.
        for parent in ("encoder", "model"):
            if hasattr(enc, parent) and hasattr(getattr(enc, parent), "layers"):
                return getattr(enc, parent).layers
        raise AttributeError(
            "Could not locate encoder transformer layers for unfreezing; "
            "inspect audio_tower structure and update _encoder_layers()."
        )

    def _init_new_weights(self) -> None:
        for module in (self.proj, self.sentiment_head, self.emotion_head):
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, input_features: Tensor) -> Tuple[Tensor, Tensor]:
        """Classify a batch of log-mel spectrograms.

        Args:
            input_features: Tensor of shape ``(batch, n_mels, seq_len)``
                (Whisper feature-extractor output, padded to 3000 frames).

        Returns:
            Tuple of ``(sentiment_logits, emotion_logits)``.
        """
        encoder_out = self.encoder(input_features)
        frames: Tensor = encoder_out.last_hidden_state   # (B, T, 1280)
        pooled: Tensor = self.attn_pool(frames)           # (B, 1280)
        hidden: Tensor = self.proj(pooled)                # (B, hidden_dim)
        return self.sentiment_head(hidden), self.emotion_head(hidden)

    def trainable_parameters(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
