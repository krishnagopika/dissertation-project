"""
Multimodal Fusion Layer
========================
Implements the fusion strategy that combines acoustic embeddings (from
Voxtral's Whisper encoder) with text embeddings (from XLM-RoBERTa's [CLS]
token) into joint sentiment and emotion predictions.

Two fusion variants are provided:

1. :class:`FusionModel` — **Concatenation fusion** (default, ablation A1–A4).
   Concatenates the text and acoustic representations, then projects through a
   two-layer MLP with LayerNorm, GELU activation, and dropout.

2. :class:`SumFusion` — **Sum fusion** (ablation A5).
   Projects each modality to a common dimension separately, then adds them
   element-wise before the classification heads.

3. :class:`FusionPipeline` — End-to-end wrapper that chains
   :class:`~src.models.voxtral.VoxtralWrapper` →
   :class:`~src.models.xlmr.XLMRobertaClassifier` →
   :class:`FusionModel` into a single callable.

Architecture diagram (FusionModel)
-----------------------------------
::

    acoustic_repr (B, acoustic_dim)  ──┐
                                       concat ──> MLP ──> sentiment head
    text_repr     (B, text_dim)      ──┘         (2-layer)  emotion head

MLP internals::

    Linear(acoustic_dim + text_dim, hidden_dim)
    LayerNorm(hidden_dim)
    GELU
    Dropout(dropout)
    Linear(hidden_dim, hidden_dim)
    LayerNorm(hidden_dim)
    GELU
    Dropout(dropout)

Classification heads::

    Linear(hidden_dim, num_sentiment_classes)   # sentiment
    Linear(hidden_dim, num_emotion_classes)     # emotion

Usage example
-------------
>>> from src.models.fusion import FusionModel, FusionPipeline
>>>
>>> # Stand-alone fusion module
>>> fusion = FusionModel(acoustic_dim=1280, text_dim=768)
>>> sentiment_logits, emotion_logits = fusion(text_repr, acoustic_repr)
>>>
>>> # End-to-end pipeline
>>> pipeline = FusionPipeline(
...     voxtral_model_name="mistralai/Voxtral-Mini-3B-2507",
...     xlmr_model_name="FacebookAI/xlm-roberta-base",
... )
>>> sentiment_logits, emotion_logits = pipeline(
...     audio_tensor=waveform,
...     input_ids=input_ids,
...     attention_mask=attention_mask,
...     sample_rate=16000,
... )
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

# Default dimensions
_ACOUSTIC_DIM: int = 1280  # Whisper Large-v3 encoder hidden size
_TEXT_DIM: int = 768        # XLM-RoBERTa-base hidden size
_HIDDEN_DIM: int = 512
_NUM_SENTIMENT: int = 3
_NUM_EMOTION: int = 7


# ---------------------------------------------------------------------------
# Shared building block: classification head
# ---------------------------------------------------------------------------


class _ClassificationHead(nn.Module):
    """Simple linear classification head with optional dropout.

    Parameters
    ----------
    in_features:
        Dimensionality of input features.
    num_classes:
        Number of output classes.
    dropout_prob:
        Dropout probability applied before the linear layer.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_prob)
        self.linear = nn.Linear(in_features, num_classes)
        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(self.dropout(x))


# ---------------------------------------------------------------------------
# 1. Concatenation Fusion (default)
# ---------------------------------------------------------------------------


class FusionModel(nn.Module):
    """Concatenation-based multimodal fusion module.

    Concatenates acoustic and text representations and projects them through
    a 2-layer MLP to produce joint sentiment and emotion logits.

    Parameters
    ----------
    acoustic_dim:
        Dimensionality of the acoustic embedding (e.g. 1280 for Whisper
        Large-v3).
    text_dim:
        Dimensionality of the text embedding (e.g. 768 for XLM-RoBERTa-base).
    hidden_dim:
        Hidden dimensionality of the projection MLP.
    num_sentiment_classes:
        Number of sentiment output classes (default 3).
    num_emotion_classes:
        Number of emotion output classes (default 7).
    dropout_prob:
        Dropout probability applied throughout the MLP and heads.
    """

    def __init__(
        self,
        acoustic_dim: int = _ACOUSTIC_DIM,
        text_dim: int = _TEXT_DIM,
        hidden_dim: int = _HIDDEN_DIM,
        num_sentiment_classes: int = _NUM_SENTIMENT,
        num_emotion_classes: int = _NUM_EMOTION,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()

        self.acoustic_dim = acoustic_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.fused_dim = acoustic_dim + text_dim

        # 2-layer projection MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
        )

        # Classification heads
        self.sentiment_head = _ClassificationHead(
            hidden_dim, num_sentiment_classes, dropout_prob
        )
        self.emotion_head = _ClassificationHead(
            hidden_dim, num_emotion_classes, dropout_prob
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        text_repr: Tensor,
        acoustic_repr: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Fuse modalities and produce classification logits.

        Parameters
        ----------
        text_repr:
            ``[CLS]`` embedding from XLM-RoBERTa. Shape ``(batch, text_dim)``.
        acoustic_repr:
            Mean-pooled encoder output from Voxtral. Shape
            ``(batch, acoustic_dim)``.

        Returns
        -------
        tuple of (sentiment_logits, emotion_logits):
            * ``sentiment_logits``: ``(batch, num_sentiment_classes)``
            * ``emotion_logits``:   ``(batch, num_emotion_classes)``
        """
        # Ensure both representations are on the same device
        acoustic_repr = acoustic_repr.to(text_repr.device)

        fused: Tensor = torch.cat([text_repr, acoustic_repr], dim=-1)  # (B, D_t + D_a)
        projected: Tensor = self.mlp(fused)                              # (B, H)

        sentiment_logits: Tensor = self.sentiment_head(projected)        # (B, S)
        emotion_logits: Tensor = self.emotion_head(projected)            # (B, E)

        return sentiment_logits, emotion_logits

    def trainable_parameters(self) -> int:
        """Return the total number of trainable parameters in this module."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 2. Sum Fusion (ablation A5)
# ---------------------------------------------------------------------------


class SumFusion(nn.Module):
    """Element-wise sum fusion for ablation study A5.

    Each modality is first projected to a common ``hidden_dim``-dimensional
    space via a separate linear layer, then the projections are added
    element-wise before the classification heads.

    Parameters
    ----------
    acoustic_dim:
        Dimensionality of acoustic embeddings.
    text_dim:
        Dimensionality of text embeddings.
    hidden_dim:
        Common projection dimensionality.
    num_sentiment_classes:
        Number of sentiment output classes.
    num_emotion_classes:
        Number of emotion output classes.
    dropout_prob:
        Dropout probability.
    """

    def __init__(
        self,
        acoustic_dim: int = _ACOUSTIC_DIM,
        text_dim: int = _TEXT_DIM,
        hidden_dim: int = _HIDDEN_DIM,
        num_sentiment_classes: int = _NUM_SENTIMENT,
        num_emotion_classes: int = _NUM_EMOTION,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()

        self.acoustic_dim = acoustic_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim

        # Per-modality projections
        self.acoustic_proj = nn.Sequential(
            nn.Linear(acoustic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
        )

        # Post-sum projection
        self.post_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
        )

        # Classification heads
        self.sentiment_head = _ClassificationHead(
            hidden_dim, num_sentiment_classes, dropout_prob
        )
        self.emotion_head = _ClassificationHead(
            hidden_dim, num_emotion_classes, dropout_prob
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        text_repr: Tensor,
        acoustic_repr: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Fuse modalities via element-wise addition and produce logits.

        Parameters
        ----------
        text_repr:
            ``[CLS]`` embedding. Shape ``(batch, text_dim)``.
        acoustic_repr:
            Acoustic embedding. Shape ``(batch, acoustic_dim)``.

        Returns
        -------
        tuple of (sentiment_logits, emotion_logits).
        """
        acoustic_repr = acoustic_repr.to(text_repr.device)

        a_proj: Tensor = self.acoustic_proj(acoustic_repr)  # (B, H)
        t_proj: Tensor = self.text_proj(text_repr)          # (B, H)

        fused: Tensor = a_proj + t_proj                     # (B, H) — element-wise sum
        projected: Tensor = self.post_fusion(fused)         # (B, H)

        sentiment_logits: Tensor = self.sentiment_head(projected)
        emotion_logits: Tensor = self.emotion_head(projected)

        return sentiment_logits, emotion_logits

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 3. End-to-end pipeline
# ---------------------------------------------------------------------------


class FusionPipeline(nn.Module):
    """End-to-end multimodal sentiment/emotion pipeline.

    Chains together:
    1. :class:`~src.models.voxtral.VoxtralWrapper` for ASR + acoustic embedding.
    2. :class:`~src.models.xlmr.XLMRobertaClassifier` for text embedding.
    3. A :class:`FusionModel` (or :class:`SumFusion`) for joint classification.

    Parameters
    ----------
    voxtral_model_name:
        HuggingFace ID for the Voxtral model.
    xlmr_model_name:
        HuggingFace ID for the XLM-RoBERTa model.
    fusion_type:
        ``"concat"`` for :class:`FusionModel`, ``"sum"`` for
        :class:`SumFusion`.
    acoustic_dim:
        Dimensionality of acoustic embeddings (1280 for Whisper Large-v3).
    text_dim:
        Dimensionality of text embeddings (768 for XLM-RoBERTa-base).
    hidden_dim:
        Hidden dimension of the fusion MLP.
    num_sentiment_classes:
        Number of sentiment output classes.
    num_emotion_classes:
        Number of emotion output classes.
    dropout_prob:
        Dropout probability for fusion and heads.
    voxtral_device_map:
        Device map for Voxtral (e.g. ``"auto"``).
    voxtral_torch_dtype:
        Dtype for Voxtral parameters.
    xlmr_freeze_encoder:
        If ``True``, freeze XLM-RoBERTa encoder parameters.
    cache_dir:
        HuggingFace cache directory.
    """

    def __init__(
        self,
        voxtral_model_name: str = "mistralai/Voxtral-Mini-3B-2507",
        xlmr_model_name: str = "FacebookAI/xlm-roberta-base",
        fusion_type: str = "concat",
        acoustic_dim: int = _ACOUSTIC_DIM,
        text_dim: int = _TEXT_DIM,
        hidden_dim: int = _HIDDEN_DIM,
        num_sentiment_classes: int = _NUM_SENTIMENT,
        num_emotion_classes: int = _NUM_EMOTION,
        dropout_prob: float = 0.1,
        voxtral_device_map: str = "auto",
        voxtral_torch_dtype: torch.dtype = torch.bfloat16,
        xlmr_freeze_encoder: bool = False,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()

        # Lazy imports to avoid circular dependencies at module level
        from src.models.voxtral import VoxtralWrapper
        from src.models.xlmr import XLMRobertaClassifier

        # --- Voxtral ---
        logger.info("Initialising VoxtralWrapper: %s", voxtral_model_name)
        self.voxtral = VoxtralWrapper(
            model_name_or_path=voxtral_model_name,
            device_map=voxtral_device_map,
            torch_dtype=voxtral_torch_dtype,
            cache_dir=cache_dir,
        )

        # --- XLM-RoBERTa ---
        logger.info("Initialising XLMRobertaClassifier: %s", xlmr_model_name)
        self.xlmr = XLMRobertaClassifier(
            model_name_or_path=xlmr_model_name,
            num_sentiment_classes=num_sentiment_classes,
            num_emotion_classes=num_emotion_classes,
            dropout_prob=dropout_prob,
            freeze_encoder=xlmr_freeze_encoder,
            cache_dir=cache_dir,
        )

        # --- Fusion module ---
        fusion_type_lower = fusion_type.lower()
        if fusion_type_lower == "concat":
            self.fusion = FusionModel(
                acoustic_dim=acoustic_dim,
                text_dim=text_dim,
                hidden_dim=hidden_dim,
                num_sentiment_classes=num_sentiment_classes,
                num_emotion_classes=num_emotion_classes,
                dropout_prob=dropout_prob,
            )
        elif fusion_type_lower == "sum":
            self.fusion = SumFusion(
                acoustic_dim=acoustic_dim,
                text_dim=text_dim,
                hidden_dim=hidden_dim,
                num_sentiment_classes=num_sentiment_classes,
                num_emotion_classes=num_emotion_classes,
                dropout_prob=dropout_prob,
            )
        else:
            raise ValueError(
                f"fusion_type must be 'concat' or 'sum', got '{fusion_type}'"
            )

        logger.info("FusionPipeline ready. Fusion strategy: %s", fusion_type)

    def forward(
        self,
        audio_tensor: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        sample_rate: int = 16_000,
    ) -> Tuple[Tensor, Tensor]:
        """Run the full multimodal pipeline.

        Parameters
        ----------
        audio_tensor:
            Raw waveform at ``sample_rate`` Hz. Shape ``(T,)`` or ``(B, T)``.
        input_ids:
            Tokenised text (e.g. ASR transcript) token IDs.
            Shape ``(batch, seq_len)``.
        attention_mask:
            Attention mask for ``input_ids``. Shape ``(batch, seq_len)``.
        sample_rate:
            Sample rate of ``audio_tensor`` in Hz.

        Returns
        -------
        tuple of (sentiment_logits, emotion_logits):
            * ``sentiment_logits``: ``(batch, num_sentiment_classes)``
            * ``emotion_logits``:   ``(batch, num_emotion_classes)``
        """
        # 1. Acoustic embeddings from Voxtral encoder
        acoustic_repr: Tensor = self.voxtral.extract_acoustic_embeddings(
            audio_tensor, sample_rate=sample_rate
        )  # (B, acoustic_dim)  — returned on CPU

        # 2. Text embeddings from XLM-RoBERTa
        text_repr: Tensor = self.xlmr.get_text_representation(
            input_ids, attention_mask
        )  # (B, text_dim)

        # 3. Fusion
        sentiment_logits, emotion_logits = self.fusion(text_repr, acoustic_repr)

        return sentiment_logits, emotion_logits

    def transcribe_and_forward(
        self,
        audio_tensor: Tensor,
        tokenizer,
        sample_rate: int = 16_000,
        max_length: int = 128,
    ) -> Tuple[Tensor, Tensor, str]:
        """ASR + classification in a single call (convenience method).

        Transcribes the audio with Voxtral, tokenises the transcript, and
        runs the full fusion pipeline.

        Parameters
        ----------
        audio_tensor:
            Waveform tensor. Shape ``(T,)`` (single clip only).
        tokenizer:
            HuggingFace tokenizer compatible with XLM-RoBERTa.
        sample_rate:
            Sample rate of the audio.
        max_length:
            Maximum token sequence length for the tokenizer.

        Returns
        -------
        (sentiment_logits, emotion_logits, transcript):
            * ``sentiment_logits``: ``(1, num_sentiment_classes)``
            * ``emotion_logits``:   ``(1, num_emotion_classes)``
            * ``transcript``:       str
        """
        # ASR
        transcript: str = self.voxtral.transcribe(
            audio_tensor, sample_rate=sample_rate
        )
        if isinstance(transcript, list):
            transcript = transcript[0]

        # Tokenise transcript
        encoding = tokenizer(
            transcript,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        device = next(self.xlmr.parameters()).device
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        sentiment_logits, emotion_logits = self.forward(
            audio_tensor=audio_tensor,
            input_ids=input_ids,
            attention_mask=attention_mask,
            sample_rate=sample_rate,
        )

        return sentiment_logits, emotion_logits, transcript

    def trainable_parameters(self) -> int:
        """Return the count of trainable parameters across the full pipeline."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_fusion_checkpoint(self, path: str) -> None:
        """Save the fusion module (and XLM-R heads) state dict to disk.

        Voxtral weights are typically not saved because they are not fine-tuned
        (or are too large for typical checkpoint storage).

        Parameters
        ----------
        path:
            File path for the ``.pt`` checkpoint.
        """
        state = {
            "fusion": self.fusion.state_dict(),
            "xlmr": self.xlmr.state_dict(),
        }
        torch.save(state, path)
        logger.info("Fusion checkpoint saved to %s", path)

    def load_fusion_checkpoint(self, path: str, strict: bool = False) -> None:
        """Load a previously saved fusion checkpoint.

        Parameters
        ----------
        path:
            File path of the ``.pt`` checkpoint.
        strict:
            Whether to require an exact match of keys.
        """
        state = torch.load(path, map_location="cpu")
        self.fusion.load_state_dict(state["fusion"], strict=strict)
        self.xlmr.load_state_dict(state["xlmr"], strict=strict)
        logger.info("Fusion checkpoint loaded from %s", path)
