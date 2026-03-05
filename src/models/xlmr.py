"""
XLM-RoBERTa Classifier Wrapper
================================
Wraps ``FacebookAI/xlm-roberta-base`` (768-dim hidden states) with two
classification heads for **joint** sentiment and emotion prediction.

Architecture
------------
::

    [XLM-RoBERTa encoder]
           |
       [CLS] token embedding  (batch, 768)
           |
    ┌──────┴──────┐
    |             |
[sentiment     [emotion
  head]          head]
(batch, 3)    (batch, 7)

Both heads are simple linear projections applied to the ``[CLS]`` embedding.
The number of output classes is configurable at construction time.

ASR-aware fine-tuning
---------------------
The typical use-case in this dissertation is to fine-tune XLM-RoBERTa on
*ASR transcripts* (not gold-standard text) so that it becomes robust to ASR
errors introduced by Voxtral. The ``get_text_representation`` method provides
a clean way to extract the ``[CLS]`` embedding for use by the fusion layer.

Usage example
-------------
>>> from src.models.xlmr import XLMRobertaClassifier
>>> clf = XLMRobertaClassifier.from_pretrained()
>>> # input_ids, attention_mask: (B, L) long tensors from AutoTokenizer
>>> sentiment_logits, emotion_logits = clf(input_ids, attention_mask)
>>> # sentiment_logits: (B, 3),  emotion_logits: (B, 7)
>>> cls_emb = clf.get_text_representation(input_ids, attention_mask)
>>> # cls_emb: (B, 768)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "FacebookAI/xlm-roberta-base"
_XLM_R_HIDDEN_DIM: int = 768


class XLMRobertaClassifier(nn.Module):
    """Joint sentiment + emotion classifier built on XLM-RoBERTa-base.

    Parameters
    ----------
    model_name_or_path:
        HuggingFace model identifier or local checkpoint path.
    num_sentiment_classes:
        Number of output classes for the sentiment head.
        Defaults to 3 (negative / neutral / positive).
    num_emotion_classes:
        Number of output classes for the emotion head.
        Defaults to 7 (MELD emotion taxonomy).
    dropout_prob:
        Dropout probability applied to the ``[CLS]`` embedding before each
        classification head. Defaults to 0.1.
    freeze_encoder:
        If ``True``, freeze all XLM-RoBERTa parameters so that only the
        classification heads are trained. Useful for a feature-extraction
        baseline (ablation A3).
    cache_dir:
        Optional HuggingFace cache directory.
    """

    def __init__(
        self,
        model_name_or_path: str = _DEFAULT_MODEL,
        num_sentiment_classes: int = 3,
        num_emotion_classes: int = 7,
        dropout_prob: float = 0.1,
        freeze_encoder: bool = False,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()

        from transformers import AutoModel

        self.model_name_or_path = model_name_or_path
        self.num_sentiment_classes = num_sentiment_classes
        self.num_emotion_classes = num_emotion_classes
        self.hidden_dim = _XLM_R_HIDDEN_DIM

        # ---- Backbone ----
        self.encoder = AutoModel.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            add_pooling_layer=False,  # We handle [CLS] manually
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info(
                "XLM-RoBERTa encoder parameters frozen (feature-extraction mode)."
            )

        # ---- Classification heads ----
        self.dropout = nn.Dropout(p=dropout_prob)

        self.sentiment_head = nn.Linear(self.hidden_dim, num_sentiment_classes)
        self.emotion_head = nn.Linear(self.hidden_dim, num_emotion_classes)

        # Initialise head weights with small values
        nn.init.normal_(self.sentiment_head.weight, std=0.02)
        nn.init.zeros_(self.sentiment_head.bias)
        nn.init.normal_(self.emotion_head.weight, std=0.02)
        nn.init.zeros_(self.emotion_head.bias)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def get_text_representation(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        """Extract the ``[CLS]`` token embedding from XLM-RoBERTa.

        Parameters
        ----------
        input_ids:
            Token ID tensor of shape ``(batch, seq_len)``.
        attention_mask:
            Binary attention mask of shape ``(batch, seq_len)``.
            1 for real tokens, 0 for padding.

        Returns
        -------
        Tensor of shape ``(batch, 768)`` containing the ``[CLS]`` embedding.
        The tensor is on the same device as the model parameters.
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        # last_hidden_state: (batch, seq_len, hidden_dim)
        # Index 0 along the sequence dimension is the [CLS] token
        cls_embedding: Tensor = outputs.last_hidden_state[:, 0, :]  # (B, D)
        return cls_embedding

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute sentiment and emotion logits from tokenised text.

        Parameters
        ----------
        input_ids:
            Token ID tensor of shape ``(batch, seq_len)``.
        attention_mask:
            Binary attention mask of shape ``(batch, seq_len)``.

        Returns
        -------
        tuple of (sentiment_logits, emotion_logits):
            * ``sentiment_logits``: ``(batch, num_sentiment_classes)``
            * ``emotion_logits``:   ``(batch, num_emotion_classes)``
        """
        cls_embedding = self.get_text_representation(input_ids, attention_mask)
        dropped = self.dropout(cls_embedding)

        sentiment_logits: Tensor = self.sentiment_head(dropped)   # (B, 3)
        emotion_logits: Tensor = self.emotion_head(dropped)       # (B, 7)

        return sentiment_logits, emotion_logits

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = _DEFAULT_MODEL,
        num_sentiment_classes: int = 3,
        num_emotion_classes: int = 7,
        dropout_prob: float = 0.1,
        freeze_encoder: bool = False,
        cache_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> "XLMRobertaClassifier":
        """Construct an :class:`XLMRobertaClassifier`, optionally loading a
        fine-tuned checkpoint.

        Parameters
        ----------
        model_name_or_path:
            HuggingFace model ID or local HuggingFace checkpoint directory.
        num_sentiment_classes:
            Number of sentiment output classes.
        num_emotion_classes:
            Number of emotion output classes.
        dropout_prob:
            Dropout probability for the classification heads.
        freeze_encoder:
            Whether to freeze the encoder parameters.
        cache_dir:
            HuggingFace cache directory.
        checkpoint_path:
            Optional path to a PyTorch ``.pt`` / ``.bin`` state-dict file
            produced by a previous training run. If provided, the state dict
            is loaded into the model after construction.

        Returns
        -------
        XLMRobertaClassifier
        """
        model = cls(
            model_name_or_path=model_name_or_path,
            num_sentiment_classes=num_sentiment_classes,
            num_emotion_classes=num_emotion_classes,
            dropout_prob=dropout_prob,
            freeze_encoder=freeze_encoder,
            cache_dir=cache_dir,
        )

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            # Allow loading state dicts that were saved with a module wrapper
            if all(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            logger.info("Loaded classifier checkpoint from %s", checkpoint_path)

        return model

    def freeze_encoder(self) -> None:
        """Freeze all XLM-RoBERTa encoder parameters in-place."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze all XLM-RoBERTa encoder parameters in-place."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def trainable_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameters(self) -> int:
        """Return the total number of parameters (trainable + frozen)."""
        return sum(p.numel() for p in self.parameters())
