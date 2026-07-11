"""
Acoustic Emotion Classifier (RAVDESS pre-training)
====================================================
A lightweight emotion classifier over pre-cached Voxtral acoustic embeddings.
Used for two purposes:

1. **Standalone training on RAVDESS** as an 8-class emotion classifier
   (neutral, calm, happy, sad, angry, fearful, disgust, surprised).
2. **Transferable backbone**: after Phase A training, the ``acoustic_proj``
   sub-module's state dict is identical in shape and key names to the
   ``acoustic_proj`` in :class:`~src.models.fusion.SumFusion`,
   :class:`~src.models.fusion.GatedFusion`, and
   :class:`~src.models.fusion.CrossModalGating` — so the pretrained weights
   load with a single ``.load_state_dict()`` call.

Architecture
------------
::

    Voxtral acoustic embedding (B, 1280)
              |
    [Linear(1280, hidden_dim)]
    [LayerNorm(hidden_dim)]
    [GELU]                       <-- "acoustic_proj" (transferable)
    [Dropout(p)]
              |
    [Linear(hidden_dim, num_classes)]   <-- task-specific head (NOT transferred)

Why the head is task-specific
-----------------------------
RAVDESS has 8 emotion classes (includes calm). MELD has 7 (no calm). Per the
supervisor's note, we keep all RAVDESS data by simply swapping the head when
moving to MELD — the backbone learns from every RAVDESS clip, the MELD head
is freshly initialised against MELD's taxonomy.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class AcousticEmotionClassifier(nn.Module):
    """Acoustic-only emotion classifier over cached Voxtral embeddings.

    Args:
        acoustic_dim: Dimension of input acoustic embedding (1280 for Whisper).
        hidden_dim: Hidden dimension of the projection. Must equal
            ``model.fusion_hidden`` if you intend to transfer the backbone
            into a fusion model.
        num_classes: Number of output classes (8 for RAVDESS, 7 for MELD).
        dropout_prob: Dropout probability inside the projection.
    """

    def __init__(
        self,
        acoustic_dim: int = 1280,
        hidden_dim: int = 512,
        num_classes: int = 8,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()

        self.acoustic_dim = acoustic_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # NOTE: keep this submodule named ``acoustic_proj`` and the layer
        # ordering identical to SumFusion / GatedFusion / CrossModalGating in
        # src/models/fusion.py — that is what makes weight transfer a
        # single load_state_dict() call.
        self.acoustic_proj = nn.Sequential(
            nn.Linear(acoustic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_prob),
        )

        self.head = nn.Linear(hidden_dim, num_classes)

        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        for module in self.acoustic_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, acoustic_repr: Tensor) -> Tensor:
        """Predict emotion logits from a cached acoustic embedding.

        Args:
            acoustic_repr: Tensor of shape ``(batch, acoustic_dim)``.

        Returns:
            Tensor of shape ``(batch, num_classes)``.
        """
        hidden: Tensor = self.acoustic_proj(acoustic_repr)
        return self.head(hidden)

    def backbone_state_dict(self) -> dict:
        """Return only the ``acoustic_proj`` sub-state-dict.

        These keys match :class:`~src.models.fusion.SumFusion.acoustic_proj`
        (and similarly for GatedFusion, CrossModalGating), so the result can
        be loaded directly into a fusion model's acoustic projection without
        any key remapping.
        """
        return {
            f"acoustic_proj.{k}": v
            for k, v in self.acoustic_proj.state_dict().items()
        }


def make_classifier_for_meld(
    acoustic_dim: int = 1280,
    hidden_dim: int = 512,
    dropout_prob: float = 0.1,
) -> AcousticEmotionClassifier:
    """Build a 7-class MELD-head version (used as a sanity-check baseline)."""
    return AcousticEmotionClassifier(
        acoustic_dim=acoustic_dim,
        hidden_dim=hidden_dim,
        num_classes=7,
        dropout_prob=dropout_prob,
    )
