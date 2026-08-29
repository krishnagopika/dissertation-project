"""Fusion over acoustic *sequences*: pooling wrapper + capacity-matched variants.

The existing fusion classes in `fusion.py` take a pooled acoustic vector,
`(B, 1280)`. The cache now holds `(T, 1280)` sequences so that pooling can be
learned (docs/DECISIONS.md ADR-002/003). This module bridges the two.

Design
------
The pooler is a **wrapper**, not a member of each fusion variant. One
implementation therefore serves all four, and the pooling ablation applies
uniformly: 4 fusion mechanisms x 4 poolers = 16 cells from one code path.
Duplicating a pooler inside each variant would make them drift.

    (B, T, 1280) + mask ──► pooler ──► (B, out_dim) ──┐
                                                       ├──► fusion ──► heads
    (B, 768) text ────────────────────────────────────┘

Comparability
-------------
Two properties are enforced here that the original variants did not have.

**Every variant projects each modality before combining.** `FusionModel`
previously concatenated raw 768+1280 dims straight into its MLP while the other
three projected each modality to a common width first. That made the concat
baseline differ from the others in *two* ways — fusion mechanism and the
presence of a projection — so a win or loss could not be attributed. All four
now share the projection stage and differ only in how the two projected
representations are combined.

**Parameter counts are reported, not assumed equal.** Different combination
mechanisms genuinely need different numbers of parameters; a gated variant has
a gate matrix that a sum variant does not. That is a real difference and should
not be hidden, but it must be *visible* so a gain can be judged against the
capacity that produced it. `parameter_report()` returns per-component counts.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.models.pooling import build_pooler


class _Head(nn.Module):
    """Two-layer classification head shared by every variant."""

    def __init__(self, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:                    # noqa: D102
        return self.net(x)


def _projector(in_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    """Per-modality projection to the common fusion width.

    Applied to BOTH modalities in EVERY variant, so the four differ only in
    their combination step.
    """
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(p=dropout),
    )


class SequenceFusion(nn.Module):
    """Pool an acoustic sequence, fuse with a text embedding, classify.

    Args:
        acoustic_dim: Frame width of the cached sequences (1280).
        text_dim: Text embedding width (768).
        hidden_dim: Common fusion width.
        num_emotion_classes: Emotion head output size.
        num_sentiment_classes: Sentiment head output size.
        dropout: Dropout throughout.
        pooling: One of ``masked_mean``, ``attention_fixed``, ``attention``,
            ``attentive_stats``.
        fusion: One of ``concat``, ``sum``, ``gated``, ``crossmodal``.
        attention_dim: Width of the attention scorer (attention poolers only).
        pooling_dropout: Dropout on attention weights.

    Raises:
        ValueError: On an unknown fusion name.
    """

    FUSIONS = ("concat", "sum", "gated", "crossmodal")

    #: Unimodal baselines share this class so that every ablation runs through
    #: one code path. A separate acoustic-only script would drift from the
    #: fusion model in pooling, projection width, head architecture and
    #: initialisation, and the fusion-vs-unimodal comparison would then be
    #: confounded by those differences rather than measuring fusion.
    MODALITIES = ("both", "acoustic", "text")

    def __init__(
        self,
        acoustic_dim: int = 1280,
        text_dim: int = 768,
        hidden_dim: int = 512,
        num_emotion_classes: int = 7,
        num_sentiment_classes: int = 3,
        dropout: float = 0.3,
        pooling: str = "attention",
        fusion: str = "concat",
        attention_dim: int = 128,
        pooling_dropout: float = 0.0,
        modality: str = "both",
    ) -> None:
        super().__init__()
        if fusion not in self.FUSIONS:
            raise ValueError(f"fusion must be one of {self.FUSIONS}, got {fusion!r}")
        if modality not in self.MODALITIES:
            raise ValueError(
                f"modality must be one of {self.MODALITIES}, got {modality!r}"
            )

        self.fusion_name = fusion
        self.pooling_name = pooling
        self.modality = modality

        # Pooler owns the acoustic sequence collapse. Its output width is NOT
        # assumed to equal acoustic_dim: attentive_stats concatenates the
        # weighted mean and std and therefore emits 2*acoustic_dim.
        # Not built for text-only: an unused pooler would inflate the reported
        # parameter count and make the unimodal comparison misleading.
        self.pooler = (
            build_pooler(pooling, acoustic_dim,
                         attention_dim=attention_dim, dropout=pooling_dropout)
            if modality in ("both", "acoustic") else None
        )
        pooled_dim = self.pooler.output_dim if self.pooler is not None else 0

        self.acoustic_proj = (_projector(pooled_dim, hidden_dim, dropout)
                              if modality in ("both", "acoustic") else None)
        self.text_proj = (_projector(text_dim, hidden_dim, dropout)
                          if modality in ("both", "text") else None)

        # A unimodal model has nothing to combine, so it skips the fusion stage
        # entirely and feeds its single projection to `post`. Keeping `post`,
        # the heads and the projection identical to the fusion model is what
        # makes "fusion beats unimodal" a statement about fusion rather than
        # about differing head capacity.
        if modality != "both":
            self.combine_in = hidden_dim
        elif fusion == "concat":
            self.combine_in = hidden_dim * 2
        else:
            self.combine_in = hidden_dim

        if modality != "both":
            pass          # no gate: nothing to gate against
        elif fusion == "gated":
            # Symmetric convex gate: g*t + (1-g)*a. Weights sum to 1, so this
            # is a soft SELECTION between modalities.
            self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        elif fusion == "crossmodal":
            # Asymmetric: each modality gates the OTHER, gates independent.
            # Not convex -- both can be suppressed or both passed through.
            self.gate_t_from_a = nn.Linear(hidden_dim, hidden_dim)
            self.gate_a_from_t = nn.Linear(hidden_dim, hidden_dim)

        self.post = nn.Sequential(
            nn.Linear(self.combine_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
        )
        self.emotion_head = _Head(hidden_dim, num_emotion_classes, dropout)
        self.sentiment_head = _Head(hidden_dim, num_sentiment_classes, dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _combine(self, t: Optional[Tensor], a: Optional[Tensor]) -> Tensor:
        """Apply the selected combination to the available representations."""
        if self.modality == "acoustic":
            return a
        if self.modality == "text":
            return t
        if self.fusion_name == "concat":
            return torch.cat([t, a], dim=-1)
        if self.fusion_name == "sum":
            return t + a
        if self.fusion_name == "gated":
            g = torch.sigmoid(self.gate(torch.cat([t, a], dim=-1)))
            return g * t + (1.0 - g) * a
        g_ta = torch.sigmoid(self.gate_t_from_a(a))
        g_at = torch.sigmoid(self.gate_a_from_t(t))
        return g_ta * t + g_at * a

    def forward(
        self,
        text: Tensor,
        acoustic: Tensor,
        acoustic_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Args:
            text: ``(B, text_dim)`` XLM-R [CLS] embeddings.
            acoustic: ``(B, T, acoustic_dim)`` padded frame sequences.
            acoustic_mask: ``(B, T)`` bool, True where the frame is real.

        Returns:
            ``(emotion_logits, sentiment_logits, attention_weights)``. The
            weights are ``None`` for parameter-free poolers, and are the
            **undropped** distribution otherwise — plot them from eval mode.
        """
        a = attn = None
        if self.modality in ("both", "acoustic"):
            pooled, attn = self.pooler(acoustic, acoustic_mask)
            a = self.acoustic_proj(pooled)

        t = None
        if self.modality in ("both", "text"):
            ref = a if a is not None else text
            t = self.text_proj(text.to(ref.device))

        fused = self.post(self._combine(t, a))
        return self.emotion_head(fused), self.sentiment_head(fused), attn

    def represent(
        self,
        text: Tensor,
        acoustic: Tensor,
        acoustic_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """The fused representation, before the classification heads.

        This is the vector the heads see: pooled acoustics and text, each
        projected, combined by the configured mechanism, then passed through
        ``post``. Exposed so a downstream model (e.g. the bc-LSTM dialogue
        context model) can consume the LEARNED representation instead of a raw
        concatenation of the two caches.

        That matters for comparability: bc-LSTM on raw features vs fusion on
        learned features differ in BOTH pooling and context, so neither result
        isolates context. Feeding this vector to bc-LSTM makes context the only
        difference between them.

        Args:
            text: ``(B, text_dim)`` XLM-R [CLS] embeddings.
            acoustic: ``(B, T, acoustic_dim)`` padded frame sequences.
            acoustic_mask: ``(B, T)`` bool, True where the frame is real.

        Returns:
            ``(B, hidden_dim)`` fused representation.
        """
        a = None
        if self.modality in ("both", "acoustic"):
            pooled, _ = self.pooler(acoustic, acoustic_mask)
            a = self.acoustic_proj(pooled)
        t = None
        if self.modality in ("both", "text"):
            ref = a if a is not None else text
            t = self.text_proj(text.to(ref.device))
        return self.post(self._combine(t, a))

    def parameter_report(self) -> Dict[str, int]:
        """Trainable parameters per component.

        Reported so a performance difference between fusion mechanisms can be
        weighed against the capacity that produced it, rather than assumed to
        come from the mechanism alone.
        """
        def n(mod: nn.Module) -> int:
            return sum(p.numel() for p in mod.parameters() if p.requires_grad)

        rep = {"post": n(self.post)}
        # A unimodal model has no projector for the absent branch, and the
        # pooler is dead weight in a text-only model, so report only what the
        # configured modality actually instantiates.
        if self.modality in ("both", "acoustic"):
            rep["pooler"] = n(self.pooler)
            rep["acoustic_proj"] = n(self.acoustic_proj)
        if self.modality in ("both", "text"):
            rep["text_proj"] = n(self.text_proj)
        rep.update({
            "emotion_head": n(self.emotion_head),
            "sentiment_head": n(self.sentiment_head),
        })
        if hasattr(self, "gate"):
            rep["gate"] = n(self.gate)
        if hasattr(self, "gate_t_from_a"):
            rep["gate_t_from_a"] = n(self.gate_t_from_a)
            rep["gate_a_from_t"] = n(self.gate_a_from_t)
        rep["TOTAL"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return rep
