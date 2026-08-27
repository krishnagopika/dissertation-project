"""
Temporal Pooling
================
Collapse a variable-length acoustic frame sequence into one fixed-size vector.

Why this module exists
----------------------
The pipeline used to mean-pool inside :mod:`src.models.voxtral` at preprocessing
time, writing one 1280-d vector per utterance to disk. That forced the pooling
to be parameter-free: a learned pooling cannot be baked into a cache produced
before training starts.

Moving pooling here — into the trainable head, operating on cached *sequences* —
lifts that restriction while keeping Voxtral frozen and run exactly once.

Two problems are deliberately kept separate
-------------------------------------------
1. **Padding.** The old pooling averaged over all 1500 encoder frames even when
   the utterance occupied ~180 of them, so most of the average was padding.
2. **Uniform weighting.** Mean pooling assumes every frame is equally
   informative. For emotion that is plainly false — affect concentrates in
   prosodic peaks.

:class:`MaskedMeanPooling` fixes (1) alone and exists as the *control*. Without
it, any gain from :class:`AttentionPooling` is unattributable: the model may
simply have learned to ignore padding rather than to find the emotional peak.
Report both.

All poolers share one interface::

    pooled, weights = pooler(frames, mask)

    frames  : (B, T, D) float
    mask    : (B, T) bool, True = real frame, False = padding. None = all real.
    pooled  : (B, out_dim)
    weights : (B, T) attention weights, or None for parameter-free poolers.

``weights`` is returned rather than discarded because it is directly useful for
the write-up: plotting it over time shows *where* in an utterance the model
looks, and comparing that across accents is evidence about dialect robustness
that WER alone cannot give.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# Added to a variance before sqrt. Guards against NaN gradients when a segment
# has (near-)zero variance, which happens on silent or single-frame clips.
_VAR_EPS: float = 1e-5

# Floor for the post-dropout attention-weight sum. Deliberately a SEPARATE
# constant from _VAR_EPS: that one is a pre-sqrt variance floor, and conflating
# the two means tuning the variance path would silently move the dropout
# fallback boundary. Also note 1e-5 is subnormal in fp16 (smallest normal
# ~6.1e-5), so reusing it here would be fragile in a half-precision head.
_WEIGHT_SUM_EPS: float = 1e-6


def _check_mask(frames: Tensor, mask: Optional[Tensor]) -> Tensor:
    """Return a validated boolean mask, defaulting to all-real frames.

    Args:
        frames: Input of shape ``(B, T, D)``.
        mask: Optional bool tensor of shape ``(B, T)``.

    Returns:
        Bool tensor of shape ``(B, T)`` on the same device as ``frames``.

    Raises:
        ValueError: If the mask shape does not match the frame sequence.
    """
    if mask is None:
        return torch.ones(
            frames.shape[:2], dtype=torch.bool, device=frames.device
        )
    if mask.shape != frames.shape[:2]:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} does not match frames "
            f"{tuple(frames.shape[:2])}"
        )
    return mask.to(device=frames.device, dtype=torch.bool)


class MaskedMeanPooling(nn.Module):
    """Mean over real frames only — the parameter-free control.

    Identical to the old ``hidden.mean(dim=1)`` when every frame is real, so any
    difference in results is attributable purely to excluding padding.

    Args:
        input_dim: Frame dimension ``D``. Stored for ``output_dim`` only.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim

    @property
    def output_dim(self) -> int:
        return self.input_dim

    def forward(
        self,
        frames: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            frames: Float tensor of shape ``(B, T, D)``.
            mask: Bool tensor ``(B, T)``, True where the frame is real.

        Returns:
            Tuple of (pooled ``(B, D)``, ``None``).
        """
        mask = _check_mask(frames, mask)
        w = mask.unsqueeze(-1).to(frames.dtype)          # (B, T, 1)
        # clamp_min(1.0): a fully-padded row would otherwise divide by zero and
        # emit NaN, which propagates silently through the whole batch loss.
        denom = w.sum(dim=1).clamp_min(1.0)              # (B, 1)
        return (frames * w).sum(dim=1) / denom, None


class AttentionPooling(nn.Module):
    """Additive (Bahdanau-style) attention pooling with a single learned query.

    Scores each frame independently, softmaxes over time, returns the weighted
    sum::

        e_t = v^T tanh(W h_t + b)
        a   = softmax(e)                (padding scored -inf)
        out = sum_t a_t h_t

    This is the "learned CLS" formulation: the query lives in the trainable head
    rather than inside the frozen encoder, so no Voxtral weights are touched.

    Args:
        input_dim: Frame dimension ``D``.
        attention_dim: Width of the scoring projection.
        dropout: Dropout applied to attention weights during training.
    """

    def __init__(
        self,
        input_dim: int,
        attention_dim: int = 128,
        dropout: float = 0.0,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.trainable = trainable
        self.project = nn.Linear(input_dim, attention_dim)
        self.tanh = nn.Tanh()
        self.score = nn.Linear(attention_dim, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

        if not trainable:
            # Frozen at random init: identical architecture and identical
            # forward cost, but ZERO trainable parameters -- so it matches
            # MaskedMeanPooling on capacity while still producing a non-uniform
            # weighting. See ADR-003: without this rung, the mean -> attention
            # delta confounds "learned frame weighting" with "extra capacity".
            for p in self.parameters():
                p.requires_grad_(False)

    @property
    def output_dim(self) -> int:
        return self.input_dim

    def _attend(self, frames: Tensor, mask: Tensor) -> Tensor:
        """Compute normalised attention weights of shape ``(B, T)``."""
        scores = self.score(self.tanh(self.project(frames))).squeeze(-1)

        # Mask BEFORE softmax. Zeroing weights afterwards would leave padding
        # contributing to the normaliser, shrinking every real weight by the
        # padding fraction — the exact bug this module exists to remove.
        # finfo.min rather than -inf: -inf produces NaN when a row is fully
        # masked, and NaN gradients cannot be recovered from.
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)

        # A fully-masked row has all scores equal to finfo.min, so softmax
        # returns a UNIFORM distribution over all T frames -- i.e. it would pool
        # pure padding. Zero the row instead, matching MaskedMeanPooling, which
        # returns a zero vector for the same input. Callers should still assert
        # mask.any(dim=1).all() at the collator so this never arises.
        return weights * mask.any(dim=1, keepdim=True)

    def _drop_and_renormalise(self, weights: Tensor) -> Tensor:
        """Apply attention dropout, then restore the sum-to-one property.

        nn.Dropout is *inverted* dropout: it zeroes with probability p and
        scales survivors by 1/(1-p), so E[sum] = 1. But that holds only in
        expectation -- for an individual sample the sum is a random variable
        around 1. Both statistics below assume a genuine weight distribution:

          mean = sum_t w_t x_t            requires sum_t w_t == 1
          var  = sum_t w_t x_t^2 - mean^2 requires it *exactly*

        With sum_t w_t = s, the variance identity evaluates to
        ``s*E'[x^2] - s^2*E'[x]^2`` instead of ``E'[x^2] - E'[x]^2``. When s > 1
        and the frames are near-constant (a steady vowel, a silent region) this
        goes NEGATIVE for genuinely positive variance, and the clamp in
        AttentiveStatsPooling then pins std to a signal-free constant.
        Measured on near-constant frames at p=0.3: 48% of elements negative.

        Renormalising is not merely a rescale -- it is EXACTLY equivalent to
        masking the dropped positions before the softmax::

            softmax over surviving subset S
                = exp(e_t) / sum_S exp(e)
                = a_t / sum_S a                 = dropped / dropped.sum()

        The inverted-dropout 1/(1-p) factor cancels in the ratio, so the result
        does not depend on p's scaling at all. The operation is therefore
        precisely "re-run attention treating the dropped frames as padding",
        which is the same semantics as the masking path above rather than a
        separate approximation of it.

        Args:
            weights: Normalised attention weights of shape ``(B, T)``.

        Returns:
            Weights of shape ``(B, T)`` summing to one along dim 1 -- except for
            a fully-masked row, which stays all-zero.
        """
        dropped = self.dropout(weights)
        total = dropped.sum(dim=1, keepdim=True)
        # The threshold in the condition and the floor in the divisor MUST be
        # the same constant. With `total > 0` against a 1e-5 floor, any row with
        # 0 < total < 1e-5 selects the division branch while the clamp changes
        # the divisor -- yielding weights that sum to total/1e-5, violating the
        # exact invariant this function exists to provide. Reachable when
        # dropout leaves only far-tail frames of a peaked distribution.
        #
        # clamp_min is still required even though the condition already excludes
        # the small case: torch.where evaluates BOTH branches, so an unclamped
        # division would produce inf in the discarded branch and poison the
        # gradient despite never being selected.
        return torch.where(
            total > _WEIGHT_SUM_EPS,
            dropped / total.clamp_min(_WEIGHT_SUM_EPS),
            weights,
        )

    def forward(
        self,
        frames: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            frames: Float tensor of shape ``(B, T, D)``.
            mask: Bool tensor ``(B, T)``, True where the frame is real.

        Returns:
            Tuple of (pooled ``(B, D)``, weights ``(B, T)``). The returned
            weights are the **undropped** distribution -- in train mode the
            vector is pooled with a dropped-and-renormalised copy, so the
            pooled output cannot be reconstructed from them. Plot them from
            eval mode, where dropout is off and the two coincide.
        """
        mask = _check_mask(frames, mask)
        weights = self._attend(frames, mask)
        w = self._drop_and_renormalise(weights)
        pooled = torch.bmm(w.unsqueeze(1), frames).squeeze(1)
        return pooled, weights


class AttentiveStatsPooling(AttentionPooling):
    """Attentive statistics pooling — attention-weighted mean AND std.

    Okabe et al., *Attentive Statistics Pooling for Deep Speaker Embedding*
    (Interspeech 2018). Concatenates the weighted mean with the weighted
    standard deviation, so ``output_dim = 2 * input_dim``.

    Relevant here because the std term retains prosodic *variability*, which a
    mean discards entirely — and variability of pitch and energy is part of how
    affect is realised in speech.

    Args:
        input_dim: Frame dimension ``D``.
        attention_dim: Width of the scoring projection.
        dropout: Dropout applied to attention weights during training.
    """

    @property
    def output_dim(self) -> int:
        return 2 * self.input_dim

    def forward(
        self,
        frames: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            frames: Float tensor of shape ``(B, T, D)``.
            mask: Bool tensor ``(B, T)``, True where the frame is real.

        Returns:
            Tuple of (pooled ``(B, 2*D)``, weights ``(B, T)``). As above, the
            returned weights are the **undropped** distribution.
        """
        mask = _check_mask(frames, mask)
        weights = self._attend(frames, mask)
        # Renormalised: the variance identity below is only valid for weights
        # that sum to exactly one. See _drop_and_renormalise.
        w = self._drop_and_renormalise(weights).unsqueeze(-1)  # (B, T, 1)

        mean = (w * frames).sum(dim=1)                         # (B, D)
        # E[x^2] - E[x]^2 under the same attention distribution. clamp_min
        # before sqrt guards fp16/bf16 rounding only -- with normalised weights
        # the identity is non-negative in exact arithmetic, so a clamp firing
        # here now indicates precision loss rather than a broken distribution.
        var = (w * frames.pow(2)).sum(dim=1) - mean.pow(2)
        std = var.clamp_min(_VAR_EPS).sqrt()

        # A fully-masked row has zero weights, so mean and var are both 0 --
        # but clamp_min then makes std sqrt(_VAR_EPS), a nonzero constant. Zero
        # the whole row so the degenerate case matches MaskedMeanPooling and
        # AttentionPooling, which both return an all-zero vector for it.
        valid = mask.any(dim=1, keepdim=True).to(frames.dtype)
        return torch.cat([mean, std], dim=1) * valid, weights


#: Registry so a config string selects the pooler — no magic strings in models.
#: ``attention_fixed`` is ``attention`` frozen at random init: same shape, same
#: forward cost, no learning. It is the control that separates "non-uniform
#: weighting" from "*learned* non-uniform weighting" (ADR-003).
POOLING_REGISTRY = {
    "masked_mean": MaskedMeanPooling,
    "attention_fixed": AttentionPooling,
    "attention": AttentionPooling,
    "attentive_stats": AttentiveStatsPooling,
}

#: Poolers instantiated with frozen parameters.
_FROZEN_POOLERS = frozenset({"attention_fixed"})


def build_pooler(
    name: str,
    input_dim: int,
    attention_dim: int = 128,
    dropout: float = 0.0,
) -> nn.Module:
    """Instantiate a pooler by config name.

    Args:
        name: One of ``masked_mean``, ``attention_fixed``, ``attention``,
            ``attentive_stats``.
        input_dim: Frame dimension ``D``.
        attention_dim: Width of the scoring projection (attention poolers only).
        dropout: Dropout on attention weights (attention poolers only).

    Returns:
        The constructed pooling module, exposing ``.output_dim``.

    Raises:
        KeyError: If ``name`` is not a registered pooler.
    """
    if name not in POOLING_REGISTRY:
        raise KeyError(
            f"Unknown pooling '{name}'. Options: {sorted(POOLING_REGISTRY)}"
        )
    cls = POOLING_REGISTRY[name]
    if cls is MaskedMeanPooling:
        return cls(input_dim)
    return cls(
        input_dim,
        attention_dim=attention_dim,
        dropout=dropout,
        trainable=name not in _FROZEN_POOLERS,
    )


def count_trainable(module: nn.Module) -> int:
    """Trainable parameter count — used to verify an ablation is capacity-matched.

    Args:
        module: Any module.

    Returns:
        Number of parameters with ``requires_grad=True``.
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
