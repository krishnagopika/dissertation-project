"""Context THEN fusion: contextualise each modality separately, then combine.

The ordering question
---------------------
The stacked model fused first and contextualised second::

    acoustic ─┐
              ├─► fuse ─► 512-d ─► BiLSTM over dialogue ─► classify
    text ─────┘

That did not work. `concat` mode kept the raw features alongside the fused
vector (2560-d, more parameters than plain) and still failed to beat a plain
bc-LSTM on gold, so the fused vector carries essentially nothing the context
model cannot already get from the raw features.

This module reverses the order::

    acoustic ─► BiLSTM_a over dialogue ─┐
                                         ├─► fuse ─► classify
    text ─────► BiLSTM_t over dialogue ─┘

Three reasons to expect this to behave differently.

**Compression happens last.** Fusion projects to a common width against a
per-utterance objective. Doing that first destroys exactly the information the
temporal model needed -- which is what the stacked results measured. Here the
temporal model sees the uncompressed per-modality features and fusion compresses
afterwards, where the objective matches.

**Context is modality-specific.** How prosody evolves across a conversation is
not how lexical content evolves: a raised voice three turns ago carries
different information from a word three turns ago. One shared BiLSTM over a
blended vector cannot represent two different temporal dynamics; two can.

**Modality identity survives the temporal model.** Once fused, the recurrent
layer cannot tell which dimensions came from audio, so it cannot learn to
weight one channel differently across a stretch of dialogue.

The ablation this is built for
------------------------------
``use_text_lstm`` and ``use_acoustic_lstm`` are independent, giving a 2x2:

    ================  ==================  ======================
    acoustic LSTM     text LSTM           what it isolates
    ================  ==================  ======================
    on                on                  both contextualised
    on                off                 acoustic-only context
    off               on                  text-only context
    off               off                 no context (= plain fusion)
    ================  ==================  ======================

The bottom row is the control: with both off this reduces to per-utterance
fusion, so any gain in the other three is attributable to context alone.

WHAT THE 2x2 ACTUALLY FOUND (12 runs, single seed)

The prediction going in was text-only ~ both and acoustic-only ~ neither: the
acoustic cache is byte-identical across gold/asr/asr_cleaned, yet dialogue
context only helped when the text was clean, which looked like a text-mediated
effect. That prediction was WRONG, and its refutation is instructive.

On DEV, acoustic-only was the strongest arm on both ASR conditions and helped
most where transcription was worst (asr +0.0156 over the no-context control),
while text context went slightly negative on asr_cleaned. That looked like a
clean robustness result.

On TEST it did not replicate. asr/acoustic-only went from best on dev (+0.0156)
to WORST in its row (-0.0114). Every test delta across all 12 cells sits within
+/-0.012, which for one seed is noise.

So the honest reading is that this architecture shows NO reliable benefit over
per-utterance fusion, and that the dev-based ordering was not real. The only
effect that survives is the gold->ASR gap of ~0.12, which dwarfs every
architectural choice measured here.

Multiple seeds (3-5 per cell, mean +/- std) are required before any ordering in
this table can be claimed. Do not quote a single-seed delta from it.

Parameter fairness
------------------
Turning off one BiLSTM removes that modality's recurrent capacity, so a loss in
a smaller arm may be capacity rather than architecture.

The knob is ``lstm_hidden``, NOT ``hidden_dim``. ``hidden_dim`` is the fusion
width -- changing it resizes the projections and ``post``, and does not touch
the BiLSTMs at all.

Scaling is SUB-quadratic in ``lstm_hidden`` here. An LSTM has
``2 * 4 * (input*h + h*h + 2h)`` parameters, and with ``input_dim`` of 768/1280
against ``h`` of 256 the ``input*h`` term dominates: doubling h from 128 to 256
takes the text LSTM from 919,552 to 2,101,248, a factor of 2.28 rather than 4.
So a "divide h by sqrt(2)" rule is the right ballpark for the wrong reason and
should not be relied on.

Measured matched pair at the default:

    acoustic_only, lstm_hidden=256  ->  4,339,210 parameters
    both,          lstm_hidden=180  ->  4,375,946 parameters   (+0.8%)

Pick the value empirically from ``parameter_report()`` rather than analytically.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class _DialogueBiLSTM(nn.Module):
    """BiLSTM over a dialogue's utterance sequence for ONE modality."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int,
                 dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            # PyTorch ignores LSTM dropout at one layer and warns; guard it.
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dim = hidden_dim * 2

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        """
        Args:
            x: ``(B, T, input_dim)`` padded utterance features.
            lengths: ``(B,)`` real dialogue lengths.

        Returns:
            ``(B, T, 2 * hidden_dim)``.
        """
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.lstm(packed)
        # total_length keeps the padded shape so logits stay aligned with the
        # label tensor even when a batch's longest dialogue is short.
        out, _ = pad_packed_sequence(out, batch_first=True,
                                     total_length=x.size(1))
        return out


class ContextThenFusion(nn.Module):
    """Per-modality dialogue context, then fusion, then per-utterance heads.

    Args:
        text_dim: Width of the cached text vectors.
        acoustic_dim: Width of the cached (already pooled) acoustic vectors.
        hidden_dim: Common fusion width, and the projection target.
        lstm_hidden: BiLSTM hidden size PER DIRECTION, per modality.
        num_layers: Stacked LSTM layers per modality.
        num_emotion_classes: Emotion head output size.
        num_sentiment_classes: Sentiment head output size.
        dropout: Dropout throughout.
        fusion: One of ``concat``, ``sum``, ``gated``, ``crossmodal``.
        use_text_lstm: Contextualise the text branch.
        use_acoustic_lstm: Contextualise the acoustic branch.

    Raises:
        ValueError: On an unknown fusion name.
    """

    FUSIONS = ("concat", "sum", "gated", "crossmodal")

    def __init__(
        self,
        text_dim: int = 768,
        acoustic_dim: int = 1280,
        hidden_dim: int = 512,
        lstm_hidden: int = 256,
        num_layers: int = 1,
        num_emotion_classes: int = 7,
        num_sentiment_classes: int = 3,
        dropout: float = 0.3,
        fusion: str = "concat",
        use_text_lstm: bool = True,
        use_acoustic_lstm: bool = True,
    ) -> None:
        super().__init__()
        if fusion not in self.FUSIONS:
            raise ValueError(f"fusion must be one of {self.FUSIONS}, got {fusion!r}")
        self.fusion_name = fusion
        self.use_text_lstm = use_text_lstm
        self.use_acoustic_lstm = use_acoustic_lstm

        self.text_lstm = (_DialogueBiLSTM(text_dim, lstm_hidden, num_layers, dropout)
                          if use_text_lstm else None)
        self.acoustic_lstm = (_DialogueBiLSTM(acoustic_dim, lstm_hidden, num_layers, dropout)
                              if use_acoustic_lstm else None)

        # A branch without a BiLSTM is projected from its raw width, so the two
        # ablation arms differ only in whether the temporal layer is present --
        # not in whether the branch reaches the fusion stage at all.
        #
        # Caveat worth stating rather than hiding: an "off" branch projects from
        # 768 (text) or 1280 (acoustic), an "on" branch from 2*lstm_hidden=512.
        # So the projection layer is also differently sized between arms, and
        # "off" is not EXACTLY "on minus the LSTM". Avoiding that would require
        # padding the raw features to 512, which introduces its own artefact.
        # The difference is small relative to the LSTM itself and is visible in
        # parameter_report().
        t_in = self.text_lstm.output_dim if use_text_lstm else text_dim
        a_in = self.acoustic_lstm.output_dim if use_acoustic_lstm else acoustic_dim

        def _proj(d_in: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(d_in, hidden_dim), nn.LayerNorm(hidden_dim),
                nn.GELU(), nn.Dropout(p=dropout))

        self.text_proj = _proj(t_in)
        self.acoustic_proj = _proj(a_in)

        combine_in = hidden_dim * 2 if fusion == "concat" else hidden_dim
        if fusion == "gated":
            # Symmetric convex gate: g*t + (1-g)*a. Weights sum to 1, so this
            # is a soft SELECTION between modalities.
            #
            # _init_weights zeroes the bias, so sigmoid(0)=0.5 and training
            # STARTS at an exact mean of the two modalities. That is a
            # deliberate neutral prior, not an accident: the model has to learn
            # its way away from equal weighting rather than toward it.
            self.gate = nn.Linear(hidden_dim * 2, hidden_dim)
        elif fusion == "crossmodal":
            # Asymmetric: each modality gates the OTHER, gates independent.
            # Not convex -- both can be suppressed or both passed through.
            #
            # NOT cross-ATTENTION. The gating is elementwise and time-local:
            # utterance i's gate is computed from utterance i's other modality
            # only, with no interaction across the dialogue. Temporal mixing is
            # the BiLSTMs' job and happens strictly before this point.
            self.gate_t_from_a = nn.Linear(hidden_dim, hidden_dim)
            self.gate_a_from_t = nn.Linear(hidden_dim, hidden_dim)

        self.post = nn.Sequential(
            nn.Linear(combine_in, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(p=dropout))
        self.emotion_head = nn.Linear(hidden_dim, num_emotion_classes)
        self.sentiment_head = nn.Linear(hidden_dim, num_sentiment_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _combine(self, t: Tensor, a: Tensor) -> Tensor:
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

    def represent(self, text: Tensor, acoustic: Tensor, lengths: Tensor) -> Tensor:
        """Fused, contextualised per-utterance vectors, before the heads."""
        t = self.text_lstm(text, lengths) if self.text_lstm is not None else text
        a = (self.acoustic_lstm(acoustic, lengths)
             if self.acoustic_lstm is not None else acoustic)
        return self.post(self._combine(self.text_proj(t), self.acoustic_proj(a)))

    def forward(
        self, text: Tensor, acoustic: Tensor, lengths: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            text: ``(B, T, text_dim)`` padded per-utterance text vectors.
            acoustic: ``(B, T, acoustic_dim)`` padded pooled acoustic vectors.
            lengths: ``(B,)`` real dialogue lengths.

        Returns:
            ``(sentiment_logits, emotion_logits)``, each ``(B, T, C)`` --
            ordered to match BiLSTMContext so the trainers stay interchangeable.
        """
        fused = self.represent(text, acoustic, lengths)
        return self.sentiment_head(fused), self.emotion_head(fused)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_report(self) -> Dict[str, int]:
        """Per-component counts, so a gain can be weighed against capacity.

        Turning off one BiLSTM removes roughly half the recurrent parameters,
        so the ablation arms are NOT capacity-matched by default.
        """
        def n(m: Optional[nn.Module]) -> int:
            return 0 if m is None else sum(p.numel() for p in m.parameters()
                                           if p.requires_grad)
        rep = {
            "text_lstm": n(self.text_lstm),
            "acoustic_lstm": n(self.acoustic_lstm),
            "text_proj": n(self.text_proj),
            "acoustic_proj": n(self.acoustic_proj),
            "post": n(self.post),
            "emotion_head": n(self.emotion_head),
            "sentiment_head": n(self.sentiment_head),
        }
        for attr in ("gate", "gate_t_from_a", "gate_a_from_t"):
            if hasattr(self, attr):
                rep[attr] = n(getattr(self, attr))
        rep["TOTAL"] = self.trainable_parameters()
        return rep
