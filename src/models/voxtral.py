"""
Voxtral Wrapper
===============
Wrapper around Mistral's Voxtral models for acoustic embedding extraction.

**Superseded for MELD preprocessing.** transcribe_all.py now captures acoustics
during the vllm transcription pass via a forward hook (CR-003), so Voxtral is
loaded once rather than twice. This module remains for RAVDESS extraction, for
standalone use, and as the reference implementation the single-pass hook must
agree with. It is never loaded during training.

Architecture
------------
VoxtralForConditionalGeneration audio path::

    Raw Audio  →  WhisperFeatureExtractor (mel spectrogram, padded to 30 s)
                       ↓
    [VoxtralEncoder / audio_tower]   Whisper large-v3, 50 Hz, 1280-dim
                       ↓                       <-- WE STOP HERE
    mean-pool over time   →   (batch, 1280)
                       ↓
    [projector: ×4 downsample → AudioLanguageAdapter]   (not used)
                       ↓  (output dim = text_config.hidden_size, 3072 for Mini)
    audio tokens the LLM attends over

Extraction takes the **encoder** output, NOT the projector output. The projector
is trained to make audio look like text tokens to the LLM, so it is free to
discard the paralinguistic detail emotion recognition depends on; it also
downsamples ×4, from 50 Hz to 12.5 Hz. See docs/DECISIONS.md ADR-001.

Note ``model.get_audio_features()`` runs encoder **and** projector, so it is
deliberately NOT used — ``self.model.audio_tower(...)`` is called directly.

Embedding dimension
-------------------
1280 for both Mini and Small: they wrap the same Whisper large-v3 encoder, so
there is no per-model value. Use ``acoustic_dim: 1280``.

Caveat: the mean is taken over all 1500 padded encoder frames with no mask, so
for a typical MELD utterance (~2.7 s) roughly 91% of the average is padding.
That defect is inherent to pooling raw encoder output and is why pooling moved
into the trainable head — see ADR-003 and src/models/pooling.py.

Usage
-----
>>> wrapper = VoxtralWrapper("mistralai/Voxtral-Mini-3B-2507")
>>> waveform  # float32 Tensor, shape (T,), 16 kHz
>>> emb = wrapper.extract_acoustic_embeddings(waveform, sample_rate=16000)
>>> # emb: (1, 1280) float32 on CPU
>>> dim = wrapper.get_acoustic_hidden_dim()   # 1280
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

# Fallback if config introspection fails. This is the WHISPER ENCODER width
# (audio_config.d_model), because extract_acoustic_embeddings returns encoder
# output — see ADR-001. Both Mini and Small wrap Whisper large-v3, so both are
# 1280; there is no per-model variant to keep.
#
# Previously this was 4096 with a second, never-referenced _SMALL_FALLBACK_DIM
# beside it. Both were wrong twice over: they named the LLM width rather than
# the encoder width, and Mini's LLM width is 3072, not 4096 (params.json:
# "dim": 3072).
_ENCODER_FALLBACK_DIM: int = 1280

VOXTRAL_MINI = "mistralai/Voxtral-Mini-3B-2507"
VOXTRAL_SMALL = "mistralai/Voxtral-Small-24B-2507"


class VoxtralWrapper(nn.Module):
    """Thin wrapper around HuggingFace ``VoxtralForConditionalGeneration``.

    Provides a stable API for acoustic embedding extraction independent of
    transformers internals. Embedding extraction uses
    ``model.get_audio_features(input_features)`` directly — no forward hooks.

    Args:
        model_name_or_path: HuggingFace model ID or local path.
        device_map: Passed to ``from_pretrained``. Use ``"auto"`` for
            automatic multi-GPU sharding (required for Small on 2× L40S).
        torch_dtype: Floating-point dtype for model weights.
            Defaults to ``torch.bfloat16`` (recommended on L40S).
        cache_dir: Optional HuggingFace cache directory override.
        load_in_8bit: Enable bitsandbytes 8-bit quantisation.
        load_in_4bit: Enable bitsandbytes 4-bit (NF4) quantisation.
        attn_implementation: Attention backend.  ``"sdpa"`` is safe on all
            Ampere/Ada GPUs; ``"flash_attention_2"`` requires flash-attn.
    """

    def __init__(
        self,
        model_name_or_path: str = VOXTRAL_MINI,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        cache_dir: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        attn_implementation: str = "sdpa",
    ) -> None:
        super().__init__()

        self.model_name_or_path = model_name_or_path
        self.torch_dtype = torch_dtype

        logger.info("Loading Voxtral model: %s", model_name_or_path)

        # ------------------------------------------------------------------
        # Model loading
        # VoxtralForConditionalGeneration is the correct class for this model.
        # Note: kwarg is `dtype=`, not `torch_dtype=`.
        # ------------------------------------------------------------------
        from transformers import AutoProcessor, VoxtralForConditionalGeneration

        quantization_config = None
        if load_in_8bit or load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=load_in_8bit,
                    load_in_4bit=load_in_4bit,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError:
                logger.warning(
                    "bitsandbytes not installed; ignoring quantisation flags."
                )

        model_kwargs: dict = dict(
            cache_dir=cache_dir,
            device_map=device_map,
            dtype=torch_dtype,          # VoxtralForConditionalGeneration uses `dtype=`
            attn_implementation=attn_implementation,
        )
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        self.model = VoxtralForConditionalGeneration.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
        )

        logger.info(
            "Voxtral loaded. audio_tower: %s | projector: %s",
            type(self.model.audio_tower).__name__,
            type(self.model.multi_modal_projector).__name__,
        )

    # ------------------------------------------------------------------
    # Audio preprocessing
    # ------------------------------------------------------------------

    def _prepare_input_features(
        self,
        audio_tensor: Tensor,
        sample_rate: int,
    ) -> Tensor:
        """Extract mel spectrogram features from a raw waveform.

        Args:
            audio_tensor: Float32 waveform, shape ``(T,)`` or ``(B, T)``.
            sample_rate: Sample rate of the audio (should be 16 kHz).

        Returns:
            ``input_features`` tensor of shape ``(B, n_mels, seq_len)``
            on the model's device.
        """
        if audio_tensor.dim() == 1:
            audios = [audio_tensor.float().cpu().numpy()]
        else:
            audios = [
                audio_tensor[i].float().cpu().numpy()
                for i in range(audio_tensor.shape[0])
            ]

        fe_out = self.processor.feature_extractor(
            audios,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding="max_length",  # always pad to 3000 mel frames (30s) — required by audio tower
            truncation=True,
        )
        input_features: Tensor = fe_out["input_features"]

        # Move to the model's device (first parameter gives the primary device)
        device = next(self.model.parameters()).device
        return input_features.to(device=device, dtype=self.torch_dtype)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def extract_acoustic_embeddings(
        self,
        audio_tensor: Tensor,
        sample_rate: int = 16_000,
    ) -> Tensor:
        """Extract mean-pooled acoustic embeddings from the MLP projector.

        Calls ``model.get_audio_features(input_features)`` which runs:
          Whisper encoder → VoxtralMultiModalProjector → returns
          ``(N, text_hidden_size)`` where N = batch × encoder_seq_len.

        The result is mean-pooled over the time dimension to produce a
        fixed-size ``(batch, text_hidden_size)`` vector per audio clip.

        Args:
            audio_tensor: Float32 waveform at ``sample_rate`` Hz.
                Shape ``(T,)`` for a single clip or ``(B, T)`` for a batch.
            sample_rate: Audio sample rate. Should be 16 kHz.

        Returns:
            Float32 CPU tensor of shape ``(batch, text_hidden_size)``.
            ``text_hidden_size`` ≈ 4096 for both Mini and Small.
        """
        input_features = self._prepare_input_features(audio_tensor, sample_rate)

        # Whisper encoder only — stop before the MLP projector
        encoder_out = self.model.audio_tower(input_features)
        hidden: Tensor = encoder_out.last_hidden_state  # (B, seq_len, 1280)

        # Mean-pool over time (dim=1) → one vector per clip → (B, 1280)
        pooled: Tensor = hidden.mean(dim=1)
        return pooled.float().cpu()

    @torch.inference_mode()
    def transcribe(
        self,
        audio_path: Union[str, Path],
        max_new_tokens: int = 448,
    ) -> str:
        """Transcribe an audio file to text using the HF model.

        Note: In the dissertation pipeline, Pass 1 transcription is handled
        by vllm (faster). This method is provided for standalone use / testing.

        Args:
            audio_path: Path to an audio file (.wav, .mp4, etc.).
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Transcript string.
        """
        audio_path = str(Path(audio_path).resolve())
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "url": f"file://{audio_path}",
                    },
                    {
                        "type": "text",
                        "text": "Write the exact words spoken in the audio. Do not infer, do not paraphrase, and do not add any extra information.",
                    },
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            return_dict=True,
            return_tensors="pt",
            tokenize=True,
        )

        device = next(self.model.parameters()).device
        inputs = {
            k: v.to(device) if isinstance(v, Tensor) else v
            for k, v in inputs.items()
        }

        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        # Decode only the newly generated tokens
        new_ids = generated_ids[:, inputs["input_ids"].shape[1]:]
        transcript: str = self.processor.batch_decode(
            new_ids, skip_special_tokens=True
        )[0]
        return transcript.strip()

    def get_acoustic_hidden_dim(self) -> int:
        """Return the dimension :meth:`extract_acoustic_embeddings` actually emits.

        That is the **Whisper encoder** width (``audio_config.d_model``), 1280 for
        both Mini and Small — not the LLM width. See ADR-001 for why extraction
        happens at the encoder rather than after the projector.

        This previously returned ``text_config.hidden_size`` — the LLM width, a
        dimension this class never produces. That mattered because
        transcribe_all.py treats this value as authoritative and overrides the
        config with it::

            actual_dim = wrapper.get_acoustic_hidden_dim()
            if actual_dim != acoustic_dim:
                acoustic_dim = actual_dim      # <- silently wrong

        ``acoustic_dim`` is then the width used to zero-fill clips whose audio is
        missing or whose extraction raised, so a failed clip could have landed in
        the cache with a different shape from every successful one. Verified not
        to have happened: all 13,708 legacy entries are (1280,), and the two
        all-zero vectors match the two clips DATA_INVENTORY.md records as lost.
        Latent, not manifested — fixed so it stays that way.

        Returns:
            Encoder hidden dimension, e.g. 1280 for Voxtral-Mini-3B-2507.
        """
        for obj, attr in (
            (getattr(self.model.config, "audio_config", None), "d_model"),
            (getattr(self.model.config, "audio_config", None), "hidden_size"),
            (getattr(self.model.audio_tower, "config", None), "d_model"),
        ):
            if obj is not None and hasattr(obj, attr):
                return int(getattr(obj, attr))

        logger.warning(
            "Could not infer the encoder dimension from the model config; "
            "using fallback %d. Verify against the model's audio_config.",
            _ENCODER_FALLBACK_DIM,
        )
        return _ENCODER_FALLBACK_DIM

    # ------------------------------------------------------------------
    # Convenience classmethod
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = VOXTRAL_MINI,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> "VoxtralWrapper":
        """Alias for the constructor.

        Args:
            model_name_or_path: HuggingFace model ID or local path.
            device_map: Device placement strategy.
            torch_dtype: Model floating-point dtype.
            **kwargs: Additional arguments forwarded to ``VoxtralWrapper.__init__``.

        Returns:
            VoxtralWrapper instance.
        """
        return cls(
            model_name_or_path=model_name_or_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            **kwargs,
        )
