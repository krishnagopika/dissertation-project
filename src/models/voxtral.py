"""
Voxtral Wrapper
===============
Wrapper around Mistral's Voxtral models for acoustic embedding extraction.

Used exclusively for **Pass 2** of preprocessing: extract fixed-size acoustic
embeddings from cached MELD audio files. Pass 1 (transcription) is handled
separately by vllm — this module is never loaded during training.

Architecture
------------
VoxtralForConditionalGeneration pipeline::

    Raw Audio  →  WhisperFeatureExtractor (mel spectrogram)
                       ↓
    [VoxtralEncoder / audio_tower]   Whisper Large-v3, 50 Hz, 1280-dim
                       ↓
    [VoxtralMultiModalProjector]     linear_1 → act → linear_2
                       ↓  (output dim = text_config.hidden_size)
    mean-pool over time   →   fixed-size embedding (batch, hidden_size)

``model.get_audio_features(input_features)`` encapsulates the encoder +
projector pass and is called directly — no forward hooks required.

Embedding dimensions
--------------------
* ``mistralai/Voxtral-Mini-3B-2507``   → ``text_config.hidden_size`` ≈ 4096
  Use ``acoustic_dim: 4096`` in config (or let the wrapper report the real dim).
* ``mistralai/Voxtral-Small-24B-2507`` → larger; requires 2× L40S via
  ``device_map="auto"``.

Usage
-----
>>> wrapper = VoxtralWrapper("mistralai/Voxtral-Mini-3B-2507")
>>> waveform  # float32 Tensor, shape (T,), 16 kHz
>>> emb = wrapper.extract_acoustic_embeddings(waveform, sample_rate=16000)
>>> # emb: (1, hidden_size) float32 on CPU
>>> dim = wrapper.get_acoustic_hidden_dim()   # e.g. 4096
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

# Per-model fallback dims if config introspection fails.
_MINI_FALLBACK_DIM: int = 4096   # Voxtral-Mini-3B text_config.hidden_size
_SMALL_FALLBACK_DIM: int = 4096  # Voxtral-Small-24B (same LLM size)

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
        """Return the acoustic embedding dimension (MLP projector output dim).

        This equals ``text_config.hidden_size`` for both Mini and Small.

        Returns:
            Integer dimension, e.g. 4096 for Voxtral-Mini-3B-2507.
        """
        try:
            return int(self.model.config.text_config.hidden_size)
        except AttributeError:
            pass

        # Fallback: try direct hidden_size on config
        try:
            return int(self.model.config.hidden_size)
        except AttributeError:
            pass

        logger.warning(
            "Could not infer acoustic_dim from model config; "
            "using fallback %d. Check config and update if wrong.",
            _MINI_FALLBACK_DIM,
        )
        return _MINI_FALLBACK_DIM

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
