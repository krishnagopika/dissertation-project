"""
Voxtral Wrapper
===============
Wrapper around Mistral's Voxtral models for two tasks:

1. **ASR / transcription** — :meth:`VoxtralWrapper.transcribe` returns the
   text transcript for one or more audio waveforms.
2. **Acoustic embedding extraction** — :meth:`VoxtralWrapper.extract_acoustic_embeddings`
   hooks into the Whisper Large-v3 encoder that backs Voxtral's audio front-end
   and returns mean-pooled hidden states from the final encoder layer.
   Output shape: ``(batch_size, hidden_dim)``.

Supported models
----------------
* ``mistralai/Voxtral-Mini-3B-2507``  (3B parameters, fits on 1× L40S)
* ``mistralai/Voxtral-Small-24B-2507`` (24B parameters, requires 2× L40S via
  ``device_map="auto"``)

Architecture notes
------------------
Voxtral is composed of a **Whisper Large-v3 encoder** (audio front-end) and a
**Mistral LLM decoder** (language model). The encoder processes mel-spectrogram
features and produces contextualised acoustic representations; the decoder
attends to these via cross-attention to generate text tokens.

The path to the encoder attribute differs between the Mini and Small variants
because they are implemented under slightly different model class names. This
module uses ``try/except`` fallback chains to handle the differences gracefully.

Cluster notes (Warwick WMLG, wmlg-ada, L40S)
---------------------------------------------
* Mini:  1 GPU, ``device_map="auto"``  or explicit ``device="cuda:0"``
* Small: 2 GPUs, ``device_map="auto"`` (model is sharded automatically by
  ``transformers``; ensure 2 GPUs are visible via ``CUDA_VISIBLE_DEVICES``)

Usage example
-------------
>>> wrapper = VoxtralWrapper("mistralai/Voxtral-Mini-3B-2507")
>>> # audio_tensor: (batch, samples) or (samples,) at 16 kHz
>>> transcript = wrapper.transcribe(audio_tensor, sample_rate=16000)
>>> embeddings = wrapper.extract_acoustic_embeddings(audio_tensor, sample_rate=16000)
>>> # embeddings shape: (batch, 1280)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

# Hidden dimension of the Whisper Large-v3 encoder
WHISPER_LARGE_V3_HIDDEN_DIM: int = 1280

# Supported model identifiers
VOXTRAL_MINI = "mistralai/Voxtral-Mini-3B-2507"
VOXTRAL_SMALL = "mistralai/Voxtral-Small-24B-2507"


class VoxtralWrapper(nn.Module):
    """Thin wrapper around a HuggingFace Voxtral model.

    Parameters
    ----------
    model_name_or_path:
        HuggingFace model identifier or local path.
        Typical values: ``VOXTRAL_MINI`` or ``VOXTRAL_SMALL``.
    device_map:
        Passed directly to ``AutoModelForSpeechSeq2Seq.from_pretrained``.
        Use ``"auto"`` for automatic multi-GPU sharding.
    torch_dtype:
        Model dtype. Defaults to ``torch.bfloat16`` for memory efficiency on
        L40S GPUs.
    cache_dir:
        Optional HuggingFace cache directory.
    load_in_8bit:
        Whether to use bitsandbytes 8-bit quantisation. Reduces VRAM at some
        cost to accuracy. Requires ``bitsandbytes`` to be installed.
    load_in_4bit:
        Whether to use bitsandbytes 4-bit (NF4) quantisation. Further reduces
        VRAM. Requires ``bitsandbytes`` to be installed.
    attn_implementation:
        Attention backend. ``"flash_attention_2"`` is recommended on Ampere/
        Ada GPUs if ``flash-attn`` is installed; otherwise use ``"sdpa"``
        (PyTorch scaled-dot-product attention) or ``"eager"``.
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

        # Lazily populated by _register_encoder_hook()
        self._encoder_hook_handle = None
        self._captured_encoder_output: Optional[Tensor] = None

        logger.info("Loading Voxtral model: %s", model_name_or_path)

        # ------------------------------------------------------------------
        # Model loading
        # ------------------------------------------------------------------
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

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
                    "bitsandbytes is not installed; ignoring quantisation flags."
                )

        model_kwargs = dict(
            cache_dir=cache_dir,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
        )

        # Resolve and cache a reference to the Whisper encoder sub-module
        self._encoder: nn.Module = self._resolve_encoder()
        logger.info(
            "Resolved Whisper encoder: %s", type(self._encoder).__name__
        )

    # ------------------------------------------------------------------
    # Encoder resolution
    # ------------------------------------------------------------------

    def _resolve_encoder(self) -> nn.Module:
        """Attempt to find the Whisper encoder in the model graph.

        Different Voxtral variants (Mini / Small) may organise sub-modules
        under slightly different attribute paths. We try a priority list of
        known paths, falling back to a breadth-first search if needed.
        """
        # Priority attribute paths — Mini and Small variants
        candidate_paths: List[List[str]] = [
            ["model", "encoder"],            # most common for Seq2Seq models
            ["model", "audio_encoder"],
            ["encoder"],
            ["model", "model", "encoder"],
            ["audio_tower"],
            ["model", "audio_tower"],
        ]

        for path in candidate_paths:
            obj = self.model
            try:
                for attr in path:
                    obj = getattr(obj, attr)
                if isinstance(obj, nn.Module):
                    return obj
            except AttributeError:
                continue

        # Fallback: breadth-first search for any module whose class name
        # contains "WhisperEncoder"
        for name, module in self.model.named_modules():
            if "WhisperEncoder" in type(module).__name__:
                logger.debug("Found encoder via BFS at: %s", name)
                return module

        raise RuntimeError(
            "Could not locate the Whisper encoder sub-module inside the Voxtral "
            "model. Please check the model architecture and update the candidate "
            "paths in VoxtralWrapper._resolve_encoder()."
        )

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _encoder_forward_hook(
        self,
        module: nn.Module,
        input: Tuple,
        output,
    ) -> None:
        """Forward hook that captures the encoder's last hidden state."""
        # The encoder output can be a tensor, a tuple, or a
        # BaseModelOutput-like object with a ``last_hidden_state`` attribute.
        if isinstance(output, Tensor):
            self._captured_encoder_output = output.detach()
        elif hasattr(output, "last_hidden_state"):
            self._captured_encoder_output = output.last_hidden_state.detach()
        elif isinstance(output, (tuple, list)) and len(output) > 0:
            # The first element of Whisper encoder output is the hidden states
            self._captured_encoder_output = output[0].detach()
        else:
            logger.warning(
                "Unrecognised encoder output type: %s. "
                "Acoustic embeddings may be unavailable.",
                type(output).__name__,
            )

    @contextmanager
    def _hook_encoder(self):
        """Context manager that registers/removes the encoder forward hook."""
        self._captured_encoder_output = None
        handle = self._encoder.register_forward_hook(self._encoder_forward_hook)
        try:
            yield
        finally:
            handle.remove()
            # Do NOT clear _captured_encoder_output here so callers can
            # retrieve it after the context exits.

    # ------------------------------------------------------------------
    # Audio preprocessing
    # ------------------------------------------------------------------

    def _preprocess_audio(
        self,
        audio_tensor: Tensor,
        sample_rate: int,
    ) -> dict:
        """Convert a raw waveform into processor inputs.

        Parameters
        ----------
        audio_tensor:
            Float32 waveform tensor. Shape ``(T,)`` or ``(B, T)``. Audio
            should already be at 16 kHz; if not, the processor will attempt
            to handle resampling via ``sampling_rate``.
        sample_rate:
            Sample rate of the input audio.

        Returns
        -------
        dict of tensors ready to be passed to ``self.model.generate()``.
        """
        # Normalise to list-of-numpy for the HuggingFace processor
        if audio_tensor.dim() == 1:
            audios = [audio_tensor.float().cpu().numpy()]
        else:
            audios = [audio_tensor[i].float().cpu().numpy() for i in range(audio_tensor.shape[0])]

        inputs = self.processor(
            audios,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )

        # Move inputs to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in inputs.items()}
        return inputs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def transcribe(
        self,
        audio_tensor: Tensor,
        sample_rate: int = 16_000,
        language: str = "en",
        max_new_tokens: int = 448,
    ) -> Union[str, List[str]]:
        """Transcribe one or more audio waveforms to text.

        Parameters
        ----------
        audio_tensor:
            Raw waveform tensor at ``sample_rate`` Hz. Shape ``(T,)`` for a
            single clip, or ``(B, T)`` for a batch.
        sample_rate:
            Sample rate of the audio. Defaults to 16 kHz.
        language:
            Target transcription language code (e.g. ``"en"``).
        max_new_tokens:
            Maximum number of new tokens to generate.

        Returns
        -------
        str or list of str:
            Transcript(s). Returns a single ``str`` when input is 1-D, or a
            ``list[str]`` for batched input.
        """
        is_single = audio_tensor.dim() == 1
        inputs = self._preprocess_audio(audio_tensor, sample_rate)

        generate_kwargs: dict = dict(
            max_new_tokens=max_new_tokens,
        )
        # Attempt to set language / task tokens if the processor supports it
        try:
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=language, task="transcribe"
            )
            if forced_decoder_ids:
                generate_kwargs["forced_decoder_ids"] = forced_decoder_ids
        except (AttributeError, Exception):
            pass

        generated_ids = self.model.generate(**inputs, **generate_kwargs)

        # Decode, removing special tokens
        transcripts: List[str] = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        if is_single:
            return transcripts[0] if transcripts else ""
        return transcripts

    @torch.inference_mode()
    def extract_acoustic_embeddings(
        self,
        audio_tensor: Tensor,
        sample_rate: int = 16_000,
    ) -> Tensor:
        """Extract mean-pooled acoustic embeddings from the Whisper encoder.

        The hook captures the encoder's last hidden state tensor of shape
        ``(batch, seq_len, hidden_dim)`` and applies mean-pooling over the
        sequence dimension to produce ``(batch, hidden_dim)``.

        Parameters
        ----------
        audio_tensor:
            Raw waveform tensor at ``sample_rate`` Hz.
            Shape ``(T,)`` or ``(B, T)``.
        sample_rate:
            Sample rate of the audio.

        Returns
        -------
        Tensor of shape ``(batch, hidden_dim)`` on CPU, float32.
        The hidden dimension is 1280 for Whisper Large-v3.
        """
        inputs = self._preprocess_audio(audio_tensor, sample_rate)

        with self._hook_encoder():
            # A single forward pass through the encoder is sufficient.
            # We call generate with max_new_tokens=1 to minimise compute; the
            # hook fires during the encoder forward pass regardless.
            try:
                self.model.generate(**inputs, max_new_tokens=1)
            except Exception as exc:
                # If generate fails for any reason, attempt a direct encoder
                # forward pass to still capture embeddings.
                logger.warning(
                    "model.generate() failed (%s); attempting encoder forward pass.",
                    exc,
                )
                # Determine the correct input key for the encoder
                input_features = inputs.get(
                    "input_features",
                    inputs.get("input_values", None),
                )
                if input_features is not None:
                    self._encoder(input_features)

        if self._captured_encoder_output is None:
            raise RuntimeError(
                "Encoder hook did not capture any output. "
                "Check that the encoder was actually called during the forward pass."
            )

        hidden_states: Tensor = self._captured_encoder_output  # (B, S, D)

        if hidden_states.dim() == 2:
            # Some encoder variants return (S, D) for a single sample
            hidden_states = hidden_states.unsqueeze(0)

        # Mean-pool over the sequence dimension
        pooled: Tensor = hidden_states.mean(dim=1)  # (B, D)
        return pooled.float().cpu()

    def get_encoder_hidden_dim(self) -> int:
        """Return the encoder hidden dimension.

        Tries to infer this from model config; falls back to the Whisper
        Large-v3 default of 1280.
        """
        for attr in ("d_model", "hidden_size", "encoder_hidden_size"):
            try:
                return getattr(self.model.config, attr)
            except AttributeError:
                continue
        return WHISPER_LARGE_V3_HIDDEN_DIM

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

        Parameters
        ----------
        model_name_or_path:
            HuggingFace model ID or local path.
        device_map:
            Device placement strategy.
        torch_dtype:
            Model floating-point dtype.
        **kwargs:
            Additional arguments forwarded to :class:`VoxtralWrapper.__init__`.

        Returns
        -------
        VoxtralWrapper
        """
        return cls(
            model_name_or_path=model_name_or_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            **kwargs,
        )
