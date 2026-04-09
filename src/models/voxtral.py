"""
Voxtral Wrapper
===============
Wrapper around Mistral's Voxtral models for two tasks:

1. **ASR / transcription** — :meth:`VoxtralWrapper.transcribe` returns the
   text transcript for one or more audio waveforms.
2. **Acoustic embedding extraction** — :meth:`VoxtralWrapper.extract_acoustic_embeddings`
   hooks into the **MLP adapter** that sits between the Whisper encoder and the
   Mistral LLM decoder. The adapter down-samples from 50 Hz to 12.5 Hz and
   projects into the LLM's embedding space, giving semantically-enriched but
   still acoustic (not text-decoded) features.
   Output shape: ``(batch_size, lm_hidden_dim)``.

   Why the adapter, not the raw encoder?
   - Whisper encoder output is 1280-dim at 50 Hz — very long sequences.
   - Adapter output is 4× shorter (12.5 Hz) and already in the LLM's embedding
     space, carrying the same information the LM itself uses for generation.
   - This is the right granularity for a fixed-size mean-pooled representation.

Supported models
----------------
* ``mistralai/Voxtral-Mini-3B-2507``  (3B parameters, fits on 1× L40S)
  LLM hidden dim ≈ 1024 — use ``acoustic_dim: 1024`` in config.
* ``mistralai/Voxtral-Small-24B-2507`` (24B parameters, requires 2× L40S via
  ``device_map="auto"``)
  LLM hidden dim ≈ 4096 — use ``acoustic_dim: 4096`` in config.

Architecture notes
------------------
Voxtral pipeline::

    Raw Audio
        ↓
    [Whisper Large-v3 Encoder]   50 Hz, 1280-dim
        ↓
    [MLP Adapter / Projector]    12.5 Hz, lm_hidden_dim  -> hook here?
        ↓
    [Mistral LLM Decoder]        text tokens

The adapter is typically named ``multi_modal_projector``, ``audio_projector``,
or similar. This module tries a priority list of known attribute paths and falls
back to a keyword BFS if needed.

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
>>> # embeddings shape: (batch, lm_hidden_dim)  e.g. (batch, 1024) for Mini
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

# Fallback hidden dimensions if we cannot infer from model config.
# These are the LLM embedding dimensions (adapter output), not the Whisper
# encoder output (which is always 1280 for Whisper Large-v3).
_MINI_FALLBACK_DIM: int = 1024   # Voxtral-Mini-3B LLM hidden size
_SMALL_FALLBACK_DIM: int = 4096  # Voxtral-Small-24B LLM hidden size
_WHISPER_ENCODER_DIM: int = 1280  # kept for reference / fallback only

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

        # Resolve the MLP adapter (preferred hook point — post-adapter features
        # are in the LLM embedding space and 4x shorter than encoder output).
        # Falls back to the raw Whisper encoder if the adapter cannot be found.
        self._hook_module: nn.Module
        self._hook_module, self._hook_label = self._resolve_hook_module()
        logger.info(
            "Acoustic hook target: %s (%s)",
            self._hook_label,
            type(self._hook_module).__name__,
        )

    # ------------------------------------------------------------------
    # Module resolution — adapter-first, encoder fallback
    # ------------------------------------------------------------------

    def _resolve_hook_module(self) -> Tuple[nn.Module, str]:
        """Find the best module to hook for acoustic embeddings.

        Priority:
        1. MLP adapter / projector (post-adapter = LLM embedding space, 12.5 Hz)
        2. Raw Whisper encoder (fallback, 1280-dim at 50 Hz)

        Returns
        -------
        (module, label) where label is a short description string for logging.
        """
        # ---- 1. Try known adapter/projector attribute paths ----
        adapter_paths: List[List[str]] = [
            ["model", "multi_modal_projector"],
            ["model", "audio_projector"],
            ["model", "mm_projector"],
            ["multi_modal_projector"],
            ["audio_projector"],
            ["model", "model", "multi_modal_projector"],
            ["model", "connector"],
            ["model", "audio_connector"],
        ]
        for path in adapter_paths:
            obj = self.model
            try:
                for attr in path:
                    obj = getattr(obj, attr)
                if isinstance(obj, nn.Module):
                    logger.debug("Found adapter via path: %s", ".".join(path))
                    return obj, "adapter"
            except AttributeError:
                continue

        # ---- 2. BFS: any module whose name suggests a projector ----
        projector_keywords = ("projector", "connector", "adapter", "bridge")
        for name, module in self.model.named_modules():
            if name and any(kw in name.lower() for kw in projector_keywords):
                if isinstance(module, nn.Module):
                    logger.debug(
                        "Found adapter via BFS keyword at: %s", name
                    )
                    return module, f"adapter({name})"

        # ---- 3. Fallback: raw Whisper encoder ----
        logger.warning(
            "MLP adapter not found — falling back to raw Whisper encoder. "
            "Embedding dim will be %d. Update acoustic_dim in your config.",
            _WHISPER_ENCODER_DIM,
        )
        return self._resolve_encoder_fallback(), "encoder(fallback)"

    def _resolve_encoder_fallback(self) -> nn.Module:
        """Locate the raw Whisper encoder as a last resort."""
        candidate_paths: List[List[str]] = [
            ["model", "encoder"],
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

        for name, module in self.model.named_modules():
            if "WhisperEncoder" in type(module).__name__:
                return module

        raise RuntimeError(
            "Could not locate the Whisper encoder or MLP adapter inside the "
            "Voxtral model. Inspect model.named_modules() and update the "
            "candidate paths in VoxtralWrapper._resolve_hook_module()."
        )

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _forward_hook(
        self,
        module: nn.Module,
        input: Tuple,
        output,
    ) -> None:
        """Forward hook that captures the hooked module's output tensor."""
        if isinstance(output, Tensor):
            self._captured_output = output.detach()
        elif hasattr(output, "last_hidden_state"):
            self._captured_output = output.last_hidden_state.detach()
        elif isinstance(output, (tuple, list)) and len(output) > 0:
            first = output[0]
            if isinstance(first, Tensor):
                self._captured_output = first.detach()
            else:
                logger.warning(
                    "Hook output[0] is not a Tensor (got %s); "
                    "acoustic embeddings may be unavailable.",
                    type(first).__name__,
                )
        else:
            logger.warning(
                "Unrecognised hook output type: %s. "
                "Acoustic embeddings may be unavailable.",
                type(output).__name__,
            )

    @contextmanager
    def _hook_active(self):
        """Context manager: register the forward hook, yield, then remove it."""
        self._captured_output: Optional[Tensor] = None
        handle = self._hook_module.register_forward_hook(self._forward_hook)
        try:
            yield
        finally:
            handle.remove()
            # Do NOT clear _captured_output — callers read it after exit.

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
        """Extract mean-pooled acoustic embeddings from the MLP adapter.

        The hook fires on the adapter module (not the raw Whisper encoder),
        capturing features already projected into the LLM's embedding space
        at 12.5 Hz (4x downsampled from 50 Hz). The captured tensor of shape
        ``(batch, seq_len, lm_hidden_dim)`` is mean-pooled over the time
        dimension to give a fixed-size ``(batch, lm_hidden_dim)`` vector.

        Parameters
        ----------
        audio_tensor:
            Raw waveform tensor at ``sample_rate`` Hz.
            Shape ``(T,)`` or ``(B, T)``.
        sample_rate:
            Sample rate of the audio.

        Returns
        -------
        Tensor of shape ``(batch, lm_hidden_dim)`` on CPU, float32.
        lm_hidden_dim is 1024 for Voxtral-Mini and 4096 for Voxtral-Small.
        """
        inputs = self._preprocess_audio(audio_tensor, sample_rate)

        with self._hook_active():
            # model.generate drives the full encoder + adapter forward pass.
            # max_new_tokens=1 minimises LLM compute; the hook fires during
            # the encoder/adapter pass regardless of how many tokens are decoded.
            try:
                self.model.generate(**inputs, max_new_tokens=1)
            except Exception as exc:
                logger.warning(
                    "model.generate() failed (%s); "
                    "hook may not have fired — embeddings could be None.",
                    exc,
                )

        if self._captured_output is None:
            raise RuntimeError(
                f"Hook on '{self._hook_label}' did not capture any output. "
                "The hooked module may not have been called during generate(). "
                "Inspect model.named_modules() and update _resolve_hook_module()."
            )

        hidden_states: Tensor = self._captured_output  # (B, S, D) or (S, D)

        if hidden_states.dim() == 2:
            # Some variants return (S, D) for a single sample
            hidden_states = hidden_states.unsqueeze(0)

        # Mean-pool over the sequence dimension → (B, D)
        pooled: Tensor = hidden_states.mean(dim=1)
        return pooled.float().cpu()

    def get_acoustic_hidden_dim(self) -> int:
        """Return the acoustic embedding dimension (adapter output dim).

        Attempts to infer from model config. Falls back to per-model defaults.
        """
        # LLM hidden size is the adapter output dimension
        for attr in ("hidden_size", "d_model", "text_config.hidden_size"):
            try:
                obj = self.model.config
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return int(obj)
            except AttributeError:
                continue

        # Model-specific fallbacks
        if VOXTRAL_SMALL in self.model_name_or_path:
            return _SMALL_FALLBACK_DIM
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
