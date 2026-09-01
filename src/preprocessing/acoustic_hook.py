"""Capture acoustic frame sequences from the vLLM transcription pass.

Why this exists
---------------
transcribe_all.py used to load Voxtral TWICE: once under vLLM to transcribe,
then again under HF transformers to run the Whisper encoder and mean-pool it.
That is pure waste -- vLLM already runs the Whisper encoder on every clip in
order to build the audio tokens the LLM attends to, then discards the encoder
output once the adapter has projected it into text-embedding space.

This module registers a forward hook on that encoder so the same tensor is
captured on its way past. One model load, both artefacts.

What we hook
------------
vllm/model_executor/models/voxtral.py::VoxtralForConditionalGeneration

    embed_multimodal(**kwargs):
        audio_embeddings = self.whisper_encoder(audio_inputs)   <-- HOOKED HERE
        ... pad / reshape / audio_language_adapter ...

``self.whisper_encoder`` is a VoxtralEncoderModel whose forward is

    forward(input_features: list[Tensor]) -> list[Tensor]

with input_features a list of raw waveforms (one per clip) and the return a
list of (seq_len, 1280) encoder states -- one per clip, SAME ORDER. That is the
same quantity HF's ``audio_tower(...).last_hidden_state`` provides, i.e. the
representation ADR-001 selected.

Sequences, not pooled vectors
-----------------------------
We store the full ``(T, 1280)`` sequence rather than a pooled vector, because
pooling moved into the trainable head (ADR-002/003). Attention pooling is
learned and therefore cannot be baked into a cache written before training.

Frames are truncated to the clip's TRUE length before storage. Voxtral emits
50 encoder frames per second (16 kHz / hop 160 -> 100 Hz mel, conv2 stride 2 ->
50 Hz), so a clip of N samples occupies ceil(N / 320) frames and anything beyond
that is padding. Truncating here rather than masking later is what makes the
cache ~6-7 GB instead of ~52 GB, and removes the padding defect at source.

Matching sequences back to utterance keys
-----------------------------------------
The hook sees a batch of waveforms with no request ids attached -- vLLM's
scheduler decides batch composition, and the mm_hash that identifies each item
lives in the model runner, not in the model. So we key on the audio itself:
blake2b over the waveform bytes. The caller hashes the same decoded waveform it
handed to vLLM and looks the sequence up afterwards.

A hash drift surfaces as a lookup MISS, never as a silently wrong pairing --
which matters, because pairing one clip's acoustics with another clip's label
would corrupt training invisibly.

State lives on the model object (``model._acoustic_capture``) rather than in a
module global, because vLLM workers may run in separate processes: apply_model
hands the same model instance to both the install and drain calls, whereas a
module global in this file would be a different object in each worker.

Note: ``llm.apply_model`` ships the callable to the worker, which requires
``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` in the environment. Without it vLLM
refuses with "Object of type <class 'function'> is not serializable".
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, List, Tuple

import numpy as np
import torch

# Attribute name stamped onto the vLLM model object.
_ATTR = "_acoustic_capture"

#: Audio samples per Whisper encoder frame at 16 kHz.
#: hop_length 160 -> 100 Hz mel frames; conv2 stride 2 -> 50 Hz encoder frames.
SAMPLES_PER_FRAME: int = 320


def num_encoder_frames(num_samples: int) -> int:
    """Encoder frames a clip of ``num_samples`` samples genuinely occupies.

    Args:
        num_samples: Length of the 16 kHz waveform in samples.

    Returns:
        Frame count, at least 1.
    """
    return max(1, int(math.ceil(num_samples / SAMPLES_PER_FRAME)))


def canonical_waveform(waveform: "np.ndarray | torch.Tensor") -> torch.Tensor:
    """Reduce a waveform to the form both sides of the hook can agree on.

    Two transformations sit between what we send vllm and what the encoder
    receives, and both must be undone before a content hash can match. Measured
    directly (job 8081, 6 MELD dev clips):

    1. **Zero-padding to 30 s.** Every clip arrives at the encoder with
       ``n = 480000`` samples regardless of true duration -- vllm pads up to its
       chunk size so a batch can be stacked. Trailing zeros are stripped here.
       This also removes any digital silence the clip genuinely ended with:
       dia1_utt0's own waveform ends in ~193 zero samples, and the driver's copy
       ends in zeros too, so BOTH sides strip it and still agree.

    2. **Cast to bfloat16.** The hook sees ``torch.bfloat16``, not float32 --
       vllm converts audio to the model dtype. bf16 keeps ~3 decimal digits, so
       casting back to float32 does NOT recover the original bits and a float32
       hash can never match. The driver must therefore quantise to bf16 too.
       This is why an earlier float32 fingerprint matched 0 of 24 clips.

    Args:
        waveform: 1-D waveform, numpy array (driver) or torch tensor (worker).

    Returns:
        1-D bfloat16 CPU tensor with trailing zeros removed. Empty input and
        all-silent input both yield a single zero sample rather than an empty
        tensor, so downstream length arithmetic stays valid.
    """
    if isinstance(waveform, torch.Tensor):
        t = waveform.detach().to(torch.bfloat16).cpu()
    else:
        t = torch.from_numpy(
            np.ascontiguousarray(waveform, dtype=np.float32)
        ).to(torch.bfloat16)

    nz = torch.nonzero(t).flatten()
    if nz.numel() == 0:
        return torch.zeros(1, dtype=torch.bfloat16)
    return t[: int(nz[-1]) + 1].contiguous()


def waveform_fingerprint(waveform: "np.ndarray | torch.Tensor") -> str:
    """Stable content hash of one clip's waveform.

    Hashes :func:`canonical_waveform`, so a driver-side float32 numpy array and
    a worker-side bf16 GPU tensor of the same audio produce the same digest.

    Args:
        waveform: 1-D waveform, numpy array or torch tensor.

    Returns:
        32-character hex digest.
    """
    t = canonical_waveform(waveform)
    # bfloat16 has no numpy dtype; reinterpret as int16 (both are 2 bytes) to
    # get at the raw buffer without a lossy conversion.
    buf = t.view(torch.int16).numpy().tobytes()
    return hashlib.blake2b(buf, digest_size=16).hexdigest()


# ---------------------------------------------------------------------------
# Functions below run INSIDE the vLLM worker, via llm.apply_model(fn).
# They are shipped by value, so imports must be local to the function body.
# ---------------------------------------------------------------------------

def install_acoustic_hook(model) -> str:
    """Attach the capture hook. Returns a short status string for logging."""
    import hashlib as _hashlib
    import math as _math

    import torch as _torch

    encoder = getattr(model, "whisper_encoder", None)
    if encoder is None:
        raise AttributeError(
            "vLLM model has no .whisper_encoder -- this build's Voxtral "
            f"implementation differs. Model class: {type(model).__name__}"
        )

    store: Dict[str, "np.ndarray"] = {}
    errors: List[str] = []
    setattr(model, _ATTR, store)
    setattr(model, _ATTR + "_errors", errors)

    def _hook(module, args, output):
        # args[0] is `input_features`; forward() normalises a bare tensor to a
        # list, so replicate that rather than assuming a list arrives.
        inputs = args[0] if args else None
        if inputs is None:
            return
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        if not isinstance(output, (list, tuple)):
            return
        if len(inputs) != len(output):
            # Never guess at an alignment -- a wrong pairing would attach one
            # clip's acoustics to another clip's label and poison training
            # silently. Dropping the batch surfaces as missing keys instead.
            return

        for wav, enc in zip(inputs, output):
            try:
                # Canonicalise: strip vllm's 30 s zero-padding and keep the
                # bf16 quantisation the encoder actually received. Inlined
                # rather than imported so the closure survives being shipped
                # to the worker by value.
                t = wav.detach().to(_torch.bfloat16).cpu()
                nz = _torch.nonzero(t).flatten()
                t = (_torch.zeros(1, dtype=_torch.bfloat16) if nz.numel() == 0
                     else t[: int(nz[-1]) + 1].contiguous())

                fp = _hashlib.blake2b(
                    t.view(_torch.int16).numpy().tobytes(), digest_size=16
                ).hexdigest()

                # Truncate the encoder output to the frames the real audio
                # occupies. Keeping vllm's padding would reintroduce the exact
                # defect ADR-003 exists to remove, and inflate the cache ~8x
                # (measured: 92 MB for 24 clips before this).
                n_true = max(1, int(_math.ceil(t.numel() / 320)))
                enc = enc[: min(n_true, enc.shape[0])]

                # fp16 halves the cache; the encoder ran in bf16 anyway, so
                # this discards no precision the values actually carried.
                store[fp] = enc.detach().to(_torch.float16).cpu().numpy()
            except Exception as exc:                           # noqa: BLE001
                # Do NOT swallow silently -- PM-002. A capture that fails here
                # becomes an unmatched key, which is loud, but the reason for
                # the failure would otherwise be invisible.
                errors.append(f"{type(exc).__name__}: {exc}")

    handle = encoder.register_forward_hook(_hook, with_kwargs=False)
    setattr(model, _ATTR + "_handle", handle)
    return f"hooked {type(encoder).__name__}"


def drain_acoustic_hook(model) -> Dict[str, "np.ndarray"]:
    """Return everything captured so far and clear the buffer."""
    store = getattr(model, _ATTR, None)
    if store is None:
        return {}
    out = dict(store)
    store.clear()
    return out


def drain_hook_errors(model) -> List[str]:
    """Return and clear any exceptions the hook caught.

    The hook cannot raise -- an exception inside a forward hook would abort the
    whole engine step -- so failures are collected here instead. Silently
    swallowing them is what turned a one-word typo into 110 blank rows in
    PM-002; the caller is expected to log whatever this returns.
    """
    errs = getattr(model, _ATTR + "_errors", None)
    if not errs:
        return []
    out = list(errs)
    errs.clear()
    return out


def remove_acoustic_hook(model) -> str:
    """Detach the hook."""
    handle = getattr(model, _ATTR + "_handle", None)
    if handle is not None:
        handle.remove()
    return "removed"


# ---------------------------------------------------------------------------
# Driver-side helper
# ---------------------------------------------------------------------------

def merge_worker_captures(
    per_worker: List[Dict[str, "np.ndarray"]],
) -> Dict[str, torch.Tensor]:
    """Collapse apply_model's per-worker results into one {fingerprint: seq}.

    Under tensor parallelism the audio encoder is replicated, so every rank
    computes the same sequences; taking the union (first writer wins) is correct
    and also tolerates a rank returning nothing.

    Args:
        per_worker: One capture dict per vLLM worker.

    Returns:
        Mapping from waveform fingerprint to a ``(T, 1280)`` fp16 tensor.
    """
    merged: Dict[str, torch.Tensor] = {}
    for captures in per_worker:
        for fp, seq in (captures or {}).items():
            if fp not in merged:
                merged[fp] = torch.from_numpy(np.asarray(seq, dtype=np.float16))
    return merged


def sequence_stats(sequences: Dict[str, torch.Tensor]) -> Tuple[int, float, float]:
    """Summarise a capture for logging.

    Args:
        sequences: Mapping of key to ``(T, D)`` tensor.

    Returns:
        Tuple of (count, total megabytes, mean frame count).
    """
    if not sequences:
        return 0, 0.0, 0.0
    total_bytes = sum(v.numel() * v.element_size() for v in sequences.values())
    mean_frames = sum(v.shape[0] for v in sequences.values()) / len(sequences)
    return len(sequences), total_bytes / 1e6, mean_frames
