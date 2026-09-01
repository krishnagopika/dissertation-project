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
    """Attach the capture hook. Returns a short status string for logging.

    TENSOR-PARALLEL-SAFE CAPTURE PATH (needed for Voxtral-Small, tp=2).
    -------------------------------------------------------------------
    Nothing here is intrinsically Small-specific -- the quantity captured is
    the same ``VoxtralEncoderModel`` output, ``(T, 1280)``, from the same
    Whisper large-v3 encoder, and this path would be valid for Mini too. It is
    introduced now because tp=2 is the first execution context in which a
    blocking side effect inside forward() is unsafe. Mini keeps the original
    in-forward version because it is proven over 13,708 clips at tp=1.

    The Mini version does its whole pipeline INSIDE the forward
    pass: ``wav.detach().to(bfloat16).cpu()``, ``torch.nonzero``, a blake2b
    hash, then ``enc.detach().to(float16).cpu().numpy()``. Two of those are
    blocking device-to-host copies, and each one synchronises the CUDA stream.

    At tp=1 that is merely slow. At tp=2 it deadlocks. Jobs 9702 and 9708 both
    loaded Small, ran warmup (which does push audio through the encoder), then
    froze on the first HOOKED audio forward with:

      * tqdm elapsed frozen at 00:00, "Processed prompts: 0/8" indefinitely
      * both VLLM::Worker_TP processes at ~100% of a core (NCCL busy-polls)
      * both GPUs at 100% utilisation with no tokens emitted

    That is a spin-wait, not slow arithmetic. NCCL_P2P_DISABLE=1 is required on
    this node (see small_smoke.sbatch), which makes NCCL stage its all-reduces
    through host memory -- the same path the hook's D2H copies use. A
    synchronising copy inside a collective-bearing forward is what wedges it.

    The fix is to do NOTHING in the hook but retain references. It performs
    only ``detach()``; all copying, hashing, truncation and host transfer are
    deferred to ``drain_acoustic_hook``, which the caller invokes immediately
    after ``llm.chat()`` returns, with no collective in flight.

    An earlier version of this file cloned in the hook, on the theory that only
    a host sync was unsafe. Job 9713 hung anyway, so allocation itself is
    implicated -- see the inline note at the capture site.

    Memory: references only, so nothing is retained beyond the tensors vllm
    already holds for the batch. The caller drains per batch regardless.
    """
    import torch as _torch

    encoder = getattr(model, "whisper_encoder", None)
    if encoder is None:
        raise AttributeError(
            "vLLM model has no .whisper_encoder -- this build's Voxtral "
            f"implementation differs. Model class: {type(model).__name__}"
        )

    pending: List = []            # [(wav_gpu, enc_gpu)] awaiting drain
    errors: List[str] = []
    setattr(model, _ATTR + "_pending", pending)
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
                # Reference-only capture. No clone(), no .cpu(), no .numpy(),
                # no nonzero(), no hashing -- and crucially NO ALLOCATION.
                #
                # The previous version cloned here, on the theory that only a
                # host sync was dangerous. That was too optimistic: clone()
                # allocates ~38 MB per batch of 8 through the caching
                # allocator, and with gpu_memory_utilization=0.85 plus a
                # preallocated KV cache, an allocation that misses the pool
                # falls through to cudaMalloc -- which DOES synchronise, and
                # can wedge an in-flight NCCL collective. Job 9713 still hung
                # with clone(), so the copy was not innocent.
                #
                # Retaining references is safe because the driver drains
                # immediately after llm.chat() returns, before the next batch.
                # If vllm's encoder cache were to reuse a buffer between the
                # several encoder invocations WITHIN one chat() call, an alias
                # would corrupt the capture -- but not silently: the waveform
                # fingerprint would then match no expected key and the run
                # reports unmatched keys. Corruption fails loudly here.
                pending.append((wav.detach(), enc.detach()))
            except Exception as exc:                           # noqa: BLE001
                # Do NOT swallow silently -- PM-002. A capture that fails here
                # becomes an unmatched key, which is loud, but the reason for
                # the failure would otherwise be invisible.
                errors.append(f"{type(exc).__name__}: {exc}")

    handle = encoder.register_forward_hook(_hook, with_kwargs=False)
    setattr(model, _ATTR + "_handle", handle)
    return f"hooked {type(encoder).__name__} (tp-safe, deferred D2H)"


def drain_acoustic_hook(model) -> Dict[str, "np.ndarray"]:
    """Do the deferred work and return {fingerprint: (T, 1280) fp16 array}.

    This is where everything the Mini hook does inline now happens: the host
    transfer, the padding strip, the fingerprint and the fp16 cast. It runs
    from ``llm.apply_model(...)`` BETWEEN batches, so no collective is in
    flight and a synchronising copy is safe here.

    The transformations must match the Mini hook exactly, because the
    fingerprint is what pairs a sequence back to its utterance key: the driver
    hashes the waveform it SENT, and this hashes the waveform the encoder
    RECEIVED. Two things sit between them -- vllm zero-pads to 30 s
    (480,000 samples) and casts to bfloat16 -- so both sides must strip the
    padding and quantise to bf16 or nothing matches. An earlier float32
    fingerprint matched 0 of 24 clips for exactly this reason (PM-004).
    """
    import hashlib
    import math

    pending = getattr(model, _ATTR + "_pending", None)
    if not pending:
        return {}

    errors = getattr(model, _ATTR + "_errors", [])
    out: Dict[str, "np.ndarray"] = {}

    for wav, enc in pending:
        try:
            # bf16 is what the encoder actually received. bf16 -> float32 does
            # NOT recover the original bits, so fingerprinting in float32 would
            # never match the driver side.
            t = wav.to(torch.bfloat16).cpu()
            nz = torch.nonzero(t).flatten()
            t = (torch.zeros(1, dtype=torch.bfloat16) if nz.numel() == 0
                 else t[: int(nz[-1]) + 1].contiguous())

            fp = hashlib.blake2b(
                t.view(torch.int16).numpy().tobytes(), digest_size=16
            ).hexdigest()

            # Truncate to the frames the real audio occupies. Keeping vllm's
            # padding would reintroduce the exact defect ADR-003 removes, and
            # inflate the cache ~8x (measured: 92 MB for 24 clips before this).
            n_true = max(1, int(math.ceil(t.numel() / 320)))
            out[fp] = enc[: min(n_true, enc.shape[0])].to(
                torch.float16).cpu().numpy()
        except Exception as exc:                              # noqa: BLE001
            errors.append(f"drain {type(exc).__name__}: {exc}")

    pending.clear()
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
