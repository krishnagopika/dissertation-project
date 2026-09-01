#!/usr/bin/env python3.12
"""End-to-end INFERENCE over accented English, through the trained MELD pipeline.

What this is
------------
Take N held-out utterances of UK/Irish accented speech and push them through
the *finished* pipeline exactly as deployed -- audio in, emotion label out --
with nothing trained or fine-tuned here. Every weight is frozen and loaded
from the checkpoints the MELD experiments produced.

    .wav -> Voxtral-Mini encoder (frozen)  -> (T, 1280) frames  -> pooler
         -> Voxtral-Mini LLM      (frozen)  -> ASR transcript
                                             -> XLM-R (frozen)  -> 768-d
                                             -> fusion / bc-LSTM -> emotion

Why no "best model" is chosen
-----------------------------
Picking the checkpoint that won on MELD test would be selecting on a different
distribution. Accented read speech is out-of-domain in three ways at once
(accent, channel, and read-vs-spontaneous), and which fusion mechanism
generalises there is precisely the open question. So EVERY fusion variant is
run and reported side by side; the spread across them is itself a result.

Both ASR conditions are run -- and "cleaned" refers to TRAINING, not input
--------------------------------------------------------------------------
There is no such thing as a "cleaned transcript" for this data, and none is
produced here. In MELD the two conditions read the SAME Voxtral transcripts
(`ctx_asr.yaml` and `ctx_asr_cleaned.yaml` share one `transcripts_path`); they
differ only in `filtering.enabled`, i.e. whether the WER-based keep-list was
applied when selecting TRAINING utterances. Test is never filtered in either
condition -- see the header of `evaluate_all.py`.

So the two conditions are two sets of WEIGHTS, produced by training on
different subsets, and each carries its own text encoder:

    asr          text <- xlmr_asr_plain/best_model.pt          (trained on all)
    asr_cleaned  text <- xlmr_asr_cleaned_plain/best_model.pt  (trained on kept)

Here both families therefore receive byte-identical inputs -- the same clips,
the same transcripts -- so for MOST models the difference between the two
condition columns is attributable to the training subset alone. Output columns
are prefixed `trainedon_<condition>` so the CSV cannot be misread as "the input
was cleaned".

EXCEPTION -- `stacked` and `stackedcat` are confounded. Both consume the
`fused512` vector from FUSED_SOURCE, and those two checkpoints differ by
MECHANISM as well as by subset (asr=concat, asr_cleaned=sum), because that is
what each condition's cached features were actually built from. Reproducing
training is the right call, but it means those two families' condition columns
are NOT a clean subset ablation and must not be read as one. `attn`, `attnraw`
and `ctxfusion` all share the single ACOUSTIC_POOL_CKPT across conditions and
ARE clean. The meta JSON records `fused_sources` so this is checkable after
the fact.

Feeding one condition's embeddings to the other condition's fusion head would
be silently wrong. Each checkpoint's own recorded `args["config"]` is checked
against the condition it is being run under, so a future reordering of the run
tables fails loudly instead of producing plausible nonsense.

One waveform, hashed and sent
-----------------------------
Encoder output is paired back to utterance keys by hashing the waveform. That
only works if the array we hash is the array vLLM sees, so the audio is decoded
ONCE here, forced to 16 kHz mono, hashed, and re-encoded as PCM_16 WAV to send.
Handing vLLM the raw file instead would let it apply its own resampling and
downmix policy, and on any corpus not already 16 kHz mono every sample would
differ and every lookup would miss.

bc-LSTM at K=0
--------------
These are isolated sentences, not dialogues, so there is no conversational
context to give the bc-LSTM. That does NOT prevent running it: at K=0 the model
sees a length-1 sequence, which is exactly the configuration that won on MELD
(see docs/BILSTM.md §4 -- the gain is the recurrent layer, not the context).
Its dialogue machinery contributes nothing here, and that is the point: it
tests whether the layer's benefit survives the domain shift.

Output
------
One CSV per run with a column per model, plus `human_emotion` left BLANK for
manual annotation. The corpus has no emotion labels -- it is read speech from a
dialect corpus -- so the only ground truth is what a human hears. Because rows
were stratified by Voxtral's PREDICTED emotion, annotating them yields
per-emotion PRECISION, not recall: we cannot know what the models failed to
detect.

Usage
-----
    python3.12 src/evaluation/dialect_end_to_end.py \\
        --selection /dcs/large/u5734759/data/dialect_probe_100/selection.csv \\
        --out_dir   /dcs/large/u5734759/data/dialect_probe_100
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import EMOTION_NAMES, SENTIMENT_NAMES
from src.utils import get_device, load_config, set_seed, setup_logging

CKPT = Path("/dcs/large/u5734759/checkpoints")
TARGET_SR = 16000

# Text encoder per condition -- see module docstring. Sourced from
# src/scripts/text_embed_v2.sbatch, which built the caches fusion trained on.
TEXT_ENCODER = {
    "asr": CKPT / "xlmr_asr_plain" / "best_model.pt",
    "asr_cleaned": CKPT / "xlmr_asr_cleaned_plain" / "best_model.pt",
}

# Every fusion mechanism, both conditions. No pre-selection.
FUSION_RUNS = [
    (cond, mech, CKPT / "fusion" / "phase3_fusion" / f"{cond}_{mech}" / "best_model.pt")
    for cond in ("asr", "asr_cleaned")
    for mech in ("concat", "sum", "gated", "crossmodal")
]

# Feature composition per context family. NOT interchangeable -- each was
# trained on a different input, and the dimensions below were read off the
# checkpoint weights (`lstm.weight_ih_l0`), not assumed:
#
#   attn        1280 = text 768  ++ acoustic-only fusion represent()  512
#   attnraw     2048 = text 768  ++ RAW pooler output (pre-projection) 1280
#   stacked      512 = both-modality fusion represent() 512
#   stackedcat  2560 = fused 512 ++ text 768 ++ masked-mean acoustic 1280
#   ctxfusion        = text 768 stream and acoustic 512 stream, separately
#
# Concatenation ORDER matters and mirrors DialogueDataset in train_context.py:
#   fused_mode "acoustic" -> cat([text, fused])
#   fused_mode "concat"   -> cat([fused, text, acoustic])
FUSED_DIM = 512

# Acoustic-only attention model that produced meld_pooled/attention. Built once
# under fusion_asr.yaml and reused for BOTH conditions by the attnpool grid, so
# its condition is deliberately not checked against the run's condition.
ACOUSTIC_POOL_CKPT = (CKPT / "fusion" / "phase1_pooling" /
                      "acoustic_attention" / "best_model.pt")

# Both-modality fusion whose represent() produced meld_fused/<cond>. These DO
# differ per condition -- see each directory's provenance.json.
FUSED_SOURCE = {
    "asr": CKPT / "fusion" / "phase3_fusion" / "asr_concat" / "best_model.pt",
    "asr_cleaned": CKPT / "fusion" / "phase3_fusion" / "asr_cleaned_sum" / "best_model.pt",
}

# Which feature models each family actually needs. ctxfusion needs the
# acoustic pooler (for repr512) and is handled alongside NEEDS_ACOUSTIC.
NEEDS_ACOUSTIC = {"attn", "attnraw"}
NEEDS_FUSED = {"stacked", "stackedcat"}

# bc-LSTM (all three orderings from docs/BILSTM.md §3) and context-then-fusion,
# K=0 (length-1 sequence; see docstring).
CONTEXT_RUNS = [
    (cond, f"bclstm_{fam}_k0", "bclstm", fam,
     CKPT / "bclstm" / f"{fam}_{cond}" / "k0" / "best_context.pt")
    for cond in ("asr", "asr_cleaned")
    for fam in ("attn", "attnraw", "stacked", "stackedcat")
] + [
    (cond, "ctxfusion_both_k0", "ctxfusion", "attn",
     CKPT / "ctxfusion" / f"attn_{cond}" / "both_k0" / "best_model.pt")
    for cond in ("asr", "asr_cleaned")
]


# Checkpoints loaded without a recorded condition, i.e. whose pairing could
# not be verified. Reported at the end of main(): an empty list is the only
# state in which _assert_condition's guarantee actually held for every load.
UNVERIFIED: List[str] = []


def _assert_condition(state: Dict, cond: str, ckpt: Path) -> None:
    """Fail loudly if a checkpoint was not trained on the condition claimed.

    The run tables pair a condition with a path by position, which is easy to
    break in a later edit and impossible to notice from the output -- the wrong
    pairing still produces plausible labels. Checkpoints record the config they
    trained under, so the claim is checkable rather than merely intended.

    Exact matching matters: "asr_cleaned" contains "asr" as a substring, so a
    naive `in` test would accept the wrong pairing in one direction.
    """
    a = state.get("args", {})
    if not isinstance(a, dict):
        a = vars(a)
    cfg = a.get("config")
    if not cfg:
        # A silent return would let this guarantee never fire even once while
        # the run still looks clean, so unverified loads are counted and
        # reported rather than passed over.
        UNVERIFIED.append(str(ckpt))
        return
    # SUFFIX matching, not a prefix whitelist. A whitelist has to enumerate
    # every config-naming scheme in the project, and anything it misses falls
    # through and fails the equality test -- reporting a CORRECT pairing as a
    # fatal error. Suffixes need no such list.
    #
    # The substring hazard that motivated exact matching does not apply here:
    #   "asr_cleaned".endswith("asr")        -> False
    #   "fusion_asr_cleaned".endswith("asr") -> False
    # so "asr" cannot swallow "asr_cleaned" in either direction.
    stem = Path(str(cfg)).stem                   # e.g. "fusion_asr_cleaned"
    for suffix in ("_plain", "_focal", "_weighted"):
        stem = stem.removesuffix(suffix)
    if not stem.endswith(cond):
        # Distinguish "trained on the wrong condition" from "I cannot parse
        # this name". Only the former is a real mispairing; the latter must
        # not kill a run that is otherwise correct.
        # "gold" is a real project condition (fusion_gold.yaml, ctx_gold.yaml,
        # xlmr_gold_plain) that this script never RUNS -- listing it here is
        # what turns "a gold checkpoint got wired into an asr run" into a
        # named error instead of an unverified pass.
        # Order matters: "asr_cleaned" must precede "asr", or a cleaned
        # checkpoint would be reported as 'asr'.
        other = [c for c in ("asr_cleaned", "asr", "gold") if stem.endswith(c)]
        if other:
            raise ValueError(
                f"{ckpt} was trained under condition {other[0]!r}, but is "
                f"being run as {cond!r}. Check FUSION_RUNS / CONTEXT_RUNS.")
        UNVERIFIED.append(str(ckpt))


# --------------------------------------------------------------------------
# Stage 1 -- acoustics + ASR, one forward pass
# --------------------------------------------------------------------------

def _canonical_wav(path: Path) -> Tuple["object", bytes]:
    """Decode to 16 kHz mono and re-encode as PCM_16 WAV.

    Returns both the array to hash and the bytes to send, guaranteeing they
    describe the same signal. PCM_16 specifically: writing float32 would leave
    our array pre-quantisation while vLLM decodes the quantised file, so the
    two would differ in the low bits and the hash would miss.
    """
    import io

    import librosa
    import numpy as np
    import soundfile as sf

    wave, sr = sf.read(str(path), dtype="float32")
    if wave.ndim > 1:                             # explicit downmix policy
        wave = wave.mean(axis=1)
    if sr != TARGET_SR:
        wave = librosa.resample(wave, orig_sr=sr, target_sr=TARGET_SR)
    buf = io.BytesIO()
    sf.write(buf, wave, TARGET_SR, format="WAV", subtype="PCM_16")
    raw = buf.getvalue()
    # Re-read so the hashed array is the QUANTISED signal vLLM will decode,
    # not the float array we happened to have in memory.
    quantised, _ = sf.read(io.BytesIO(raw), dtype="float32")
    return np.asarray(quantised), raw


def extract_acoustic_and_asr(
    wavs: List[Tuple[str, Path]],
    model_id: str,
    logger,
    batch_size: int = 8,
) -> Tuple[Dict[str, str], Dict[str, Tensor]]:
    """Transcribe and capture encoder frames in ONE pass, as MELD was done.

    The encoder output is taken via a forward hook on ``whisper_encoder`` and
    paired back to utterance keys by hashing the exact waveform handed to vLLM.
    Doing ASR and extraction separately would run the encoder twice and risk
    the two drifting; this mirrors ``transcribe_all.py`` deliberately.

    Args:
        wavs: ``(key, path)`` pairs.
        model_id: HuggingFace id of the Voxtral model.
        logger: Logger instance.
        batch_size: Clips per vLLM batch; the hook is drained per batch so the
            worker-side buffer stays bounded.

    Returns:
        ``(transcripts, sequences)`` keyed by utterance key. Sequences are
        ``(T, 1280)`` float tensors.
    """
    import base64

    import numpy as np
    from vllm import LLM, SamplingParams

    from src.preprocessing.acoustic_hook import (drain_acoustic_hook,
                                                 drain_hook_errors,
                                                 install_acoustic_hook,
                                                 merge_worker_captures,
                                                 waveform_fingerprint)

    llm = LLM(model=model_id, tokenizer_mode="mistral", max_model_len=8192,
              dtype="bfloat16", gpu_memory_utilization=0.85,
              tensor_parallel_size=1, enforce_eager=True)
    logger.info("hook: %s", llm.apply_model(install_acoustic_hook))

    sampling = SamplingParams(temperature=0.0, max_tokens=200)
    transcripts: Dict[str, str] = {}
    sequences: Dict[str, Tensor] = {}
    captured: Dict[str, "np.ndarray"] = {}
    fingerprints: Dict[str, str] = {}

    for start in range(0, len(wavs), batch_size):
        batch = wavs[start:start + batch_size]
        msgs, keys = [], []
        for key, path in batch:
            wave, raw = _canonical_wav(path)
            fingerprints[key] = waveform_fingerprint(wave)
            b64 = base64.b64encode(raw).decode()
            msgs.append([{"role": "user", "content": [
                {"type": "audio_url",
                 "audio_url": {"url": f"data:audio/wav;base64,{b64}"}},
                {"type": "text", "text": "Transcribe this audio."}]}])
            keys.append(key)

        for k, o in zip(keys, llm.chat(msgs, sampling_params=sampling)):
            transcripts[k] = o.outputs[0].text.strip()

        captured.update(merge_worker_captures(llm.apply_model(drain_acoustic_hook)))
        for errs in llm.apply_model(drain_hook_errors):
            for m in errs[:5]:
                logger.warning("acoustic hook: %s", m)

        for key in keys:
            arr = captured.pop(fingerprints[key], None)
            if arr is None:
                continue
            sequences[key] = (torch.from_numpy(arr)
                              if isinstance(arr, np.ndarray) else arr)
        logger.info("  %d/%d | %d sequences",
                    min(start + batch_size, len(wavs)), len(wavs), len(sequences))

    missing = [k for k, _ in wavs if k not in sequences]
    if missing:
        logger.warning("%d clip(s) produced no encoder output: %s",
                       len(missing), missing[:5])
    # Captures that matched no key are the signature of fingerprint DRIFT --
    # the encoder ran fine, we just hashed something else. Without this line a
    # hashing bug and a capture failure look identical in the log.
    if captured:
        logger.warning(
            "%d captured sequence(s) matched no key -- fingerprint drift "
            "between the array hashed here and the audio vLLM decoded. "
            "If this equals the number missing, it is a hashing bug, not a "
            "capture failure.", len(captured))
        captured.clear()

    # vLLM holds GPU memory in worker processes that `del` alone does not free,
    # and four fusion models plus XLM-R load onto the same device next.
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel
        destroy_model_parallel()
    except Exception as exc:                                      # noqa: BLE001
        logger.warning("destroy_model_parallel unavailable: %s", exc)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return transcripts, sequences


# --------------------------------------------------------------------------
# Stage 2 -- text embeddings, per condition
# --------------------------------------------------------------------------

def embed_text(texts: Dict[str, str], ckpt: Path, cond: str, config: Dict,
               device, logger) -> Dict[str, Tensor]:
    """XLM-R [CLS] embeddings from THIS condition's fine-tuned encoder.

    Args:
        texts: key -> ASR transcript.
        ckpt: Path to the condition's XLM-R checkpoint.
        cond: Condition name, checked against the checkpoint's own record.
        config: Loaded config (supplies model id, dropout, class counts).
        device: Torch device.
        logger: Logger instance.

    Returns:
        key -> 768-d [CLS] tensor on CPU.
    """
    from transformers import AutoTokenizer

    from src.models.xlmr import XLMRobertaClassifier

    if not ckpt.exists():
        raise FileNotFoundError(f"XLM-R checkpoint not found: {ckpt}")
    xlmr_id = config["model"]["xlmr_id"]
    tok = AutoTokenizer.from_pretrained(xlmr_id)
    model = XLMRobertaClassifier(
        model_name_or_path=xlmr_id,
        num_sentiment_classes=config["model"]["num_sentiment_classes"],
        num_emotion_classes=config["model"]["num_classes"],
        dropout_prob=config["model"]["dropout"])
    # weights_only=False: these checkpoints carry an argparse namespace and
    # optimiser state. torch>=2.6 defaults this to True and would refuse.
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    _assert_condition(state, cond, ckpt)
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device).eval()
    logger.info("XLM-R (%s): %s", cond, ckpt)

    out: Dict[str, Tensor] = {}
    keys = list(texts)
    with torch.no_grad():
        for i in range(0, len(keys), 32):
            chunk = keys[i:i + 32]
            enc = tok([texts[k] for k in chunk], max_length=128,
                      padding="max_length", truncation=True, return_tensors="pt")
            cls = model.get_text_representation(
                enc["input_ids"].to(device), enc["attention_mask"].to(device))
            for k, v in zip(chunk, cls.cpu()):
                out[k] = v
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------
# Stage 3 -- classify
# --------------------------------------------------------------------------

def _pad_sequences(seqs: List[Tensor]) -> Tuple[Tensor, Tensor]:
    """Right-pad frame sequences and build the real-frame mask."""
    T = max(s.size(0) for s in seqs)
    D = seqs[0].size(1)
    out = torch.zeros(len(seqs), T, D)
    mask = torch.zeros(len(seqs), T, dtype=torch.bool)
    for i, s in enumerate(seqs):
        out[i, :s.size(0)] = s.float()
        mask[i, :s.size(0)] = True
    return out, mask


def _build_fusion(state: Dict, config: Dict):
    """Reconstruct a SequenceFusion from its checkpoint's recorded args."""
    from src.models.fusion_seq import SequenceFusion

    a = state.get("args", {})
    if not isinstance(a, dict):
        a = vars(a)
    model = SequenceFusion(
        acoustic_dim=config["model"]["acoustic_dim"],
        text_dim=config["model"]["text_dim"],
        hidden_dim=config["model"]["fusion_hidden"],
        num_emotion_classes=config["model"]["num_classes"],
        num_sentiment_classes=config["model"]["num_sentiment_classes"],
        dropout=config["model"]["dropout"],
        pooling=a.get("pooling", "attention"),
        fusion=a.get("fusion", "concat"),
        modality=a.get("modality", "both"))
    model.load_state_dict(state["model_state_dict"])
    return model


def run_fusion(ckpt: Path, cond: str, keys: List[str], text: Dict[str, Tensor],
               seqs: Dict[str, Tensor], config: Dict, device,
               logger) -> Optional[Dict[str, Tuple[str, str]]]:
    """SequenceFusion. Uses the checkpoint's OWN trained attention pooler.

    The pooler is part of the fusion model, so raw ``(T, 1280)`` frames are
    passed in and each checkpoint pools with the weights it was trained with.
    Pre-pooling once and sharing across checkpoints would silently apply one
    model's pooler to another's head.

    Returns:
        key -> ``(emotion_name, sentiment_name)``. ``None`` ONLY when the
        checkpoint is absent -- an empty dict means the checkpoint loaded but
        nothing survived the input filter, which is a different failure.
    """
    if not ckpt.exists():
        logger.warning("missing checkpoint: %s", ckpt)
        return None
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    _assert_condition(state, cond, ckpt)
    model = _build_fusion(state, config).to(device).eval()

    preds: Dict[str, Tuple[str, str]] = {}
    with torch.no_grad():
        for i in range(0, len(keys), 16):
            chunk = [k for k in keys[i:i + 16] if k in seqs and k in text]
            if not chunk:
                continue
            ac, mask = _pad_sequences([seqs[k] for k in chunk])
            tx = torch.stack([text[k] for k in chunk]).float()
            e, s, _ = model(tx.to(device), ac.to(device), mask.to(device))
            for k, ei, si in zip(chunk, e.argmax(-1).cpu(), s.argmax(-1).cpu()):
                preds[k] = (EMOTION_NAMES[ei], SENTIMENT_NAMES[si])
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return preds


def _acoustic_views(frames: Tensor, mask: Tensor, acoustic_model,
                    fused_model, text: Tensor
                    ) -> Dict[str, Callable[[], Tensor]]:
    """The four acoustic reductions the context families were trained on.

    Each family consumed a DIFFERENT reduction of the same frames, so all four
    are produced here and selected by name. Producing one and reusing it
    everywhere would feed three of the five families an input distribution
    they never saw.

    Args:
        frames: ``(B, T, 1280)`` padded encoder output.
        mask: ``(B, T)`` bool, True on real frames.
        acoustic_model: The acoustic-only attention model behind
            meld_pooled/ (512-d ``represent``) and meld_pooled_raw/ (1280-d
            raw ``pooler``).
        fused_model: This condition's both-modality fusion model, whose
            ``represent`` produced meld_fused/<cond>.
        text: ``(B, 768)`` text embeddings, needed by the fused view.

    Returns:
        Dict of THUNKS keyed ``repr512``, ``raw1280``, ``fused512``,
        ``mean1280``. Call the one you need. The two model-backed views are
        deferred, so a family never pays for a reduction it does not use;
        ``mean1280`` is computed eagerly because it is a masked sum.
    """
    # Masked mean over REAL frames only -- dividing by T would scale every
    # vector by however much padding its batch happened to carry.
    denom = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
    mean1280 = (frames * mask.unsqueeze(-1)).sum(dim=1) / denom
    # Lazy: `attn` and `attnraw` never touch fused512, and forcing it would
    # make them depend on a FUSED_SOURCE checkpoint they do not use.
    views = {
        "raw1280": lambda: acoustic_model.pooler(frames, mask)[0],
        "repr512": lambda: acoustic_model.represent(text, frames, mask),
        "fused512": lambda: fused_model.represent(text, frames, mask),
        "mean1280": lambda: mean1280,
    }
    return views


def _context_features(family: str, views: Dict[str, Callable[[], Tensor]],
                      text: Tensor) -> Tensor:
    """Assemble one family's input, in the exact order training used.

    Mirrors ``DialogueDataset`` in train_context.py:
        fused_mode "acoustic" -> cat([text, pooled])
        fused_mode "concat"   -> cat([fused, text, acoustic])
        fused_mode "replace"  -> fused alone
    """
    if family == "attn":                       # 768 + 512 = 1280
        return torch.cat([text, views["repr512"]()], dim=-1)
    if family == "attnraw":                    # 768 + 1280 = 2048
        return torch.cat([text, views["raw1280"]()], dim=-1)
    if family == "stacked":                    # 512
        return views["fused512"]()
    if family == "stackedcat":                 # 512 + 768 + 1280 = 2560
        return torch.cat([views["fused512"](), text, views["mean1280"]()], dim=-1)
    raise ValueError(f"unknown context family: {family}")


def run_context(ckpt: Path, kind: str, cond: str, acoustic_model, fused_model,
                keys: List[str], text: Dict[str, Tensor],
                seqs: Dict[str, Tensor], config: Dict, device, logger,
                family: str = "attn"
                ) -> Optional[Dict[str, Tuple[str, str]]]:
    """bc-LSTM or ContextThenFusion on length-1 sequences (K=0).

    These consume POOLED per-utterance vectors, so the acoustic frames must be
    reduced first. ``acoustic_model`` and ``fused_model`` supply those
    reductions -- the same checkpoints ``extract_pooled_acoustic.py`` and
    ``extract_fused_features.py`` used to build the cached features these
    models were trained on, so the input distribution matches. Both are built
    once per condition in ``main`` and passed in; whichever this family does
    not need is ``None``.

    Both models return ``(sentiment_logits, emotion_logits)`` -- sentiment
    FIRST, opposite to SequenceFusion. Getting that backwards silently swaps
    the two label spaces.
    """
    if not ckpt.exists():
        logger.warning("missing checkpoint: %s", ckpt)
        return None
    # The two feature models are built ONCE PER CONDITION in main() and passed
    # in: rebuilding them inside each of the ten CONTEXT_RUNS calls cost ~18
    # redundant loads. Whichever this family does not need is None, so a
    # missing checkpoint skips only the families that actually depend on it
    # rather than aborting the run.
    if family in NEEDS_ACOUSTIC or kind == "ctxfusion":
        if acoustic_model is None:
            logger.warning("%s needs the acoustic pooler, which is "
                           "unavailable -- skipping", family)
            return None
    if family in NEEDS_FUSED and fused_model is None:
        logger.warning("%s needs the fused source for %s, which is "
                       "unavailable -- skipping", family, cond)
        return None

    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    _assert_condition(state, cond, ckpt)
    sd = state["model_state_dict"]

    if kind == "bclstm":
        from src.models.context_lstm import BiLSTMContext
        # Shapes from the WEIGHTS, not the config -- the config carries no
        # hidden_dim, so a default would be a guess that happens to be right.
        hidden = sd["lstm.weight_hh_l0"].shape[1]
        expected_in = sd["lstm.weight_ih_l0"].shape[1]
        n_layers = sum(1 for k in sd if k.startswith("lstm.weight_ih_l")
                       and not k.endswith("_reverse"))
        model = BiLSTMContext(
            input_dim=expected_in, hidden_dim=hidden,
            num_emotion_classes=config["model"]["num_classes"],
            num_sentiment_classes=config["model"]["num_sentiment_classes"],
            num_layers=n_layers, dropout=config["model"]["dropout"])
    else:
        from src.models.context_fusion import ContextThenFusion
        a = state.get("args", {})
        if not isinstance(a, dict):
            a = vars(a)
        # Shapes from the WEIGHTS. The attn ctxfusion runs were trained on the
        # 512-d pooled represent, NOT the 1280-d config acoustic_dim, so
        # trusting the config here raises a size mismatch on load.
        # Read the arm flags FIRST: an arm with a stream disabled has no
        # corresponding lstm.* key, and indexing it would raise KeyError
        # instead of the clear failure the caller wants.
        use_text = a.get("use_text_lstm", True)
        use_acoustic = a.get("use_acoustic_lstm", True)
        text_dim = (sd["text_lstm.lstm.weight_ih_l0"].shape[1] if use_text
                    else config["model"]["text_dim"])
        acoustic_dim = (sd["acoustic_lstm.lstm.weight_ih_l0"].shape[1]
                        if use_acoustic else FUSED_DIM)
        model = ContextThenFusion(
            text_dim=text_dim, acoustic_dim=acoustic_dim,
            hidden_dim=config["model"]["fusion_hidden"],
            lstm_hidden=a.get("lstm_hidden", 256),
            num_layers=a.get("num_layers", 1),
            num_emotion_classes=config["model"]["num_classes"],
            num_sentiment_classes=config["model"]["num_sentiment_classes"],
            dropout=config["model"]["dropout"],
            fusion=a.get("fusion", "concat"),
            use_text_lstm=use_text, use_acoustic_lstm=use_acoustic)
    model.load_state_dict(sd)
    model = model.to(device).eval()

    preds: Dict[str, Tuple[str, str]] = {}
    with torch.no_grad():
        for i in range(0, len(keys), 16):
            chunk = [k for k in keys[i:i + 16] if k in seqs and k in text]
            if not chunk:
                continue
            ac, mask = _pad_sequences([seqs[k] for k in chunk])
            ac, mask = ac.to(device), mask.to(device)
            tx = torch.stack([text[k] for k in chunk]).float().to(device)
            views = _acoustic_views(ac, mask, acoustic_model, fused_model, tx)
            # (B, 1, D): one utterance per "dialogue" -- K=0.
            lengths = torch.ones(len(chunk), dtype=torch.long)
            if kind == "bclstm":
                feats = _context_features(family, views, tx)
                # `raise`, not `assert`: this is the guard that stops a
                # wrong-but-plausible feature vector from classifying
                # silently, and `python -O` strips assertions.
                if feats.shape[-1] != expected_in:
                    raise ValueError(
                        f"{family}: built {feats.shape[-1]}-d input but the "
                        f"checkpoint expects {expected_in}-d.")
                s_log, e_log = model(feats.unsqueeze(1), lengths)
            else:
                # ctxfusion takes the two streams separately; its acoustic
                # branch was trained on the 512-d represent, not raw frames.
                acou = views["repr512"]()
                if tx.shape[-1] != text_dim or acou.shape[-1] != acoustic_dim:
                    raise ValueError(
                        f"ctxfusion: built {tx.shape[-1]}/{acou.shape[-1]}-d "
                        f"streams but the checkpoint expects "
                        f"{text_dim}/{acoustic_dim}-d.")
                s_log, e_log = model(tx.unsqueeze(1), acou.unsqueeze(1), lengths)
            e = e_log[:, 0].argmax(-1).cpu()
            s = s_log[:, 0].argmax(-1).cpu()
            for k, ei, si in zip(chunk, e, s):
                preds[k] = (EMOTION_NAMES[ei], SENTIMENT_NAMES[si])
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return preds


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run accented speech through the trained MELD pipeline.")
    ap.add_argument("--config", default="src/configs/extract_mini.yaml")
    ap.add_argument("--selection", required=True,
                    help="CSV from the stratified sampler (needs key, audio_path).")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    config = load_config(args.config)
    set_seed(config["data"]["seed"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(out_dir), "dialect_end_to_end")
    # NOTE: `device` is resolved LATER, after Voxtral is torn down.
    # get_device() calls torch.cuda.is_available(), which initialises CUDA in
    # this process; vLLM then forks its EngineCore and dies with
    # "Cannot re-initialize CUDA in forked subprocess". transcribe_all.py
    # imports get_device but never calls it, for exactly this reason.

    sel = pd.read_csv(args.selection)
    wavs = [(r.key, Path(r.audio_path)) for r in sel.itertuples()]
    missing = [k for k, p in wavs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} audio file(s) missing, "
                                f"e.g. {missing[:3]}")
    keys = [k for k, _ in wavs]
    # Fingerprint lookup POPS, so a repeated key (or byte-identical audio)
    # would make the second occurrence silently vanish into `missing`.
    if len(set(keys)) != len(keys):
        dup = [k for k in set(keys) if keys.count(k) > 1]
        raise ValueError(f"duplicate key(s) in selection: {dup[:5]}")
    logger.info("%d clips | %d accents | %d emotions (voxtral-predicted)",
                len(sel), sel.accent.nunique(), sel.pred_emotion.nunique())

    # 1 -- acoustics + ASR. The cache is keyed by the SELECTION, not just the
    # output directory: reusing another selection's acoustics against this
    # sheet would map every new key to NaN without erroring.
    sel_hash = hashlib.blake2b("\n".join(sorted(keys)).encode(),
                               digest_size=8).hexdigest()
    cache = out_dir / f"acoustic_cache_{sel_hash}.pt"
    if cache.exists():
        blob = torch.load(cache, map_location="cpu", weights_only=False)
        transcripts, sequences = blob["transcripts"], blob["sequences"]
        stale = set(keys) - set(blob.get("keys", []))
        if stale:
            raise ValueError(f"{cache} does not cover {len(stale)} selected "
                             f"key(s); delete it and rerun.")
        logger.info("reusing %s (%d clips)", cache.name, len(blob.get("keys", [])))
    else:
        transcripts, sequences = extract_acoustic_and_asr(
            wavs, config["model"]["voxtral_id"], logger, args.batch_size)
        torch.save({"transcripts": transcripts, "sequences": sequences,
                    "keys": keys, "selection": str(args.selection)}, cache)
        logger.info("cached -> %s", cache)

    # Safe now: vLLM has forked, run and been torn down, so initialising CUDA
    # here cannot poison a subsequent fork.
    device = get_device()
    logger.info("classifier device: %s", device)

    # 2 + 3 -- per condition: embed with that condition's encoder, then run
    # every model trained on that condition.
    results: Dict[str, Dict[str, Tuple[str, str]]] = {}
    for cond in ("asr", "asr_cleaned"):
        logger.info("=== condition: %s ===", cond)
        text = embed_text(transcripts, TEXT_ENCODER[cond], cond, config,
                          device, logger)

        for c, mech, ckpt in FUSION_RUNS:
            if c != cond:
                continue
            p = run_fusion(ckpt, cond, keys, text, sequences, config, device, logger)
            if p is not None:
                results[f"trainedon_{cond}__fusion_{mech}"] = p
                logger.info("  fusion/%-10s -> %d preds", mech, len(p))

        # Build the two feature models ONCE for this condition, and only if
        # some family in it actually needs them. A missing checkpoint leaves
        # the corresponding model None, which skips just the dependent
        # families instead of killing the run.
        wanted = {f for c, _, k, f, _ in CONTEXT_RUNS if c == cond}
        kinds = {k for c, _, k, _, _ in CONTEXT_RUNS if c == cond}
        acoustic_model = fused_model = None
        if (wanted & NEEDS_ACOUSTIC) or "ctxfusion" in kinds:
            if ACOUSTIC_POOL_CKPT.exists():
                astate = torch.load(str(ACOUSTIC_POOL_CKPT), map_location="cpu",
                                    weights_only=False)
                # NOT condition-checked: built once under fusion_asr.yaml and
                # reused for both conditions by the attnpool grids.
                acoustic_model = _build_fusion(astate, config).to(device).eval()
            else:
                logger.warning("missing acoustic pooler: %s", ACOUSTIC_POOL_CKPT)
        if wanted & NEEDS_FUSED:
            fsrc = FUSED_SOURCE[cond]
            if fsrc.exists():
                fstate = torch.load(str(fsrc), map_location="cpu",
                                    weights_only=False)
                _assert_condition(fstate, cond, fsrc)
                fused_model = _build_fusion(fstate, config).to(device).eval()
            else:
                logger.warning("missing fused source: %s", fsrc)

        for c, name, kind, family, ckpt in CONTEXT_RUNS:
            if c != cond:
                continue
            p = run_context(ckpt, kind, cond, acoustic_model, fused_model,
                            keys, text, sequences, config, device, logger,
                            family=family)
            if p is not None:
                results[f"trainedon_{cond}__{name}"] = p
                logger.info("  %-18s -> %d preds", name, len(p))

        del acoustic_model, fused_model
        gc.collect()
        torch.cuda.empty_cache()

    # 4 -- annotation sheet
    sheet = sel[["key", "accent", "gender", "speaker_id", "duration_s",
                 "gold_text", "audio_path"]].copy()
    sheet["voxtral_asr_text"] = sheet.key.map(transcripts)
    sheet["voxtral_emotion"] = sel.set_index("key").pred_emotion.reindex(sheet.key).values
    for name in sorted(results):
        sheet[f"pred__{name}"] = sheet.key.map(
            {k: v[0] for k, v in results[name].items()})
    sheet["human_emotion"] = ""          # <- fill by listening
    sheet["notes"] = ""

    path = out_dir / "dialect_predictions.csv"
    sheet.to_csv(path, index=False)
    with open(out_dir / "dialect_predictions_meta.json", "w") as f:
        json.dump({"n_clips": len(sheet), "models": sorted(results),
                   "selection_hash": sel_hash,
                   "text_encoders": {k: str(v) for k, v in TEXT_ENCODER.items()},
                   "fused_sources": {k: str(v) for k, v in FUSED_SOURCE.items()},
                   "acoustic_pool_ckpt": str(ACOUSTIC_POOL_CKPT),
                   "unverified_condition_loads": sorted(set(UNVERIFIED)),
                   "confound": "stacked/stackedcat consume fused512 from "
                               "fused_sources, which differ by MECHANISM "
                               "(asr=concat, asr_cleaned=sum) as well as by "
                               "training subset. Their two condition columns "
                               "are NOT a clean subset ablation. attn, "
                               "attnraw and ctxfusion share one acoustic "
                               "pooler across conditions and ARE clean.",
                   "note": "'trainedon_<cond>' names the TRAINING subset, not "
                           "the input: both families saw identical transcripts. "
                           "human_emotion is blank by design; rows were "
                           "stratified by Voxtral's predicted emotion, so "
                           "annotation yields per-emotion PRECISION not recall."},
                  f, indent=2)

    if UNVERIFIED:
        logger.warning("%d checkpoint load(s) could not be condition-verified: "
                       "%s", len(set(UNVERIFIED)), sorted(set(UNVERIFIED))[:5])
    else:
        logger.info("all checkpoint loads were condition-verified")
    logger.info("wrote %s (%d rows, %d model columns)",
                path, len(sheet), len(results))
    print(f"\n  {len(results)} models run over {len(sheet)} clips")
    print(f"  -> {path}")
    print("\n  agreement with Voxtral's label (no ground truth yet):")
    for name in sorted(results):
        col = sheet[f"pred__{name}"]
        # Denominator is the ROWS THE MODEL SCORED. NaN never equals anything,
        # so including unscored rows would silently cap agreement at the fill
        # rate and read like an accuracy.
        ok = col.notna()
        if not ok.any():
            # distinct=0 would read as "collapsed to one class" rather than
            # "scored nothing", which is the opposite diagnosis.
            print(f"    {name:38s}     --     UNSCORED (0 rows)")
            continue
        agree = (col[ok] == sheet.voxtral_emotion[ok]).mean()
        print(f"    {name:38s} {agree:6.1%}   "
              f"distinct={col[ok].nunique()}  filled={int(ok.sum())}")


if __name__ == "__main__":
    main()
