"""Serverless inference endpoint for the trained MELD emotion pipeline.

What this serves
----------------
The deployable end of the dissertation: **audio in, a full analysis out**, for
a clip picked from the MELD corpus, uploaded, or recorded in a browser. It is
the `asr` condition -- no gold transcript is assumed, because no deployed
system has one -- and it runs the three stages the project actually trained:

    audio ──► Voxtral-Mini-3B ──┬──► transcript ──► XLM-R (fine-tuned) ──► 768-d
                                │                                            │
                                └──► Whisper encoder frames                  │
                                     masked mean over REAL frames ──► 1280-d │
                                                          │                  │
                                                          └──► ContextThenFusion (T=1)
                                                                    │
                                                        sentiment (3) + emotion (7)

Alongside the prediction it returns the two diagnostics the project uses to
decide whether an utterance is worth trusting at all:

    VAD    Silero speech_ratio, plus the speech segments, so a clip that is
           mostly silence or laughter is visibly that rather than mysteriously
           misclassified.
    WER    against MELD gold for a corpus clip, or against a reference the
           caller supplies. Reported under BOTH normalisation policies
           (`text_normalisation.py`), because the gap between them localises
           how much of the error is formatting rather than content.

and combines them into the project's own **quality gate** -- the `wer25`
policy from the configs, WER <= 0.25 and speech_ratio >= 0.20. That is the
keep-list that produced the `asr_cleaned` training set, so the gate answers a
question a live app genuinely needs: *is this clip the kind of input the model
was trained on?*

Why K=0 makes this deployable at all
------------------------------------
`RESULTS.md` §4 decomposes the context model and finds the recurrent *layer* is
worth +0.019 to +0.058 while the dialogue *context* inside it is worth ~0.000,
with K=0 the best cell in two of three conditions. That is a null result on
conversational context, and it is what makes a single-utterance API honest:
the served model is not a degraded version of the evaluated one, it IS the
evaluated one. No dialogue history has to be assembled at request time.

Which checkpoint, and why this one
----------------------------------
`ctxfusion/asr_cleaned/both_k0` -- "best trained model" in `RESULTS.md` §4b,
which scores it 0.635 weighted F1 on the UK/Irish dialect probe against 0.478
for Voxtral prompted zero-shot on the same clips. It is NOT the best cell on
MELD test (that is bc-LSTM + attention at 0.5334). §4b argues explicitly that
in-domain rank order is a poor guide to deployment -- this architecture is last
on MELD and first out of domain -- so the out-of-domain winner is the
defensible thing to put behind a URL. Swap ``HEAD_CKPT`` to change that; the
acoustic pooling must change with it (this head was trained on the masked-mean
cache, the ``attnraw`` heads were not).

Endpoints
---------
    POST /classify    one clip -> prediction + VAD + WER + quality gate
    GET  /clips       manifest of the bundled MELD subset
    GET  /clip        one clip's audio as base64 wav
    POST /admin       status / warm -- container state, no pinning

All four require Modal proxy-auth headers (``Modal-Key`` / ``Modal-Secret``),
so the Vercel backend holds the credentials and the browser never sees them.

Cost model
----------
Modal bills per second of container life and scales to zero, so a deployed but
idle endpoint costs the volume only (~12 GiB, inside the 1 TiB free allowance
-> $0). At the container shape below, live time is about $0.00031/sec
($1.12/GPU-hour) against a $30/month Starter allowance. The dominant term for
bursty use is the COLD START, not the request. ``benchmark`` measures both.

Usage
-----
    export PYTHONPATH=/dcs/large/u5734759/modal_env

    # once: pull Voxtral-Mini + XLM-R base into the weights volume (no GPU)
    modal run src/modal/serve_emotion.py::download_models

    # once: upload the trained checkpoints and the clip corpus
    python3.12 src/modal/prepare_checkpoints.py --out /tmp/modal_weights
    modal volume put emotion-weights /tmp/modal_weights/xlmr.pt /xlmr.pt
    modal volume put emotion-weights /tmp/modal_weights/head.pt /head.pt
    python3.12 src/modal/prepare_clips.py --out /tmp/modal_clips
    modal volume put emotion-clips /tmp/modal_clips /

    # measure cold start and per-clip cost before committing to anything
    modal run src/modal/serve_emotion.py::benchmark --n 20

    # live URLs, scale-to-zero
    modal deploy src/modal/serve_emotion.py
"""

from __future__ import annotations

import modal

APP_NAME = "xlm-voxemotion"

VOXTRAL_ID = "mistralai/Voxtral-Mini-3B-2507"
XLMR_ID = "FacebookAI/xlm-roberta-base"

# --- Serving shape -----------------------------------------------------------
# L4 is the cheapest card that holds Voxtral-Mini in bf16 (9.3 GB weights) with
# headroom for a 30 s mel batch. A10 is 1.4x the price for the same fit; L40S
# and A100 are 2.4x and 3.1x for a model that does not need them. T4 is cheaper
# still but has no bf16 and only 16 GB, which would force fp16 and leave the
# activation headroom uncomfortably thin.
GPU = "L4"

# Seconds a container stays alive after its last request.
#
# This is now the ONLY warmth control, and it replaced a pair of Start/Stop
# buttons that pinned min_containers=1. The pin was the wrong shape: during an
# active session requests arrive far more often than the window, so it bought
# nothing, while forgetting to press Stop cost ~$1.12/hour indefinitely --
# about $80 over a long weekend. A window cannot be forgotten.
#
# 900 s so a demo never goes cold between questions, at a worst case of
# 15 minutes of idle tail: 0.25 h x $1.116 = $0.28 per session, which is
# cheaper than one extra cold start is annoying.
SCALEDOWN_WINDOW = 900

# A hard ceiling on concurrent spend. Without it a burst of traffic (or a loop
# in a client) fans out to as many containers as Modal will give, each one a
# fresh cold start on a paid GPU. Raise it deliberately, not by default.
MAX_CONTAINERS = 2

WEIGHTS_DIR = "/weights"
CLIPS_DIR = "/clips"
HF_DIR = "/weights/hf"
XLMR_CKPT = "/weights/xlmr.pt"
HEAD_CKPT = "/weights/head.pt"

# --- Label schema ------------------------------------------------------------
# Mirrors EMOTION_NAMES / SENTIMENT_NAMES in src/evaluation/metrics.py. Not
# imported from there because the serving image has no business pulling in the
# training-side dependency tree.
#
# WHICH mapping matters, and the repo has two. `src/data/meld.py` defines
# SENTIMENT2IDX as neutral=0, positive=1, negative=2; `src/evaluation/
# metrics.py` defines SENTIMENT_NAMES as negative=0, neutral=1, positive=2.
# The context models are trained through src/training/train_context.py, which
# builds its SENTIMENT2IDX from SENTIMENT_NAMES -- so the METRICS ordering is
# the one the served heads emit, and it is the one CLAUDE.md §15 documents.
#
# This is not cosmetic. Scoring the head's sentiment logits through meld.py's
# ordering gives weighted F1 0.2125 on dev; through this one it gives 0.5763.
# The emotion orderings agree in both files, which is why only sentiment ever
# looked wrong.
IDX2EMOTION = {0: "neutral", 1: "surprise", 2: "fear", 3: "sadness",
               4: "joy", 5: "disgust", 6: "anger"}
IDX2SENTIMENT = {0: "negative", 1: "neutral", 2: "positive"}

# hop 160 -> 100 Hz mel, conv2 stride 2 -> 50 Hz encoder frames. Same constant
# as src/preprocessing/acoustic_hook.py; the masked mean is only correct if the
# frame count matches the one the training cache was built with.
SAMPLES_PER_FRAME = 320
MAX_ENCODER_FRAMES = 1500          # Voxtral pads every clip to 30 s
TARGET_SR = 16_000

# Transcription decode settings, matched to transcribe_all.py Pass 1 so the
# transcripts XLM-R sees at serving time are drawn from the same distribution
# as the ones it was fine-tuned on.
TRANSCRIBE_PROMPT = "Transcribe this audio."
TRANSCRIBE_MAX_TOKENS = 200

MAX_TEXT_LENGTH = 128              # data.max_text_length in the configs

# --- Quality gate ------------------------------------------------------------
# The `wer25` policy from the configs -- the keep-list that produced the
# asr_cleaned training set. Served so a live app can say "this clip is the kind
# of input the model was trained on" instead of silently scoring noise.
GATE_WER_MAX = 0.25
GATE_VAD_MIN = 0.20
# Which normalisation the gate reads. The training keep-lists were built from
# compute_filter_metadata.py, which normalises before scoring, so `normalised`
# is the policy that reproduces them.
GATE_WER_POLICY = "normalised"

# Upper bound on request audio. Voxtral's encoder truncates at 30 s regardless;
# beyond that the extra audio is decoded, VAD-ed and then thrown away.
MAX_AUDIO_SECONDS = 30.0

# --- Rates -------------------------------------------------------------------
# Modal's published prices, September 2026 (modal.com/pricing). Kept in the
# source so the admin panel reports money rather than seconds, and so the
# figure is traceable to a source rather than to someone's memory.
GPU_USD_PER_SEC = {"T4": 0.000164, "L4": 0.000222, "A10": 0.000306,
                   "L40S": 0.000542, "A100-40GB": 0.000583,
                   "A100-80GB": 0.000694, "H100": 0.001097}
CPU_USD_PER_CORE_SEC = 0.0000131
MEM_USD_PER_GIB_SEC = 0.00000222

CONTAINER_CPU = 4
CONTAINER_MEMORY_GIB = 16

#: What one second of a live inference container costs, all-in.
CONTAINER_USD_PER_SEC = (
    GPU_USD_PER_SEC[GPU]
    + CONTAINER_CPU * CPU_USD_PER_CORE_SEC
    + CONTAINER_MEMORY_GIB * MEM_USD_PER_GIB_SEC
)

app = modal.App(APP_NAME)

# One volume for the models, one for the clip corpus. Split so the corpus can
# be regenerated or withdrawn without touching 10 GB of weights. Volumes are
# $0.09/GiB/mo with 1 TiB free, so both are free at this size.
weights_volume = modal.Volume.from_name("emotion-weights", create_if_missing=True)
clips_volume = modal.Volume.from_name("emotion-clips", create_if_missing=True)

# Shared state for the warm-container controls. A Dict rather than a field on
# the class because the whole point is to read warmth WITHOUT starting a GPU
# container -- asking the pipeline whether it is awake would wake it, and bill
# for the privilege. The GPU container writes heartbeats here; the CPU-only
# admin endpoint reads them.
warm_state = modal.Dict.from_name("emotion-warm-state", create_if_missing=True)

_base = (
    modal.Image.debian_slim(python_version="3.12")
    # MELD clips arrive as .mp4 and browsers upload webm/opus; ffmpeg is the
    # project's only audio decoder (torchaudio/torchcodec are deliberately
    # absent -- see DECISIONS.md).
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.8.0",
        "transformers==4.56.0",
        "mistral-common[audio]",
        "accelerate",
        "huggingface_hub",
        "hf_transfer",
        "soundfile",
        "numpy",
        # transformers' Voxtral chat template resolves `audio` parts through
        # load_audio_as(), which hard-requires librosa. Nothing imports it
        # directly, so it is invisible until the first transcription call
        # raises ImportError deep inside apply_chat_template.
        "librosa",
        # The two diagnostics. silero-vad ships its model inside the wheel, so
        # it needs no network at load time -- which matters because the image
        # runs with HF_HUB_OFFLINE set.
        "silero-vad",
        # Required even though this project uses silero's TORCH model:
        # silero_vad/__init__.py imports sequence_vad, which imports
        # onnxruntime unconditionally at package import. Without it the
        # container crash-loops on `from silero_vad import load_silero_vad`
        # before any audio is seen.
        "onnxruntime",
        "jiwer",
        "fastapi[standard]",
    )
    .env({
        "HF_HOME": HF_DIR,
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
)

# Ship the real source so the served model classes, and the WER normalisation,
# are the ones the results were computed with.
#
# add_local_dir MUST be the last step of each image. Modal refuses a build
# step after it -- local files are layered on at container start rather than
# baked in, so anything after them would silently not apply. That is why the
# two variants below branch from `_base` and each add the source themselves,
# rather than one deriving from the other.
image = _base.add_local_dir("src", "/root/src")

# Serving runs offline: once the volume is populated nothing should reach the
# network, and a silent re-download would show up as cold-start cost rather
# than as an error.
#
# A SEPARATE IMAGE rather than an env var flipped at runtime. huggingface_hub
# reads HF_HUB_OFFLINE once, into a module constant, when it is first
# imported -- so `os.environ[...] = "0"` inside the download function is read
# too late and the download fails with LocalEntryNotFoundError against an
# empty cache. That is not hypothetical; it is what the first deploy did.
serving_image = (
    _base
    .env({"HF_HUB_OFFLINE": "1"})
    .add_local_dir("src", "/root/src")
)

# Proxy auth. Callers present Modal-Key / Modal-Secret headers; the Vercel
# route holds them server-side so they never reach the browser. Create with:
#   modal token new --profile <p>   (account tokens)
#   modal secret create ...          (not needed -- proxy auth is account-level)
# and attach the resulting proxy auth token in the Modal dashboard.
REQUIRES_PROXY_AUTH = True


# -----------------------------------------------------------------------------
# One-time setup
# -----------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={WEIGHTS_DIR: weights_volume},
    timeout=60 * 60,
    # Voxtral is a gated repo; the download 401s without an accepted licence.
    secrets=[modal.Secret.from_name("huggingface-token")],
    # No GPU: this is a pure download, and paying L4 rates to wait on
    # HuggingFace would be the most avoidable cost in the whole deployment.
)
def download_models() -> dict:
    """Populate the weights volume with Voxtral-Mini and XLM-R base. Run once.

    Returns:
        Sizes in GB of each snapshot, so the volume footprint is known before
        anything is billed against it.
    """
    import os

    from huggingface_hub import snapshot_download

    # Only the HF-format weights and the tokeniser. Unfiltered, these two
    # repos land 25 GB on the volume: Voxtral ships Mistral's
    # consolidated.safetensors alongside the HF shards (a full duplicate of
    # weights this endpoint does not load, since it uses
    # VoxtralForConditionalGeneration), and xlm-roberta-base ships TF, Flax,
    # ONNX and .bin copies of the same 1.1 GB model. Filtered it is ~10 GB.
    ignore = ["consolidated*.safetensors", "*.msgpack", "*.h5", "*.ot",
              "*.tflite", "onnx/*", "rust_model.ot", "pytorch_model.bin"]

    sizes = {}
    for model_id in (VOXTRAL_ID, XLMR_ID):
        path = snapshot_download(model_id, ignore_patterns=ignore)
        total = sum(
            os.path.getsize(os.path.join(r, f))
            for r, _, fs in os.walk(path) for f in fs
        )
        sizes[model_id] = round(total / 1e9, 2)
        print(f"{model_id}: {sizes[model_id]} GB at {path}")

    weights_volume.commit()
    return sizes


# -----------------------------------------------------------------------------
# Diagnostics: VAD and WER
# -----------------------------------------------------------------------------

def _vad_report(waveform, vad_model) -> dict:
    """Silero speech ratio plus the segments behind it.

    ``compute_filter_metadata.py`` keeps only the ratio, because that is all
    the keep-lists need. A live app wants the segments too: "speech_ratio
    0.31" is a number, but a timeline showing one 0.9 s burst inside a 3 s clip
    is an explanation.

    Args:
        waveform: 1-D float32 torch tensor at 16 kHz.
        vad_model: Loaded Silero-VAD model.

    Returns:
        Dictionary with ``speech_ratio``, ``speech_seconds``, ``num_segments``
        and ``segments`` (start/end in seconds). A clip under 100 ms, or one
        VAD fails on, reports a ratio of 0.0 -- matching the training-time
        behaviour rather than raising.
    """
    from silero_vad import get_speech_timestamps

    n = int(waveform.numel())
    if n < TARGET_SR // 10:                      # < 100 ms -> treat as no speech
        return {"speech_ratio": 0.0, "speech_seconds": 0.0,
                "num_segments": 0, "segments": []}

    try:
        segments = get_speech_timestamps(
            waveform, vad_model, sampling_rate=TARGET_SR, threshold=0.5)
    except Exception:                                            # noqa: BLE001
        return {"speech_ratio": 0.0, "speech_seconds": 0.0,
                "num_segments": 0, "segments": []}

    speech_samples = sum(s["end"] - s["start"] for s in segments)
    return {
        "speech_ratio": round(speech_samples / n, 4),
        "speech_seconds": round(speech_samples / TARGET_SR, 2),
        "num_segments": len(segments),
        "segments": [{"start": round(s["start"] / TARGET_SR, 2),
                      "end": round(s["end"] / TARGET_SR, 2)}
                     for s in segments],
    }


def _wer_report(reference: str, hypothesis: str) -> dict:
    """Word- and character-level error rates under BOTH normalisation policies.

    Both are reported because the project reports both: EXACT embeds zero
    judgement calls and reads high, NORMALISED is standard ASR practice, and
    the gap between them is how much of the error is formatting rather than
    content (`text_normalisation.py`).

    Args:
        reference: Ground-truth text -- MELD gold, or caller-supplied.
        hypothesis: The transcript Voxtral produced.

    Returns:
        Dictionary keyed by policy name, each with wer/mer/wil/wip/cer, plus
        the word counts. An empty reference yields ``None`` for every rate
        rather than a divide-by-zero or a misleading 1.0.
    """
    import sys

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    import jiwer

    from src.evaluation.text_normalisation import POLICIES, normalise_text

    out: dict = {
        "reference_words": len(normalise_text(reference).split()),
        "hypothesis_words": len(normalise_text(hypothesis).split()),
        "policies": {},
    }
    if not reference.strip():
        out["policies"] = {name: None for name in POLICIES}
        return out

    for name, (words, chars) in POLICIES.items():
        measures = jiwer.process_words(
            reference, hypothesis,
            reference_transform=words, hypothesis_transform=words)
        out["policies"][name] = {
            "wer": round(measures.wer, 4),
            "mer": round(measures.mer, 4),
            "wil": round(measures.wil, 4),
            "wip": round(measures.wip, 4),
            "cer": round(jiwer.cer(
                reference, hypothesis,
                reference_transform=chars, hypothesis_transform=chars), 4),
            "substitutions": measures.substitutions,
            "deletions": measures.deletions,
            "insertions": measures.insertions,
            "hits": measures.hits,
        }
    return out


def _quality_gate(wer: dict, vad: dict) -> dict:
    """Apply the project's `wer25` keep-list policy to a single clip.

    This is the same rule that built the `asr_cleaned` training set: WER <=
    0.25 AND speech_ratio >= 0.20. Serving it lets a caller distinguish "the
    model thinks this is anger" from "this clip would not have been in the
    training set at all", which are very different claims to put in front of a
    user.

    Args:
        wer: Output of :func:`_wer_report`, or ``None`` when no reference
            exists -- in which case the WER half of the gate is unevaluable
            and the gate reports ``None`` rather than guessing.
        vad: Output of :func:`_vad_report`.

    Returns:
        Dictionary with ``passes`` (bool or None), the thresholds used, and a
        list of human-readable failure reasons.
    """
    reasons = []

    speech_ok = vad["speech_ratio"] >= GATE_VAD_MIN
    if not speech_ok:
        reasons.append(
            f"speech_ratio {vad['speech_ratio']:.2f} below {GATE_VAD_MIN:.2f} "
            f"-- mostly non-speech")

    wer_value = None
    if wer is not None and wer["policies"].get(GATE_WER_POLICY):
        wer_value = wer["policies"][GATE_WER_POLICY]["wer"]
        if wer_value > GATE_WER_MAX:
            reasons.append(
                f"WER {wer_value:.2f} above {GATE_WER_MAX:.2f} "
                f"-- transcript unreliable")

    return {
        "policy": "wer25",
        "wer_max": GATE_WER_MAX,
        "vad_min_speech_ratio": GATE_VAD_MIN,
        "wer_policy": GATE_WER_POLICY,
        "speech_ratio_ok": speech_ok,
        "wer_ok": None if wer_value is None else wer_value <= GATE_WER_MAX,
        # Unevaluable rather than False when there is no reference: an upload
        # with no ground truth has not failed the gate, it has not taken it.
        "passes": None if wer_value is None else (speech_ok and
                                                  wer_value <= GATE_WER_MAX),
        "reasons": reasons,
    }


# -----------------------------------------------------------------------------
# Serving
# -----------------------------------------------------------------------------

@app.cls(
    image=serving_image,
    gpu=GPU,
    volumes={WEIGHTS_DIR: weights_volume, CLIPS_DIR: clips_volume},
    cpu=CONTAINER_CPU,
    memory=CONTAINER_MEMORY_GIB * 1024,
    scaledown_window=SCALEDOWN_WINDOW,
    # Scale to zero. This is the default, stated explicitly because it is the
    # entire answer to "can I pay only for inference": with no floor on
    # containers, an idle endpoint costs nothing but volume storage.
    min_containers=0,
    max_containers=MAX_CONTAINERS,
    # Snapshot the loaded models so a cold start restores memory instead of
    # re-running from_pretrained. The GPU half is experimental and is the part
    # that matters here -- 9.3 GB of bf16 weights is what makes the cold start
    # expensive in the first place.
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    timeout=600,
)
class EmotionPipeline:
    """Voxtral-Mini + fine-tuned XLM-R + ContextThenFusion, loaded once."""

    @modal.enter(snap=True)
    def load(self) -> None:
        """Load all four models. Captured in the snapshot, so paid for once."""
        import json
        import sys
        import time
        from pathlib import Path

        t0 = time.time()

        # The repo has no __init__.py files (PEP 420 namespace packages), so
        # the parent of src/ must be importable.
        if "/root" not in sys.path:
            sys.path.insert(0, "/root")

        import torch
        from silero_vad import load_silero_vad
        from transformers import (
            AutoProcessor,
            AutoTokenizer,
            VoxtralForConditionalGeneration,
        )

        from src.models.context_fusion import ContextThenFusion
        from src.models.xlmr import XLMRobertaClassifier

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- Voxtral-Mini: transcript AND encoder frames from one load ------
        # kwarg is `dtype=`, not `torch_dtype=` (PROGRESS bug 2).
        self.voxtral = VoxtralForConditionalGeneration.from_pretrained(
            VOXTRAL_ID, dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(VOXTRAL_ID)

        # --- XLM-R, fine-tuned on the served condition ----------------------
        self.tokenizer = AutoTokenizer.from_pretrained(XLMR_ID)
        self.xlmr = XLMRobertaClassifier(model_name_or_path=XLMR_ID)
        state = torch.load(XLMR_CKPT, map_location="cpu", weights_only=False)
        # Accept a bare state dict or a training checkpoint; the stripped
        # artefact is the latter minus the optimizer moments.
        self.xlmr.load_state_dict(state.get("model_state_dict", state), strict=False)
        self.xlmr = self.xlmr.to(self.device).eval()

        # --- The trained head -----------------------------------------------
        # Rebuilt from the checkpoint's own argparse record rather than from
        # constants here, so a head trained with different arms or a different
        # fusion cannot be silently loaded into the wrong architecture.
        head_state = torch.load(HEAD_CKPT, map_location="cpu", weights_only=False)
        head_args = head_state["args"]
        self.head = ContextThenFusion(
            text_dim=768,
            acoustic_dim=1280,
            hidden_dim=512,
            lstm_hidden=head_args["lstm_hidden"],
            num_layers=head_args["num_layers"],
            fusion=head_args["fusion"],
            use_text_lstm=head_args["use_text_lstm"],
            use_acoustic_lstm=head_args["use_acoustic_lstm"],
        )
        self.head.load_state_dict(head_state["model_state_dict"])
        self.head = self.head.to(self.device).eval()

        # --- Silero VAD ------------------------------------------------------
        # CPU: the model is ~1 MB and the clips are seconds long, so moving it
        # to the GPU would cost more in transfer than it saves in compute.
        self.vad = load_silero_vad()

        # --- Clip corpus manifest --------------------------------------------
        # Optional: the endpoint still classifies uploads without it.
        manifest_path = Path(CLIPS_DIR) / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as fh:
                self.manifest = {c["key"]: c for c in json.load(fh)}
        else:
            self.manifest = {}

        self.model_info = {
            "head": head_args["tag"],
            "head_dev_emotion_wf1": round(head_state["emotion_weighted_f1"], 4),
            "fusion": head_args["fusion"],
            "context_window": head_args.get("context_window", 0),
            "voxtral": VOXTRAL_ID,
            "xlmr": XLMR_ID,
        }
        self.load_seconds = time.time() - t0

        # Heartbeat, so the admin endpoint can report warmth without waking a
        # GPU container to ask. Written on load and refreshed on every request.
        warm_state["loaded_at"] = time.time()
        warm_state["load_seconds"] = round(self.load_seconds, 1)
        warm_state["last_seen"] = time.time()
        warm_state["model"] = self.model_info

        print(f"loaded in {self.load_seconds:.1f}s | {self.model_info} "
              f"| {len(self.manifest)} corpus clips")

    @modal.exit()
    def unload(self) -> None:
        """Record the scaledown so the admin panel stops claiming warmth."""
        import time

        warm_state["stopped_at"] = time.time()
        warm_state.pop("loaded_at", None)

    @modal.method()
    def warmup(self) -> dict:
        """Force a container up and the models in, without doing any work.

        The Start button's payload. Returns once :meth:`load` has finished, so
        a caller that waits on it knows the next real request will be warm.

        Returns:
            Load time and the model identifiers now resident.
        """
        return {"warm": True,
                "load_seconds": round(self.load_seconds, 1),
                "model": self.model_info,
                "corpus_clips": len(self.manifest)}

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _decode(raw: bytes) -> "object":
        """Decode arbitrary container bytes to a 16 kHz mono float32 waveform.

        Args:
            raw: Bytes of any ffmpeg-readable audio or video container --
                MELD's .mp4, an uploaded .wav, or the browser's webm/opus.

        Returns:
            1-D float32 numpy array at 16 kHz.

        Raises:
            ValueError: If ffmpeg cannot decode the input.
        """
        import subprocess

        import numpy as np

        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", "pipe:0",
             "-ac", "1", "-ar", str(TARGET_SR), "-f", "f32le", "pipe:1"],
            input=raw, capture_output=True, check=False,
        )
        if proc.returncode != 0 or not proc.stdout:
            raise ValueError(
                f"ffmpeg could not decode this audio: "
                f"{proc.stderr.decode('utf-8', 'replace')[:200]}")
        # .copy() rather than the raw buffer view: np.frombuffer is read-only,
        # and torch.from_numpy on a non-writable array warns and produces a
        # tensor whose mutation is undefined behaviour. Silero takes a tensor.
        return np.frombuffer(proc.stdout, dtype=np.float32).copy()

    def _corpus_bytes(self, key: str) -> bytes:
        """Read one bundled corpus clip off the clips volume.

        Args:
            key: Corpus key, e.g. ``dia0_utt0``.

        Returns:
            The clip's wav bytes.

        Raises:
            ValueError: If the key is unknown or its file is missing, which
                are different faults and say so differently.
        """
        from pathlib import Path

        entry = self.manifest.get(key)
        if entry is None:
            # Name a few real keys. The corpus is a random class-balanced
            # sample, so guessing a plausible-looking key like "dia0_utt0"
            # usually misses -- the error should not make that a guessing game.
            examples = ", ".join(sorted(self.manifest)[:5]) or "(manifest empty)"
            raise ValueError(
                f"unknown clip key: {key}. {len(self.manifest)} clips are "
                f"loaded, for example: {examples}")

        path = Path(CLIPS_DIR) / entry["file"]
        if not path.exists():
            raise ValueError(f"clip file missing on the volume: {entry['file']}")
        return path.read_bytes()

    def _acoustic(self, wave) -> "object":
        """Masked-mean pooled Whisper-encoder representation for one clip.

        The mask is the whole point. Voxtral zero-pads every clip to 30 s, so
        an unmasked mean over 1500 frames dilutes a two-second utterance by
        ~15x -- and the cached features this head was trained on were pooled
        over real frames only (FUSION.md §2).

        Args:
            wave: 1-D float32 waveform at 16 kHz.

        Returns:
            Float32 tensor of shape ``(1280,)`` on the model device.
        """
        import math

        torch = self.torch

        features = self.processor.feature_extractor(
            [wave], sampling_rate=TARGET_SR, return_tensors="pt",
            padding="max_length",      # the audio tower needs exactly 3000 mel frames
            truncation=True,
        )["input_features"].to(self.device, dtype=torch.bfloat16)

        n_frames = min(
            MAX_ENCODER_FRAMES,
            max(1, math.ceil(len(wave) / SAMPLES_PER_FRAME)),
        )
        hidden = self.voxtral.audio_tower(features).last_hidden_state  # (1, 1500, 1280)
        return hidden[0, :n_frames].float().mean(dim=0)

    def _transcribe(self, wave) -> str:
        """Greedy transcript for one clip, using Pass 1's prompt and cap.

        The waveform is handed over as a base64 data URI rather than a file.
        The chat template accepts ``url``, ``path`` or ``base64`` for an audio
        part, but the first two go through ``load_audio_as``, which cannot
        resolve a ``file://`` URI -- writing a temp wav and passing
        ``file://{path}`` fails with "Error loading audio: File not found"
        even though the file is plainly there. The base64 branch skips that
        loader entirely and lands on ``audio_url``, which is byte-for-byte the
        form transcribe_all.py Pass 1 used through vLLM.

        Args:
            wave: 1-D float32 waveform at 16 kHz.

        Returns:
            Transcript string, stripped.
        """
        import base64
        import io

        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, wave, TARGET_SR, format="WAV")
        encoded = base64.b64encode(buffer.getvalue()).decode()

        conversation = [{"role": "user", "content": [
            {"type": "audio", "base64": f"data:audio/wav;base64,{encoded}"},
            {"type": "text", "text": TRANSCRIBE_PROMPT},
        ]}]
        inputs = self.processor.apply_chat_template(
            conversation, return_dict=True, return_tensors="pt", tokenize=True,
        )
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}
        generated = self.voxtral.generate(
            **inputs,
            max_new_tokens=TRANSCRIBE_MAX_TOKENS,
            do_sample=False,       # temperature=0.0 in Pass 1
        )

        new_ids = generated[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            new_ids, skip_special_tokens=True)[0].strip()

    def _text(self, transcript: str) -> "object":
        """Fine-tuned XLM-R ``[CLS]`` vector for one transcript.

        Args:
            transcript: The ASR output for this clip.

        Returns:
            Float32 tensor of shape ``(768,)`` on the model device.
        """
        encoded = self.tokenizer(
            transcript, max_length=MAX_TEXT_LENGTH, truncation=True,
            padding="max_length", return_tensors="pt",
        ).to(self.device)
        cls = self.xlmr.get_text_representation(
            encoded["input_ids"], encoded["attention_mask"])
        return cls[0].float()

    # -- inference ------------------------------------------------------------

    @modal.method()
    def analyse(
        self,
        raw: bytes,
        reference: str = "",
        key: str = "",
    ) -> dict:
        """Full analysis of one clip.

        Args:
            raw: Clip bytes, any ffmpeg-readable container. May be empty when
                ``key`` is given, in which case the audio is read from the
                clips volume.
            reference: Ground-truth transcript for WER. Ignored when ``key``
                names a corpus clip, which carries its own MELD gold.
            key: Optional corpus key (``dia0_utt0``). When set, the MELD gold
                transcript and labels are attached and the prediction is
                scored against them.

        Returns:
            A single dictionary carrying the prediction, the VAD report, the
            WER report, the quality gate, and per-stage timings. WER and the
            gate degrade to ``None`` rather than being omitted when no
            reference exists, so the client's shape never changes.
        """
        import time

        import torch as _torch

        torch = self.torch
        timings = {}
        warm_state["last_seen"] = time.time()

        # Resolving the key HERE, not only in the HTTP endpoint. A corpus clip
        # already lives on this container's volume, so making the caller ship
        # its bytes is both pointless and a trap: `analyse(b"", key=...)` from
        # another Modal function looks obviously correct and used to die in
        # ffmpeg with "Invalid data found when processing input".
        if not raw and key:
            raw = self._corpus_bytes(key)

        t0 = time.time()
        wave = self._decode(raw)
        truncated = False
        if len(wave) > MAX_AUDIO_SECONDS * TARGET_SR:
            wave = wave[: int(MAX_AUDIO_SECONDS * TARGET_SR)]
            truncated = True
        timings["decode_ms"] = round((time.time() - t0) * 1000)

        # VAD runs on the clip as heard, before any model sees it, so a
        # silence-dominated clip is diagnosable even if the rest fails.
        t0 = time.time()
        vad = _vad_report(_torch.from_numpy(wave), self.vad)
        timings["vad_ms"] = round((time.time() - t0) * 1000)

        with torch.inference_mode():
            t0 = time.time()
            acoustic = self._acoustic(wave)
            timings["acoustic_ms"] = round((time.time() - t0) * 1000)

            t0 = time.time()
            transcript = self._transcribe(wave)
            timings["transcribe_ms"] = round((time.time() - t0) * 1000)

            t0 = time.time()
            text = self._text(transcript)
            timings["text_ms"] = round((time.time() - t0) * 1000)

            t0 = time.time()
            # The head expects a dialogue: (B, T, D) with real lengths. K=0
            # means T=1 -- the BiLSTM runs with all its parameters over a
            # length-1 sequence, which is exactly how it was trained.
            s_logits, e_logits = self.head(
                text.view(1, 1, -1), acoustic.view(1, 1, -1), torch.tensor([1]))
            e_probs = torch.softmax(e_logits[0, 0], dim=-1)
            s_probs = torch.softmax(s_logits[0, 0], dim=-1)
            timings["head_ms"] = round((time.time() - t0) * 1000)

        emotion_idx = int(e_probs.argmax())
        sentiment_idx = int(s_probs.argmax())

        # --- Reference text: corpus gold beats anything the caller sends -----
        clip = self.manifest.get(key) if key else None
        if clip is not None:
            reference, reference_source = clip["utterance"], "meld_gold"
        elif reference.strip():
            reference_source = "caller"
        else:
            reference_source = None

        wer = _wer_report(reference, transcript) if reference_source else None
        if wer is not None:
            wer["reference"] = reference
            wer["reference_source"] = reference_source

        result = {
            "key": key or None,
            "audio": {
                "duration_sec": round(len(wave) / TARGET_SR, 2),
                "sample_rate": TARGET_SR,
                "truncated_to_30s": truncated,
            },
            "transcript": transcript,
            "prediction": {
                "emotion": IDX2EMOTION[emotion_idx],
                "emotion_confidence": round(float(e_probs[emotion_idx]), 4),
                "sentiment": IDX2SENTIMENT[sentiment_idx],
                "sentiment_confidence": round(float(s_probs[sentiment_idx]), 4),
                "emotion_probs": {IDX2EMOTION[i]: round(float(p), 4)
                                  for i, p in enumerate(e_probs)},
                "sentiment_probs": {IDX2SENTIMENT[i]: round(float(p), 4)
                                    for i, p in enumerate(s_probs)},
            },
            "vad": vad,
            "wer": wer,
            "quality_gate": _quality_gate(wer, vad),
            "model": self.model_info,
            "timings_ms": timings,
        }

        # --- Ground truth, when the clip came from the corpus ---------------
        if clip is not None:
            result["ground_truth"] = {
                "emotion": clip["emotion"],
                "sentiment": clip["sentiment"],
                "utterance": clip["utterance"],
                "speaker": clip.get("speaker"),
                "split": clip.get("split"),
            }
            result["correct"] = {
                "emotion": clip["emotion"] == result["prediction"]["emotion"],
                "sentiment": clip["sentiment"] == result["prediction"]["sentiment"],
            }

        result["timings_ms"]["total_ms"] = sum(
            v for k, v in timings.items() if k.endswith("_ms"))
        return result

    # -- HTTP -----------------------------------------------------------------

    @modal.fastapi_endpoint(method="POST", docs=True, requires_proxy_auth=REQUIRES_PROXY_AUTH)
    def classify(self, item: dict) -> dict:
        """Classify one clip.

        Args:
            item: ``{"audio_b64": "<base64>"}`` for an upload or recording, or
                ``{"key": "dia0_utt0"}`` to analyse a bundled corpus clip.
                An optional ``"reference"`` supplies ground-truth text for WER
                on an upload; corpus clips use their MELD gold instead.

        Returns:
            The analysis dictionary, or ``{"error": ...}`` with the reason.
        """
        import base64
        import time

        # Heartbeat here as well as in analyse(): a request that fails
        # validation still means a container is up and billing, and a status
        # panel that missed those would under-report what is running.
        warm_state["last_seen"] = time.time()

        key = (item.get("key") or "").strip()
        payload = item.get("audio_b64")

        if key:
            # analyse() resolves the key against the volume itself.
            raw = b""
        elif payload:
            try:
                raw = base64.b64decode(payload)
            except Exception as exc:                             # noqa: BLE001
                return {"error": f"audio_b64 is not valid base64: {exc}"}
        else:
            return {"error": "expected either 'key' or 'audio_b64'"}

        try:
            return self.analyse.local(
                raw, reference=item.get("reference", ""), key=key)
        except ValueError as exc:
            return {"error": str(exc)}


# -----------------------------------------------------------------------------
# Corpus browsing -- CPU only
# -----------------------------------------------------------------------------
#
# These were originally methods on EmotionPipeline, which was a costly
# mistake: the class carries the GPU, so listing the corpus or fetching a wav
# for the audio player woke an L4. Simply OPENING the page did it, because the
# picker loads on mount -- two containers were found idling on a deployment
# nobody had run an analysis against.
#
# Neither endpoint needs a model. They read a manifest and a file off the
# clips volume, so they belong on a CPU container at roughly 3% of the cost,
# and the GPU now wakes only for an actual analysis.

def _manifest() -> dict:
    """Load the clip manifest keyed by clip id.

    Returns:
        ``{key: record}``, empty when the volume has no manifest yet.
    """
    import json
    from pathlib import Path

    path = Path(CLIPS_DIR) / "manifest.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return {c["key"]: c for c in json.load(fh)}


@app.function(
    image=image,
    volumes={CLIPS_DIR: clips_volume},
    cpu=0.125,
    memory=512,
    timeout=60,
)
@modal.fastapi_endpoint(method="GET", docs=True, requires_proxy_auth=REQUIRES_PROXY_AUTH)
def clips() -> dict:
    """The bundled MELD subset, for the corpus picker.

    Returns:
        ``{"count": n, "clips": [...]}`` -- each entry carries the key, split,
        gold emotion and sentiment, gold utterance, duration and speaker.
        Audio is fetched separately via :func:`clip`.
    """
    manifest = _manifest()
    return {"count": len(manifest),
            "clips": sorted(manifest.values(),
                            key=lambda c: (c["emotion"], c["key"]))}


@app.function(
    image=image,
    volumes={CLIPS_DIR: clips_volume},
    cpu=0.125,
    memory=512,
    timeout=60,
)
@modal.fastapi_endpoint(method="GET", docs=True, requires_proxy_auth=REQUIRES_PROXY_AUTH)
def clip(key: str) -> dict:
    """One corpus clip's audio, for browser playback.

    Args:
        key: Corpus key, e.g. ``dia0_utt0``.

    Returns:
        ``{"key":, "content_type": "audio/wav", "audio_b64": ...}``.
    """
    import base64
    from pathlib import Path

    entry = _manifest().get(key)
    if entry is None:
        return {"error": f"unknown clip key: {key}"}

    path = Path(CLIPS_DIR) / entry["file"]
    if not path.exists():
        return {"error": f"clip file missing on the volume: {entry['file']}"}

    return {"key": key, "content_type": "audio/wav",
            "audio_b64": base64.b64encode(path.read_bytes()).decode()}


# -----------------------------------------------------------------------------
# Admin: the warm-container controls
# -----------------------------------------------------------------------------

@app.function(
    image=image,
    # Deliberately no GPU and minimal resources. This endpoint exists so an
    # operator can ask "is it warm?" and "shut it down" without those
    # questions themselves costing GPU seconds. It runs at roughly
    # $0.0000075/sec -- about 3% of one percent of the inference container.
    cpu=0.125,
    memory=256,
    timeout=300,
)
@modal.fastapi_endpoint(method="POST", docs=True, requires_proxy_auth=REQUIRES_PROXY_AUTH)
def admin(item: dict) -> dict:
    """Inspect the inference container, or start one early.

    Two actions, and neither can leave a GPU running indefinitely:

        status  read-only, free, and served from a CPU container so asking
                never wakes a GPU to answer.
        warm    fire a warmup so the models load NOW rather than on whoever
                arrives first. The container then scales down on the normal
                window like any other -- this does not pin it.

    There is deliberately no "keep warm forever". An earlier version set
    min_containers=1 and offered a Stop button, which meant the cheapest
    possible mistake -- forgetting to press Stop -- cost about $80 over a
    long weekend. SCALEDOWN_WINDOW does the same job and cannot be forgotten.

    ``unpin`` remains as a safety net for a deployment left pinned by that
    older version; it is a no-op on a healthy one.

    Args:
        item: ``{"action": "status" | "warm" | "unpin"}``.

    Returns:
        Container state, the cold-start estimate, and the live hourly rate.
    """
    import time

    action = (item.get("action") or "status").lower()
    if action not in ("status", "warm", "unpin"):
        return {"error": f"unknown action {action!r}; "
                         f"expected status, warm or unpin"}

    pipeline = modal.Cls.from_name(APP_NAME, "EmotionPipeline")()
    now = time.time()

    if action == "warm":
        # spawn, not remote: the caller gets an immediate answer and polls
        # status, rather than holding an HTTP connection open for a minute.
        pipeline.warmup.spawn()
    elif action == "unpin":
        pipeline.update_autoscaler(min_containers=0)
        warm_state.pop("pinned_at", None)

    last_seen = warm_state.get("last_seen")

    containers = None
    stats_error = None
    try:
        # Through the CLASS handle, not Function.from_name. Modal bundles a
        # class's methods into one service function named "EmotionPipeline.*",
        # so looking up "EmotionPipeline.analyse" as a Function raises NotFound
        # -- and swallowing that silently is how this once reported no
        # containers while two were running.
        containers = pipeline.analyse.get_current_stats().num_total_runners
    except Exception as exc:                                     # noqa: BLE001
        stats_error = f"{type(exc).__name__}: {exc}"[:200]
        print("container stats unavailable:", stats_error, flush=True)

    warm = containers > 0 if containers is not None else bool(
        last_seen and (now - last_seen) < SCALEDOWN_WINDOW)

    return {
        "action": action,
        "warm": warm,
        "containers": containers,
        "containers_error": stats_error,
        "model": warm_state.get("model"),
        # What the next request will actually wait, rather than a guess: the
        # last measured load, or the observed ~60 s round trip if never run.
        "cold_start_seconds": warm_state.get("load_seconds") or 60,
        "scaledown_window_sec": SCALEDOWN_WINDOW,
        "container_usd_per_hour": round(CONTAINER_USD_PER_SEC * 3600, 3),
        "seconds_since_last_request": (None if last_seen is None
                                       else round(now - last_seen)),
        # Worst case left on the clock if nobody calls again.
        "idle_tail_usd": round(SCALEDOWN_WINDOW * CONTAINER_USD_PER_SEC, 3),
    }


# -----------------------------------------------------------------------------
# Cost measurement
# -----------------------------------------------------------------------------

@app.function(
    image=image,
    volumes={CLIPS_DIR: clips_volume},
    # CPU only, on purpose. The benchmark MEASURES the GPU container; it must
    # not be one, or every run pays for two. An earlier version ran on a GPU
    # and called the `@modal.enter` hook directly, which is not a callable
    # method on a Modal class -- it failed with KeyError('load').
    cpu=0.5,
    memory=1024,
    timeout=60 * 30,
)
def benchmark(n: int = 8) -> dict:
    """Measure cold start and per-clip time against the DEPLOYED endpoint.

    Every cost claim about this service reduces to two numbers -- how long the
    models take to load and how long a clip takes -- and both are cheaper to
    measure once than to argue about.

    Cold start is timed as the round trip of the first ``warmup`` call, which
    is what a user actually waits for: container schedule, image pull, then
    model load. Timing the load alone would flatter it.

    Args:
        n: Corpus clips to analyse after warm-up.

    Returns:
        Timings and the derived per-clip and per-hour cost.
    """
    import json
    import time
    from pathlib import Path

    manifest_path = Path(CLIPS_DIR) / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(
            f"No manifest on the clips volume. Run prepare_clips.py and "
            f"`modal volume put emotion-clips <dir>/manifest.json /manifest.json`.")
    with open(manifest_path, "r", encoding="utf-8") as fh:
        keys = [c["key"] for c in json.load(fh)]
    if not keys:
        raise RuntimeError("The clip manifest is empty.")

    pipeline = modal.Cls.from_name(APP_NAME, "EmotionPipeline")()

    t0 = time.time()
    warm = pipeline.warmup.remote()
    cold_start_s = time.time() - t0
    print(f"cold start {cold_start_s:.1f}s (model load {warm['load_seconds']}s)")

    results, t0 = [], time.time()
    for key in (keys * n)[:n]:
        results.append(pipeline.analyse.remote(b"", key=key))
    infer_s = time.time() - t0
    per_clip = infer_s / max(1, n)

    correct = sum(1 for r in results if r.get("correct", {}).get("emotion"))
    stages = ("decode_ms", "vad_ms", "acoustic_ms", "transcribe_ms",
              "text_ms", "head_ms")

    out = {
        "gpu": GPU,
        "clips": n,
        "container_usd_per_hour": round(CONTAINER_USD_PER_SEC * 3600, 3),
        "cold_start_seconds": round(cold_start_s, 1),
        "model_load_seconds": warm["load_seconds"],
        "cold_start_usd": round(cold_start_s * CONTAINER_USD_PER_SEC, 4),
        "seconds_per_clip": round(per_clip, 2),
        "usd_per_clip": round(per_clip * CONTAINER_USD_PER_SEC, 6),
        "usd_per_1000_clips": round(per_clip * 1000 * CONTAINER_USD_PER_SEC, 2),
        "free_tier_hours_per_month": round(
            30 / (CONTAINER_USD_PER_SEC * 3600), 1),
        "emotion_correct": f"{correct}/{n}",
        "mean_stage_ms": {
            stage: round(sum(r["timings_ms"][stage] for r in results) / len(results))
            for stage in stages
        },
        "example": {
            "key": results[0]["key"],
            "transcript": results[0]["transcript"],
            "predicted": results[0]["prediction"]["emotion"],
            "gold": results[0].get("ground_truth", {}).get("emotion"),
            "wer": (results[0].get("wer") or {}).get(
                "policies", {}).get("normalised", {}).get("wer"),
            "speech_ratio": results[0]["vad"]["speech_ratio"],
        },
    }
    print(json.dumps(out, indent=2))
    return out


@app.local_entrypoint()
def main(key: str = "dia0_utt0") -> None:
    """Smoke the deployed class against one corpus clip.

    Args:
        key: Corpus key to analyse. The container reads the audio from its own
            clips volume, so nothing has to be uploaded from here -- which
            also means this entrypoint needs no local audio dependencies.
    """
    import json

    print(json.dumps(EmotionPipeline().analyse.remote(b"", key=key), indent=2))
