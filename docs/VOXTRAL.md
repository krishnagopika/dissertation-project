# Voxtral — the audio front-end

What Voxtral is, which variant we use, where the acoustic hook is placed and
why, how the prompts were arrived at, and how both artefacts are extracted from
a single model load.

**Scope: methodology.** What the system does and why it is built that way.
Decision rationale and alternatives rejected are in `DECISIONS.md`;
implementation history is in `CHANGE_RECORD.md`; failure analysis is in
`POSTMORTEMS.md`. This document does not repeat them.

---

## 1. What Voxtral is

Voxtral is Mistral's audio-language model: a **Whisper large-v3 encoder** bolted
to a **Mistral LLM decoder** through a small projection adapter. It takes audio
plus a text instruction and generates text — so it does ASR, audio QA, and
audio-conditioned generation through one interface.

```
raw audio (16 kHz)
   │
   ▼  log-mel, hop 160  →  100 Hz frames
[ Whisper large-v3 encoder ]  32 layers, d_model 1280
   │
   ▼  (T, 1280) @ 50 Hz          ◄── WE TAP HERE
[ ×4 temporal downsample ]  concat 4 adjacent frames → 5120
   │
[ AudioLanguageAdapter ]  Linear(5120→3072) → GELU → Linear(3072→3072)
   │
   ▼  (T/4, 3072) audio tokens @ 12.5 Hz
[ Mistral decoder ]  d_model 3072
   │
   ▼  generated text
```

All dimensions confirmed from the model's own `params.json`, not from
documentation: encoder `dim: 1280`, `downsample_factor: 4`, LLM `dim: 3072`,
`hop_length: 160`, `max_source_positions: 1500`.

### Variants

| variant | params | encoder | LLM dim | local size |
|---|---|---|---|---|
| `Voxtral-Mini-3B-2507` | 3B | Whisper large-v3 | 3072 | 9.3 GB |
| `Voxtral-Small-24B-2507` | 24B | Whisper large-v3 | larger | 48.5 GB |

**Both wrap the same Whisper large-v3 encoder.** That matters for this project:
the acoustic branch is 1280-d for both, so Mini and Small differ in *decoder*
capacity — i.e. in transcription quality — not in the acoustic representation.

We use **Mini** for everything reported so far. Small is a scale comparison, not
a prerequisite (ADR-006).

---

## 2. Where the hook goes, and why not after the MLP

The acoustic branch taps the **encoder output** — `(T, 1280)` at 50 Hz — and not
the adapter output. Three reasons, in order of weight.

### 2.1 The adapter is trained for the wrong objective

Its job is to make audio look like *text tokens* to the decoder, optimised for
autoregressive transcription. Anything the LLM does not need in order to emit
the correct words — speaker identity, prosodic contour, voice quality, channel
character — is free to be discarded there. Those are precisely the
paralinguistic cues emotion recognition depends on.

The encoder output has not yet been filtered through "what is needed to produce
the right words".

### 2.2 The adapter destroys temporal resolution

`downsample_factor: 4` concatenates four adjacent frames, taking the sequence
from **50 Hz (20 ms) to 12.5 Hz (80 ms)**.

This only became visible as a *decisive* argument once attention pooling was
adopted (ADR-003): attention weights frames individually, so pooling after the
adapter would attend over units 4× coarser. At MELD's median 2.7 s utterance
that is 134 frames versus 33 — the difference between resolving a prosodic peak
and averaging over it.

### 2.3 Comparability

Whisper-encoder features are a standard frozen representation in speech-emotion
research, so results sit alongside existing baselines. Post-adapter features are
Voxtral-specific with nothing to compare against.

### The honest counter-argument

Post-adapter is the representation the decoder *actually receives*. If the
question were "why did Voxtral's own zero-shot emotion predictions come out this
way", post-adapter would be the faithful choice. It is the wrong choice for
*training a separate classifier*, which is what we do.

### Do not use `get_audio_features()`

HF's `model.get_audio_features(input_features)` runs encoder **and** adapter, so
it returns the post-adapter representation. `src/models/voxtral.py` deliberately
calls `self.model.audio_tower(...)` directly instead. This is easy to get wrong:
the convenience method is the obvious one to reach for and silently gives the
other thing.

---

## 3. Prompt design

### 3.1 Transcription

Final prompt (`transcribe_all.py`):

```
Output only the verbatim spoken words from this audio.
Plain text only. No timestamps, no speaker labels, no formatting.
If unclear, output your best guess. Do not say you did not understand.
```

Every clause exists because of an observed failure mode:

| clause | failure it suppresses |
|---|---|
| "Output only the verbatim spoken words" | the model summarising or paraphrasing instead of transcribing |
| "Plain text only. No timestamps, no speaker labels, no formatting" | Whisper-style `[00:00:03]` and `Speaker 1:` prefixes leaking into the transcript and being scored as insertions against MELD gold |
| "If unclear, output your best guess" | refusal on noisy audio |
| **"Do not say you did not understand"** | the model emitting *"I'm not sure what you're asking"* as the transcript |

**The limit of prompting.** These clauses constrain output format; they cannot
supply missing input. A 0.28 s clip yields roughly 11 encoder frames at 50 Hz,
and at that point the decoder is generating from language-model priors rather
than from audio — the final clause reduces the resulting refusal-style output
but does not remove it.

Sub-second clips are therefore handled downstream by *flagging*, not by
instruction. See `MELD_ANALYSIS.md` §5 for the defect taxonomy.

### 3.2 Zero-shot classification

Used for the dialect probe, not for the MELD pipeline
(`evaluation/voxtral_zeroshot.py`):

```
You are an expert at recognising emotion and sentiment from speech.
Listen to the audio and classify the speaker's emotional state.
Judge from tone, pitch, pace and intonation as well as the words.

Choose EXACTLY ONE emotion from this list: {…}.
Choose EXACTLY ONE sentiment from this list: {…}.

Respond in EXACTLY this format and nothing else:
Emotion: <emotion>
Sentiment: <sentiment>
```

Design points:

- **"Judge from tone, pitch, pace and intonation as well as the words"** —
  without it the model classifies from transcript semantics alone, which defeats
  the purpose of giving it audio.
- **Labels enumerated explicitly** — otherwise it invents labels outside MELD's
  seven-class schema and nothing parses.
- **Format pinned** — the response is machine-parsed; free-form prose is
  unparseable at 17,877 clips.
- `max_tokens=32` for classification versus 256 for ASR — a classification that
  runs long has gone wrong anyway.

### 3.3 Decoding

`temperature=0.0` — greedy, for both passes. Transcription must be reproducible:
the WER keep-lists are derived from these transcripts, so a sampled decode would
make the filter non-deterministic.

`max_tokens=200` for ASR. This is what bounds runaway repetition — the failure
mode in `MELD_ANALYSIS.md` §5 where a 1-word gold produces a 176-word output. The
cap turns an unbounded loop into a bounded one; it does not prevent it.

---

## 4. Extraction — one model load, both artefacts

### 4.1 Why one pass, not two

A naive implementation loads Voxtral twice — once under vLLM to transcribe,
once under HF transformers to run the encoder for embeddings — and, if the
engine is rebuilt per split, four times for a three-split corpus.

That is unnecessary, because vLLM **already runs the Whisper encoder on every clip** — it must, to build
the audio tokens the decoder attends to — and then discards the output once the
adapter has projected it. Pass 2 was recomputing something Pass 1 had already
computed and thrown away.

### 4.2 The hook

`vllm/model_executor/models/voxtral.py`:

```python
def embed_multimodal(self, **kwargs):
    audio_embeddings = self.whisper_encoder(audio_inputs)   # ← hooked
    ...                                                      # pad → reshape
    audio_embeddings_packed = self.audio_language_adapter(...)   # discarded after
```

`whisper_encoder` is a `VoxtralEncoderModel` whose forward is
`forward(list[Tensor]) -> list[Tensor]` — one waveform in, one `(T, 1280)`
encoder state out, **same order**. A forward hook captures it in flight.

Registered via `llm.apply_model()`, which requires
`VLLM_ALLOW_INSECURE_SERIALIZATION=1` — vLLM refuses to ship a Python callable
to the worker under its default msgpack-only serialisation.

Result: **4 model loads → 1**.

### 4.3 Matching captures back to utterances

The hook sees a batch of waveforms with **no request IDs**. vLLM's scheduler
decides batch composition, and the `mm_hash` identifying each item lives in the
model *runner*, not the model. So captures are keyed by a content hash of the
audio itself.

Two transformations sit between what is submitted and what the encoder
receives, both measured directly from the running model:

| | measured |
|---|---|
| padding | vLLM **zero-pads every clip to exactly 30 s** — 480,000 samples |
| dtype | the encoder receives **bfloat16**, cast to the model dtype, not float32 |

Both must be undone before a content hash can agree across the boundary. bf16 →
float32 does not recover the original bits, so the hash must be computed on the
bf16 form.

`canonical_waveform()` applies one reduction on both sides: cast to bf16, strip
trailing zeros, hash. The 30 s padding is trailing zeros, so a single rule covers
both transformations. Stripping is safe for clips ending in genuine digital
silence, because the driver-side copy carries the same zeros.

**The scheme fails closed.** A hash mismatch produces a *missing* key, never a
wrong pairing — which matters, because pairing one clip's acoustics with another
clip's label would corrupt training invisibly. Misses are zero-filled, logged at
ERROR, and written to `{split}_unmatched.json`.

### 4.4 Padding removed at capture

vLLM pads every clip to 30 s, so the tail of the encoder output is activation
over silence. Frames are truncated to the clip's true length at capture:

```
50 encoder frames per second  →  n_frames = ceil(n_samples / 320)
```

This is why the cache is **6.1 GB rather than ~52 GB**, and it removes the
padding defect at source rather than masking around it downstream. Verified: 24
clips went from a constant 1500 frames (an impossible 30.0 s for every clip) to
41–550 frames, median 134 = **2.7 s**, matching MELD's real durations.

### 4.5 What lands on disk

Per split, under `data/meld_extracted/{model}/`:

| artefact | contents |
|---|---|
| `transcripts/{split}_transcripts.json` | `"diaD_uttU"` → transcript string |
| `acoustic/{split}_acoustic_seq.pt` | `"diaD_uttU"` → **fp16 (T, 1280)**, padding stripped |
| `acoustic/{split}_embeddings_maskedmean.pt` | fp32 (1280,) — the parameter-free control |
| `acoustic/{split}_unmatched.json` | keys the hook failed to capture |

fp16 because the encoder ran in bf16 anyway — the extra precision was never real.

Sequences rather than pooled vectors because pooling moved into the trainable
head (ADR-002/003): **a learned pooling cannot be baked into a cache written
before training starts.** One extraction now serves every pooling variant.

Writes are atomic: `torch.save` to `.partial`, then `os.replace`. A reader
therefore observes either the complete file or no file — never a truncated one —
and resumption validates archive integrity rather than mere existence, since a
partially-written `.pt` satisfies an `exists()` check.

### 4.6 Why sequences, and where the pooling decision lives

The cache holds `(T, 1280)` sequences, not one pooled vector per utterance.
Voxtral's responsibility ends at producing those frames; **the choice of how to
collapse them belongs to the model that consumes them**, which is why the
pooling ablation is specified in `DECISIONS.md` ADR-003 and implemented in
`src/models/pooling.py` rather than here.

The reason it cannot live on this side is concrete: **attention pooling has
learned parameters, and they do not exist until training starts.** A pooled
cache would have to be written before any training, which forces the pooling to
be parameter-free. Moving the cache boundary one stage earlier removes that
constraint while keeping Voxtral frozen and run exactly once (ADR-002).

One extraction therefore serves every pooling variant:

| pooler | trainable params | output | isolates |
|---|---:|---|---|
| unmasked mean | 0 | 1280 | the historical baseline |
| `masked_mean` | 0 | 1280 | how much was *only* the padding defect |
| `attention_fixed` | 0 (frozen at random init) | 1280 | non-uniform but *uninformative* weighting |
| `attention` | ~164 k | 1280 | **learned** frame weighting |
| `attentive_stats` | ~164 k | **2560** | + prosodic variance (mean ⊕ std) |

`masked_mean` is the control that matters. Without it, a gain from attention
pooling is unattributable: the model may have learned *where the emotion is*, or
merely *to ignore the padding* — and §4.4 showed the padding was ~91% of a
median utterance's frames before truncation. `masked_mean` is verified to
reproduce the old `.mean(dim=1)` exactly when no frame is padded, so the two are
directly comparable.

Attention weights are returned rather than discarded, because plotting them over
time shows *where* in an utterance the model attends — and comparing that across
accents is dialect-robustness evidence that WER alone cannot provide.

#### What the consuming model must provide

```
CURRENT
  {split}_embeddings.pt   ──►  (B, 1280)  ──►  FusionModel

NEEDED
  {split}_acoustic_seq.pt ──► collate ──► (B, T, 1280) + mask (B, T)
                                              │
                                        build_pooler(cfg)
                                              ▼
                                          (B, out_dim)  ──►  FusionModel
```

The fusion variants themselves operate on a pooled vector and are unaffected by
the choice of pooler. Two requirements fall on the consuming model:

- **A collator** that pads variable-length sequences into `(B, T, 1280)` and
  emits the corresponding boolean mask.
- **A pooler owned by the fusion model**, so its parameters are trained jointly
  with the classifier rather than fixed in advance.

The acoustic input dimension must be taken from `pooler.output_dim` rather than
from `acoustic_dim`: `attentive_stats` concatenates the weighted mean and
standard deviation and therefore outputs 2560, not 1280.

---

## 5. Measured output (Voxtral-Mini, MELD)

| | train | dev | test |
|---|---:|---:|---:|
| utterances | 9,989 | 1,109 | 2,610 |
| corpus WER (normalised) | 0.3461 | 0.3124 | 0.3823 |
| corpus WER, defects excluded | 0.2908 | 0.2392 | 0.2691 |
| sequence cache | 4.02 GB | 0.44 GB | 1.13 GB |

Per-accent WER on `english_dialects` (17,877 clips): Southern 0.060, Scottish
0.061, Midlands 0.064, Welsh 0.073, Northern 0.075, **Irish 0.123**.

---

## Reproducing

```bash
sbatch --gres=gpu:1 --export=ALL,MODEL=mini \
  --job-name=extract_mini src/scripts/extract_meld.sbatch
```

`158 frames avg` in the log is the expected signal — 158/50 ≈ 3.2 s, matching
MELD. A constant `1500` means padding is being captured instead of audio.
