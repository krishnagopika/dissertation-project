# Architecture Decision Records

A running log of design decisions, why they were made, what was rejected, and
what evidence or literature backs them. Append new records; do not rewrite old
ones — if a decision is reversed, mark the original **Superseded** and add a new
record explaining what changed.

Each record carries an explicit **Status**, because several of these are agreed
in principle but not yet implemented. Do not read a record as a description of
what the code currently does unless its status says so.

| Status | Meaning |
|---|---|
| Implemented | In the code now, this is current behaviour |
| Partial | Component written and tested, not yet wired into the pipeline |
| Proposed | Agreed in principle, no code yet |
| Superseded | Replaced — see the record that supersedes it |

---

## ADR-001 — Acoustic embeddings are taken at the encoder output, not after the MLP projector

**Status:** Implemented
**Date:** recorded 2026-08-23 (decision predates this record)
**Files:** `src/models/voxtral.py::extract_acoustic_embeddings`, `src/configs/mini.yaml` (`acoustic_dim: 1280`)

### Context

Voxtral's audio path has two stages after the mel front-end:

```
waveform → Whisper-large-v3 encoder → (T, 1280) @ 50 Hz
         → ×4 temporal downsample (concat 4 adjacent frames → 5120)
         → AudioLanguageAdapter (5120 → 3072)   ["the MLP projector"]
         → audio tokens the LLM attends over
```

Confirmed from `params.json`: encoder `dim: 1280`, `downsample_factor: 4`,
LLM `dim: 3072`, `hop_length: 160`, `max_source_positions: 1500`.

Either stage could supply the acoustic branch. We take the **encoder** output.

### Decision

Extract at `audio_tower(...).last_hidden_state` — 1280-d, 50 Hz.

### Rationale

1. **The projector is trained for the wrong objective.** Its job is to make audio
   look like *text tokens* to the language model, optimised for autoregressive
   transcription. Anything the LLM does not need in order to emit the right words
   — speaker identity, prosodic contour, channel character — is free to be
   discarded there. Those are exactly the paralinguistic cues emotion recognition
   depends on.

2. **The projector destroys temporal resolution.** The ×4 downsample concatenates
   four adjacent frames, taking the sequence from 50 Hz (20 ms) to 12.5 Hz
   (80 ms). Attention pooling (ADR-003) weights frames individually, so it wants
   the finest resolution available. Pooling after the projector would attend over
   4× coarser units.

3. **Comparability with the SER literature.** Whisper-encoder features are a
   standard frozen representation in speech-emotion work, so results sit next to
   existing baselines. Post-projector features are Voxtral-specific with nothing
   to compare against.

4. **Cost.** 1280-d vs 3072-d — smaller cache, smaller first fusion layer.

### Alternatives considered

- **Post-projector (3072-d).** Rejected for the reasons above. It has one real
  advantage worth recording: it is the representation the LLM actually receives,
  so it is the faithful choice for *explaining Voxtral's own zero-shot emotion
  predictions*. If we ever analyse those predictions rather than train on the
  features, revisit this.
- **An intermediate encoder layer.** Common in SER — mid-stack layers often carry
  more paralinguistic information than the final layer, which specialises toward
  the decoder's transcription objective. Not tried. This is a genuine open option
  and a cheap ablation once sequences are cached (ADR-002).

### Consequences

- `acoustic_dim: 1280` throughout.
- **Known documentation defect:** the `src/models/voxtral.py` module docstring
  describes the projector path and says the embedding is `text_config.hidden_size`.
  The code does not do this. `_MINI_FALLBACK_DIM = 4096` in that file is also
  wrong — Mini's LLM dim is 3072. Config and code agree; only the prose is wrong.

---

## ADR-002 — Cache acoustic *sequences*, not pooled vectors

**Status:** Implemented in preprocessing (CR-003, verified job 8082). Downstream
consumers not yet migrated — `fusion.py`, `train_fusion.py`, `train_context.py`
still read the legacy fixed-size cache.
**Files:** `src/preprocessing/transcribe_all.py`, `src/preprocessing/acoustic_hook.py`

### Context

The pipeline caches one 1280-d vector per utterance and never loads Voxtral
during training. That constraint forces the pooling to be
parameter-free — a *learned* pooling cannot be baked into a cache written before
training begins.

### Decision

Move the cache boundary one stage earlier: store the `(T, 1280)` frame sequence
per utterance, and pool inside the trainable head.

### Rationale

Voxtral stays frozen and still runs exactly once, so §16 is honoured in
substance. Only the boundary moves. This unlocks every pooling variant in
ADR-003 without re-running Voxtral per experiment — one extraction, N pooling
ablations.

### Storage estimate

Frames arrive at 50 Hz, so `T = samples / 320`.

| Variant | Size (13,708 utterances) |
|---|---|
| Padded to 30 s (T=1500), fp16 | ≈ 52 GB |
| True length, fp16 | ≈ 6–7 GB (MELD utterances are short) |

**Measured** (job 8082, 24 dev clips): 10.7 MB → **6.1 GB projected** for all
13,708 utterances. Frame counts 41–550, median 134 = 2.7 s at 50 Hz, which
tracks MELD's actual utterance durations.

Store **unpadded, fp16**. This also removes the padding defect in ADR-004 at
source rather than masking around it.

### Consequences

- Invalidates every cached `{split}_embeddings.pt`.
- Every downstream model that assumes a fixed-size acoustic vector must change.
- All acoustic and fusion results must be regenerated. bc-LSTM retrains in ~1 min;
  fusion is the real cost.
- **Open question, unresolved:** whether existing `{split}_transcripts.json` and
  the `*_filtered_keys_*.json` keep-lists derived from them are preserved
  byte-for-byte, or regenerated. Keep-lists were computed against the existing
  transcripts, so regenerating transcripts silently invalidates them.

---

## ADR-003 — Attention pooling in the trainable head, with masked mean as the control

**Status:** Partial — `src/models/pooling.py` written and unit-tested; not wired
into any training script. Depends on ADR-002.
**Files:** `src/models/pooling.py`

### Context

The original pooling was `hidden.mean(dim=1)` — an unweighted average over the
time axis. Two independent problems:

1. **Padding.** Every clip reaches the encoder padded to 30 s, and the mean is
   taken over all 1500 frames with **no mask**. For a short utterance most of
   the averaged frames are padding — at MELD's median 2.7 s, ~91% of them.

   This was originally attributed to HF's `padding="max_length"` in
   `_prepare_input_features`. That was incomplete: **vLLM pads too** (measured
   job 8081 — every waveform arrives at the encoder with exactly 480,000
   samples). The defect is a property of Voxtral's audio front-end, not of one
   loading path, so *any* pooling over raw encoder output in either framework
   averages across ~30 s regardless of utterance length.
2. **Uniform weighting.** Mean pooling assumes every frame is equally
   informative. For emotion this is false — affect concentrates in prosodic
   peaks, not uniformly across an utterance.

### Decision

Implement three poolers behind one interface and treat pooling as an **ablation
axis**, not a swap:

| Pooler | Trainable | Output | Isolates |
|---|---|---|---|
| unmasked mean | 0 | 1280 | historical baseline |
| `MaskedMeanPooling` | 0 | 1280 | how much was *only* the padding defect |
| `attention_fixed` | 0 | 1280 | non-uniform but *uninformative* weighting |
| `AttentionPooling` | 164,096 | 1280 | **learned** frame weighting |
| `AttentiveStatsPooling` | 164,096 | 2560 | additional gain from prosodic *variance* |

**Why `attention_fixed` is in the ladder.** `masked_mean → attention` moves two
variables at once: the weighting scheme *and* trainable capacity (0 → 164,096
parameters). That delta cannot separate "learned frame weighting helps" from
"extra capacity helps" — a reviewer is entitled to push on it. `attention_fixed`
is the same module frozen at random init: identical architecture, identical
forward cost, zero trainable parameters. Verified to produce genuinely
non-uniform weights on realistic frames (max/min 3.2×, KL from uniform 0.038), so
it is a real rung rather than a disguised mean. `attention_fixed → attention` is
then the clean test of whether *learning* the weights helps, architecture held
constant.

### Rationale

**Why masked mean must exist.** If attention pooling were introduced alone and it
won, the result would be unattributable: the model may have learned *where the
emotion is*, or merely *to ignore the padding*. Those are very different claims.
Masked mean is the control that separates them, and it is parameter-free, so any
remaining gap is genuinely attributable to learned weighting. `MaskedMeanPooling`
is verified to reproduce the old `.mean(dim=1)` exactly when no frame is padded,
which makes the two directly comparable.

**Why attentive stats.** A mean discards variability entirely, but variability of
pitch and energy is part of how affect is realised. Okabe et al. pair the
attention-weighted mean with the attention-weighted standard deviation for
exactly this reason.

**Why the query lives outside the encoder.** Attention pooling with a single
learned query is functionally "a CLS token in the trainable head" — see ADR-005
for why a real CLS token is not available here.

### Alternatives considered

- **Keep mean pooling, just add a mask.** Cheapest fix, no cache change, no
  retraining of the fusion contract. Rejected as the *only* change because it
  leaves the uniform-weighting assumption untouched — but retained as the
  control, so nothing is lost.
- **Multi-head attention pooling.** More capacity. Rejected for now: MELD's
  minority classes are tiny (dev has 5 disgust, 9 fear after wer25 filtering) and
  extra parameters are more likely to overfit than to help.
- **Self-attention over frames before pooling.** A transformer layer on top of
  the frozen encoder. Rejected as redundant — the encoder is already 32
  self-attention layers.

### Consequences

- Attention weights are returned, not discarded. Plotting them over time shows
  *where* the model looks, and comparing that across accents is dialect-robustness
  evidence that WER alone cannot provide.
- Blocked on ADR-002: attention pooling cannot run against a cache of pooled
  vectors.

### References

- Okabe, Koshinaka & Shinoda (2018). *Attentive Statistics Pooling for Deep
  Speaker Embedding*. Interspeech 2018. — attentive-stats formulation.
- Bahdanau, Cho & Bengio (2015). *Neural Machine Translation by Jointly Learning
  to Align and Translate*. ICLR 2015. — additive attention.
- Reimers & Gurevych (2019). *Sentence-BERT*. EMNLP 2019. — evidence that frozen
  CLS underperforms mean pooling, relevant to ADR-005.

---

## ADR-004 — tanh as the attention scoring activation

**Status:** Implemented in `src/models/pooling.py::AttentionPooling._attend`

### Context

Additive attention scores each frame as `e_t = vᵀ f(W h_t + b)`, then softmaxes
over time. `f` was chosen as tanh. ReLU was raised as an alternative.

### Decision

tanh, with the scorer's activation left as a candidate ablation flag.

### Rationale

1. **Bounded output keeps softmax out of saturation.** tanh bounds `f` to (−1, 1),
   so `e_t` is bounded by `‖v‖₁`. ReLU is unbounded above, so scores can grow
   without limit; once the gap between frames exceeds ~10 the softmax is
   effectively one-hot, every other frame's gradient vanishes, and it cannot
   recover. With a **single** query and no multi-head redundancy, that collapse is
   terminal. This is the main reason.
2. **ReLU collapses low-scoring frames together.** It zeroes negative
   pre-activations, so frames negative across most hidden dims map to the same
   all-zero vector → identical scores → identical weights. The layer's entire job
   is *ranking* frames; tanh is zero-centred and sign-preserving, so negative
   evidence stays distinguishable.
3. **The usual argument for ReLU does not apply.** ReLU addresses vanishing
   gradients through *deep* stacks. This scorer is 1280 → 128 → 1: one
   nonlinearity, no depth to compound through.
4. **Citability.** Bahdanau et al. and Okabe et al. both specify tanh. Since
   Okabe is cited for the stats formulation, deviating on the activation means
   owning the justification rather than inheriting it.

### Alternatives considered

- **ReLU.** Rejected — see above.
- **Scaled dot-product scoring**, `e_t = (q·h_t)/√d`, no nonlinearity. The more
  interesting alternative: it is what modern attention actually uses and handles
  the saturation problem via √d scaling instead of a bounded activation. Not yet
  tried; a reasonable ablation row.
- **GELU / SiLU.** Unbounded above like ReLU, so they inherit problem (1) while
  only partially fixing (2).

### Consequences

Implementation detail worth keeping: masking uses `torch.finfo(dtype).min`, not
`-inf`. A fully-masked row softmaxed over `-inf` yields NaN, and NaN gradients
are unrecoverable. Verified: a fully-padded row returns finite values from all
three poolers.

---

## ADR-005 — No CLS token on the acoustic branch

**Status:** Implemented (by omission) — recorded to close the question

### Context

CLS pooling was considered as an alternative to mean and attention pooling.

### Decision

Not available on the audio path. Use attention pooling (ADR-003) instead.

### Rationale

Whisper's encoder has **no CLS token** — conv front-end, transformer with
sinusoidal positions, no prepended special token. Whisper was trained with a
decoder cross-attending over the full encoder sequence, so nothing ever pressured
any single position into becoming a summary. Voxtral adds none either: its
adapter projects every frame and the LLM attends over all of them.

Inventing one would mean prepending a learnable token and letting self-attention
aggregate into it — which fails here because **the encoder is frozen**. A
randomly-initialised token pushed through attention weights never trained to
route information into it produces noise. Making it work would require unfreezing
and training the encoder, breaking the frozen-Voxtral constraint the pipeline is
built on.

Even where CLS legitimately exists it underperforms when frozen: Sentence-BERT's
central finding is that frozen BERT's CLS is worse than plain mean pooling, and
CLS only becomes good under fine-tuning pressure.

### Consequences

- On the **text** branch CLS *is* used and *is* legitimate — XLM-R has a
  pretrained one (`extract_text_embeddings.py` takes the 768-d `[CLS]`). Because
  CLS sits at position 0, padding to 128 tokens does not dilute it, so the text
  branch never had the audio branch's padding defect.
- A single learned attention query is the nearest working equivalent, and is what
  ADR-003 implements.

---

## ADR-006 — Both Voxtral models run through the SAME vLLM pipeline

**Status:** Implemented — `src/scripts/extract_meld.sbatch` + `src/configs/extract_{mini,small}.yaml`
**Date:** 2026-08-23

### Context

Voxtral-Small never ran. Every attempt failed inside
`vllm/model_executor/models/voxtral.py` — first
`AttributeError: 'VoxtralEncoderConfig' object has no attribute 'downsample_factor'`,
then, after patching that, `TypeError: 'NoneType' object is not iterable`.

The cause is a **weight format** mismatch, not model size or GPU capacity. vLLM
selects its loader by looking for one file:

```python
def is_mistral_model_repo(...):        # picks config_format
    allow_patterns=["consolidated*.safetensors"]

if load_format == "auto":              # picks load_format
    load_format = "mistral" if <consolidated*.safetensors present> else "hf"
```

Both repos publish **both** formats. Mini's local snapshot happened to contain
`consolidated.safetensors` + `params.json` + `tekken.json` (Mistral), so both
checks selected `mistral` and it worked. Small's snapshot contained the
HF-format shards `model-0000N-of-00011.safetensors` instead, so both checks
selected `hf` — a path vLLM's Voxtral cannot serve, because
`VoxtralEncoderModel.load_weight` only remaps *Mistral* weight names
(`mm_whisper_embeddings.whisper_encoder.transformer.layers.N.attention.wq` →
`whisper_encoder.layers.N.self_attn.q_proj`). There is no HF→vLLM mapping.

Patching `downsample_factor` therefore bought one step down a list of missing
mappings, which is why the next error was different but no more tractable.

### Decision

Fetch `consolidated.safetensors` (48.5 GB) for Small and delete its HF shards.
Run **both** models through one pipeline, parameterised by config.

### Alternative rejected: run Small under HF transformers

`VoxtralWrapper` uses `VoxtralForConditionalGeneration.from_pretrained`, which
reads the HF shards directly — so Small could have run today with no download.
Rejected: it would mean Mini on vLLM and Small on HF transformers, so any
Mini-vs-Small difference would confound **model** with **inference framework**
and be unattributable. This is the same reasoning `dialect_probe.py` already
applies in reverse ("using Small here would confound accent shift with
different model").

The engineering argument points the same way: one pipeline configured by
environment is modular; two pipelines is duplicated surface that must be kept
in step.

### Consequences

- One script, `extract_meld.sbatch --export=MODEL=mini|small`; one config
  schema, `extract_${MODEL}.yaml` carrying `voxtral_id`,
  `tensor_parallel_size`, and output paths. A third model is one YAML file.
- Each model writes only to `data/meld_extracted/${MODEL}/`, so they cannot
  overwrite each other and neither touches the legacy caches.
- Small's HF shards deleted; every file recorded in
  `docs/small_hf_shards_deleted.txt` with the `allow_patterns` to re-fetch.
- Quota made this ordering necessary: delete first (118 → 71 GB), download
  second. Not risk-free, but the shards are re-downloadable and HF resumes
  partial transfers, so a failure costs transfer time, not work.

---

## ADR-007 — XLM-R fine-tuning: text source, loss variants, and run isolation

**Status:** Implemented — `src/configs/xlmr_{gold,asr,asr_cleaned}_{plain,weighted,focal}.yaml`,
`src/scripts/xlmr_grid.sbatch`, `src/training/finetune.py`
**Date:** 2026-08-26
**Related:** `POSTMORTEMS.md` PM-001, PM-010 · `MELD_ANALYSIS.md` §5.3

### Context

Phase 1 fine-tunes XLM-RoBERTa on MELD utterances for joint emotion + sentiment
classification. Four questions had to be settled before any run was meaningful.

### Decision 1 — Train on ASR transcripts, with gold as an explicit control

`TranscriptDataset` accepted `transcripts_path`, documented itself as serving
ASR transcripts, and then read `row["utterance"]` — MELD's **gold** text —
regardless. The parameter was never used. Meanwhile `evaluate_text_only.py`
correctly scored ASR transcripts.

So every text-only and fusion result produced before 2026-08-26 **trained on
gold text and was evaluated on ASR text**: a train/test domain mismatch nobody
chose. `text_source` is now an explicit, required decision at the call site.

This is not a cosmetic fix. It makes WER-filtering the training data
*meaningful for the first time*: a branch that never sees a transcription error
cannot benefit from removing utterances that have them. The previously observed
null result for training-set filtering (±0.3 F1) is therefore uninterpretable —
it was measured under a configuration where the treatment could not act.

`gold` is retained as a run condition, and gold-trained models are additionally
scored on ASR text (`_ASRMISMATCH` tag). The gap between those two numbers
measures what the bug cost.

### Decision 2 — Three loss variants forming an incremental ladder

| run | loss | alpha | gamma |
|---|---|---|---|
| A `plain` | `CrossEntropyLoss(weight=None)` | — | — |
| B `weighted` | `CrossEntropyLoss(weight=w)` | inverse frequency | — |
| C `focal` | `FocalLoss(weight=w, gamma=2.0)` | inverse frequency | 2.0 |

`w = N / (C * count_c)` — inverse frequency. `use_weighted_sampler: false` in
all nine configs, so sampling never confounds the loss comparison.

**A exists because there was no unweighted baseline.** Class weights were
previously applied unconditionally, so nothing in the project measured what
weighting actually buys.

**C keeps alpha deliberately**, though a literal reading of "focal loss with
gamma=2" would drop it. With alpha in both B and C, **B→C isolates gamma** and
**A→B isolates alpha** — each step changes exactly one variable. Dropping alpha
from C would make B and C differ in two variables simultaneously, so a
"C beats B" result could not be attributed. This also matches the standard
formulation (Lin et al., 2017), which uses both.

Motivated by the label imbalance measured in `MELD_ANALYSIS.md` §2: 17.6:1
neutral-to-fear in train, with only 22 disgust and 40 fear in dev.

### Decision 3 — Filter the text branch, not the acoustic branch

`asr_cleaned` uses `{split}_keys_text_clean.json`: audio defects **plus**
WER > 0.50. `asr` uses everything.

Per `MELD_ANALYSIS.md` §5.3, the WER gate removes ~2,214 train utterances whose
**audio is fine** and whose emotion label is valid — only the transcript is
unusable. Those clips are legitimate training data for the acoustic branch and
noise for the text branch. Applying one gate to both discards usable acoustic
data, and is the leading candidate explanation for the earlier null result.

Threshold 0.50 rather than 0.40: 36% of MELD references are <= 4 words, so WER
is quantised to {0, 0.25, 0.50, 0.75, 1.0} on more than a third of the corpus.
A 0.40 gate sits in a gap where nothing can land — its only effect is to exclude
the WER == 0.50 bucket. See `MELD_ANALYSIS.md` §5.2.

### Decision 4 — One checkpoint directory per run, and no resuming

`finetune.py` reads `checkpoint_dir` straight from config with **no run tag**,
and resumes from any checkpoint it finds there. Two failure modes follow:

- runs sharing a directory silently overwrite each other (**PM-001**, which
  destroyed a bc-LSTM checkpoint);
- a run finding a previous checkpoint **warm-starts from the wrong model**
  instead of base XLM-R, which would invalidate the comparison without any
  error.

Each of the nine runs therefore gets its own `checkpoint_dir`, and the sbatch
**refuses to train** into a directory that already holds `best_model.pt` rather
than resuming into it.

### Decision 5 — Early stopping on dev weighted F1

Patience 3, on **dev weighted F1** — the model-selection metric — not on
validation loss. `best_model.pt` already tracks the best epoch, so stopping
early costs nothing but avoids wasted epochs across nine runs.

Weighted F1 rather than loss because loss is dominated by the majority class
under this imbalance and can improve while minority-class F1 degrades.

### Consequences

- Nine runs: 3 data conditions x 3 losses, each differing in one variable.
- Results in `results_new/xml_roberta/{gold,asr,asr_cleaned}/`, one JSON per
  split (train/dev/test) per loss, all produced by the same evaluator so no
  number comes from a different code path than any other.
- TensorBoard per run at `logs/xlmr/{condition}/{loss}/tb`: train and dev loss,
  weighted and macro F1, sentiment F1, learning rate, epochs-since-best, and
  **per-class F1** — the weighted average hides exactly the minority classes
  these variants target.
- **All of this is Voxtral-Mini data.** Every transcript, WER figure and
  keep-list comes from `meld_extracted/mini/`. Small requires the same
  treatment once its weights land (ADR-006); the configs are parameterised by
  model rather than hardcoded so that is a config change, not a code change.

### Not settled

- Whether `asr_cleaned` should also drop defects from the **dev** set. Filtering
  dev changes the model-selection criterion, not just the training data.
- Whether to report test numbers on the full test set, the clean subset, or
  both. Filtering test breaks comparability with published MELD results
  (`MELD_ANALYSIS.md` §7).

---

## Related documents

- **`CHANGE_RECORD.md`** — dated implementation notes: what changed in the code
  and why. Decisions live here; changes live there.
- **`POSTMORTEMS.md`** — incidents: what went wrong, and what would have caught it.

---

## Open questions not yet decided

1. **Does the text branch use gold text or ASR transcripts?**
   `extract_text_embeddings.py` reads `row["utterance"]` — human gold text from
   the MELD CSVs — while `extract_text_embeddings_asr.py` exists separately. Which
   one produced the cached `meld_text_embeddings/` determines whether published
   fusion and bc-LSTM results reflect an end-to-end speech pipeline at all. It may
   also explain why filtering the *training* data was worth only ±0.3 points: if
   the text branch never saw ASR errors, removing high-WER utterances from
   training would not change much. **Unresolved — check before interpreting any
   existing result.**

2. **Single-pass preprocessing.** Voxtral is currently loaded four times for a
   full run (once per split under vLLM, plus once under HF for embeddings), and
   the HF pass recomputes an encoder forward that vLLM already ran and discarded.
   A forward hook on vLLM's `whisper_encoder` can capture it during
   transcription. Prototype at `src/preprocessing/acoustic_hook.py`; the smoke
   test failed on `apply_model` serialisation (needs
   `VLLM_ALLOW_INSECURE_SERIALIZATION=1`) and has not been re-run. **Not
   integrated.**

3. **Which encoder layer.** ADR-001 takes the final encoder layer. Mid-stack
   layers may carry more paralinguistic information. Cheap to ablate once
   sequences are cached.
