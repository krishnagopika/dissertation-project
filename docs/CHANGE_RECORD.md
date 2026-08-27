# Change Record

Dated implementation notes: what changed in the code, why, and what evidence
justified it.

Distinct from the other two docs in this folder:

| Doc | Records | Lifecycle |
|---|---|---|
| `DECISIONS.md` | *Decisions* — why the design is the way it is | Durable; superseded, never edited |
| `CHANGE_RECORD.md` | *Changes* — what was modified and when | Append-only, dated |
| `POSTMORTEMS.md` | *Failures* — what went wrong and what would have caught it | Append-only, dated |

A change may implement a decision, fix a bug, or refactor — it does not need a
corresponding ADR. Conversely an ADR may sit unimplemented for a long time with
no change record. Where a change relates to a decision or an incident, link it.

**Entry format:** `CR-NNN — one-line summary`, then date, affected files, and
sections for *What changed*, *Why*, *Evidence*, *Consequences*.

---

## CR-001 — Renormalise attention weights after dropout

**Date:** 2026-08-23
**Type:** bug fix
**Related:** `DECISIONS.md` ADR-003, ADR-004 · `POSTMORTEMS.md` PM-007
**File:** `src/models/pooling.py::AttentionPooling._drop_and_renormalise`
**Raised by:** code review — flagged as "test rather than blindly assume"

### What changed

Both `AttentionPooling` and `AttentiveStatsPooling` previously fed
`self.dropout(weights)` straight into the pooling sum. They now route it through
`_drop_and_renormalise`, which divides by the post-dropout sum so the weights
again sum to exactly one.

### Why

`nn.Dropout` is *inverted* dropout: it zeroes with probability `p` and scales
survivors by `1/(1-p)`, so `E[Σw] = 1`. That holds **in expectation only** — for
any individual sample `Σw = s` is a random variable around 1. Both statistics
assume a genuine distribution:

```
mean = Σ w_t x_t              requires Σ w_t == 1
var  = Σ w_t x_t² − mean²     requires it exactly
```

With `Σ w_t = s`, the variance identity evaluates to `s·E'[x²] − s²·E'[x]²`
rather than `E'[x²] − E'[x]²`. When `s > 1` and frames are near-constant — a
steady vowel, a silent region, both common in real audio — this goes **negative
for genuinely positive variance**, and the `clamp_min(_VAR_EPS)` guard then pins
`std` to `sqrt(1e-5) = 0.00316`, a constant carrying no signal. The clamp was
written as a float-precision guard and was silently absorbing a structural error.

The mean term was affected too, though less severely: without renormalisation the
output scale differs per-sample between train and eval, since eval has dropout
off and weights already summing to one.

### Evidence

Measured at `p=0.3`, `T=40`, 256 rows:

| Condition | Before | After |
|---|---|---|
| `Σw` range after dropout | 0.673 – 1.263 | 1.000000 – 1.000000 |
| Rows with `Σw` off by >0.1 | 40% | 0 |
| Median relative error in `var` (σ=3 frames) | 5.8% | 0 |
| Negative variance elements (near-constant frames) | **988 / 2048 (48.2%)** | **0 / 2048** |
| `std` on near-constant frames | pinned at 0.00316 | 0.0052 – 0.0158 |

### Consequences

- Renormalising cancels inverted dropout's `1/(1-p)` scaling entirely, making the
  operation exactly *"attend over a random subset of frames"* — a coherent
  regulariser, and one whose output scale now matches eval mode.
- Eval mode is bit-identical to before: with dropout off, renormalisation is a
  verified no-op.
- Added a fallback for the pathological case where dropout zeroes every frame of
  a short sequence. Dividing by ~0 would emit inf/NaN; returning zeros would emit
  an all-zero embedding indistinguishable from a padded clip. It now falls back
  to the undropped weights. Verified finite at `p=1.0`.
- A clamp firing in `AttentiveStatsPooling` now genuinely indicates fp16/bf16
  precision loss, which is what the guard was for.

### Alternatives considered

- **Use undropped weights for the statistics.** Simpler and mathematically clean,
  but discards the regularisation entirely.
- **Drop the attention dropout.** Defensible given `dropout=0.0` is the default,
  but leaves a latent trap for anyone who sets it later.
- **Drop out the scores *pre-softmax*** instead of the weights post-softmax.
  Equivalent in effect and arguably cleaner to read, since the renormalisation
  then happens implicitly inside the softmax.

#### Note on why renormalisation is exact, not approximate

`dropped / dropped.sum()` on a softmax output is *identical* to having masked
those positions before the softmax:

```
softmax over surviving subset S = exp(e_t) / Σ_S exp(e) = a_t / Σ_S a
```

The inverted-dropout `1/(1−p)` factor cancels in the ratio, so the result does
not depend on `p`'s scaling at all. The operation is therefore precisely
*"re-run attention treating the dropped frames as padding"* — the same semantics
as the masking path the module already relies on, rather than a separate
approximation of it. This is a stronger justification than "attend over a random
subset" and is now recorded in the method docstring.

Renormalising was chosen because it keeps the regularisation *and* the
statistical interpretation, rather than trading one for the other.


---

## CR-002 — Fix threshold mismatch in the dropout fallback; add fixed-query pooler

**Date:** 2026-08-23
**Type:** bug fix + new component
**Related:** `CHANGE_RECORD.md` CR-001 · `DECISIONS.md` ADR-003 · `POSTMORTEMS.md` PM-007
**File:** `src/models/pooling.py`
**Raised by:** code review of CR-001

### 1. Threshold mismatch in `_drop_and_renormalise` (bug)

**What changed.** The guard condition and the divisor floor were different
constants:

```python
torch.where(total > 0, dropped / total.clamp_min(_VAR_EPS), weights)   # before
```

For any row with `0 < total < 1e-5`, the condition selected the *division*
branch while `clamp_min` silently changed the divisor — so the result summed to
`total/1e-5`, not to 1. That is the exact invariant the function exists to
guarantee, violated in precisely the regime it was written to protect. Reachable
whenever dropout leaves only far-tail frames of a peaked distribution — rare, but
it lands on near-degenerate rows, which are the ones already prone to the
negative-variance path from CR-001.

Both now use a single dedicated constant `_WEIGHT_SUM_EPS = 1e-6`.

**Why a separate constant.** `_VAR_EPS` is documented as a pre-`sqrt` variance
floor; reusing it as a weight-sum floor conflates two unrelated numerical guards,
so tuning one would silently move the other's boundary. `1e-5` is also subnormal
in fp16 (smallest normal ≈ 6.1e-5), which would make the reuse fragile in a
half-precision head.

**`clamp_min` is still required** even though the condition now excludes the
small case: `torch.where` evaluates *both* branches, so an unclamped division
would produce `inf` in the discarded branch and poison the gradient despite never
being selected.

| Input `total` | Before | After |
|---|---|---|
| 1e-9 | sum = 1e-4 | sum = 1e-9 (falls back to undropped) |
| 1e-7 | sum = 1e-2 | sum = 1e-7 (falls back to undropped) |
| 0.5 | sum = 1.0 | sum = 1.0 |

### 2. Fully-masked rows made consistent (bug)

A fully-masked row has every score at `finfo.min`, so softmax returned a
**uniform** distribution over all `T` frames — i.e. it pooled pure padding.
`_attend` now zeroes such rows, and `AttentiveStatsPooling` zeroes its output
row too (otherwise `clamp_min` left `std` at a nonzero constant `sqrt(1e-5) =
0.00316` while `mean` was 0).

All three poolers now return exactly `0` for a fully-masked row. Callers should
still assert `mask.any(dim=1).all()` at the collator so it never arises.

| Pooler | Before | After |
|---|---|---|
| `masked_mean` | 0.0 | 0.0 |
| `attention` | uniform over padding | 0.0 |
| `attentive_stats` | std pinned at 0.00316 | 0.0 |

### 3. `attention_fixed` — capacity-matched control (new)

**Why.** `masked_mean → attention` differs in *two* variables at once: weighting
scheme **and** trainable capacity (0 → 164,096 parameters). The delta therefore
cannot separate "learned frame weighting helps" from "extra capacity helps".

`AttentionPooling` gained a `trainable: bool` flag; `attention_fixed` builds it
frozen at random init. Identical architecture, identical forward cost, **zero**
trainable parameters.

**Verified it is a real rung, not a disguised mean.** On realistic varied frames
its weights are genuinely non-uniform — max/min ratio 3.2×, KL from uniform
0.038 at `attention_dim=16`. Gradients do not reach it, but flow normally to the
downstream head.

Ablation ladder at `input_dim=1280`:

| Pooler | Trainable | out_dim | Isolates |
|---|---|---|---|
| unmasked mean | 0 | 1280 | historical baseline |
| `masked_mean` | 0 | 1280 | the padding defect alone |
| `attention_fixed` | 0 | 1280 | non-uniform but *uninformative* weighting |
| `attention` | 164,096 | 1280 | **learned** weighting |
| `attentive_stats` | 164,096 | 2560 | + prosodic variance |

`attention_fixed → attention` is now the clean test of whether *learning* the
weights helps, with architecture held constant.

### Documentation

`AttentionPooling.forward` and `AttentiveStatsPooling.forward` now state in
`Returns:` that the returned weights are the **undropped** distribution — in
train mode the vector is pooled with a dropped-and-renormalised copy, so the
pooled output cannot be reconstructed from them. Plot from eval mode, where the
two coincide.


---

## CR-003 — Single-pass preprocessing: one Voxtral load, both artefacts

**Date:** 2026-08-23
**Type:** architecture change
**Related:** `DECISIONS.md` ADR-001/002/003 · `POSTMORTEMS.md` PM-008
**Files:** `src/preprocessing/transcribe_all.py`,
`src/preprocessing/acoustic_hook.py`,
`src/scripts/transcribe_single_pass.sbatch`
**Verified:** job 8082 (24 MELD dev clips, throwaway output dir)

### What changed

A full run loaded Voxtral **four times**: once per split under vllm, plus a
fourth under HF transformers to recompute an encoder forward vllm had already
performed and discarded. Now **once**.

- `build_llm()` constructs the engine once and shares it across splits.
- `transcribe_and_extract_split()` transcribes and captures acoustics in the
  same forward, via a hook on vllm's `whisper_encoder`.
- `--legacy_two_pass` preserves the old path for reproducing existing
  `{split}_embeddings.pt` files.
- Fails fast if `VLLM_ALLOW_INSECURE_SERIALIZATION=1` is absent, rather than
  dying at the first `apply_model` deep into a long run.

### Why the hook point works

`VoxtralForConditionalGeneration.embed_multimodal` calls
`self.whisper_encoder(audio_inputs)`, whose signature is
`forward(list[Tensor]) -> list[Tensor]` — one waveform in, one `(T, 1280)`
encoder state out, same order. That is the same quantity HF's
`audio_tower(...).last_hidden_state` provides, i.e. exactly the representation
ADR-001 selected. vllm then projects it through the adapter and drops it.

### The hard part: matching captures to utterance keys

The hook sees a batch with no request ids — vllm's scheduler decides batch
composition, and the `mm_hash` identifying each item lives in the model *runner*,
not the model. So captures are keyed by a content hash of the audio.

The first attempt hashed the float32 waveform and matched **0 of 24** clips.
Diagnostic job 8081 found two transformations sitting between what we send and
what the encoder receives:

| Assumption | Measured reality |
|---|---|
| hook sees the waveform we sent | vllm **zero-pads to exactly 30 s** — 480,000 samples, every clip |
| waveform is float32 | it is **bfloat16**, cast to model dtype |

The dtype was fatal: bf16 → float32 does not recover the original bits, so a
float32 hash can never match. `canonical_waveform()` now applies the same
reduction on both sides — cast to bf16, strip trailing zeros, hash. The 30 s
padding is just more trailing zeros, so one rule covers both.

Trailing-zero stripping is safe even for a clip that genuinely ends in digital
silence (`dia1_utt0` ends in ~193 zero samples): the driver's copy carries those
zeros too, so both sides strip identically and still agree.

### Evidence

| Check | Before fix | After fix |
|---|---|---|
| Clips matched | 0 / 24 | **24 / 24** |
| Frames per clip | 1500 (constant) | 41–550, median 134 |
| Median duration implied | 30.0 s (impossible) | **2.7 s** (correct for MELD) |
| Capture size | 92 MB / 24 clips | 10.7 MB / 24 clips |
| Projected full corpus | ~52 GB | **6.1 GB** |
| Model loads per full run | 4 | **1** |

`masked_mean` over stored frames equals the stored pooled vector exactly.

### Consequences

- Padding is removed **at capture**, so the cache holds only real frames and the
  ADR-003 padding defect cannot reach downstream code.
- `{split}_transcripts.json` is never overwritten — keep-lists depend on it.
- The masked-mean vector is written under a distinct filename so the legacy
  `{split}_embeddings.pt` stays byte-identical and old results remain
  reproducible.
- Unmatched clips are zero-filled, logged at ERROR, **and** listed in
  `{split}_unmatched.json`. The hook banks exceptions rather than raising (a
  raise inside a forward hook aborts the engine step) and the pipeline drains
  and logs them each batch — PM-002's lesson applied.

### Not done

Downstream consumers still read the legacy fixed-size cache. `fusion.py`,
`train_fusion.py` and `train_context.py` must be migrated to sequences before
the poolers in `src/models/pooling.py` can actually run.
