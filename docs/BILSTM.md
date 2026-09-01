# bc-LSTM — the dialogue context model

What it is, how it is wired into text and acoustics, what the windowing does,
and what it actually bought us. Companion to `FUSION.md`; the raw experimental
log is in `EXPERIMENTS.md` §5–§7 and the headline numbers in `RESULTS.md`.

**Every number here is test weighted F1 on MELD's full 2,610-utterance test
split**, scored through `src/evaluation/evaluate_all.py` or the per-run
`test_results_*.json`. Dev numbers are labelled where they appear.

---

## 1. Why a context model at all

MELD is conversation data. The same words carry different emotion depending on
what came before:

> A: "I got the promotion."
> B: **"Oh, great."**   ← joy
>
> A: "I crashed your car."
> B: **"Oh, great."**   ← anger

A model that sees only the utterance cannot separate those two, because the
input is byte-identical. Whatever ceiling a per-utterance classifier has, this
is part of it.

bc-LSTM (Poria et al., 2019, the paper that introduced MELD) is the standard
answer: run a bidirectional LSTM over the *sequence of utterances* in a
dialogue, so each utterance's representation is informed by its neighbours on
both sides, then classify each position. It is the reference architecture for
this dataset, which is why it is the context model here — using it makes our
numbers comparable to the published line rather than to a bespoke design.

---

## 2. Architecture

`src/models/context_lstm.py`. It consumes **pre-cached per-utterance vectors**,
so no large model is loaded and a full run takes minutes.

```
utterance features        (B, T, input_dim)     one vector per utterance, padded
        │  pack_padded_sequence(lengths)        padding never enters the LSTM
   BiLSTM (num_layers, bidirectional)           each utterance sees its neighbours
        │  (B, T, 2 * hidden_dim)
   Dropout
        │
   sentiment_head (B, T, 3)    emotion_head (B, T, 7)
```

Two details that matter:

**Packing.** `pack_padded_sequence` with real `lengths` means padded positions
are never fed to the LSTM. Without it, a short dialogue in a batch with a long
one would have its final states contaminated by padding — and because the LSTM
is *bidirectional*, the backward pass would start from padding and corrupt
every position, not just the tail.

**Per-position heads.** The heads are applied at every timestep, so one forward
pass over a dialogue of T utterances produces T predictions. Loss is masked to
real positions via `PAD_LABEL`.

`represent()` exposes the post-dropout `(B, T, 2*hidden_dim)` state — the exact
vector the heads consume — so the representation analysis can measure what
context does to class separability rather than inferring it from accuracy.

---

## 3. How text and acoustics enter

The BiLSTM is modality-agnostic: it sees one vector per utterance. How text and
audio get combined into that vector is the design axis, and we tested three
orderings.

### 3a. Concatenate, then contextualise (the default bc-LSTM)

```
XLM-R [CLS]        (768)  ─┐
                           ├─ concat → (2048) → BiLSTM → heads
Voxtral pooled    (1280)  ─┘
```

`input_dim = 768 + 1280 = 2048`. The two modalities are joined by raw
concatenation before the LSTM ever runs, so the LSTM does the fusion *and* the
contextualisation in one set of weights. This is the classic formulation.

### 3b. Fuse first, then contextualise (`stacked` / `stackedcat`)

```
text + acoustic → SequenceFusion → 512-d fused → BiLSTM → heads
```

The trained fusion model (`fusion_seq.py`) produces a learned 512-d joint
representation via `represent()`; the BiLSTM then contextualises *that*.
`stacked` replaces the input with the fused vector; `stackedcat` concatenates
the fused vector onto the originals. Cached by
`src/preprocessing/extract_fused_features.py`.

### 3c. Contextualise first, then fuse (`ContextThenFusion`)

```
text     (768)  → BiLSTM_text     ─┐
                                    ├─ fuse → heads
acoustic (1280) → BiLSTM_acoustic ─┘
```

`src/models/context_fusion.py`. Each modality gets its own BiLSTM, so context
is applied *within* a modality before the modalities meet. The intuition was
that text context and acoustic context are different phenomena — lexical
coherence versus prosodic carry-over — and forcing them through one LSTM makes
the model learn both in shared weights.

`use_text_lstm` / `use_acoustic_lstm` give a 2×2 ablation. **These arms are not
capacity-matched** (6.3 M / 4.3 M / 3.6 M / 1.6 M parameters), so an arm winning
may be winning on capacity; this is recorded in the model docstring.

### Which ordering won

| ordering | gold | asr | asr_cleaned |
|---|---|---|---|
| **concat → context** (`attn`) | 0.6259 | **0.5145** | **0.5334** |
| fuse → context (`stacked`) | 0.6197 | 0.5061 | 0.5030 |
| fuse → context (`stackedcat`) | 0.6172 | 0.5000 | 0.5033 |
| context → fuse (`ctxfusion`) | 0.6258 | 0.5032 | 0.4927 |

All at K=0 except ctxfusion (best cell). **The orderings are equivalent within
noise on gold** (0.617–0.626) and the plain concatenation is at least as good
as anything more elaborate on ASR. The extra machinery did not pay.

---

## 4. Windowing

### What the window is

A MELD dialogue is 1–33 utterances. `WindowedDialogueDataset`
(`src/training/train_context.py`) replaces each dialogue with one training
example *per utterance*, containing a ±K neighbourhood:

```python
start = max(0, i - context_window)
end   = min(T, i + context_window + 1)
```

All labels in the window are set to `PAD_LABEL` except the centre, so loss and
metrics score only the centre utterance while the BiLSTM still sees its
neighbours as context.

### Why window at all, rather than use whole dialogues

Three reasons:

1. **A fixed receptive field.** With whole dialogues, an utterance in a 33-turn
   conversation gets far more context than one in a 2-turn conversation. K makes
   the amount of context a controlled variable instead of an artefact of where
   the utterance happens to sit.
2. **It matches deployment.** A live system has the preceding turns but not the
   following ones, and not an unbounded history. Bounded K is the realistic
   setting; `full` is the optimistic one.
3. **It is the actual research question.** "How much context does MELD emotion
   need?" is answerable by sweeping K, and not by any single architecture.

### K = 0 is not "no model", it is "no neighbours"

This distinction caused a real misreading and is worth stating plainly. At K=0
the window is `[i, i+1)` — **a sequence of length 1**. The BiLSTM still runs,
still has all its parameters, and still applies its input-to-hidden transform.
What it does *not* have is any neighbour to look at.

So K=0 isolates the bc-LSTM **layer** from the bc-LSTM **context**. That turns
out to be the whole story.

### The sweep

Test weighted F1, `attn` family (attention-pooled acoustics):

| condition | K=0 | K=1 | K=2 | K=4 | full |
|---|---|---|---|---|---|
| gold | 0.6259 | **0.6283** | 0.6162 | 0.6182 | 0.6252 |
| asr | **0.5145** | 0.5055 | 0.5140 | 0.5119 | 0.5138 |
| asr_cleaned | **0.5334** | 0.5259 | 0.5326 | 0.5317 | 0.5243 |

And across every other bc-LSTM variant (`attnraw`, `stacked`, `stackedcat`),
the spread from K=0 to full stays under 0.01 in all three conditions.

**This is a null result on dialogue context.** K=0 — no neighbours at all — is
the best cell in two of three conditions. The full dialogue never wins.

### Reconciling it with "context helps"

`RESULTS.md` §4 reports a gain of +0.032 to +0.050 for "+ dialogue context"
over "fusion alone". That is **not** in tension with the null above, but the
label is wrong. Decomposed on test:

| condition | fusion alone (no BiLSTM) | + BiLSTM, K=0 (no neighbours) | + neighbours (best K) |
|---|---|---|---|
| gold | 0.5957 | 0.6259 (**+0.030**) | 0.6283 (+0.002) |
| asr | 0.4952 | 0.5145 (**+0.019**) | 0.5145 (+0.000) |
| asr_cleaned | 0.4754 | 0.5334 (**+0.058**) | 0.5334 (+0.000) |

**The gain is the layer, not the context.** Adding the BiLSTM's parameters is
worth +0.019 to +0.058; adding neighbours on top of it is worth ~0.000. The
honest claim is *"a recurrent layer over the utterance representation helps;
the dialogue context inside it does not"* — which is a more interesting finding
than the one it replaces, and a considerably more awkward one for the
architecture's stated purpose.

### Why context might be failing here

Stated as hypotheses, not conclusions — we did not test these:

- **The features are frozen.** XLM-R and Voxtral are cached, so the LSTM
  contextualises vectors that were never trained to be contextualised.
- **Speaker identity is absent.** MELD is multi-party. The model sees an
  undifferentiated utterance sequence with no notion of who is speaking, so
  "the previous turn" may be a different speaker mid-argument or the same
  speaker continuing. COSMIC's gains come substantially from modelling exactly
  this.
- **Class imbalance dominates.** At 47% neutral, a model can do well by
  tracking frequency, and context helps least on the majority class.

Independent support: Zhang & Poellabauer (NeurIPS 2024 workshop, `RESULTS.md`
§2e) find that even an LLM reading context as natural language gains only
~0.02 UA on MELD, saturating by 10 utterances. Context is not where the
headroom is on this corpus.

---

## 5. Hidden size

`hidden` family, one layer, whole dialogues:

| hidden | gold | asr | asr_cleaned |
|---|---|---|---|
| 128 | 0.6161 | 0.4977 | 0.5099 |
| 256 | 0.6220 | 0.4945 | 0.5075 |
| 512 | 0.6234 | 0.4952 | **0.5105** |

A 4× capacity increase moves gold by 0.007 and ASR by nothing. **Capacity is
not the binding constraint.** 256 is the default and there is no evidence for
paying more.

Note the BiLSTM compresses 2048 → 2×256 = 512. That is a 4× reduction, and the
h512 row (2048 → 1024) shows the compression is not what is costing us.

---

## 6. What actually mattered

Ranked by measured effect on the ASR condition, the thing the thesis is about:

| change | effect |
|---|---|
| gold → ASR transcripts | **−0.114** |
| acoustic pooling: masked-mean → attention | **+0.027** |
| adding the BiLSTM layer at all | **+0.019 to +0.058** |
| WER-based cleaning of training data | +0.019 |
| fusion ordering (3 variants) | ≤0.010 |
| **dialogue context width (K=0 → full)** | **≤0.010, sign varies** |
| hidden size (128 → 512) | ≤0.007 |

The pooling comparison is robust despite the single seed: 43 of 45 cells
improved, which is a sign test at **p ≈ 5.9 × 10⁻¹¹**. The window and hidden
sweeps are the opposite — differences well inside what one seed can produce.

---

## 7. Reproducing

```bash
# per-utterance ±K windows, attention-pooled acoustics
sbatch src/scripts/attnpool_context_grid.sbatch    # 45 runs
sbatch src/scripts/bclstm_grid.sbatch              # base grid
sbatch src/scripts/hidden_sweep.sbatch             # 18 runs
sbatch src/scripts/ctxfusion_window_grid.sbatch    # context-then-fusion

# score every checkpoint through one code path
sbatch src/scripts/evaluate_all.sbatch
```

Checkpoints under `/dcs/large/u5734759/checkpoints/bclstm/<variant>_<condition>/<window>/`.

---

## 8. Caveats

- **Single seed per cell.** Differences below ~0.02 are not distinguishable
  from seed noise, which covers the entire window and hidden sweeps. The
  pooling result survives only because of the 43/45 sign test.
- **Dev used twice** — early stopping and run selection — so dev is mildly
  optimistic. Test is the honest number.
- **`ctxfusion` arms are not capacity-matched** (see §3c).
- **No speaker modelling**, which is the most likely reason context
  underperforms here and the most obvious next experiment.
