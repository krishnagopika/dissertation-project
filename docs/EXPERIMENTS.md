# Experiments

Every experiment run, in pipeline order, with what it measured and what it
found. Methodology for each component lives in its own document
(`VOXTRAL.md`, `XLM_ROBERTA.md`, `FUSION.md`, `MELD_ANALYSIS.md`); this file is
the record of runs and results.

**Metric throughout: weighted F1 on the 7-class emotion task**, with sentiment
reported alongside. Weighted rather than accuracy because MELD is 47% neutral.

**Read the caveats in §9 before quoting any number from here.**

---

## 1. The Mini line — what the pipeline is

```
MELD audio (.mp4, 13,708 utterances)
     │
     ▼  Voxtral-Mini-3B, ONE load, forward hook on the Whisper encoder
     ├──────────────► transcripts        {split}_transcripts.json
     └──────────────► acoustic (T,1280)  {split}_acoustic_seq.pt
                                          {split}_embeddings_maskedmean.pt
     │
     ▼  XLM-RoBERTa-base fine-tuned on the transcripts
            text [CLS] 768-d              {split}_text_embeddings.pt
     │
     ▼  downstream models, all on frozen caches
        fusion · bc-LSTM · stacked · context-then-fusion
```

Splits: **9,989 train / 1,109 dev / 2,610 test**.

Voxtral is never trained. Both artefacts come from a single load — the encoder
output vLLM computes for audio tokens is captured on its way past rather than
recomputed under a second framework.

---

## 2. ASR quality — the WER baseline

Voxtral-Mini transcripts scored against MELD's gold text.

| split | corpus WER | corpus CER | mean utterance WER | n |
|---|---|---|---|---|
| train | 0.3461 | 0.2957 | 0.6785 | 9,988 |
| dev | 0.3124 | 0.2498 | 0.5898 | 1,109 |
| test | **0.3823** | 0.3357 | 0.9093 | 2,610 |

Two things this establishes:

**Corpus WER ≪ mean per-utterance WER** (0.38 vs 0.68 on train) because short
utterances produce enormous per-utterance ratios — a one-word reference against
a runaway hallucination scores WER 169. The corpus figure is the honest
aggregate; the mean is dominated by a tail.

**Test is the hardest split** (0.3823 vs 0.3124 dev), which matters when reading
every test number below: the ASR conditions are working from worse transcripts
at test than the dev numbers implied.

The `text_clean` keep-list derives from this analysis: **6,729 / 752 / 1,787**
utterances retained. An earlier `wer25` list kept only 3,539 train rows, but was
computed before the punctuation fix (`KL-5` → `kl5` scored WER 2.0 against ASR
`KL 5`), before cp1252 mojibake repair, and at threshold 0.25 rather than 0.50.
Roughly half its exclusions were a scoring artefact.

---

## 3. XLM-RoBERTa — text-only, 9 runs

3 text conditions × 3 losses. Encoder fine-tuned; this is the only stage where
XLM-R is trainable.

**Dev weighted F1 (emotion):**

| condition | plain | weighted | focal |
|---|---|---|---|
| gold | **0.5733** | 0.5565 | 0.5617 |
| asr | **0.4594** | 0.4409 | 0.4445 |
| asr_cleaned | **0.4769** | 0.4568 | 0.4366 |

- `plain` wins every condition — class weighting and focal loss both hurt.
- **`asr_cleaned` beats `asr` on all three losses**, so WER filtering earns its
  place: training on cleaner transcripts helps even though the test set is not
  filtered.
- The gold→asr gap is **0.114**.

**Cross-evaluation matrix: 27/27 cells** — every trained model scored on every
test condition (gold / asr / asr_clean), with the test subset chosen by CLI so
a model trained on filtered data can still be scored on the full 2,610.

Test weighted F1 (emotion), `plain` loss:

| trained on | test=gold | test=asr | test=asr_clean |
|---|---|---|---|
| gold | 0.6070 | 0.4634 | **0.5074** |
| asr | — | 0.4594 | 0.5047 |
| **asr_cleaned** | — | **0.4859** | 0.4923 |

n = 2,610 for gold/asr; **1,787** for asr_clean.

**Read DOWN a column, never across a row.** The asr_clean column is a different
and easier subset -- it excludes precisely the utterances whose transcripts are
worst -- so every model scores ~0.04-0.05 higher there. A cross-row comparison
would credit the filter for the exam being easier.

Read correctly, two things follow.

**Training on cleaned data helps** on the full ASR test set: 0.4859 against
0.4594 for asr-trained and 0.4634 for gold-trained, a gain of +0.027. WER
filtering earns its place at training time.

**But the cleaned-trained model LOSES on the cleaned test set** (0.4923 vs the
gold-trained model's 0.5074). Training on filtered data buys robustness to
noise; it does not buy accuracy on clean input, where a gold-trained model
transfers better.

Reporting convention: the **full** test set is the headline, because deployment
cannot filter what has not been transcribed yet. The asr_clean column is
reported alongside as the "with an inference-time quality filter"
configuration, which is a legitimate deployed system and a fair question to
ask of these models.

The three `plain` checkpoints supply the frozen text encoder for everything
downstream: each condition uses **its own** encoder, so the loss is held
constant across conditions and the encoder choice is not a confound.

---

## 4. Fusion — 25 runs, 4 staged phases

Per-utterance. Learns its acoustic pooling and its modality combination.
Full detail in `FUSION.md`.

| phase | runs | varies | best (dev) |
|---|---|---|---|
| 0 — learning rate × batch | 6 | 3 LR × 2 batch | lr 1e-3, bs 32 |
| 1 — pooling | 4 | 4 poolers, acoustic-only | `attention` 0.5044 |
| 2 — text-only baselines | 3 | 3 conditions | gold **0.5620** |
| 3 — fusion | 12 | 4 mechanisms × 3 conditions | gold_sum **0.5652** |

Staged rather than crossed: a full cross is 288 runs. Phase 0's winner is fixed
for all 19 later runs so no comparison can be explained by a luckier learning
rate.

### Fusion's value is CONDITIONAL on transcription quality

Comparing fusion against its own unimodal baselines -- same grid, same head,
same data, so the only difference is which modalities reach the classifier.
Test weighted F1 (emotion):

| condition | text-only | acoustic-only | fusion | **fusion − text** |
|---|---|---|---|---|
| gold | 0.5968 | — | 0.5957 | **−0.001** |
| **asr** | 0.4350 | 0.4893 | **0.4952** | **+0.060** |
| asr_cleaned | 0.4533 | — | 0.4754 | **+0.022** |

**Fusion is worthless on clean text and worth +0.060 on ASR.** Reading only the
gold row -- as an earlier version of this document did -- gives exactly the
wrong conclusion.

The mechanism is confirmed independently by the classical baselines (§8b):

- text degrades **0.5968 → 0.4350** (−0.162) when transcripts become noisy;
- **acoustic is unaffected** -- 0.4986 on both conditions, because the acoustic
  cache is byte-identical across text conditions (same audio, same encoder);
- so on ASR the acoustic branch (0.4893) actually **beats** text (0.4350).

Fusion has nothing to add when the text is reliable, and becomes the difference
when it is not. **That is the dialect/accent robustness argument in its most
direct form**: accented or noisy speech produces worse transcripts, and the
acoustic channel is what carries the model through.

One qualification: on ASR the acoustic-only model (0.4893) nearly matches full
fusion (0.4952), so most of fusion's gain there is **having the audio channel at
all**, not the sophistication of the combination mechanism. The four fusion
mechanisms differ from each other by far less than the presence of audio does.

---

## 5. bc-LSTM — dialogue context, 15 cells

Whole dialogues; a BiLSTM contextualises each utterance with its neighbours.
Input is the raw 768 + 1280 concatenation.

**Dev weighted F1, 3 conditions × 5 context widths:**

| | k0 | k1 | k2 | k4 | full |
|---|---|---|---|---|---|
| gold | 0.5853 | 0.5925 | 0.5954 | **0.6109** | 0.5931 |
| asr | 0.5018 | 0.4984 | 0.5044 | 0.5066 | 0.4975 |
| asr_cleaned | 0.5003 | 0.4982 | 0.4983 | 0.5041 | 0.5054 |

**Three findings:**

1. **Context helps far more than fusion** — gold k4 at 0.6109 against fusion's
   0.5652.
2. **But only on clean text.** Gold gains **+0.026** from k0→k4; asr gains
   +0.005 and asr_cleaned +0.004, both within noise.
3. **A bounded window beats the full dialogue** on gold (0.6109 at k4 vs 0.5931
   full) — distant utterances add noise rather than signal.

Note K=0 is not a context-free control in the architectural sense: the BiLSTM
keeps its own parameters over a length-1 sequence. It controls for context
*width*.

---

## 6. Stacked — fusion → bc-LSTM, 30 cells

Feeds fusion's learned 512-d representation into the context model, so that
context is the only difference from plain fusion.

Two variants, 15 cells each:

- **`replace`** — the BiLSTM sees only the fused 512-d vector.
- **`concat`** — it sees `fused ⊕ text ⊕ acoustic` (2560-d), so **no
  information can be lost**.

**Best per condition (dev):**

| condition | plain | stacked `replace` | stacked `concat` |
|---|---|---|---|
| gold | **0.6109** | 0.5922 | 0.5988 |
| asr | 0.5066 | **0.5171** | 0.5125 |
| asr_cleaned | 0.5054 | 0.4962 | **0.5081** |

**Finding: stacking does not work.** `concat` is the decisive test — it keeps
the raw features *and* adds the fused vector, with more parameters than plain —
and still loses on gold. So the fusion representation carries essentially
nothing the context model cannot already recover from the raw features.

An initial reading blamed a 512-d bottleneck. `concat` removes that explanation.

---

## 7. Context-then-fusion — reversing the order, 12 cells (+36 windowed)

Per-modality BiLSTM **first**, fusion **second**, so compression happens last.
Independent switches give a 2×2 ablation.

| arm | acoustic BiLSTM | text BiLSTM | params |
|---|---|---|---|
| both | ✓ | ✓ | 6,309,386 |
| acoustic_only | ✓ | — | 4,339,210 |
| text_only | — | ✓ | 3,552,778 |
| neither (control) | — | — | 1,582,602 |

**Dev** suggested acoustic context was the robustness story: `acoustic_only`
won both ASR conditions and helped most where transcription was worst
(asr **+0.0156** over control), with *fewer* parameters than `both`.

**Test did not replicate it.**

| test, arm − control | acoustic_only | text_only | both |
|---|---|---|---|
| gold | −0.0024 | +0.0066 | +0.0055 |
| asr | **−0.0114** | −0.0089 | −0.0035 |
| asr_cleaned | +0.0000 | −0.0075 | +0.0026 |

`asr/acoustic_only` went from **best on dev to worst in its row on test**. Every
test delta sits within ±0.012. **This architecture shows no reliable benefit**,
and the dev ordering was not real.

A window sweep (K=0,1,2,4 × 3 arms × 3 conditions = 36 runs) is in progress to
test whether prosodic and lexical context have different optimal widths.

---

## 8. Voxtral-Small

Small is a second acoustic front-end, not a scaled Mini: **0 of 6 sampled
encoder tensors match Mini's**, because Mistral fine-tuned each encoder jointly
with its own decoder (paper Table 1; §3.1 — the encoder is frozen only for the
warm-up pass). Its acoustics are a genuinely different representation.

**On-prem it does not run.** 24.3 B params ≈ 48.5 GB in bf16 against a 46 GB
L40S forces `tensor_parallel_size=2`, and every `tp=2` generation deadlocks.
Eleven jobs eliminated: the acoustic hook (4 variants including a no-op), audio
itself (text-only hangs), fork vs spawn, async scheduling, GPUDirect RDMA,
`NCCL_SHM_DISABLE`, `/dev/shm` capacity, and a corrupt download (SHA-256
verified). NCCL *init* succeeds — "Connected all rings/trees" — then the first
collective spin-waits. That signature matches NVIDIA/nccl#2079: PCIe ACS/IOMMU
on two Ada GPUs without NVLink. A host-level fix, not a configuration one.

**On Modal it works.** One A100-80GB, `tensor_parallel_size=1`, no collectives.
Model loads in ~103 s; measured **0.70 s per clip**, so all 13,708 project to
~2.7 h ≈ $6.70.

| split | status |
|---|---|
| dev | **done** — 1,109 sequences, 0 unmatched, 1280-d, median 92 frames |
| test | failed — macOS `._*.mp4` resource-fork files reach ffmpeg |
| train | not completed — same cause |

The fix is to exclude `._` prefixed files at clip discovery.

### The WER gate: does Small transcribe better than Mini?

Same normalisation, `repair_encoding` on both sides, identical clip set:

| split | n | Mini WER | Small WER | delta | Mini CER | Small CER |
|---|---|---|---|---|---|---|
| dev | 1,108 | 0.3120 | 0.3294 | **+0.017 worse** | 0.2587 | 0.2930 |
| test | 2,610 | 0.3823 | **0.3619** | **−0.020 better** | 0.3300 | 0.3173 |

**The effect flips sign between splits.** Small is worse on dev and better on
test, by about the same margin in each direction. An 8x larger model therefore
gives **no consistent transcription improvement on MELD**.

That is a decision, not just a number. Transcription quality is the only factor
that moved results anywhere in this project (~0.12 between gold and ASR), so a
front-end that does not reliably improve WER cannot be expected to improve
anything downstream. **Re-running the 127-model pipeline on Small transcripts
is not justified by this evidence.**

Two things keep Small worth having anyway:

**Its acoustics are a genuinely different representation.** 0 of 6 sampled
encoder tensors match Mini's, because Mistral fine-tuned each encoder jointly
with its own decoder. The acoustic branch is a real second data point even
though the transcripts are not.

**"A 24B model does not beat a 3B model at transcribing MELD" is itself a
finding.** MELD is short, overlapping, background-heavy television dialogue;
the errors look audio-limited rather than capacity-limited, which is consistent
with everything else measured here.

**Zero-shot baseline (Voxtral-Mini, no training at all), test:**

| task | accuracy | weighted F1 |
|---|---|---|
| emotion | 0.5487 | **0.5043** |
| sentiment | 0.5644 | **0.5515** |

Worth keeping in view: a frozen model with a prompt reaches 0.5043, which is
close to the trained fusion pipeline's ASR-condition performance.

---

## 8b. Classical baselines — do the neural models earn their complexity?

Three classical models on **exactly the cached vectors the neural models
consume**, so the comparison isolates the model and nothing else.

**Gold condition, test weighted F1 (emotion), macro in brackets:**

| features | dim | logreg | linear SVM | XGBoost |
|---|---|---|---|---|
| text | 768 | 0.5491 (0.364) | 0.5850 (0.387) | **0.5953 (0.399)** |
| acoustic | 1280 | 0.3959 (0.280) | 0.4262 (0.281) | 0.4986 (0.290) |

**Sentiment:**

| features | logreg | linear SVM | XGBoost |
|---|---|---|---|
| text | 0.6735 | 0.6812 | **0.6818** |
| acoustic | 0.5379 | 0.5578 | 0.6038 |

### This reframes the neural results

| model | gold emotion WF1 |
|---|---|
| **XGBoost on frozen text** | **0.5953** |
| linear SVM on frozen text | 0.5850 |
| XLM-R fine-tuned (best of 9) | 0.5733 |
| fusion (best of 25) | 0.5652 |
| logistic regression on frozen text | 0.5491 |

**Gradient boosting on frozen 768-d embeddings beats the entire neural
pipeline** -- fine-tuning, attention pooling, four fusion mechanisms, the lot --
in 64 seconds of CPU. Everything the fusion grid added over a text baseline sits
inside the spread between three off-the-shelf classifiers on identical features.

### Three consequences

**The structure is nonlinear.** XGBoost 0.5953 > SVM 0.5850 > logreg 0.5491 is a
0.046 spread on the same vectors, so there is nonlinear structure in the XLM-R
embeddings that linear probes miss entirely.

**Linear probe F1 is a LOWER BOUND on separability**, not a measurement of it.
The representation analysis (§10) uses logistic regression, which understates
linear separability by ~0.036 against SVM and misses nonlinear structure by
~0.046. Its numbers must be read as a floor.

**Text dominates acoustic at the utterance level** -- a gap of 0.153 (logreg) to
0.097 (XGBoost). That is why fusion gained so little: it was adding a much
weaker channel to a strong one. Note XGBoost closes a third of the gap, so a
meaningful part of the acoustic signal is nonlinear and invisible to a linear
probe.

**Macro F1 stays at 0.28-0.29 for acoustic across all three models.** The
minority emotions are not recoverable from pooled prosody by any classifier
tried. The weighted metric is carried by neutral.

### Naive concatenation actively HURTS

| features | dim | logreg | linear SVM |
|---|---|---|---|
| text | 768 | 0.5491 | **0.5850** |
| both | 2048 | 0.5282 | 0.5263 |
| **delta** | | −0.021 | **−0.059** |

Adding the acoustic channel by concatenation makes the linear models *worse*,
and worse by more for SVM than for logistic regression. Sentiment shows the
same pattern (SVM 0.6812 text vs 0.6432 both).

...but XGBoost GAINS from the same concatenation:

| features | logreg | SVM | XGBoost |
|---|---|---|---|
| text | 0.5491 | 0.5850 | 0.5953 |
| acoustic | 0.3959 | 0.4262 | 0.4986 |
| **both** | 0.5282 | 0.5263 | **0.6189** (macro **0.418**) |
| vs text alone | −0.021 | −0.059 | **+0.024** |

So the acoustic channel DOES carry information complementary to text -- it is
simply not linearly accessible. Linear models are diluted by 1280 largely
uninformative dimensions and have no way to discount them; axis-aligned splits
just ignore the unhelpful ones.

**This corrects the "fusion barely helps" reading.** That conclusion was
specific to the fusion architectures tried, not to the modality combination:
gradient boosting extracts +0.024 from exactly the two caches our learned
fusion extracted +0.003 from. Its macro F1 of 0.418 is the highest recorded
anywhere in this project, so the gain is concentrated in the minority classes
that every neural model collapsed.

XGBoost on the raw concatenation reaches **0.6189 test weighted F1** -- within
0.01 of the best neural result in the entire project (0.6283), with no dialogue
context, no attention pooling, and 205 seconds of CPU.

### But that parity is a GOLD result, and gold is not the realistic condition

Split by text condition, the comparison inverts:

| condition | best neural | best classical (text) | gap |
|---|---|---|---|
| gold | 0.6283 | 0.5953 | +0.033 |
| gold (vs classical on BOTH) | 0.6283 | 0.6189 | **+0.009 — parity** |
| **asr** | **0.5089** | 0.4670 | **+0.042** |

On clean transcripts the signal is easy enough that gradient boosting on frozen
features extracts most of it, and architecture adds almost nothing. On DEGRADED
transcripts -- the realistic case, and the one this project exists to study --
the neural models lead by **+0.042**, twice the ~0.02 seed-noise threshold and
more than four times the gold margin.

That is the more defensible robustness claim, and it has a mechanism: the
architectures carry an acoustic branch and dialogue context, and those matter
precisely when the text they supplement is unreliable. Quoting the gold parity
alone would understate what the models contribute under realistic conditions.

The number that sharpens or blunts this is XGBoost on `both` for ASR. If it
reaches ~0.50, the neural advantage is really about having the ACOUSTIC CHANNEL
rather than about architecture; if it stays near 0.47, the architectures are
contributing something a strong classical model on the same features cannot.

---

## 9. Results on test, and how to read them

All 91 checkpoints scored on the **full 2,610-utterance test set** through one
code path, so the table has a single provenance.

| family | runs scored | best test emotion | best test sentiment |
|---|---|---|---|
| fusion | 25 | 0.5968 (`phase2_textonly/gold`) | 0.6808 |
| bc-LSTM (incl. stacked) | 45 | **0.6283** (`stackedcat_gold/k2`) | **0.7039** |
| context-then-fusion | 12 | 0.6258 (`gold/text_only`) | 0.6995 |

### Caveats — these bind every number above

**Single seed.** Every cell is one run. Differences below roughly 0.02 are not
distinguishable from seed noise, and §7 is a worked example of a dev ordering
that reversed on test. **No ordering in this document should be quoted as a
finding without 3–5 seeds and a mean ± std.**

**Dev was used twice** — for early stopping *and* for choosing between runs — so
dev is mildly optimistic. Test numbers are the honest ones. Test scores here run
about 0.03 *above* dev, consistently across all 91 runs.

**Arms are not parameter-matched** in §7. `acoustic_only` carries 4.34 M against
`text_only`'s 3.55 M purely because the acoustic input is 1280-d against 768-d.
A measured matched pair: `both` at `lstm_hidden=180` ≈ 4.38 M ≈ `acoustic_only`
at 256.

**Loss differs between families.** The fusion family offers plain / weighted /
focal with inverse-frequency alpha; both context families use focal with
**no** alpha. That is an uncontrolled difference across families and is recorded
in every run marker.

### What actually holds

Only one effect is large enough to survive the caveats:

> **The gold → ASR gap is ~0.12 weighted F1, and it dwarfs every architectural
> choice measured here** — fusion mechanism, pooling, context width, context
> ordering, and stacking are all worth ≲0.02 individually.

Transcription quality dominates. That is the finding the evidence supports, and
it is a direct argument for the dialect/accent robustness question: improving
the front-end matters more than improving what sits on top of it.

---

## 10. Representation analysis — what the pipeline does to class structure

Five cached representations, measured on the full high-dimensional vectors
(never on 2-D projection coordinates) and visualised with PCA / UMAP / t-SNE.

**Test, gold condition:**

| stage | dim | probe F1 | cosine gap | PCA EVR |
|---|---|---|---|---|
| text (XLM-R `[CLS]`) | 768 | 0.5494 | 0.0806 | 0.629 |
| acoustic (masked mean) | 1280 | 0.3957 | **0.0138** | 0.393 |
| fused | 512 | 0.5705 | **0.1245** | 0.601 |
| context (bc-LSTM) | 512 | **0.5736** | 0.0873 | 0.282 |
| ctxfusion | 512 | 0.5576 | 0.0618 | 0.360 |

Probe F1 is a logistic regression fit on train and scored on test -- the only
measure here comparable ACROSS stages. Cosine gap is mean within-class minus
mean between-class cosine similarity. Silhouette is recorded in the JSON but is
dimension-dependent and must not be compared across stages.

**Probe F1 is a lower bound.** On the same text vectors, SVM reaches 0.5850 and
XGBoost 0.5953 against logistic regression's 0.5494 -- so the probe understates
linear separability by ~0.036 and misses nonlinear structure by ~0.046.

### Pooling, not the encoder, was destroying the acoustic signal

The `acoustic` row above is the MASKED MEAN, and its cosine gap of 0.0138 means
same-emotion clips are 1.4% more similar than different-emotion ones --
essentially nothing. Its UMAP is a single uniform blob with no class occupying
any region.

That looked like a limitation of the Whisper encoder: it is trained for
transcription, so discarding speaker and prosodic variation is what it is *for*.
Extracting the same audio through the four Phase 1 poolers shows otherwise:

| pooling | dim | probe F1 | cosine gap |
|---|---|---|---|
| masked mean (raw cache) | 1280 | 0.3957 | 0.0138 |
| **attention** | 512 | **0.4623** | **0.0779** |
| attentive stats | 512 | 0.4562 | 0.0665 |
| learned masked mean | 512 | 0.4178 | 0.0657 |

**Attention pooling raises acoustic class separation 5.6x**, to a level
comparable with raw text (0.0806). Its UMAP shows four or five identifiable
regions -- sadness upper-left, anger lower-left, surprise bottom, joy right --
where the masked mean showed none.

So the emotion signal was present in the frames all along and a naive mean was
averaging it away. Averaging over every frame dilutes exactly the moments where
emotion is expressed; attention finds them. Note the learned masked mean also
reaches 0.0657, so part of the gain is the trainable projection rather than
attention specifically -- but attention wins on both measures.

### The consequence: the context models were handicapped

| family | acoustic input | cosine gap of that input |
|---|---|---|
| fusion | **attention** (Phase 1 winner) | 0.0779 |
| bc-LSTM | masked mean | 0.0138 |
| context-then-fusion | masked mean | 0.0138 |

Fusion always had the good pooling. Both context families read
`{split}_embeddings_maskedmean.pt` through `DialogueDataset`, so **every context
result in sections 5-7 ran on the degraded acoustic representation** while
fusion did not.

That makes "context helps less than fusion" partly an artefact of a
preprocessing choice rather than an architectural fact. A 45-run grid re-runs
bc-LSTM (3 conditions x 5 widths) and ContextThenFusion `both` and
`acoustic_only` (3 x 2 x 5) on the attention-pooled vector, each cell pairing
against its existing masked-mean counterpart so only the pooling differs.
`text_only` and `neither` are not re-run: neither has an acoustic BiLSTM, so
changing the acoustic pooling does not test what those arms isolate.

---

## Reproducing

```bash
sbatch src/scripts/xlmr_finish_and_matrix.sbatch   # XLM-R + 3x3 matrix
sbatch src/scripts/fusion_grid.sbatch              # 25 fusion runs
sbatch src/scripts/bclstm_grid.sbatch              # 15 bc-LSTM cells
sbatch src/scripts/stacked_grid.sbatch             # stacked, replace
sbatch src/scripts/stacked_concat_grid.sbatch      # stacked, concat
sbatch src/scripts/ctxfusion_grid.sbatch           # 2x2 ablation
sbatch src/scripts/ctxfusion_window_grid.sbatch    # + window sweep
sbatch src/scripts/evaluate_all.sbatch             # every checkpoint on test
```

Every grid skips runs whose `TRAINING_COMPLETE.json` exists, so a resubmission
after a walltime kill continues rather than restarting.

Reports rebuild any time from the checkpoint tree:

```bash
python3.12 src/evaluation/fusion_report.py
python3.12 src/evaluation/bclstm_report.py
python3.12 src/evaluation/ctxfusion_report.py
```
---

## 11. Dialect probe — the trained pipeline on unseen accented speech

The only experiment in this project run on data the models were never trained
on, and the only one scored against labels this project produced itself.

### Setup

100 clips from the CSTR/Google UK–Irish English dialect corpus
(`ylacombe/english_dialects`) — read speech, 6 accents, 23 speakers. Sampled
from a 17,877-clip pool stratified so all seven emotions appear at 14–16 each,
because the pool is 97% neutral by Voxtral's zero-shot label and a random
sample would have been uninformative.

**Inference only.** Every weight frozen, loaded from the finished MELD
checkpoints. No optimiser is constructed anywhere in
`src/evaluation/dialect_end_to_end.py`.

    .wav -> Voxtral-Mini encoder + LLM -> frames (T,1280) + ASR transcript
         -> XLM-R (per condition)       -> 768-d
         -> 18 classifiers              -> emotion

18 models = 2 conditions x (4 fusion mechanisms + 4 bc-LSTM orderings +
ctxfusion). No "best model" was pre-selected: MELD rank is selection on a
different distribution, and which architecture generalises was the question.

### Ground truth

The corpus has NO emotion labels. All 100 clips were annotated by ear by the
author, blind to model output — the listening pack deliberately excluded any
predicted emotion from its filenames. Resulting distribution:

| neutral | surprise | anger | sadness | disgust | joy | fear |
|---|---|---|---|---|---|---|
| 45 | 13 | 12 | 12 | 8 | 8 | **2** |

45% neutral confirms the read-speech caveat: most clips genuinely carry no
emotion, and fear at n=2 cannot support a per-class claim.

### Results — weighted F1 against the human labels

| model | overall | macro | Irish | Midl. | North. | Scot. | South. | Welsh | spread |
|---|---|---|---|---|---|---|---|---|---|
| **asr_cleaned ctxfusion k0** | **0.635** | 0.524 | 0.590 | 0.727 | 0.576 | 0.593 | 0.789 | 0.333 | 0.456 |
| asr ctxfusion k0 | 0.629 | 0.503 | 0.645 | 0.732 | 0.601 | 0.571 | 0.796 | 0.363 | 0.433 |
| asr bclstm attnraw k0 | 0.619 | 0.495 | 0.667 | 0.576 | 0.633 | 0.653 | 0.698 | 0.289 | 0.409 |
| asr_cleaned bclstm attn k0 | 0.604 | 0.483 | 0.645 | 0.727 | 0.524 | 0.543 | 0.744 | 0.317 | 0.426 |
| asr bclstm stackedcat k0 | 0.597 | 0.486 | 0.667 | 0.732 | 0.645 | 0.571 | 0.623 | 0.399 | 0.333 |
| asr bclstm attn k0 | 0.587 | 0.466 | 0.758 | 0.576 | 0.529 | 0.558 | 0.701 | 0.317 | 0.440 |
| asr fusion gated | 0.587 | 0.457 | 0.773 | 0.671 | 0.625 | 0.561 | 0.603 | 0.306 | 0.467 |
| asr bclstm stacked k0 | 0.583 | 0.423 | 0.535 | 0.671 | 0.548 | 0.599 | 0.649 | 0.382 | 0.289 |
| asr_cleaned bclstm stacked k0 | 0.580 | 0.442 | 0.503 | 0.756 | 0.524 | 0.656 | 0.652 | 0.369 | 0.387 |
| asr_cleaned bclstm stackedcat k0 | 0.578 | 0.440 | 0.473 | 0.756 | 0.561 | 0.656 | 0.683 | 0.281 | 0.475 |
| asr fusion concat | 0.575 | 0.444 | 0.742 | 0.833 | 0.606 | 0.436 | 0.599 | 0.448 | 0.397 |
| asr_cleaned fusion sum | 0.574 | 0.440 | 0.488 | 0.726 | 0.533 | 0.670 | 0.713 | 0.205 | 0.521 |
| asr_cleaned fusion concat | 0.573 | 0.446 | 0.570 | 0.671 | 0.575 | 0.572 | 0.621 | 0.317 | 0.354 |
| asr_cleaned bclstm attnraw k0 | 0.569 | 0.444 | 0.533 | 0.635 | 0.571 | 0.610 | 0.671 | 0.300 | 0.371 |
| asr fusion sum | 0.553 | 0.423 | 0.658 | 0.576 | 0.683 | 0.458 | 0.578 | 0.421 | 0.262 |
| asr_cleaned fusion crossmodal | 0.540 | 0.394 | 0.488 | 0.671 | 0.526 | 0.641 | 0.622 | 0.244 | 0.427 |
| asr fusion crossmodal | 0.528 | 0.392 | 0.667 | 0.732 | 0.590 | 0.465 | 0.538 | 0.264 | 0.468 |
| asr_cleaned fusion gated | 0.505 | 0.356 | 0.473 | 0.671 | 0.512 | 0.565 | 0.486 | 0.329 | 0.343 |
| *voxtral zero-shot* | *0.478* | *0.443* | *0.473* | *0.648* | *0.396* | *0.486* | *0.566* | *0.403* | *0.252* |

n per accent: Southern 24, Northern 21, Scottish 21, Welsh 12, Irish 11,
Midlands 11.

### Finding 1 — the pipeline beats the zero-shot LLM, everywhere

**All 18 trained models beat Voxtral zero-shot** (0.478); the best by **+0.157**.
This is the clearest justification the pipeline has: on data it never saw,
scored against labels produced independently of it, the trained stack is worth
substantially more than prompting the same audio LLM directly. Compare
`project_context.md`, where zero-shot Voxtral BEAT the text-only model on MELD.

### Finding 2 — the MELD ranking does not transfer

| model | MELD asr_cleaned | dialect | rank change |
|---|---|---|---|
| ctxfusion | 0.4927 (**worst of family**) | **0.635 (1st)** | last -> first |
| bclstm attn | **0.5334 (best)** | 0.604 (4th) | first -> fourth |
| bclstm attnraw | 0.5075 | 0.619 (3rd) | up |

Both ctxfusion variants take 1st and 2nd of 18. A single cell would be noise;
the whole family at the top is harder to dismiss.

**Read this carefully.** ctxfusion is "context-then-fusion", but at K=0 it sees
NO neighbouring utterances — these are isolated sentences. What generalises is
its ARCHITECTURE, a separate BiLSTM per modality before fusing, not dialogue
context. The defensible statement is: *per-modality recurrent processing before
fusion generalises better out-of-domain than it performs in-domain.*

This vindicates running all 18 rather than the MELD winner. Selecting on MELD
would have shipped the 4th-best model.

### Finding 3 — accuracy and accent-robustness pull apart

| | mean accent spread |
|---|---|
| 18 trained models | **0.403** |
| Voxtral zero-shot | **0.252** |

**Welsh is the worst accent for 17 of 18 models**, and only 2 of 18 beat
Voxtral on Welsh. The trained pipeline is more accurate overall and
*less accent-robust* than the zero-shot baseline it beats — it has learned the
training distribution, and Southern English (0.789, closest to MELD's American
TV speech) is where it does best.

This is the thesis result: **fine-tuning buys accuracy at the cost of accent
robustness.**

### Finding 4 — the accent gap is NOT an ASR effect

The corpus ships gold transcripts, so ASR quality is measurable per accent
without any annotation. Full analysis in `ENGLISH_DIALECT_DATA.md`; the
decisive table:

| accent | ASR WER | emotion F1 (best model) |
|---|---|---|
| Scottish | 0.0128 | 0.593 |
| **Welsh** | **0.0143** | **0.333** |
| Midlands | 0.0333 | 0.727 |
| Irish | 0.0455 | 0.590 |
| Southern | 0.0481 | 0.789 |
| Northern | 0.0848 | 0.576 |

Across the six accents: **Pearson r = +0.32 (p = 0.54), Spearman rho = +0.09
(p = 0.87)** -- no relationship, and the sign is backwards if anything.

**Welsh has the second-lowest WER and the worst emotion F1 by a wide margin.**
Its transcripts are near-perfect and the emotion models still fail on it. So
the accent degradation lives in the acoustic and semantic REPRESENTATIONS, not
in the transcription front-end.

This separation is impossible on MELD, where corpus WER is 0.38 and dominates
every other effect. Here it is 0.044 on the probe (0.060 on the full 17,879
clips) with **zero defects at MELD's thresholds** -- 100 of 100 clips clean,
median per-utterance WER 0.0000 in every accent.

A corollary: with nothing for a keep-list to remove, the `asr` vs
`asr_cleaned` comparison on this data is purely a difference between two sets
of WEIGHTS, on byte-identical defect-free input.

*(n = 6 accents, so this rules out a strong WER-to-F1 relationship, not a
modest one.)*

### Finding 5 — per-class F1, and where the gain actually comes from

Best model (asr_cleaned ctxfusion k0) against Voxtral zero-shot:

| class | n | F1 | prec | rec | vox F1 | vox prec | vox rec |
|---|---|---|---|---|---|---|---|
| neutral | 45 | **0.756** | 0.838 | 0.689 | 0.492 | **0.938** | 0.333 |
| joy | 8 | **0.842** | 0.727 | 1.000 | 0.636 | 0.500 | 0.875 |
| disgust | 8 | **0.667** | 0.714 | 0.625 | 0.571 | 0.462 | 0.750 |
| surprise | 13 | **0.600** | 0.857 | 0.462 | 0.444 | 0.429 | 0.462 |
| sadness | 12 | 0.400 | 0.385 | 0.417 | 0.385 | 0.357 | 0.417 |
| anger | 12 | 0.400 | 0.333 | 0.500 | **0.444** | 0.400 | 0.500 |
| fear | **2** | 0.000 | 0.000 | 0.000 | 0.125 | 0.071 | 0.500 |

Per-class F1 across all 18 models (mean / best / worst):

| class | n | mean | best | worst |
|---|---|---|---|---|
| neutral | 45 | 0.735 | 0.775 | 0.676 |
| joy | 8 | 0.720 | 0.842 | 0.588 |
| surprise | 13 | 0.491 | 0.609 | 0.333 |
| anger | 12 | 0.458 | 0.571 | 0.333 |
| sadness | 12 | 0.361 | 0.500 | 0.273 |
| disgust | 8 | 0.350 | **0.800** | **0.000** |
| fear | 2 | 0.019 | 0.333 | 0.000 |

Three readings:

**The +0.157 gain is restraint, not better emotion detection.** Voxtral's
neutral PRECISION is 0.938 -- it almost never calls something neutral
wrongly -- but its RECALL is 0.333. The trained pipeline trades a little
precision (0.838) for double the recall (0.689). It is better at knowing when
NOT to predict an emotion, which on read speech is most of the time.

**Fear's 0.000 is uninterpretable at n=2**, and it is the one class where
Voxtral "wins" (0.125) purely by predicting fear 14 times and catching one.
That is not a Voxtral advantage and should not be reported as one.

**Disgust ranges 0.000 to 0.800 across the 18 models** on n=8. That spread is
the clearest single illustration that per-class numbers here are unstable, and
the strongest concrete argument for a larger annotated set.

### Finding 6 — fear/disgust collapse survives the domain change

Per-emotion recall, best model:

| emotion | n | recall |
|---|---|---|
| joy | 8 | 1.00 |
| neutral | 45 | 0.69 |
| disgust | 8 | 0.62 |
| anger | 12 | 0.50 |
| surprise | 13 | 0.46 |
| sadness | 12 | 0.42 |
| **fear** | **2** | **0.00** |

Fear fails here as it does on MELD — but n=2, so this corroborates rather than
demonstrates.

### Caveats

- **n=100, one annotator, no inter-annotator agreement.** The labels are the
  author's own ear. A second annotator on even 30 clips would give a kappa and
  materially strengthen this.
- **Margins are thin.** 1st to 3rd spans 0.016. Only the gap to Voxtral
  (+0.157) and the accent spread are comfortably outside what n=100 supports.
- **Read speech, not conversation.** 45% neutral; fear n=2.
- **Accent is confounded with speaker.** Different speakers per accent, so
  "Welsh collapses" is partly a claim about 12 clips from a few voices.
  VoxMovies-style within-speaker pairing would fix this and is the obvious
  next experiment.
- **`stacked`/`stackedcat` condition columns are confounded** — their fused
  source differs by mechanism as well as by training subset.
- **10 of 18 checkpoint loads were condition-unverified** (all `bclstm/*` plus
  both XLM-R encoders record no `args["config"]`); their pairing rests on the
  path convention and the explicit `TEXT_ENCODER` map.

### Reproducing

```bash
sbatch src/scripts/dialect_end_to_end.sbatch      # ~5 min, 18 models, gpu:1
```

Outputs under `/dcs/large/u5734759/data/dialect_probe_100/`:
`dialect_predictions.csv`, `dialect_scores.csv`, `dialect_scored_clean.csv`,
`dialect_predictions_meta.json`, `dialect_listening_pack.zip`.
