# English dialect data — corpus, WER/VAD analysis, and the accent probe

The out-of-domain evaluation set: what it is, how clean it is, and what it
supports. Experiment results live in `EXPERIMENTS.md` §11; this document is
about the **data**.

---

## 1. The corpus

**CSTR/Google Crowdsourced UK & Ireland English Dialect speech**
(`ylacombe/english_dialects` on HuggingFace), 11 accent × gender configs.

| | |
|---|---|
| utterances | **17,879** (with gold text) |
| accents | Irish, Midlands, Northern, Scottish, Southern, Welsh |
| speakers | 73 |
| mean duration | 6.2 s |
| audio | 16 kHz mono PCM_16 WAV |
| gold transcripts | **yes** — every utterance |
| emotion labels | **no** — see §4 |

Sampled by `src/preprocessing/sample_dialects.py`, which streams rather than
downloading the full 8.98 GB, and decodes with `decode=False` + soundfile
because torchcodec cannot load on this cluster (missing `libnvrtc.so.13`).

### Composition is heavily unbalanced

| accent | clips | speakers | mean dur |
|---|---|---|---|
| Southern | 8,494 | 31 | 6.19 s |
| Welsh | 2,849 | 11 | 6.87 s |
| Northern | 2,847 | 14 | 6.32 s |
| Scottish | 2,543 | 11 | 6.13 s |
| Midlands | 696 | 3 | 6.22 s |
| Irish | 450 | 3 | 5.72 s |

Southern is 48% of the corpus; Irish and Midlands have **3 speakers each**.
Any per-accent claim about those two rests on very few voices, and speaker
identity is confounded with accent throughout — there is no speaker who
appears in more than one accent.

---

## 2. ASR quality — the objective half

Voxtral-Mini transcripts against the corpus gold text, scored with the SAME
code and normalisation policy as MELD (`src/evaluation/text_normalisation.py`,
corpus WER = Σ(S+D+I)/Σ(S+D+H)), so the two corpora are directly comparable.

### The headline: this corpus is easy for ASR

| corpus | corpus WER (normalised) |
|---|---|
| MELD test | **0.3823** |
| MELD train | 0.3461 |
| **dialect, full 17,879** | **0.0602** |
| **dialect, the 100-clip probe** | **0.0440** |

**Roughly 6× lower WER than MELD.** That is not a statement about accents being
easy; it is a statement about the two corpora. MELD is overlapping multi-party
TV dialogue with laughter and music; this is prompted read speech recorded for
a speech corpus. Li et al. (`RESULTS.md` §2c) explain the MELD side: 71.5% of
MELD utterances are ≤10 words, and WER collapses on short references.

### Per-accent WER

Full corpus (17,879):

| accent | n | corpus WER |
|---|---|---|
| Southern | 8,494 | 0.0541 |
| Scottish | 2,543 | 0.0546 |
| Midlands | 696 | 0.0555 |
| Welsh | 2,849 | 0.0683 |
| Northern | 2,847 | 0.0690 |
| **Irish** | 450 | **0.1126** |

Irish is transcribed roughly **2× worse** than any other accent. With only 3
speakers, that may be a speaker effect rather than an accent effect — but it is
the one clear ASR disparity in the corpus, and it is worth naming as a
fairness observation in its own right: the front-end is measurably worse for
one accent group.

The 100-clip probe (deliberately stratified by predicted emotion, so not a
representative sample of WER):

| accent | n | corpus WER | median | speech ratio |
|---|---|---|---|---|
| Scottish | 21 | 0.0128 | 0.0000 | 0.66 |
| Welsh | 12 | 0.0143 | 0.0000 | 0.68 |
| Midlands | 11 | 0.0333 | 0.0000 | 0.69 |
| Irish | 11 | 0.0455 | 0.0000 | 0.67 |
| Southern | 24 | 0.0481 | 0.0000 | 0.74 |
| Northern | 21 | 0.0848 | 0.0000 | 0.72 |

**Median per-utterance WER is 0.0000 in every accent** — more than half of all
clips are transcribed perfectly.

---

## 3. Defect analysis — the MELD filtering pipeline applied here

Same thresholds as MELD (`analyse_wer_vad.py`): `SHORT_SEC` 1.0 s, `LOW_VAD`
0.20 speech ratio, `HIGH_WER` 0.50, runaway = ASR > 5× gold words AND ≥ 40
words. Silero-VAD for the speech ratio.

### Full corpus (17,879)

| flag | n | % |
|---|---|---|
| is_short | 0 | 0.0% |
| is_empty_asr | 0 | 0.0% |
| is_runaway | 4 | 0.0% |
| is_high_wer | 254 | 1.4% |
| is_wer_over_1 | 9 | 0.1% |
| **clean (no flags)** | **17,625** | **98.6%** |

Clean-subset corpus WER: 0.0483 (from 0.0602).

### The 100-clip probe

| flag | n |
|---|---|
| **every flag** | **0** |
| **clean** | **100 (100%)** |

**Zero defects at MELD's thresholds.** For comparison, MELD test is 85.2%
clean and its defect classes carry corpus WERs up to 2.89.

This matters for interpreting the emotion results: on MELD, the WER keep-list
removes ~15% of utterances and measurably changes what a model learns. Here
there is nothing to remove. **The `asr_cleaned` vs `asr` comparison on this
data is purely a difference between two sets of weights** — both conditions
received byte-identical, defect-free input.

---

## 4. What this data cannot do

**No emotion labels.** The corpus ships speaker IDs, accent, gender and gold
text. Emotion had to be annotated by ear — 100 clips, one annotator, no
inter-annotator agreement. That is the binding constraint on every emotion
claim, not compute.

**Read speech, not conversation.** People reciting prompts. The author's
annotation came out **45% neutral** with **fear n=2**, which is the honest
distribution for this material and is why per-class emotion claims here are
weak. It is also why the models' ~32% neutral rate reads as over-prediction.

**Accent is confounded with speaker.** No speaker appears in two accents, and
Irish/Midlands have 3 speakers each. A VoxMovies-style design — the same
identity in calm and expressive conditions — would separate the two, and is
the obvious next corpus.

**Stratified, not representative.** The 100 were sampled to hold all seven
predicted emotions at 14–16 each, against a pool that is 97% neutral by
Voxtral's zero-shot label. Annotating them yields per-emotion **precision**,
not recall: we cannot know what the models failed to detect.

---

## 5. The finding this data enables

Because WER here is near-zero and uniform, the corpus separates two effects
that MELD cannot:

| accent | ASR WER | emotion F1 (best model) |
|---|---|---|
| Scottish | 0.0128 | 0.593 |
| **Welsh** | **0.0143** | **0.333** |
| Midlands | 0.0333 | 0.727 |
| Irish | 0.0455 | 0.590 |
| Southern | 0.0481 | 0.789 |
| Northern | 0.0848 | 0.576 |

Across the six accents: **Pearson r = +0.32 (p = 0.54), Spearman ρ = +0.09
(p = 0.87)**. No relationship, and the sign is if anything backwards.

**Welsh has the second-lowest WER and by far the worst emotion F1.** Its
transcripts are almost perfect; the emotion models still fail on it.

So the accent gap measured in `EXPERIMENTS.md` §11 — mean spread 0.403 across
18 models, Welsh worst for 17 of 18 — **is not mediated by transcription
quality**. It is a property of the acoustic and semantic representations, not
of the ASR front-end. On MELD that distinction is impossible to draw, because
WER there is 0.38 and dominates everything.

That is the strongest thing this dataset does for the thesis, and it needed no
human labels to establish the WER half of it.

*(n = 6 accents, so the correlation is a weak test. It rules out a strong
WER→F1 relationship; it does not rule out a modest one.)*

---

## 6. Files

| path | contents |
|---|---|
| `data/dialect_probe_full/manifest.csv` | 17,877 clips, accent/gender/speaker/gold text |
| `data/dialect_probe_full/dialect_annotation_sheet.csv` | + Voxtral ASR, WER, zero-shot emotion |
| `data/dialect_probe_full/dialect_full_wer_summary.json` | §2, §3 full-corpus numbers |
| `data/dialect_probe_100/selection.csv` | the stratified 100 |
| `data/dialect_probe_100/dialect_wer_vad.csv` | per-utterance WER, VAD, defect flags |
| `data/dialect_probe_100/dialect_wer_vad_summary.json` | §2, §3 probe numbers |
| `data/dialect_probe_100/dialect_predictions.csv` | 18 model predictions + human labels |
| `data/dialect_probe_100/dialect_scores.csv` | per-model F1 |
| `data/dialect_probe_100/dialect_listening_pack.zip` | audio + annotation sheet |

Reproduce with `src/scripts/dialect_end_to_end.sbatch` (~5 min, gpu:1).
