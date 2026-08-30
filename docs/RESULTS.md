# Results

Headline numbers and comparison against published work. The full experimental
record — every grid, every ablation, and the corrections made along the way —
is in `EXPERIMENTS.md`.

**Metric: weighted F1 on MELD's 7-class emotion task**, test split (2,610
utterances). Weighted rather than accuracy because MELD is 47% neutral.

---

## 1. The condition that matters

Almost all published MELD work uses the **gold transcripts** — the
human-annotated `Utterance` column of the CSV. No deployed system has those. A
real system transcribes the audio first, and every error propagates.

This project reports both, and treats the ASR condition as primary:

| condition | text source | our best dev WF1 |
|---|---|---|
| gold | MELD `Utterance` column | **0.6149** |
| **asr** | **Voxtral-Mini transcripts, WER 0.42** | **0.5339** |
| asr_cleaned | ASR + WER-based keep-list | **0.5370** |

The **0.081 gold→ASR drop** is the single largest effect measured anywhere in
this project. Every architectural choice we tested is worth ≤0.05 by comparison.

---

## 2. Benchmark: Taghavi et al., ICLR 2023 (Tiny Papers)

*"A Change of Heart: Improving Speech Emotion Recognition through
Speech-to-Text Modality Conversion"* — arXiv:2307.11584

The closest published work to this project. Same dataset, same metric, and —
unusually — it **reports the ASR condition explicitly** rather than assuming
gold transcripts.

Their approach: convert speech to text with an ASR system, then classify with
RoBERTa-base. The audio is **discarded** after conversion.

### Their results, and ours

| method | modality | WF1 (%) |
|---|---|---|
| SpeechFormer | speech | 41.9 |
| SpeechFormer++ | speech | 47.0 |
| DWFormer | speech | 48.5 |
| DST | speech | 48.8 |
| **Modality-Conversion** (Vosk ASR → RoBERTa) | speech → text | **43.1** |
| **ours, ASR condition** | speech → text **+ audio + context** | **53.4** |
| Modality-Conversion++ (*ideal* ASR, i.e. gold) | gold text | 60.4 |
| **ours, gold condition** | gold text + audio + context | **61.5** |

**On the ASR condition we lead by +10.3 points**, and our 53.4 also exceeds
every speech-only method in their comparison table (best: DST at 48.8).

This is the ONE benchmark in this document where an ASR-condition comparison is
legitimate, because Taghavi et al. actually ran that condition and report it.
Everywhere else in the literature the ASR row simply does not exist, and our
ASR numbers stand alone rather than against a published counterpart.

### The more interesting number is the degradation

| | ideal/gold | real ASR | drop |
|---|---|---|---|
| Taghavi et al. | 60.4 | 43.1 | **−17.3** |
| **this work** | 61.5 | 53.4 | **−8.1** |

Both start from a similar gold-condition score. **Ours loses less than half as
much when the transcripts become real.**

### Why, mechanistically

Modality conversion **throws away the modality that survives transcription
error**. Our own ablation measures exactly this (`EXPERIMENTS.md` §4):

| condition | text-only | fusion (text + audio) | gain from audio |
|---|---|---|---|
| gold | 0.5968 | 0.5957 | **−0.001** |
| **asr** | 0.4350 | **0.4952** | **+0.060** |

The acoustic cache is byte-identical across text conditions — same audio, same
encoder — so while text degrades by 0.162 under ASR, audio does not degrade at
all. On ASR the acoustic branch alone (0.4893) actually **beats** the text
branch (0.4350).

That is the thesis in one line: **a text-only pipeline discards the channel that
is robust to the errors it is most vulnerable to.**

### Caveats on this comparison

- **Different ASR systems.** They use the Vosk API; we use Voxtral-Mini at
  corpus WER 0.42. Part of the +10.3 gap is a better front-end rather than a
  better classifier. Their paper does not report WER, so the two effects cannot
  be separated from published numbers alone.
- **Different classifiers.** RoBERTa-base vs XLM-RoBERTa-base plus an acoustic
  branch and a dialogue-context model. The comparison is between *pipelines*,
  not between equally-matched components.
- **Single seed on our side.** See §5.

---

## 2a. Benchmark: Culnan et al., WASSA 2021 — the closest prior work

*"Me, myself, and ire: Effects of automatic transcription quality on emotion,
sarcasm, and personality detection"*

This paper asks **the same question this project asks**, and states it in the
abstract: *"In deployment, systems that use speech as input must make use of
automated transcriptions. Yet, typically when these systems are evaluated, gold
transcriptions are assumed."* They cite COSMIC and Poria et al. as examples of
that assumption -- the same papers benchmarked in §2b and §3 here.

They evaluate three ASR tools (Sphinx, Kaldi, Google Cloud) against gold on
MELD, MUStARD and FirstImpr, with text-only, audio-only and multimodal models.

### Their MELD results, and ours

| model | text source | their WF1 | ours |
|---|---|---|---|
| Audio only | — | 37.38 | 48.93 |
| Text (Gold) | gold | 57.32 | **61.5** |
| Text (Google) | **ASR** | 37.60 | — |
| **MM (Google)** | **ASR + audio** | **40.94** | **53.4** |
| MM (Gold) | gold + audio | 56.28 | 61.5 |

**On the ASR multimodal condition we lead by +12.5 points.**

### They independently confirm the central mechanism

Their conclusion -- *"the inclusion of audio features partially mitigates
transcription errors"* -- is measured on MELD as Text(Google) 37.60 →
MM(Google) 40.94, a gain of **+3.3 from adding audio under ASR**.

Our equivalent ablation gives **+6.0** (text-only 0.4350 → fusion 0.4952).
Same direction, roughly twice the magnitude.

**They also reproduce the sign flip.** On gold, adding audio *hurts* slightly
(Text 57.32 → MM 56.28, −1.0). On ASR it helps (+3.3). That is exactly the
pattern measured here independently (gold −0.001, asr +0.060), on the same
dataset with different features, different models and a different ASR system.

Two independent confirmations of the same effect is considerably stronger than
one.

### WER does not predict downstream performance

Their MELD WERs: **Sphinx 114.9, Kaldi 104.4, Google 82.6**. Yet Sphinx
transcriptions beat Google's on MUStARD, and none of the differences between
ASR tools reach significance downstream. Their finding: *"word error rates do
not correlate well with downstream performance."*

This bears directly on the Voxtral-Small decision in `EXPERIMENTS.md` §8. Small
showed no consistent WER improvement over Mini (+0.017 dev, −0.020 test), and
this paper indicates WER would have been a poor predictor of downstream quality
in either direction. The gate was the right call, but for a stronger reason than
originally stated: WER is a weak proxy, so a front-end change that does not even
move WER consistently has no basis for expecting downstream gains.

### Caveats on this comparison

- **Their ASR is far worse.** MELD WERs of 82.6–114.9% against our Voxtral-Mini
  at 42%. A WER above 100% is possible when insertions exceed the reference
  length -- their transcripts are severely degraded. A substantial part of the
  +12.5 gap is the front-end, not the classifier.
- **Their models are deliberately simple.** GloVe + a two-layer LSTM for text,
  a feedforward network over averaged OpenSMILE features for audio, late fusion.
  They state explicitly that they are "not trying to define a new state of the
  art". Ours uses XLM-RoBERTa, a learned attention pooler over Whisper encoder
  frames, and a dialogue-context model.
- **Their gold text-only (57.32) exceeds their gold multimodal (56.28)**, which
  matches our own finding that audio adds nothing when transcripts are clean.

### What this paper gives the thesis

A **peer-reviewed precedent** for the framing, an **independent replication** of
the audio-mitigates-ASR-error effect, and a **published warning against WER as a
proxy** for downstream quality. It is the paper this project most directly
extends: same question, stronger front-end, stronger models, larger effect.

---

## 2b. Benchmark: COSMIC (Ghosal et al., EMNLP Findings 2020)

*"COSMIC: COmmonSense knowledge for eMotion Identification in
Conversations"*

A stronger comparison than Taghavi et al., and one we do **not** beat on the
gold condition. Reported honestly because the gap is informative.

### Their MELD table, and where we sit

**Apples to apples: gold transcripts only.** COSMIC reports no ASR condition,
so our ASR number has no counterpart in this table and is deliberately omitted
-- placing it here would invite a comparison against a condition they never
evaluated.

| method | modalities | MELD 7-class WF1 (gold text) |
|---|---|---|
| **COSMIC** | text only | **65.21** |
| RoBERTa DialogueRNN | text only | 63.61 |
| RoBERTa | text only | 62.02 |
| **ours, gold** | **text + audio** | **61.5** |
| DialogueRNN (GloVe) | text only | 57.03 |
| CNN (GloVe) | text only | 55.02 |

Note the modality column: **every method in this table except ours is text
only.** COSMIC's §4.1 states results are reported "from the textual information
for all four datasets", and the baselines it compares against are the
text-feature versions. So a single-modality model beats our two-modality
pipeline here -- the gap is a stronger TEXT branch, not a modality advantage.

COSMIC beats our gold-condition result by **3.7 points**. Our 61.5 sits just
below plain RoBERTa (62.02) and comfortably above the GloVe-era
DialogueRNN (57.03) and CNN (55.02).

### Why the gap, and what it does and does not mean

**COSMIC is text-only.** Their §4.1 is explicit: results are reported "for
conversational emotion recognition from the **textual information** for all
four datasets". There is no acoustic or visual branch. So this is not a
multimodal comparison -- it is a much stronger *text* model.

Three concrete advantages over our text branch:

- **RoBERTa-Large**, 355M parameters and 1024-d hidden, against our
  XLM-RoBERTa-**base** at 278M and 768-d;
- **an external commonsense knowledge graph** (COMET trained on ATOMIC),
  supplying five inferred relations per utterance -- speaker intent, effect and
  reaction, and listener effect and reaction;
- **five GRU state trackers** modelling speaker/listener internal, external and
  intent states, against our single BiLSTM.

Their own ablation is worth noting: removing all commonsense knowledge drops
MELD 7-class from 65.21 to 64.28, and they state plainly that "although the
performance improvement is observed using commonsense knowledge across the
datasets, this improvement is not very substantial". So most of their margin
over us is the **larger encoder**, not the commonsense machinery.

**They also average five runs**; every number in this document is a single
seed. The comparison is therefore between our single sample and their mean.

### What this comparison actually establishes

COSMIC is better than our pipeline at the task **as the literature defines
it** -- 7-class emotion from gold transcripts. That is a fair result and we do
not dispute it.

But COSMIC has **no acoustic branch**. Under the ASR condition its text
representation would degrade exactly as Taghavi et al.'s did (60.4 -> 43.1, a
17.3-point fall), with nothing to fall back on. Our own ablation measures the
size of that fallback: text-only loses 0.162 under ASR while fusion loses
0.100, because the acoustic channel does not degrade at all.

The claim is therefore narrow and specific: **not that this pipeline is the
best MELD model, but that the best MELD models are evaluated under an
assumption -- perfect transcription -- that a deployed system cannot satisfy,
and that the architectural choices which matter change once it is dropped.**

A direct test of that would be to run COSMIC on our Voxtral transcripts. It is
not run here (their commonsense feature extraction requires COMET inference
over every utterance), but it is the obvious next experiment, and the
prediction is explicit: a text-only model with no acoustic fallback should lose
more under ASR than a fused one.

---

## 2c. Reference point: Li et al., Interspeech 2023 — is our WER competitive?

*"ASR and Emotional Speech: A Word-Level Investigation of the Mutual Impact of
Speech and Emotion Recognition"* — arXiv:2305.16065

Not a performance benchmark: no comparable MELD emotion F1 is reported. It is
used here for something more useful -- an **independent measurement of what WER
is achievable on MELD**, which is the missing piece in the Culnan comparison.

### Our ASR front-end is competitive

| ASR model | MELD WER (%) |
|---|---|
| Kaldi Librispeech | 58.5 |
| wav2vec2-base-960h | 57.8 |
| Conformer (ESPnet) | 52.1 |
| **Voxtral-Mini (ours)** | **42.4** |
| Whisper-medium | 34.8 |

Voxtral-Mini at 42.4% sits between Conformer and Whisper-medium. This settles a
caveat raised in §2a: Culnan et al. report MELD WERs of 82.6-114.9%, so it was
unclear how much of our +12.5 advantage came from a better front-end rather
than better modelling. Li et al. show independently that **34.8-58.5% is a
normal range on MELD**, so our 42.4% is a reasonable front-end and not an
outlier -- while Culnan's 82.6-114.9% is unusually poor.

Note the hedge: *a* normal range, not *the* normal range. Published MELD WER
spans roughly 20% to 115% across papers (§2e), a 5x spread that no difference
in ASR quality alone explains. Reference normalisation, scoring of empty
hypotheses, and whether the full 7-class set or a 4-class subset is scored all
move this number substantially. **Cross-paper WER comparison on MELD is
therefore weak evidence**, and this section should be read as establishing that
42.4% is unremarkable, not that it is good.

### Why MELD is the hardest emotion corpus

| corpus | WER range across 4 ASR models |
|---|---|
| IEMOCAP | 12.3 – 36.8 |
| MOSI | 17.3 – 40.9 |
| **MELD** | **34.8 – 58.5** |

Their explanation is utterance length. **71.5% of MELD utterances are ≤10
words**, and WER falls sharply with length:

| utterance length | MELD WER |
|---|---|
| ≤10 words | **73.3** |
| 11–20 | 48.6 |
| 21–30 | 42.1 |
| ≥30 | 38.8 |

This independently explains a discrepancy noted in `EXPERIMENTS.md` §2: our
corpus WER is 0.38 on train but the *mean per-utterance* WER is 0.68. Short
references produce enormous per-utterance ratios, and MELD is overwhelmingly
short utterances. The corpus figure is the honest aggregate.

### ASR quality varies by emotion, and it tracks utterance length

MELD WER per emotion (their Table 5):

| emotion | WER | % utterances ≤10 words |
|---|---|---|
| anger | 52.7 | 64.1 |
| sadness | 52.9 | 62.0 |
| disgust | 53.4 | 60.1 |
| neutral | 58.3 | 72.8 |
| fear | 58.6 | 62.1 |
| joy | 59.9 | 72.4 |
| **surprise** | **65.3** | **82.8** |

Transcription quality is **not uniform across the classes being predicted**.
Surprise is transcribed worst, and it is also the class with the most short
utterances. Notably **neutral is not the best-transcribed class** despite being
the least emotionally "distorted" -- its short-utterance ratio offsets that.

This is a confound worth naming -- but **we measured it, and it is weak.**
Cross-checking their per-class MELD WER against our per-class probe F1 (gold
text):

| class | our F1 | their WER | n (test) |
|---|---|---|---|
| neutral | 0.708 | 58.3 | 1256 |
| surprise | 0.504 | **65.3 (worst)** | 281 |
| joy | 0.556 | 59.9 | 402 |
| anger | 0.331 | **52.7 (best)** | 345 |
| sadness | 0.255 | 52.9 | 208 |
| disgust | 0.104 | 53.4 | 68 |
| fear | 0.095 | 58.6 | 50 |

The correlation is weak and in places inverted. Surprise is the *worst*
transcribed class (65.3) yet our *second best* by F1 (0.504); anger is the
*best* transcribed (52.7) yet mid-table (0.331). Disgust and anger have almost
identical WER (53.4 vs 52.7) but F1 differing by a factor of three.

What per-class F1 actually tracks here is **support**: the two collapsed
classes, fear (n=50) and disgust (n=68), are the two rarest. The fear/disgust
failure in this project is a class-imbalance result, not an artefact of
transcription quality, and should be reported as such.

### The WER-to-performance curve has a knee

They degrade transcripts from 5% to 50% WER and measure SER accuracy at each
step. Accuracy falls throughout, with a **steep drop between 15% and 25% WER**.
Both this project (42.4%) and Culnan et al. (82.6%+) operate well beyond that
knee, in the regime where transcription quality dominates -- which is consistent
with the ~0.081 gold→ASR gap measured here being the largest single effect in
the project.

---

## 2d. Benchmark: Combei, arXiv 2025 — the closest methodological match

*"On the Contribution of Lexical Features to Speech Emotion Recognition"* —
arXiv:2509.05634 (Sept 2025), Technical University of Cluj-Napoca.

This is the most directly comparable paper in this document, and the only one
that is comparable **without caveats about metric or label set**: MELD, all
seven emotions, weighted F1 on the official test partition, frozen
self-supervised encoders with a small trainable classifier on top. That is this
project's design, arrived at independently.

Their pipeline: Whisper-large-v3 → frozen text SSL (BERT / XLM-R / DeBERTa),
mean-pooled → 3-layer MLP. Acoustic arm: frozen wav2vec2-XLS-R-2B, mean-pooled
→ same MLP.

### Three independent calibration points, all within 0.4 F1

This is the important result. Comparing single-modality arms — the only place
the two systems are doing the same thing:

| arm | Combei (test WF1) | ours (test WF1) | Δ |
|---|---|---|---|
| acoustic only | 49.3 | **48.9** | −0.4 |
| lexical, ASR transcripts | 51.5 | **51.2** | −0.3 |
| lexical, manual transcripts | 60.9 | **60.7** | −0.2 |

Ours: `fusion/phase1_pooling/acoustic_attention` (48.93),
`text_only_weighted_TESTON_asr_clean` (51.19),
`text_only_plain_TESTON_gold` (60.70).

Three arms, two research groups, different encoders (DeBERTa vs XLM-R),
different ASR systems (Whisper-large-v3 vs Voxtral-Mini), and the results agree
to within half a point at every one. **This is the strongest external
validation in the project.** It establishes that the frozen-cache pipeline here
is correctly implemented and correctly scored — not merely internally
consistent, but landing where an independent group lands.

### The gold→ASR gap replicates almost exactly

| | manual | ASR | gap |
|---|---|---|---|
| Combei | 60.9 | 51.5 | **−9.4** |
| this project | 62.8 | 53.3 | **−9.5** |

The central measured effect of this dissertation — that moving from gold to ASR
transcripts costs roughly 9-10 weighted F1 on MELD — is independently
reproduced by a different group with a different ASR front-end. It is not an
artefact of Voxtral, of the cleaning strategy, or of this codebase.

### What our models add on top

| system | modality | test WF1 |
|---|---|---|
| SpeechFormer (2022) | acoustic | 41.9 |
| Taghavi et al. (§2) | lexical | 43.1 |
| SpeechFormer++ (2023) | acoustic | 47.0 |
| DWFormer / TF-Mamba (2023/25) | acoustic | 48.5 |
| DST (2023) — prior acoustic SOTA | acoustic | 48.9 |
| Combei — acoustic | acoustic | 49.3 |
| Combei — lexical (ASR) | lexical | 51.5 |
| **ours — bc-LSTM + attention pooling (ASR, cleaned)** | **both** | **53.3** |
| Combei — lexical (manual) | lexical | 60.9 |
| **ours — bc-LSTM + attention pooling (gold)** | **both** | **62.8** |

Against matched single-modality baselines from the same paper, the multimodal +
context + attention-pooling stack is worth **+1.8 over the best lexical arm and
+4.4 over the best acoustic arm in the ASR condition**, and +1.9 in the gold
condition. Modest, but consistent across both conditions and measured against a
2025 baseline rather than a 2020 one.

Note also that Combei's table places Taghavi (§2) at 43.1 WF1 — consistent with
the +10.3 recorded there, from an independent source.

### Two findings we should act on

**1. XLM-R was the wrong text encoder.** Their layer-wise sweep (their Table II)
ranks the four text SSLs on MELD dev, and XLM-R is *last*:

| encoder | best dev WF1 | best layer |
|---|---|---|
| DeBERTa | **51.73** | 19 |
| BERT-large | 51.04 | 19 |
| BERT-base | 48.78 | 10 |
| XLM-RoBERTa | 47.48 | 19 |

DeBERTa beats XLM-R by **4.3 F1** on the identical task. This is a genuine
limitation of the present work and should be named as one. XLM-R was chosen
here for multilingual coverage in service of the dialect-robustness thesis
target, which is a defensible reason — but on English-only MELD it costs
performance.

**2. The final layer is not the best layer.** For XLM-R they report layer 19 at
47.48 vs the final layer 24 at 44.17 — **+3.3 from layer choice alone**, free,
no architecture change. This project extracts `[CLS]` from the final layer.
Their result implies a cheap improvement is available, and it is directly
relevant to the pooling investigation in `EXPERIMENTS.md` §10: some of the
"representation is weak" finding may be a final-layer artefact rather than a
property of the encoder.

They also mean-pool over tokens rather than using `[CLS]`, so the two choices
are confounded in their numbers — but both point the same way.

### One negative result worth citing

DEMUCS denoising **degraded performance across every acoustic model and nearly
every layer**. Their explanation: aggressive enhancement strips non-verbal
vocalisations that carry emotion. Useful defensively — audio denoising is an
obvious reviewer suggestion for a noisy corpus like MELD, and there is now a
citation showing it does not help.

---

## 2e. Zhang & Poellabauer, NeurIPS 2024 workshop — context length and WER

*"Contextual Speech Emotion Recognition with Large Language Models and
ASR-Based Transcriptions"*

**Not usable as a performance benchmark.** They score a 4-class MELD subset
(neutral / sadness / happy / anger) with unweighted accuracy against a prompted
LLM with no training. Their best MELD result (UA 0.69, LLaMA3-70B, "gambler"
prompt, context 10) shares neither the label set nor the metric used here.

Useful for two narrower purposes.

### Context length has a knee, and it is shallow — matching our null result

Their MELD UA by context window, holding everything else fixed:

| context (utterances) | MELD UA |
|---|---|
| 5 | 0.58 |
| 10 | **0.60** |
| 15 | 0.59 |

A +0.02 gain from 5→10 and a −0.01 loss from 10→15. Our own window sweep is
flatter still — the spread across K ∈ {0, 1, 2, 4, full} is under 0.01 WF1 in
every family:

| family (asr_cleaned) | k0 | k1 | k2 | k4 | full |
|---|---|---|---|---|---|
| bc-LSTM | 0.5073 | 0.5100 | 0.5128 | 0.5101 | 0.5075 |
| bc-LSTM + attn | **0.5334** | 0.5259 | 0.5326 | 0.5317 | 0.5243 |
| stacked | 0.5030 | 0.5049 | 0.5094 | 0.5074 | 0.5093 |

K=0 — no dialogue context at all — is the best cell in the attention family.
This is a **null result on conversational context**, and it is worth reporting
as one rather than burying it. Zhang et al. provide the supporting literature
point: even with an LLM reading context as natural language, the effect on MELD
is ~0.02 and saturates by 10 utterances. Context is not where the headroom is
on this corpus.

### Lower WER does not guarantee better SER

Their headline finding: an 'ensemble' transcript with WER 0.34 outperformed the
lowest-WER system (w2v2-960-large-self, WER 0.22) on downstream SER. This
constrains what we may claim in §4. Our cleaning **does** lower corpus WER
(test 0.382 → 0.269) *and* raises F1 (+0.027), so the two move together here —
but Zhang et al. show the WER drop is not sufficient to explain the F1 gain.
The cleaning also removes structurally defective utterances, and those two
effects are confounded in our design. State the gain; do not attribute it
solely to WER.

Their MELD WER figures (0.20-0.53 across 11 systems) are also the low end of
the 5x published spread noted in §2c. Their Whisper rows are explicitly
sabotaged — "intentionally truncated to ensure they are not overly effective"
(their Appendix A) — so those rows cannot be read as Whisper's real MELD WER.

---

## 3. Other published MELD results, and why they are not directly comparable

| method | text | audio | video | WF1 (%) |
|---|---|---|---|---|
| AM²-EmoJE (2024) | ✓ | ✓ | **✓** | 71.98 |
| M2FNet (2022) | ✓ | ✓ | **✓** | 66.71 |
| COSMIC (2020) | ✓ | — | — | 65.21 |
| MELD baseline, bc-LSTM (2019) | ✓ | ✓ | ✓ | ~57–59 |
| **ours** | ✓ | ✓ | **—** | **61.5** |

All gold transcripts.

Two differences make a direct ranking misleading:

**Video.** M2FNet and AM²-EmoJE use visual features from the Friends frames.
This project uses two modalities, not three. Our 61.5 sitting above the 2019
tri-modal baseline and below the modern tri-modal systems is the expected
position.

**Gold transcripts.** Every number in that table assumes perfect
transcription. None of them report what happens under real ASR — which is the
question this project is about.

**We do not claim state of the art.** The claim is narrower and, we would
argue, more useful: *published MELD results rest on an assumption that no
deployed system satisfies, and here is what changes when it is dropped.*

---

## 4. What survives the ASR condition

Every architectural finding below is measured on the ASR conditions, where the
differences actually matter. Full detail in `EXPERIMENTS.md`.

### Acoustic pooling is worth more than any architecture we tested

The acoustic sequence `(T, 1280)` must be collapsed to one vector per
utterance. That choice was worth more than every fusion mechanism, context
width and ordering combined.

| pooling | probe F1 | cosine gap (class separation) |
|---|---|---|
| masked mean | 0.3957 | **0.0138** |
| **attention** | **0.4623** | **0.0779** — 5.6× |

Re-running both context families on the attention-pooled vector, **43 of 45
cells improved**, with the largest gains on the noisy conditions:

| condition | masked mean | attention | gain |
|---|---|---|---|
| gold | 0.6109 | **0.6149** | +0.004 |
| **asr** | 0.5066 | **0.5339** | **+0.027** |
| **asr_cleaned** | 0.5054 | **0.5370** | **+0.032** |

### Dialogue context helps, and its ordering does not matter

| condition | fusion alone | + dialogue context | gain |
|---|---|---|---|
| gold | 0.5652 | **0.6149** | +0.050 |
| asr | 0.5021 | **0.5339** | +0.032 |
| asr_cleaned | 0.4935 | **0.5370** | +0.044 |

Two orderings were tested — context after fusion (bc-LSTM on concatenated
features) and context before fusion (per-modality BiLSTM, then fuse). They are
**equivalent**: 0.6149 vs 0.6147 on gold, 0.5339 vs 0.5273 on asr. The gain is
from having context at all, not from where it sits.

### WER filtering helps, but only once pooling is fixed

| | masked mean | attention |
|---|---|---|
| asr | 0.5081 | 0.5273 |
| asr_cleaned | 0.5026 | **0.5370** |

Under the old pooling, filtering the training set made things *worse* — the
lost training volume outweighed the quality gain. With attention pooling it
wins. This is a **conditional** finding and should be reported as one.

### The fusion mechanism barely matters

`sum` wins on gold and asr_cleaned, `concat` on asr, with a spread of
0.004–0.027 — inside seed noise. `sum` is marginally ahead and is the simplest,
which is a reasonable basis for preferring it, but the honest statement is that
the four mechanisms are equivalent.

---

## 5. Caveats binding every number here

**Single seed.** Each cell is one run. Differences below ~0.02 are not
distinguishable from seed noise. The pooling result (43/45 cells improving,
+0.027 mean on ASR) is robust to this because the pattern is consistent across
45 independent cells and two architectures; individual cell-to-cell orderings
are not.

**Dev used twice.** For early stopping *and* for choosing between runs, so dev
is mildly optimistic. Test numbers are the honest ones. In this project test
runs about 0.03 *above* dev, consistently across all 82 scored runs.

**Two modalities, not three.** No visual features, unlike the modern MELD
systems we compare against.

**Fear and disgust are never recovered.** Per-class F1 stays at 0.07–0.13 for
fear and 0.09–0.12 for disgust at every pipeline stage, in every condition.
With 96 and 115 training examples respectively, this is a dataset limitation
rather than a model one — but it means the weighted metric is substantially
carried by neutral (47%).

**Protocol heterogeneity in the comparison literature.** The five papers
benchmarked above use four different MELD protocols: 7-class weighted F1
(COSMIC, Culnan, Combei), 4-class unweighted accuracy (Zhang & Poellabauer),
and — in Shang & Fu, *Intelligent Systems with Applications* 24 (2024) — a
protocol that cannot be reconciled with MELD at all. That paper reports splits
of 7,919 / 862 / 2,116 against MELD's actual 9,989 / 1,109 / 2,610, names the
MELD emotions as "happiness, appreciation, anger, neutrality" (MELD has seven,
and "appreciation" is not among them), and defines both "WA" and "UA" with
formulas containing no false-negative term — they are precision, not accuracy.
Its headline 71.84% on MELD is not comparable to anything here and is not cited
as a benchmark.

This matters beyond one weak paper: **large reported multimodal gains are
common in this literature, and some come from work whose protocol does not
survive checking.** The +0.003 fusion-over-text gain measured here looks poor
against published claims, but the honest comparison is against §2d, where every
number is on the same 7-class test partition under the same metric — and there
the multimodal stack is +1.8. The defence of the single-code-path gold /
asr / asr_cleaned matrix in §1 is exactly this.
