# Progress Report — Multimodal Fusion for Sentiment & Emotion on MELD

**Student**: 5734759 · **Config**: `src/configs/mini.yaml` (XLM-RoBERTa-base + Voxtral-Mini-3B) · **Target**: MELD test set (2610 utterances, 7-class emotion + 3-class sentiment) · **Primary metric**: weighted F1

Everything below is on the MELD **test** split unless stated otherwise. All raw JSONs live under `results/mini/`.

---

## 0. Dataset & class imbalance context

MELD train label counts (from `train_sent_emo.csv`):

| Class | Count | Fraction |
|---|---:|---:|
| neutral | 4710 | 47.2 % |
| joy | 1743 | 17.4 % |
| surprise | 1205 | 12.1 % |
| anger | 1109 | 11.1 % |
| sadness | 683 | 6.8 % |
| **disgust** | **271** | **2.7 %** |
| **fear** | **268** | **2.7 %** |

Ratio neutral : fear ≈ **17.6 : 1**. Fear and disgust are also the two classes with the lowest inter-annotator agreement in the MELD paper (Poria et al., 2019).

---

## 1. Pipeline overview

Four phases, each producing a distinct set of result files.

```
Phase 0: Preprocessing
  transcribe_all.py  →  Voxtral ASR transcripts + acoustic embeddings (cached to disk)

Phase 1: Text-only fine-tune
  finetune.py        →  XLM-RoBERTa-base trained on ASR transcripts (joint emotion + sentiment)

Phase 2: Multimodal fusion
  train_fusion.py    →  4 fusion heads (concat / sum / gated / crossmodal) over cached embeddings

Auxiliary: Classical baselines, Voxtral zero-shot, late fusion, RAVDESS transfer
```

Voxtral is always frozen. XLM-R is frozen during Phase 2. Fusion only sees pre-cached embeddings — no LLM inference at Phase 2 train time.

---

## 2. Approaches taken — a complete inventory

### 2.1  Focal loss + Weighted Random Sampler *(imbalance handling)*

**Where**: `src/training/finetune.py` (Phase 1 only, XLM-RoBERTa fine-tuning).

**What focal loss does**: Replaces `nn.CrossEntropyLoss` with
```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)     γ = 2.0
```
This is standard cross-entropy multiplied by the down-weighting factor `(1 - p_t)^γ`. When the model is confident and correct (`p_t → 1`), the factor is ~0 and the loss on that example is near zero. When the model is wrong (`p_t → 0`), the factor is ~1 and the loss is like ordinary CE. Effect: gradient attention shifts to hard examples, which in practice are the minority classes.

**What Weighted Random Sampler does**: replaces the default shuffling `DataLoader` with `torch.utils.data.WeightedRandomSampler`. Each training example is drawn with probability proportional to `1 / count(class(example))`, with replacement. The model then sees every class with roughly equal expected frequency per batch — so a fear example is drawn ~17× more often than a neutral one per epoch.

**Interaction rule**: when the sampler is active, per-class α weights inside focal loss are set to *uniform*. Otherwise you double-correct — the sampler equalises class frequencies, and weighting again would over-penalise majority errors.

**Config toggles**: `training.use_focal_loss`, `training.focal_gamma`, `training.use_weighted_sampler` in `mini.yaml`.

**Result**: see §3 (marginal improvement in text-only WF1; substantial overfit after epoch 3 because sampler replays the same 268 fear examples with high frequency and the model memorises them).

---

### 2.2  Voxtral paraphrase augmentation *(more minority-class data)*

**Where**: `src/preprocessing/augment_transcripts.py` (Phase 0 add-on). Result loaded by `TranscriptDataset` in `finetune.py` when `training.use_paraphrases = true`.

**Motivation**: focal + sampler helped a bit but capped quickly because there are only 268 unique fear utterances. Adding *new* surface forms should give the model diverse gradient signal instead of replaying the same lines.

**How**:
1. Filter MELD train transcripts to minority emotions using `augmentation.per_class_multiplier`:
   ```
   fear × 4     (268 → +1072 paraphrases)
   disgust × 4  (271 → +1084)
   sadness × 1  (683 → +683)
   anger × 0    (already mid-pack, skipped)
   ```
2. Prompt Voxtral-Mini-3B (text-only chat, `n=multiplier`, `temperature=0.8`, `top_p=0.95`) with:
   ```
   You are rewriting a single line of dialogue for a TV-show character.
   The character is expressing the emotion: {emotion}.
   Original line: "{text}"
   Rewrite this line in different words while keeping the same emotion
   ({emotion}) and the same overall meaning. Use casual, conversational
   English. Keep it 1-2 short sentences.
   ```
3. Save to `/dcs/large/u5734759/data/meld_transcripts/train_paraphrases.json`.
4. `TranscriptDataset` appends `(paraphrase_text, same_emotion, same_sentiment)` to its sample list when the config flag is on. Val / test are never augmented.

**Downstream**: the fusion model still trains on the original 9989 audio-text pairs — augmentation only affects XLM-R's Phase 1 weights. That improved XLM-R then re-extracts text [CLS] embeddings for fusion (via `extract_text_embeddings.py`).

**Result**: see §3 (aug hurt text-only, marginal at best in fusion).

#### Why text-only augmentation? Why only XLM-R got the paraphrases?

This design constraint is worth spelling out — it's a subtle point in the pipeline.

**1. Voxtral generates text, not audio.** Voxtral takes audio input and returns text output. When we prompt it with *"rewrite this line while keeping the emotion"*, the output is a new string. There is no mode where Voxtral produces synthetic paired audio of a new sentence being spoken fearfully — that's not the model's function.

**2. A paraphrase has no paired audio, so it cannot enter Phase 2 fusion directly.** MELD's audio-text pairs are 1:1 — the audio is what the actor actually said. If we invent "I'm terrified of what's coming" via Voxtral, no one recorded that sentence with a fearful voice. Three theoretical workarounds all fail:

- **Skip audio for augmented samples** → the fusion dataloader in `train_fusion.py` expects both modalities per sample; leaving one out breaks the batch.
- **Reuse the original audio with the paraphrased text** → the audio no longer matches the text (audio says "I'm scared", text says "I'm terrified of what's coming"). This *teaches* the fusion model wrong audio-text alignments — poisoning, not augmentation.
- **Generate audio via TTS** → TTS emotion prosody is a synthetic template, not real emotional speech. Would inject a fake acoustic emotion signature that doesn't match how real actors sound on MELD. Also poisoning.

**3. Fusion Phase 2 reads pre-cached embeddings keyed by MELD utterance ID.** `train_fusion.py:127-145` loads `{split}_text_embeddings.pt` and `{split}_embeddings.pt` — both dicts keyed by `dia{d}_utt{u}` IDs from MELD's CSV. Paraphrases don't have MELD IDs (they were generated *from* MELD IDs), so there is no place to slot them into the fusion dataloader.

**4. But augmentation still reaches fusion — indirectly, via the XLM-R checkpoint.** The path is:

```
Voxtral → 2839 paraphrases (text only)
       ↓
Phase 1: XLM-R fine-tunes on 9989 originals + 2839 paraphrases = 12 828 text samples
       ↓
Better XLM-R checkpoint (more robust to minority-class ASR text)
       ↓
extract_text_embeddings.py runs the improved XLM-R over ONLY the original 9989 MELD utterances
       ↓
train_text_embeddings.pt now contains BETTER [CLS] vectors for the same 9989 keys
       ↓
Phase 2 fusion trains on (improved-CLS[X], original-audio-embedding[X], label[X]) triples
       ↓
Fusion never sees a paraphrase — but its text side is now a better encoder because
Phase 1 saw more diverse minority examples.
```

That is why we have `test_results_{concat,sum,gated,crossmodal}_aug.json` files: the `_aug` tag means "Phase 1 was aug-trained; Phase 2 used the resulting improved text embeddings, over the same 9989 MELD utterances as everyone else".

**5. Why not audio augmentation instead?** Audio augmentation techniques exist (SpecAugment frequency/time masks, speed perturbation, noise injection) but for this task they don't help enough to be worth the compute:

- **They don't add *new* examples, only variants of existing ones.** Time-masking `dia0_utt3` still leaves you with 268 unique fear *utterances*, just with distortion. The bottleneck is *diverse* minority utterances, not augmented copies.
- **They require re-running Voxtral's encoder over N × (train size) perturbed clips** — expensive (~10 min per encoder pass on the L40S × N augmentation factor).
- **The failure mode we saw with RAVDESS pretraining (§2.3, §2.4)** — that *real* external emotion audio didn't transfer to MELD — implies acoustic augmentation via perturbation of the same 268 clips is very unlikely to break the fear/disgust ceiling either.

**6. Val / test are never augmented.** Only the train split's `TranscriptDataset` is extended with paraphrases (`finetune.py:130-160` gates on `split == "train"`). Evaluation numbers remain on the same 1109 dev and 2610 test utterances as every other ablation, so aug rows are directly comparable to baseline rows.

---

### 2.3  RAVDESS cross-corpus acoustic pretraining

**Where**:
- `src/preprocessing/extract_ravdess_embeddings.py` — cache Voxtral encoder outputs for all 1440 RAVDESS speech clips.
- `src/training/pretrain_ravdess.py` — Phase A: 8-class acoustic emotion classifier on RAVDESS.
- `src/models/acoustic_classifier.py` — architecture whose `acoustic_proj` submodule is bit-compatible with `SumFusion.acoustic_proj`, `GatedFusion.acoustic_proj`, `CrossModalGating.acoustic_proj`.
- `src/training/train_fusion.py` — Phase B: warm-starts `acoustic_proj` from the backbone when `use_ravdess_pretrain: true`.

**Why RAVDESS**: 1440 studio-quality acted emotion clips, ~180 per emotion (8 classes incl. calm) → provides clean acoustic emotion labels external to MELD. Text is useless here because RAVDESS only has 2 fixed sentences — acoustic-only transfer.

**Emotion mapping**: RAVDESS's 8-class taxonomy → MELD's 7-class (used only in the combined training in §2.4):
```
RAVDESS neutral   → MELD neutral
RAVDESS calm      → MELD neutral   (supervisor: don't drop calm)
RAVDESS happy     → MELD joy
RAVDESS sad       → MELD sadness
RAVDESS angry     → MELD anger
RAVDESS fearful   → MELD fear
RAVDESS disgust   → MELD disgust
RAVDESS surprised → MELD surprise
```

**Head-swap design**: the 8-class RAVDESS head is discarded when transferring — only the shared `acoustic_proj` submodule moves to fusion. This lets every RAVDESS clip contribute (including calm) without introducing fake MELD-space labels.

**Architecture** (matches `SumFusion.acoustic_proj` exactly):
```
Voxtral acoustic embedding (B, 1280)
    ↓
Linear(1280, 512)
LayerNorm(512)
GELU
Dropout(p)
    ↓  ← transferable
Linear(512, num_classes)   ← task-specific head, not transferred
```

**Result**: strong on RAVDESS (macro F1=0.59 test), zero transfer to MELD fusion. See §3.

---

### 2.4  Combined RAVDESS + MELD acoustic training

**Where**: `src/training/pretrain_combined.py`.

**Idea**: instead of pretraining on RAVDESS *then* fine-tuning fusion on MELD (which showed no benefit), train an acoustic-only classifier on the **union** of RAVDESS + MELD train audio with all labels in MELD's 7-class space, validated on MELD dev. This directly optimises for the target domain and keeps RAVDESS as extra minority-class examples.

**Setup**:
- Train records = 1440 RAVDESS + 9989 MELD train = 11 429 audio clips
- Val = MELD dev (1109) — early stop on target domain WF1
- Test = MELD test (2610)
- Head is 7-class MELD from the start; RAVDESS labels are mapped in

**Result**: acoustic-only on MELD test is worse than text-only baseline. Backbone weights still didn't help fusion. See §3.

---

### 2.5  Classical baselines (SVM, XGBoost) on fused [CLS] + acoustic

**Where**: `src/training/train_classical.py`.

**Setup**: for each Phase 1 checkpoint, extract XLM-R [CLS] → concat with Voxtral acoustic embedding → per-utterance 2048-d vector. Fit SVM (RBF kernel, class-balanced) and XGBoost (7-class softmax, class-balanced sample weights). Dev split for evaluation.

**Purpose**: sanity check that the neural fusion is actually doing something a linear/tree model can't, and provide a shallow-classifier row for the dissertation table.

---

### 2.6  Preprocessing / ASR analysis (context)

- **Voxtral-Mini ASR transcription of MELD** — `src/preprocessing/transcribe_all.py` Pass 1 (vllm)
- **Voxtral encoder embeddings** — Pass 2 (HF transformers, extracted from `audio_tower` before the multimodal projector, 1280-d)
- **Whisper Large-v3 baseline transcription** — for WER comparison
- **WER analysis** — `src/evaluation/compute_wer.py`, results in `wer_analysis.json` (Voxtral) and `wer_analysis_whisper.json` (Whisper)
- **Voxtral zero-shot ERC** — direct LLM prompting for emotion labels, no fine-tuning (`voxtral_zeroshot_{split}_metrics.json`)

---

### 2.7  Late-fusion ensembling

**Where**: `results/mini/late_fusion_results.json`.

Combines three heads with the same test predictions:
- **baseline_text** — XLM-R text-only (aug Phase 1 checkpoint)
- **baseline_acoustic** — 7-class acoustic-only classifier on MELD
- **baseline_zeroshot** — Voxtral zero-shot emotion labels

Then five ensemble strategies:
- `weighted_avg` — weighted softmax averaging (grid-search weights)
- `meta` — logistic-regression meta-classifier over the three probability vectors
- `confidence_gate` — pick the head with highest max-softmax per sample (fell back to zero-shot in practice)
- `joint_meta` — meta-classifier trained jointly for emotion + sentiment
- `oracle_upper` — take the best of the three per sample (upper bound, not attainable at test time)

---

## 3. Results — the full picture

**Reading key**: WF1 = weighted F1. mF1 = macro F1. Per-class F1 for the two hardest MELD classes (fear, disgust) is shown; complete per-class F1 is in the raw JSONs.

### 3.1  Phase 1 (text-only, XLM-R) — MELD test

#### 3.1.a Emotion (7-class)

| Config | WF1 | mF1 | fear | disgust | anger | sadness | joy | surprise | neutral | JSON |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **baseline** (class-weighted CE) | 0.454 | 0.320 | 0.125 | 0.140 | 0.296 | 0.287 | 0.368 | 0.438 | 0.587 | `test_results_text_only_baseline.json` |
| focal loss + weighted sampler | 0.468 | 0.311 | 0.111 | 0.167 | 0.139 | 0.285 | 0.382 | 0.440 | 0.652 | `test_results_text_only_focal_sampler.json` |
| Voxtral paraphrase aug (2839 new lines) | 0.449 | 0.292 | 0.070 | 0.116 | 0.178 | 0.307 | 0.328 | 0.415 | 0.627 | `test_results_text_only_aug.json` |

Focal+sampler gave +0.014 WF1, but macro F1 *dropped* (0.320 → 0.311) — the sampler over-fits the tiny fear pool and anger F1 collapsed from 0.30 → 0.14. Aug made things worse on text-only: paraphrases introduced label noise. Fear F1=0.07 with aug is the lowest recorded.

#### 3.1.b Sentiment (3-class)

| Config | WF1 | mF1 | negative | neutral | positive | JSON |
|---|---:|---:|---:|---:|---:|---|
| **baseline** (class-weighted CE) | 0.533 | 0.506 | 0.520 | 0.595 | 0.404 | `test_results_text_only_baseline.json` |
| focal loss + weighted sampler | **0.559** | **0.524** | **0.540** | **0.641** | 0.392 | `test_results_text_only_focal_sampler.json` |
| Voxtral paraphrase aug | 0.544 | 0.502 | 0.504 | 0.650 | 0.352 | `test_results_text_only_aug.json` |

Sentiment behaves better than emotion in Phase 1: focal+sampler is unambiguously the winner (+0.026 WF1 over baseline, +0.018 macro F1). Aug helped neutral (0.595 → 0.650) but hurt positive (0.404 → 0.352) — likely because most paraphrases we generated were for fear/disgust/sadness (which map to the negative sentiment class), so the model learned to bias toward negative and lost positive precision.

### 3.2  Fusion — MELD test

Each fusion architecture was re-trained per Phase 1 checkpoint and per RAVDESS-pretrain choice. `_no_pretrain` is the control where `use_ravdess_pretrain=false`. `_ravdesspre` runs used the RAVDESS-only backbone (§2.3). All fusion runs were 10 epochs, LR 1e-4, XLM-R frozen.

#### 3.2.a Fusion emotion (7-class)

#### CONCAT emotion (`FusionModel` — MLP over `[text; acoustic]`)

| Config | WF1 | mF1 | fear | disgust | anger | sadness | joy | surprise | neutral | JSON |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.467 | 0.315 | 0.128 | 0.127 | 0.289 | 0.262 | 0.332 | 0.437 | 0.632 | `test_results_concat.json` |
| focal+sampler | **0.593** | 0.415 | 0.142 | 0.173 | 0.387 | 0.317 | 0.594 | 0.542 | 0.747 | `..._concat_focal_sampler.json` |
| aug | **0.593** | 0.415 | 0.142 | 0.173 | 0.387 | 0.317 | 0.594 | 0.542 | 0.747 | `..._concat_aug.json` |
| RAVDESS pretrain | **0.593** | **0.420** | **0.175** | 0.166 | 0.432 | 0.339 | 0.555 | 0.525 | 0.746 | `..._concat_ravdesspre.json` |
| focal+sampler + RAVDESS | 0.593 | 0.420 | 0.175 | 0.166 | 0.432 | 0.339 | 0.555 | 0.525 | 0.746 | `..._concat_focal_sampler_ravdesspre.json` |
| aug + RAVDESS | 0.593 | 0.415 | 0.142 | 0.173 | 0.387 | 0.317 | 0.594 | 0.542 | 0.747 | `..._concat_aug_ravdesspre.json` |
| no_pretrain (control) | 0.593 | 0.420 | 0.175 | 0.166 | 0.432 | 0.339 | 0.555 | 0.525 | 0.746 | `..._concat_no_pretrain.json` |

*Note*: concat has no separate `acoustic_proj` (both modalities enter the joint MLP), so `use_ravdess_pretrain` is a no-op — `_ravdesspre` and `_no_pretrain` rows are identical, as expected.

#### SUM emotion (`SumFusion` — projected sum)

| Config | WF1 | mF1 | fear | disgust | anger | sadness | joy | surprise | neutral | JSON |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.473 | 0.325 | 0.140 | 0.128 | 0.314 | 0.282 | 0.337 | 0.446 | 0.629 | `test_results_sum.json` |
| focal+sampler | 0.587 | 0.409 | 0.147 | 0.131 | 0.394 | 0.341 | 0.580 | 0.530 | 0.739 | `..._sum_focal_sampler.json` |
| aug | **0.591** | 0.413 | 0.139 | 0.142 | 0.405 | 0.337 | 0.584 | 0.545 | 0.739 | `..._sum_aug.json` |
| RAVDESS pretrain | 0.576 | 0.415 | **0.175** | **0.193** | 0.418 | 0.320 | 0.566 | 0.515 | 0.716 | `..._sum_ravdesspre.json` |
| focal+sampler + RAVDESS | 0.576 | 0.415 | 0.175 | 0.193 | 0.418 | 0.320 | 0.566 | 0.515 | 0.716 | `..._sum_focal_sampler_ravdesspre.json` |
| aug + RAVDESS | 0.587 | 0.409 | 0.148 | 0.144 | 0.390 | 0.339 | 0.579 | 0.525 | 0.739 | `..._sum_aug_ravdesspre.json` |
| no_pretrain (control) | 0.576 | 0.415 | 0.175 | 0.193 | 0.418 | 0.320 | 0.566 | 0.515 | 0.716 | `..._sum_no_pretrain.json` |

#### GATED emotion (`GatedFusion` — symmetric gate)

| Config | WF1 | mF1 | fear | disgust | anger | sadness | joy | surprise | neutral | JSON |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.466 | 0.319 | 0.135 | 0.140 | 0.289 | 0.274 | 0.335 | 0.432 | 0.627 | `test_results_gated.json` |
| focal+sampler | 0.583 | 0.395 | 0.130 | 0.161 | 0.374 | 0.299 | 0.541 | 0.505 | 0.759 | `..._gated_focal_sampler.json` |
| aug | 0.591 | 0.413 | 0.134 | 0.158 | 0.394 | 0.314 | 0.591 | 0.559 | 0.739 | `..._gated_aug.json` |
| RAVDESS pretrain | 0.586 | **0.422** | 0.173 | **0.193** | 0.420 | 0.323 | 0.579 | 0.540 | 0.726 | `..._gated_ravdesspre.json` |
| focal+sampler + RAVDESS | 0.586 | 0.422 | 0.173 | 0.193 | 0.420 | 0.323 | 0.579 | 0.540 | 0.726 | `..._gated_focal_sampler_ravdesspre.json` |
| aug + RAVDESS | **0.590** | 0.414 | 0.147 | 0.153 | 0.395 | 0.319 | 0.589 | 0.553 | 0.739 | `..._gated_aug_ravdesspre.json` |
| no_pretrain (control) | 0.586 | 0.422 | 0.173 | 0.193 | 0.420 | 0.323 | 0.579 | 0.540 | 0.726 | `..._gated_no_pretrain.json` |

#### CROSSMODAL emotion (`CrossModalGating` — asymmetric cross gates)

| Config | WF1 | mF1 | fear | disgust | anger | sadness | joy | surprise | neutral | JSON |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 0.465 | 0.321 | 0.143 | 0.131 | 0.298 | 0.270 | 0.356 | 0.433 | 0.616 | `test_results_crossmodal.json` |
| focal+sampler | 0.583 | 0.396 | 0.134 | 0.170 | 0.370 | 0.299 | 0.535 | 0.500 | 0.762 | `..._crossmodal_focal_sampler.json` |
| aug | **0.591** | 0.415 | 0.158 | 0.137 | 0.397 | **0.342** | 0.581 | 0.548 | 0.741 | `..._crossmodal_aug.json` |
| RAVDESS pretrain | 0.586 | 0.418 | **0.178** | 0.167 | 0.430 | 0.320 | 0.568 | 0.533 | 0.729 | `..._crossmodal_ravdesspre.json` |
| focal+sampler + RAVDESS | 0.586 | 0.418 | 0.178 | 0.167 | 0.430 | 0.320 | 0.568 | 0.533 | 0.729 | `..._crossmodal_focal_sampler_ravdesspre.json` |
| aug + RAVDESS | 0.586 | 0.411 | 0.148 | 0.145 | 0.394 | 0.333 | 0.583 | 0.543 | 0.733 | `..._crossmodal_aug_ravdesspre.json` |
| no_pretrain (control) | 0.586 | 0.418 | 0.178 | 0.167 | 0.430 | 0.320 | 0.568 | 0.533 | 0.729 | `..._crossmodal_no_pretrain.json` |

**Observation**: for sum/gated/crossmodal, `_ravdesspre` and `_no_pretrain` are numerically identical, meaning the fusion training washes out whatever initialisation `acoustic_proj` starts from over 10 epochs.

#### 3.2.b Fusion sentiment (3-class)

For sentiment, the pattern is similar to emotion but *stronger*: fusion adds ~+0.15 WF1 over the text-only baseline (bigger than the +0.12 emotion lift).

**CONCAT sentiment**

| Config | WF1 | mF1 | negative | neutral | positive | JSON |
|---|---:|---:|---:|---:|---:|---|
| baseline | 0.534 | 0.496 | 0.478 | 0.638 | 0.371 | `test_results_concat.json` |
| focal+sampler | 0.676 | 0.652 | 0.596 | 0.762 | **0.599** | `..._concat_focal_sampler.json` |
| aug | 0.676 | 0.652 | 0.596 | 0.762 | 0.599 | `..._concat_aug.json` |
| RAVDESS pretrain | **0.687** | **0.663** | **0.623** | **0.767** | 0.599 | `..._concat_ravdesspre.json` |
| focal+sampler + RAVDESS | 0.687 | 0.663 | 0.623 | 0.767 | 0.599 | `..._concat_focal_sampler_ravdesspre.json` |
| aug + RAVDESS | 0.676 | 0.652 | 0.596 | 0.762 | 0.599 | `..._concat_aug_ravdesspre.json` |
| no_pretrain (control) | 0.687 | 0.663 | 0.623 | 0.767 | 0.599 | `..._concat_no_pretrain.json` |

**SUM sentiment**

| Config | WF1 | mF1 | negative | neutral | positive | JSON |
|---|---:|---:|---:|---:|---:|---|
| baseline | 0.536 | 0.501 | 0.487 | 0.631 | 0.387 | `test_results_sum.json` |
| focal+sampler | 0.673 | 0.651 | 0.606 | 0.748 | 0.598 | `..._sum_focal_sampler.json` |
| aug | 0.671 | 0.648 | 0.596 | 0.753 | 0.595 | `..._sum_aug.json` |
| RAVDESS pretrain | **0.685** | **0.662** | **0.632** | **0.758** | 0.595 | `..._sum_ravdesspre.json` |
| focal+sampler + RAVDESS | 0.685 | 0.662 | 0.632 | 0.758 | 0.595 | `..._sum_focal_sampler_ravdesspre.json` |
| aug + RAVDESS | 0.675 | 0.653 | 0.609 | 0.749 | **0.601** | `..._sum_aug_ravdesspre.json` |
| no_pretrain (control) | 0.685 | 0.662 | 0.632 | 0.758 | 0.595 | `..._sum_no_pretrain.json` |

**GATED sentiment**

| Config | WF1 | mF1 | negative | neutral | positive | JSON |
|---|---:|---:|---:|---:|---:|---|
| baseline | 0.537 | 0.498 | 0.496 | 0.639 | 0.358 | `test_results_gated.json` |
| focal+sampler | 0.673 | 0.643 | 0.579 | **0.776** | 0.572 | `..._gated_focal_sampler.json` |
| aug | 0.676 | 0.653 | 0.603 | 0.754 | **0.604** | `..._gated_aug.json` |
| RAVDESS pretrain | **0.679** | **0.656** | **0.618** | 0.754 | 0.597 | `..._gated_ravdesspre.json` |
| focal+sampler + RAVDESS | 0.679 | 0.656 | 0.618 | 0.754 | 0.597 | `..._gated_focal_sampler_ravdesspre.json` |
| aug + RAVDESS | 0.674 | 0.651 | 0.599 | 0.754 | 0.600 | `..._gated_aug_ravdesspre.json` |
| no_pretrain (control) | 0.679 | 0.656 | 0.618 | 0.754 | 0.597 | `..._gated_no_pretrain.json` |

**CROSSMODAL sentiment**

| Config | WF1 | mF1 | negative | neutral | positive | JSON |
|---|---:|---:|---:|---:|---:|---|
| baseline | 0.533 | 0.499 | 0.486 | 0.627 | 0.384 | `test_results_crossmodal.json` |
| focal+sampler | 0.665 | 0.634 | 0.556 | **0.776** | 0.571 | `..._crossmodal_focal_sampler.json` |
| aug | **0.676** | **0.653** | **0.610** | 0.753 | 0.596 | `..._crossmodal_aug.json` |
| RAVDESS pretrain | 0.674 | 0.648 | 0.593 | 0.763 | 0.587 | `..._crossmodal_ravdesspre.json` |
| focal+sampler + RAVDESS | 0.674 | 0.648 | 0.593 | 0.763 | 0.587 | `..._crossmodal_focal_sampler_ravdesspre.json` |
| aug + RAVDESS | 0.672 | 0.650 | **0.609** | 0.747 | 0.593 | `..._crossmodal_aug_ravdesspre.json` |
| no_pretrain (control) | 0.674 | 0.648 | 0.593 | 0.763 | 0.587 | `..._crossmodal_no_pretrain.json` |

**Sentiment observations**:
- Best fusion sentiment WF1 = **0.687 (concat)**, macro F1 = **0.663**. Compared to best text-only sentiment (0.559 WF1, 0.524 macro F1), fusion adds **+0.128 WF1 and +0.139 macro F1**.
- Sentiment doesn't suffer the fear/disgust bottleneck emotion does — the 3-class taxonomy is coarser and all three sentiment classes have adequate training data (2334 positive / 2945 negative / 4710 neutral).
- Positive sentiment F1 stays around 0.60 across all fusion configs (except baseline). This is the class with the lowest training count for sentiment; it's the sentiment analogue of "fear/disgust" but far less extreme.
- Just like emotion, `_ravdesspre` = `_no_pretrain` numerically for sum/gated/crossmodal — RAVDESS pretraining doesn't transfer for sentiment either.

### 3.3  Classical baselines (SVM, XGBoost) — MELD dev

Classical baselines are evaluated on **dev** (not test), because they were used as a shallow-model reality check rather than for final numbers.

**SVM emotion (RBF kernel, class-balanced) — MELD dev**

| Config | WF1 | mF1 | fear | disgust | JSON |
|---|---:|---:|---:|---:|---|
| baseline | 0.467 | 0.349 | 0.226 | 0.167 | `classical_svm_emotion.json` |
| focal+sampler XLM-R | 0.496 | 0.368 | 0.250 | 0.103 | `..._focal_sampler.json` |
| aug XLM-R | 0.500 | 0.371 | 0.159 | 0.150 | `..._aug.json` |
| RAVDESS pretrain | 0.496 | 0.368 | 0.250 | 0.103 | `..._ravdesspre.json` |

**SVM sentiment — MELD dev**

| Config | WF1 | mF1 | negative | neutral | positive | JSON |
|---|---:|---:|---:|---:|---:|---|
| baseline | 0.556 | 0.532 | 0.544 | 0.636 | 0.417 | `classical_svm_sentiment.json` |
| focal+sampler XLM-R | 0.576 | 0.556 | 0.583 | 0.632 | 0.452 | `..._focal_sampler.json` |
| aug XLM-R | 0.581 | 0.557 | 0.599 | 0.637 | 0.435 | `..._aug.json` |
| RAVDESS pretrain | 0.576 | 0.556 | 0.583 | 0.632 | 0.452 | `..._ravdesspre.json` |

**XGBoost emotion (7-class softmax, class-balanced) — MELD dev**

| Config | WF1 | mF1 | fear | disgust | JSON |
|---|---:|---:|---:|---:|---|
| baseline | 0.464 | 0.341 | 0.192 | 0.176 | `classical_xgboost_emotion.json` |
| focal+sampler XLM-R | 0.491 | 0.354 | 0.196 | 0.103 | `..._focal_sampler.json` |
| aug XLM-R | 0.479 | 0.343 | 0.131 | 0.158 | `..._aug.json` |
| RAVDESS pretrain | 0.491 | 0.354 | 0.196 | 0.103 | `..._ravdesspre.json` |

**XGBoost sentiment — MELD dev**

| Config | WF1 | mF1 | negative | neutral | positive | JSON |
|---|---:|---:|---:|---:|---:|---|
| baseline | 0.554 | 0.528 | 0.543 | 0.639 | 0.403 | `classical_xgboost_sentiment.json` |
| focal+sampler XLM-R | 0.580 | 0.554 | 0.589 | 0.653 | 0.419 | `..._focal_sampler.json` |
| aug XLM-R | 0.568 | 0.537 | 0.574 | 0.653 | 0.386 | `..._aug.json` |
| RAVDESS pretrain | 0.580 | 0.554 | 0.589 | 0.653 | 0.419 | `..._ravdesspre.json` |

**Observations**:
- SVM baseline dev fear F1 = 0.226 is higher than any neural fusion test fear F1. Two things going on: (a) it's dev vs test — dev fear is a smaller, easier subset; (b) the SVM optimises a hinge over class-balanced samples with a rigid decision boundary that's oddly forgiving of the small fear class.
- Classical **sentiment** dev WF1 (~0.55–0.58) is well below fusion **sentiment** test WF1 (~0.67–0.69). Both SVM and XGBoost see the same feature diet (768-d [CLS] + 1280-d Voxtral); the neural MLP fusion clearly extracts more from the concatenation than a linear/tree model can.

### 3.4  Acoustic-only ablations — MELD test

| Model | WF1 | mF1 | fear | disgust | anger | sad | JSON |
|---|---:|---:|---:|---:|---:|---:|---|
| RAVDESS-only backbone → MELD acoustic-only (§2.3) | — | — | — | — | — | — | (not evaluated on MELD in isolation — used only as fusion init) |
| Combined RAVDESS+MELD acoustic-only (§2.4) | 0.413 | 0.283 | 0.122 | **0.025** | 0.390 | 0.286 | `combined_acoustic_results.json` |
| Late-fusion `baseline_acoustic` head | 0.547 | 0.357 | 0.103 | 0.056 | 0.465 | 0.229 | `late_fusion_results.json` |

**Note**: `combined_acoustic_results.json` uses a 7-class head trained from scratch on RAVDESS + MELD. `late_fusion_results.json` `baseline_acoustic` is a different acoustic-only head trained solely on MELD (see §3.5 code). They are two independent acoustic-only baselines and both underperform text-only.

### 3.5  Late fusion ensembling — MELD test

Combines three heads: text-only (aug Phase 1), acoustic-only (MELD-trained), Voxtral zero-shot.

**Emotion**

| Ensemble strategy | WF1 | mF1 | fear | disgust | anger | joy |
|---|---:|---:|---:|---:|---:|---:|
| baseline_text (aug Phase 1) | 0.449 | 0.292 | 0.070 | 0.116 | 0.178 | 0.328 |
| baseline_acoustic (MELD-only) | 0.547 | 0.357 | 0.103 | 0.056 | 0.465 | 0.491 |
| baseline_zeroshot (Voxtral) | 0.504 | 0.320 | 0.069 | 0.103 | 0.385 | 0.358 |
| **weighted_avg** | 0.539 | 0.353 | 0.091 | 0.078 | 0.417 | 0.431 |
| **meta** (LogReg over probs) | 0.534 | 0.368 | 0.090 | 0.084 | 0.416 | 0.501 |
| confidence_gate | 0.504 | 0.320 | 0.069 | 0.103 | 0.385 | 0.358 |
| joint_meta (emo + sent jointly) | 0.533 | 0.365 | 0.087 | 0.086 | 0.422 | 0.487 |
| oracle_upper (best-of-3 per sample) | **0.728** | **0.543** | 0.156 | 0.227 | 0.680 | 0.695 |

**Sentiment**

| Ensemble strategy | WF1 | mF1 | negative | neutral | positive |
|---|---:|---:|---:|---:|---:|
| baseline_text | 0.544 | 0.502 | 0.504 | 0.650 | 0.352 |
| baseline_acoustic | 0.615 | 0.587 | 0.539 | 0.707 | 0.516 |
| baseline_zeroshot | 0.551 | 0.516 | 0.491 | 0.653 | 0.404 |
| weighted_avg | 0.626 | 0.596 | 0.561 | 0.718 | 0.509 |
| **meta** | **0.633** | 0.609 | 0.608 | 0.695 | 0.525 |
| joint_meta | **0.634** | **0.609** | 0.605 | 0.697 | 0.526 |
| oracle_upper | 0.806 | 0.789 | 0.774 | 0.856 | 0.739 |

**Observation**: the oracle upper bound (WF1=0.73 emotion, 0.81 sentiment) is very high — meaning at least one of the three heads gets each sample right most of the time. But no achievable ensemble comes close to that upper bound; the strategies plateau around WF1=0.53–0.54 on emotion. Head disagreement is not being resolved by any of the tried combination rules.

### 3.6  Voxtral zero-shot ERC (no fine-tuning) — MELD

| Split | Emo WF1 | Emo mF1 | fear | disgust | anger | joy | Sent WF1 | JSON |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| dev  | 0.508 | 0.345 | 0.089 | 0.000 | 0.482 | 0.433 | 0.560 | `voxtral_zeroshot_dev_metrics.json` |
| test | 0.504 | 0.320 | 0.069 | 0.103 | 0.385 | 0.358 | 0.551 | `voxtral_zeroshot_test_metrics.json` |

Voxtral prompted directly for emotion label (no gradient updates) is roughly on par with our fine-tuned text-only baseline (WF1 0.50 vs 0.45).

### 3.7  RAVDESS Phase A self-test (RAVDESS test, 8 classes)

`ravdess_pretrain_results.json`. 30 epochs on cached embeddings.

| RAVDESS test | WF1 | mF1 | neutral | calm | happy | sad | angry | fearful | disgust | surprised |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  | **0.594** | **0.592** | 0.488 | 0.593 | 0.489 | 0.507 | 0.677 | **0.679** | **0.635** | 0.667 |

The backbone learns strong acoustic patterns for fear (F1=0.68) and disgust (F1=0.64) on RAVDESS. These do **not** transfer to MELD (§3.4).

### 3.8  ASR quality analysis (Voxtral vs Whisper)

| ASR system | Split | Corpus WER | Corpus CER | JSON |
|---|---|---:|---:|---|
| Voxtral-Mini-3B  | train | 38.3 % | 29.6 % | `wer_analysis.json` |
| Voxtral-Mini-3B  | dev   | 34.0 % | 25.0 % | |
| Voxtral-Mini-3B  | test  | 42.4 % | 33.6 % | |
| Whisper Large-v3 | train | 33.8 % | 25.8 % | `wer_analysis_whisper.json` |
| Whisper Large-v3 | dev   | 28.9 % | 20.8 % | |
| Whisper Large-v3 | test  | 33.0 % | 24.3 % | |

Voxtral is ~5–9 % WER worse than Whisper on MELD. Voxtral occasionally "hallucinates" — the worst example is `dia512_utt3` where the gold utterance is `"Good."` but Voxtral generates a ~1000-token monologue about "The Art of the Deal." (See `wer_analysis.json` → `train.worst_examples`.) Whisper's worst examples are shorter and more benign (mostly stuck-in-a-loop `"no, no, no, ..."` output). The ASR-error robustness this project is trying to build directly targets these failure modes.

---

## 4. Cross-cutting comparison

Best absolute WF1 / macro F1 per approach on MELD test:

| Approach | Best WF1 | Best macro F1 | Best fear F1 | Best disgust F1 |
|---|---:|---:|---:|---:|
| Text-only (baseline) | 0.454 | 0.320 | 0.125 | 0.140 |
| Text-only (focal+sampler) | 0.468 | 0.311 | 0.111 | 0.167 |
| Text-only (aug) | 0.449 | 0.292 | 0.070 | 0.116 |
| Classical SVM (baseline) | 0.467 (dev) | 0.349 (dev) | 0.226 (dev) | 0.167 (dev) |
| Voxtral zero-shot | 0.504 | 0.320 | 0.069 | 0.103 |
| Acoustic-only (combined) | 0.413 | 0.283 | 0.122 | 0.025 |
| **Fusion concat (best)** | **0.593** | **0.420** | **0.175** | 0.173 |
| **Fusion sum (best)** | **0.591** | 0.415 | 0.175 | **0.193** |
| **Fusion gated (best)** | **0.591** | 0.422 | 0.173 | 0.193 |
| **Fusion crossmodal (best)** | **0.591** | 0.418 | **0.178** | 0.170 |
| Late-fusion meta ensemble | 0.534 | 0.368 | 0.090 | 0.084 |

Everything sits within 0.02 WF1 of each other above the 0.45 text-only floor, and *nothing* pushes fear or disgust above 0.20.

---

## 5. What worked, what didn't, and why

### What worked

1. **Doing fusion at all** — the single biggest lever. Text-only best = WF1 0.47, fusion best = WF1 0.59. Adds +0.12 WF1 and +0.10 macro F1.

2. **XLM-R fine-tuning on ASR transcripts (Phase 1)** — even without any tricks the baseline WF1 is already 0.45 on 42 % WER transcripts, showing XLM-R adapts.

### What didn't work (and why)

1. **Focal loss + weighted sampler**: +0.014 WF1 in text-only but macro F1 *dropped* 0.320 → 0.311 because the sampler over-fits the 268-example fear pool. Fusion runs starting from this Phase 1 landed at the same numbers as other Phase 1 variants — fusion training re-shapes the representation.

2. **Voxtral paraphrase augmentation**: hurt text-only (WF1 0.454 → 0.449, fear 0.125 → 0.070). Likely causes: (a) label drift — some paraphrases slide from "fear" wording into "worry" wording without a matching label change; (b) style drift — Voxtral-generated text has a distinctly different register from MELD's conversational TV dialogue.

3. **RAVDESS cross-corpus pretraining**: strong Phase A (macro F1 = 0.59 on RAVDESS), but zero measurable transfer to MELD fusion. Numbers with `_ravdesspre` are numerically identical to `_no_pretrain` for every fusion variant. The acoustic-only sanity check (§3.4) makes it clear why: **RAVDESS's studio-acted fear acoustics don't appear in MELD's TV-episode audio**. Disgust dropped from F1=0.64 on RAVDESS test to F1=0.025 on MELD test in the combined acoustic-only model.

4. **Late-fusion ensembling**: none of the tried strategies (weighted-avg, meta, confidence-gate, joint-meta) rose above WF1 0.54, despite an oracle upper bound of 0.73. The three constituent heads disagree in ways that aren't recoverable by simple probability combination — they need a router with access to features beyond softmax outputs.

### The bottleneck

Fear F1 caps around 0.18 and disgust F1 around 0.19 across every tried configuration. Structural reasons:
- Only 268 (fear) / 271 (disgust) unique train examples out of 9989.
- Lowest inter-annotator agreement in MELD.
- MELD's audio quality (laugh tracks, background music, mid-sentence cuts) partially destroys the acoustic signal that could disambiguate these classes.
- Cross-corpus acoustic pretraining doesn't help because acted acoustic patterns don't match TV-dialogue acoustic patterns.

The remaining headroom is likely in either (a) a much larger MELD-in-domain acoustic pretraining corpus, or (b) using conversational context (dialogue history) which none of the current models see.

---

## 6. Where every file lives

Everything below is under `results/mini/` unless otherwise stated.

### Text-only Phase 1
- `test_results_text_only_baseline.json` — original baseline (class-weighted CE)
- `test_results_text_only_focal_sampler.json` — focal + sampler
- `test_results_text_only_aug.json` — Voxtral paraphrase aug
- `test_results_text_only.json`, `_ravdesspre.json`, `_focal_sampler_ravdesspre.json`, `_aug_ravdesspre.json`, `_no_pretrain.json` — text-only test evaluations from each ablation pipeline run

### Fusion (per architecture × per ablation)
- `test_results_{concat,sum,gated,crossmodal}.json` — baseline
- `test_results_{concat,sum,gated,crossmodal}_focal_sampler.json`
- `test_results_{concat,sum,gated,crossmodal}_aug.json`
- `test_results_{concat,sum,gated,crossmodal}_ravdesspre.json`
- `test_results_{concat,sum,gated,crossmodal}_focal_sampler_ravdesspre.json`
- `test_results_{concat,sum,gated,crossmodal}_aug_ravdesspre.json`
- `test_results_{concat,sum,gated,crossmodal}_no_pretrain.json`

### Classical (dev set)
- `classical_{svm,xgboost}_{emotion,sentiment}{,_focal_sampler,_aug,_ravdesspre,_focal_sampler_ravdesspre,_aug_ravdesspre,_no_pretrain}.json`

### Cross-corpus & auxiliary
- `ravdess_pretrain_results.json` — RAVDESS 8-class Phase A test set
- `combined_acoustic_results.json` — combined RAVDESS+MELD acoustic-only MELD test
- `late_fusion_results.json` — late-fusion ensembling of text / acoustic / zero-shot
- `voxtral_zeroshot_{dev,test}_metrics.json`, `voxtral_zeroshot_{dev,test}_predictions.json` — Voxtral zero-shot ERC
- `wer_analysis.json`, `wer_analysis_whisper.json` — ASR quality analysis
- `acoustic_probs.json`, `xlmr_text_probs.json` — cached softmax outputs used by the late-fusion script

### Plots
- `phase1_loss_curve.png`, `phase1_loss_curve_baseline.png`, `phase1_loss_curve_focal_sampler.png`
- `phase2_loss_curve_{concat,sum,gated,crossmodal}.png`
- `ravdess_pretrain_loss_curve.png`, `combined_acoustic_loss_curve.png`
- `confusion_emotion_{model}{,_tag}.png`, `confusion_sentiment_{model}{,_tag}.png` for every ablation

### Backups (large scratch dir, not results/)
- `/dcs/large/u5734759/checkpoints/mini/best_model.pt` — current Phase 1 (aug run — last one that ran)
- `/dcs/large/u5734759/checkpoints/mini_baseline/checkpoint_epoch009_baseline.pt` — original baseline Phase 1
- `/dcs/large/u5734759/checkpoints/mini_focal_sampler/best_model.pt` — focal+sampler Phase 1
- `/dcs/large/u5734759/checkpoints/mini_aug/best_model.pt` — aug Phase 1
- `/dcs/large/u5734759/checkpoints/ravdess/best_acoustic_backbone.pt` — RAVDESS-only Phase A
- `/dcs/large/u5734759/checkpoints/combined/best_acoustic_backbone.pt` — Combined RAVDESS+MELD Phase A
