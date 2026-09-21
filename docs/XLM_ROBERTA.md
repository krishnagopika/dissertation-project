# XLM-RoBERTa — the text branch

What the text branch is, how it is trained, and why each hyperparameter has the
value it has. Every number here is read from the configs and the code, not from
a paper.

**References**
- Devlin et al. (2019), *BERT: Pre-training of Deep Bidirectional Transformers
  for Language Understanding*, NAACL. Appendix A.3 — the fine-tuning grid.
  https://arxiv.org/abs/1810.04805
- Mosbach, Andriushchenko & Klakow (2021), *On the Stability of Fine-tuning
  BERT: Misconceptions, Explanations, and Strong Baselines*, ICLR.
  https://arxiv.org/abs/2006.04884
- Lin et al. (2017), *Focal Loss for Dense Object Detection*, ICCV — the focal
  formulation used in run C. https://arxiv.org/abs/1708.02002
- Conneau et al. (2020), *Unsupervised Cross-lingual Representation Learning at
  Scale*, ACL — XLM-R. https://arxiv.org/abs/1911.02116

**Scope: methodology.** What the text branch is, how it is trained, and why each
hyperparameter has the value it has. Decision rationale and alternatives
rejected are in `DECISIONS.md`; implementation history is in
`CHANGE_RECORD.md`; failure analysis is in `POSTMORTEMS.md`. This document does
not repeat them.

**Related:** `MELD_ANALYSIS.md` — the data this trains on.

---

## 1. Role in the pipeline

MELD gives audio; the emotion label depends on both *what was said* and *how it
sounded*. The pipeline splits those:

```
audio.mp4
   ├─► Voxtral encoder ──► (T, 1280) acoustic sequence   how it SOUNDED
   └─► Voxtral decoder ──► transcript ──► XLM-R ──► 768-d   what was SAID
                                                      │
                                        fusion ◄──────┘
```

XLM-R is the **text** branch. It never sees audio. Its input is a transcript,
its output is a 768-d `[CLS]` representation plus emotion and sentiment logits.

Phase 1 (this document) fine-tunes it standalone. Phase 2 freezes it and trains
fusion on top , consuming the cached `[CLS]` vectors.

### Why XLM-RoBERTa rather than BERT or RoBERTa

MELD is English, so a multilingual encoder is not required by the data. It is
required by the *thesis question*: the target is dialect and accent robustness,
and a monolingual English model trained predominantly on US-standard text is a
poor starting point for claims about non-standard varieties. XLM-R's pretraining
(CommonCrawl, 100 languages) contains far more orthographic and dialectal
variation, which matters once the input is ASR output containing
transcription errors rather than clean edited text.

---

## 2. Architecture

`src/models/xlmr.py::XLMRobertaClassifier`

```
input_ids (B, 128) ──► XLM-RoBERTa-base encoder ──► last_hidden_state (B, 128, 768)
                                                          │
                                              [CLS] = [:, 0, :]   (B, 768)
                                                          │
                                                    Dropout(0.3)
                                                    ┌─────┴─────┐
                                        Linear(768→3)         Linear(768→7)
                                          sentiment              emotion
```

| | value | why |
|---|---|---|
| encoder | `FacebookAI/xlm-roberta-base` | 278M params. `large` would need a smaller batch on one L40S and is not the variable under study |
| pooling | `[CLS]` token, position 0 | XLM-R has a genuinely pretrained `[CLS]`, unlike Whisper's encoder — see ADR-005 |
| dropout | 0.3 | applied to `[CLS]` before both heads |
| heads | two linear, joint | multi-task: `loss = CE(emotion) + CE(sentiment)` |

### Why two heads rather than one

Emotion (7-way) and sentiment (3-way) are not independent — sentiment is close
to a coarsening of emotion. Training both is cheap multi-task regularisation:
the sentiment head gives a denser, better-conditioned gradient signal (3 classes
at 17.6:1 imbalance is far easier than 7), which stabilises the shared encoder
early in training when the emotion head is still near-random.

The losses are summed unweighted. That is a choice, not a derivation, and a
weighting term is an untested option.

### Why `[CLS]` pooling is safe here

`[CLS]` sits at position 0, so padding to 128 tokens does not dilute it. This is
in deliberate contrast to the acoustic branch, where mean-pooling over a
30 s-padded sequence averaged mostly padding (`MELD_ANALYSIS.md` §5,
`DECISIONS.md` ADR-003). The text branch never had that defect.

---

## 3. Training regime — supervised fine-tuning

Full fine-tuning: the encoder is **not** frozen in Phase 1. All 278M parameters
receive gradients.

```
loss = CrossEntropy(emotion_logits, emotion_label)
     + CrossEntropy(sentiment_logits, sentiment_label)
```

### Hyperparameters

All identical across the nine runs, so the ablation varies only what it means to.

| parameter | value | justification |
|---|---:|---|
| `phase1_lr` | **2e-5** | Directly from Devlin et al. (2019) Appendix A.3, which recommends selecting from {5e-5, 3e-5, 2e-5}. See §3.1 for what this does and does *not* justify |
| optimiser | **AdamW** | Decoupled weight decay. Adam's L2 interacts badly with adaptive scaling; AdamW is the standard for transformer fine-tuning |
| `weight_decay` | **0.01** | Standard for AdamW on transformers |
| `batch_size` | **16** | In Devlin et al.'s recommended set {16, 32}. 16 over 32 doubles the optimiser steps — 625 vs 312 per epoch at n=9,989 — which matters more than throughput on a small dataset. See §3.1 |
| `max_text_length` | **128** tokens | MELD's median gold utterance is 6 words; p90 well under 128. Longer wastes compute on padding |
| `warmup_steps` | **100** | ~1/6 of one epoch at batch 16. Prevents large early updates from disrupting pretrained weights while Adam's moment estimates are still poor |
| scheduler | **linear decay with warmup** | `get_linear_schedule_with_warmup`. Standard; anneals to 0 by the final step |
| `max_grad_norm` | **1.0** | Gradient clipping. Standard safeguard against loss spikes |
| `epochs_phase1` | **10** (max) | An upper bound, not a target — early stopping decides the real number |
| `patience` | **3** | On dev weighted F1 |
| `dropout` | **0.3** | On `[CLS]` before both heads |
| `seed` | **42** | Fixed across all runs |

### 3.1 What the hyperparameter citations actually support

**The values are sourced. The mechanism usually given for them is contested.**

[Devlin et al. (2019)](https://arxiv.org/abs/1810.04805), Appendix A.3, gives
the fine-tuning grid this configuration is drawn from:

| | recommended | ours |
|---|---|---|
| batch size | {16, 32} | **16** |
| learning rate (Adam) | {5e-5, 3e-5, 2e-5} | **2e-5** |
| epochs | {2, 3, 4} | **10 max, early-stopped** |

That is a citation for the *value*, and it is a grid to search, not a derived
optimum. **No sweep was run here** — the values were fixed so the nine runs
differ only in text condition and loss.

#### The citation is for BERT, and we fine-tune XLM-R

This should be stated plainly rather than glossed. Appendix A.3 is BERT's
fine-tuning grid. A primary citation giving XLM-R's own recommended fine-tuning
learning rate was **not located** when this document was written; neither was
RoBERTa's Appendix C retrieved directly (only a third-party summary, which
described RoBERTa-*large*, not base).

The extrapolation is defensible as an *argument*, not as a citation:

- XLM-RoBERTa is architecturally RoBERTa — same encoder stack, 768-d base, same
  AdamW fine-tuning setup. It differs in pretraining corpus (CommonCrawl, 100
  languages) and vocabulary, not in the fine-tuning optimisation problem.
- RoBERTa is architecturally BERT — the differences are pretraining choices
  (no NSP, dynamic masking, larger batches), again not the fine-tuning recipe.

**Where that argument could fail.** XLM-R's vocabulary is 250k SentencePiece
tokens against BERT's 30k, so the embedding matrix is roughly 192M of its 278M
parameters — a far larger share of the model than in BERT-base. Whether that
shifts the optimal fine-tuning learning rate is not something this project has
established or found a source for.

**What the XLM-R paper itself specifies.** The paper (Conneau et al. 2020, ACL)
was text-extracted in full (7,641 words). It contains **no fine-tuning learning rate anywhere** — the only occurrences of
"Adam", "optimiz*" or "learning rate" are in the bibliography (the RoBERTa
title, and an author named Adam Roberts). The only batch size given, 8192, is
for *pretraining* on 500 V100s.

So 2e-5 is neither inside nor outside an XLM-R recommendation: **XLM-R publishes
no fine-tuning range.** The BERT/RoBERTa grid is the de-facto standard the field
applies across this architecture family, and 2e-5 sits inside it. There is no
authoritative value to defer to, so the only real justification available is
empirical — and none has been run here.

**What this does and does not threaten.** All nine runs share the same learning
rate, so it cannot bias gold-vs-ASR or A-vs-B-vs-C; a suboptimal value shifts
all nine together. It affects absolute performance, not the comparisons the
conclusions rest on.

**Priority for further validation** — these are not equally valuable:

| | fixes | cost | priority |
|---|---|---|---|
| multi-seed (3-5 seeds, ONE condition) | whether observed differences are real | 3-5 runs, ~40 min | **first** |
| LR sweep (1e-5 / 2e-5 / 3e-5) | absolute performance | 2 runs | second |

Multi-seed matters more, and Mosbach et al. is the reason: the same model on the
same data with a different seed produces large variance in task performance.
Every conclusion here is a *difference between conditions*. If seed variance is
+/-0.02 weighted F1 and a gap is 0.015, the gap is not evidence — and at present
there is no way to tell which case holds.

Suggested: `asr_weighted` at seeds 42/43/44/45/46, giving an error bar that can
be attached to every other cell in the grid without re-running all nine.

**A claim to avoid.** The common justification — "higher learning rates cause
catastrophic forgetting of pretrained representations" — is not supported.
[Mosbach et al., ICLR 2021](https://arxiv.org/abs/2006.04884) tested both
catastrophic forgetting and small-dataset size as explanations for fine-tuning
instability and found **both fail to explain it**. Their finding is that
instability arises from *optimisation difficulties causing vanishing gradients*.

Their prescription is small learning rates **with bias correction** (which AdamW
provides) and **training for more iterations, to near-zero training loss**.

**That is in tension with `patience=3`.** Aggressive early stopping is the
opposite of "train to near-zero training loss". Two defensible positions:

- *ours*: dev weighted F1 is the selection metric, `best_model.pt` retains the
  best epoch regardless, and nine runs must fit one job. Early stopping cannot
  make the selected model worse — only cheaper to reach.
- *Mosbach et al.*: stopping early on a small dataset may halt runs still in the
  unstable optimisation regime, so a run's score partly reflects where it
  happened to stop rather than its converged quality.

The second is a real risk and is **not controlled for here**. It compounds the
single-seed limitation in §7: with one seed and early stopping, a small gap
between conditions could be an artefact of stopping point rather than treatment.
Treat differences below ~0.01 weighted F1 as unsupported.

### Why 10 epochs is a ceiling, not a setting

Fine-tuning a pretrained encoder on ~10k examples converges in a handful of
epochs and then overfits; a previous run in this project reached its best dev
weighted F1 at epoch 2 of 3. Setting 10 with `patience=3` lets each of the nine
runs stop where *it* converges rather than forcing a shared epoch count that
would under-train some and over-train others.

This matters because the runs have different dataset sizes — `asr_cleaned` has
6,729 training examples versus 9,989 — so an equal epoch count is not an equal
amount of training. Early stopping on a shared criterion is the fairer control.

### Why early stopping tracks weighted F1, not validation loss

Loss under 17.6:1 imbalance is dominated by the majority class. A model can
reduce loss by becoming *more* confidently neutral while minority-class F1
degrades. Weighted F1 is also the project's declared primary metric
, so selecting on it aligns the stopping criterion with the
reporting criterion.

`best_model.pt` already holds the best epoch, so stopping early costs nothing —
it only avoids wasted epochs.

---

## 4. The nine runs

Two axes, three levels each, one variable per comparison.

```
                    TRAIN
              ┌───────┼───────────┐
            GOLD     ASR     ASR-CLEANED
              └───────┼───────────┘
                      ↓
                same ASR TEST
                      ↓
              primary comparison
```

### Axis 1 — training text

| condition | text | n (train) |
|---|---|---:|
| `gold` | MELD CSV `utterance` — human reference | 9,989 |
| `asr` | Voxtral-Mini transcripts | 9,989 |
| `asr_cleaned` | Voxtral-Mini transcripts, `text_clean` keep-list | 6,729 |

`text_clean` = drop duplicate audio, clips < 1 s, `speech_ratio` < 0.20,
runaway ASR, empty ASR, **and** WER > 0.50. See `MELD_ANALYSIS.md` §5.

**Evaluation is fixed at ASR for all nine.** At inference there are no gold
transcripts, only what the ASR produced — so scoring the gold-trained model on
gold text would measure a condition that does not exist in deployment. Gold
runs are additionally scored on gold text as a *ceiling* diagnostic
(`_goldceiling` tag): the gap to their ASR-test number is the cost of ASR error.

`gold` is the control condition. It quantifies the ceiling — what the text
branch achieves given perfect transcription — and, scored on ASR text, the cost
of transcription error to a model that never saw one during training.

### Axis 2 — loss

| run | loss | alpha | gamma |
|---|---|---|---|
| A `plain` | `CrossEntropyLoss(weight=None)` | — | — |
| B `weighted` | `CrossEntropyLoss(weight=w)` | inverse frequency | — |
| C `focal` | `FocalLoss(weight=w, gamma=2)` | inverse frequency | 2.0 |

`w_c = N / (C · count_c)` — inverse frequency.

**A → B isolates alpha. B → C isolates gamma.** C keeps alpha deliberately: a
literal "focal loss with gamma=2" would drop it, but then B and C would differ
in two variables and a "C beats B" result could not be attributed. This also
matches Lin et al. (2017), whose formulation uses both.

`use_weighted_sampler: false` in all nine, and `finetune.py` **raises** if it is
enabled with `weighted` or `focal` — the sampler suppresses alpha to avoid
double-correcting, which would silently collapse the ladder to
(CE+sampler, CE+sampler, focal+sampler).

Motivation is the imbalance in `MELD_ANALYSIS.md` §2: 17.6:1 neutral-to-fear in
train, and only 22 disgust / 40 fear in dev.

### The dev set is deliberately not filtered

`asr_cleaned` trains on 6,729 but selects on the **full** 1,109-utterance dev
set (`filter_dev: false`). Filtering dev would make the filter change both the
training data and the early-stopping criterion, so a difference could not be
attributed to either — and dev would stop being comparable across the nine runs.
Dev is a fixed yardstick; the filter is the treatment.

---

## 5. Run isolation

Each of the nine runs writes to its own `checkpoint_dir`, `log_dir` and
TensorBoard directory. `finetune.py` takes `checkpoint_dir` directly from config
with no run identifier appended, so a shared directory would let runs overwrite
one another's weights and would let a run warm-start from a previous run's
checkpoint rather than from base XLM-R — the second being the more dangerous,
since it invalidates the comparison silently.

**Automatic resume is disabled**; `--resume` must be passed explicitly. Resume
is unsuitable for a controlled comparison for two reasons: it restores
`best_metric` from the latest checkpoint rather than the best one, so a later
inferior epoch can overwrite a better `best_model.pt`; and it rebuilds the
learning-rate scheduler from step 0 while the optimizer continues from its saved
state, re-running warmup part-way through training.

Completion is recorded in `TRAINING_COMPLETE.json`, written only after the
training loop exits normally. `best_model.pt` is not a completion signal — it
appears at the first improving epoch, so it exists for a run that stopped early
for any reason.

---

## 6. What gets logged

**TensorBoard** — `logs/xlmr/{condition}/{loss}/tb`:

| scalar | why |
|---|---|
| `loss/train`, `loss/dev` | overfitting shows as divergence |
| `emotion/dev_weighted_f1` | the selection metric |
| `emotion/dev_macro_f1` | treats all classes equally — moves differently from weighted under imbalance |
| `emotion_f1_per_class/*` | **the important one.** A weighted average hides exactly the minority classes A/B/C target |
| `sentiment/dev_weighted_f1` | the auxiliary task |
| `lr` | confirms warmup and decay behaved |
| `train/epochs_since_best` | early-stopping counter |

**Results JSON** — `results_new/xml_roberta/{condition}/`, one per split per
loss, with `_asrtest` on primary numbers and `_goldceiling` on the diagnostic.

---

## 7. Limitations

- **Speaker-dependent evaluation.** MELD splits by dialogue, not speaker, and
  the principal cast appears in all splits (`MELD_ANALYSIS.md` §1). Any result
  here is an upper bound relative to unseen speakers.
- **Sentiment loss weighting is untested.** `loss = CE_emotion + CE_sentiment`
  with equal weight is assumed, not tuned.
- **Test-set choice unresolved.** All nine report on the full ASR test set,
  which keeps comparability with published MELD numbers. Whether to *also*
  report on the clean subset is open (ADR-007).
- **Early stopping is not validated against convergence.** Mosbach et al. (2021)
  recommend training to near-zero training loss for stability; `patience=3` does
  the opposite. Untested here — see §3.1.
- **No hyperparameter sweep.** Values come from Devlin et al.'s recommended grid
  and were held fixed, not tuned on this data.
- **Single seed.** Every run uses seed 42. Differences smaller than
  seed-to-seed variance are not interpretable, and that variance has not been
  measured. Any conclusion resting on a gap of ~0.005 weighted F1 should be
  treated as unsupported until a multi-seed run exists.
- **Mini only.** All transcripts come from Voxtral-Mini. Small requires the
  same treatment once its weights land (ADR-006).

---

## Reproducing

```bash
sbatch src/scripts/xlmr_grid.sbatch          # all nine
tensorboard --logdir logs/xlmr --port 6006
```

Single run:

```bash
python3.12 src/training/finetune.py \
    --config src/configs/xlmr_asr_cleaned_weighted.yaml \
    --text_source asr --loss weighted --patience 3

python3.12 src/evaluation/evaluate_text_only.py \
    --config src/configs/xlmr_asr_cleaned_weighted.yaml \
    --split test --text_source asr --tag _weighted_test_asrtest
```
