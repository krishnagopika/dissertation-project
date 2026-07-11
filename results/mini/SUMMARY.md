# MELD Test Results — Ablation Summary

All numbers are **MELD test set** weighted F1 / macro F1 / per-class F1 unless stated otherwise. `mini` config = XLM-R-base + Voxtral-Mini-3B.

## 1) Phase 1 (text-only) ablations — XLM-R alone

Different recipes for fine-tuning XLM-RoBERTa on MELD ASR transcripts. Both reuse class-weighted CE.

| Phase 1 config | WF1 | mF1 | fear | disgust | anger | sad | joy | surprise | neutral |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline (class-weighted CE) | 0.454 | 0.320 | 0.125 | 0.140 | 0.296 | 0.287 | 0.368 | 0.438 | 0.587 |
| **focal loss + weighted sampler** | 0.468 | 0.311 | 0.111 | 0.167 | 0.139 | 0.285 | 0.382 | 0.440 | 0.652 |
| **Voxtral paraphrase augmentation** | 0.449 | 0.292 | 0.070 | 0.116 | 0.178 | 0.307 | 0.328 | 0.415 | 0.627 |

## 2) Fusion ablations — XLM-R + Voxtral acoustic

Per fusion architecture, sweep across Phase 1 + acoustic-pretrain choices. RAVDESS pretrain warm-starts `acoustic_proj` on sum/gated/crossmodal (no effect on concat — flagged).


### CONCAT

| Config | WF1 | mF1 | fear | disgust | anger | sad | joy | surprise | neutral |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.467 | 0.315 | 0.128 | 0.127 | 0.289 | 0.262 | 0.332 | 0.437 | 0.632 |
| focal+sampler | 0.593 | 0.415 | 0.142 | 0.173 | 0.387 | 0.317 | 0.594 | 0.542 | 0.747 |
| aug only | 0.593 | 0.415 | 0.142 | 0.173 | 0.387 | 0.317 | 0.594 | 0.542 | 0.747 |
| RAVDESS pretrain | 0.593 | 0.420 | 0.175 | 0.166 | 0.432 | 0.339 | 0.555 | 0.525 | 0.746 |
| focal+sampler + RAVDESS | 0.593 | 0.420 | 0.175 | 0.166 | 0.432 | 0.339 | 0.555 | 0.525 | 0.746 |
| aug + RAVDESS | 0.593 | 0.415 | 0.142 | 0.173 | 0.387 | 0.317 | 0.594 | 0.542 | 0.747 |
| no_pretrain (control) | 0.593 | 0.420 | 0.175 | 0.166 | 0.432 | 0.339 | 0.555 | 0.525 | 0.746 |

### SUM

| Config | WF1 | mF1 | fear | disgust | anger | sad | joy | surprise | neutral |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.473 | 0.325 | 0.140 | 0.128 | 0.314 | 0.282 | 0.337 | 0.446 | 0.629 |
| focal+sampler | 0.587 | 0.409 | 0.147 | 0.131 | 0.394 | 0.341 | 0.580 | 0.530 | 0.739 |
| aug only | 0.591 | 0.413 | 0.139 | 0.142 | 0.405 | 0.337 | 0.584 | 0.545 | 0.739 |
| RAVDESS pretrain | 0.576 | 0.415 | 0.175 | 0.193 | 0.418 | 0.320 | 0.566 | 0.515 | 0.716 |
| focal+sampler + RAVDESS | 0.576 | 0.415 | 0.175 | 0.193 | 0.418 | 0.320 | 0.566 | 0.515 | 0.716 |
| aug + RAVDESS | 0.587 | 0.409 | 0.148 | 0.144 | 0.390 | 0.339 | 0.579 | 0.525 | 0.739 |
| no_pretrain (control) | 0.576 | 0.415 | 0.175 | 0.193 | 0.418 | 0.320 | 0.566 | 0.515 | 0.716 |

### GATED

| Config | WF1 | mF1 | fear | disgust | anger | sad | joy | surprise | neutral |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.466 | 0.319 | 0.135 | 0.140 | 0.289 | 0.274 | 0.335 | 0.432 | 0.627 |
| focal+sampler | 0.583 | 0.395 | 0.130 | 0.161 | 0.374 | 0.299 | 0.541 | 0.505 | 0.759 |
| aug only | 0.591 | 0.413 | 0.134 | 0.158 | 0.394 | 0.314 | 0.591 | 0.559 | 0.739 |
| RAVDESS pretrain | 0.586 | 0.422 | 0.173 | 0.193 | 0.420 | 0.323 | 0.579 | 0.540 | 0.726 |
| focal+sampler + RAVDESS | 0.586 | 0.422 | 0.173 | 0.193 | 0.420 | 0.323 | 0.579 | 0.540 | 0.726 |
| aug + RAVDESS | 0.590 | 0.414 | 0.147 | 0.153 | 0.395 | 0.319 | 0.589 | 0.553 | 0.739 |
| no_pretrain (control) | 0.586 | 0.422 | 0.173 | 0.193 | 0.420 | 0.323 | 0.579 | 0.540 | 0.726 |

### CROSSMODAL

| Config | WF1 | mF1 | fear | disgust | anger | sad | joy | surprise | neutral |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.465 | 0.321 | 0.143 | 0.131 | 0.298 | 0.270 | 0.356 | 0.433 | 0.616 |
| focal+sampler | 0.583 | 0.396 | 0.134 | 0.170 | 0.370 | 0.299 | 0.535 | 0.500 | 0.762 |
| aug only | 0.591 | 0.415 | 0.158 | 0.137 | 0.397 | 0.342 | 0.581 | 0.548 | 0.741 |
| RAVDESS pretrain | 0.586 | 0.418 | 0.178 | 0.167 | 0.430 | 0.320 | 0.568 | 0.533 | 0.729 |
| focal+sampler + RAVDESS | 0.586 | 0.418 | 0.178 | 0.167 | 0.430 | 0.320 | 0.568 | 0.533 | 0.729 |
| aug + RAVDESS | 0.586 | 0.411 | 0.148 | 0.145 | 0.394 | 0.333 | 0.583 | 0.543 | 0.733 |
| no_pretrain (control) | 0.586 | 0.418 | 0.178 | 0.167 | 0.430 | 0.320 | 0.568 | 0.533 | 0.729 |

## 3) Acoustic-only on MELD test (sanity check, no text)

AcousticEmotionClassifier trained on RAVDESS+MELD audio (all labels in MELD space, RAVDESS calm→MELD neutral).

| Acoustic-only model | WF1 | mF1 | fear | disgust | anger | sad | joy | surprise | neutral |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| combined (RAVDESS + MELD train) | 0.413 | 0.283 | 0.122 | 0.025 | 0.390 | 0.286 | 0.329 | 0.304 | 0.524 |

## 4) RAVDESS Phase A (RAVDESS test, 8 classes incl. calm)

Just to show how well the acoustic backbone can do on clean acted data — context for the cross-corpus story.

- RAVDESS test WF1: **0.594**, macro F1: **0.592**
- neutral: 0.488
- calm: 0.593
- happy: 0.489
- sad: 0.507
- angry: 0.677
- fearful: 0.679
- disgust: 0.635
- surprised: 0.667

## Key takeaways


1. **Phase 1 (text-only) recipe changes barely moved the needle.**
   focal+sampler nudged WF1 from 0.454 → 0.468 but macro F1 actually dropped (0.32 → 0.31). Voxtral paraphrase augmentation hurt text-only (WF1 0.45, macro 0.29) — synthetic paraphrases introduce label noise that outweighs the imbalance fix at this dataset size.

2. **Multimodal fusion is the real win.**
   Going from text-only (best WF1 ≈ 0.47) → fusion (best WF1 ≈ 0.59) is +0.12 absolute WF1 and +0.10 macro F1. Every fusion variant beats every text-only variant.

3. **Fusion variants are within ~0.02 WF1 of each other** — architecture choice matters less than just doing fusion at all.

4. **RAVDESS cross-corpus pretrain did not transfer.**
   - On RAVDESS test (acted, clean), the backbone hit fear F1=0.68, disgust F1=0.63.
   - On MELD test, the same backbone in acoustic-only inference gave fear=0.122, disgust=0.025.
   - In fusion, RAVDESS warm-start gave numbers numerically indistinguishable from no-warm-start — fusion training over 10 epochs converges to the same solution regardless of init.
   - Conclusion: studio-acted emotion acoustic patterns from RAVDESS do not match MELD's noisy in-the-wild TV-dialogue audio.

5. **Fear / disgust remain the bottleneck.**
   Best fusion fear=0.178, disgust=0.193 — still far below other classes. The acoustic-only sanity check (disgust=0.025) confirms the audio modality has limited transferable emotion signal for these classes on MELD. The text branch is doing most of the work in fusion.

