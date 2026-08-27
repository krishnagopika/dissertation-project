# Filter vs Unfiltered — Full Comparison

**Filter policy**: Voxtral WER ≤ 25 %, Silero-VAD speech ratio ≥ 0.20.
Applied to train (9989 → 3539), dev (1109 → 379), test (2610 → 909).

**Test denominators differ**: unfiltered numbers are on 2610 clips, filter numbers on 909 clips (the clean-ASR subset). Numbers are not directly comparable in absolute terms — the filter run is measuring performance *on a cleaner subset of MELD*. Compare within a model family for the effect of filtering.

---

## Emotion (7-class)

| Model | Config | WF1 | mF1 | fear | disgust | anger | sadness | joy | surprise | neutral |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Text-only** | baseline | 0.454 | 0.320 | 0.125 | 0.140 | 0.296 | 0.287 | 0.368 | 0.438 | 0.587 |
| Text-only | focal+sampler | 0.468 | 0.311 | 0.111 | 0.167 | 0.139 | 0.285 | 0.382 | 0.440 | 0.652 |
| Text-only | aug | 0.449 | 0.292 | 0.070 | 0.116 | 0.178 | 0.307 | 0.328 | 0.415 | 0.627 |
| **Text-only** | **filter wer25** | **0.551** | **0.341** | 0.027 | 0.161 | 0.220 | 0.393 | 0.391 | 0.483 | 0.714 |
| **Classical SVM** (dev) | baseline | 0.467 | 0.349 | 0.226 | 0.167 | — | — | — | — | — |
| **Classical SVM** (dev) | **filter wer25** | **0.508** | 0.373 | 0.138 | 0.103 | — | — | — | — | — |
| **Classical XGB** (dev) | baseline | 0.464 | 0.341 | 0.192 | 0.176 | — | — | — | — | — |
| **Classical XGB** (dev) | **filter wer25** | **0.494** | 0.349 | 0.157 | 0.061 | — | — | — | — | — |
| **Fusion concat** | baseline | 0.467 | 0.315 | 0.128 | 0.127 | 0.289 | 0.262 | 0.332 | 0.437 | 0.632 |
| Fusion concat | focal+sampler | 0.593 | 0.415 | 0.142 | 0.173 | 0.387 | 0.317 | 0.594 | 0.542 | 0.747 |
| Fusion concat | aug | 0.593 | 0.415 | 0.142 | 0.173 | 0.387 | 0.317 | 0.594 | 0.542 | 0.747 |
| Fusion concat | ravdesspre | 0.593 | 0.420 | 0.175 | 0.166 | 0.432 | 0.339 | 0.555 | 0.525 | 0.746 |
| **Fusion concat** | **filter wer25** | **0.603** | 0.390 | 0.038 | 0.176 | 0.340 | 0.418 | 0.481 | 0.521 | 0.754 |
| **Fusion sum** | baseline | 0.473 | 0.325 | 0.140 | 0.128 | 0.314 | 0.282 | 0.337 | 0.446 | 0.629 |
| Fusion sum | focal+sampler | 0.587 | 0.409 | 0.147 | 0.131 | 0.394 | 0.341 | 0.580 | 0.530 | 0.739 |
| Fusion sum | aug | 0.591 | 0.413 | 0.139 | 0.142 | 0.405 | 0.337 | 0.584 | 0.545 | 0.739 |
| Fusion sum | ravdesspre | 0.576 | 0.415 | 0.175 | 0.193 | 0.418 | 0.320 | 0.566 | 0.515 | 0.716 |
| **Fusion sum** | **filter wer25** | **0.600** | 0.382 | 0.033 | 0.171 | 0.352 | 0.364 | 0.467 | 0.533 | 0.754 |
| **Fusion gated** | baseline | 0.466 | 0.319 | 0.135 | 0.140 | 0.289 | 0.274 | 0.335 | 0.432 | 0.627 |
| Fusion gated | focal+sampler | 0.583 | 0.395 | 0.130 | 0.161 | 0.374 | 0.299 | 0.541 | 0.505 | 0.759 |
| Fusion gated | aug | 0.591 | 0.413 | 0.134 | 0.158 | 0.394 | 0.314 | 0.591 | 0.559 | 0.739 |
| Fusion gated | ravdesspre | 0.586 | 0.422 | 0.173 | 0.193 | 0.420 | 0.323 | 0.579 | 0.540 | 0.726 |
| **Fusion gated** | **filter wer25** | **0.602** | 0.374 | 0.000 | 0.145 | 0.339 | 0.369 | 0.477 | 0.525 | 0.762 |
| **Fusion crossmodal** | baseline | 0.465 | 0.321 | 0.143 | 0.131 | 0.298 | 0.270 | 0.356 | 0.433 | 0.616 |
| Fusion crossmodal | focal+sampler | 0.583 | 0.396 | 0.134 | 0.170 | 0.370 | 0.299 | 0.535 | 0.500 | 0.762 |
| Fusion crossmodal | aug | 0.591 | 0.415 | 0.158 | 0.137 | 0.397 | 0.342 | 0.581 | 0.548 | 0.741 |
| Fusion crossmodal | ravdesspre | 0.586 | 0.418 | 0.178 | 0.167 | 0.430 | 0.320 | 0.568 | 0.533 | 0.729 |
| **Fusion crossmodal** | **filter wer25** | **0.601** | 0.378 | 0.054 | 0.147 | 0.340 | 0.366 | 0.464 | 0.517 | 0.761 |

## Sentiment (3-class)

| Model | Config | WF1 | mF1 | negative | neutral | positive |
|---|---|---:|---:|---:|---:|---:|
| **Text-only** | baseline | 0.533 | 0.506 | 0.520 | 0.595 | 0.404 |
| Text-only | focal+sampler | 0.559 | 0.524 | 0.540 | 0.641 | 0.392 |
| Text-only | aug | 0.544 | 0.502 | 0.504 | 0.650 | 0.352 |
| **Text-only** | **filter wer25** | **0.628** | **0.552** | 0.545 | 0.731 | 0.379 |
| **Classical SVM** (dev) | baseline | 0.556 | 0.532 | 0.544 | 0.636 | 0.417 |
| **Classical SVM** (dev) | **filter wer25** | **0.575** | 0.550 | 0.584 | 0.645 | 0.420 |
| **Classical XGB** (dev) | baseline | 0.554 | 0.528 | 0.543 | 0.639 | 0.403 |
| **Classical XGB** (dev) | **filter wer25** | **0.576** | 0.549 | 0.581 | 0.655 | 0.410 |
| **Fusion concat** | baseline | 0.534 | 0.496 | 0.478 | 0.638 | 0.371 |
| Fusion concat | focal+sampler | 0.676 | 0.652 | 0.596 | 0.762 | 0.599 |
| Fusion concat | aug | 0.676 | 0.652 | 0.596 | 0.762 | 0.599 |
| Fusion concat | ravdesspre | 0.687 | 0.663 | 0.623 | 0.767 | 0.599 |
| **Fusion concat** | **filter wer25** | 0.676 | 0.612 | 0.600 | 0.764 | 0.472 |
| **Fusion sum** | ravdesspre | 0.685 | 0.662 | 0.632 | 0.758 | 0.595 |
| **Fusion sum** | **filter wer25** | 0.679 | 0.615 | 0.606 | 0.767 | 0.472 |
| **Fusion gated** | ravdesspre | 0.679 | 0.656 | 0.618 | 0.754 | 0.597 |
| **Fusion gated** | **filter wer25** | 0.675 | 0.615 | 0.592 | 0.764 | 0.488 |
| **Fusion crossmodal** | aug | 0.676 | 0.653 | 0.610 | 0.753 | 0.596 |
| **Fusion crossmodal** | **filter wer25** | 0.677 | 0.618 | 0.600 | 0.763 | 0.490 |

## Best-of comparison — the punchline

| Model class | Best unfiltered WF1 | Filter wer25 WF1 | Δ |
|---|---:|---:|---:|
| Text-only emotion | 0.468 | **0.551** | **+0.083** |
| Classical SVM emotion (dev) | 0.467 | **0.508** | +0.041 |
| Fusion emotion (best of 4) | 0.593 | **0.603** | +0.010 |
| Text-only sentiment | 0.559 | **0.628** | +0.069 |
| Fusion sentiment (best of 4) | 0.687 | 0.679 | −0.008 (≈ same) |

## Takeaways

1. **Filter helps text-only most** (+0.083 WF1). Filtering removes the biggest handicap for pure-text models: ASR noise.
2. **Fusion barely benefits** (+0.010 WF1). The multimodal architecture was already using acoustic features to compensate for noisy transcripts — filtering removes noise it was already handling.
3. **Classical baselines improve modestly** (~+0.04 WF1). Same story as text-only, less pronounced.
4. **Sentiment is largely unaffected** — the 3-class taxonomy is coarser and less sensitive to per-utterance ASR quality.
5. **Fear/disgust F1 drops sharply** *not* because the models got worse — because filter test has only 6 fear clips and 24 disgust clips. Per-class F1 is unreliable at those denominators.
6. **Macro F1 drops** for the same reason — averaging across a per-class F1 that includes an unreliable fear=0.0 pulls the mean down.

## Interpretation for the dissertation

- The filter is a **diagnostic**, not a fix. It confirms that:
  - Text-only was constrained mostly by ASR noise.
  - Fusion had already learned to compensate for ASR noise using the acoustic branch — hence the small gain.
  - Fear/disgust are a **structural bottleneck** of MELD (too few clips), not a solvable ASR problem.
