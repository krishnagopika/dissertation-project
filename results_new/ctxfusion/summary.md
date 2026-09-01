# Context-then-fusion: 2x2 context ablation

108 of 12 cells complete.

## Dev weighted F1 (emotion)

| condition | both | acoustic_only | text_only | neither |
|---|---|---|---|---|
| `gold` | 0.6018 | 0.5972 | 0.5927 | 0.5898 |
| `asr` | 0.5081 | 0.5166 | 0.5068 | 0.5010 |
| `asr_cleaned` | 0.4999 | 0.5127 | 0.5023 | 0.5079 |

## Value of context (arm − neither)

| condition | acoustic_only | text_only | both |
|---|---|---|---|
| `gold` | +0.0074 | +0.0029 | +0.0119 |
| `asr` | +0.0156 | +0.0058 | +0.0071 |
| `asr_cleaned` | +0.0048 | -0.0056 | -0.0080 |

`neither` = per-utterance fusion, no context. Deltas are the value of contextualising each channel.

## Parameters (arms are NOT matched)

| condition | both | acoustic_only | text_only | neither |
|---|---|---|---|---|
| `gold` | 6,309,386 | 4,339,210 | 3,552,778 | 1,582,602 |
| `asr` | 6,309,386 | 4,339,210 | 3,552,778 | 1,582,602 |
| `asr_cleaned` | 6,309,386 | 4,339,210 | 3,552,778 | 1,582,602 |

## Comparability

- dev key-set hashes: `['72c2797a81a4']`
- CONSISTENT
