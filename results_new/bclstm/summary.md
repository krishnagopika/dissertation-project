# bc-LSTM context ablation

15 of 15 cells complete.

## Dev weighted F1 (emotion)

| condition | k0 | k1 | k2 | k4 | full |
|---|---|---|---|---|---|
| `gold` | 0.5853 | 0.5925 | 0.5954 | 0.6109 | 0.5931 |
| `asr` | 0.5018 | 0.4984 | 0.5044 | 0.5066 | 0.4975 |
| `asr_cleaned` | 0.5003 | 0.4982 | 0.4983 | 0.5041 | 0.5054 |

## Test weighted F1 (emotion)

| condition | k0 | k1 | k2 | k4 | full |
|---|---|---|---|---|---|
| `gold` | 0.6185 | 0.6258 | 0.6203 | 0.6180 | 0.6220 |
| `asr` | 0.4919 | 0.4930 | 0.4925 | 0.4935 | 0.4945 |
| `asr_cleaned` | 0.5073 | 0.5100 | 0.5128 | 0.5101 | 0.5075 |

## Per-run detail

| condition | context | dev WF1 | test WF1 | best ep | epochs | early stop | params |
|---|---|---|---|---|---|---|---|
| `asr` | `k0` | 0.5018 | 0.4919 | 8 | 14 | yes | 4727818 |
| `asr` | `k1` | 0.4984 | 0.4930 | 6 | 12 | yes | 4727818 |
| `asr` | `k2` | 0.5044 | 0.4925 | 5 | 11 | yes | 4727818 |
| `asr` | `k4` | 0.5066 | 0.4935 | 5 | 11 | yes | 4727818 |
| `asr` | `full` | 0.4975 | 0.4945 | 3 | 9 | yes | 4727818 |
| `asr_cleaned` | `k0` | 0.5003 | 0.5073 | 14 | 20 | yes | 4727818 |
| `asr_cleaned` | `k1` | 0.4982 | 0.5100 | 2 | 8 | yes | 4727818 |
| `asr_cleaned` | `k2` | 0.4983 | 0.5128 | 4 | 10 | yes | 4727818 |
| `asr_cleaned` | `k4` | 0.5041 | 0.5101 | 1 | 7 | yes | 4727818 |
| `asr_cleaned` | `full` | 0.5054 | 0.5075 | 9 | 15 | yes | 4727818 |
| `gold` | `k0` | 0.5853 | 0.6185 | 5 | 11 | yes | 4727818 |
| `gold` | `k1` | 0.5925 | 0.6258 | 5 | 11 | yes | 4727818 |
| `gold` | `k2` | 0.5954 | 0.6203 | 2 | 8 | yes | 4727818 |
| `gold` | `k4` | 0.6109 | 0.6180 | 16 | 22 | yes | 4727818 |
| `gold` | `full` | 0.5931 | 0.6220 | 3 | 9 | yes | 4727818 |

## Comparability

- dev key-set hashes: `['72c2797a81a4']`
- CONSISTENT — all runs scored on the same dev set
- note: `asr_cleaned` filters TRAIN labels only; dev and test keep every utterance, so all cells score the same dev set
