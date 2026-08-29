# Fusion grid results

25 completed runs.

## Phase 0 — learning rate x batch size

| run | dev WF1 | best ep | epochs | early stop | lr | bs | train n | params |
|---|---|---|---|---|---|---|---|---|
| `asr_lr1e-3_bs32` | **0.5021** | 3 | 7 | yes | 0.001 | 32 | 9988 | 2,006,794 |
| `asr_lr5e-4_bs32` | **0.4969** | 7 | 11 | yes | 0.0005 | 32 | 9988 | 2,006,794 |
| `asr_lr1e-3_bs64` | **0.4909** | 3 | 7 | yes | 0.001 | 64 | 9988 | 2,006,794 |
| `asr_lr2e-4_bs32` | **0.4866** | 6 | 10 | yes | 0.0002 | 32 | 9988 | 2,006,794 |
| `asr_lr5e-4_bs64` | **0.4844** | 3 | 7 | yes | 0.0005 | 64 | 9988 | 2,006,794 |
| `asr_lr2e-4_bs64` | **0.4814** | 5 | 9 | yes | 0.0002 | 64 | 9988 | 2,006,794 |

## Phase 1 — acoustic-only, pooling ablation

| run | dev WF1 | best ep | epochs | early stop | lr | bs | train n | params |
|---|---|---|---|---|---|---|---|---|
| `acoustic_attention` | **0.5044** | 3 | 7 | yes | 0.001 | 32 | 9988 | 1,349,898 |
| `acoustic_attentive_stats` | **0.4809** | 7 | 11 | yes | 0.001 | 32 | 9988 | 2,005,258 |
| `acoustic_masked_mean` | **0.4671** | 4 | 8 | yes | 0.001 | 32 | 9988 | 1,185,802 |
| `acoustic_attention_fixed` | **0.4603** | 1 | 5 | yes | 0.001 | 32 | 9988 | 1,185,802 |

## Phase 2 — text-only baselines

| run | dev WF1 | best ep | epochs | early stop | lr | bs | train n | params |
|---|---|---|---|---|---|---|---|---|
| `gold` | **0.5620** | 4 | 8 | yes | 0.001 | 32 | 9988 | 923,658 |
| `asr_cleaned` | **0.4697** | 2 | 6 | yes | 0.001 | 32 | 6729 | 923,658 |
| `asr` | **0.4549** | 2 | 6 | yes | 0.001 | 32 | 9988 | 923,658 |

## Phase 3 — fusion mechanisms x text conditions

| run | dev WF1 | best ep | epochs | early stop | lr | bs | train n | params |
|---|---|---|---|---|---|---|---|---|
| `gold_sum` | **0.5652** | 3 | 7 | yes | 0.001 | 32 | 9988 | 1,744,650 |
| `gold_crossmodal` | **0.5637** | 4 | 8 | yes | 0.001 | 32 | 9988 | 2,269,962 |
| `gold_gated` | **0.5621** | 2 | 6 | yes | 0.001 | 32 | 9988 | 2,269,450 |
| `gold_concat` | **0.5617** | 0 | 4 | yes | 0.001 | 32 | 9988 | 2,006,794 |
| `asr_concat` | **0.5021** | 3 | 7 | yes | 0.001 | 32 | 9988 | 2,006,794 |
| `asr_cleaned_sum` | **0.4935** | 6 | 10 | yes | 0.001 | 32 | 6729 | 1,744,650 |
| `asr_cleaned_gated` | **0.4921** | 8 | 12 | yes | 0.001 | 32 | 6729 | 2,269,450 |
| `asr_sum` | **0.4866** | 3 | 7 | yes | 0.001 | 32 | 9988 | 1,744,650 |
| `asr_cleaned_crossmodal` | **0.4865** | 2 | 6 | yes | 0.001 | 32 | 6729 | 2,269,962 |
| `asr_crossmodal` | **0.4835** | 3 | 7 | yes | 0.001 | 32 | 9988 | 2,269,962 |
| `asr_gated` | **0.4805** | 3 | 7 | yes | 0.001 | 32 | 9988 | 2,269,450 |
| `asr_cleaned_concat` | **0.4752** | 2 | 6 | yes | 0.001 | 32 | 6729 | 2,006,794 |

## Comparability

- dev key-set hashes: `['1d9f866342c4']`
- CONSISTENT — all runs scored on the same dev set
