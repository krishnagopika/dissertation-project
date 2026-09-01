# WER / VAD defect analysis

Source: `results_new/mini/wer`  
Measurement only — nothing filtered.

Flags are additive: a clip can be short AND duplicated AND runaway.

## train — 9989 utterances

| class | count | % | corpus WER | median WER |
|---|---:|---:|---:|---:|
| ALL | 9988 | 100.00 | 0.3461 | 0.1667 |
| dup_copy | 19 | 0.19 | 0.6832 | 0.5 |
| short | 942 | 9.43 | 1.7818 | 0.2857 |
| low_vad | 950 | 9.51 | 1.4538 | 0.6667 |
| no_speech | 862 | 8.63 | 1.6786 | 1.0 |
| runaway | 23 | 0.23 | 25.1837 | 24.2857 |
| empty_asr | 1 | 0.01 | 1.0 | 1.0 |
| high_wer | 2378 | 23.81 | 1.3261 | 1.0 |
| wer_over_1 | 1043 | 10.44 | 2.8465 | 2.3333 |
| **clean (no flags)** | 8469 | 84.78 | **0.2908** | 0.1667 |

### WER threshold sensitivity

How many utterances a WER gate removes, and how many of those the
audio-based flags (dup / short / low_vad / runaway) already catch.
A gate that removes only clips the audio flags catch adds nothing.

| WER > | removed | % | also audio-flagged | **only** WER | kept | kept corpus WER |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 5904 | 59.11 | 873 | 5031 | 4085 | 0.0253 |
| 0.15 | 5288 | 52.94 | 859 | 4429 | 4701 | 0.0448 |
| 0.20 | 4518 | 45.23 | 831 | 3687 | 5471 | 0.068 |
| 0.25 | 3955 | 39.59 | 794 | 3161 | 6034 | 0.0865 |
| 0.30 | 3649 | 36.53 | 789 | 2860 | 6340 | 0.0989 |
| 0.40 | 2947 | 29.5 | 733 | 2214 | 7042 | 0.127 |
| 0.50 | 2378 | 23.81 | 637 | 1741 | 7611 | 0.1497 |
| 0.75 | 1814 | 18.16 | 570 | 1244 | 8175 | 0.1864 |
| 1.00 | 1043 | 10.44 | 306 | 737 | 8946 | 0.227 |

### Flag combinations

| combination | count |
|---|---:|
| clean | 6729 |
| high_wer | 1740 |
| short | 431 |
| short|low_vad|no_speech|high_wer | 266 |
| low_vad|no_speech | 264 |
| low_vad|no_speech|high_wer | 200 |
| short|low_vad|no_speech | 114 |
| short|high_wer | 114 |
| low_vad | 63 |
| low_vad|high_wer | 25 |
| short|low_vad|no_speech|runaway|high_wer | 15 |
| dup_copy|high_wer | 8 |
| dup_copy | 8 |
| runaway|high_wer | 6 |
| empty_asr|high_wer | 1 |
| low_vad|no_speech|runaway|high_wer | 1 |
| dup_copy|low_vad|no_speech|high_wer | 1 |
| dup_copy|short | 1 |
| dup_copy|low_vad|no_speech | 1 |
| short|runaway|high_wer | 1 |

### Defect rate by emotion

| emotion | n | defective | % |
|---|---:|---:|---:|
| neutral | 4710 | 1516 | 32.2 |
| joy | 1743 | 595 | 34.1 |
| surprise | 1205 | 491 | 40.7 |
| anger | 1109 | 327 | 29.5 |
| sadness | 683 | 171 | 25.0 |
| disgust | 271 | 79 | 29.2 |
| fear | 268 | 81 | 30.2 |

## dev — 1109 utterances

| class | count | % | corpus WER | median WER |
|---|---:|---:|---:|---:|
| ALL | 1109 | 100.00 | 0.3124 | 0.1667 |
| dup_copy | 1 | 0.09 | 1.2 | 1.2 |
| short | 107 | 9.65 | 2.5635 | 0.5 |
| low_vad | 130 | 11.72 | 1.4665 | 0.5 |
| no_speech | 118 | 10.64 | 1.7808 | 0.6667 |
| runaway | 2 | 0.18 | 31.5455 | 35.0 |
| empty_asr | 1 | 0.09 | 1.0 | 1.0 |
| high_wer | 242 | 21.82 | 1.4287 | 1.0 |
| wer_over_1 | 99 | 8.93 | 3.4879 | 2.6667 |
| **clean (no flags)** | 918 | 82.78 | **0.2392** | 0.1538 |

### WER threshold sensitivity

How many utterances a WER gate removes, and how many of those the
audio-based flags (dup / short / low_vad / runaway) already catch.
A gate that removes only clips the audio flags catch adds nothing.

| WER > | removed | % | also audio-flagged | **only** WER | kept | kept corpus WER |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 646 | 58.25 | 111 | 535 | 463 | 0.0314 |
| 0.15 | 570 | 51.4 | 110 | 460 | 539 | 0.0507 |
| 0.20 | 486 | 43.82 | 107 | 379 | 623 | 0.0715 |
| 0.25 | 420 | 37.87 | 101 | 319 | 689 | 0.089 |
| 0.30 | 389 | 35.08 | 98 | 291 | 720 | 0.1018 |
| 0.40 | 313 | 28.22 | 91 | 222 | 796 | 0.1253 |
| 0.50 | 242 | 21.82 | 75 | 167 | 867 | 0.1449 |
| 0.75 | 189 | 17.04 | 70 | 119 | 920 | 0.174 |
| 1.00 | 99 | 8.93 | 44 | 55 | 1010 | 0.2061 |

### Flag combinations

| combination | count |
|---|---:|
| clean | 752 |
| high_wer | 166 |
| low_vad|no_speech | 47 |
| short | 45 |
| short|low_vad|no_speech|high_wer | 34 |
| low_vad|no_speech|high_wer | 23 |
| short|high_wer | 14 |
| short|low_vad|no_speech | 12 |
| low_vad | 11 |
| short|low_vad|no_speech|runaway|high_wer | 2 |
| dup_copy|high_wer | 1 |
| low_vad|high_wer | 1 |
| empty_asr|high_wer | 1 |

### Defect rate by emotion

| emotion | n | defective | % |
|---|---:|---:|---:|
| neutral | 470 | 152 | 32.3 |
| joy | 163 | 46 | 28.2 |
| anger | 153 | 55 | 35.9 |
| surprise | 150 | 58 | 38.7 |
| sadness | 111 | 27 | 24.3 |
| fear | 40 | 13 | 32.5 |
| disgust | 22 | 6 | 27.3 |

## test — 2610 utterances

| class | count | % | corpus WER | median WER |
|---|---:|---:|---:|---:|
| ALL | 2610 | 100.00 | 0.3823 | 0.1667 |
| dup_copy | 22 | 0.84 | 2.8852 | 3.5 |
| short | 223 | 8.54 | 3.5687 | 0.9 |
| low_vad | 254 | 9.73 | 2.5596 | 1.0 |
| no_speech | 235 | 9.0 | 2.8688 | 1.0 |
| runaway | 13 | 0.5 | 31.2264 | 41.6667 |
| empty_asr | 0 | 0.0 | None | None |
| high_wer | 637 | 24.41 | 1.7541 | 1.0 |
| wer_over_1 | 286 | 10.96 | 4.212 | 2.625 |
| **clean (no flags)** | 2223 | 85.17 | **0.2691** | 0.1667 |

### WER threshold sensitivity

How many utterances a WER gate removes, and how many of those the
audio-based flags (dup / short / low_vad / runaway) already catch.
A gate that removes only clips the audio flags catch adds nothing.

| WER > | removed | % | also audio-flagged | **only** WER | kept | kept corpus WER |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 1540 | 59.0 | 250 | 1290 | 1070 | 0.0263 |
| 0.15 | 1376 | 52.72 | 246 | 1130 | 1234 | 0.046 |
| 0.20 | 1184 | 45.36 | 239 | 945 | 1426 | 0.067 |
| 0.25 | 1033 | 39.58 | 230 | 803 | 1577 | 0.0866 |
| 0.30 | 948 | 36.32 | 229 | 719 | 1662 | 0.0993 |
| 0.40 | 763 | 29.23 | 214 | 549 | 1847 | 0.126 |
| 0.50 | 637 | 24.41 | 201 | 436 | 1973 | 0.145 |
| 0.75 | 512 | 19.62 | 188 | 324 | 2098 | 0.1721 |
| 1.00 | 286 | 10.96 | 113 | 173 | 2324 | 0.2141 |

### Flag combinations

| combination | count |
|---|---:|
| clean | 1787 |
| high_wer | 436 |
| short|low_vad|no_speech|high_wer | 79 |
| short | 78 |
| low_vad|no_speech | 71 |
| low_vad|no_speech|high_wer | 50 |
| short|high_wer | 32 |
| short|low_vad|no_speech | 23 |
| dup_copy|high_wer | 21 |
| low_vad | 14 |
| short|low_vad|no_speech|runaway|high_wer | 11 |
| low_vad|high_wer | 5 |
| runaway|high_wer | 2 |
| dup_copy|low_vad|no_speech|high_wer | 1 |

### Defect rate by emotion

| emotion | n | defective | % |
|---|---:|---:|---:|
| neutral | 1256 | 372 | 29.6 |
| joy | 402 | 138 | 34.3 |
| anger | 345 | 105 | 30.4 |
| surprise | 281 | 118 | 42.0 |
| sadness | 208 | 52 | 25.0 |
| disgust | 68 | 17 | 25.0 |
| fear | 50 | 21 | 42.0 |
