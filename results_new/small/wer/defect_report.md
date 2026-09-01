# WER / VAD defect analysis

Source: `results_new/small/wer`  
Measurement only — nothing filtered.

Flags are additive: a clip can be short AND duplicated AND runaway.

## train — 9989 utterances

| class | count | % | corpus WER | median WER |
|---|---:|---:|---:|---:|
| ALL | 9988 | 100.00 | 0.3748 | 0.1667 |
| dup_copy | 19 | 0.19 | 1.5842 | 0.3333 |
| short | 942 | 9.43 | 2.0106 | 0.0 |
| low_vad | 950 | 9.51 | 2.1658 | 0.5714 |
| no_speech | 862 | 8.63 | 2.5071 | 0.6667 |
| runaway | 37 | 0.37 | 16.0058 | 20.0 |
| empty_asr | 1 | 0.01 | 1.0 | 1.0 |
| high_wer | 2280 | 22.83 | 1.6141 | 1.0 |
| wer_over_1 | 1039 | 10.4 | 3.5847 | 2.5 |
| **clean (no flags)** | 8464 | 84.73 | **0.2839** | 0.1538 |

### WER threshold sensitivity

How many utterances a WER gate removes, and how many of those the
audio-based flags (dup / short / low_vad / runaway) already catch.
A gate that removes only clips the audio flags catch adds nothing.

| WER > | removed | % | also audio-flagged | **only** WER | kept | kept corpus WER |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 5696 | 57.02 | 832 | 4864 | 4293 | 0.0246 |
| 0.15 | 5088 | 50.94 | 819 | 4269 | 4901 | 0.0434 |
| 0.20 | 4338 | 43.43 | 793 | 3545 | 5651 | 0.066 |
| 0.25 | 3804 | 38.08 | 757 | 3047 | 6185 | 0.083 |
| 0.30 | 3515 | 35.19 | 750 | 2765 | 6474 | 0.0953 |
| 0.40 | 2843 | 28.46 | 693 | 2150 | 7146 | 0.1214 |
| 0.50 | 2280 | 22.83 | 604 | 1676 | 7709 | 0.1447 |
| 0.75 | 1770 | 17.72 | 542 | 1228 | 8219 | 0.1788 |
| 1.00 | 1039 | 10.4 | 256 | 783 | 8950 | 0.2141 |

### Flag combinations

| combination | count |
|---|---:|
| clean | 6789 |
| high_wer | 1675 |
| short | 441 |
| low_vad|no_speech | 275 |
| short|low_vad|no_speech|high_wer | 254 |
| low_vad|no_speech|high_wer | 177 |
| short|low_vad|no_speech | 131 |
| short|high_wer | 104 |
| low_vad | 63 |
| low_vad|high_wer | 24 |
| low_vad|no_speech|runaway|high_wer | 13 |
| runaway|high_wer | 11 |
| short|low_vad|no_speech|runaway|high_wer | 10 |
| dup_copy | 8 |
| dup_copy|high_wer | 7 |
| low_vad|runaway|high_wer | 1 |
| empty_asr|high_wer | 1 |
| short|runaway|high_wer | 1 |
| dup_copy|runaway|high_wer | 1 |
| dup_copy|low_vad|no_speech|high_wer | 1 |

### Defect rate by emotion

| emotion | n | defective | % |
|---|---:|---:|---:|
| neutral | 4710 | 1463 | 31.1 |
| joy | 1743 | 602 | 34.5 |
| surprise | 1205 | 476 | 39.5 |
| anger | 1109 | 331 | 29.8 |
| sadness | 683 | 171 | 25.0 |
| disgust | 271 | 77 | 28.4 |
| fear | 268 | 80 | 29.9 |

## dev — 1109 utterances

| class | count | % | corpus WER | median WER |
|---|---:|---:|---:|---:|
| ALL | 1109 | 100.00 | 0.3298 | 0.1429 |
| dup_copy | 1 | 0.09 | 1.4 | 1.4 |
| short | 107 | 9.65 | 2.8968 | 0.5 |
| low_vad | 130 | 11.72 | 1.7165 | 0.5 |
| no_speech | 118 | 10.64 | 2.0985 | 0.5 |
| runaway | 4 | 0.36 | 33.1429 | 47.0 |
| empty_asr | 1 | 0.09 | 1.0 | 1.0 |
| high_wer | 244 | 22.0 | 1.7209 | 1.0 |
| wer_over_1 | 96 | 8.66 | 4.2331 | 2.0 |
| **clean (no flags)** | 918 | 82.78 | **0.2309** | 0.1429 |

### WER threshold sensitivity

How many utterances a WER gate removes, and how many of those the
audio-based flags (dup / short / low_vad / runaway) already catch.
A gate that removes only clips the audio flags catch adds nothing.

| WER > | removed | % | also audio-flagged | **only** WER | kept | kept corpus WER |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 624 | 56.27 | 110 | 514 | 485 | 0.0271 |
| 0.15 | 550 | 49.59 | 108 | 442 | 559 | 0.0454 |
| 0.20 | 478 | 43.1 | 105 | 373 | 631 | 0.0642 |
| 0.25 | 415 | 37.42 | 100 | 315 | 694 | 0.0813 |
| 0.30 | 383 | 34.54 | 97 | 286 | 726 | 0.0951 |
| 0.40 | 313 | 28.22 | 91 | 222 | 796 | 0.1189 |
| 0.50 | 244 | 22.0 | 76 | 168 | 865 | 0.1392 |
| 0.75 | 193 | 17.4 | 71 | 122 | 916 | 0.1655 |
| 1.00 | 96 | 8.66 | 33 | 63 | 1013 | 0.1958 |

### Flag combinations

| combination | count |
|---|---:|
| clean | 751 |
| high_wer | 167 |
| low_vad|no_speech | 44 |
| short | 43 |
| short|low_vad|no_speech|high_wer | 30 |
| low_vad|no_speech|high_wer | 25 |
| short|low_vad|no_speech | 16 |
| short|high_wer | 15 |
| low_vad | 11 |
| short|low_vad|no_speech|runaway|high_wer | 2 |
| low_vad|no_speech|runaway|high_wer | 1 |
| dup_copy|high_wer | 1 |
| low_vad|high_wer | 1 |
| short|runaway|high_wer | 1 |
| empty_asr|high_wer | 1 |

### Defect rate by emotion

| emotion | n | defective | % |
|---|---:|---:|---:|
| neutral | 470 | 147 | 31.3 |
| joy | 163 | 51 | 31.3 |
| anger | 153 | 53 | 34.6 |
| surprise | 150 | 61 | 40.7 |
| sadness | 111 | 26 | 23.4 |
| fear | 40 | 15 | 37.5 |
| disgust | 22 | 5 | 22.7 |

## test — 2610 utterances

| class | count | % | corpus WER | median WER |
|---|---:|---:|---:|---:|
| ALL | 2610 | 100.00 | 0.3619 | 0.1538 |
| dup_copy | 22 | 0.84 | 3.2295 | 4.5 |
| short | 223 | 8.54 | 2.5183 | 0.6667 |
| low_vad | 254 | 9.73 | 2.2569 | 1.0 |
| no_speech | 235 | 9.0 | 2.5276 | 1.0 |
| runaway | 7 | 0.27 | 29.8947 | 43.6667 |
| empty_asr | 0 | 0.0 | None | None |
| high_wer | 625 | 23.95 | 1.7394 | 1.0 |
| wer_over_1 | 279 | 10.69 | 4.0422 | 3.0 |
| **clean (no flags)** | 2223 | 85.17 | **0.2595** | 0.1429 |

### WER threshold sensitivity

How many utterances a WER gate removes, and how many of those the
audio-based flags (dup / short / low_vad / runaway) already catch.
A gate that removes only clips the audio flags catch adds nothing.

| WER > | removed | % | also audio-flagged | **only** WER | kept | kept corpus WER |
|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 1490 | 57.09 | 243 | 1247 | 1120 | 0.0244 |
| 0.15 | 1316 | 50.42 | 237 | 1079 | 1294 | 0.0444 |
| 0.20 | 1127 | 43.18 | 229 | 898 | 1483 | 0.0653 |
| 0.25 | 990 | 37.93 | 224 | 766 | 1620 | 0.0829 |
| 0.30 | 917 | 35.13 | 222 | 695 | 1693 | 0.0942 |
| 0.40 | 744 | 28.51 | 209 | 535 | 1866 | 0.1184 |
| 0.50 | 625 | 23.95 | 202 | 423 | 1985 | 0.1375 |
| 0.75 | 496 | 19.0 | 183 | 313 | 2114 | 0.1655 |
| 1.00 | 279 | 10.69 | 98 | 181 | 2331 | 0.2031 |

### Flag combinations

| combination | count |
|---|---:|
| clean | 1800 |
| high_wer | 423 |
| short|low_vad|no_speech|high_wer | 83 |
| short | 80 |
| low_vad|no_speech | 63 |
| low_vad|no_speech|high_wer | 54 |
| short|high_wer | 30 |
| short|low_vad|no_speech | 29 |
| dup_copy|high_wer | 21 |
| low_vad | 13 |
| low_vad|high_wer | 6 |
| low_vad|no_speech|runaway|high_wer | 4 |
| runaway|high_wer | 2 |
| dup_copy|low_vad|no_speech|high_wer | 1 |
| short|low_vad|no_speech|runaway|high_wer | 1 |

### Defect rate by emotion

| emotion | n | defective | % |
|---|---:|---:|---:|
| neutral | 1256 | 360 | 28.7 |
| joy | 402 | 139 | 34.6 |
| anger | 345 | 105 | 30.4 |
| surprise | 281 | 115 | 40.9 |
| sadness | 208 | 55 | 26.4 |
| disgust | 68 | 17 | 25.0 |
| fear | 50 | 19 | 38.0 |
