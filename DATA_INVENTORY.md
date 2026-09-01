# MELD Data Inventory — where data is lost in the pipeline

Audited 2026-08-03. Sources: `meld_raw/*.csv`, `meld_transcripts/*.json`,
`meld_embeddings/*.pt`, `logs/mini/apply_filter.log`, `logs/mini/compute_filter_metadata.log`.

## Stage 0 — Raw MELD (baseline)

| Split | Utterances | Dialogues | Median utt/dialogue |
|-------|-----------:|----------:|--------------------:|
| train | 9,989 | 1,038 | 9 |
| dev   | 1,109 |   114 | 10 |
| test  | 2,610 |   280 | 9 |
| **total** | **13,708** | **1,432** | — |

These match the official MELD release counts — nothing lost at CSV load.

## Stage 1 — Voxtral transcription (Mini)

| Split | Keys written | Blank transcripts | Loss |
|-------|-------------:|------------------:|-----:|
| train | 9,989 | 1 | 0.01% |
| dev   | 1,109 | 1 | 0.09% |
| test  | 2,610 | 0 | 0.00% |

**Effectively lossless (2 utterances of 13,708).** The "1 missing" in the run log
is a missing source audio file, stored as an empty string rather than dropped.

## Stage 2 — Acoustic embeddings (Whisper encoder, 1280-d)

| Split | Keys | Loss |
|-------|-----:|-----:|
| train | 9,989 | 0 |
| dev   | 1,109 | 0 |
| test  | 2,610 | 0 |

**Zero loss.** Every utterance has a cached acoustic vector.

## Stage 3 — WER/VAD filtering ← THIS IS WHERE THE DATA GOES

All policies use `source=vox`, `vad_min=0.20`. Kept counts (and % of raw):

| Policy | train | dev | test |
|--------|------:|----:|-----:|
| wer10 | 1,587 (15.9%) | 176 (15.9%) | 412 (15.8%) |
| wer15 | 2,071 (20.7%) | 227 (20.5%) | 545 (20.9%) |
| wer20 | 2,829 (28.3%) | 302 (27.2%) | 736 (28.2%) |
| **wer25** | **3,539 (35.4%)** | **379 (34.2%)** | **909 (34.8%)** |
| wer25_eval_wer40_train | 5,174 (51.8%) | 379 (34.2%) | 909 (34.8%) |

At the wer25 policy the pipeline keeps **4,827 of 13,708 utterances = 35.2%**.
**We discard ~65% of MELD.** Dominant exclusion reason is `wer_above_threshold`
(e.g. train wer25: 5,499 WER / 795 WER+VAD / 156 VAD-only).

### Per-emotion survival at wer25 — the filter is NOT class-neutral

train: neutral 1901/4710 (40%) · disgust 112/271 (41%) · sadness 252/683 (37%)
· fear 87/268 (32%) · anger 342/1109 (31%) · surprise 372/1205 (31%) · joy 473/1743 (27%)

dev: neutral 193/470 (41%) · sadness 43/111 (39%) · joy 51/163 (31%)
· anger 43/153 (28%) · **disgust 5/22 (23%)** · surprise 35/150 (23%) · **fear 9/40 (22%)**

test: neutral 517/1256 (41%) · sadness 74/208 (36%) · disgust 24/68 (35%)
· anger 108/345 (31%) · surprise 76/281 (27%) · **fear 13/50 (26%)** · joy 97/402 (24%)

## Two problems this creates

**1. Eval sets become too small to trust.** Filtered dev = 379 utterances, with
**5 disgust and 9 fear**. Test = 909, with **24 disgust and 13 fear**. A per-class
F1 computed over 5–13 samples is noise, not a measurement — one flipped prediction
moves disgust F1 by ~0.1. Any rare-class claim on filtered dev/test is unreportable.

**2. The filter shifts the label distribution.** It keeps neutral at 40–41% but
joy at only 24–27%. So filtering *worsens* the neutral skew that is already the
main problem on MELD, and filtered scores are **not comparable** to unfiltered
ones — a WF1 gain may just be more neutral in the eval set.

Recommended: keep filtering as a *training-data* intervention only, and always
report on the **full unfiltered dev/test** so numbers stay comparable and rare
classes retain enough support.
