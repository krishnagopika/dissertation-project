# ASR Quality Report — Voxtral vs Whisper on MELD

**What this measures:** Word Error Rate (and related metrics) of the ASR transcripts
used by the text modality, scored against the MELD gold `Utterance` text. This quantifies
the noise behind the gold→ASR fusion gap documented in [`EXPERIMENTS.md`](../../EXPERIMENTS.md) §3.

**Scoring:** `src/evaluation/compute_wer.py` — text normalised (lowercase, punctuation
stripped, whitespace collapsed) on both sides before alignment. Corpus-level (micro-averaged
over words/chars), not per-utterance means.

- Scripts: [`compute_wer.py`](../../src/evaluation/compute_wer.py) (metrics), [`export_wer_pairs.py`](../../src/evaluation/export_wer_pairs.py) (side-by-side CSV)
- Voxtral results: [`wer_analysis.json`](wer_analysis.json) · error CSV: [`wer_pairs_voxtral.csv`](wer_pairs_voxtral.csv)
- Whisper results: [`wer_analysis_whisper.json`](wer_analysis_whisper.json) · error CSV: [`wer_pairs_whisper.csv`](wer_pairs_whisper.csv)

---

## 1. Metrics glossary

| Metric | Meaning | Direction |
|--------|---------|:---------:|
| **WER** | Word Error Rate = (S+D+I)/N — fraction of words wrong. Can exceed 1.0 (insertions). | ↓ |
| **MER** | Match Error Rate = (S+D+I)/(H+S+D+I) — bounded [0,1]. | ↓ |
| **WIL** | Word Information Lost = 1 − H²/(N·P) — penalises precision & recall jointly. | ↓ |
| **WIP** | Word Information Preserved = 1 − WIL. | ↑ |
| **CER** | Character Error Rate — same formula as WER but over characters. | ↓ |

*S = substitutions, D = deletions, I = insertions, H = hits, N = reference words, P = hypothesis words.*

---

## 2. Voxtral-Mini-3B — results

| Split | n | WER ↓ | MER ↓ | WIL ↓ | WIP ↑ | CER ↓ |
|-------|---|:-----:|:-----:|:-----:|:-----:|:-----:|
| train | 9989 | 0.383 | 0.326 | 0.424 | 0.576 | 0.296 |
| dev   | 1109 | 0.340 | 0.295 | 0.394 | 0.606 | 0.250 |
| **test** | 2610 | **0.424** | 0.348 | 0.447 | 0.553 | 0.336 |

- **Test WER ≈ 0.42** is the headline; **CER ≈ 0.34 < WER** → a meaningful share of errors are
  sub-word (contractions, curly quotes, minor spelling), i.e. partly cosmetic.
- Decoding was already greedy (`temperature=0.0`, `max_tokens=200` in `transcribe_all.py`
  Pass 1); the config's `voxtral_temperature`/`voxtral_max_tokens` are used only by the
  augmentation script, not transcription. Gold utterances max at 69 words « 200-token cap,
  so **no truncation**.

---

## 3. The failure mode — LLM decoder goes rogue on short clips

The corpus WER is inflated by a **small number of catastrophic hallucinations**, not by
uniform mishearing. On short/ambiguous audio, Voxtral's 3B LLM decoder stops transcribing
and instead **refuses and loops** — these are the exact short, high-emotion utterances that
carry the emotion signal.

### Highest-error utterances (MELD test)

| Key | Gold | Voxtral ASR (truncated) | WER |
|-----|------|--------------------------|----:|
| `dia124_utt8` | "Chandler!" | *"I'm not sure if I can help with that. I'm not a doctor…"* (same sentence looped ×10) | **154.0** |
| `dia95_utt6` | "Y'know?" | *"I'm not a doctor… If you're having a medical emergency, call 911…"* | 121.0 |
| `dia246_utt5` | "Hey, Peter!" | *"I'm not a doctor…"* (looped) | 77.0 |
| `dia70_utt13` | "Huh?" | *"I'm not sure if I can help with that. I'm not a doctor…"* | 48.0 |
| `dia261_utt2` | "Thanks!" | *"I'm not sure if I can help with that. I'm not a doctor…"* | 47.0 |

### Lowest-WER utterances (MELD test, gold ≥ 5 words)

| Key | Gold | Voxtral ASR | WER |
|-----|------|-------------|----:|
| `dia46_utt7` | "I stepped in something icky." | "I stepped in something icky." | 0.0 |
| `dia244_utt10` | "So how many more do you have tomorrow?" | "So how many more do you have tomorrow?" | 0.0 |
| `dia240_utt10` | "Oh that was a real person?!" | "Oh that was a real person?" | 0.0 |
| `dia244_utt15` | "Sure, your dresser is missing but this she notices." | "Sure, your dresser is missing, but this she notices." | 0.0 |
| `dia70_utt10` | "No ... the leather sticks to my ass." | "No. The leather sticks to my ass." | 0.0 |

**Takeaway:** on normal-length speech Voxtral transcribes near-perfectly; the WER damage
is concentrated in a handful of rogue-decoder blowups on 1–2 word clips. This directly
motivates the standalone Whisper baseline (§4), which has no LLM decoder to derail.

Full best/worst lists for every split are in [`wer_analysis.json`](wer_analysis.json)
(`worst_examples` / `best_examples` per split).

---

## 4. Whisper-large-v3 — results

Whisper-large-v3 **is** Voxtral's audio encoder, but uses Whisper's own seq2seq decoder
instead of the 3B LLM. Same audio, same encoder — only the decoder differs.

- Transcription: [`src/preprocessing/transcribe_whisper.py`](../../src/preprocessing/transcribe_whisper.py) · Slurm: [`src/scripts/transcribe_whisper.sbatch`](../../src/scripts/transcribe_whisper.sbatch) (`gpu:1`)

| Split | n | WER ↓ | MER ↓ | WIL ↓ | WIP ↑ | CER ↓ |
|-------|---|:-----:|:-----:|:-----:|:-----:|:-----:|
| train | 9989 | 0.338 | 0.295 | 0.388 | 0.612 | 0.258 |
| dev   | 1109 | 0.289 | 0.259 | 0.354 | 0.646 | 0.208 |
| **test** | 2610 | **0.330** | 0.289 | 0.386 | 0.614 | 0.243 |

### Highest-error utterances (MELD test)

Whisper also over-generates on short/ambiguous clips (it transcribes surrounding dialogue),
but **without the refusal loops** — the blowups are far milder (worst WER 35 vs Voxtral's 154).

| Key | Gold | Whisper ASR (truncated) | WER |
|-----|------|--------------------------|----:|
| `dia212_utt3` | "Y'know?" | *"Oh, please. Every moment is precious. You know, besides, somebody had to ride in…"* | **35.0** |
| `dia49_utt9` | "Hey!" | *"I know, me too. Hey, you know what? What if we went away for the whole weekend?…"* | 22.0 |
| `dia220_utt0` | "What's that smell?" | *"Well, this is Julie. Uh, Julie, this is Rachel. Hi…"* | 16.0 |
| `dia93_utt4` | "Yeah?" | *"Well, I don't want to talk to you, Wayne. I hate you. You ruined my life."* | 16.0 |
| `dia71_utt5` | "Oh!" | *"Uh, Joey, this is the awkward part. Oh, hey. Right. Not a problem."* | 12.0 |

### Lowest-WER utterances (MELD test, gold ≥ 5 words)

| Key | Gold | Whisper ASR | WER |
|-----|------|-------------|----:|
| `dia237_utt5` | "Eight and a half hours of aptitude tests, intelligence tests, personality tests... and what do I learn?" | *(exact match)* | 0.0 |
| `dia107_utt2` | "Do you still have that, um, Navy uniform?" | *(exact match)* | 0.0 |
| `dia106_utt2` | "Then why are you smoking?" | *(exact match)* | 0.0 |
| `dia102_utt2` | "How could you not tell me that she has hair?" | *(exact match)* | 0.0 |
| `dia102_utt0` | "You said she was bald." | *(exact match)* | 0.0 |

Full best/worst lists per split in [`wer_analysis_whisper.json`](wer_analysis_whisper.json);
all utterances with WER > 0.1 (worst-first, with audio paths) in
[`wer_pairs_whisper.csv`](wer_pairs_whisper.csv) — **9,011 rows** (train 6,562 · dev 733 · test 1,716).

---

## 5. Head-to-head (MELD test)

| System | WER ↓ | MER ↓ | WIL ↓ | WIP ↑ | CER ↓ | worst WER | utts WER>0.1 |
|--------|:-----:|:-----:|:-----:|:-----:|:-----:|:--------:|:-----------:|
| Voxtral-Mini-3B | 0.424 | 0.348 | 0.447 | 0.553 | 0.336 | 154.0 | 1,772 |
| **Whisper-large-v3** | **0.330** | **0.289** | **0.386** | **0.614** | **0.243** | **35.0** | **1,716** |
| *absolute gain* | *−0.094* | *−0.059* | *−0.061* | *+0.061* | *−0.093* | | |

**Whisper wins on every metric** — test WER **0.330 vs 0.424 (≈9.4 pts / 22% relative)**.
Same encoder, so the entire gap is the **decoder**: Voxtral's 3B LLM derails on short clips
(refusals + repetition loops, WER up to 154), while Whisper's seq2seq decoder degrades far
more gracefully (worst 35). The count of WER>0.1 utterances is similar (1,716 vs 1,772) —
i.e. both mishear a comparable *number* of clips, but Voxtral's errors are far more *severe*.
This is the argument for using Whisper (or Whisper transcripts) for the text leg rather than
Voxtral's LLM output.

---

*Generated by `src/evaluation/compute_wer.py` and `export_wer_pairs.py`.*
