# Postmortems

Incidents where something was lost, silently wrong, or reported wrong — and what
would have caught it. Kept separate from `DECISIONS.md` (which records intended
design) because these are *failures*, and the value is in the pattern across
them, not in any single entry.

Written blamelessly but not vaguely: an incident with no identified root cause
has not been understood, and will recur.

**Severity:** `S1` irrecoverable loss · `S2` wrong results reported as correct ·
`S3` wasted time, no bad output escaped

---

## PM-001 — bc-LSTM checkpoint destroyed by overwrite · S1

**Date:** 2026-08-06 06:24 · **Recovered:** no

### What happened

`/dcs/large/u5734759/checkpoints/mini/context_bclstm/best_context.pt` went from
67,223,824 bytes (2026-07-30) to 18,918,736 bytes during job 1410944
(06:22:40–06:24:03). The original weights are gone. No backups, no snapshots.

### Root cause

`train_context.py` wrote every run to a **fixed** `context_bclstm/` directory.
`--tag` only renamed the output results JSON — it did not affect the checkpoint
path. Four ablation configs were run that all shared
`checkpoint_dir: /dcs/large/u5734759/checkpoints/mini`, so each run silently
overwrote the previous run's weights.

### Contributing factor

The failure was invisible because the *results* files were correctly named per
tag. Everything downstream looked right; only the weights were being clobbered.

### Mitigating fact

Both of the user's own recorded results already pointed at the same fixed path,
so the filtered and unfiltered models had **never coexisted on disk**. The
comparison being attempted was impossible before the incident too, and required
retraining regardless. bc-LSTM retrains in ~1 minute.

### Fix applied

```python
# The tag MUST be part of the checkpoint directory, not just the results JSON.
ckpt_dir = Path(config["training"]["checkpoint_dir"]) / f"context_bclstm{tag}"
```

### Prevention

- Any script writing a checkpoint must derive the **directory** from the run
  identifier, not just the filename of its metrics output.
- Refuse to overwrite an existing checkpoint unless `--overwrite` is passed.
- Before running a sweep, confirm each cell writes to a distinct path — check the
  path, not the results filename.

---

## PM-002 — Bare `except` blanked every WER value · S2

**Date:** 2026-08-06 · **File:** `src/evaluation/dialect_probe.py::safe_wer`

### What happened

Every row of the dialect annotation sheet had an empty `wer` column. The pipeline
reported success.

### Root cause

The call used `jiwer.wer(..., truth_transform=...)`. The correct kwarg is
`reference_transform=`. The wrong one raises `TypeError` — which was caught by a
bare `except Exception: return ""`, converting a hard programming error into a
plausible-looking blank cell for all 110 rows.

### The real lesson

The bug was a one-word typo. What made it *dangerous* was the exception handler
written to tolerate bad data, which also silently tolerated broken code. A
`TypeError` from a keyword argument is never a data problem — it cannot be
recovered from and should never be caught.

### Fix applied

Corrected the kwarg; verified against known cases (identical strings → 0.0,
1-of-6 wrong → 0.1667).

### Prevention

- Catch specific exceptions, not `Exception`.
- Where a broad catch is genuinely needed, **log the exception** rather than
  swallowing it — a silent `except` should be treated as a defect in review.
- Sanity-check any all-empty or all-identical output column before trusting a run.

---

## PM-003 — Sampler key collision silently dropped clips · S2

**Date:** 2026-08-06 · **File:** `src/preprocessing/sample_dialects.py`

### What happened

The manifest listed 110 rows but only 108 WAV files existed on disk.

### Root cause

Files were keyed on `line_id`, which is the **prompt** id, not a clip id. Several
speakers within one accent config read the same prompt, so two clips shared a key
and the second write silently overwrote the first.

### How it was caught

Only by comparing the manifest row count against `ls | wc -l`. Nothing in the
pipeline complained.

### Fix applied

```python
key = f"{config}__{row['speaker_id']}__{row['line_id']}"
```

### Prevention

- Never assume a corpus field is unique because it looks like an identifier —
  verify with a count of distinct values against the row count.
- Assert `len(set(keys)) == len(keys)` when building any keyed artefact.
- Cross-check written-file count against intended-row count at the end of any
  extraction job.

---

## PM-004 — "Evaluate the old models" silently retrained them · S2

**Date:** 2026-08-06 · **Detected by:** the user, twice — *"thi sis worng"*

### What happened

The request was to evaluate *existing* checkpoints on a common test set. The 2×2
grid that was run instead trained a fresh model in every cell. Each reported
number came from a different random init with a different best epoch, so nothing
in the table was comparable to anything else — while looking exactly like a valid
comparison.

### Root cause

`train_context.py` always trains before it scores; it has no eval-only path. It
was used as though it did.

### Contributing factor

Compounded by PM-001: even the intention to evaluate old checkpoints could not
have worked, because those checkpoints had already been overwritten.

### Fix applied

Wrote `src/evaluation/eval_context.py`, which trains nothing — it loads a
checkpoint, reads the architecture **from the state dict** rather than the config
(so a config/checkpoint mismatch cannot silently score a differently-shaped
model), and writes `"eval_only": true` into its results JSON as provenance.

### Prevention

- Results JSON must record provenance: `eval_only`, `checkpoint`, `trained_epoch`.
- Before reporting a comparison, confirm the cells actually share a model —
  reading the provenance fields, not assuming from the script name.
- When asked to *evaluate*, verify an evaluation path exists before running
  anything.

---

## PM-005 — Underpowered pilot produced an overstated headline · S2

**Date:** 2026-08-06

### What happened

A 110-clip pilot gave per-accent WER of Midlands 0.036 → Irish 0.238, reported as
a **6.6×** accent gap. The full 17,877-clip run gave Midlands 0.064 → Irish 0.123
— a **2×** gap. Irish had n=10 in the pilot.

### Root cause

Per-accent cells of 10–20 clips. WER is high-variance per utterance, so cell
means at that size are dominated by noise. The direction of the finding survived;
the magnitude did not.

### Why it mattered

The 6.6× figure was on its way into a dissertation as a headline result. The
qualitative claim (Irish degrades most) held up. The number would not have.

### Prevention

- Report `n` beside every per-group statistic, always.
- Treat any cell with n < 30 as directional only — never quote its magnitude.
- Run the full corpus before a number enters the write-up, when the full corpus
  is affordable. Here it cost ~1.5 h.

---

## PM-006 — Result from one model reported as if it covered another · S2

**Date:** 2026-08-06 · **Detected by:** the user — *"we had the resulst from the
new modle and it reached a lot more than the 0.5 someyhing"*

### What happened

Text-only XLM-R comparisons were run, and the conclusion "filtering-based
training didn't work" was stated without qualifying that it applied only to the
text branch. The user had bc-LSTM results showing otherwise.

### Root cause

Scope error in reporting, not in code. The experiment was valid; the claim
generalised past its evidence.

### Prevention

- State the model and the test set in the same sentence as any metric.
- A claim about "the pipeline" requires evidence from the pipeline —
  Voxtral + XLM-R + fusion + bc-LSTM — not from one branch.

---

## PM-007 — Attention dropout broke the variance identity · S3

**Date:** 2026-08-23 · **Caught in review before any training run**

Full detail in `DECISIONS.md` → CR-001. Recorded here for the pattern only.

### The pattern worth keeping

A `clamp_min` written as an fp16 precision guard was silently absorbing a
**structural** error — 48% of variance elements going negative because
post-dropout weights no longer summed to one. The guard made a real bug look like
rounding.

### A second defect in the fix itself

Review of the fix found the guard condition and the divisor floor using
*different* thresholds — `total > 0` against `clamp_min(1e-5)` — so any row with
`0 < total < 1e-5` took the division branch with a silently altered divisor,
producing weights summing to `total/1e-5`. The function violated its own
invariant in exactly the regime it was written to protect. See CR-002.

### Prevention

- A defensive guard should be *rare* in normal operation. Count how often it
  fires; if it fires routinely, it is masking a bug, not preventing one.
- When an identity depends on a precondition (`Σw = 1`), assert the precondition
  rather than clamping its violated consequence.
- **A guard's condition and its saturating constant must be the same value.** If
  the branch test and the clamp disagree, there is a window where the branch is
  taken and the clamp silently alters the result.
- Give each numerical guard its own named constant. Reusing one (`_VAR_EPS` as
  both a variance floor and a weight-sum floor) means tuning one path silently
  moves the other's boundary.
- This was caught because the reviewer said "test rather than blindly assume."
  Both the bug and the fix were then demonstrated numerically. Do that first —
  and re-review the fix, since this one shipped with a defect of its own.

---

## PM-008 — Single-pass hook captured everything and matched nothing · S3

**Date:** 2026-08-23 · **Caught by:** the smoke test, before any real cache was written

### What happened

The first end-to-end run of the single-pass pipeline transcribed all 24 smoke
clips correctly and captured all 24 acoustic sequences — then paired **zero** of
them to an utterance key. Every clip was zero-filled.

### Root cause

One fact explained both symptoms: **vLLM pads every waveform to its 30 s chunk
before the encoder runs.** So:

- the hook's `arr.shape[0]` was 480,000 for every clip, making
  `ceil(480000/320) = 1500` and the truncation a no-op — hence `1500 frames avg`
  and a 92 MB capture for 24 clips (~8× the intended size);
- the fingerprint was computed over *padded* audio, which the driver's unpadded
  decode can never reproduce — hence 24/24 unmatched.

The design assumed the hook would see the same waveform the driver sent. It sees
a preprocessed one.

### Why this was survivable

Two guards written earlier did their job. The fingerprint scheme fails *closed*:
a mismatch produces a miss, never a wrong pairing, so no clip was silently given
another clip's acoustics. And unmatched keys were logged as ERROR and written to
`{split}_unmatched.json` rather than passing as valid zero vectors. The smoke
test ran against a throwaway output directory, so no real cache was touched.

### Wider consequence

ADR-003 attributed the padding defect to HF's `padding="max_length"`. That was
incomplete: **vLLM pads too**, so the defect is a property of Voxtral's audio
front-end, not of one loading path. Any pooling over raw encoder output — in
either framework — averages over ~30 s regardless of utterance length.

### Prevention

- Do not assume a framework hands your data to an inner module unmodified.
  Verify what the hook actually receives before building on it.
- A capture whose count looks right (`24 seqs`) is not a working capture. The
  per-batch log said `1500 frames avg`, which was the tell — a physically
  implausible constant across clips of obviously different durations. **Log a
  derived physical quantity, not just a count**, so implausibility is visible.

---

## Cross-cutting patterns

Ranked by how many incidents they explain.

1. **Silent overwrite** (PM-001, PM-003). Two of the worst incidents were writes
   to a path that already held something. Neither raised an error. *Assert
   uniqueness before writing; refuse to clobber without an explicit flag.*

2. **Exception handlers that hide programming errors** (PM-002, and the same
   `except Exception` pattern still present in `transcribe_all.py` and
   `dialect_probe.py`). *Catch narrowly; log what you swallow.*

3. **Numbers reported without provenance** (PM-004, PM-005, PM-006). Every case
   involved a metric whose origin — which model, which test set, what n — was not
   carried alongside it. *Provenance travels with the number or the number is not
   reportable.*

4. **Guards that mask rather than prevent** (PM-007). *If it fires often, it is
   hiding something.*

---

## Still unresolved

- **`mini` and `mini_aug` checkpoints have identical weights.** Both report
  epoch=4 and identical probed tensor sums (`word_embeddings.weight sum =
  2086090.750000`); `mini_wer25_filter` differs (epoch=9, sum=2086510.000000).
  Project notes describe `mini_aug` as the best text checkpoint used by late
  fusion. If the two are genuinely the same file, the augmentation run either did
  not happen or did not save — and any result attributed to `mini_aug` is
  actually `mini`. **Not investigated.**

- **Voxtral-Small never ran.** Root cause identified as a packaging mismatch, not
  model size or GPU capacity: Mini's snapshot ships Mistral-format `params.json`
  (which carries `downsample_factor: 4`), Small ships an HF `config.json` that
  routes into a loader path failing first on
  `AttributeError: 'VoxtralEncoderConfig' object has no attribute 'downsample_factor'`
  and then on `TypeError: 'NoneType' object is not iterable`. Patching HF keys one
  at a time was abandoned; the recommendation is a Mistral-format re-download
  (~46 GB). **Parked by the user.**
