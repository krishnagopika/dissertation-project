# Fusion — combining the audio and text branches

## 1. Role in the pipeline

Fusion is the last trainable stage. Everything upstream is frozen by the time
it runs:

| stage | model | state during fusion |
|---|---|---|
| audio | Voxtral-Mini-3B (Whisper large-v3 encoder) | **frozen**, outputs cached |
| text | XLM-RoBERTa-base | **frozen**, `[CLS]` vectors cached |
| fusion | `SequenceFusion` | **trained**, ~2.1 M parameters |

Neither Voxtral nor XLM-R is loaded during fusion training. Both artefacts are
read from disk:

```
{split}_acoustic_seq.pt      "dia{d}_utt{u}" -> fp16 (T, 1280)   variable length
{split}_text_embeddings.pt   "dia{d}_utt{u}" -> fp32 (768,)
```

Training therefore costs minutes rather than hours, which is what makes a
25-run grid affordable at all.

## 2. What arrives, and why pooling is needed

The acoustic side is a **sequence**, not a vector. Voxtral's encoder emits
50 frames per second (16 kHz audio, hop 160 → 100 Hz mel, conv stride 2 →
50 Hz), so a clip of `N` samples occupies `ceil(N / 320)` frames. MELD
utterances vary from well under a second to tens of seconds, so `T` varies
per clip while the width is always 1280.

The classifier needs one vector per utterance. Something must collapse the
time axis, and *what* does the collapsing is a modelling decision rather than a
formatting detail — which is why the cache stores frames and the head does the
pooling, instead of caching a pooled vector.

### The four poolers

| pooler | output width | trainable | what it does |
|---|---|---|---|
| `masked_mean` | 1280 | no | mean over real frames only; padding excluded |
| `attention_fixed` | 1280 | no | fixed (non-learned) attention scores |
| `attention` | 1280 | yes | Bahdanau additive attention over frames |
| `attentive_stats` | **2560** | yes | weighted mean **and** weighted std, concatenated |

`masked_mean` is the control. It is the honest baseline the learned poolers
have to beat, and masking matters: Voxtral zero-pads every clip to 30 s, so an
unmasked mean over 1500 frames dilutes a two-second utterance by ~15×.

`attentive_stats` returning 2560 is why the model reads
`self.pooler.output_dim` rather than assuming `acoustic_dim`. Hard-coding 1280
would silently truncate half its output.

## 3. Architecture

```
(B, T, 1280) + mask ──► pooler ──► (B, P) ──► proj ──┐
                                                      ├──► combine ──► post ──► heads
(B, 768) text ──────────────────► proj ──────────────┘
```

Two properties are enforced so that a difference between variants can be
attributed to the variant:

**Every mechanism projects both modalities before combining.** An earlier
implementation concatenated raw 768 + 1280 dimensions straight into its MLP
while the other mechanisms projected each modality to a common width first.
That made `concat` differ from the others in *two* ways — the combination rule
and the presence of a projection — so a win or loss could not be attributed to
either. All four now share the projection stage.

**Parameter counts are reported, not assumed equal.** A gated mechanism has a
gate matrix that a sum does not. That is a real difference and should not be
hidden, but it must be *visible*, so `parameter_report()` records per-component
counts in every run's completion marker.

### The four combination mechanisms

| mechanism | rule | character |
|---|---|---|
| `concat` | `[t ; a]` → 2×hidden | lets the head learn any interaction |
| `sum` | `t + a` | forces a shared space, no extra parameters |
| `gated` | `g·t + (1−g)·a`, `g = σ(W[t;a])` | **convex** — a soft selection between modalities |
| `crossmodal` | `σ(W₁a)·t + σ(W₂a)·a` | asymmetric, each modality gates the other; **not** convex, so both can be suppressed or both passed |

### Unimodal baselines share the class

`SequenceFusion` takes `modality ∈ {both, acoustic, text}`. The unimodal
baselines are the *same class* with one branch disabled, not separate scripts.
A separate acoustic-only script would drift from the fusion model in pooling,
projection width, head architecture and initialisation, and "fusion beats
unimodal" would then be a statement about those differences rather than about
fusion.

For the same reason, Phase 2's text-only baseline is **not** the XLM-R grid
result. The XLM-R runs fine-tuned their encoder; Phase 2 uses a frozen `[CLS]`
vector through the identical head fusion uses. That is the comparison matched
on everything except modality.

## 4. The 25 runs

### Why four staged phases and not one crossed grid

A full cross is 4 poolers × 4 mechanisms × 3 text conditions × 3 learning
rates × 2 batch sizes = **288 runs**. Each phase instead fixes one variable
using the previous phase's winner, giving 25.

| phase | runs | varies | fixed from |
|---|---|---|---|
| **0** — learning rate × batch | 6 | 3 LR × 2 batch | — |
| **1** — pooling | 4 | 4 poolers, acoustic-only | Phase 0's lr/batch |
| **2** — text-only baselines | 3 | 3 text conditions | Phase 0's lr/batch |
| **3** — fusion | 12 | 4 mechanisms × 3 conditions | Phase 0 + Phase 1 |

**Phase 0** sweeps optimiser settings once, on one representative configuration
(`concat` + `attention` on ASR — it exercises both branches, the learned
pooler and the text path, so the winner is not tuned to a degenerate case).
The winner is then held **fixed** for all 19 later runs, so no comparison
between poolers, mechanisms or conditions can be explained by one arm having
had a luckier learning rate.

**Phase 1** ablates pooling with **no text at all**. Pooling is thereby
isolated with zero text confound, and its winner sets the pooler for Phase 3.

**Phase 3** crosses the four mechanisms with the three text conditions at the
fixed pooler.

Loss is held at `weighted` throughout, so the loss choice is never a confound.

### The honest limitation of staging

Staging assumes the pooler that wins for acoustic-only also wins inside fusion,
and that the best learning rate transfers across configurations. Neither is
guaranteed. A full cross would test both, and is not affordable. The assumption
is stated here rather than left implicit.

## 5. The three text conditions

| condition | text | filtering | train rows |
|---|---|---|---|
| `gold` | MELD CSV `Utterance` | none | 9,989 |
| `asr` | Voxtral transcripts | none | 9,989 |
| `asr_cleaned` | Voxtral transcripts | `text_clean` keep-list | 6,729 |

`asr` and `asr_cleaned` use the **same transcripts**. They differ only in which
utterances the keep-list admits — it is a data-quality condition, not a
different signal.

Each condition has its **own** frozen text encoder, taken from that condition's
best XLM-R checkpoint. `plain` won every condition on dev weighted F1
(gold 0.5733, asr 0.4594, asr_cleaned 0.4769), so the loss is held constant
across conditions and the encoder choice is not a confound.

### The dev set is never filtered

`train_fusion_seq.py` passes `filtered_keys_path=None` for dev unconditionally.
Filtering dev would change both the training data *and* the early-stopping
criterion, so a difference could not be attributed to either — and dev would
stop being comparable across runs. Dev is a fixed yardstick for all 25.

This is verified rather than assumed: every run records a **SHA-1 of its
sorted dev key set**, and the report fails loudly if two runs disagree.

## 6. Run isolation

The validity of every unimodal number rests on the unused branch having no
influence. That is asserted, not trusted. Before training, each unimodal run:

1. builds a probe batch with **real** acoustic data (not the padding-avoidance
   placeholder a text-only loader would otherwise emit),
2. checks the probe is non-degenerate — non-empty, and 1280 wide,
3. perturbs the unused input two ways — `randn_like` (different values) and
   `roll` along the batch (identical distribution, different pairing),
4. requires **both** the emotion and sentiment logits to be bit-identical.

Any change raises and the run aborts. A leak into the sentiment head alone
would otherwise pass silently, and a probe built from the placeholder would
pass while testing nothing.

## 7. What is recorded

Every run writes `TRAINING_COMPLETE.json` next to its checkpoint:

```
best_dev_weighted_f1   best_epoch          checkpoint_written
epochs_run             early_stopped       seed
lr                     batch_size          max_frames
train_size             dev_size            dev_key_hash
lambda_sentiment       parameters{...}     modality/pooling/fusion/loss
```

`best_epoch` is `None` until a checkpoint is actually written, and
`checkpoint_written` records that explicitly — the marker's purpose is to
certify a *usable artefact* exists, which `best_dev_weighted_f1` alone cannot
do.

Runs are separated on disk by phase:

```
checkpoints/fusion/
  phase0_lr_batch/   {cond}_lr{lr}_bs{bs}
  phase1_pooling/    acoustic_{pooler}
  phase2_textonly/   {cond}
  phase3_fusion/     {cond}_{mechanism}
```

`src/evaluation/fusion_report.py` walks that tree and writes
`results_new/fusion/summary.{csv,md,json}`. It runs after **every** phase, not
only at the end, so a walltime kill still leaves whatever finished recorded and
readable. It also fails loudly on the two conditions that invalidate a
comparison: a run that never wrote a checkpoint, and runs that disagree on the
dev key hash.

TensorBoard scalars go to `logs/<run>/tb`: train and dev loss, dev weighted and
macro F1 for both tasks, per-class emotion F1, learning rate, and epochs since
best.

## 8. Training regime

| setting | value | why |
|---|---|---|
| epochs | 30 (ceiling) | early stopping decides the actual length |
| patience | 3 | on dev **weighted F1**, not validation loss |
| scheduler | `ReduceLROnPlateau`, factor 0.5, patience `early_stop − 3` | leaves two epochs at the reduced rate, so the reduction is actually exercised rather than firing one epoch before the run dies |
| optimiser | AdamW | |
| dev shuffle | never | |
| train shuffle | seeded generator | pinned independently of global RNG, so two runs shuffle identically |

Early stopping tracks weighted F1 rather than loss because the two disagree on
MELD: dev loss frequently rises while F1 continues improving, as the model
grows confident on the majority classes.

## 9. Limitations

- **Single seed.** Every number is one run. Differences of a point or two in
  weighted F1 are within seed noise and should not be read as ordering.
- **Staged, not crossed** — see §4.
- **Frozen upstream.** Fusion cannot recover information the frozen encoders
  discarded.
- **Head-truncation.** `max_frames`, when set, keeps the *first* frames. For
  emotion the salient part of an utterance is often the end (rising anger, a
  sighed ending), so this is a real choice; the number of affected clips is
  logged per split so it is never silently large.
- **Dev is used twice** — for early stopping and for model selection across
  the grid. The selected configuration is therefore mildly optimistic on dev;
  test numbers are the honest ones.

## Reproducing

```bash
sbatch src/scripts/fusion_grid.sbatch          # all 25 runs, resumable
python3.12 src/evaluation/fusion_report.py     # rebuild the summary anytime
```

The grid skips any run whose `TRAINING_COMPLETE.json` already exists, so a
re-submission after a walltime kill continues rather than restarting.
