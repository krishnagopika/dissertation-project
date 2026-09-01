# Runbook

Copy-paste commands. Run everything from `/dcs/pg25/u5734759/dissertation-project`.

```bash
cd /dcs/pg25/u5734759/dissertation-project
```

Check space before any big job — soft quota 120 G, hard 132 G:

```bash
quota -s -u u5734759 | tail -2
```

---

## 1. Download Voxtral-Small consolidated weights

vLLM picks its loader by looking for `consolidated*.safetensors`. Without that
file it falls back to the HF loader, which cannot serve Voxtral — see
`DECISIONS.md` ADR-006. The HF-format shards were deleted; this fetches the one
file that works.

**Must run on the login node** — compute nodes have no internet.

```bash
export HF_HOME=/dcs/large/u5734759/hf_cache
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

setsid nohup /dcs/large/u5734759/venv/bin/python3.12 \
  /dcs/large/u5734759/dl_small.py \
  > /dcs/large/u5734759/dl_small.log 2>&1 < /dev/null &
```

`setsid` detaches it, so it survives your SSH session closing. Resumes from any
existing `.incomplete` — safe to re-run after a disconnect.

Progress (48.5 GB total):

```bash
ls -la /dcs/large/u5734759/hf_cache/hub/models--mistralai--Voxtral-Small-24B-2507/blobs/*.incomplete \
  | awk '{printf "%.2f GB of 48.5\n", $5/1e9}'
tail -3 /dcs/large/u5734759/dl_small.log
pgrep -f dl_small.py || echo "NOT RUNNING"
```

Done when the log says `DONE in … min` and the snapshot lists
`consolidated.safetensors`:

```bash
ls -la /dcs/large/u5734759/hf_cache/hub/models--mistralai--Voxtral-Small-24B-2507/snapshots/*/
```

Expect `consolidated.safetensors`, `params.json`, `tekken.json`.

> Slow (~3 MB/s) because it is a single 48.5 GB file — `max_workers` parallelises
> across files, not within one. `pip install hf_transfer` +
> `HF_HUB_ENABLE_HF_TRANSFER=1` would chunk it across parallel streams, ~5-10x,
> at the cost of adding a dependency mid-project.

---

## 2. Extract transcripts + acoustics

One script, both models, selected by `MODEL`. Each writes only under
`data/meld_extracted/$MODEL/`, so the two cannot collide and the legacy caches
are untouched. **Unfiltered** — all 13,708 utterances, no VAD or WER gate.

### Mini

```bash
sbatch --gres=gpu:1 --export=ALL,MODEL=mini \
  --job-name=extract_mini src/scripts/extract_meld.sbatch
```

### Small — only after the download completes

```bash
sbatch --gres=gpu:2 --export=ALL,MODEL=small \
  --job-name=extract_small src/scripts/extract_meld.sbatch
```

`gpu:2` because Small is 24 B; its config sets `tensor_parallel_size: 2`. The
NCCL env vars in the sbatch are what got it past an earlier init hang.

### Watch it

```bash
squeue -u u5734759
J=<jobid>
grep -oE "'(train|dev|test)' \| [0-9]+ / [0-9]+ transcribed \| [0-9]+ seqs" logs/slurm_$J.err | tail -3
sed -n '/=== VERIFY/,$p' logs/slurm_$J.out
```

Healthy output looks like `158 frames avg` — that is 158/50 ≈ 3.2 s, matching
MELD's real utterance length. A constant 1500 means padding is being captured
instead of audio.

### Resuming

Safe to re-submit: a split is skipped only when its transcripts exist **and**
its sequence file passes an archive-integrity check. A truncated file from a
killed job is detected and re-extracted rather than silently accepted.

### Output

```
/dcs/large/u5734759/data/meld_extracted/{mini,small}/
├── transcripts/{split}_transcripts.json          "diaD_uttU" -> str
├── acoustic/{split}_acoustic_seq.pt              fp16 (T, 1280), padding stripped
├── acoustic/{split}_embeddings_maskedmean.pt     fp32 (1280,)
├── acoustic/{split}_unmatched.json               keys the hook missed
├── filter_metadata/                              step 3 writes here
└── text/                                         XLM-R embeddings, later
```

Roughly 6.1 GB per model for all three splits.

---

## 3. WER + VAD report

Two steps. The first measures, the second reports. **Neither filters anything.**

### 3a. Compute per-utterance metadata

Silero-VAD `speech_ratio` + Voxtral WER for every utterance. Needs the
transcripts from step 2.

```bash
sbatch --gres=gpu:1 --export=ALL,MODEL=mini \
  --job-name=wer_vad_mini src/scripts/wer_vad.sbatch
```

Writes `data/meld_extracted/mini/filter_metadata/{split}_filter_metadata.json`,
one record per utterance:

```json
{"key": "dia0_utt0", "vox_wer": 0.14, "speech_ratio": 0.87,
 "duration_sec": 2.4, "gold_words": 7, "vox_words": 7}
```

### 3b. Build the report

CPU only, seconds to run — 3a already did the work.

```bash
/dcs/large/u5734759/venv/bin/python3.12 src/evaluation/wer_vad_report.py \
    --config src/configs/extract_mini.yaml --splits train dev test
```

Writes to `results/extract_mini/wer_vad/`:

| File | Contents |
|---|---|
| `report.md` | all splits, human-readable — start here |
| `{split}_utterances.csv` | one row per utterance + emotion, gold text, ASR text |
| `{split}_summary.json` | percentiles, per-emotion WER, threshold table |
| `all_splits_summary.json` | the three summaries together |

The CSV joins WER and `speech_ratio` to each utterance's emotion label and both
texts side by side, so you can sort by WER and read what actually went wrong.

`report.md` includes a **"what a filter would keep"** table for WER ≤ 0.10 …
0.40 with VAD ≥ 0.20. It is informational — the cost of a threshold, visible
before you commit. `apply_filter.py` is the step that actually creates
keep-lists.

For Small, swap the config: `--config src/configs/extract_small.yaml`.

---

## 4. Order of operations

```
1. download consolidated.safetensors      (login node, ~4 h, background)
2. sbatch extract_meld  MODEL=mini        (~1 h)     ─┐ independent of 1
3. sbatch wer_vad       MODEL=mini        (~1 h)      │
4. wer_vad_report.py    extract_mini      (seconds)  ─┘
5. sbatch extract_meld  MODEL=small       needs 1
6. sbatch wer_vad       MODEL=small       needs 5
7. wer_vad_report.py    extract_small
```

Steps 2-4 do not wait on the download. Chain jobs with
`--dependency=afterok:<jobid>` to queue them back-to-back.

---

## Troubleshooting

**`VLLM_ALLOW_INSECURE_SERIALIZATION` error.** `llm.apply_model()` ships the
capture hook to the worker; vLLM refuses that under its default msgpack
serialisation. The sbatch sets it. Only relevant if running the python directly.

**Job dies with no error, partway through.** Almost certainly memory. The
extract sbatch requests `--mem=96G` for this reason — job 8086 had no `--mem`,
took the partition default, and was OOM-killed mid-`torch.save`. Check with
`sacct -j <jobid> --format=JobID,State,MaxRSS,ReqMem,Elapsed`.

**`frames avg` is 1500.** The hook is capturing vLLM's 30 s zero-padding rather
than real audio. See `POSTMORTEMS.md` PM-008.

**Small fails at load.** Check that `consolidated.safetensors` is actually
present in the snapshot. Without it vLLM silently selects the HF loader and
fails with `downsample_factor` or `'NoneType' object is not iterable`.

**Over quota.** `quota -s -u u5734759`. Reclaimable: strip optimizer state from
finished checkpoints (`src/scripts/strip_optimizer_state.py <ckpt> --apply`,
~2.2 GB per XLM-R checkpoint — evaluable afterwards but not resumable).
