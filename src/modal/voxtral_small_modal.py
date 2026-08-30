"""Voxtral-Small extraction on Modal — single pass, one GPU, no tensor parallelism.

Why this exists
---------------
Voxtral-Small is 24.3 B parameters ≈ 48.5 GB in bf16. A Warwick L40S has 46 GB,
so Small could only run split across two GPUs (``tensor_parallel_size=2``) — and
on that node every tp=2 generation deadlocks. Eleven jobs established that the
hang is in the tensor-parallel path itself, not in anything project-specific:

    hook (4 variants, incl. a literal no-op) ..... all hang
    audio (text-only prompt, no hook) ............ hangs
    fork vs spawn ................................ both hang
    async scheduling disabled .................... hangs
    GPUDirect RDMA disabled ...................... hangs
    NCCL_SHM_DISABLE / OMP_NUM_THREADS / socket .. hangs
    NCCL init itself ............................. SUCCEEDS ("Connected all
                                                   rings", "Connected all trees")

The signature — communicator init completing, then the first real collective
spin-waiting with both GPUs at 100% and both workers burning a full core — is
NVIDIA/nccl#2079, where PCIe ACS/IOMMU breaks peer traffic on two Ada-class
GPUs without NVLink. That is a host-level fix, not a config one.

An 80 GB GPU removes the problem rather than working around it: Small fits
whole, ``tensor_parallel_size=1``, no NCCL collectives, nothing that can
deadlock. A100-80GB is the cheapest such GPU on Modal.

Design is unchanged from Mini
-----------------------------
This is the SAME single-pass architecture, not a reimplementation. One Voxtral
load produces both artefacts: a forward hook on the Whisper encoder captures
the ``(T, 1280)`` sequences vLLM computes for audio tokens and would otherwise
discard, so the acoustic branch costs nothing beyond the transcription pass.

It imports ``src.preprocessing.acoustic_hook`` — Mini's ORIGINAL hook, proven
over 13,708 clips — not the tp-safe variant. At tp=1 there are no collectives,
so the in-forward host copies that variant was written to avoid are harmless
here, and using the identical module means the two models cannot drift.

Cost
----
A100-80GB is $0.000694/sec ≈ $2.50/hr. At a pessimistic 0.5 s/clip the full
13,708-clip extraction is ~1.9 h ≈ $4.75 against a $30/month allowance.
Volumes are $0.09/GiB/mo with 1 TiB free, so the ~65 GB footprint costs $0.

Do NOT set a region: a broad region selection multiplies price by 1.5x and a
narrow one by 1.75x, for no benefit here.

Usage
-----
    export PYTHONPATH=/dcs/large/u5734759/modal_env
    modal setup                                   # once, interactive
    modal run src/modal/voxtral_small_modal.py::download_model
    modal volume put meld-audio <local dir> /dev_splits_complete
    modal run src/modal/voxtral_small_modal.py::smoke
    modal run --detach src/modal/voxtral_small_modal.py::extract --split dev

IMPORTANT: use ``--detach`` for anything long. Without it the remote app is
killed the moment the local client disconnects -- "Stopping app - local client
disconnected" -- so a 2.7 h train extraction would die with the ssh session, a
network blip, or a Ctrl-C. The smoke test is short enough not to care; the full
splits are not.
"""

from __future__ import annotations

import modal

APP_NAME = "voxtral-small-extract"
MODEL_ID = "mistralai/Voxtral-Small-24B-2507"
MODEL_REVISION = "da5b42409f279fdd92febee0511a6c32828569c1"

# One GPU, 80 GB. This is the whole point: Small fits, so tp=1, so no NCCL.
GPU = "A100-80GB"

app = modal.App(APP_NAME)

# Persisted across runs so the 48.5 GB download happens once. Modal pulls it
# from HuggingFace at datacenter bandwidth, not through the Warwick filer.
model_volume = modal.Volume.from_name("voxtral-small-weights", create_if_missing=True)
data_volume = modal.Volume.from_name("meld-audio", create_if_missing=True)
out_volume = modal.Volume.from_name("meld-extracted-small", create_if_missing=True)

MODEL_DIR = "/models"
DATA_DIR = "/data"
OUT_DIR = "/out"

image = (
    modal.Image.debian_slim(python_version="3.12")
    # ffmpeg decodes MELD's .mp4 to 16 kHz mono; the pipeline shells out to it
    # rather than using torchcodec, which needs libnvrtc.
    .apt_install("ffmpeg")
    .pip_install(
        "vllm==0.19.0",
        "torch==2.10.0",
        "mistral-common[audio]",
        "huggingface_hub",
        # Required because HF_HUB_ENABLE_HF_TRANSFER=1 is set below;
        # enabling it without the package is a hard error, not a fallback.
        "hf_transfer",
        "soundfile",
        "numpy",
        "pyyaml",
    )
    .env({
        "HF_HOME": MODEL_DIR,
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # apply_model() ships the hook callable to the worker, which vLLM
        # refuses without this.
        "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
    })
    # Ship the real source so the hook here IS the hook used on Mini.
    .add_local_dir("src", "/root/src")
)


def _real_clips(root):
    """Every genuine .mp4 under `root`, excluding macOS resource forks.

    `modal volume put` uploads AppleDouble sidecar files (`._dia0_utt0.mp4`)
    alongside the real clips. They end in .mp4 so rglob matches them, but they
    are metadata, not media: ffmpeg exits 1 and kills the whole extraction.
    That is what failed the test split AFTER the 48.5 GB model had loaded.
    """
    return [p for p in root.rglob("*.mp4") if not p.name.startswith("._")]


@app.function(
    image=image,
    volumes={MODEL_DIR: model_volume},
    timeout=60 * 60,
    # Voxtral is a GATED repo: the download 401s without an accepted-licence
    # token. Supplied as a Modal secret so it never appears in the source.
    secrets=[modal.Secret.from_name("huggingface-token")],
    # No GPU: this is a pure download.
)
def download_model() -> str:
    """Pull Voxtral-Small into the persistent model volume. Run once."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        MODEL_ID,
        revision=MODEL_REVISION,
        # consolidated.safetensors is the Mistral-format weight file. The HF
        # shards are NOT interchangeable: vLLM selects its loader by format,
        # and the HF path has no weight-name mapping for Voxtral.
        allow_patterns=["*.json", "consolidated.safetensors", "tekken.json"],
    )
    model_volume.commit()
    import os

    total = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, fs in os.walk(path)
        for f in fs
    )
    print(f"model at {path} ({total / 1e9:.2f} GB)")
    return path


def _build_llm():
    """Construct the vLLM engine. tp=1 — the entire reason for using Modal."""
    from vllm import LLM

    return LLM(
        model=MODEL_ID,
        revision=MODEL_REVISION,
        tokenizer_mode="mistral",
        # Forced, not auto-detected. Small's snapshot contains BOTH a
        # Mistral params.json and an HF config.json, so "auto" can pick the HF
        # parser — which never reads downsample_factor from params.json's
        # multimodal block and fails with AttributeError. Mini has no
        # config.json, which is why auto worked there and not here.
        config_format="mistral",
        load_format="mistral",
        max_model_len=8192,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        tensor_parallel_size=1,
        enforce_eager=True,
    )


def _transcribe_and_capture(llm, clips, max_tokens: int = 200,
                            on_shard=None, shard_every: int = 500):
    """One pass: transcribe, capturing encoder output via the forward hook.

    Mirrors ``transcribe_and_extract_split`` in transcribe_all.py, including
    draining the hook PER BATCH so the worker-side buffer stays bounded.

    Incremental flushing (``on_shard``)
    -----------------------------------
    Originally this accumulated every sequence in RAM and returned them all at
    once, so ``extract`` wrote nothing until the final clip. That was wrong for
    two independent reasons, and a train run lost ~2 hours of A100 time to it:

    1. **No durability.** A container restart, a preemption or the 6-hour
       function timeout discarded the entire run, and Modal restarted from
       clip 0. Nothing was recoverable because nothing had been committed.
    2. **Unbounded memory.** 9,989 sequences of roughly (180, 1280) float32 is
       about 9 GB, held twice over (``captured`` keyed by fingerprint and
       ``sequences`` keyed by utterance) -- a credible OOM on its own, and the
       most likely trigger for the restart in the first place.

    Passing ``on_shard`` fixes both: sequences are resolved to utterance keys
    at each batch boundary and handed off every ``shard_every`` clips, after
    which the in-memory buffers are cleared. The caller is responsible for
    persisting each shard and for committing the volume.
    """
    import base64
    import subprocess

    from vllm import SamplingParams

    # The repo has no __init__.py files (PEP 420 namespace packages), so the
    # parent of src/ must be importable. On-prem scripts do the same insert;
    # relying on Modal's default working directory would be fragile.
    import sys
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    from src.preprocessing.acoustic_hook import (
        drain_acoustic_hook,
        drain_hook_errors,
        install_acoustic_hook,
        merge_worker_captures,
        waveform_fingerprint,
    )

    print("hook:", llm.apply_model(install_acoustic_hook))

    def wav_bytes(path):
        """Decode to 16 kHz mono, or return None if the clip is undecodable.

        MELD ships at least one corrupt file: train_splits/dia125_utt3.mp4 is
        1.9 MB but has no moov atom, so ffmpeg exits 1. With check=True that
        raised and killed the whole extraction at clip 672 of 9,989, discarding
        every sequence captured up to that point.
        
        One bad clip out of ~14,000 must not cost the run. Returning None skips
        it; the caller records the key so a missing utterance is visible in the
        output rather than silently absent.
        """
        r = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
             "-ac", "1", "-ar", "16000", "-f", "wav", "-"],
            capture_output=True, check=False)
        if r.returncode != 0 or not r.stdout:
            return None
        return r.stdout

    import numpy as np
    import torch

    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    transcripts, captured, fingerprints = {}, {}, {}
    sequences: dict = {}
    unmatched: list = []
    undecodable: list = []

    BATCH = 8
    for start in range(0, len(clips), BATCH):
        batch = clips[start:start + BATCH]
        msgs, keys = [], []
        for key, path in batch:
            raw = wav_bytes(path)
            if raw is None:
                undecodable.append(key)
                print(f"  SKIP {key}: ffmpeg could not decode it", flush=True)
                continue
            # Fingerprint the SAME decoded audio we hand to vLLM, so the
            # driver-side and worker-side hashes agree.
            import io

            import soundfile as sf
            wave, _ = sf.read(io.BytesIO(raw), dtype="float32")
            fingerprints[key] = waveform_fingerprint(wave)
            # "audio_url", NOT "audio" -- mistral-common raises
            # NotImplementedError: Unknown part type: audio for the latter.
            # This mirrors transcribe_all.py exactly.
            msgs.append([{"role": "user", "content": [
                {"type": "audio_url", "audio_url":
                    {"url": f"data:audio/wav;base64,{base64.b64encode(raw).decode()}"}},
                {"type": "text", "text": "Transcribe this audio."}]}])
            keys.append(key)

        outs = llm.chat(msgs, sampling_params=sampling)
        for k, o in zip(keys, outs):
            transcripts[k] = o.outputs[0].text.strip()

        captured.update(merge_worker_captures(llm.apply_model(drain_acoustic_hook)))
        for errs in llm.apply_model(drain_hook_errors):
            for m in errs[:5]:
                print("acoustic hook:", m)

        # Resolve THIS batch's fingerprints to utterance keys now, rather than
        # once at the end, so the buffers can be flushed and freed.
        for key in keys:
            fp = fingerprints.get(key)
            arr = captured.pop(fp, None) if fp is not None else None
            if arr is None:
                unmatched.append(key)
                continue
            # merge_worker_captures already returns torch.Tensor; the hook
            # stores numpy internally. Accept either rather than assuming one.
            sequences[key] = (torch.from_numpy(arr)
                              if isinstance(arr, np.ndarray) else arr)

        done = min(start + BATCH, len(clips))
        print(f"  {done}/{len(clips)} | {len(sequences)} pending sequences")

        if on_shard is not None and len(sequences) >= shard_every:
            on_shard(transcripts, sequences, unmatched + undecodable)
            transcripts, sequences = {}, {}
            unmatched, undecodable = [], []

    if undecodable:
        print(f"\n  {len(undecodable)} clip(s) were undecodable and skipped: "
              f"{undecodable[:5]}", flush=True)
    if on_shard is not None:
        # Final partial shard. Returning empties keeps the non-sharded contract
        # below unambiguous: with on_shard set, everything went to the caller.
        if transcripts or sequences or unmatched or undecodable:
            on_shard(transcripts, sequences, unmatched + undecodable)
        return {}, {}, []
    return transcripts, sequences, unmatched + undecodable


@app.function(
    image=image, gpu=GPU,
    volumes={MODEL_DIR: model_volume, DATA_DIR: data_volume, OUT_DIR: out_volume},
    secrets=[modal.Secret.from_name("huggingface-token")],
    timeout=60 * 60,
)
def smoke(n: int = 24) -> dict:
    """Prove Small loads AND generates on one GPU before paying for the full run.

    Deliberately the same shape as the on-prem smoke test: a handful of clips,
    throwaway output, and explicit verification rather than "it didn't crash".
    """
    import time
    from pathlib import Path

    clips = sorted(_real_clips(Path(DATA_DIR)))[:n]
    if not clips:
        raise RuntimeError(
            f"No .mp4 under {DATA_DIR}. Upload MELD first:\n"
            f"  modal volume put meld-audio <local dir> /dev_splits_complete")
    records = [(c.stem, c) for c in clips]

    t0 = time.time()
    llm = _build_llm()
    load_s = time.time() - t0
    print(f"LOADED in {load_s:.1f}s")

    t0 = time.time()
    transcripts, sequences, unmatched = _transcribe_and_capture(llm, records)
    gen_s = time.time() - t0

    dims = {tuple(v.shape[1:]) for v in sequences.values()}
    frames = sorted(v.shape[0] for v in sequences.values())
    result = {
        "clips": len(records),
        "transcripts": len(transcripts),
        "sequences": len(sequences),
        "unmatched": unmatched[:5],
        "n_unmatched": len(unmatched),
        "dims": [list(d) for d in dims],
        "median_frames": frames[len(frames) // 2] if frames else None,
        "load_seconds": round(load_s, 1),
        "seconds_per_clip": round(gen_s / max(1, len(records)), 2),
        "projected_hours_13708": round(gen_s / max(1, len(records)) * 13708 / 3600, 2),
    }
    for k in list(transcripts)[:3]:
        print(f"  {k}: {transcripts[k][:70]!r}")
    print(result)

    ok = (len(sequences) == len(records) and dims == {(1280,)}
          and not unmatched)
    print("=== SMALL WORKS ON MODAL ===" if ok else "=== CHECK ABOVE ===")
    result["ok"] = ok
    return result


@app.function(
    image=image, gpu=GPU,
    volumes={MODEL_DIR: model_volume, DATA_DIR: data_volume, OUT_DIR: out_volume},
    secrets=[modal.Secret.from_name("huggingface-token")],
    timeout=6 * 60 * 60,
)
def extract(split: str = "dev", shard_every: int = 500) -> dict:
    """Full single-pass extraction for one split, written to the output volume.

    Resumable. Work is committed to ``{split}_shards/`` every ``shard_every``
    clips; on restart any clip already present in a shard is skipped. A train
    run was lost at ~7,000 of 9,989 clips because this function committed
    nothing until the final clip, so a container restart discarded everything
    and began again from zero. A restart now costs at most one shard.
    """
    import json
    from pathlib import Path

    import torch

    subdir = {"train": "train_splits",
              "dev": "dev_splits_complete",
              "test": "output_repeated_splits_test"}[split]
    clips = sorted(_real_clips(Path(DATA_DIR) / subdir))
    if not clips:
        raise RuntimeError(f"No .mp4 under {Path(DATA_DIR) / subdir}")
    records = [(c.stem, c) for c in clips]
    print(f"{split}: {len(records)} clips")

    shard_dir = Path(OUT_DIR) / f"{split}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    # --- Resume -----------------------------------------------------------
    # A key counts as done if it appears in ANY completed shard, whether it
    # produced a sequence, went unmatched, or was undecodable. Retrying a clip
    # that legitimately has no sequence would loop forever across restarts.
    done_keys: set = set()
    existing = sorted(shard_dir.glob("shard_*.pt"))
    for p in existing:
        try:
            done_keys |= set(torch.load(p, map_location="cpu",
                                        weights_only=False)["keys"])
        except Exception as exc:                                  # noqa: BLE001
            # A shard half-written when the container died is not fatal: drop
            # it and redo those clips rather than aborting the whole run.
            print(f"  discarding unreadable shard {p.name}: {exc}", flush=True)
            p.unlink(missing_ok=True)
    if done_keys:
        records = [(k, p) for k, p in records if k not in done_keys]
        print(f"  resuming: {len(done_keys)} clip(s) already done, "
              f"{len(records)} remaining", flush=True)
    if not records:
        print("  nothing left to extract; merging existing shards")

    shard_no = len(list(shard_dir.glob("shard_*.pt")))

    def write_shard(t_delta, s_delta, u_delta):
        nonlocal shard_no
        torch.save({"transcripts": t_delta, "sequences": s_delta,
                    "unmatched": u_delta,
                    "keys": list(t_delta) + list(s_delta) + list(u_delta)},
                   shard_dir / f"shard_{shard_no:05d}.pt")
        out_volume.commit()
        shard_no += 1
        print(f"  committed shard {shard_no} "
              f"({len(s_delta)} sequences)", flush=True)

    if records:
        llm = _build_llm()
        _transcribe_and_capture(llm, records, on_shard=write_shard,
                                shard_every=shard_every)

    # --- Merge ------------------------------------------------------------
    transcripts, sequences, unmatched = {}, {}, []
    for p in sorted(shard_dir.glob("shard_*.pt")):
        sh = torch.load(p, map_location="cpu", weights_only=False)
        transcripts.update(sh["transcripts"])
        sequences.update(sh["sequences"])
        unmatched.extend(sh["unmatched"])

    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{split}_transcripts.json", "w", encoding="utf-8") as f:
        json.dump(transcripts, f, ensure_ascii=False, indent=2)
    torch.save(sequences, out / f"{split}_acoustic_seq.pt")
    # Masked-mean control (ADR-003), computed over REAL frames only.
    pooled = {k: v.float().mean(dim=0) for k, v in sequences.items()}
    torch.save(pooled, out / f"{split}_embeddings_maskedmean.pt")
    if unmatched:
        with open(out / f"{split}_unmatched.json", "w", encoding="utf-8") as f:
            json.dump(unmatched, f, indent=2)
    out_volume.commit()

    # `records` was filtered by the resume step, so it counts only what THIS
    # invocation did. Completeness must be judged against the clips on disk.
    total = len(clips)
    accounted = len(sequences) + len(unmatched)
    complete = accounted >= total
    print(f"\n  {split}: {len(sequences)} sequences + {len(unmatched)} "
          f"unmatched = {accounted} of {total} clips "
          f"{'-- COMPLETE' if complete else '-- INCOMPLETE, rerun to resume'}",
          flush=True)

    return {"split": split, "clips_total": total,
            "clips_this_run": len(records),
            "transcripts": len(transcripts), "sequences": len(sequences),
            "n_unmatched": len(unmatched), "complete": complete}


@app.function(
    image=image, gpu=GPU,
    volumes={MODEL_DIR: model_volume, DATA_DIR: data_volume, OUT_DIR: out_volume},
    secrets=[modal.Secret.from_name("huggingface-token")],
    timeout=6 * 60 * 60,
)
def zeroshot(split: str = "test", max_samples: int = 0) -> dict:
    """Zero-shot emotion/sentiment classification — the untrained baseline.

    Voxtral classifies directly from audio rather than transcribing, so this
    measures what a frozen model with a prompt achieves before any of the
    pipeline's training exists. Mini's numbers on test are emotion WF1 0.5043
    and sentiment WF1 0.5515; Small is the missing half of that comparison.

    Reuses ``src/evaluation/voxtral_zeroshot.py`` -- the same prompt and the
    same parser as the Mini run. Re-implementing either would make the two
    models incomparable for reasons that have nothing to do with the models.
    """
    import base64
    import json
    import subprocess
    import sys
    from pathlib import Path

    if "/root" not in sys.path:
        sys.path.insert(0, "/root")

    from vllm import SamplingParams

    from src.evaluation.voxtral_zeroshot import (
        build_classification_prompt,
        build_messages,
        parse_prediction,
    )

    subdir = {"train": "train_splits",
              "dev": "dev_splits_complete",
              "test": "output_repeated_splits_test"}[split]
    clips = sorted(_real_clips(Path(DATA_DIR) / subdir))
    if max_samples:
        clips = clips[:max_samples]
    if not clips:
        raise RuntimeError(f"No .mp4 under {Path(DATA_DIR) / subdir}")
    print(f"{split}: {len(clips)} clips")

    llm = _build_llm()
    instruction = build_classification_prompt()
    # max_tokens=30 matches the Mini run: a classification that runs long has
    # gone wrong anyway, and the response is machine-parsed.
    sampling = SamplingParams(max_tokens=30, temperature=0.0)

    preds = {}
    BATCH = 16
    for start in range(0, len(clips), BATCH):
        batch = clips[start:start + BATCH]
        msgs = []
        for c in batch:
            raw = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-i", str(c),
                 "-ac", "1", "-ar", "16000", "-f", "wav", "-"],
                capture_output=True, check=True).stdout
            msgs.append(build_messages(base64.b64encode(raw).decode(), instruction))
        outs = llm.chat(msgs, sampling_params=sampling)
        for c, o in zip(batch, outs):
            text = o.outputs[0].text.strip()
            emo, sen = parse_prediction(text)
            preds[c.stem] = {"emotion_idx": emo, "sentiment_idx": sen, "raw": text}
        print(f"  {min(start + BATCH, len(clips))}/{len(clips)}")

    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"voxtral_small_zeroshot_{split}_predictions.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2)
    out_volume.commit()
    # Scored locally against the MELD CSV, so the metric code is identical to
    # Mini's rather than a second implementation living on Modal.
    return {"split": split, "n": len(preds), "path": str(path)}


@app.local_entrypoint()
def main(split: str = "", n: int = 24):
    """Default entrypoint: smoke test, or a full split with --split."""
    if split:
        print(extract.remote(split=split))
    else:
        print(smoke.remote(n=n))
