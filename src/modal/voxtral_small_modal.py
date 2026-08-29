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
    modal run src/modal/voxtral_small_modal.py::extract --split dev
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


def _transcribe_and_capture(llm, clips, max_tokens: int = 200):
    """One pass: transcribe, capturing encoder output via the forward hook.

    Mirrors ``transcribe_and_extract_split`` in transcribe_all.py, including
    draining the hook PER BATCH so the worker-side buffer stays bounded.
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
        return subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path),
             "-ac", "1", "-ar", "16000", "-f", "wav", "-"],
            capture_output=True, check=True).stdout

    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    transcripts, captured, fingerprints = {}, {}, {}

    BATCH = 8
    for start in range(0, len(clips), BATCH):
        batch = clips[start:start + BATCH]
        msgs, keys = [], []
        for key, path in batch:
            raw = wav_bytes(path)
            # Fingerprint the SAME decoded audio we hand to vLLM, so the
            # driver-side and worker-side hashes agree.
            import io

            import soundfile as sf
            wave, _ = sf.read(io.BytesIO(raw), dtype="float32")
            fingerprints[key] = waveform_fingerprint(wave)
            msgs.append([{"role": "user", "content": [
                {"type": "audio", "audio_url":
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
        print(f"  {min(start + BATCH, len(clips))}/{len(clips)} | "
              f"{len(captured)} sequences captured")

    # Pair sequences back to utterance keys by content hash.
    import torch
    sequences, unmatched = {}, []
    for key, fp in fingerprints.items():
        arr = captured.get(fp)
        if arr is None:
            unmatched.append(key)
        else:
            sequences[key] = torch.from_numpy(arr)
    return transcripts, sequences, unmatched


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

    clips = sorted(Path(DATA_DIR).rglob("*.mp4"))[:n]
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
def extract(split: str = "dev") -> dict:
    """Full single-pass extraction for one split, written to the output volume."""
    import json
    from pathlib import Path

    import torch

    subdir = {"train": "train_splits",
              "dev": "dev_splits_complete",
              "test": "output_repeated_splits_test"}[split]
    clips = sorted((Path(DATA_DIR) / subdir).rglob("*.mp4"))
    if not clips:
        raise RuntimeError(f"No .mp4 under {Path(DATA_DIR) / subdir}")
    records = [(c.stem, c) for c in clips]
    print(f"{split}: {len(records)} clips")

    llm = _build_llm()
    transcripts, sequences, unmatched = _transcribe_and_capture(llm, records)

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

    return {"split": split, "clips": len(records),
            "transcripts": len(transcripts), "sequences": len(sequences),
            "n_unmatched": len(unmatched)}


@app.local_entrypoint()
def main(split: str = "", n: int = 24):
    """Default entrypoint: smoke test, or a full split with --split."""
    if split:
        print(extract.remote(split=split))
    else:
        print(smoke.remote(n=n))
