#!/usr/bin/env python3.12
"""Validate single-pass acoustic extraction before rewiring the pipeline.

Answers three questions that cannot be settled by reading source:

  Q1  Does a forward hook installed via llm.apply_model actually fire during
      llm.chat, and does it see one waveform + one encoder output per clip?
  Q2  Does the waveform reaching the hook hash to the same fingerprint we can
      compute on the driver side? (If not, embeddings cannot be keyed back to
      utterances and the whole approach is dead.)
  Q3  Does mean-pooling vLLM's encoder output reproduce the vectors HF's
      VoxtralWrapper produced? If cosine ~= 1.0 the existing checkpoints stay
      valid; if not, everything downstream must be re-extracted and retrained.

Run under Slurm with a GPU. Prints a verdict block at the end.
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.acoustic_hook import (
    drain_acoustic_hook,
    install_acoustic_hook,
    merge_worker_captures,
    waveform_fingerprint,
)
from src.preprocessing.transcribe_all import (
    audio_to_wav_base64,
    build_utterance_index,
    load_audio_mono_16k,
)

N_CLIPS = 8
MELD_ROOT = Path("/dcs/large/u5734759/data/meld_raw")
MODEL_ID = "mistralai/Voxtral-Mini-3B-2507"

INSTRUCTION = (
    "Output only the verbatim spoken words from this audio. "
    "Plain text only. No timestamps, no speaker labels, no formatting."
)


def decode_wav_b64(b64: str) -> np.ndarray:
    """Decode our own WAV bytes the way an audio loader would, to float32."""
    import soundfile as sf
    data, sr = sf.read(io.BytesIO(base64.b64decode(b64)), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), sr


def main() -> None:
    records = [r for r in build_utterance_index(MELD_ROOT, "dev")
               if r[1].exists()][:N_CLIPS]
    print(f"Using {len(records)} dev clips")

    b64 = {}
    driver_fp = {}
    driver_wav = {}
    for key, path in records:
        enc = audio_to_wav_base64(path)
        b64[key] = enc
        arr, sr = decode_wav_b64(enc)
        driver_wav[key] = arr
        driver_fp[key] = waveform_fingerprint(arr)
        print(f"  {key:20s} samples={len(arr):7d} sr={sr} "
              f"mean={arr.mean():+.6f} std={arr.std():.6f} fp={driver_fp[key][:12]}")

    # ---------------------------------------------------------------- vLLM --
    from vllm import LLM, SamplingParams
    llm = LLM(model=MODEL_ID, tokenizer_mode="mistral", max_model_len=8192,
              dtype="bfloat16", gpu_memory_utilization=0.85,
              tensor_parallel_size=1, enforce_eager=True)

    print("\n--- installing hook ---")
    print(llm.apply_model(install_acoustic_hook))

    # A diagnostic hook alongside, recording what the encoder actually receives.
    def _diag(model):
        import torch as t
        diag = []
        enc = model.whisper_encoder

        def h(mod, args, output):
            ins = args[0] if isinstance(args[0], (list, tuple)) else [args[0]]
            outs = output if isinstance(output, (list, tuple)) else [output]
            for w, o in zip(ins, outs):
                diag.append({
                    "n": int(w.numel()), "dtype": str(w.dtype),
                    "dev": str(w.device),
                    "mean": float(w.detach().to(t.float32).mean()),
                    "std": float(w.detach().to(t.float32).std()),
                    "out_shape": tuple(o.shape), "out_dtype": str(o.dtype),
                })
        enc.register_forward_hook(h)
        model._diag = diag
        return "diag installed"
    print(llm.apply_model(_diag))

    messages = [[{"role": "user", "content": [
        {"type": "audio_url",
         "audio_url": {"url": f"data:audio/wav;base64,{b64[k]}"}},
        {"type": "text", "text": INSTRUCTION},
    ]}] for k, _ in records]

    outs = llm.chat(messages, sampling_params=SamplingParams(max_tokens=128,
                                                             temperature=0.0))
    for (k, _), o in zip(records, outs):
        print(f"  ASR {k:20s} {o.outputs[0].text.strip()[:70]!r}")

    diag = llm.apply_model(lambda m: getattr(m, "_diag", []))[0]
    print(f"\n--- hook fired for {len(diag)} clip(s) ---")
    for d in diag:
        print("   ", d)

    captured = merge_worker_captures(llm.apply_model(drain_acoustic_hook))
    print(f"\n--- captured {len(captured)} embedding(s) ---")

    matched = {k: captured[fp] for k, fp in driver_fp.items() if fp in captured}
    print(f"Q2 fingerprint match: {len(matched)} / {len(records)}")
    if matched:
        any_v = next(iter(matched.values()))
        print(f"    embedding shape {tuple(any_v.shape)} dtype {any_v.dtype}")

    del llm
    torch.cuda.empty_cache()
    import gc; gc.collect()

    # ------------------------------------------------------------------ HF --
    print("\n--- HF reference pass ---")
    from src.models.voxtral import VoxtralWrapper
    w = VoxtralWrapper(MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16)
    hf = {}
    for key, path in records:
        wav = load_audio_mono_16k(path, 40.0)
        hf[key] = w.extract_acoustic_embeddings(wav, sample_rate=16000).squeeze(0)

    print("\n" + "=" * 68)
    print("VERDICT")
    print("=" * 68)
    print(f"Q1 hook fired            : {'YES' if diag else 'NO'} "
          f"({len(diag)} clips seen, {len(records)} submitted)")
    print(f"Q2 keyable by fingerprint: {len(matched)}/{len(records)}")
    if matched:
        print("Q3 vLLM-hook vs HF-pass cosine / relative-norm:")
        for key in matched:
            a = matched[key].float()
            b = hf[key].float()
            if a.shape != b.shape:
                print(f"    {key:20s} SHAPE MISMATCH {tuple(a.shape)} vs {tuple(b.shape)}")
                continue
            cos = torch.nn.functional.cosine_similarity(a[None], b[None]).item()
            print(f"    {key:20s} cos={cos:+.6f} "
                  f"|a|={a.norm():.3f} |b|={b.norm():.3f}")
    else:
        print("Q3 skipped -- nothing matched")


if __name__ == "__main__":
    main()
