"""Determine exactly how vllm pads audio before the Whisper encoder."""
from __future__ import annotations
import sys, base64, io
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.preprocessing.transcribe_all import (audio_to_wav_base64, build_utterance_index,
                                              wav_b64_to_array)

recs = [r for r in build_utterance_index(Path("/dcs/large/u5734759/data/meld_raw"), "dev")
        if r[1].exists()][:6]
b64, drv = {}, {}
for k, p in recs:
    e = audio_to_wav_base64(p); b64[k] = e; drv[k] = wav_b64_to_array(e)
    print(f"DRIVER {k:16s} n={len(drv[k]):7d}  last5={drv[k][-5:]}")

from vllm import LLM, SamplingParams
llm = LLM(model="mistralai/Voxtral-Mini-3B-2507", tokenizer_mode="mistral",
          max_model_len=8192, dtype="bfloat16", gpu_memory_utilization=0.85,
          tensor_parallel_size=1, enforce_eager=True)

def install(model):
    import torch as t
    diag = []
    def h(mod, args, output):
        ins = args[0] if isinstance(args[0], (list, tuple)) else [args[0]]
        for w in ins:
            a = w.detach().to(t.float32).cpu().numpy()
            nz = np.nonzero(a)[0]
            last = int(nz[-1]) + 1 if len(nz) else 0
            diag.append({"n": int(a.shape[0]), "dtype": str(w.dtype),
                         "last_nonzero": last,
                         "trail_zeros": int(a.shape[0]) - last,
                         "head5": a[:5].tolist()})
    model.whisper_encoder.register_forward_hook(h)
    model._diag = diag
    return "ok"
print(llm.apply_model(install))

msgs = [[{"role":"user","content":[
    {"type":"audio_url","audio_url":{"url":f"data:audio/wav;base64,{b64[k]}"}},
    {"type":"text","text":"Transcribe."}]}] for k,_ in recs]
llm.chat(msgs, sampling_params=SamplingParams(max_tokens=16, temperature=0.0))

diag = llm.apply_model(lambda m: m._diag)[0]
print(f"\n=== hook saw {len(diag)} clips ===")
for (k,_), d in zip(recs, diag):
    a = drv[k]
    print(f"HOOK   {k:16s} n={d['n']:7d} last_nonzero={d['last_nonzero']:7d} "
          f"trail_zeros={d['trail_zeros']:7d} dtype={d['dtype']}")
    print(f"       driver n={len(a):7d} | trimmed==driver_len: {d['last_nonzero']==len(a)} "
          f"| head matches: {np.allclose(np.array(d['head5']), a[:5], atol=1e-6)}")
