#!/bin/bash
#SBATCH --job-name=async_off
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=96G --gres=gpu:2
#SBATCH --account=wmlg --time=00:40:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Disable vllm's ASYNC SCHEDULING at tp=2.
#
# Ruled out so far: the acoustic hook (4 variants incl. a no-op all hung),
# audio itself (text-only hangs too, 9717), and the mp start method
# (spawn hangs identically, 9920 config A).
#
# Every run blocks in the SAME place:
#     core.py:499  step_with_batch_queue  ->  future.result()
# and every log says:
#     [vllm.py:790] Asynchronous scheduling is enabled.
#
# step_with_batch_queue is the ASYNC path -- it queues a batch and waits on a
# future. vllm itself knows this mode is not universally safe; config/vllm.py
# disables it automatically for executor backends that do not support it
# ("Async scheduling will be disabled because it is not supported with the %s
# distributed executor backend"). Our backend is mp, and vllm is ENABLING it.
#
# So: same model, same tp=2, same NCCL config, async_scheduling=False.

set -u
source /etc/profile.d/modules.sh; source /etc/profile.d/conda.sh
export HF_HOME=/dcs/large/u5734759/hf_cache VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0 VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
cd /dcs/pg25/u5734759/dissertation-project
mkdir -p logs

$VENV/bin/python3.12 - <<'PY'
import time

def main():
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model="mistralai/Voxtral-Small-24B-2507", tokenizer_mode="mistral",
              config_format="mistral", load_format="mistral",
              max_model_len=4096, dtype="bfloat16", gpu_memory_utilization=0.85,
              tensor_parallel_size=2, enforce_eager=True,
              async_scheduling=False)          # <-- the one change
    print(f"LOADED in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    out = llm.chat([[{"role": "user", "content": "Name three colours."}]],
                   SamplingParams(temperature=0.0, max_tokens=16))
    print(f"TEXT OK in {time.time()-t0:.1f}s: {out[0].outputs[0].text.strip()[:60]!r}", flush=True)

    # If text works, go straight at the real thing: audio through the encoder.
    import base64, subprocess
    from pathlib import Path
    clips = sorted(Path("/dcs/large/u5734759/data/meld_raw/dev_splits_complete").glob("*.mp4"))[:4]
    msgs = []
    for c in clips:
        wav = subprocess.run(["ffmpeg","-nostdin","-v","error","-i",str(c),
                              "-ac","1","-ar","16000","-f","wav","-"],
                             capture_output=True, check=True).stdout
        b64 = base64.b64encode(wav).decode()
        msgs.append([{"role":"user","content":[
            {"type":"audio","audio_url":f"data:audio/wav;base64,{b64}"},
            {"type":"text","text":"Transcribe this audio."}]}])
    t0 = time.time()
    out = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=100))
    print(f"AUDIO OK in {time.time()-t0:.1f}s ({time.time()-t0:.1f}s / {len(out)} clips)", flush=True)
    for c, o in zip(clips, out):
        print(f"   {c.stem}: {o.outputs[0].text.strip()[:70]!r}", flush=True)
    print("ASYNC_OFF_WORKS", flush=True)

if __name__ == "__main__":
    main()
PY
