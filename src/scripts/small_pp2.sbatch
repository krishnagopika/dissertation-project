#!/bin/bash
#SBATCH --job-name=small_pp2
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=96G --gres=gpu:2
#SBATCH --account=wmlg --time=00:50:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# PIPELINE parallelism instead of TENSOR parallelism.
#
# Every failure so far is a tp=2 all-reduce deadlock. Matching bug report:
# NVIDIA/nccl#2079, "ncclAllReduce deadlocks on 2x RTX 5880 Ada (PCIe PHB
# topology, no NVLink)" -- communicator init COMPLETES, then the first real
# collective deadlocks in a CUDA kernel spin-wait on P2P-mapped memory
# signalling. Root cause is PCIe ACS / IOMMU redirecting peer traffic through
# the CPU root complex. Our node matches: 2x L40S (Ada), topology NODE, no
# NVLink; NCCL reports "Connected all rings/trees" then execute_model times out
# with both GPUs at 100% util and both workers burning a full core.
#
# Ruled out: hook (4 variants incl. no-op, 9708/9713/9715/9716), audio
# (text-only hangs, 9717), fork vs spawn (9920), async scheduling (9922),
# GDR off (9925-E), enforce_eager=False (9925-D, fails at startup),
# NCCL_SHM_DISABLE + OMP_NUM_THREADS=1 + socket ifname (9926-H), /dev/shm
# capacity, corrupt download.
#
# tp splits WITHIN a layer -> all-reduce every layer (the broken primitive).
# pp splits BY layer       -> point-to-point send/recv, no all-reduce.
# VoxtralForConditionalGeneration declares SupportsPP, so this is legitimate.
# Memory is equivalent: ~24 GB per GPU either way.

set -u
source /etc/profile.d/modules.sh; source /etc/profile.d/conda.sh
export HF_HOME=/dcs/large/u5734759/hf_cache VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0 VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
export NCCL_DEBUG=INFO          # set at script level so workers inherit it
cd /dcs/pg25/u5734759/dissertation-project
mkdir -p logs

timeout --signal=KILL 1500 $VENV/bin/python3.12 - <<'PY'
import base64, subprocess, time
from pathlib import Path

def main():
    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model="mistralai/Voxtral-Small-24B-2507", tokenizer_mode="mistral",
              config_format="mistral", load_format="mistral",
              max_model_len=4096, dtype="bfloat16", gpu_memory_utilization=0.85,
              tensor_parallel_size=1,          # no all-reduce
              pipeline_parallel_size=2,        # split by layers instead
              enforce_eager=True)
    print(f"LOADED in {time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    out = llm.chat([[{"role": "user", "content": "Name three colours."}]],
                   SamplingParams(temperature=0.0, max_tokens=16))
    print(f"TEXT OK in {time.time()-t0:.1f}s: {out[0].outputs[0].text.strip()[:60]!r}", flush=True)

    clips = sorted(Path("/dcs/large/u5734759/data/meld_raw/dev_splits_complete").glob("*.mp4"))[:8]
    msgs = []
    for c in clips:
        wav = subprocess.run(["ffmpeg","-nostdin","-v","error","-i",str(c),
                              "-ac","1","-ar","16000","-f","wav","-"],
                             capture_output=True, check=True).stdout
        msgs.append([{"role":"user","content":[
            {"type":"audio","audio_url":f"data:audio/wav;base64,{base64.b64encode(wav).decode()}"},
            {"type":"text","text":"Transcribe this audio."}]}])
    t0 = time.time()
    out = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=100))
    dt = time.time() - t0
    print(f"AUDIO OK: {len(out)} clips in {dt:.1f}s ({dt/len(out):.1f}s/clip)", flush=True)
    for c, o in zip(clips, out):
        print(f"   {c.stem}: {o.outputs[0].text.strip()[:70]!r}", flush=True)
    print(f"PROJECTED 13,708 clips: {dt/len(out)*13708/3600:.1f} hours", flush=True)
    print("PP2_WORKS", flush=True)

if __name__ == "__main__":
    main()
PY
echo "exit: $?"
