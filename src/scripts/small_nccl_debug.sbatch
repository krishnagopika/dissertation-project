#!/bin/bash
#SBATCH --job-name=nccl_dbg
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=96G --gres=gpu:2
#SBATCH --account=wmlg --time=00:30:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Stop guessing at config flags; make NCCL report where it is stuck.
#
# Ruled out: the acoustic hook (4 variants incl. a no-op), audio (text-only
# hangs, 9717), fork-vs-spawn (spawn hangs, 9920), async scheduling (confirmed
# "disabled", still hangs, 9922), /dev/shm capacity (284 GB free), and a
# corrupt download (sha-256 verified, exact byte size, model loads + warms up).
#
# Symptom: both GPUs at 100% util, zero tokens, EngineCore times out after
# 5 min waiting on workers. 100% GPU with no progress is NCCL busy-polling a
# collective that never completes -- the shm timeout is downstream of that.
#
# NCCL_DEBUG=INFO prints transport selection and ring/channel setup;
# NCCL_DEBUG_SUBSYS=INIT,COLL,P2P,GRAPH shows which transport each rank chose
# and which collective is outstanding. With P2P disabled it must fall back to
# SHM or NET -- if that fallback is misconfigured, this is where it shows.
#
# Also sets a 120 s NCCL watchdog so a stuck collective ABORTS with a message
# naming the operation, instead of spinning until the job walltime.

set -u
source /etc/profile.d/modules.sh; source /etc/profile.d/conda.sh
export HF_HOME=/dcs/large/u5734759/hf_cache VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0 VLLM_DISABLE_CUSTOM_ALL_REDUCE=1

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL,P2P,GRAPH,ENV
export TORCH_NCCL_BLOCKING_WAIT=1          # surface the hang as an error
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=120                    # abort a stuck collective
export TORCH_DISTRIBUTED_DEBUG=DETAIL

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
              tensor_parallel_size=2, enforce_eager=True)
    print(f"LOADED in {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    out = llm.chat([[{"role": "user", "content": "Hi"}]],
                   SamplingParams(temperature=0.0, max_tokens=8))
    print(f"TEXT OK in {time.time()-t0:.1f}s: {out[0].outputs[0].text.strip()!r}", flush=True)
    print("NCCL_PATH_WORKS", flush=True)

if __name__ == "__main__":
    main()
PY
