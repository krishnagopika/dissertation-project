#!/bin/bash
#SBATCH --job-name=small_textonly
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=64G --gres=gpu:2
#SBATCH --account=wmlg --time=00:25:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# FINAL DIAGNOSTIC: is the tp=2 hang about AUDIO, or about tp=2 at all?
#
# The hook is exonerated. Four variants all hung identically at 0/8 with a
# 5-minute vLLM TimeoutError:
#   9708 blocking .cpu()   9713 clone()   9715 detach()   9716 no-op (pass)
# A hook that does nothing cannot be the cause.
#
# So: same model, same tp=2, same NCCL config, NO AUDIO and NO HOOK.
#   generates -> the audio path under TP is at fault (narrow, tractable)
#   hangs     -> tp=2 generation is broken on this node, full stop
set -u
source /etc/profile.d/modules.sh; source /etc/profile.d/conda.sh
export HF_HOME=/dcs/large/u5734759/hf_cache VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0 VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
cd /dcs/pg25/u5734759/dissertation-project

$VENV/bin/python3.12 - <<'PY'
import time
from vllm import LLM, SamplingParams

print("loading Small, tp=2, text-only, no hook ...", flush=True)
t0 = time.time()
llm = LLM(model="mistralai/Voxtral-Small-24B-2507", tokenizer_mode="mistral",
          config_format="mistral", load_format="mistral",
          max_model_len=8192, dtype="bfloat16", gpu_memory_utilization=0.85,
          tensor_parallel_size=2, enforce_eager=True)
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

msgs = [[{"role": "user", "content": "Name three colours."}] for _ in range(4)]
print("generating (text only) ...", flush=True)
t0 = time.time()
out = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=32))
dt = time.time() - t0
print(f"\n=== TEXT GENERATION OK: {len(out)} prompts in {dt:.1f}s ===", flush=True)
for o in out[:2]:
    print(f"  {o.outputs[0].text.strip()[:60]!r}")
print("\n=== tp=2 generation WORKS -> the hang is specific to the AUDIO path ===")
PY
