#!/bin/bash
#SBATCH --job-name=pp2_fix
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=96G --gres=gpu:2
#SBATCH --account=wmlg --time=00:50:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Pipeline parallelism, with the config problem fixed.
#
# Job 9927 died before any communication with:
#   ValidationError: No model architectures are specified
# Under PP each worker re-validates VllmConfig in its own process, and
# config_format="mistral" reads params.json, which has no "architectures"
# field -- that lives in HF's config.json. So PP itself is still UNTESTED.
#
# Why PP is worth testing: every tp=2 failure is an all-reduce deadlock
# (NVIDIA/nccl#2079 -- init completes, first collective spin-waits, 2x Ada
# PCIe no NVLink, ACS/IOMMU). PP splits BY LAYER and uses point-to-point
# send/recv, never all-reduce. VoxtralForConditionalGeneration declares
# SupportsPP.
#
# A: architectures injected via hf_overrides, keeping the Mistral config.
# B: config from HF config.json (which has architectures), weights still
#    Mistral format. Note ADR-006: the HF path previously failed with
#    AttributeError: downsample_factor -- but that was the WEIGHT loader, and
#    load_format stays "mistral" here. If B reproduces that error we learn
#    the config and load formats are not independently selectable.

set -u
source /etc/profile.d/modules.sh; source /etc/profile.d/conda.sh
export HF_HOME=/dcs/large/u5734759/hf_cache VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0 VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
cd /dcs/pg25/u5734759/dissertation-project
mkdir -p logs

PROBE=/tmp/pp2_$$.py
cat > $PROBE <<'PY'
import os, time

def main():
    from vllm import LLM, SamplingParams
    mode = os.environ["PP_MODE"]
    kw = dict(model="mistralai/Voxtral-Small-24B-2507", tokenizer_mode="mistral",
              load_format="mistral", max_model_len=4096, dtype="bfloat16",
              gpu_memory_utilization=0.85, tensor_parallel_size=1,
              pipeline_parallel_size=2, enforce_eager=True)
    if mode == "A":
        kw["config_format"] = "mistral"
        kw["hf_overrides"] = {"architectures": ["VoxtralForConditionalGeneration"]}
    else:
        kw["config_format"] = "auto"
    t0 = time.time()
    llm = LLM(**kw)
    print(f"LOADED in {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    out = llm.chat([[{"role": "user", "content": "Name three colours."}]],
                   SamplingParams(temperature=0.0, max_tokens=16))
    print(f"TEXT OK in {time.time()-t0:.1f}s: {out[0].outputs[0].text.strip()[:60]!r}", flush=True)
    print("PP2_WORKS", flush=True)

if __name__ == "__main__":
    main()
PY

for m in A B; do
  log=logs/pp2_${m}_${SLURM_JOB_ID}.log
  echo ""; echo "=== PP MODE $m ==="
  ( export PP_MODE=$m; timeout --signal=KILL 600 $VENV/bin/python3.12 $PROBE ) > $log 2>&1
  rc=$?
  grep -E "LOADED in|TEXT OK|PP2_WORKS" $log | sed 's/^/    /'
  if grep -q PP2_WORKS $log; then echo "    >>> $m : *** PIPELINE PARALLEL WORKS ***"
  elif [ $rc -eq 137 ]; then echo "    >>> $m : TIMED OUT (hung)"
  else
    echo "    >>> $m : failed (exit $rc)"
    grep -oiE "Value error[^\"]{0,110}|AttributeError[^\"]{0,90}|TypeError[^\"]{0,90}|not supported[^\"]{0,80}" $log | sort -u | head -4 | sed 's/^/      /'
  fi
  rm -f /dev/shm/psm_* 2>/dev/null || true
  sleep 8
done
rm -f $PROBE
