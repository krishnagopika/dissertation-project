#!/bin/bash
#SBATCH --job-name=tp2_matrix
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=96G --gres=gpu:2
#SBATCH --account=wmlg --time=01:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Find a tp=2 configuration that generates on wmlg1.
#
# Established: the hang is NOT the hook and NOT audio. Text-only with no hook
# hangs identically (9717); four hook variants incl. a literal no-op all hung
# the same (9708, 9713, 9715, 9716). Model loads, KV cache allocates, warmup
# forward SUCCEEDS, then the first real scheduler step never returns and vllm
# raises TimeoutError from multiproc_executor.get_response -> mq.dequeue.
# That is EngineCore waiting on its WORKER processes.
#
# Suspect: VLLM_WORKER_MULTIPROC_METHOD defaults to "fork"; forking after CUDA
# init is a classic deadlock. vllm 0.19.0 has no V0 fallback (VLLM_USE_V1
# absent) and ray is not installed, so the mp start method is the lever.
#
# NOTE: the probe MUST guard with if __name__ == "__main__". Under spawn the
# child re-imports the module, so module-level LLM() construction recurses and
# dies with "An attempt has been made to start a new process before the current
# process has finished its bootstrapping phase" -- which is what killed the
# first attempt at this matrix (job 9919) and was mistaken for spawn failing.

set -u
source /etc/profile.d/modules.sh; source /etc/profile.d/conda.sh
export HF_HOME=/dcs/large/u5734759/hf_cache VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0 VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
cd /dcs/pg25/u5734759/dissertation-project
mkdir -p logs

PROBE=/tmp/tp2_probe_$$.py
cat > $PROBE <<'PY'
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
    out = llm.chat([[{"role": "user", "content": "Name three colours."}]],
                   SamplingParams(temperature=0.0, max_tokens=16))
    print(f"GENERATED in {time.time()-t0:.1f}s: {out[0].outputs[0].text.strip()[:50]!r}", flush=True)
    print("CONFIG_WORKS", flush=True)

# Required for spawn: without it the re-imported child re-runs main().
if __name__ == "__main__":
    main()
PY

run_cfg () {
  local name="$1"; shift
  local log=logs/tp2_${name}_${SLURM_JOB_ID}.log
  echo ""
  echo "=================================================================="
  echo "CONFIG: $name   env: $*"
  echo "=================================================================="
  # Full output to a per-config log so nothing is filtered away this time.
  ( export "$@"; timeout --signal=KILL 420 $VENV/bin/python3.12 $PROBE ) > $log 2>&1
  local rc=$?
  grep -E "LOADED in|GENERATED in|CONFIG_WORKS" $log | sed 's/^/    /'
  if [ $rc -eq 137 ]; then
    echo "    >>> $name : TIMED OUT after 420s (hung)"
  elif grep -q CONFIG_WORKS $log; then
    echo "    >>> $name : *** WORKS ***"
  else
    echo "    >>> $name : failed (exit $rc), last error:"
    grep -iE "error|exception|raise|Traceback" $log | tail -6 | cut -c1-150 | sed 's/^/      /'
  fi
  echo "    full log: $log"
  rm -f /dev/shm/psm_* 2>/dev/null || true
  sleep 10
}

run_cfg "A_spawn"          VLLM_WORKER_MULTIPROC_METHOD=spawn
run_cfg "C_spawn_noV1mp"   VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ENABLE_V1_MULTIPROCESSING=0
run_cfg "B_fork_control"   VLLM_WORKER_MULTIPROC_METHOD=fork

rm -f $PROBE
echo ""
echo "=== SUMMARY ==="
grep -l CONFIG_WORKS logs/tp2_*_${SLURM_JOB_ID}.log 2>/dev/null | sed 's/^/  WORKS: /' || echo "  no config worked"
