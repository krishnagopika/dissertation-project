#!/bin/bash
#SBATCH --job-name=tp2_transport
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=96G --gres=gpu:2
#SBATCH --account=wmlg --time=01:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Force NCCL off the shared-memory transport.
#
# Diagnosis from NVIDIA/nccl#2079 ("ncclAllReduce deadlocks on 2x RTX 5880 Ada,
# PCIe PHB topology, no NVLink"): communicator init COMPLETES, then the first
# real collective deadlocks in a CUDA kernel spin-wait on P2P-mapped memory
# signalling across the PCIe host bridge. Root cause is PCIe ACS / IOMMU
# redirecting peer traffic through the CPU root complex.
#
# That matches this node exactly: init succeeds ("Connected all rings",
# "Connected all trees"), then execute_model times out with both GPUs at 100%
# util and both workers burning a full core. 2x L40S (Ada), topology NODE,
# no NVLink -- same class of hardware.
#
# The documented workaround is NCCL_P2P_DISABLE=1, which we ALREADY set. With
# P2P off NCCL fell back to "SHM/direct/direct" -- but shared-memory mappings
# are subject to the same IOMMU redirection, so that may deadlock too.
# NCCL_SHM_DISABLE=1 forces the NET/Socket path instead, avoiding both.
#
# H is the combination; if it works we bisect. If H fails, F and G almost
# certainly fail too, so ordering it first saves the allocation.

set -u
source /etc/profile.d/modules.sh; source /etc/profile.d/conda.sh
export HF_HOME=/dcs/large/u5734759/hf_cache VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0 VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
cd /dcs/pg25/u5734759/dissertation-project
mkdir -p logs

PROBE=/tmp/tp2_tr_$$.py
cat > $PROBE <<'PY'
import os, time

def main():
    from vllm import LLM, SamplingParams
    util = float(os.environ.get("PROBE_UTIL", "0.85"))
    t0 = time.time()
    llm = LLM(model="mistralai/Voxtral-Small-24B-2507", tokenizer_mode="mistral",
              config_format="mistral", load_format="mistral",
              max_model_len=4096, dtype="bfloat16",
              gpu_memory_utilization=util,
              tensor_parallel_size=2, enforce_eager=True)
    print(f"LOADED in {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    out = llm.chat([[{"role": "user", "content": "Name three colours."}]],
                   SamplingParams(temperature=0.0, max_tokens=16))
    print(f"GENERATED in {time.time()-t0:.1f}s: {out[0].outputs[0].text.strip()[:50]!r}", flush=True)
    print("TP2_WORKS", flush=True)

if __name__ == "__main__":
    main()
PY

run_cfg () {
  local name="$1"; shift
  local log=logs/tr_${name}_${SLURM_JOB_ID}.log
  echo ""; echo "=== CONFIG $name : $* ==="
  ( export "$@"; timeout --signal=KILL 540 $VENV/bin/python3.12 $PROBE ) > $log 2>&1
  local rc=$?
  grep -E "LOADED in|GENERATED in|TP2_WORKS" $log | sed 's/^/    /'
  if grep -q TP2_WORKS $log; then echo "    >>> $name : *** WORKS ***"
  elif [ $rc -eq 137 ]; then echo "    >>> $name : TIMED OUT (hung)"
  else echo "    >>> $name : failed (exit $rc)"; grep -iE "error|raise" $log | tail -3 | cut -c1-140 | sed 's/^/      /'; fi
  grep -oE "via [A-Za-z/]+" $log | sort -u | head -3 | sed 's/^/      transport: /'
  rm -f /dev/shm/psm_* 2>/dev/null || true
  sleep 10
}

run_cfg "H_all"       NCCL_SHM_DISABLE=1 OMP_NUM_THREADS=1 NCCL_SOCKET_IFNAME=lo PROBE_UTIL=0.80 NCCL_DEBUG=INFO
run_cfg "F_shm_off"   NCCL_SHM_DISABLE=1 NCCL_DEBUG=INFO
run_cfg "G_omp1"      OMP_NUM_THREADS=1

rm -f $PROBE
echo ""; echo "=== SUMMARY ==="
grep -l TP2_WORKS logs/tr_*_${SLURM_JOB_ID}.log 2>/dev/null | sed 's/^/  WORKS: /' || echo "  none worked"
