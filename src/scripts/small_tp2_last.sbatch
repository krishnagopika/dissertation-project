#!/bin/bash
#SBATCH --job-name=tp2_last
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1 --cpus-per-task=8 --mem=96G --gres=gpu:2
#SBATCH --account=wmlg --time=01:30:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# The two variables never tested at tp=2.
#
# Ruled out: hook (4 variants incl. no-op), audio (text-only hangs, 9717),
# fork vs spawn (9920), async scheduling (9922, confirmed disabled),
# /dev/shm capacity, corrupt download (sha verified), and NCCL transport --
# 9923's trace shows NCCL connects fine: "Channel 00 : 0[0] -> 1[1] via
# SHM/direct/direct", "Connected all rings", "Connected all trees". Init is
# healthy; execute_model still times out.
#
# D: enforce_eager=False. Set in EVERY run so far, including all failures --
#    the one constant never varied. It skips Inductor/CUDA-graph compilation.
#    Without it vllm takes the graph-capture path instead, which is different
#    code. May be slow to start, hence the long timeout.
#
# E: NCCL_NET_GDR_LEVEL=0. 9923 reported "use ring PXN 0 GDR 1" -- GPUDirect
#    RDMA is on, on a node where P2P already misbehaves (P2P reports OK via
#    nvidia-smi topo yet deadlocks init, 9705/9707).

set -u
source /etc/profile.d/modules.sh; source /etc/profile.d/conda.sh
export HF_HOME=/dcs/large/u5734759/hf_cache VENV=/dcs/large/u5734759/venv
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export NCCL_P2P_DISABLE=1 NCCL_CUMEM_ENABLE=0 VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
cd /dcs/pg25/u5734759/dissertation-project
mkdir -p logs

PROBE=/tmp/tp2_last_$$.py
cat > $PROBE <<'PY'
import os, time

def main():
    from vllm import LLM, SamplingParams
    eager = os.environ.get("PROBE_EAGER", "1") == "1"
    t0 = time.time()
    llm = LLM(model="mistralai/Voxtral-Small-24B-2507", tokenizer_mode="mistral",
              config_format="mistral", load_format="mistral",
              max_model_len=4096, dtype="bfloat16", gpu_memory_utilization=0.85,
              tensor_parallel_size=2, enforce_eager=eager)
    print(f"LOADED in {time.time()-t0:.1f}s (enforce_eager={eager})", flush=True)
    t0 = time.time()
    out = llm.chat([[{"role": "user", "content": "Name three colours."}]],
                   SamplingParams(temperature=0.0, max_tokens=16))
    print(f"GENERATED in {time.time()-t0:.1f}s: {out[0].outputs[0].text.strip()[:50]!r}", flush=True)
    print("TP2_WORKS", flush=True)

if __name__ == "__main__":
    main()
PY

run_cfg () {
  local name="$1" secs="$2"; shift 2
  local log=logs/tp2last_${name}_${SLURM_JOB_ID}.log
  echo ""; echo "=== CONFIG $name  (timeout ${secs}s)  $* ==="
  ( export "$@"; timeout --signal=KILL "$secs" $VENV/bin/python3.12 $PROBE ) > $log 2>&1
  local rc=$?
  grep -E "LOADED in|GENERATED in|TP2_WORKS" $log | sed 's/^/    /'
  if grep -q TP2_WORKS $log; then echo "    >>> $name : *** WORKS ***"
  elif [ $rc -eq 137 ]; then echo "    >>> $name : TIMED OUT (hung)"
  else echo "    >>> $name : failed (exit $rc)"; grep -iE "error|raise" $log | tail -4 | cut -c1-140 | sed 's/^/      /'; fi
  rm -f /dev/shm/psm_* 2>/dev/null || true
  sleep 10
}

# GDR off first: fast, and the cheaper hypothesis.
run_cfg "E_gdr_off"    600 PROBE_EAGER=1 NCCL_NET_GDR_LEVEL=0
# Graph capture may compile for a long time; give it room.
run_cfg "D_no_eager"  2400 PROBE_EAGER=0

rm -f $PROBE
echo ""; echo "=== SUMMARY ==="
grep -l TP2_WORKS logs/tp2last_*_${SLURM_JOB_ID}.log 2>/dev/null | sed 's/^/  WORKS: /' || echo "  neither worked"
