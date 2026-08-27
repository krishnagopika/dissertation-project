#!/bin/bash
#SBATCH --job-name=small_smoke
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --account=wmlg
#SBATCH --time=00:40:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# SMOKE TEST: 20 dev utterances through Voxtral-Small-24B at tp=2.
#
# Jobs 1401458 and 1406769 both hung at vllm engine init (rank 1 never returned
# from NCCL setup) and burned the full 2-day walltime without transcribing a
# single utterance. 2x L40S have no NVLink, so NCCL falls back to PCIe P2P,
# which on Ada commonly deadlocks — especially with vllm's custom all-reduce.
#
# This job answers "does tp=2 initialise at all" in ~10 minutes instead of 48h.

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

export HF_HOME=/dcs/large/u5734759/hf_cache
export TRANSFORMERS_CACHE=/dcs/large/u5734759/hf_cache
export VENV=/dcs/large/u5734759/venv

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# --- the actual fix under test ---
export NCCL_P2P_DISABLE=1              # route peer transfers via host, not PCIe P2P
export NCCL_CUMEM_ENABLE=0             # cuMem allocator has known Ada hangs
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1  # fall back to plain NCCL all-reduce
export NCCL_DEBUG=INFO                 # if it still hangs, name the collective
export NCCL_DEBUG_SUBSYS=INIT,GRAPH
export OMP_NUM_THREADS=8

set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a

mkdir -p logs

# Hard 25-min cap so a hang exits with logs intact rather than being SIGKILLed
# at the walltime wall. Exit 124 == still hanging; NCCL_DEBUG output will say where.
timeout 1500 $VENV/bin/python3.12 src/preprocessing/transcribe_all.py \
    --config src/configs/small_smoke.yaml \
    --skip_embeddings \
    --max_samples 20 \
    --splits dev

rc=$?
if [ $rc -eq 124 ]; then
    echo "=== SMOKE FAILED: still hanging after 25 min. Check NCCL_DEBUG output above. ==="
elif [ $rc -ne 0 ]; then
    echo "=== SMOKE FAILED: exit code $rc ==="
else
    echo "=== SMOKE PASSED: tp=2 initialised and transcribed 20 utterances ==="
fi
exit $rc
