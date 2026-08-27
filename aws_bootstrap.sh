#!/bin/bash
# Bootstrap the Voxtral/XLM-R dissertation pipeline on a fresh AWS g5.xlarge
# (Deep Learning Base OSS Nvidia Driver GPU AMI on Ubuntu 22.04).
#
# Usage (on the AWS instance after ssh):
#   cd ~/project && bash aws_bootstrap.sh
#
# Idempotent — safe to re-run.

set -euo pipefail

echo "[1/5] Apt packages..."
sudo apt-get update -qq
sudo apt-get install -yqq ffmpeg tmux rsync git

echo "[2/5] Python venv..."
if [ ! -d ~/venv ]; then
    python3 -m venv ~/venv
fi
source ~/venv/bin/activate
pip install --upgrade pip -q

echo "[3/5] Python packages..."
pip install -q \
    torch transformers accelerate \
    scikit-learn xgboost jiwer \
    pandas matplotlib pyyaml \
    silero-vad hf_transfer

echo "[4/5] HuggingFace cache + env..."
mkdir -p ~/hf_cache ~/data ~/checkpoints
if ! grep -q "HF_HOME" ~/.bashrc; then
    cat >> ~/.bashrc <<'EOF'
export HF_HOME=~/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export PATH=~/venv/bin:$PATH
EOF
fi
source ~/.bashrc

echo "[5/5] Verify GPU + torch..."
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo ""
echo "Bootstrap done."
echo "Next: bash aws_run_all_ablations.sh (once data is rsync'd into ~/data/)"
