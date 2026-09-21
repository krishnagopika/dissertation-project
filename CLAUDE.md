# Project Coding Constitution
# Multimodal Fusion for Sentiment and Emotion Extraction

This file defines the coding standards, conventions, and rules for every file
in this project. All code must follow these guidelines without exception.
When in doubt, refer back here before writing anything.

---

## 1. Environment

- **Python version**: 3.12 (enforced by Warwick HPC — do not use 3.11 or below)
- **Framework**: PyTorch + HuggingFace Transformers
- **Hardware target**: Warwick WMLG wmlg-ada partition (4x Nvidia L40S 48GB)
- **User GPU limit**: 2 GPUs maximum per job (96GB VRAM total)
- **Job time limit**: 48 hours maximum — all training loops must checkpoint

---

## 2. Device Handling

Every script must support CPU, CUDA, and MPS (Apple Silicon) transparently.
Always use the `get_device()` utility — never hardcode `"cuda"` or `"cpu"`.

```python
def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

When using multiple GPUs on the HPC, use `torch.nn.DataParallel` or
`torch.nn.parallel.DistributedDataParallel`. Always check GPU count:

```python
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

---

## 3. Seed Setup

Every script that involves randomness must call `set_seed()` at the very start,
before any model or data loading. The seed is read from the config file.

```python
import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 4. Config Loading

Never hardcode paths, hyperparameters, or model IDs anywhere.
Everything is read from `configs/mini.yaml` or `configs/small.yaml`.
The config file path is always passed as a command-line argument.

```python
import yaml
import argparse
from pathlib import Path

def load_config(config_path: str) -> dict:
    """Load YAML config and return as dictionary."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Standard argument parsing pattern for every script
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True,
                    help="Path to config yaml (mini.yaml or small.yaml)")
args = parser.parse_args()
config = load_config(args.config)
```

---

## 5. Model Class Structure

Every model class follows this exact structure:

```python
import torch
import torch.nn as nn
from typing import Optional, Tuple

class MyModel(nn.Module):
    """
    One-line description of what this model does.

    Args:
        input_dim: Dimension of input features.
        hidden_dim: Dimension of hidden layer.
        num_classes: Number of output classes.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # Define all layers here — no logic, only architecture
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass — this is where order matters.
        Data flows through layers defined in __init__.

        Args:
            x: Input tensor of shape [batch_size, input_dim].

        Returns:
            Logits tensor of shape [batch_size, num_classes].
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

Rules:
- `__init__` defines architecture only — no data, no logic
- `forward` defines data flow only — layers applied in order
- Type hints on every argument and return value
- Docstring on every class and every method

---

## 6. Tokenizer and AutoModel Loading

Always use `AutoTokenizer` and `AutoModel` from HuggingFace.
Tokenizers are models themselves — always load them from the same
checkpoint as the model to guarantee compatibility.

```python
from transformers import AutoTokenizer, AutoModel

def load_xlmr(model_id: str, device: torch.device):
    """
    Load XLM-RoBERTa tokenizer and model from HuggingFace.

    Args:
        model_id: HuggingFace model identifier string.
        device: Target device for the model.

    Returns:
        Tuple of (tokenizer, model).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model = model.to(device)
    return tokenizer, model
```

Rules:
- Always load tokenizer and model from the same `model_id`
- Always move model to device immediately after loading
- Never instantiate tokenizers manually — always use `AutoTokenizer`

---

## 7. Dataset Class Structure

Every dataset class inherits from `torch.utils.data.Dataset`
and implements exactly three methods:

```python
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict
import torch

class MyDataset(Dataset):
    """
    Dataset description.

    Args:
        data_root: Path to raw data directory.
        split: One of 'train', 'dev', 'test'.
        config: Loaded config dictionary.
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        config: dict,
    ) -> None:
        assert split in ("train", "dev", "test"), \
            f"split must be train/dev/test, got {split}"
        self.data_root = Path(data_root)
        self.split = split
        self.config = config
        self.samples = self._load_samples()

    def _load_samples(self) -> list:
        """Load and return list of (audio_path, label) tuples."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
```

DataLoader rules:
- Always set `num_workers=4` (or match `--cpus-per-task` in sbatch)
- Always set `pin_memory=True` when using GPU
- Never shuffle validation or test splits
- Always set `drop_last=False` for evaluation

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(
    dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=True,          # train only
    num_workers=4,
    pin_memory=True,
)

val_loader = DataLoader(
    dataset,
    batch_size=config["training"]["batch_size"],
    shuffle=False,         # never shuffle val/test
    num_workers=4,
    pin_memory=True,
)
```

---

## 8. Logging

Never use `print()` in any training or evaluation script.
Always use Python's `logging` module. Set up logging at the
top of every script before anything else runs.

```python
import logging
from pathlib import Path

def setup_logging(log_dir: str, script_name: str) -> logging.Logger:
    """
    Set up logging to both file and console.

    Args:
        log_dir: Directory to write log file.
        script_name: Name used for log file.

    Returns:
        Configured logger instance.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    fh = logging.FileHandler(Path(log_dir) / f"{script_name}.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
```

Log at the start of every epoch:
```python
logger.info(f"Epoch {epoch}/{total_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Weighted F1: {wf1:.4f}")
```

---

## 9. Checkpointing (CRITICAL on HPC — 48 hour limit)

Every training loop must save checkpoints and be able to resume from them.
This is non-negotiable given the 48 hour Slurm job limit.

```python
import torch
from pathlib import Path

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metric: float,
    config: dict,
    checkpoint_dir: str,
    is_best: bool = False,
) -> None:
    """
    Save model checkpoint with epoch and metric in filename.

    Args:
        model: Model to save.
        optimizer: Optimizer state to save (for resuming).
        epoch: Current epoch number.
        metric: Validation weighted F1 at this epoch.
        config: Full config dictionary.
        checkpoint_dir: Directory to save checkpoints.
        is_best: If True, also save as best_model.pt.
    """
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metric": metric,
        "config": config,
    }
    filename = f"checkpoint_epoch{epoch:03d}_f1{metric:.4f}.pt"
    path = Path(checkpoint_dir) / filename
    torch.save(state, path)

    if is_best:
        best_path = Path(checkpoint_dir) / "best_model.pt"
        torch.save(state, best_path)

def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[int, float]:
    """
    Load checkpoint and restore model and optimizer state.

    Args:
        checkpoint_path: Path to checkpoint file.
        model: Model to restore weights into.
        optimizer: Optimizer to restore state into.

    Returns:
        Tuple of (start_epoch, best_metric).
    """
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    return state["epoch"] + 1, state["metric"]
```

Training loop must check for existing checkpoint at startup:

```python
start_epoch = 0
best_metric = 0.0

if Path(config["training"]["checkpoint_dir"]).exists():
    checkpoints = sorted(
        Path(config["training"]["checkpoint_dir"]).glob("checkpoint_*.pt")
    )
    if checkpoints:
        latest = checkpoints[-1]
        logger.info(f"Resuming from checkpoint: {latest}")
        start_epoch, best_metric = load_checkpoint(
            latest, model, optimizer
        )
```

---

## 10. Inference Rules

Always use `model.eval()` and `torch.no_grad()` during inference.
Never run inference without these — they prevent gradient accumulation
and enable inference-specific optimisations.

```python
model.eval()
with torch.no_grad():
    outputs = model(inputs)
```

After large inference runs (e.g. transcribing all MELD audio with Voxtral),
clear GPU cache to free memory before the next phase:

```python
del model
torch.cuda.empty_cache()
import gc; gc.collect()
```

---

## 11. Memory Management

Given the 48GB per GPU limit and large model sizes:

- Voxtral Small (24B) must run in `torch.bfloat16` or `torch.float16`
- Always load large models with `torch_dtype=torch.bfloat16`
- Cache acoustic embeddings to disk after extraction — never recompute them
- Cache transcripts to disk after ASR — never retranscribe during training
- Use `del` on tensors immediately when no longer needed in loops

```python
# Loading Voxtral with reduced precision
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    config["model"]["voxtral_id"],
    torch_dtype=torch.bfloat16,
    device_map="auto",   # handles multi-GPU automatically
)
```

---

## 12. Paths and File Handling

Never hardcode paths. Always use `pathlib.Path`, never string concatenation.
Always check files exist before reading them.

```python
from pathlib import Path

def load_embedding(path: str) -> torch.Tensor:
    """
    Load cached acoustic embedding from disk.

    Args:
        path: Path to .pt embedding file.

    Returns:
        Embedding tensor.

    Raises:
        FileNotFoundError: If embedding file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Embedding not found at {p}. "
            f"Run preprocessing/transcribe_all.py first."
        )
    return torch.load(p, map_location="cpu")
```

---

## 13. Evaluation Metrics

Primary metric is always **weighted F1**. Never use accuracy alone
due to MELD class imbalance. Always report per-class F1 alongside
the primary metric.

```python
from sklearn.metrics import f1_score, classification_report

def compute_metrics(
    preds: list,
    labels: list,
    class_names: list,
) -> dict:
    """
    Compute weighted F1 and per-class F1.

    Args:
        preds: List of predicted class indices.
        labels: List of ground truth class indices.
        class_names: List of class name strings.

    Returns:
        Dictionary with weighted_f1 and per_class_f1.
    """
    weighted_f1 = f1_score(labels, preds, average="weighted")
    report = classification_report(
        labels, preds,
        target_names=class_names,
        output_dict=True,
    )
    return {
        "weighted_f1": weighted_f1,
        "per_class_f1": {
            name: report[name]["f1-score"]
            for name in class_names
        },
    }
```

---

## 14. Slurm / HPC Rules

Every long-running script must have a corresponding `.sbatch` file in `scripts/`.
Use this template:

```bash
#!/bin/bash
#SBATCH --job-name=YOUR_JOB_NAME
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --account=wmlg
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

python3.12 src/YOUR_SCRIPT.py --config src/configs/mini.yaml
```

Rules:
- Use `gpu:2` only for Voxtral Small inference and fusion training with Small
- Use `gpu:1` for XLM-RoBERTa fine-tuning and Voxtral Mini
- Always set `--mail-type=END,FAIL,TIME_LIMIT_80` so you know when jobs finish
- Always write stdout and stderr to `logs/` directory
- Submit from `kudu-taught` only

---

## 15. MELD Emotion Label Schema

These are the canonical label indices used throughout the project.
Never use different numbering in different files.

```python
MELD_EMOTIONS = {
    0: "neutral",
    1: "surprise",
    2: "fear",
    3: "sadness",
    4: "joy",
    5: "disgust",
    6: "anger",
}

MELD_SENTIMENTS = {
    0: "negative",
    1: "neutral",
    2: "positive",
}
```

---

## 16. Pipeline Phases — What Is Frozen When

This is critical to understand before writing any training code:

| Phase | Script | Voxtral | XLM-RoBERTa | Fusion |
|-------|--------|---------|-------------|--------|
| Preprocessing | transcribe_all.py | Frozen (inference only) | Not loaded | Not loaded |
| Phase 1 | finetune.py | Not loaded (transcripts cached) | Trainable | Not loaded |
| Phase 2 | train_fusion.py | Not loaded (embeddings cached) | Frozen | Trainable |
| Evaluation | evaluate.py | Not loaded | Frozen | Frozen |

Voxtral is NEVER trained. It is always frozen.
Embeddings and transcripts are ALWAYS cached to disk before training begins.

---

## 17. Code Style

- Type hints on every function argument and return value
- Docstring on every class and every non-trivial function
- No magic numbers anywhere — all values come from config
- Maximum line length: 88 characters (Black formatter standard)
- Imports ordered: stdlib → third party → local, separated by blank lines
- No `import *` anywhere

---

*Last updated: April 2026*
*Student ID: 5734759*


modal token set --token-id 


modal profile activate krishnagopika1701