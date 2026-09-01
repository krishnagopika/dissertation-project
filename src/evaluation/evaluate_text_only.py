"""
evaluate_text_only.py — Phase 1 Text-Only Baseline Evaluation
==============================================================
Evaluates the Phase 1 XLM-RoBERTa checkpoint on the MELD test set
using ASR transcripts only (no acoustic features). Used to measure
how much the fusion adds over the text-only model.

Usage
-----
  python3.12 src/evaluation/evaluate_text_only.py --config src/configs/mini.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.evaluate import compute_all_metrics, log_full_metrics, plot_confusion_matrix
from src.evaluation.metrics import EMOTION_NAMES, SENTIMENT_NAMES
from src.models.xlmr import XLMRobertaClassifier
from src.utils import get_device, load_config, set_seed, setup_logging

EMOTION2IDX: Dict[str, int] = {name: i for i, name in enumerate(EMOTION_NAMES)}
SENTIMENT2IDX: Dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}


class TextDataset(Dataset):
    """Dataset serving tokenised ASR transcripts + labels (no acoustics).

    Args:
        meld_root: Path to MELD root directory.
        transcripts_path: Directory containing {split}_transcripts.json.
        split: One of 'train', 'dev', 'test'.
        tokenizer: HuggingFace tokenizer for XLM-RoBERTa.
        max_length: Maximum tokenised sequence length.
    """

    def __init__(
        self,
        meld_root: str,
        transcripts_path: str,
        split: str,
        tokenizer,
        max_length: int = 128,
        filtered_keys_path: Optional[str] = None,
        text_source: str = "asr",
    ) -> None:
        assert text_source in ("asr", "gold"), text_source
        self.text_source = text_source
        self.tokenizer = tokenizer
        self.max_length = max_length

        csv_path = Path(meld_root) / f"{split}_sent_emo.csv"
        df = pd.read_csv(csv_path)
        df.columns = (
            df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
        )
        df = df.dropna(subset=["emotion", "sentiment"]).reset_index(drop=True)
        df["emotion"]   = df["emotion"].str.strip().str.lower()
        df["sentiment"] = df["sentiment"].str.strip().str.lower()

        transcripts: Dict[str, str] = {}
        if text_source == "asr":
            with open(
                Path(transcripts_path) / f"{split}_transcripts.json",
                "r", encoding="utf-8",
            ) as f:
                transcripts = json.load(f)

        keep_set: Optional[set] = None
        self.num_before_filter = len(df)
        if filtered_keys_path is not None:
            p = Path(filtered_keys_path)
            if not p.exists():
                raise FileNotFoundError(
                    f"filtered_keys_path set but file not found at {p}."
                )
            with open(p, "r", encoding="utf-8") as f:
                keep_set = set(json.load(f)["keys"])

        self.samples: List[Tuple[str, int, int]] = []
        for _, row in df.iterrows():
            key = f"dia{int(row['dialogue_id'])}_utt{int(row['utterance_id'])}"
            if keep_set is not None and key not in keep_set:
                continue
            # Must match how the model was TRAINED. Scoring a gold-trained
            # model on ASR text is a train/test domain mismatch, which is
            # exactly what the pipeline did unnoticed until 2026-08-26.
            text = (transcripts.get(key, "") if text_source == "asr"
                    else str(row.get("utterance", "") or "").strip())
            self.samples.append((
                text,
                EMOTION2IDX.get(row["emotion"], 0),
                SENTIMENT2IDX.get(row["sentiment"], 1),
            ))
        self.num_after_filter = len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        text, emotion_label, sentiment_label = self.samples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "emotion_label":  torch.tensor(emotion_label, dtype=torch.long),
            "sentiment_label": torch.tensor(sentiment_label, dtype=torch.long),
        }


@torch.no_grad()
def run_inference(model, loader, device):
    """Run text-only inference; return preds, labels, probs for both tasks."""
    model.eval()
    emotion_preds, emotion_labels, emotion_probs_l = [], [], []
    sentiment_preds, sentiment_labels, sentiment_probs_l = [], [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        e_labels       = batch["emotion_label"].to(device)
        s_labels       = batch["sentiment_label"].to(device)

        sentiment_logits, emotion_logits = model(input_ids, attention_mask)

        emotion_preds.extend(emotion_logits.argmax(dim=-1).cpu().tolist())
        emotion_labels.extend(e_labels.cpu().tolist())
        emotion_probs_l.append(F.softmax(emotion_logits, dim=-1).cpu().numpy())

        sentiment_preds.extend(sentiment_logits.argmax(dim=-1).cpu().tolist())
        sentiment_labels.extend(s_labels.cpu().tolist())
        sentiment_probs_l.append(F.softmax(sentiment_logits, dim=-1).cpu().numpy())

    return (
        emotion_preds, emotion_labels, np.concatenate(emotion_probs_l),
        sentiment_preds, sentiment_labels, np.concatenate(sentiment_probs_l),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Text-only baseline evaluation using Phase 1 checkpoint."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--filtered_keys_path", type=str, default=None,
        help=("Override the config's keep-list for the scored split. Needed for "
              "a cross-evaluation matrix, where the TEST subset must be chosen "
              "independently of whatever the model was TRAINED with."),
    )
    parser.add_argument(
        "--text_source", type=str, default="asr", choices=["asr", "gold"],
        help="Must match the model's training text source.",
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "dev", "test"],
        help="Which split to score. dev lets the same code report validation "
             "metrics, so train/dev/test numbers are all produced identically.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help=(
            "Path to Phase 1 best_model.pt. Defaults to "
            "<checkpoint_dir>/best_model.pt from the config."
        ),
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="",
        help=(
            "Suffix added to output filenames (test_results_text_only<tag>.json, "
            "confusion_*_text_only<tag>.png). Use to keep ablation runs distinct. "
            "Example: '_focal_sampler'."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["data"]["seed"])
    device = get_device()

    logger = setup_logging(config["training"]["log_dir"], "evaluate_text_only")
    logger.info(
        "Config: %s | Device: %s | Text-only baseline | tag=%r",
        args.config, device, args.tag,
    )

    xlmr_id   = config["model"]["xlmr_id"]
    tokenizer = AutoTokenizer.from_pretrained(xlmr_id)

    # The TEST subset is decided ENTIRELY by the CLI, never by the training
    # config. A model trained on asr_cleaned has filtering.enabled=true, and if
    # that leaked into evaluation it could only ever be scored on the filtered
    # subset -- making the off-diagonal cells of the cross-evaluation matrix
    # impossible. Absent --filtered_keys_path means the FULL split.
    #
    # The previous form raised unconditionally: it referenced an undefined
    # `use_filter`, and its branch assigned None and then raised *because* the
    # value was None. Every gold and asr cell of the matrix failed on it.
    test_filter_keys: Optional[str] = args.filtered_keys_path
    if test_filter_keys is not None and not Path(test_filter_keys).exists():
        raise FileNotFoundError(
            f"--filtered_keys_path given but not found: {test_filter_keys}")
    logger.info("test subset: %s",
                Path(test_filter_keys).name if test_filter_keys
                else f"FULL {args.split} split")

    test_ds = TextDataset(
        meld_root          = config["data"]["meld_root"],
        transcripts_path   = config["data"]["transcripts_path"],
        split              = args.split,
        tokenizer          = tokenizer,
        max_length         = config["data"]["max_text_length"],
        filtered_keys_path = test_filter_keys,
        text_source        = args.text_source,
    )
    logger.info("%s set: %d samples | text_source=%s", args.split, len(test_ds), args.text_source)
    if test_filter_keys is not None:
        logger.info(
            "Filter enabled | test: %d/%d kept",
            test_ds.num_after_filter, test_ds.num_before_filter,
        )

    test_loader = DataLoader(
        test_ds,
        batch_size  = config["training"]["batch_size"],
        shuffle     = False,
        num_workers = config["data"]["num_workers"],
        pin_memory  = True,
        drop_last   = False,
    )

    # Load Phase 1 checkpoint (CLI override takes precedence over config path)
    ckpt_path = Path(args.checkpoint_path) if args.checkpoint_path else (
        Path(config["training"]["checkpoint_dir"]) / "best_model.pt"
    )
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Phase 1 checkpoint not found: {ckpt_path}")

    model = XLMRobertaClassifier(
        model_name_or_path    = xlmr_id,
        num_sentiment_classes = config["model"]["num_sentiment_classes"],
        num_emotion_classes   = config["model"]["num_classes"],
        dropout_prob          = config["model"]["dropout"],
    )
    state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    logger.info("Loaded Phase 1 checkpoint: %s", ckpt_path)

    # Inference
    (
        emotion_preds, emotion_labels, emotion_probs,
        sentiment_preds, sentiment_labels, sentiment_probs,
    ) = run_inference(model, test_loader, device)

    # Metrics
    emotion_metrics   = compute_all_metrics(emotion_preds, emotion_labels, emotion_probs, EMOTION_NAMES)
    sentiment_metrics = compute_all_metrics(sentiment_preds, sentiment_labels, sentiment_probs, SENTIMENT_NAMES)

    log_full_metrics(emotion_metrics,   "emotion",   logger)
    log_full_metrics(sentiment_metrics, "sentiment", logger)

    # Save results
    output_dir = Path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(
        emotion_preds, emotion_labels, EMOTION_NAMES,
        title     = f"Emotion — XLM-RoBERTa text-only (test){args.tag}",
        save_path = output_dir / f"confusion_emotion_text_only{args.tag}.png",
    )
    plot_confusion_matrix(
        sentiment_preds, sentiment_labels, SENTIMENT_NAMES,
        title     = f"Sentiment — XLM-RoBERTa text-only (test){args.tag}",
        save_path = output_dir / f"confusion_sentiment_text_only{args.tag}.png",
    )

    results = {
        "config":       args.config,
        "model":        "text_only",
        "tag":          args.tag,
        "checkpoint":   str(ckpt_path),
        "test_samples": len(test_ds),
        "emotion":      {k: v for k, v in emotion_metrics.items() if k != "report"},
        "sentiment":    {k: v for k, v in sentiment_metrics.items() if k != "report"},
    }
    results_path = output_dir / f"test_results_text_only{args.tag}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
