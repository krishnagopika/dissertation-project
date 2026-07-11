"""
augment_transcripts.py — Generate Voxtral paraphrases for minority emotion classes
====================================================================================
Imbalance in MELD (fear=268, disgust=271 vs neutral=4710) caps Phase 1 per-class
F1 even with class-weighted loss. This script generates additional training-only
text data by prompting Voxtral to paraphrase existing minority-class transcripts
while preserving the emotion label.

Pipeline
--------
  train_sent_emo.csv + train_transcripts.json
        ↓
  filter to emotions with per_class_multiplier > 0
        ↓
  Voxtral-Mini-3B (vllm, text-only chat, temperature>0, n=multiplier)
        ↓
  data/meld_transcripts/train_paraphrases.json
        list of {key, emotion, sentiment, original, text}

Downstream: finetune.py loads this file in addition to train_transcripts.json
when training.use_paraphrases=true.

Usage
-----
  python3.12 src/preprocessing/augment_transcripts.py --config src/configs/mini.yaml

Slurm: see src/scripts/augment.sbatch
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import load_config, set_seed, setup_logging


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TRAIN_CSV = "train_sent_emo.csv"


_PROMPT_TEMPLATE = (
    "You are rewriting a single line of dialogue for a TV-show character.\n"
    "The character is expressing the emotion: {emotion}.\n\n"
    "Original line: \"{text}\"\n\n"
    "Rewrite this line in different words while keeping the same emotion "
    "({emotion}) and the same overall meaning. Use casual, conversational "
    "English. Keep it 1-2 short sentences. Return ONLY the rewritten line — "
    "no quotes, no preamble, no explanation, no list."
)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def build_minority_index(
    meld_root: Path,
    transcripts_path: Path,
    per_class_multiplier: Dict[str, int],
    logger: logging.Logger,
) -> List[Tuple[str, str, str, str, int]]:
    """Build list of (key, transcript, emotion, sentiment, n_paraphrases) for
    every train utterance whose emotion has a positive multiplier.

    Args:
        meld_root: Directory containing train_sent_emo.csv.
        transcripts_path: Directory containing train_transcripts.json.
        per_class_multiplier: emotion name → number of paraphrases to generate.
        logger: Logger instance.

    Returns:
        List of (key, transcript, emotion, sentiment, n_paraphrases) tuples.

    Raises:
        FileNotFoundError: If CSV or transcripts JSON are missing.
    """
    csv_path = meld_root / _TRAIN_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"MELD train CSV not found: {csv_path}")

    transcripts_file = transcripts_path / "train_transcripts.json"
    if not transcripts_file.exists():
        raise FileNotFoundError(
            f"train_transcripts.json not found at {transcripts_file}. "
            f"Run preprocessing/transcribe_all.py first."
        )

    with open(transcripts_file, "r", encoding="utf-8") as f:
        transcripts: Dict[str, str] = json.load(f)

    df = pd.read_csv(csv_path)
    df.columns = (
        df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    )
    df = df.dropna(subset=["emotion", "sentiment"]).reset_index(drop=True)
    df["emotion"] = df["emotion"].str.strip().str.lower()
    df["sentiment"] = df["sentiment"].str.strip().str.lower()

    records: List[Tuple[str, str, str, str, int]] = []
    skipped_missing = 0
    skipped_empty = 0
    for _, row in df.iterrows():
        emo = row["emotion"]
        n = int(per_class_multiplier.get(emo, 0))
        if n <= 0:
            continue
        dia = int(row["dialogue_id"])
        utt = int(row["utterance_id"])
        key = f"dia{dia}_utt{utt}"
        transcript = transcripts.get(key, "")
        if transcript is None:
            skipped_missing += 1
            continue
        transcript = transcript.strip()
        if not transcript:
            skipped_empty += 1
            continue
        records.append((key, transcript, emo, row["sentiment"], n))

    counts: Dict[str, int] = {}
    for _, _, emo, _, n in records:
        counts[emo] = counts.get(emo, 0) + n
    logger.info(
        "Indexed %d minority utterances. Paraphrases to generate per emotion: %s "
        "(skipped %d missing, %d empty)",
        len(records), counts, skipped_missing, skipped_empty,
    )
    return records


# ---------------------------------------------------------------------------
# Voxtral generation
# ---------------------------------------------------------------------------

def generate_paraphrases(
    records: List[Tuple[str, str, str, str, int]],
    model_id: str,
    tensor_parallel_size: int,
    temperature: float,
    max_tokens: int,
    batch_size: int,
    logger: logging.Logger,
) -> List[Dict[str, str]]:
    """Generate paraphrases for every record via Voxtral on vllm.

    Each record is sent once with sampling param n=multiplier so vllm returns
    that many paraphrases in a single call. Calls are batched across records.

    Args:
        records: Output of build_minority_index.
        model_id: HuggingFace model identifier for Voxtral.
        tensor_parallel_size: vllm tp size.
        temperature: Sampling temperature (>0 for diversity across paraphrases).
        max_tokens: Maximum tokens to generate per paraphrase.
        batch_size: Number of records to send per vllm.chat() call.
        logger: Logger instance.

    Returns:
        List of dicts with keys: key, emotion, sentiment, original, text.
    """
    from vllm import LLM, SamplingParams

    logger.info(
        "Loading Voxtral via vllm: %s | tp=%d | temperature=%.2f | "
        "max_tokens=%d | batch_size=%d",
        model_id, tensor_parallel_size, temperature, max_tokens, batch_size,
    )
    llm = LLM(
        model=model_id,
        tokenizer_mode="mistral",
        max_model_len=4096,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=True,
    )

    paraphrases: List[Dict[str, str]] = []
    total = len(records)

    for batch_start in range(0, total, batch_size):
        batch = records[batch_start: batch_start + batch_size]
        messages_batch = []
        meta = []
        for key, transcript, emo, sent, n in batch:
            prompt = _PROMPT_TEMPLATE.format(emotion=emo, text=transcript)
            messages_batch.append(
                [{"role": "user", "content": prompt}]
            )
            meta.append((key, transcript, emo, sent, n))

        # n varies per record; vllm.chat accepts a list of SamplingParams
        sampling_params_list = [
            SamplingParams(
                n=n,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.95,
            )
            for (_, _, _, _, n) in meta
        ]

        try:
            outputs = llm.chat(
                messages_batch, sampling_params=sampling_params_list
            )
        except Exception as exc:
            logger.error(
                "vllm batch error (batch starting %d): %s", batch_start, exc
            )
            continue

        for (key, transcript, emo, sent, _), output in zip(meta, outputs):
            for completion in output.outputs:
                text = completion.text.strip()
                # Strip surrounding quotes the model often adds despite the prompt
                if len(text) >= 2 and text[0] in ('"', "'") and text[-1] == text[0]:
                    text = text[1:-1].strip()
                if not text:
                    continue
                paraphrases.append({
                    "key": key,
                    "emotion": emo,
                    "sentiment": sent,
                    "original": transcript,
                    "text": text,
                })

        done = min(batch_start + batch_size, total)
        logger.info(
            "  generated batch | %d / %d source utterances | "
            "%d paraphrases so far",
            done, total, len(paraphrases),
        )

    del llm
    torch.cuda.empty_cache()
    gc.collect()
    return paraphrases


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Voxtral paraphrases of minority-class MELD transcripts "
            "for Phase 1 text augmentation."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config yaml (mini.yaml or small.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["data"]["seed"])

    log_dir = config["training"]["log_dir"]
    logger = setup_logging(log_dir, "augment_transcripts")
    logger.info("Config: %s", args.config)

    if "augmentation" not in config:
        raise KeyError(
            "Config missing 'augmentation' block — see mini.yaml for the "
            "expected schema (paraphrases_path, per_class_multiplier, etc.)."
        )

    meld_root = Path(config["data"]["meld_root"])
    transcripts_path = Path(config["data"]["transcripts_path"])
    out_path = Path(config["augmentation"]["paraphrases_path"])

    if out_path.exists():
        logger.info(
            "Paraphrases already exist at %s — skipping. Delete the file to "
            "regenerate.", out_path,
        )
        return

    multiplier = config["augmentation"]["per_class_multiplier"]
    records = build_minority_index(
        meld_root, transcripts_path, multiplier, logger
    )
    if not records:
        logger.warning("No minority utterances indexed — nothing to do.")
        return

    paraphrases = generate_paraphrases(
        records=records,
        model_id=config["model"]["voxtral_id"],
        tensor_parallel_size=int(
            config["model"].get("tensor_parallel_size", 1)
        ),
        temperature=float(config["augmentation"]["voxtral_temperature"]),
        max_tokens=int(config["augmentation"]["voxtral_max_tokens"]),
        batch_size=int(config["augmentation"]["voxtral_batch_size"]),
        logger=logger,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"paraphrases": paraphrases}, f, ensure_ascii=False, indent=2)

    per_emotion: Dict[str, int] = {}
    for p in paraphrases:
        per_emotion[p["emotion"]] = per_emotion.get(p["emotion"], 0) + 1
    logger.info(
        "Done. Wrote %d paraphrases → %s | per-emotion: %s",
        len(paraphrases), out_path, per_emotion,
    )


if __name__ == "__main__":
    main()
