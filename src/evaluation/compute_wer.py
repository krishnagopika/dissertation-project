"""
compute_wer.py — ASR Quality Evaluation via Word Error Rate
===========================================================
Computes WER between MELD gold transcripts (CSV Utterance column) and
Voxtral ASR transcriptions (cached JSON files) for all three splits.

This provides a quantitative measure of Voxtral's transcription quality
on the MELD dataset, justifying the pipeline design choice of fine-tuning
XLM-RoBERTa on ASR-noisy transcripts for domain robustness.

Inputs:
  data/meld_raw/{split}_sent_emo.csv           — gold text (Utterance column)
  data/meld_transcripts/{split}_transcripts.json — Voxtral ASR output

Output:
  results/mini/wer_analysis.json  — corpus WER, per-utterance stats per split

Usage
-----
  python3.12 src/evaluation/compute_wer.py --config src/configs/mini.yaml

Slurm: see src/scripts/compute_wer.sbatch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import jiwer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import load_config, setup_logging


# ---------------------------------------------------------------------------
# Text normalisation transform (standard ASR evaluation practice)
# ---------------------------------------------------------------------------

NORMALISE = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.RemoveMultipleSpaces(),
])

_SPLIT_CSV: Dict[str, str] = {
    "train": "train_sent_emo.csv",
    "dev":   "dev_sent_emo.csv",
    "test":  "test_sent_emo.csv",
}


# ---------------------------------------------------------------------------
# Per-split WER computation
# ---------------------------------------------------------------------------

def compute_split_wer(
    split: str,
    meld_root: Path,
    transcripts_dir: Path,
    logger,
) -> Dict:
    """Compute WER for one MELD split.

    Args:
        split: One of 'train', 'dev', 'test'.
        meld_root: Path to directory containing MELD CSV files.
        transcripts_dir: Path to directory containing {split}_transcripts.json.
        logger: Logger instance.

    Returns:
        Dictionary with corpus_wer, mean_utt_wer, std_utt_wer,
        n_matched, n_missing, n_total.
    """
    # ---- Load CSV ----
    csv_path = meld_root / _SPLIT_CSV[split]
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    df = df.dropna(subset=["utterance"]).reset_index(drop=True)

    # ---- Load transcripts ----
    transcript_file = transcripts_dir / f"{split}_transcripts.json"
    if not transcript_file.exists():
        raise FileNotFoundError(f"Transcripts not found: {transcript_file}")
    with open(transcript_file, "r", encoding="utf-8") as f:
        transcripts: Dict[str, str] = json.load(f)

    # ---- Align gold text with ASR transcripts ----
    gold_texts:  List[str] = []
    hyp_texts:   List[str] = []
    n_missing = 0

    for _, row in df.iterrows():
        key = f"dia{int(row['dialogue_id'])}_utt{int(row['utterance_id'])}"
        gold = str(row["utterance"]).strip()
        hyp  = transcripts.get(key, "")

        if not hyp:
            n_missing += 1
            logger.debug("Missing transcript for key: %s", key)

        gold_texts.append(gold)
        hyp_texts.append(hyp)

    n_total   = len(gold_texts)
    n_matched = n_total - n_missing
    logger.info(
        "[%s] Total: %d | Matched: %d | Missing ASR: %d",
        split, n_total, n_matched, n_missing,
    )

    # ---- Compute corpus-level WER ----
    corpus_wer = jiwer.wer(
        gold_texts,
        hyp_texts,
        reference_transform=NORMALISE,
        hypothesis_transform=NORMALISE,
    )

    # ---- Compute per-utterance WER for mean/std ----
    utt_wers: List[float] = []
    for gold, hyp in zip(gold_texts, hyp_texts):
        try:
            w = jiwer.wer(
                gold,
                hyp,
                reference_transform=NORMALISE,
                hypothesis_transform=NORMALISE,
            )
        except Exception:
            w = 1.0  # treat any failure as 100% WER
        utt_wers.append(w)

    utt_arr = np.array(utt_wers)
    mean_wer = float(np.mean(utt_arr))
    std_wer  = float(np.std(utt_arr))

    logger.info(
        "[%s] Corpus WER: %.4f | Mean utt WER: %.4f ± %.4f",
        split, corpus_wer, mean_wer, std_wer,
    )

    # ---- Log worst examples ----
    worst_idx = np.argsort(utt_arr)[-5:][::-1]
    logger.info("[%s] Worst 5 utterances by WER:", split)
    for i in worst_idx:
        row = df.iloc[i]
        key = f"dia{int(row['dialogue_id'])}_utt{int(row['utterance_id'])}"
        logger.info(
            "  %s | WER=%.4f\n    GOLD: %s\n    ASR : %s",
            key, utt_arr[i], gold_texts[i], hyp_texts[i],
        )

    return {
        "corpus_wer":   round(corpus_wer, 6),
        "mean_utt_wer": round(mean_wer, 6),
        "std_utt_wer":  round(std_wer, 6),
        "n_total":      n_total,
        "n_matched":    n_matched,
        "n_missing":    n_missing,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute WER between MELD gold text and Voxtral ASR transcriptions."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config yaml (mini.yaml or small.yaml)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "test"],
        choices=["train", "dev", "test"],
        help="Which splits to evaluate (default: all three)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    log_dir = config["training"]["log_dir"]
    logger  = setup_logging(log_dir, "compute_wer")
    logger.info("Config: %s | Splits: %s", args.config, args.splits)

    meld_root       = Path(config["data"]["meld_root"])
    transcripts_dir = Path(config["data"]["transcripts_path"])
    output_dir      = Path(config["evaluation"]["output_dir"])

    # ---- Compute WER for each split ----
    all_results: Dict[str, Dict] = {}

    for split in args.splits:
        logger.info("=" * 50)
        logger.info("Processing split: %s", split)
        results = compute_split_wer(split, meld_root, transcripts_dir, logger)
        all_results[split] = results

    # ---- Summary ----
    logger.info("=" * 50)
    logger.info("WER SUMMARY — Voxtral ASR vs MELD Gold Transcripts")
    logger.info("=" * 50)
    for split, res in all_results.items():
        logger.info(
            "%5s | Corpus WER: %.4f | Mean utt WER: %.4f ± %.4f | "
            "Matched: %d / %d",
            split,
            res["corpus_wer"],
            res["mean_utt_wer"],
            res["std_utt_wer"],
            res["n_matched"],
            res["n_total"],
        )

    # ---- Save results ----
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "wer_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
