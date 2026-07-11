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
    jiwer.ReduceToListOfListOfWords(),
])

# Same normalisation but reduced to characters, for CER.
NORMALISE_CHARS = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ReduceToListOfListOfChars(),
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
    filename_tmpl: str = "{split}_transcripts.json",
    n_examples: int = 5,
) -> Dict:
    """Compute WER for one MELD split.

    Args:
        split: One of 'train', 'dev', 'test'.
        meld_root: Path to directory containing MELD CSV files.
        transcripts_dir: Path to directory containing the transcript JSON.
        logger: Logger instance.
        filename_tmpl: Template for the transcript filename; must contain
            "{split}" (e.g. "{split}_transcripts.json" for Voxtral or
            "{split}_transcripts_whisper.json" for Whisper).
        n_examples: How many best / worst utterances to record.

    Returns:
        Dictionary with corpus measures, per-utterance stats, and the
        best/worst example utterances.
    """
    # ---- Load CSV ----
    csv_path = meld_root / _SPLIT_CSV[split]
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    df = df.dropna(subset=["utterance"]).reset_index(drop=True)

    # ---- Load transcripts ----
    transcript_file = transcripts_dir / filename_tmpl.format(split=split)
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

    # ---- Compute corpus-level word measures (WER/MER/WIL/WIP) ----
    word_out = jiwer.process_words(
        gold_texts,
        hyp_texts,
        reference_transform=NORMALISE,
        hypothesis_transform=NORMALISE,
    )
    corpus_wer = word_out.wer
    corpus_mer = word_out.mer
    corpus_wil = word_out.wil
    corpus_wip = word_out.wip

    # ---- Compute corpus-level CER ----
    corpus_cer = jiwer.cer(
        gold_texts,
        hyp_texts,
        reference_transform=NORMALISE_CHARS,
        hypothesis_transform=NORMALISE_CHARS,
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
        "[%s] Corpus WER: %.4f | MER: %.4f | WIL: %.4f | WIP: %.4f | "
        "CER: %.4f",
        split, corpus_wer, corpus_mer, corpus_wil, corpus_wip, corpus_cer,
    )
    logger.info(
        "[%s] Mean utt WER: %.4f ± %.4f",
        split, mean_wer, std_wer,
    )

    # ---- Build per-utterance keys ----
    keys = [
        f"dia{int(r['dialogue_id'])}_utt{int(r['utterance_id'])}"
        for _, r in df.iterrows()
    ]

    def _example(i: int) -> Dict:
        return {
            "key":  keys[i],
            "wer":  round(float(utt_arr[i]), 6),
            "gold": gold_texts[i],
            "asr":  hyp_texts[i],
        }

    # ---- Worst examples (highest WER) ----
    worst_idx = np.argsort(utt_arr)[::-1][:n_examples]
    worst = [_example(i) for i in worst_idx]
    logger.info("[%s] Worst %d utterances by WER:", split, n_examples)
    for ex in worst:
        logger.info(
            "  %s | WER=%.4f\n    GOLD: %s\n    ASR : %s",
            ex["key"], ex["wer"], ex["gold"], ex["asr"],
        )

    # ---- Best examples (lowest WER, gold >= 5 words so they are not
    #      trivial one-word utterances like "What?") ----
    gold_lens = np.array([len(g.split()) for g in gold_texts])
    meaningful = np.where(gold_lens >= 5)[0]
    best_pool = meaningful if len(meaningful) else np.arange(len(utt_arr))
    best_idx = best_pool[np.argsort(utt_arr[best_pool])][:n_examples]
    best = [_example(i) for i in best_idx]
    logger.info(
        "[%s] Best %d utterances by WER (gold >= 5 words):", split, n_examples
    )
    for ex in best:
        logger.info(
            "  %s | WER=%.4f\n    GOLD: %s\n    ASR : %s",
            ex["key"], ex["wer"], ex["gold"], ex["asr"],
        )

    return {
        "corpus_wer":   round(corpus_wer, 6),
        "corpus_mer":   round(corpus_mer, 6),
        "corpus_wil":   round(corpus_wil, 6),
        "corpus_wip":   round(corpus_wip, 6),
        "corpus_cer":   round(corpus_cer, 6),
        "mean_utt_wer": round(mean_wer, 6),
        "std_utt_wer":  round(std_wer, 6),
        "n_total":      n_total,
        "n_matched":    n_matched,
        "n_missing":    n_missing,
        "worst_examples": worst,
        "best_examples":  best,
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
    parser.add_argument(
        "--asr",
        type=str,
        default="voxtral",
        choices=["voxtral", "whisper"],
        help="Which ASR transcripts to score (default: voxtral)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Transcript filename template + output filename per ASR system.
    filename_tmpl = {
        "voxtral": "{split}_transcripts.json",
        "whisper": "{split}_transcripts_whisper.json",
    }[args.asr]
    out_filename = {
        "voxtral": "wer_analysis.json",
        "whisper": "wer_analysis_whisper.json",
    }[args.asr]

    log_dir = config["training"]["log_dir"]
    logger  = setup_logging(log_dir, "compute_wer")
    logger.info("Config: %s | ASR: %s | Splits: %s", args.config, args.asr, args.splits)

    meld_root       = Path(config["data"]["meld_root"])
    transcripts_dir = Path(config["data"]["transcripts_path"])
    output_dir      = Path(config["evaluation"]["output_dir"])

    # ---- Compute WER for each split ----
    all_results: Dict[str, Dict] = {}

    for split in args.splits:
        logger.info("=" * 50)
        logger.info("Processing split: %s", split)
        results = compute_split_wer(
            split, meld_root, transcripts_dir, logger,
            filename_tmpl=filename_tmpl,
        )
        all_results[split] = results

    # ---- Summary ----
    logger.info("=" * 50)
    logger.info("WER SUMMARY — %s ASR vs MELD Gold Transcripts", args.asr.upper())
    logger.info("=" * 50)
    for split, res in all_results.items():
        logger.info(
            "%5s | WER: %.4f | MER: %.4f | WIL: %.4f | WIP: %.4f | "
            "CER: %.4f | Matched: %d / %d",
            split,
            res["corpus_wer"],
            res["corpus_mer"],
            res["corpus_wil"],
            res["corpus_wip"],
            res["corpus_cer"],
            res["n_matched"],
            res["n_total"],
        )

    # ---- Save results ----
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / out_filename
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
