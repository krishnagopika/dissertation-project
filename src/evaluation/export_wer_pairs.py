"""
export_wer_pairs.py — Side-by-side gold / ASR / audio for error review
=======================================================================
For every MELD utterance whose WER exceeds a threshold, writes a CSV row
with the audio path, the gold transcript, and the ASR transcript, so the
errors can be listened to and read side by side.

Columns: split, key, dialogue_id, utterance_id, wer, audio_path,
         gold_text, asr_text

Rows are sorted by WER descending (worst first). WER is computed per
utterance with the same normalisation as compute_wer.py (lowercase,
punctuation-stripped).

Output: {output_dir}/wer_pairs_{asr}.csv

Usage
-----
  python3.12 src/evaluation/export_wer_pairs.py --config src/configs/mini.yaml
  python3.12 src/evaluation/export_wer_pairs.py --config src/configs/mini.yaml \\
      --asr whisper --splits test --threshold 0.1
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import jiwer
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import load_config, setup_logging

# Same normalisation as compute_wer.py, so per-utterance WER is consistent.
NORMALISE = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.Strip(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ReduceToListOfListOfWords(),
])

_SPLIT_CSV: Dict[str, str] = {
    "train": "train_sent_emo.csv",
    "dev":   "dev_sent_emo.csv",
    "test":  "test_sent_emo.csv",
}


def utt_wer(gold: str, hyp: str) -> float:
    """Per-utterance WER with the shared normalisation (1.0 on failure)."""
    try:
        return float(
            jiwer.wer(
                gold, hyp,
                reference_transform=NORMALISE,
                hypothesis_transform=NORMALISE,
            )
        )
    except Exception:  # noqa: BLE001 — degenerate pairs score as 100% WER
        return 1.0


def collect_rows(
    split: str,
    meld_root: Path,
    transcripts_dir: Path,
    filename_tmpl: str,
    threshold: float,
    logger,
) -> List[dict]:
    """Build the list of above-threshold rows for one split.

    Args:
        split: One of 'train', 'dev', 'test'.
        meld_root: Directory with MELD CSVs and per-split audio folders.
        transcripts_dir: Directory with the ASR transcript JSON.
        filename_tmpl: Transcript filename template containing "{split}".
        threshold: Keep utterances with WER strictly greater than this.
        logger: Logger instance.

    Returns:
        List of row dicts (unsorted).
    """
    df = pd.read_csv(meld_root / _SPLIT_CSV[split])
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    df = df.dropna(subset=["utterance"]).reset_index(drop=True)

    transcript_file = transcripts_dir / filename_tmpl.format(split=split)
    if not transcript_file.exists():
        raise FileNotFoundError(f"Transcripts not found: {transcript_file}")
    with open(transcript_file, "r", encoding="utf-8") as f:
        hyps: Dict[str, str] = json.load(f)

    audio_dir = meld_root / split
    rows: List[dict] = []
    for _, r in df.iterrows():
        dia, utt = int(r["dialogue_id"]), int(r["utterance_id"])
        key = f"dia{dia}_utt{utt}"
        gold = str(r["utterance"]).strip()
        hyp = str(hyps.get(key, "")).strip()
        w = utt_wer(gold, hyp)
        if w > threshold:
            rows.append({
                "split": split,
                "key": key,
                "dialogue_id": dia,
                "utterance_id": utt,
                "wer": round(w, 4),
                "audio_path": str(audio_dir / f"{key}.mp4"),
                "gold_text": gold,
                "asr_text": hyp,
            })

    logger.info(
        "[%s] %d / %d utterances with WER > %.2f",
        split, len(rows), len(df), threshold,
    )
    return rows


def main() -> None:
    """Export above-threshold gold/ASR/audio triples to a CSV."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config yaml (mini.yaml or small.yaml)")
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"],
                        choices=["train", "dev", "test"],
                        help="Which splits to export (default: all three)")
    parser.add_argument("--asr", type=str, default="voxtral",
                        choices=["voxtral", "whisper"],
                        help="Which ASR transcripts to use (default: voxtral)")
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Keep utterances with WER strictly above this")
    args = parser.parse_args()

    config = load_config(args.config)
    filename_tmpl = {
        "voxtral": "{split}_transcripts.json",
        "whisper": "{split}_transcripts_whisper.json",
    }[args.asr]

    logger = setup_logging(config["training"]["log_dir"], "export_wer_pairs")
    logger.info("ASR: %s | threshold: %.2f | splits: %s",
                args.asr, args.threshold, args.splits)

    meld_root = Path(config["data"]["meld_root"])
    transcripts_dir = Path(config["data"]["transcripts_path"])
    output_dir = Path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[dict] = []
    for split in args.splits:
        all_rows.extend(collect_rows(
            split, meld_root, transcripts_dir, filename_tmpl,
            args.threshold, logger,
        ))

    # Worst first.
    all_rows.sort(key=lambda r: r["wer"], reverse=True)

    out_path = output_dir / f"wer_pairs_{args.asr}.csv"
    fields = ["split", "key", "dialogue_id", "utterance_id", "wer",
              "audio_path", "gold_text", "asr_text"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    logger.info("Wrote %d rows to %s", len(all_rows), out_path)


if __name__ == "__main__":
    main()
