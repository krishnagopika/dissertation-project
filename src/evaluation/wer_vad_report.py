#!/usr/bin/env python3.12
"""WER + VAD report over EVERY utterance — measurement only, no filtering.

Reads the per-utterance metadata that compute_filter_metadata.py produces
(Silero-VAD speech_ratio + Voxtral WER) and turns it into something readable:
a per-utterance CSV and a summary you can put in a write-up.

Deliberately does NOT filter anything. apply_filter.py is the step that turns
these numbers into keep-lists; this script only describes the distribution. The
"survival at threshold" table is informational — it tells you what a filter
WOULD discard, so the cost of a threshold is visible before you commit to one.

Outputs, per split, under {evaluation.output_dir}/wer_vad/:

    {split}_utterances.csv   one row per utterance, joined to its MELD labels
    {split}_summary.json     percentiles, per-emotion means, threshold table
    report.md                all splits together, human-readable

Usage:
    python3.12 src/evaluation/wer_vad_report.py \\
        --config src/configs/extract_mini.yaml --splits train dev test
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import load_config, setup_logging

_SPLIT_CSV: Dict[str, str] = {
    "train": "train_sent_emo.csv",
    "dev": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv",
}

#: Thresholds reported for reference. These mirror the policies in mini.yaml so
#: the report can be read against existing keep-lists, but nothing is applied.
_WER_THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.40)
_VAD_MIN = 0.20


def percentile(values: List[float], q: float) -> float:
    """Linear-interpolated percentile. Avoids a numpy import for six numbers."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = q / 100.0 * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def load_labels(meld_root: Path, split: str) -> Dict[str, Dict[str, str]]:
    """Map utterance key -> {emotion, sentiment, utterance} from the MELD CSV."""
    import pandas as pd

    df = pd.read_csv(meld_root / _SPLIT_CSV[split])
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    df["emotion"] = df["emotion"].str.strip().str.lower()
    df["sentiment"] = df["sentiment"].str.strip().str.lower()

    out: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        key = f"dia{int(row['dialogue_id'])}_utt{int(row['utterance_id'])}"
        out[key] = {
            "emotion": row["emotion"],
            "sentiment": row["sentiment"],
            "gold_text": str(row.get("utterance", "")).strip(),
        }
    return out


def summarise(records: List[dict], labels: Dict[str, Dict[str, str]]) -> dict:
    """Compute the distribution summary for one split."""
    wers = [r["vox_wer"] for r in records if r.get("vox_wer") is not None]
    vads = [r["speech_ratio"] for r in records if r.get("speech_ratio") is not None]
    durs = [r["duration_sec"] for r in records if r.get("duration_sec") is not None]

    # Per-emotion mean WER. Reported with n, because a mean over a handful of
    # utterances is noise -- see POSTMORTEMS.md PM-005.
    by_emotion: Dict[str, List[float]] = defaultdict(list)
    for r in records:
        if r.get("vox_wer") is None:
            continue
        emo = labels.get(r["key"], {}).get("emotion", "unknown")
        by_emotion[emo].append(r["vox_wer"])

    # What a filter WOULD keep. Informational only -- nothing is applied here.
    survival = {}
    for th in _WER_THRESHOLDS:
        kept = [r for r in records
                if r.get("vox_wer") is not None
                and r["vox_wer"] <= th
                and (r.get("speech_ratio") or 0.0) >= _VAD_MIN]
        per_emo = defaultdict(int)
        for r in kept:
            per_emo[labels.get(r["key"], {}).get("emotion", "unknown")] += 1
        survival[f"wer{int(th * 100)}"] = {
            "kept": len(kept),
            "pct": round(100.0 * len(kept) / max(1, len(records)), 1),
            "per_emotion": dict(per_emo),
        }

    return {
        "n_utterances": len(records),
        "n_scored": len(wers),
        "wer": {
            "mean": round(sum(wers) / len(wers), 4) if wers else None,
            "p10": round(percentile(wers, 10), 4),
            "median": round(percentile(wers, 50), 4),
            "p90": round(percentile(wers, 90), 4),
            "perfect_zero": sum(1 for w in wers if w == 0.0),
            "over_1.0": sum(1 for w in wers if w > 1.0),
        },
        "speech_ratio": {
            "mean": round(sum(vads) / len(vads), 4) if vads else None,
            "p10": round(percentile(vads, 10), 4),
            "median": round(percentile(vads, 50), 4),
            "below_0.20": sum(1 for v in vads if v < _VAD_MIN),
        },
        "duration_sec": {
            "mean": round(sum(durs) / len(durs), 2) if durs else None,
            "median": round(percentile(durs, 50), 2),
            "p90": round(percentile(durs, 90), 2),
            "max": round(max(durs), 2) if durs else None,
            "over_30s": sum(1 for d in durs if d > 30.0),
        },
        "wer_by_emotion": {
            emo: {"n": len(v), "mean_wer": round(sum(v) / len(v), 4)}
            for emo, v in sorted(by_emotion.items(), key=lambda kv: -len(kv[1]))
        },
        "survival_if_filtered": survival,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"],
                    choices=["train", "dev", "test"])
    args = ap.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config["training"]["log_dir"], "wer_vad_report")

    meld_root = Path(config["data"]["meld_root"])
    transcripts_dir = Path(config["data"]["transcripts_path"])
    metadata_dir = Path(config["filtering"]["metadata_dir"])
    out_dir = Path(config["evaluation"]["output_dir"]) / "wer_vad"
    out_dir.mkdir(parents=True, exist_ok=True)

    md_lines: List[str] = [
        "# WER + VAD report",
        "",
        f"Config: `{args.config}`  ",
        f"Metadata: `{metadata_dir}`",
        "",
        "Measurement only — no filtering applied. The survival tables show what a",
        "threshold *would* discard, so the cost is visible before committing.",
        "",
    ]

    all_summaries = {}
    for split in args.splits:
        meta_path = metadata_dir / f"{split}_filter_metadata.json"
        if not meta_path.exists():
            logger.error("%s missing — run compute_filter_metadata.py first: %s",
                         split, meta_path)
            continue

        with open(meta_path, encoding="utf-8") as f:
            records = json.load(f)
        labels = load_labels(meld_root, split)

        tr_path = transcripts_dir / f"{split}_transcripts.json"
        transcripts = {}
        if tr_path.exists():
            with open(tr_path, encoding="utf-8") as f:
                transcripts = json.load(f)

        # --- per-utterance CSV -------------------------------------------
        csv_path = out_dir / f"{split}_utterances.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["key", "emotion", "sentiment", "vox_wer", "speech_ratio",
                        "duration_sec", "gold_words", "vox_words",
                        "gold_text", "asr_text"])
            for r in records:
                lab = labels.get(r["key"], {})
                w.writerow([
                    r["key"], lab.get("emotion", ""), lab.get("sentiment", ""),
                    r.get("vox_wer", ""), r.get("speech_ratio", ""),
                    r.get("duration_sec", ""), r.get("gold_words", ""),
                    r.get("vox_words", ""),
                    lab.get("gold_text", ""), transcripts.get(r["key"], ""),
                ])

        summary = summarise(records, labels)
        all_summaries[split] = summary
        with open(out_dir / f"{split}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info("%s | n=%d | mean WER %.4f | median %.4f | VAD<0.20: %d",
                    split, summary["n_utterances"], summary["wer"]["mean"] or 0,
                    summary["wer"]["median"], summary["speech_ratio"]["below_0.20"])

        # --- markdown ------------------------------------------------------
        md_lines += [
            f"## {split}  (n = {summary['n_utterances']})", "",
            "| metric | mean | p10 | median | p90 |",
            "|---|---:|---:|---:|---:|",
            f"| WER | {summary['wer']['mean']} | {summary['wer']['p10']} | "
            f"{summary['wer']['median']} | {summary['wer']['p90']} |",
            f"| speech_ratio | {summary['speech_ratio']['mean']} | "
            f"{summary['speech_ratio']['p10']} | {summary['speech_ratio']['median']} | — |",
            f"| duration_sec | {summary['duration_sec']['mean']} | — | "
            f"{summary['duration_sec']['median']} | {summary['duration_sec']['p90']} |",
            "",
            f"- WER exactly 0: **{summary['wer']['perfect_zero']}**  ·  "
            f"WER > 1.0 (more errors than words): **{summary['wer']['over_1.0']}**",
            f"- speech_ratio < 0.20: **{summary['speech_ratio']['below_0.20']}**  ·  "
            f"clips > 30 s: **{summary['duration_sec']['over_30s']}**",
            "",
            "### Mean WER by emotion", "",
            "| emotion | n | mean WER |", "|---|---:|---:|",
        ]
        for emo, v in summary["wer_by_emotion"].items():
            md_lines.append(f"| {emo} | {v['n']} | {v['mean_wer']} |")
        md_lines += [
            "", "### What a filter would keep (NOT applied)", "",
            "| policy | kept | % |", "|---|---:|---:|",
        ]
        for pol, v in summary["survival_if_filtered"].items():
            md_lines.append(f"| {pol} + VAD≥0.20 | {v['kept']} | {v['pct']}% |")
        md_lines.append("")

    report = out_dir / "report.md"
    report.write_text("\n".join(md_lines), encoding="utf-8")
    with open(out_dir / "all_splits_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    logger.info("Report → %s", report)


if __name__ == "__main__":
    main()
