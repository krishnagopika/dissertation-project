#!/usr/bin/env python3.12
"""Enrich the per-utterance WER/VAD table with defect flags, then report.

Takes the CSV from build_wer_vad_table.py and adds one column per known defect
class, so each can be counted, cross-tabulated, and excluded independently.

The point is separation. A single "high WER" bucket conflates at least four
distinct failure modes with different causes and different fixes:

  duplicate audio   MELD shipped one .mp4 for several utterances. Byte-identical
                    files, different gold labels. Confirmed not an ASR problem:
                    Voxtral and Whisper produce identical output because they
                    read identical bytes. Unfixable by any model.

  runaway ASR       Sub-second clips give the decoder almost nothing to condition
                    on, so it free-runs on language-model priors and loops until
                    max_tokens. Gold 1-4 words -> ASR 100-176 words.

  no speech         VAD finds no speech at all. Usually laughter, music, or a
                    reaction shot with a label attached.

  short clip        Under a second. Overlaps heavily with the two above but is
                    not identical to either, so it is flagged separately.

Flags are additive, not exclusive: a clip can be short AND duplicated AND
runaway. The report shows the overlap rather than forcing a single category.

Usage:
    python3.12 src/evaluation/analyse_wer_vad.py \\
        --config src/configs/extract_mini.yaml --splits train dev test
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import load_config, setup_logging

#: WER above this is flagged as a transcription failure.
#:
#: 0.50, not 0.40. Two reasons, the second decisive:
#:
#: 1. Standard WER quality bands put 25-50% at "moderate / usable as
#:    supplementary training data" and only recommend filtering from 50%. A
#:    0.40 gate discards data the guidance calls usable.
#:
#: 2. 36% of MELD references are <= 4 words, so WER is severely quantised --
#:    on a 4-word reference it can only be 0, 0.25, 0.50, 0.75, 1.0. A 0.40
#:    threshold therefore sits in a gap where nothing can land; its ONLY effect
#:    is to exclude the WER == 0.50 bucket (358 train utterances, median
#:    reference length 4 words). That is not a stricter quality bar, it is the
#:    same bar with one quantisation level flipped.
#:
#: Only 569 train utterances (5.7%) lie in 0.40 < WER <= 0.50 -- the entire
#: difference between the two settings.
HIGH_WER = 0.50

#: WER thresholds swept for the sensitivity table. The point is to show what a
#: gate costs BEFORE one is chosen -- the four constants in this module are
#: engineering choices, and a sweep is what turns them into evidenced ones.
WER_SWEEP = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00)

#: Audio shorter than this is flagged. 1.0 s ~= 50 encoder frames at 50 Hz.
SHORT_SEC = 1.0
#: Silero speech fraction below this is flagged as effectively non-speech.
LOW_VAD = 0.20
#: An ASR output this many times longer than gold, and this long absolutely,
#: is repetition rather than transcription.
RUNAWAY_RATIO = 5.0
RUNAWAY_MIN_WORDS = 40

_FLAGS = [
    "dup_group_id", "dup_group_size", "is_dup_copy",
    "is_short", "is_low_vad", "is_no_speech",
    "is_runaway", "is_empty_asr", "is_high_wer", "is_wer_over_1",
    "defect_classes",
]


def _f(value: str) -> float | None:
    """Parse a CSV cell to float, or None when blank/unparseable."""
    try:
        return float(value) if str(value).strip() != "" else None
    except ValueError:
        return None


def _i(value: str) -> int:
    """Parse a CSV cell to int, 0 when blank."""
    try:
        return int(float(value)) if str(value).strip() != "" else 0
    except ValueError:
        return 0


def load_dup_map(path: Path) -> Dict[str, Dict[str, Dict]]:
    """Load the duplicate-audio scan into {split: {key: {...}}}.

    Args:
        path: JSON written by the duplicate scan.

    Returns:
        Per-split map from utterance key to its group id, size, and whether it
        is a redundant copy (i.e. not the first key in its group).
    """
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    out: Dict[str, Dict[str, Dict]] = {}
    for split, info in raw.items():
        m: Dict[str, Dict] = {}
        for gid, keys in info.get("detail", {}).items():
            ordered = sorted(keys)
            for i, k in enumerate(ordered):
                m[k] = {"dup_group_id": gid[:12],
                        "dup_group_size": len(ordered),
                        "is_dup_copy": int(i > 0)}
        out[split] = m
    return out


def flag_row(row: dict, dup: Dict[str, Dict], high_wer: float = HIGH_WER) -> dict:
    """Compute the defect flags for one utterance row."""
    dur = _f(row.get("audio_duration_sec", ""))
    vad = _f(row.get("speech_ratio", ""))
    wer = _f(row.get("wer_normalised", ""))
    gw, aw = _i(row.get("gold_words", "")), _i(row.get("asr_words", ""))

    d = dup.get(row["key"], {"dup_group_id": "", "dup_group_size": "",
                             "is_dup_copy": 0})
    flags = {
        **d,
        "is_short": int(dur is not None and dur < SHORT_SEC),
        "is_low_vad": int(vad is not None and vad < LOW_VAD),
        "is_no_speech": int(vad is not None and vad == 0.0),
        "is_runaway": int(gw > 0 and aw > RUNAWAY_RATIO * gw
                          and aw >= RUNAWAY_MIN_WORDS),
        "is_empty_asr": int(aw == 0),
        "is_high_wer": int(wer is not None and wer > high_wer),
        "is_wer_over_1": int(wer is not None and wer > 1.0),
    }
    names = [n for n in ("is_dup_copy", "is_short", "is_low_vad", "is_no_speech",
                         "is_runaway", "is_empty_asr", "is_high_wer") if flags.get(n)]
    flags["defect_classes"] = "|".join(n[3:] for n in names)
    return flags


def summarise(rows: List[dict]) -> dict:
    """Counts, overlaps, and WER by defect class for one split."""
    n = len(rows)
    scored = [r for r in rows if _f(r.get("wer_normalised", "")) is not None]

    def wer_stats(subset):
        vals = sorted(_f(r["wer_normalised"]) for r in subset
                      if _f(r.get("wer_normalised", "")) is not None)
        if not vals:
            return {"n": 0}
        edits = sum(_i(r["sub_norm"]) + _i(r["del_norm"]) + _i(r["ins_norm"])
                    for r in subset)
        refw = sum(_i(r["ref_words_norm"]) for r in subset)
        return {
            "n": len(vals),
            "corpus_wer": round(edits / refw, 4) if refw else None,
            "median_wer": round(vals[len(vals) // 2], 4),
            "mean_wer": round(sum(vals) / len(vals), 4),
        }

    classes = ["is_dup_copy", "is_short", "is_low_vad", "is_no_speech",
               "is_runaway", "is_empty_asr", "is_high_wer", "is_wer_over_1"]
    per_class = {}
    for c in classes:
        sub = [r for r in rows if _i(r.get(c, 0))]
        per_class[c] = {"count": len(sub),
                        "pct": round(100 * len(sub) / max(1, n), 2),
                        **wer_stats(sub)}

    clean = [r for r in rows
             if not any(_i(r.get(c, 0)) for c in
                        ("is_dup_copy", "is_short", "is_low_vad",
                         "is_runaway", "is_empty_asr"))]

    # Sensitivity: what a WER gate costs at each threshold, and -- crucially --
    # how much of that cost is ALREADY covered by the audio-based flags. A gate
    # that only removes clips the audio flags catch anyway adds nothing.
    audio_flags = ("is_dup_copy", "is_short", "is_low_vad", "is_runaway")
    sweep = {}
    for th in WER_SWEEP:
        over = [r for r in rows
                if (_f(r.get("wer_normalised", "")) or -1) > th]
        also_audio = [r for r in over if any(_i(r.get(c, 0)) for c in audio_flags)]
        kept = [r for r in rows if r not in over]
        sweep[f"{th:.2f}"] = {
            "removed": len(over),
            "removed_pct": round(100 * len(over) / max(1, n), 2),
            "already_flagged_by_audio": len(also_audio),
            "uniquely_removed_by_wer": len(over) - len(also_audio),
            "kept": len(kept),
            "kept_corpus_wer": wer_stats(kept).get("corpus_wer"),
        }

    combos = Counter(r.get("defect_classes", "") or "clean" for r in rows)

    by_emotion = defaultdict(lambda: {"n": 0, "defective": 0})
    for r in rows:
        e = r.get("emotion", "?")
        by_emotion[e]["n"] += 1
        if r.get("defect_classes"):
            by_emotion[e]["defective"] += 1
    for e, v in by_emotion.items():
        v["pct_defective"] = round(100 * v["defective"] / max(1, v["n"]), 1)

    return {
        "n_utterances": n,
        "n_scored": len(scored),
        "all": wer_stats(rows),
        "clean_only": {"count": len(clean),
                       "pct": round(100 * len(clean) / max(1, n), 2),
                       **wer_stats(clean)},
        "per_class": per_class,
        "wer_threshold_sweep": sweep,
        "combinations": dict(combos.most_common(20)),
        "by_emotion": dict(sorted(by_emotion.items(),
                                  key=lambda kv: -kv[1]["n"])),
    }


def write_keep_lists(rows: List[dict], split: str, out_dir: Path,
                     logger) -> Dict[str, int]:
    """Emit keep-lists in the same format as the existing *_filtered_keys_*.json.

    TWO lists, because the branches consume different things (MELD_ANALYSIS §5.3):

      {split}_keys_audio_clean.json
          Audio-property defects only: duplicate, short, low VAD, runaway,
          empty ASR. These clips are broken in BOTH modalities.
          -> use for the acoustic branch AND as the floor for everything.

      {split}_keys_text_clean.json
          The above PLUS high WER. A high-WER clip usually has fine audio and a
          correct emotion label; only its transcript is unusable.
          -> use for the TEXT branch only.

    Applying the text list to the acoustic branch discards ~1,700 clips of good
    acoustic training data to protect the text branch, which is the likely
    reason the earlier WER-filtered bc-LSTM result came out as noise.

    Args:
        rows: Flagged utterance rows for one split.
        split: Split name.
        out_dir: Directory to write the keep-lists into.
        logger: Logger instance.

    Returns:
        Counts per list.
    """
    audio_defects = ("is_dup_copy", "is_short", "is_low_vad",
                     "is_runaway", "is_empty_asr")

    audio_clean = [r["key"] for r in rows
                   if not any(_i(r.get(c, 0)) for c in audio_defects)]
    text_clean = [r["key"] for r in rows
                  if not any(_i(r.get(c, 0)) for c in audio_defects)
                  and not _i(r.get("is_high_wer", 0))]

    from collections import Counter as _C
    emo = {r["key"]: r.get("emotion", "") for r in rows}

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, keys in (("audio_clean", audio_clean), ("text_clean", text_clean)):
        p = out_dir / f"{split}_keys_{name}.json"
        # Same schema apply_filter.py emits, so finetune.py and the dataset
        # classes consume these through the existing filtered_keys_path path
        # rather than needing a second loader. A bare list silently fails
        # there: the reader does json.load(f)["keys"].
        payload = {
            "policy": name,
            "split": split,
            "n_input": len(rows),
            "n_kept": len(keys),
            "keys": sorted(keys),
            "per_emotion_counts": dict(_C(emo.get(k, "") for k in keys)),
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("%s | %s: %d / %d kept (%.1f%%) → %s",
                    split, name, len(keys), len(rows),
                    100 * len(keys) / max(1, len(rows)), p)
    return {"audio_clean": len(audio_clean), "text_clean": len(text_clean),
            "total": len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"],
                    choices=["train", "dev", "test"])
    ap.add_argument("--high_wer", type=float, default=HIGH_WER,
                    help=f"WER above this is flagged (default {HIGH_WER}).")
    ap.add_argument("--dup_json", type=Path,
                    default=Path("/dcs/large/u5734759/meld_duplicate_audio.json"))
    args = ap.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config["training"]["log_dir"], "analyse_wer_vad")
    wer_dir = Path(config["evaluation"]["output_dir"]) / "wer"

    dup_map = load_dup_map(args.dup_json)
    logger.info("Duplicate map: %s", args.dup_json if dup_map else "NOT FOUND")

    summaries = {}
    md: List[str] = ["# WER / VAD defect analysis", "",
                     f"Source: `{wer_dir}`  ", "Measurement only — nothing filtered.",
                     "", "Flags are additive: a clip can be short AND duplicated AND runaway.",
                     ""]

    for split in args.splits:
        src = wer_dir / f"{split}_wer_vad.csv"
        if not src.exists():
            logger.error("%s | missing %s", split, src)
            continue
        with open(src, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        dup = dup_map.get(split, {})
        for r in rows:
            r.update(flag_row(r, dup, args.high_wer))

        out = wer_dir / f"{split}_wer_vad_flagged.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        keep_dir = Path(config["filtering"]["keys_dir"])
        counts = write_keep_lists(rows, split, keep_dir, logger)

        s = summarise(rows)
        s["keep_lists"] = counts
        summaries[split] = s
        with open(wer_dir / f"{split}_defect_summary.json", "w", encoding="utf-8") as f:
            json.dump(s, f, indent=2)

        logger.info("%s | n=%d | clean %d (%.1f%%) | corpus WER all=%s clean=%s",
                    split, s["n_utterances"], s["clean_only"]["count"],
                    s["clean_only"]["pct"], s["all"]["corpus_wer"],
                    s["clean_only"]["corpus_wer"])

        md += [f"## {split} — {s['n_utterances']} utterances", "",
               "| class | count | % | corpus WER | median WER |",
               "|---|---:|---:|---:|---:|"]
        md.append(f"| ALL | {s['all']['n']} | 100.00 | {s['all']['corpus_wer']} "
                  f"| {s['all']['median_wer']} |")
        for c, v in s["per_class"].items():
            md.append(f"| {c[3:]} | {v['count']} | {v['pct']} | "
                      f"{v.get('corpus_wer')} | {v.get('median_wer')} |")
        md.append(f"| **clean (no flags)** | {s['clean_only']['count']} | "
                  f"{s['clean_only']['pct']} | **{s['clean_only']['corpus_wer']}** "
                  f"| {s['clean_only']['median_wer']} |")
        md += ["", "### WER threshold sensitivity", "",
               "How many utterances a WER gate removes, and how many of those the",
               "audio-based flags (dup / short / low_vad / runaway) already catch.",
               "A gate that removes only clips the audio flags catch adds nothing.",
               "",
               "| WER > | removed | % | also audio-flagged | **only** WER | kept | kept corpus WER |",
               "|---:|---:|---:|---:|---:|---:|---:|"]
        for th, v in s["wer_threshold_sweep"].items():
            md.append(f"| {th} | {v['removed']} | {v['removed_pct']} | "
                      f"{v['already_flagged_by_audio']} | {v['uniquely_removed_by_wer']} | "
                      f"{v['kept']} | {v['kept_corpus_wer']} |")
        md += ["", "### Flag combinations", "", "| combination | count |", "|---|---:|"]
        for k, v in s["combinations"].items():
            md.append(f"| {k} | {v} |")
        md += ["", "### Defect rate by emotion", "",
               "| emotion | n | defective | % |", "|---|---:|---:|---:|"]
        for e, v in s["by_emotion"].items():
            md.append(f"| {e} | {v['n']} | {v['defective']} | {v['pct_defective']} |")
        md.append("")

    (wer_dir / "defect_report.md").write_text("\n".join(md), encoding="utf-8")
    with open(wer_dir / "defect_summary_all.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    logger.info("Report → %s", wer_dir / "defect_report.md")


if __name__ == "__main__":
    main()
