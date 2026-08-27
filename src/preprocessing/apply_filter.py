"""
apply_filter.py — Convert VAD + WER metadata into filter keep-lists
====================================================================
Reads the per-utterance metadata JSONs produced by
``compute_filter_metadata.py`` and produces one **keep-list** JSON per
policy defined in the config. Fast (no audio, no ASR — pure filtering).

Downstream training scripts read a keep-list and drop utterances outside it.

Policies are declared in ``filtering.policies`` in mini.yaml, for example:

  filtering:
    metadata_dir: /dcs/large/u5734759/data/meld_filter_metadata
    keys_dir:     /dcs/large/u5734759/data/meld_filter_keys
    policies:
      wer10:
        wer_source: vox            # "vox" or "whisper"
        wer_max: 0.10
        vad_min_speech_ratio: 0.20
      wer15:
        wer_max: 0.15
        vad_min_speech_ratio: 0.20
      wer20:
        wer_max: 0.20
        vad_min_speech_ratio: 0.20
      wer25:
        wer_max: 0.25
        vad_min_speech_ratio: 0.20
      wer25_train_lax:
        # "eval strict, train lax" — split-dependent thresholds
        per_split:
          train: {wer_max: 1.00, vad_min_speech_ratio: 0.20}
          dev:   {wer_max: 0.25, vad_min_speech_ratio: 0.20}
          test:  {wer_max: 0.25, vad_min_speech_ratio: 0.20}

For each policy P and each split S, this script writes THREE artefacts:

  1. {keys_dir}/{S}_filtered_keys_{P}.json         — machine-readable keep-list
       {"policy": ..., "split": ..., "n_input": ..., "n_kept": ...,
        "keys": ["dia0_utt0", ...], "per_emotion_counts": {...}}

  2. {keys_dir}/{S}_filter_summary_{P}.json        — human-readable stats
       Totals, kept vs excluded counts, mean/median WER + VAD for each group,
       per-emotion pass rates, exclusion-reason breakdown.

  3. {keys_dir}/{S}_filter_report_{P}.csv          — full per-utterance table
       key, dialogue_id, utterance_id, emotion, sentiment,
       gold_text, voxtral_asr, whisper_asr,
       vox_wer, whisper_wer, speech_ratio, duration_sec,
       kept, exclusion_reason

Usage
-----
  python3.12 src/preprocessing/apply_filter.py --config src/configs/mini.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils import load_config, setup_logging


_SPLIT_CSV: Dict[str, str] = {
    "train": "train_sent_emo.csv",
    "dev":   "dev_sent_emo.csv",
    "test":  "test_sent_emo.csv",
}


# ---------------------------------------------------------------------------
# Policy resolution
# ---------------------------------------------------------------------------

def resolve_policy_for_split(
    policy: Dict, split: str,
) -> Tuple[str, float, float]:
    """Resolve the concrete (wer_source, wer_max, vad_min) for a given split.

    Supports both flat policies (same thresholds for every split) and
    per-split policies via a ``per_split`` sub-dict.

    Args:
        policy: One entry from ``config["filtering"]["policies"]``.
        split: 'train' | 'dev' | 'test'.

    Returns:
        Tuple ``(wer_source, wer_max, vad_min_speech_ratio)``.
    """
    if "per_split" in policy and split in policy["per_split"]:
        sp = policy["per_split"][split]
        wer_source = sp.get("wer_source", policy.get("wer_source", "vox"))
        wer_max = float(sp.get("wer_max", policy.get("wer_max", 1.0)))
        vad_min = float(sp.get(
            "vad_min_speech_ratio",
            policy.get("vad_min_speech_ratio", 0.0),
        ))
    else:
        wer_source = policy.get("wer_source", "vox")
        wer_max = float(policy.get("wer_max", 1.0))
        vad_min = float(policy.get("vad_min_speech_ratio", 0.0))
    return wer_source, wer_max, vad_min


# ---------------------------------------------------------------------------
# Filter application
# ---------------------------------------------------------------------------

def apply_policy(
    records: List[Dict],
    wer_source: str,
    wer_max: float,
    vad_min_speech_ratio: float,
) -> List[Dict]:
    """Annotate every record with kept/excluded + exclusion reason.

    Args:
        records: Per-utterance metadata list from compute_filter_metadata.
        wer_source: 'vox' or 'whisper'.
        wer_max: Maximum allowed WER (inclusive).
        vad_min_speech_ratio: Minimum allowed speech ratio (inclusive).

    Returns:
        List of dicts extending each record with:
            - ``kept``: bool
            - ``exclusion_reason``: one of
                ""  (empty for kept)
                "no_wer"                       (WER not computed)
                "wer_above_threshold"
                "vad_below_threshold"
                "wer_and_vad_failed"
    """
    field = f"{wer_source}_wer"
    annotated: List[Dict] = []
    for r in records:
        w = r.get(field)
        vad = r.get("speech_ratio", 0.0)
        if w is None:
            reason = "no_wer"
            kept = False
        else:
            wer_ok = (w <= wer_max)
            vad_ok = (vad >= vad_min_speech_ratio)
            if wer_ok and vad_ok:
                reason = ""
                kept = True
            elif not wer_ok and not vad_ok:
                reason = "wer_and_vad_failed"
                kept = False
            elif not wer_ok:
                reason = "wer_above_threshold"
                kept = False
            else:
                reason = "vad_below_threshold"
                kept = False
        annotated.append({
            **r,
            "kept": kept,
            "exclusion_reason": reason,
        })
    return annotated


def load_split_context(
    meld_root: Path,
    transcripts_dir: Path,
    split: str,
) -> Tuple[
    Dict[str, Dict[str, str]],
    Dict[str, str],
    Dict[str, str],
]:
    """Load MELD CSV + cached ASR transcripts, keyed by ``dia{d}_utt{u}``.

    Returns:
        (row_by_key, vox_by_key, whisper_by_key)
        row_by_key values contain: dialogue_id, utterance_id, gold_text,
        emotion, sentiment (all strings).
    """
    csv_path = meld_root / _SPLIT_CSV[split]
    row_by_key: Dict[str, Dict[str, str]] = {}
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df.columns = (
            df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
        )
        for _, row in df.iterrows():
            key = (
                f"dia{int(row['dialogue_id'])}"
                f"_utt{int(row['utterance_id'])}"
            )
            row_by_key[key] = {
                "dialogue_id":  str(int(row["dialogue_id"])),
                "utterance_id": str(int(row["utterance_id"])),
                "gold_text":    str(row.get("utterance", "")).strip(),
                "emotion":      str(row.get("emotion", "")).strip().lower(),
                "sentiment":    str(row.get("sentiment", "")).strip().lower(),
            }

    def _load_json(p: Path) -> Dict[str, str]:
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    vox = _load_json(transcripts_dir / f"{split}_transcripts.json")
    wsp = _load_json(transcripts_dir / f"{split}_transcripts_whisper.json")
    return row_by_key, vox, wsp


def summarise(
    annotated: List[Dict],
    row_by_key: Dict[str, Dict[str, str]],
) -> Dict:
    """Build a human-readable summary of the filter outcome.

    Args:
        annotated: Output of ``apply_policy`` (records with kept + reason).
        row_by_key: MELD CSV lookup (from load_split_context).

    Returns:
        Dict with totals, group means, per-emotion pass rates, exclusion
        reason counts.
    """
    n_total = len(annotated)
    n_kept = sum(1 for r in annotated if r["kept"])
    n_excluded = n_total - n_kept

    def _stats(records: List[Dict]) -> Dict:
        if not records:
            return {"count": 0}
        vox = [r["vox_wer"] for r in records if r.get("vox_wer") is not None]
        vad = [r["speech_ratio"] for r in records]
        dur = [r["duration_sec"] for r in records]
        def _stat(xs: List[float]) -> Dict:
            if not xs:
                return {}
            arr = sorted(xs)
            n = len(arr)
            return {
                "mean":   sum(arr) / n,
                "median": arr[n // 2],
                "min":    arr[0],
                "max":    arr[-1],
            }
        return {
            "count":        len(records),
            "vox_wer":      _stat(vox),
            "speech_ratio": _stat(vad),
            "duration_sec": _stat(dur),
        }

    kept_recs = [r for r in annotated if r["kept"]]
    excl_recs = [r for r in annotated if not r["kept"]]

    # Per-emotion pass counts + rates
    per_emotion_total: Counter = Counter()
    per_emotion_kept: Counter = Counter()
    for r in annotated:
        meta = row_by_key.get(r["key"])
        if meta is None:
            continue
        emo = meta.get("emotion", "")
        if not emo:
            continue
        per_emotion_total[emo] += 1
        if r["kept"]:
            per_emotion_kept[emo] += 1
    per_emotion = {
        emo: {
            "total": per_emotion_total[emo],
            "kept":  per_emotion_kept[emo],
            "kept_pct": (
                100.0 * per_emotion_kept[emo] / per_emotion_total[emo]
                if per_emotion_total[emo] else 0.0
            ),
        }
        for emo in sorted(per_emotion_total)
    }

    reason_counts = Counter(
        r["exclusion_reason"] for r in annotated if not r["kept"]
    )

    return {
        "n_total":     n_total,
        "n_kept":      n_kept,
        "n_excluded":  n_excluded,
        "kept_pct":    100.0 * n_kept / max(n_total, 1),
        "kept_stats":     _stats(kept_recs),
        "excluded_stats": _stats(excl_recs),
        "per_emotion":    per_emotion,
        "exclusion_reasons": dict(reason_counts),
    }


def write_report_csv(
    annotated: List[Dict],
    row_by_key: Dict[str, Dict[str, str]],
    vox_by_key: Dict[str, str],
    whisper_by_key: Dict[str, str],
    out_path: Path,
) -> None:
    """Emit a per-utterance CSV combining metadata + gold + ASR + kept/reason."""
    rows: List[Dict] = []
    for r in annotated:
        key = r["key"]
        meta = row_by_key.get(key, {})
        rows.append({
            "key":              key,
            "dialogue_id":      meta.get("dialogue_id", ""),
            "utterance_id":     meta.get("utterance_id", ""),
            "emotion":          meta.get("emotion", ""),
            "sentiment":        meta.get("sentiment", ""),
            "gold_text":        meta.get("gold_text", ""),
            "voxtral_asr":      vox_by_key.get(key, ""),
            "whisper_asr":      whisper_by_key.get(key, ""),
            "gold_words":       r.get("gold_words"),
            "vox_words":        r.get("vox_words"),
            "whisper_words":    r.get("whisper_words"),
            "vox_wer":          r.get("vox_wer"),
            "whisper_wer":      r.get("whisper_wer"),
            "speech_ratio":     r.get("speech_ratio"),
            "duration_sec":     r.get("duration_sec"),
            "kept":             r.get("kept"),
            "exclusion_reason": r.get("exclusion_reason", ""),
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply threshold policies to compute_filter_metadata output and "
            "produce per-policy, per-split keep-lists."
        )
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "test"],
        choices=["train", "dev", "test"],
    )
    args = parser.parse_args()

    config = load_config(args.config)
    filt = config.get("filtering")
    if filt is None:
        raise KeyError("Config missing 'filtering' block.")

    policies: Dict[str, Dict] = filt.get("policies", {})
    if not policies:
        raise ValueError(
            "filtering.policies is empty — declare at least one policy "
            "in mini.yaml."
        )

    log_dir = config["training"]["log_dir"]
    logger = setup_logging(log_dir, "apply_filter")
    logger.info("Config: %s | policies: %s", args.config, list(policies))

    metadata_dir = Path(filt["metadata_dir"])
    keys_dir = Path(filt["keys_dir"])
    keys_dir.mkdir(parents=True, exist_ok=True)
    meld_root = Path(config["data"]["meld_root"])
    transcripts_dir = Path(config["data"]["transcripts_path"])

    # Load metadata + CSV + transcripts once per split
    metadata: Dict[str, List[Dict]] = {}
    context: Dict[str, Tuple[Dict, Dict, Dict]] = {}
    for split in args.splits:
        meta_path = metadata_dir / f"{split}_filter_metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Missing metadata: {meta_path}. "
                "Run preprocessing/compute_filter_metadata.py first."
            )
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata[split] = json.load(f)
        logger.info("Loaded %d records for %s", len(metadata[split]), split)
        context[split] = load_split_context(meld_root, transcripts_dir, split)

    # For each policy × split, write keep-list + summary + per-utterance CSV
    for policy_name, policy in policies.items():
        logger.info("\n=== Policy: %s ===", policy_name)
        for split in args.splits:
            source, wer_max, vad_min = resolve_policy_for_split(policy, split)
            row_by_key, vox_by_key, whisper_by_key = context[split]

            annotated = apply_policy(
                metadata[split],
                wer_source=source,
                wer_max=wer_max,
                vad_min_speech_ratio=vad_min,
            )
            kept_keys = [r["key"] for r in annotated if r["kept"]]
            summary = summarise(annotated, row_by_key)

            policy_desc = {
                "name": policy_name,
                "wer_source": source,
                "wer_max": wer_max,
                "vad_min_speech_ratio": vad_min,
            }

            # 1. Machine-readable keep-list
            keys_out = {
                "policy": policy_desc,
                "split":  split,
                "n_input": summary["n_total"],
                "n_kept":  summary["n_kept"],
                "keys":    kept_keys,
                "per_emotion_counts": {
                    emo: v["kept"] for emo, v in summary["per_emotion"].items()
                },
            }
            keys_path = keys_dir / f"{split}_filtered_keys_{policy_name}.json"
            with open(keys_path, "w", encoding="utf-8") as f:
                json.dump(keys_out, f, indent=2)

            # 2. Human-readable summary
            summary_out = {"policy": policy_desc, "split": split, **summary}
            summary_path = keys_dir / (
                f"{split}_filter_summary_{policy_name}.json"
            )
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_out, f, indent=2)

            # 3. Full per-utterance CSV (kept + excluded rows, gold + ASR + reason)
            csv_path = keys_dir / f"{split}_filter_report_{policy_name}.csv"
            write_report_csv(
                annotated, row_by_key, vox_by_key, whisper_by_key, csv_path,
            )

            logger.info(
                "  %s | source=%s wer_max=%.2f vad_min=%.2f → "
                "kept %d / %d (%.1f%%) | excluded reasons: %s",
                split, source, wer_max, vad_min,
                summary["n_kept"], summary["n_total"], summary["kept_pct"],
                summary["exclusion_reasons"],
            )
            logger.info("    keep-list  → %s", keys_path)
            logger.info("    summary    → %s", summary_path)
            logger.info("    report CSV → %s", csv_path)
            logger.info(
                "    per-emotion pass rates: %s",
                {
                    emo: f"{v['kept']}/{v['total']} ({v['kept_pct']:.0f}%)"
                    for emo, v in summary["per_emotion"].items()
                },
            )


if __name__ == "__main__":
    main()
