#!/usr/bin/env python3.12
"""Run the Voxtral pipeline over sampled UK/Irish accent clips.

Produces one CSV that supports two different evaluations:

  OBJECTIVE (no annotation needed) -- per-accent WER. english_dialects ships
  gold transcripts, so ASR degradation across accents is measurable directly.
  This is the stronger half of the dialect-robustness result.

  SUBJECTIVE (needs annotation) -- per-accent emotion accuracy. The corpus has
  NO emotion labels, so the CSV leaves `human_emotion` and `pred_correct`
  blank for manual listening. Because rows are stratified by PREDICTED emotion,
  the annotated result yields PRECISION per emotion per accent, not recall:
  we cannot know what the model failed to detect.

Caveat worth carrying into the write-up: this is READ speech (people reciting
prompts), so most clips are genuinely neutral. A non-neutral prediction is
therefore usually an error, and the useful signal is whether the model
over-predicts non-neutral more often on unfamiliar accents.

Model note: uses Voxtral MINI deliberately. The MELD baselines were produced
with Mini, so using Small here would confound "accent shift" with "different
model". bc-LSTM cannot be run at all -- it needs dialogue context, and these
are isolated sentences.

Usage:
    python3.12 src/evaluation/dialect_probe.py --config src/configs/mini.yaml
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jiwer

from src.evaluation.compute_wer import NORMALISE
from src.evaluation.metrics import EMOTION_NAMES, SENTIMENT_NAMES
from src.evaluation.voxtral_zeroshot import (
    build_classification_prompt,
    build_messages,
    parse_prediction,
)
from src.preprocessing.transcribe_all import audio_to_wav_base64

TRANSCRIBE_INSTRUCTION = (
    "Output only the verbatim spoken words from this audio. "
    "Plain text only. No timestamps, no speaker labels, no formatting. "
    "If unclear, output your best guess. Do not say you did not understand."
)


def build_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    return logging.getLogger("dialect_probe")


def safe_wer(reference: str, hypothesis: str) -> float | str:
    """WER with the same normalisation compute_wer.py uses on MELD.

    Returns "" when the reference normalises to nothing, so an empty gold line
    cannot silently contribute a 0.0 (or a divide-by-zero) to per-accent means.
    """
    try:
        ref_words = NORMALISE(reference)
        if not ref_words or not ref_words[0]:
            return ""
        # kwarg is reference_transform, NOT truth_transform -- the latter raises
        # TypeError, which the except below swallowed, blanking every single row.
        return round(jiwer.wer(reference, hypothesis,
                               reference_transform=NORMALISE,
                               hypothesis_transform=NORMALISE), 4)
    except Exception:                                    # noqa: BLE001
        return ""


def run_batched(llm, sampling_params, messages: List[list], keys: List[str],
                batch_size: int, label: str, logger: logging.Logger) -> Dict[str, str]:
    """Run llm.chat over messages in batches, returning {key: generated_text}."""
    out: Dict[str, str] = {}
    for start in range(0, len(messages), batch_size):
        chunk_msgs = messages[start:start + batch_size]
        chunk_keys = keys[start:start + batch_size]
        try:
            outputs = llm.chat(chunk_msgs, sampling_params=sampling_params)
            for k, o in zip(chunk_keys, outputs):
                out[k] = o.outputs[0].text.strip()
        except Exception as exc:                          # noqa: BLE001
            logger.error("%s | batch at %d failed: %s", label, start, exc)
            for k in chunk_keys:
                out[k] = ""
        logger.info("  %s | %d / %d done", label,
                    min(start + batch_size, len(messages)), len(messages))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, default="src/configs/mini.yaml")
    parser.add_argument("--probe_dir", type=Path,
                        default=Path("/dcs/large/u5734759/data/dialect_probe"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    logger = build_logger()

    from src.utils import load_config
    config = load_config(args.config)
    model_id = config["model"]["voxtral_id"]
    tp = int(config["model"].get("tensor_parallel_size", 1))

    manifest_path = args.probe_dir / "manifest.csv"
    if not manifest_path.exists():
        logger.error("Manifest not found: %s — run sample_dialects.py first.",
                     manifest_path)
        sys.exit(1)

    with open(manifest_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if args.max_samples:
        rows = rows[: args.max_samples]
    logger.info("Loaded %d clips from %s", len(rows), manifest_path)

    # ---- Encode audio once, reuse for both prompts ----
    encoded: Dict[str, str] = {}
    for r in rows:
        p = Path(r["audio_path"])
        if not p.exists():
            logger.warning("%s | missing audio, skipping", r["key"])
            continue
        try:
            encoded[r["key"]] = audio_to_wav_base64(p)
        except Exception as exc:                          # noqa: BLE001
            logger.warning("%s | encode failed: %s", r["key"], exc)

    rows = [r for r in rows if r["key"] in encoded]
    logger.info("%d clips encoded successfully", len(rows))
    if not rows:
        logger.error("No usable clips — aborting.")
        sys.exit(1)

    from vllm import LLM, SamplingParams
    logger.info("Loading %s (tp=%d)", model_id, tp)
    llm = LLM(
        model=model_id,
        tokenizer_mode="mistral",
        max_model_len=8192,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        tensor_parallel_size=tp,
        enforce_eager=True,
    )

    keys = [r["key"] for r in rows]

    logger.info("=== Pass 1: ASR transcription ===")
    asr = run_batched(
        llm, SamplingParams(max_tokens=256, temperature=0.0),
        [build_messages(encoded[k], TRANSCRIBE_INSTRUCTION) for k in keys],
        keys, args.batch_size, "ASR", logger,
    )

    logger.info("=== Pass 2: zero-shot emotion/sentiment ===")
    instruction = build_classification_prompt()
    cls = run_batched(
        llm, SamplingParams(max_tokens=32, temperature=0.0),
        [build_messages(encoded[k], instruction) for k in keys],
        keys, args.batch_size, "CLS", logger,
    )

    # ---- Assemble annotation sheet ----
    out_rows = []
    for r in rows:
        k = r["key"]
        hyp = asr.get(k, "")
        e_idx, s_idx = parse_prediction(cls.get(k, ""))
        out_rows.append({
            "key": k,
            "accent": r["accent"],
            "gender": r["gender"],
            "speaker_id": r["speaker_id"],
            "duration_s": r["duration_s"],
            "gold_text": r["gold_text"],
            "asr_text": hyp,
            "wer": safe_wer(r["gold_text"], hyp),
            "pred_emotion": EMOTION_NAMES[e_idx],
            "pred_sentiment": SENTIMENT_NAMES[s_idx],
            "raw_model_output": cls.get(k, "").replace("\n", " | "),
            # --- blank for manual annotation ---
            "human_emotion": "",
            "pred_correct": "",
            "notes": "",
        })

    # Group by accent then predicted emotion so the sheet is easy to work
    # through and every (accent, predicted emotion) cell is visible.
    out_rows.sort(key=lambda x: (x["accent"], x["pred_emotion"], x["key"]))

    out_path = args.probe_dir / "dialect_annotation_sheet.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    # ---- Objective summary: WER by accent (needs no annotation) ----
    from collections import defaultdict
    by_accent = defaultdict(list)
    for r in out_rows:
        if r["wer"] != "":
            by_accent[r["accent"]].append(float(r["wer"]))

    logger.info("=" * 58)
    logger.info("WER by accent (gold transcripts, no annotation needed)")
    for acc in sorted(by_accent, key=lambda a: sum(by_accent[a]) / len(by_accent[a])):
        vals = by_accent[acc]
        logger.info("  %-10s n=%-3d mean WER %.3f", acc, len(vals), sum(vals) / len(vals))

    pred_counts = defaultdict(int)
    for r in out_rows:
        pred_counts[r["pred_emotion"]] += 1
    logger.info("Predicted-emotion distribution: %s", dict(pred_counts))
    logger.info("Annotation sheet → %s", out_path)


if __name__ == "__main__":
    main()
