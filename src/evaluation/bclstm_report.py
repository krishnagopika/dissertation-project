"""Collect every bc-LSTM run into one readable report.

Walks ``checkpoints/bclstm/<condition>/<k>/TRAINING_COMPLETE.json`` and writes
``results_new/bclstm/summary.{csv,md,json}``.

The markdown includes the ablation as a matrix -- conditions down, context
width across -- because that is the shape the result is actually read in:

                K=0     K=1     K=2     K=4    full
    gold          .       .       .       .      .
    asr           .       .       .       .      .
    asr_cleaned   .       .       .       .      .

Reading the matrix
------------------
Across a row is the value of dialogue context at a fixed text quality. Down a
column is the cost of ASR error at a fixed context width. The interesting
question is whether they interact: if context helps *more* when the text is
noisier, the row deltas grow as you go down, and context is partly compensating
for transcription error rather than adding independent signal.

K=0 keeps the BiLSTM's own parameters over a length-1 sequence, so it controls
for context WIDTH, not for having a recurrent model at all. The K=0 column is
not a "no model" baseline.

Integrity checks, not just formatting
-------------------------------------
Fails loudly on the two conditions that invalidate a comparison: a run that
certified completion without writing a checkpoint, and runs that disagree on
``dev_key_hash`` (they were early-stopped against different dev sets).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

CKPT_ROOT = Path("/dcs/large/u5734759/checkpoints/bclstm")
OUT_DIR = Path("results_new/bclstm")

CONDITIONS = ["gold", "asr", "asr_cleaned"]
WIDTHS = ["k0", "k1", "k2", "k4", "full"]

COLUMNS = [
    "condition", "context", "best_dev_weighted_f1", "test_emotion_weighted_f1",
    "best_epoch", "epochs_run", "early_stopped", "checkpoint_written",
    "filter_enabled", "seed", "dev_key_hash", "trainable_parameters",
]


def collect() -> List[Dict]:
    """Read every completion marker under the bc-LSTM checkpoint tree."""
    rows: List[Dict] = []
    for marker in sorted(CKPT_ROOT.glob("*/*/TRAINING_COMPLETE.json")):
        with open(marker, encoding="utf-8") as f:
            d = json.load(f)
        d["condition"] = marker.parent.parent.name
        d["context"] = marker.parent.name
        rows.append(d)
    return rows


def _cell(rows: List[Dict], cond: str, width: str, field: str) -> str:
    for r in rows:
        if r["condition"] == cond and r["context"] == width:
            v = r.get(field)
            return f"{v:.4f}" if isinstance(v, float) else (str(v) if v is not None else "-")
    return "·"


def write_markdown(rows: List[Dict], path: Path, problems: List[str]) -> None:
    lines = ["# bc-LSTM context ablation", ""]
    if problems:
        lines += ["> **Integrity problems detected — see the end of this file.**", ""]
    lines += [f"{len(rows)} of {len(CONDITIONS) * len(WIDTHS)} cells complete.", ""]

    for field, title in (("best_dev_weighted_f1", "Dev weighted F1 (emotion)"),
                         ("test_emotion_weighted_f1", "Test weighted F1 (emotion)")):
        lines += [f"## {title}", "",
                  "| condition | " + " | ".join(WIDTHS) + " |",
                  "|---" * (len(WIDTHS) + 1) + "|"]
        for cond in CONDITIONS:
            cells = [_cell(rows, cond, w, field) for w in WIDTHS]
            lines.append(f"| `{cond}` | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## Per-run detail", "",
              "| condition | context | dev WF1 | test WF1 | best ep | epochs | early stop | params |",
              "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x["condition"],
                                         WIDTHS.index(x["context"])
                                         if x["context"] in WIDTHS else 99)):
        dev = r.get("best_dev_weighted_f1")
        tst = r.get("test_emotion_weighted_f1")
        dev_s = f"{dev:.4f}" if isinstance(dev, float) else "-"
        tst_s = f"{tst:.4f}" if isinstance(tst, float) else "-"
        lines.append(
            f"| `{r['condition']}` | `{r['context']}` | {dev_s} | {tst_s} "
            f"| {r.get('best_epoch')} | {r.get('epochs_run')} "
            f"| {'yes' if r.get('early_stopped') else 'no'} "
            f"| {r.get('trainable_parameters', '-')} |"
        )
    lines.append("")

    hashes = {r.get("dev_key_hash") for r in rows if r.get("dev_key_hash")}
    verdict = ("CONSISTENT — all runs scored on the same dev set"
               if len(hashes) <= 1 else
               "DIFFER — these runs are NOT comparable to each other")
    lines += ["## Comparability", "",
              f"- dev key-set hashes: `{sorted(hashes)}`", f"- {verdict}",
              "- note: `asr_cleaned` filters TRAIN labels only; dev and test "
              "keep every utterance, so all cells score the same dev set", ""]
    if problems:
        lines += ["## Integrity problems", ""] + [f"- {p}" for p in problems] + [""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect()
    if not rows:
        print(f"No completed runs under {CKPT_ROOT}")
        return

    problems: List[str] = []
    for r in rows:
        if not r.get("checkpoint_written", True):
            problems.append(
                f"`{r['condition']}/{r['context']}` certified completion but "
                "never wrote a checkpoint.")
    hashes = {r.get("dev_key_hash") for r in rows if r.get("dev_key_hash")}
    if len(hashes) > 1:
        problems.append(
            f"runs disagree on dev key-set hash ({sorted(hashes)}) — they were "
            "scored against different dev sets and cannot be compared.")

    with open(OUT_DIR / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["condition"], x["context"])):
            w.writerow(r)
    write_markdown(rows, OUT_DIR / "summary.md", problems)
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)

    print(f"  {len(rows)}/{len(CONDITIONS) * len(WIDTHS)} cells "
          f"-> {OUT_DIR}/summary.{{csv,md,json}}")
    for cond in CONDITIONS:
        cells = [_cell(rows, cond, w, "best_dev_weighted_f1") for w in WIDTHS]
        print(f"    {cond:<12s} " + "  ".join(f"{w}:{c}" for w, c in zip(WIDTHS, cells)))
    if problems:
        print("\n  INTEGRITY PROBLEMS:")
        for p in problems:
            print(f"    - {p}")


if __name__ == "__main__":
    main()
