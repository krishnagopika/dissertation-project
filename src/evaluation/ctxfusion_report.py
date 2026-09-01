"""Collect ContextThenFusion runs into one readable report.

Walks ``checkpoints/ctxfusion/<condition>/<arm>/TRAINING_COMPLETE.json`` and
writes ``results_new/ctxfusion/summary.{csv,md,json}``.

The 2x2 it reports
------------------
                    both   acoustic_only   text_only   neither
    gold             .           .             .          .
    asr              .           .             .          .
    asr_cleaned      .           .             .          .

`neither` is the control: with both BiLSTMs off the model reduces to
per-utterance fusion, so every other column's gain over it is attributable to
context alone.

How to read it
--------------
**text_only - neither** is the value of contextualising the lexical channel.
**acoustic_only - neither** is the value of contextualising prosody.
**both - (the better single)** is whether the two are additive.

Existing results predict text_only ~ both and acoustic_only ~ neither: the
acoustic cache is byte-identical across the three conditions, yet dialogue
context only helped when the text was clean. If that holds here, the whole
context story is lexical.

**Parameter counts are NOT matched across arms.** acoustic_lstm takes a 1280-d
input against text_lstm's 768-d, so both-on is 6.31M, acoustic_only 4.34M,
text_only 3.55M and neither 1.58M. A weaker arm may be weaker on capacity
rather than architecture, so the report prints parameters beside every score
and any claim about an arm losing must survive that column.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

CKPT_ROOT = Path("/dcs/large/u5734759/checkpoints/ctxfusion")
OUT_DIR = Path("results_new/ctxfusion")

CONDITIONS = ["gold", "asr", "asr_cleaned"]
ARMS = ["both", "acoustic_only", "text_only", "neither"]

COLUMNS = [
    "condition", "arm", "best_dev_weighted_f1", "best_epoch", "epochs_run",
    "early_stopped", "use_acoustic_lstm", "use_text_lstm", "fusion",
    "lstm_hidden", "lr", "batch_size", "seed", "dev_key_hash",
    "filter_enabled", "total_params",
]


def collect() -> List[Dict]:
    rows: List[Dict] = []
    for m in sorted(CKPT_ROOT.glob("*/*/TRAINING_COMPLETE.json")):
        with open(m, encoding="utf-8") as f:
            d = json.load(f)
        d["condition"] = m.parent.parent.name
        d["arm"] = m.parent.name
        d["total_params"] = (d.get("parameters") or {}).get("TOTAL")
        rows.append(d)
    return rows


def _get(rows, cond, arm, field):
    for r in rows:
        if r["condition"] == cond and r["arm"] == arm:
            return r.get(field)
    return None


def _fmt(v):
    if v is None:
        return "·"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def write_markdown(rows: List[Dict], path: Path, problems: List[str]) -> None:
    lines = ["# Context-then-fusion: 2x2 context ablation", ""]
    if problems:
        lines += ["> **Integrity problems — see the end.**", ""]
    lines += [f"{len(rows)} of {len(CONDITIONS) * len(ARMS)} cells complete.", "",
              "## Dev weighted F1 (emotion)", "",
              "| condition | " + " | ".join(ARMS) + " |",
              "|---" * (len(ARMS) + 1) + "|"]
    for c in CONDITIONS:
        cells = [_fmt(_get(rows, c, a, "best_dev_weighted_f1")) for a in ARMS]
        lines.append(f"| `{c}` | " + " | ".join(cells) + " |")
    lines.append("")

    # The deltas that answer the question.
    lines += ["## Value of context (arm − neither)", "",
              "| condition | acoustic_only | text_only | both |",
              "|---|---|---|---|"]
    for c in CONDITIONS:
        base = _get(rows, c, "neither", "best_dev_weighted_f1")
        cells = []
        for a in ("acoustic_only", "text_only", "both"):
            v = _get(rows, c, a, "best_dev_weighted_f1")
            cells.append(f"{v - base:+.4f}" if (v is not None and base is not None) else "·")
        lines.append(f"| `{c}` | " + " | ".join(cells) + " |")
    lines += ["", "`neither` = per-utterance fusion, no context. Deltas are the "
              "value of contextualising each channel.", ""]

    lines += ["## Parameters (arms are NOT matched)", "",
              "| condition | " + " | ".join(ARMS) + " |",
              "|---" * (len(ARMS) + 1) + "|"]
    for c in CONDITIONS:
        cells = []
        for a in ARMS:
            v = _get(rows, c, a, "total_params")
            cells.append(f"{v:,}" if v else "·")
        lines.append(f"| `{c}` | " + " | ".join(cells) + " |")
    lines.append("")

    hashes = {r.get("dev_key_hash") for r in rows if r.get("dev_key_hash")}
    lines += ["## Comparability", "",
              f"- dev key-set hashes: `{sorted(hashes)}`",
              f"- {'CONSISTENT' if len(hashes) <= 1 else 'DIFFER — not comparable'}",
              ""]
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
            problems.append(f"`{r['condition']}/{r['arm']}` wrote no checkpoint.")
    hashes = {r.get("dev_key_hash") for r in rows if r.get("dev_key_hash")}
    if len(hashes) > 1:
        problems.append(f"dev key hashes differ ({sorted(hashes)}) — not comparable.")

    with open(OUT_DIR / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["condition"], x["arm"])):
            w.writerow(r)
    write_markdown(rows, OUT_DIR / "summary.md", problems)
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)

    print(f"  {len(rows)}/{len(CONDITIONS) * len(ARMS)} cells "
          f"-> {OUT_DIR}/summary.{{csv,md,json}}")
    for c in CONDITIONS:
        cells = [f"{a}:{_fmt(_get(rows, c, a, 'best_dev_weighted_f1'))}" for a in ARMS]
        print(f"    {c:<12s} " + "  ".join(cells))
    if problems:
        print("\n  INTEGRITY PROBLEMS:")
        for p in problems:
            print(f"    - {p}")


if __name__ == "__main__":
    main()
