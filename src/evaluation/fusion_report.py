"""Collect every fusion run into one readable report.

Walks ``checkpoints/fusion/<phase>/<run>/TRAINING_COMPLETE.json`` and writes
three artefacts to ``results_new/fusion/``:

    summary.csv      one row per run, machine-readable, every recorded field
    summary.md       the same grouped by phase, for reading and for the writeup
    summary.json     the raw records, so nothing recorded is ever discarded

Why this exists
---------------
Job 10043's Phase 0 wrote all six learning-rate/batch configurations into a
SINGLE directory, because the grid built a run name in the shell while the
trainer derived its own from modality/pooling/fusion/loss and ignored it. Five
results were silently overwritten and the sixth was reported as though it were
the winner. Runs are now separated by ``--tag`` into a phase tree, and this
script is what makes the resulting tree legible without opening 25 JSON files.

Integrity checks, not just formatting
-------------------------------------
The report FAILS LOUDLY on the two conditions that make a comparison invalid:

* a run whose ``checkpoint_written`` is false -- it certified completion but
  never saved weights, so its metric refers to nothing on disk;
* runs that disagree on ``dev_key_hash`` -- they were early-stopped and scored
  against different dev sets, so their weighted F1 values are not comparable.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

CKPT_ROOT = Path("/dcs/large/u5734759/checkpoints/fusion")
OUT_DIR = Path("results_new/fusion")

#: Column order for the CSV. Anything present in the marker but absent here is
#: still written to summary.json, so adding a field to the trainer never
#: silently drops it from the record.
COLUMNS = [
    "phase", "run", "modality", "pooling", "fusion", "loss",
    "best_dev_weighted_f1", "best_epoch", "epochs_run", "early_stopped",
    "checkpoint_written", "lr", "batch_size", "train_size", "dev_size",
    "seed", "dev_key_hash", "max_frames", "lambda_sentiment", "total_params",
]

PHASE_TITLES = {
    "phase0_lr_batch": "Phase 0 — learning rate x batch size",
    "phase1_pooling": "Phase 1 — acoustic-only, pooling ablation",
    "phase2_textonly": "Phase 2 — text-only baselines",
    "phase3_fusion": "Phase 3 — fusion mechanisms x text conditions",
}


def collect() -> List[Dict]:
    """Read every completion marker under the fusion checkpoint tree."""
    rows: List[Dict] = []
    for marker in sorted(CKPT_ROOT.glob("*/*/TRAINING_COMPLETE.json")):
        with open(marker, encoding="utf-8") as f:
            d = json.load(f)
        d["phase"] = marker.parent.parent.name
        d["run"] = marker.parent.name
        d["total_params"] = (d.get("parameters") or {}).get("TOTAL")
        rows.append(d)
    return rows


def write_csv(rows: List[Dict], path: Path) -> None:
    """One row per run, every column in COLUMNS."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["phase"], -x["best_dev_weighted_f1"])):
            w.writerow(r)


def write_markdown(rows: List[Dict], path: Path, problems: List[str]) -> None:
    """Grouped by phase, sorted by dev weighted F1 within each."""
    by_phase: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_phase[r["phase"]].append(r)

    lines = ["# Fusion grid results", ""]
    if problems:
        lines += ["> **Integrity problems detected — see the end of this file.**", ""]
    lines += [f"{len(rows)} completed runs.", ""]

    for phase in sorted(by_phase):
        lines += [f"## {PHASE_TITLES.get(phase, phase)}", ""]
        lines += ["| run | dev WF1 | best ep | epochs | early stop | lr | bs | train n | params |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(by_phase[phase], key=lambda x: -x["best_dev_weighted_f1"]):
            params = r.get("total_params")
            lines.append(
                f"| `{r['run']}` "
                f"| **{r['best_dev_weighted_f1']:.4f}** "
                f"| {r.get('best_epoch')} "
                f"| {r.get('epochs_run')} "
                f"| {'yes' if r.get('early_stopped') else 'no'} "
                f"| {r.get('lr')} "
                f"| {r.get('batch_size')} "
                f"| {r.get('train_size')} "
                f"| {format(params, ',') if params else '-'} |"
            )
        lines.append("")

    hashes = {r.get("dev_key_hash") for r in rows if r.get("dev_key_hash")}
    verdict = ("CONSISTENT — all runs scored on the same dev set"
               if len(hashes) <= 1 else
               "DIFFER — these runs are NOT comparable to each other")
    lines += ["## Comparability", "",
              f"- dev key-set hashes: `{sorted(hashes)}`",
              f"- {verdict}", ""]
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
                f"`{r['phase']}/{r['run']}` certified completion but never wrote "
                f"a checkpoint — its metric refers to no saved weights.")
    hashes = {r.get("dev_key_hash") for r in rows if r.get("dev_key_hash")}
    if len(hashes) > 1:
        problems.append(
            f"runs disagree on dev key-set hash ({sorted(hashes)}) — they were "
            "scored against different dev sets and cannot be compared.")

    write_csv(rows, OUT_DIR / "summary.csv")
    write_markdown(rows, OUT_DIR / "summary.md", problems)
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, sort_keys=True)

    print(f"  {len(rows)} runs -> {OUT_DIR}/summary.{{csv,md,json}}")
    for phase in sorted({r["phase"] for r in rows}):
        sub = [r for r in rows if r["phase"] == phase]
        best = max(sub, key=lambda x: x["best_dev_weighted_f1"])
        print(f"    {phase:<20s} {len(sub):2d} runs | best "
              f"{best['best_dev_weighted_f1']:.4f}  {best['run']}")
    if problems:
        print("\n  INTEGRITY PROBLEMS:")
        for p in problems:
            print(f"    - {p}")


if __name__ == "__main__":
    main()
