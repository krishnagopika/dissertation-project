"""Validate a re-scoring pass against the figures written at training time.

bc-LSTM is the only family whose test metrics were computed by the training
script itself (``results_new/bclstm/test_results_*.json``). Those are the
reference. ``fusion`` and ``ctxfusion`` carry ``best_dev_weighted_f1`` only, so
a re-scoring pass is the *sole* source of test numbers for them -- which means
the pass has to be shown correct somewhere it can be checked before its
uncheckable output is believed.

Run after evaluate_all:

    python3 src/evaluation/check_windowed_eval.py [--table test_all_v6_windowed.json]

Exit status is 0 only if every bc-LSTM cell matches to within --tol.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, Tuple

RESULTS = Path("results_new")
TOL = 1e-3


def reference() -> Dict[str, float]:
    """Test weighted F1 per bc-LSTM run, as written at training time."""
    ref: Dict[str, float] = {}
    for f in glob.glob(str(RESULTS / "bclstm" / "test_results_*.json")):
        with open(f, encoding="utf-8") as fh:
            d = json.load(fh)
        tag = d.get("tag")
        if tag:
            ref[f"bclstm/{tag}"] = d["emotion"]["weighted_f1"]
    return ref


def scored(table: str) -> Dict[str, float]:
    with open(RESULTS / "test_all" / table, encoding="utf-8") as fh:
        rows = json.load(fh)
    return {r["run"]: r["test_emotion_weighted_f1"] for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="test_all_v6_windowed.json")
    ap.add_argument("--tol", type=float, default=TOL)
    args = ap.parse_args()

    ref, got = reference(), scored(args.table)
    common = sorted(set(ref) & set(got))
    if not common:
        print("no overlapping bc-LSTM runs -- nothing to validate against")
        return 1

    # asr_cleaned is scored on a different test set in the two sources and is
    # therefore not comparable. The training script applies the defect filter to
    # test as well as train (train_context.py builds test_ds via make_ds, which
    # passes filtered_keys_path), so its reference figures are on the 1,787
    # utterances that survive the filter. evaluate_all deliberately never
    # filters test and scores all 2,610. Both are intended; only the comparison
    # between them is meaningless, so these cells are reported and not failed.
    bad: list[Tuple[str, float, float]] = []
    skipped: list[Tuple[str, float, float]] = []
    for k in common:
        if abs(ref[k] - got[k]) > args.tol:
            (skipped if "asr_cleaned" in k else bad).append((k, ref[k], got[k]))

    n_cmp = len(common) - len(skipped)
    print(f"\n  bc-LSTM validation: {n_cmp - len(bad)}/{n_cmp} comparable cells "
          f"match training-time test metrics (tol {args.tol})")
    if skipped:
        print(f"  {len(skipped)} asr_cleaned cells excluded: reference is on the "
              f"1,787-utterance filtered test set, this pass on all 2,610.\n")
    if bad:
        print(f"  {'run':<40s}{'training':>10s}{'re-scored':>11s}{'diff':>9s}")
        print("  " + "-" * 70)
        for k, a, b in bad:
            print(f"  {k:<40s}{a:10.4f}{b:11.4f}{a - b:+9.4f}")
        print(f"\n  FAILED: {len(bad)} mismatches. The pass does not reproduce "
              f"known-good numbers, so its ctxfusion/fusion output -- which has "
              f"no independent check -- must not be used.\n")
        return 1

    n_ctx = sum(1 for k in got if k.startswith("ctxfusion/"))
    n_fus = sum(1 for k in got if k.startswith("fusion/"))
    print(f"  PASSED. Windowing is faithful, so the {n_ctx} ctxfusion and "
          f"{n_fus} fusion cells in this table can be used.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
