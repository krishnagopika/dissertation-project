#!/usr/bin/env python3.12
"""Score the dialect probe's model predictions against the human annotations.

Every number in `EXPERIMENTS.md` §11 and `ENGLISH_DIALECT_DATA.md` §5 comes
from this script. It was originally run inline, which made the results
unreproducible from the repository -- this file exists so the scoring is a
citable artefact rather than a transcript entry.

Metric
------
Weighted F1, matching the MELD headline metric so the two are comparable.
Macro F1 alongside, because it is far more sensitive to the minority classes
collapsing, which is exactly what happens here (fear n=2).

Reading the CSV
---------------
`dialect_predictions.csv` was hand-edited to add the `human_emotion` column,
and the editor stripped pandas' quoting. Rows whose transcripts contain commas
therefore split into extra fields, and a naive `read_csv` fails.

Recovery is positional and safe: `key` is the FIRST field, and every label
column is a single word with no commas, so counting from the END is stable
regardless of how many extra fields the transcripts introduced:

    [-1]                notes
    [-2]                human_emotion
    [-2-N : -2]         the N model predictions, in header order
    [-3-N]              voxtral_emotion

Only the free-text fields in the middle were damaged, and none are needed for
scoring. The recovery is verified by asserting every recovered label lies in
the seven-class vocabulary.

Usage
-----
    python3.12 src/evaluation/score_dialect.py \\
        --predictions /dcs/large/u5734759/data/dialect_probe_100/dialect_predictions.csv \\
        --wer_summary /dcs/large/u5734759/data/dialect_probe_100/dialect_wer_vad_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

EMOTIONS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]

#: Annotation typos corrected before scoring. Recorded explicitly rather than
#: silently normalised, because altering a human label is a judgement call and
#: the reader is entitled to see which ones were altered.
LABEL_FIXES = {"neutal": "neutral"}


def load_predictions(path: Path) -> pd.DataFrame:
    """Recover key, human label, voxtral label and model predictions.

    See the module docstring for why this cannot use ``pd.read_csv``.

    Args:
        path: The hand-edited predictions CSV.

    Returns:
        One row per clip with columns ``key``, ``human``, ``voxtral`` and one
        column per model.

    Raises:
        ValueError: If any recovered label is outside the seven-class
            vocabulary, which means the positional recovery went wrong.
    """
    lines = path.read_text().splitlines()
    header = lines[0].split(",")
    pred_cols = [h for h in header if h.startswith("pred__")]
    n = len(pred_cols)

    rows: List[Dict[str, str]] = []
    for line in lines[1:]:
        if not line.strip():
            continue
        f = [x.strip() for x in line.split(",")]
        rec = {"key": f[0],
               "human": LABEL_FIXES.get(f[-2].lower(), f[-2].lower()),
               "voxtral": f[-3 - n]}
        rec.update(dict(zip(pred_cols, f[-2 - n:-2])))
        rows.append(rec)
    d = pd.DataFrame(rows)

    for c in ["human", "voxtral"] + pred_cols:
        bad = set(d[c]) - set(EMOTIONS)
        if bad:
            raise ValueError(f"column {c!r} holds non-emotion values {bad} -- "
                             f"positional recovery failed")
    return d


def score_models(d: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    """Overall accuracy, weighted F1 and macro F1 per model."""
    out = []
    for c in models:
        out.append({
            "model": c.replace("pred__trainedon_", ""),
            "accuracy": (d[c] == d.human).mean(),
            "weighted_f1": f1_score(d.human, d[c], average="weighted",
                                    zero_division=0),
            "macro_f1": f1_score(d.human, d[c], average="macro",
                                 zero_division=0),
        })
    return pd.DataFrame(out).sort_values("weighted_f1", ascending=False)


def score_by_accent(d: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    """Weighted F1 per accent, plus the spread across accents.

    The spread is the accent-robustness measure: a model that is uniformly
    mediocre is more robust than one that is excellent on Southern and fails
    on Welsh, even at equal overall F1.
    """
    accents = sorted(d.accent.unique())
    out = []
    for c in models:
        row = {"model": c.replace("pred__trainedon_", "")}
        for a in accents:
            s = d[d.accent == a]
            row[a] = f1_score(s.human, s[c], average="weighted",
                              zero_division=0)
        row["spread"] = max(row[a] for a in accents) - min(row[a] for a in accents)
        out.append(row)
    return pd.DataFrame(out)


def score_per_class(d: pd.DataFrame, model: str) -> pd.DataFrame:
    """Per-class F1, precision and recall for one model.

    ``labels=[e], average="micro"`` isolates a single class: with one label in
    scope, micro-averaging reduces to that class's own score.
    """
    out = []
    for e in EMOTIONS:
        out.append({
            "emotion": e,
            "n": int((d.human == e).sum()),
            "f1": f1_score(d.human, d[model], labels=[e], average="micro",
                           zero_division=0),
            "precision": precision_score(d.human, d[model], labels=[e],
                                         average="micro", zero_division=0),
            "recall": recall_score(d.human, d[model], labels=[e],
                                   average="micro", zero_division=0),
        })
    return pd.DataFrame(out)


def wer_vs_f1(d: pd.DataFrame, model: str, wer_by_accent: Dict[str, float]):
    """Does ASR quality explain the accent gap? (Answer: no.)

    Correlating per-accent WER with per-accent emotion F1 separates two
    effects MELD cannot, because MELD's corpus WER of 0.38 dominates
    everything. Here WER is ~0.04, so a residual accent gap cannot be
    attributed to transcription.

    n = 6 accents, so this rules out a strong relationship, not a modest one.
    """
    from scipy.stats import pearsonr, spearmanr

    rows = []
    for a in sorted(d.accent.unique()):
        s = d[d.accent == a]
        rows.append({"accent": a, "n": len(s), "wer": wer_by_accent.get(a),
                     "f1": f1_score(s.human, s[model], average="weighted",
                                    zero_division=0)})
    r = pd.DataFrame(rows).sort_values("wer")
    pr, pp = pearsonr(r.wer, r.f1)
    sr, sp = spearmanr(r.wer, r.f1)
    return r, {"pearson_r": pr, "pearson_p": pp,
               "spearman_rho": sr, "spearman_p": sp}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--wer_summary", default=None,
                    help="dialect_wer_vad_summary.json, for the WER/F1 test.")
    ap.add_argument("--out_dir", default=None,
                    help="Defaults to the predictions file's directory.")
    args = ap.parse_args()

    pred_path = Path(args.predictions)
    out = Path(args.out_dir) if args.out_dir else pred_path.parent
    out.mkdir(parents=True, exist_ok=True)

    d = load_predictions(pred_path)
    sel = pd.read_csv(pred_path.parent / "selection.csv")[
        ["key", "accent", "gender", "speaker_id"]]
    d = d.merge(sel, on="key", how="left")
    models = [c for c in d.columns if c.startswith("pred__")]
    print(f"{len(d)} clips | {len(models)} models | "
          f"labels: {dict(d.human.value_counts())}\n")

    overall = score_models(d, models + ["voxtral"])
    overall.to_csv(out / "dialect_scores.csv", index=False)
    print("=== overall (weighted F1) ===")
    print(overall.to_string(index=False, float_format="%.3f"))

    by_acc = score_by_accent(d, models + ["voxtral"])
    by_acc.to_csv(out / "dialect_scores_by_accent.csv", index=False)
    trained = by_acc[by_acc.model != "voxtral"]
    print(f"\nmean accent spread: trained {trained.spread.mean():.3f} | "
          f"voxtral {by_acc[by_acc.model=='voxtral'].spread.iloc[0]:.3f}")

    best = "pred__trainedon_" + overall.iloc[0].model
    per_class = score_per_class(d, best)
    per_class.to_csv(out / "dialect_per_class.csv", index=False)
    print(f"\n=== per-class ({overall.iloc[0].model}) ===")
    print(per_class.to_string(index=False, float_format="%.3f"))

    if args.wer_summary and Path(args.wer_summary).exists():
        w = json.loads(Path(args.wer_summary).read_text())["by_accent"]
        r, stats = wer_vs_f1(d, best, w)
        r.to_csv(out / "dialect_wer_vs_f1.csv", index=False)
        print("\n=== WER vs emotion F1 across accents ===")
        print(r.to_string(index=False, float_format="%.4f"))
        print(f"  Pearson r={stats['pearson_r']:+.3f} p={stats['pearson_p']:.3f} | "
              f"Spearman rho={stats['spearman_rho']:+.3f} p={stats['spearman_p']:.3f}")
        json.dump(stats, open(out / "dialect_wer_f1_correlation.json", "w"),
                  indent=2)

    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
