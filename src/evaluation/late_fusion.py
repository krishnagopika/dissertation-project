"""
late_fusion.py — Weighted-scoring late fusion of N modality models
===================================================================
Decision-level multimodal fusion: combine the per-class scores of several
unimodal models and pick the highest-scoring class. Tests whether fusing
  * text      — XLM-R (mini_aug)
  * zeroshot  — Voxtral LLM prompted classification
  * acoustic  — acoustic-only classifier (encoder fine-tune / cached-embedding head)
beats any single model. Core multimodal-fusion claim of the project.

Each source is a JSON of per-utterance probability vectors:
  { "dev":  { "dia0_utt0": {"emotion_probs": [...7], "sentiment_probs": [...3]}, ... },
    "test": { ... } }

Sources are passed as ``name=path`` (soft probs) or, for argmax-only outputs
(e.g. the zero-shot predictions JSON), ``name=template`` via --predictions
where ``template`` contains ``{split}`` — these are converted to one-hot.

Weighting schemes evaluated (all "weighted scoring"):
  * per-modality baselines (argmax of each source)
  * weighted average with a simplex grid search on dev (only if <= 3 sources)
  * meta : LogisticRegression over all concatenated prob vectors (learned
           per-sample weighting — the main N-way fuser)

Tuning is on dev; final numbers reported on test.

Usage
-----
  python3.12 src/evaluation/late_fusion.py --config src/configs/mini.yaml \\
      --probs text=results/mini/xlmr_text_probs.json \\
              acoustic=results/mini/acoustic_probs.json \\
      --predictions zeroshot=results/mini/voxtral_zeroshot_{split}_predictions.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import (
    EMOTION_NAMES,
    SENTIMENT_NAMES,
    compute_emotion_metrics,
    compute_sentiment_metrics,
)
from src.utils import load_config, setup_logging

NUM_EMOTION = len(EMOTION_NAMES)
NUM_SENTIMENT = len(SENTIMENT_NAMES)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_soft(path: Path) -> Dict[str, Dict[str, Dict]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_onehot(template: str, splits: List[str]) -> Dict[str, Dict[str, Dict]]:
    """Convert a zero-shot argmax predictions JSON to one-hot prob vectors."""
    out: Dict[str, Dict[str, Dict]] = {}
    for split in splits:
        with open(Path(template.format(split=split)), "r", encoding="utf-8") as f:
            preds = json.load(f)
        out[split] = {}
        for key, rec in preds.items():
            e = np.zeros(NUM_EMOTION); e[int(rec["emotion_pred"])] = 1.0
            s = np.zeros(NUM_SENTIMENT); s[int(rec["sentiment_pred"])] = 1.0
            out[split][key] = {"emotion_probs": e.tolist(), "sentiment_probs": s.tolist()}
    return out


def stack(probs: Dict[str, Dict], keys: List[str], field: str, dim: int) -> np.ndarray:
    arr = np.zeros((len(keys), dim))
    for i, k in enumerate(keys):
        v = probs.get(k, {}).get(field)
        if v is not None:
            arr[i] = np.asarray(v, dtype=float)
    return arr


def load_gold(meld_root: Path, split: str) -> Dict[str, Tuple[int, int]]:
    import pandas as pd
    from src.preprocessing.transcribe_all import CSV_MAP
    e2i = {n: i for i, n in enumerate(EMOTION_NAMES)}
    s2i = {n: i for i, n in enumerate(SENTIMENT_NAMES)}
    df = pd.read_csv(meld_root / CSV_MAP[split])
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    df = df.dropna(subset=["emotion", "sentiment"]).reset_index(drop=True)
    gold: Dict[str, Tuple[int, int]] = {}
    for _, row in df.iterrows():
        key = f"dia{int(row['dialogue_id'])}_utt{int(row['utterance_id'])}"
        gold[key] = (e2i.get(str(row["emotion"]).strip().lower(), 0),
                     s2i.get(str(row["sentiment"]).strip().lower(), 1))
    return gold


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def score(preds: List[int], labels: List[int], dim: int) -> Dict:
    fn = compute_emotion_metrics if dim == NUM_EMOTION else compute_sentiment_metrics
    return fn(preds, labels)


def simplex_grid(n: int, step: float = 0.1) -> List[Tuple[float, ...]]:
    """All weight tuples of length n on a grid that sum to 1."""
    ticks = int(round(1.0 / step))
    combos = []
    for raw in product(range(ticks + 1), repeat=n):
        if sum(raw) == ticks:
            combos.append(tuple(r / ticks for r in raw))
    return combos


# ---------------------------------------------------------------------------
# Per-task fusion
# ---------------------------------------------------------------------------

def run_task(
    task: str,
    field: str,
    dim: int,
    sources: Dict[str, Dict[str, Dict]],
    gold_dev: Dict[str, Tuple[int, int]],
    gold_test: Dict[str, Tuple[int, int]],
    logger: logging.Logger,
) -> Dict:
    names = list(sources.keys())
    gi = 0 if task == "emotion" else 1

    def common_keys(split: str, gold) -> List[str]:
        sets = [set(sources[n][split]) for n in names]
        common = set.intersection(*sets) & set(gold)
        return sorted(common)

    dev_keys = common_keys("dev", gold_dev)
    test_keys = common_keys("test", gold_test)
    logger.info("[%s] sources=%s | dev keys=%d | test keys=%d",
                task, names, len(dev_keys), len(test_keys))

    dev = {n: stack(sources[n]["dev"], dev_keys, field, dim) for n in names}
    test = {n: stack(sources[n]["test"], test_keys, field, dim) for n in names}
    y_dev = np.array([gold_dev[k][gi] for k in dev_keys])
    y_test = np.array([gold_test[k][gi] for k in test_keys])

    results: Dict[str, Dict] = {}

    # Per-modality baselines
    for n in names:
        results[f"baseline_{n}"] = score(test[n].argmax(1).tolist(), y_test.tolist(), dim)

    # Weighted-average simplex search on dev (only feasible for few sources)
    if len(names) <= 3:
        best_w, best_wf1 = None, -1.0
        for w in simplex_grid(len(names), step=0.1):
            mix = sum(w[i] * dev[names[i]] for i in range(len(names)))
            wf1 = score(mix.argmax(1).tolist(), y_dev.tolist(), dim)["weighted_f1"]
            if wf1 > best_wf1:
                best_wf1, best_w = wf1, w
        mix_test = sum(best_w[i] * test[names[i]] for i in range(len(names)))
        results["weighted_avg"] = score(mix_test.argmax(1).tolist(), y_test.tolist(), dim)
        results["weighted_avg"]["weights"] = {names[i]: round(best_w[i], 2) for i in range(len(names))}
    else:
        logger.info("  >3 sources — skipping simplex grid, using meta only")

    # Meta classifier over all concatenated probs (learned weighting)
    X_dev = np.concatenate([dev[n] for n in names], axis=1)
    X_test = np.concatenate([test[n] for n in names], axis=1)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    clf.fit(X_dev, y_dev)
    results["meta"] = score(clf.predict(X_test).tolist(), y_test.tolist(), dim)

    # Confidence gating: per utterance, trust whichever modality is most confident.
    test_preds = {n: test[n].argmax(1) for n in names}
    conf = np.stack([test[n].max(1) for n in names], axis=1)   # (N, M)
    most_conf = conf.argmax(1)
    gated = np.array([test_preds[names[most_conf[i]]][i] for i in range(len(y_test))])
    results["confidence_gate"] = score(gated.tolist(), y_test.tolist(), dim)

    # Oracle upper bound: correct if ANY modality is correct (not a usable model;
    # shows the ceiling of perfect per-sample model selection).
    oracle = np.array([
        y_test[i] if any(test_preds[n][i] == y_test[i] for n in names)
        else test_preds[names[0]][i]
        for i in range(len(y_test))
    ])
    results["oracle_upper"] = score(oracle.tolist(), y_test.tolist(), dim)

    logger.info("=== %s (test WF1 | macro | per-class F1) ===", task)
    for name, m in results.items():
        extra = f"  weights={m['weights']}" if "weights" in m else ""
        logger.info("  %-18s WF1 %.4f | macro %.4f%s",
                    name, m["weighted_f1"], m["macro_f1"], extra)
        pcf = "  ".join(f"{k}={v:.3f}" for k, v in m["per_class_f1"].items())
        logger.info("        per-class: %s", pcf)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_joint_meta(
    sources: Dict[str, Dict[str, Dict]],
    gold_dev: Dict[str, Tuple[int, int]],
    gold_test: Dict[str, Tuple[int, int]],
    logger: logging.Logger,
) -> Dict[str, Dict]:
    """Joint meta-classifier: predict each task from ALL signals of BOTH tasks.

    Feature vector per utterance = for every modality, its emotion_probs (7) and
    sentiment_probs (3) concatenated. This lets the emotion predictor exploit the
    (stronger) sentiment signal and vice versa, learning the emotion<->sentiment
    relationship from data — a learned 'final weighted formula' over everything.
    """
    names = list(sources.keys())

    def common(split: str, gold) -> List[str]:
        sets = [set(sources[n][split]) for n in names]
        return sorted(set.intersection(*sets) & set(gold))

    def feat(split: str, keys: List[str]) -> np.ndarray:
        mats = []
        for n in names:
            mats.append(stack(sources[n][split], keys, "emotion_probs", NUM_EMOTION))
            mats.append(stack(sources[n][split], keys, "sentiment_probs", NUM_SENTIMENT))
        return np.concatenate(mats, axis=1)

    dev_keys = common("dev", gold_dev)
    test_keys = common("test", gold_test)
    X_dev, X_test = feat("dev", dev_keys), feat("test", test_keys)

    out: Dict[str, Dict] = {}
    logger.info("=== joint_meta (uses all modalities x both tasks; %d features) ===",
                X_dev.shape[1])
    for task, gi, dim in [("emotion", 0, NUM_EMOTION), ("sentiment", 1, NUM_SENTIMENT)]:
        y_dev = np.array([gold_dev[k][gi] for k in dev_keys])
        y_test = np.array([gold_test[k][gi] for k in test_keys])
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
        clf.fit(X_dev, y_dev)
        out[task] = score(clf.predict(X_test).tolist(), y_test.tolist(), dim)
        logger.info("  joint_meta %-9s WF1 %.4f | macro %.4f",
                    task, out[task]["weighted_f1"], out[task]["macro_f1"])
    return out


def parse_named(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"Expected name=path, got '{it}'")
        name, path = it.split("=", 1)
        out[name] = path
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Weighted-scoring late fusion of N modality models.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--probs", nargs="*", default=[],
                        help="Soft-prob sources as name=path.json")
    parser.add_argument("--predictions", nargs="*", default=[],
                        help="Argmax-pred sources as name=template_with_{split}.json (→ one-hot)")
    args = parser.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config["training"]["log_dir"], "late_fusion")
    meld_root = Path(config["data"]["meld_root"])

    sources: Dict[str, Dict[str, Dict]] = {}
    for name, path in parse_named(args.probs).items():
        sources[name] = load_soft(Path(path))
        logger.info("Source '%s': soft probs (%s)", name, path)
    for name, tmpl in parse_named(args.predictions).items():
        sources[name] = load_onehot(tmpl, ["dev", "test"])
        logger.info("Source '%s': one-hot from predictions (%s)", name, tmpl)

    if len(sources) < 2:
        raise ValueError("Need >= 2 sources to fuse.")

    gold_dev = load_gold(meld_root, "dev")
    gold_test = load_gold(meld_root, "test")

    all_results = {
        "emotion": run_task("emotion", "emotion_probs", NUM_EMOTION, sources, gold_dev, gold_test, logger),
        "sentiment": run_task("sentiment", "sentiment_probs", NUM_SENTIMENT, sources, gold_dev, gold_test, logger),
    }

    # Joint meta: predict each task using ALL signals from BOTH tasks.
    joint = run_joint_meta(sources, gold_dev, gold_test, logger)
    all_results["emotion"]["joint_meta"] = joint["emotion"]
    all_results["sentiment"]["joint_meta"] = joint["sentiment"]

    out_path = Path(config["evaluation"]["output_dir"]) / "late_fusion_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info("Saved fusion results → %s", out_path)


if __name__ == "__main__":
    main()
