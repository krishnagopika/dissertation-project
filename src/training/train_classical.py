"""Classical baselines on the same cached features the neural models use.

Why this matters
----------------
Every neural result in this project sits on frozen caches: XLM-R ``[CLS]``
768-d and Voxtral masked-mean 1280-d. If logistic regression on those same
vectors reaches a comparable weighted F1, then the fusion head, the pooling
ablation and the context models are not earning their complexity, and the
representation is doing the work.

That is a real possibility here, not a rhetorical one. The measured effects so
far are small: fusion beat text-only by 0.003, and the context-then-fusion 2x2
showed no reliable benefit on test. A strong classical baseline would put those
numbers in proportion.

Three models, chosen for what each rules out
--------------------------------------------
**Logistic regression** -- the linear baseline. Identical in form to the probe
used in the representation analysis, so its score IS the linear separability of
the feature set. Anything a neural model gains over this is nonlinearity or
context, nothing else.

**Linear SVM** -- a different loss (hinge, max-margin) on the same hypothesis
class. Distinguishes "the features are linearly separable" from "logistic
regression's particular objective found it".

**XGBoost** -- nonlinear, axis-aligned splits, no notion of distance. If it
beats the linear models the structure is nonlinear; if it does not, the
representation is essentially linear and depth buys nothing.

Feature sets mirror the neural ablation exactly
-----------------------------------------------
    text      768   XLM-R [CLS] only
    acoustic  1280  masked-mean Voxtral only
    both      2048  concatenated -- the same input the bc-LSTM receives

so "classical vs fusion" and "classical vs bc-LSTM" are like-for-like on
inputs, differing only in the model.

Class imbalance
---------------
MELD is 47% neutral. ``class_weight="balanced"`` for the linear models and
per-sample weights for XGBoost, so a baseline is not simply the majority class
wearing a hat. Weighted F1 is still the headline, with macro F1 alongside
because it is far more sensitive to the minority classes collapsing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import (EMOTION_NAMES, SENTIMENT_NAMES,
                                    compute_emotion_metrics,
                                    compute_sentiment_metrics)
from src.utils import load_config, set_seed, setup_logging

EMOTION2IDX = {n: i for i, n in enumerate(EMOTION_NAMES)}
SENTIMENT2IDX = {"negative": 0, "neutral": 1, "positive": 2}
_SPLIT_CSV = {"train": "train_sent_emo.csv", "dev": "dev_sent_emo.csv",
              "test": "test_sent_emo.csv"}
_SEED = 42


def load_split(config: Dict, split: str, features: str,
               keep: Optional[str] = None
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble (X, emotion, sentiment) for one split.

    Utterances missing either cached vector are DROPPED and counted, never
    zero-filled: a zero vector is indistinguishable from a real embedding to
    any of these models, so a failed extraction would become a training example
    with a valid label.
    """
    df = pd.read_csv(Path(config["data"]["meld_root"]) / _SPLIT_CSV[split])
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(" ", "_", regex=False))
    df = df.dropna(subset=["emotion", "sentiment"])

    keep_set = None
    if keep:
        with open(keep, encoding="utf-8") as f:
            keep_set = set(json.load(f)["keys"])

    text = torch.load(
        str(Path(config["data"]["text_embeddings_path"]) /
            f"{split}_text_embeddings.pt"), map_location="cpu", weights_only=True)
    acou = torch.load(
        str(Path(config["data"]["embeddings_path"]) /
            f"{split}_embeddings_maskedmean.pt"), map_location="cpu",
        weights_only=True)

    X, E, S, missing = [], [], [], 0
    for _, r in df.iterrows():
        key = f"dia{int(r['dialogue_id'])}_utt{int(r['utterance_id'])}"
        if keep_set is not None and key not in keep_set:
            continue
        e, s = str(r["emotion"]).strip().lower(), str(r["sentiment"]).strip().lower()
        if e not in EMOTION2IDX or s not in SENTIMENT2IDX:
            raise ValueError(f"{key}: unrecognised label {e!r}/{s!r}")
        t, a = text.get(key), acou.get(key)
        if (features in ("text", "both") and t is None) or \
           (features in ("acoustic", "both") and a is None):
            missing += 1
            continue
        if features == "text":
            v = t.float().numpy()
        elif features == "acoustic":
            v = a.float().numpy()
        else:
            v = np.concatenate([t.float().numpy(), a.float().numpy()])
        X.append(v); E.append(EMOTION2IDX[e]); S.append(SENTIMENT2IDX[s])

    if missing:
        print(f"    {split}: dropped {missing} utterance(s) missing a cached vector")
    return np.stack(X), np.array(E), np.array(S)


def fit_score(model_name: str, Xtr, ytr, Xte, yte, names: List[str],
              n_jobs: int = 8) -> Dict:
    """Fit one classical model and score it. Returns metrics + timing."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC

    t0 = time.time()
    if model_name == "logreg":
        # Standardised: these features are unnormalised network activations
        # with very different per-dimension scales, and a regularised linear
        # model without scaling silently weights the high-variance dimensions.
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=_SEED))
    elif model_name == "svm":
        clf = make_pipeline(
            StandardScaler(),
            LinearSVC(class_weight="balanced", random_state=_SEED,
                      max_iter=5000))
    elif model_name == "xgboost":
        from xgboost import XGBClassifier
        # Trees are scale-invariant, so no StandardScaler -- adding one would
        # only cost time.
        counts = np.bincount(ytr, minlength=len(names)).astype(float)
        w = len(ytr) / (len(names) * np.maximum(counts, 1))
        clf = XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=_SEED,
            n_jobs=n_jobs, tree_method="hist",
            num_class=len(names), objective="multi:softprob")
        clf.fit(Xtr, ytr, sample_weight=w[ytr])
        pred = clf.predict(Xte)
        fn = (compute_emotion_metrics if len(names) == 7
              else compute_sentiment_metrics)
        m = fn(pred.tolist(), yte.tolist(), names)
        return {"weighted_f1": m["weighted_f1"], "macro_f1": m.get("macro_f1"),
                "per_class_f1": m.get("per_class_f1"),
                "fit_seconds": round(time.time() - t0, 1)}
    else:
        raise ValueError(model_name)

    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    fn = compute_emotion_metrics if len(names) == 7 else compute_sentiment_metrics
    m = fn(pred.tolist(), yte.tolist(), names)
    return {"weighted_f1": m["weighted_f1"], "macro_f1": m.get("macro_f1"),
            "per_class_f1": m.get("per_class_f1"),
            "fit_seconds": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Classical baselines on the cached neural features.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--condition", required=True,
                    choices=["gold", "asr", "asr_cleaned"])
    ap.add_argument("--features", nargs="+", default=["text", "acoustic", "both"],
                    choices=["text", "acoustic", "both"])
    ap.add_argument("--models", nargs="+",
                    default=["logreg", "svm", "xgboost"],
                    choices=["logreg", "svm", "xgboost"])
    ap.add_argument("--test_split", default="test", choices=["dev", "test"])
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--out_dir", default="results_new/classical")
    args = ap.parse_args()

    config = load_config(args.config)
    set_seed(config["data"]["seed"])
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(out), f"classical_{args.condition}")

    filt = config.get("filtering", {})
    keep = (filt.get("keys_paths", {}).get("train")
            if bool(filt.get("enabled", False)) else None)
    logger.info("condition=%s | train keep-list: %s", args.condition,
                Path(keep).name if keep else "none (full train)")

    results: List[Dict] = []
    for feat in args.features:
        # The test split is NEVER filtered -- every model is scored on the same
        # utterances so the numbers are comparable across conditions.
        Xtr, Etr, Str = load_split(config, "train", feat, keep)
        Xte, Ete, Ste = load_split(config, args.test_split, feat, None)
        logger.info("%s | train %s -> test %s", feat, Xtr.shape, Xte.shape)

        for model in args.models:
            for task, ytr, yte, names in (("emotion", Etr, Ete, EMOTION_NAMES),
                                          ("sentiment", Str, Ste, SENTIMENT_NAMES)):
                try:
                    m = fit_score(model, Xtr, ytr, Xte, yte, names, args.n_jobs)
                except Exception as exc:                       # noqa: BLE001
                    logger.error("%s/%s/%s FAILED: %s: %s", feat, model, task,
                                 type(exc).__name__, exc)
                    continue
                rec = {"condition": args.condition, "features": feat,
                       "model": model, "task": task,
                       "split": args.test_split, "n_train": len(ytr),
                       "n_test": len(yte), "dim": int(Xtr.shape[1]), **m}
                results.append(rec)
                logger.info("%-9s %-8s %-9s | WF1 %.4f macro %.4f | %.1fs",
                            feat, model, task, m["weighted_f1"],
                            m["macro_f1"] or 0.0, m["fit_seconds"])

    path = out / f"classical_{args.condition}_{args.test_split}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n  {args.condition} — {args.test_split} weighted F1\n")
    print(f"  {'features':<10s}{'model':<10s}{'emotion':>10s}{'sentiment':>11s}")
    print("  " + "-" * 41)
    for feat in args.features:
        for model in args.models:
            def _v(task):
                return next((r["weighted_f1"] for r in results
                             if r["features"] == feat and r["model"] == model
                             and r["task"] == task), None)
            e, sm = _v("emotion"), _v("sentiment")
            es = f"{e:10.4f}" if e is not None else f"{'·':>10s}"
            ss = f"{sm:11.4f}" if sm is not None else f"{'·':>11s}"
            print(f"  {feat:<10s}{model:<10s}{es}{ss}")
    print(f"\n  -> {path}")


if __name__ == "__main__":
    main()
