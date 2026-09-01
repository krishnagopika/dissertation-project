"""Representation analysis across the pipeline's five stages.

    stage  representation                     dim   produced by
    -----  --------------------------------   ----  -------------------------
    1      text, XLM-R [CLS]                   768  extract_text_embeddings*
    2      acoustic, masked mean               1280  transcribe_all.py
    3      fused                                512  extract_fused_features.py
    4      contextualised (bc-LSTM)             512  extract_context_features.py
    5      context-then-fusion                  512  extract_context_features.py

The question
------------
Does class structure SHARPEN as representations move through the pipeline? If
fusion and context are doing what the F1 numbers suggest, linear separability
should rise 1,2 -> 3 -> 4,5. If it does not, the F1 gains are coming from the
classifier head rather than from the representation.

Linear probe F1 is the primary measure and the ONLY one comparable across
stages. Cosine silhouette is reported per stage but must not be compared
between stages: distance concentration is dimension-dependent, so a 512-d
silhouette is not on the same scale as a 768-d or 1280-d one.

Projections, and what each is good for
--------------------------------------
**PCA** -- always reported WITH explained variance. Without that number a plot
showing no separation is uninterpretable: the structure may exist and simply
not lie in the first two components.

**UMAP** -- fitted on TRAIN and applied to test, so the test plot is not fitted
to itself.

**t-SNE** -- has no ``transform()``, so it is fitted jointly on whatever is
plotted. Two t-SNE plots are therefore SEPARATE embeddings whose axes have no
relationship: they show local neighbourhood structure and nothing about
movement between panels. Every t-SNE figure carries that caption.

Neutral is 47% of MELD
----------------------
Colouring all seven classes equally produces a grey cloud with specks. Neutral
is drawn in light grey at low alpha so the six minority classes are visible,
and a `--drop_neutral` variant is emitted separately. The dropped version is
NOT the primary figure -- removing the majority class changes what is being
looked at.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import EMOTION_NAMES, SENTIMENT_NAMES
from src.evaluation.represent_inputs import probe, silhouette
from src.utils import load_config, setup_logging

EMOTION2IDX = {n: i for i, n in enumerate(EMOTION_NAMES)}
SENTIMENT2IDX = {"negative": 0, "neutral": 1, "positive": 2}
_SPLIT_CSV = {"train": "train_sent_emo.csv", "dev": "dev_sent_emo.csv",
              "test": "test_sent_emo.csv"}
_SEED = 42

#: Neutral is ~47% of MELD; drawn grey so the minority classes are legible.
_NEUTRAL_COLOUR = "#cccccc"


def load_labels(meld_root: str, split: str) -> Dict[str, Tuple[int, int]]:
    """Map utterance key -> (emotion idx, sentiment idx) from the MELD CSV."""
    df = pd.read_csv(Path(meld_root) / _SPLIT_CSV[split])
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(" ", "_", regex=False))
    df = df.dropna(subset=["emotion", "sentiment"])
    out = {}
    for _, r in df.iterrows():
        key = f"dia{int(r['dialogue_id'])}_utt{int(r['utterance_id'])}"
        e = str(r["emotion"]).strip().lower()
        s = str(r["sentiment"]).strip().lower()
        if e in EMOTION2IDX and s in SENTIMENT2IDX:
            out[key] = (EMOTION2IDX[e], SENTIMENT2IDX[s])
    return out


def load_stage(path: Path, labels: Dict[str, Tuple[int, int]]
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load one cached representation and align it with labels.

    Returns (X, emotion, sentiment, keys) over the INTERSECTION of the cache
    and the labelled utterances -- MELD's audio directories hold more clips
    than the CSV labels, and a representation cache may cover either set.
    """
    d = torch.load(str(path), map_location="cpu", weights_only=True)
    keys = sorted(set(d) & set(labels))
    if not keys:
        raise ValueError(f"{path}: no overlap between cache and labels")
    X = np.stack([d[k].float().numpy().ravel() for k in keys])
    E = np.array([labels[k][0] for k in keys])
    S = np.array([labels[k][1] for k in keys])
    return X, E, S, keys


def cosine_separation(X: np.ndarray, y: np.ndarray,
                      class_names: List[str]) -> Dict:
    """Mean cosine similarity WITHIN a class vs BETWEEN classes.

    Answers "are embeddings of the same emotion actually more similar?"
    directly, without the dimensionality dependence that makes silhouette
    incomparable across stages. The GAP (within - between) is the summary:
    positive means the class is cohesive relative to the rest of the space.

    Computed on the full-dimensional embedding, never on 2-D projection
    coordinates -- a projection discards most of the variance and a cosine
    measured there describes the projection, not the representation.
    """
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    per_class, wit, bet = {}, [], []
    for i, name in enumerate(class_names):
        m = y == i
        n = int(m.sum())
        if n < 2:
            per_class[name] = {"n": n, "within": None, "between": None,
                               "gap": None}
            continue
        A = Xn[m]
        # Exclude the diagonal: a vector's similarity with itself is 1 and
        # would inflate every within-class mean toward 1/n.
        S = A @ A.T
        w = float((S.sum() - np.trace(S)) / (n * (n - 1)))
        B = Xn[~m]
        b = float((A @ B.T).mean()) if len(B) else float("nan")
        per_class[name] = {"n": n, "within": w, "between": b, "gap": w - b}
        wit.append(w); bet.append(b)
    return {"per_class": per_class,
            "macro_within": float(np.mean(wit)) if wit else None,
            "macro_between": float(np.mean(bet)) if bet else None,
            "macro_gap": float(np.mean(wit) - np.mean(bet)) if wit else None}


def project(Xtr: np.ndarray, X: np.ndarray, method: str,
            perplexity: float = 30.0) -> Tuple[np.ndarray, Optional[float]]:
    """2-D projection. Returns (coords, explained_variance_ratio or None).

    PCA and UMAP are FIT ON TRAIN and applied to X, so the plotted split is not
    fitted to itself. t-SNE has no transform, so it is fit on X alone -- which
    is why t-SNE axes are not comparable between figures.
    """
    if method == "pca":
        from sklearn.decomposition import PCA
        p = PCA(n_components=2, random_state=_SEED).fit(Xtr)
        return p.transform(X), float(p.explained_variance_ratio_.sum())
    if method == "umap":
        import umap
        u = umap.UMAP(n_components=2, random_state=_SEED).fit(Xtr)
        return u.transform(X), None
    if method == "tsne":
        from sklearn.manifold import TSNE
        n = min(perplexity, max(5.0, (len(X) - 1) / 3.0))
        return TSNE(n_components=2, random_state=_SEED,
                    perplexity=n, init="pca").fit_transform(X), None
    raise ValueError(f"unknown method {method!r}")


def scatter(ax, xy: np.ndarray, y: np.ndarray, names: List[str],
            drop_neutral: bool = False) -> None:
    """One projection panel, with neutral de-emphasised."""
    import matplotlib.cm as cm
    neutral = names.index("neutral") if "neutral" in names else -1
    palette = cm.get_cmap("tab10")
    for i, name in enumerate(names):
        m = y == i
        if not m.any():
            continue
        if i == neutral:
            if drop_neutral:
                continue
            ax.scatter(xy[m, 0], xy[m, 1], s=3, c=_NEUTRAL_COLOUR,
                       alpha=0.35, linewidths=0, label=f"{name} ({m.sum()})",
                       zorder=1)
        else:
            ax.scatter(xy[m, 0], xy[m, 1], s=5, color=palette(i % 10),
                       alpha=0.75, linewidths=0, label=f"{name} ({m.sum()})",
                       zorder=2)
    ax.set_xticks([]); ax.set_yticks([])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Probe / silhouette / projections across pipeline stages.")
    ap.add_argument("--config", required=True,
                    help="Any config carrying data.meld_root (e.g. ctx_gold.yaml).")
    ap.add_argument("--condition", required=True,
                    choices=["gold", "asr", "asr_cleaned"])
    ap.add_argument("--splits", nargs="+", default=["test"],
                    choices=["train", "dev", "test"])
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--stages", nargs="+",
                    default=["text", "acoustic", "fused", "context", "ctxfusion"])
    ap.add_argument("--no_umap", action="store_true")
    ap.add_argument("--no_tsne", action="store_true")
    ap.add_argument("--tsne_perplexity", type=float, default=30.0)
    ap.add_argument("--max_points", type=int, default=4000,
                    help="Subsample before t-SNE; it is O(n^2) and unreadable "
                         "beyond a few thousand points anyway.")
    args = ap.parse_args()

    config = load_config(args.config)
    cond = args.condition
    root = Path("/dcs/large/u5734759/data")
    out = Path(args.out_dir or f"results_new/represent/{cond}")
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(out), "represent_learned")

    # stage name -> (directory, filename template)
    STAGES = {
        "text":      (root / f"meld_text_emb_v2/{cond}", "{split}_text_embeddings.pt"),
        "acoustic":  (root / "meld_extracted/mini/acoustic", "{split}_embeddings_maskedmean.pt"),
        "fused":     (root / f"meld_fused/{cond}", "{split}_fused.pt"),
        "context":   (root / f"meld_context/{cond}", "{split}_context.pt"),
        "ctxfusion": (root / f"meld_context/ctxfusion_{cond}", "{split}_context.pt"),
        # ContextThenFusion with ONLY the acoustic branch contextualised. This
        # is the closest thing to "acoustics after a dialogue model": bc-LSTM
        # consumes the 2048-d concatenation and emits one blended vector, so
        # the modalities are not separable after it. Here the text branch skips
        # its BiLSTM, so whatever temporal structure appears comes from audio.
        "ctxfusion_acoustic": (root / f"meld_context/ctxfusion_aonly_{cond}",
                               "{split}_context.pt"),
        # Acoustic-only fusion models, one per pooler. The `acoustic` stage
        # above is the MASKED MEAN, but Phase 1 found learned attention worth
        # +0.037 dev WF1 over it (0.5044 vs 0.4671) -- so the masked mean is
        # the WORSE representation and visualising only that understates what
        # the audio channel carries. These are text-free, so the vector is the
        # pooled acoustic signal and nothing else.
        "acoustic_attn": (root / f"meld_pooled/attention", "{split}_fused.pt"),
        "acoustic_attnstats": (root / f"meld_pooled/attentive_stats", "{split}_fused.pt"),
        "acoustic_attnfixed": (root / f"meld_pooled/attention_fixed", "{split}_fused.pt"),
        "acoustic_mean_learned": (root / f"meld_pooled/masked_mean", "{split}_fused.pt"),
    }

    results: Dict = {"condition": cond, "stages": {}}
    for stage in args.stages:
        if stage not in STAGES:
            logger.warning("unknown stage %s -- skipping", stage); continue
        d, tmpl = STAGES[stage]
        # Probes are fit on TRAIN and scored on the target split, so train is
        # always needed even when only test is being plotted.
        f_tr = d / tmpl.format(split="train")
        if not f_tr.exists():
            logger.warning("stage %s: %s missing -- skipping", stage, f_tr)
            continue

        lab_tr = load_labels(config["data"]["meld_root"], "train")
        Xtr, Etr, Str, _ = load_stage(f_tr, lab_tr)
        results["stages"][stage] = {"dim": int(Xtr.shape[1]), "splits": {}}

        for split in args.splits:
            f = d / tmpl.format(split=split)
            if not f.exists():
                logger.warning("stage %s split %s missing -- skipping", stage, split)
                continue
            lab = load_labels(config["data"]["meld_root"], split)
            X, E, S, keys = load_stage(f, lab)

            r = {
                "n": len(keys),
                "probe_emotion": probe(Xtr, Etr, X, E, EMOTION_NAMES),
                "probe_sentiment": probe(Xtr, Str, X, S, SENTIMENT_NAMES),
                "silhouette_emotion": silhouette(X, E, EMOTION_NAMES),
                "silhouette_sentiment": silhouette(X, S, SENTIMENT_NAMES),
                "cosine_emotion": cosine_separation(X, E, EMOTION_NAMES),
                "cosine_sentiment": cosine_separation(X, S, SENTIMENT_NAMES),
            }
            logger.info("%s | %-10s | n=%5d dim=%4d | probe emo %.4f sent %.4f "
                        "| cos gap emo %+.4f",
                        split, stage, len(keys), X.shape[1],
                        r["probe_emotion"]["weighted_f1"],
                        r["probe_sentiment"]["weighted_f1"],
                        r["cosine_emotion"]["macro_gap"] or 0.0)

            methods = ["pca"] + ([] if args.no_umap else ["umap"]) \
                              + ([] if args.no_tsne else ["tsne"])
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Subsample once so every projection plots the SAME points.
            idx = np.arange(len(X))
            if len(X) > args.max_points:
                rng = np.random.default_rng(_SEED)
                idx = rng.choice(len(X), args.max_points, replace=False)

            for meth in methods:
                try:
                    xy, evr = project(Xtr, X[idx], meth, args.tsne_perplexity)
                except Exception as exc:                        # noqa: BLE001
                    logger.error("%s %s %s failed: %s", stage, split, meth, exc)
                    continue
                if evr is not None:
                    r[f"{meth}_explained_variance"] = evr
                for task, y, names in (("emotion", E[idx], EMOTION_NAMES),
                                       ("sentiment", S[idx], SENTIMENT_NAMES)):
                    for drop in (False, True):
                        if drop and "neutral" not in names:
                            continue
                        fig, ax = plt.subplots(figsize=(6, 5))
                        scatter(ax, xy, y, names, drop_neutral=drop)
                        cap = f"{cond} | {stage} ({X.shape[1]}-d) | {split} | {meth}"
                        if evr is not None:
                            cap += f" | EVR {evr:.3f}"
                        if meth == "tsne":
                            cap += "\nt-SNE axes are not comparable between figures"
                        ax.set_title(cap, fontsize=8)
                        ax.legend(fontsize=5, markerscale=2, loc="best",
                                  frameon=False)
                        fig.tight_layout()
                        suffix = "_noneutral" if drop else ""
                        fig.savefig(out / f"{split}_{stage}_{meth}_{task}{suffix}.png",
                                    dpi=150)
                        plt.close(fig)

            results["stages"][stage]["splits"][split] = r

    with open(out / "representations.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n  {cond} — probe weighted F1 (the comparable measure)\n")
    print(f"  {'stage':<12s}{'dim':>6s}" +
          "".join(f"{s + ' probe':>13s}{s + ' cosgap':>14s}" for s in args.splits))
    print("  " + "-" * (18 + 27 * len(args.splits)))
    for stage, sd in results["stages"].items():
        row = ""
        for sp in args.splits:
            if sp in sd["splits"]:
                d = sd["splits"][sp]
                row += f"{d['probe_emotion']['weighted_f1']:13.4f}"
                row += f"{d['cosine_emotion']['macro_gap']:14.4f}"
            else:
                row += f"{'·':>13s}{'·':>14s}"
        print(f"  {stage:<12s}{sd['dim']:>6d}{row}")
    print("\n  probe = linear separability, comparable across stages")
    print("  cosgap = mean(within-class cosine) - mean(between-class cosine)")
    print("  silhouette is in the JSON but is NOT comparable across stages")
    print(f"\n  -> {out}")


if __name__ == "__main__":
    main()
