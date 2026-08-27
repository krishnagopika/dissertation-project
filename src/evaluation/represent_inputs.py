#!/usr/bin/env python3.12
"""RQ1 — what emotion/sentiment information is in each modality, before fusion?

Operates on the CACHED input representations, so it depends on no trained
fusion model and can run while training is in progress:

    text      XLM-R [CLS], 768-d          {split}_text_embeddings.pt
    acoustic  masked-mean pooled, 1280-d  {split}_embeddings_maskedmean.pt

Four measurements per modality.

**Linear probes** — the primary result. A logistic regression is fitted on
TRAIN and scored on TEST. Probe F1 answers "is the information linearly
recoverable", which is what matters when a nonlinear head sits downstream. It
is also the only measure here that is comparable across representations of
different width, which silhouette is not.

**Cosine silhouette** — secondary. `metric="cosine"`, not the sklearn default
of Euclidean: at 768 and 1280 dimensions, distance concentration drives
Euclidean silhouette toward 0 regardless of real structure, so the default
would report "no clustering" for any input. Reported PER CLASS as well as
overall, because at 47% neutral the mean is largely a statement about neutral.

Silhouette is NOT comparable between the 768-d text and 1280-d acoustic
representations — concentration is dimension-dependent. Compare within a
modality (e.g. sentiment vs emotion), not across.

**PCA** — two components, with explained-variance ratio reported. Without that
ratio a plot showing no separation is uninterpretable: the structure may exist
and have been discarded by the projection.

**UMAP** — seeded (`random_state=42`), which forces single-threaded execution
but makes the figure reproducible.

**t-SNE** — seeded likewise, but it CANNOT follow the fit-on-train protocol the
other two use: sklearn's TSNE exposes only `fit_transform`, because the
algorithm optimises point positions directly rather than learning a mapping, so
there is nothing to apply to unseen data. It is therefore fitted on train and
test JOINTLY and the coordinates split afterwards, which keeps a shared
coordinate system at the cost of letting test influence the layout. Acceptable
for a figure; it would not be acceptable for a metric, and no metric here uses
it.

Read t-SNE with care: it preserves local neighbourhoods and distorts global
geometry, so between-cluster distances and cluster sizes in a t-SNE plot are
not meaningful. It can also produce apparent clusters from unstructured data at
an unsuitable perplexity. PCA remains the primary figure for that reason.

Usage:
    python3.12 src/evaluation/represent_inputs.py \\
        --config src/configs/extract_mini.yaml --splits train test
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
from src.utils import load_config, setup_logging

EMOTION2IDX = {n: i for i, n in enumerate(EMOTION_NAMES)}
SENTIMENT2IDX = {"negative": 0, "neutral": 1, "positive": 2}
_SPLIT_CSV = {"train": "train_sent_emo.csv", "dev": "dev_sent_emo.csv",
              "test": "test_sent_emo.csv"}
_SEED = 42


def load_modality(
    meld_root: Path,
    split: str,
    text_dir: Path,
    acoustic_dir: Path,
    logger,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load aligned text, acoustic, and labels for one split.

    Returns:
        (text, acoustic, emotion_idx, sentiment_idx, keys). Only utterances
        present in BOTH caches are returned, so the two modalities are aligned
        row-for-row and every comparison is on the same utterances.
    """
    df = pd.read_csv(meld_root / _SPLIT_CSV[split])
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(" ", "_", regex=False))
    df = df.dropna(subset=["emotion", "sentiment"]).reset_index(drop=True)
    df["emotion"] = df["emotion"].str.strip().str.lower()
    df["sentiment"] = df["sentiment"].str.strip().str.lower()

    tpath = text_dir / f"{split}_text_embeddings.pt"
    apath = acoustic_dir / f"{split}_embeddings_maskedmean.pt"
    for p in (tpath, apath):
        if not p.exists():
            raise FileNotFoundError(p)
    text_emb = torch.load(str(tpath), map_location="cpu")
    aco_emb = torch.load(str(apath), map_location="cpu")

    keys, T, A, E, S = [], [], [], [], []
    skipped = 0
    for _, row in df.iterrows():
        k = f"dia{int(row['dialogue_id'])}_utt{int(row['utterance_id'])}"
        t, a = text_emb.get(k), aco_emb.get(k)
        if t is None or a is None:
            skipped += 1
            continue
        # An all-zero acoustic vector is what extraction writes on a hook miss.
        # It is not a representation and would sit at the origin, distorting
        # both silhouette and the probe.
        if float(np.abs(np.asarray(a, dtype=np.float32)).sum()) == 0.0:
            skipped += 1
            continue
        keys.append(k)
        T.append(np.asarray(t, dtype=np.float32).ravel())
        A.append(np.asarray(a, dtype=np.float32).ravel())
        E.append(EMOTION2IDX[row["emotion"]])
        S.append(SENTIMENT2IDX[row["sentiment"]])

    logger.info("%s | %d aligned utterances (%d skipped: missing or zero-filled)",
                split, len(keys), skipped)
    return (np.stack(T), np.stack(A), np.array(E), np.array(S), keys)


def probe(
    x_train: np.ndarray, y_train: np.ndarray,
    x_test: np.ndarray, y_test: np.ndarray,
    class_names: List[str],
) -> Dict:
    """Linear probe: fit on TRAIN, score on TEST.

    Fitting and scoring on the same split would measure memorisation, not
    recoverable structure. Features are standardised because logistic
    regression with L2 is scale-sensitive and the two modalities have very
    different scales.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(
        StandardScaler(),
        # multi_class= was removed in sklearn 1.7+; multinomial is the default
        # for multi-class problems. class_weight="balanced" matters here: at
        # 17.6:1 imbalance an unweighted probe predicts neutral almost always
        # and its macro F1 says nothing about the representation.
        LogisticRegression(max_iter=2000, class_weight="balanced",
                           random_state=_SEED),
    )
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    per_class = f1_score(y_test, pred, average=None,
                         labels=list(range(len(class_names))), zero_division=0)
    return {
        "weighted_f1": round(float(f1_score(y_test, pred, average="weighted",
                                            zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_test, pred, average="macro",
                                         zero_division=0)), 4),
        "per_class_f1": {n: round(float(v), 4)
                         for n, v in zip(class_names, per_class)},
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


def silhouette(x: np.ndarray, y: np.ndarray, class_names: List[str]) -> Dict:
    """Cosine silhouette, overall and per class.

    Cosine rather than Euclidean: these are high-dimensional embeddings, where
    Euclidean distances concentrate and drive the score toward 0 irrespective
    of structure.
    """
    from sklearn.metrics import silhouette_samples, silhouette_score

    if len(np.unique(y)) < 2:
        return {"overall": None, "note": "fewer than two classes present"}
    overall = float(silhouette_score(x, y, metric="cosine", random_state=_SEED))
    samples = silhouette_samples(x, y, metric="cosine")
    per_class = {}
    for i, name in enumerate(class_names):
        m = y == i
        per_class[name] = (round(float(samples[m].mean()), 4)
                           if m.sum() else None)
    return {"overall": round(overall, 4), "per_class": per_class,
            "metric": "cosine"}


def fit_reducer(x_train: np.ndarray, method: str):
    """Fit a 2-D projection on TRAIN only.

    Fitted once on train and then APPLIED to every split, rather than refitted
    per split. Refitting would place each split in its own coordinate system,
    and the train-vs-test panels would then be unrelated pictures -- you could
    not tell whether test occupies the same region as train, which is the whole
    point of showing them together.

    With a shared projection, "test looks like train" and "test is displaced"
    become visually distinguishable, and that displacement is what overfitting
    looks like in representation space.

    Returns:
        (transform_fn, explained_variance_ratio). The ratio is None for UMAP,
        which has no such notion; for PCA it must be reported, since a plot
        showing no separation is uninterpretable without knowing how much
        variance the projection retained.
    """
    from sklearn.preprocessing import StandardScaler

    if method == "pca":
        from sklearn.decomposition import PCA
        scaler = StandardScaler().fit(x_train)
        p = PCA(n_components=2, random_state=_SEED).fit(scaler.transform(x_train))
        evr = [round(float(v), 4) for v in p.explained_variance_ratio_]
        return (lambda x: p.transform(scaler.transform(x))), evr

    import umap
    scaler = StandardScaler().fit(x_train)
    # random_state forces single-threaded execution but makes the figure
    # reproducible, which matters more here than speed.
    reducer = umap.UMAP(n_components=2, random_state=_SEED,
                        n_neighbors=15, min_dist=0.1,
                        metric="cosine").fit(scaler.transform(x_train))
    return (lambda x: reducer.transform(scaler.transform(x))), None


def plot_2d(coords, labels, class_names, title, path, evr=None) -> None:
    """Scatter coloured by class, saved to `path`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    for i, name in enumerate(class_names):
        m = labels == i
        if not m.sum():
            continue
        ax.scatter(coords[m, 0], coords[m, 1], s=6, alpha=0.5,
                   color=cmap(i % 10), label=f"{name} (n={int(m.sum())})")
    sub = (f"  ·  PC1 {evr[0]*100:.1f}%, PC2 {evr[1]*100:.1f}% of variance"
           if evr else "")
    ax.set_title(title + sub, fontsize=10)
    ax.legend(markerscale=2, fontsize=7, loc="best")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "test"],
                    choices=["train", "dev", "test"])
    ap.add_argument("--text_dir", type=str, default=None,
                    help="Override config's text_embeddings_path.")
    ap.add_argument("--no_umap", action="store_true")
    ap.add_argument("--no_tsne", action="store_true")
    ap.add_argument("--tsne_perplexity", type=float, default=30.0,
                    help="t-SNE perplexity. Roughly the effective neighbourhood "
                         "size; must be < n_samples/3.")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    config = load_config(args.config)
    logger = setup_logging(config["training"]["log_dir"], "represent_inputs")
    meld_root = Path(config["data"]["meld_root"])
    text_dir = Path(args.text_dir or config["data"]["text_embeddings_path"])
    acoustic_dir = Path(config["data"]["embeddings_path"])
    out = Path(config["evaluation"]["output_dir"]) / f"represent_inputs{args.tag}"
    (out / "figures").mkdir(parents=True, exist_ok=True)

    logger.info("text: %s", text_dir)
    logger.info("acoustic: %s", acoustic_dir)

    data = {s: load_modality(meld_root, s, text_dir, acoustic_dir, logger)
            for s in args.splits}
    if "train" not in data:
        raise ValueError("--splits must include train: probes are fitted on it.")

    Ttr, Atr, Etr, Str, _ = data["train"]
    results: Dict = {"text_dir": str(text_dir), "acoustic_dir": str(acoustic_dir),
                     "dims": {"text": int(Ttr.shape[1]),
                              "acoustic": int(Atr.shape[1])},
                     "protocol": {
                         "probes": "fitted on train, scored on each split",
                         "projections": "PCA/UMAP fitted on train, applied to "
                                        "each split; t-SNE fitted on all splits "
                                        "jointly (no transform() exists) and "
                                        "coordinates sliced per split",
                         "silhouette": "computed independently per split "
                                       "(unsupervised, nothing is fitted)",
                     }}

    # Projections fitted ONCE on train, then applied to every split, so all
    # panels share a coordinate system and are directly comparable.
    methods = ["pca"]
    if not args.no_umap:
        methods.append("umap")
    reducers: Dict[str, Dict] = {}
    for mod, Xtr in (("text", Ttr), ("acoustic", Atr)):
        reducers[mod] = {}
        for meth in methods:
            logger.info("fitting %s on train | %s", meth.upper(), mod)
            reducers[mod][meth] = fit_reducer(Xtr, meth)

    # t-SNE separately: no transform(), so it is fitted on all splits at once
    # and the coordinates are sliced back out per split.
    tsne_coords: Dict[str, Dict[str, np.ndarray]] = {}
    if not args.no_tsne:
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        for mod in ("text", "acoustic"):
            stacked, bounds, off = [], {}, 0
            for sp in args.splits:
                X = data[sp][0] if mod == "text" else data[sp][1]
                stacked.append(X)
                bounds[sp] = (off, off + len(X))
                off += len(X)
            allX = StandardScaler().fit_transform(np.concatenate(stacked))
            logger.info("fitting t-SNE jointly on %s | %d points, perplexity %.0f",
                        mod, len(allX), args.tsne_perplexity)
            emb = TSNE(n_components=2, random_state=_SEED, init="pca",
                       perplexity=args.tsne_perplexity,
                       metric="cosine").fit_transform(allX)
            tsne_coords[mod] = {sp: emb[a:b] for sp, (a, b) in bounds.items()}
        methods.append("tsne")

    for split in args.splits:
        T, A, E, S, _ = data[split]
        split_res: Dict = {"n": int(len(E))}
        for mod, X in (("text", T), ("acoustic", A)):
            Xtr = Ttr if mod == "text" else Atr
            split_res[mod] = {
                "probe_emotion": probe(Xtr, Etr, X, E, EMOTION_NAMES),
                "probe_sentiment": probe(Xtr, Str, X, S, SENTIMENT_NAMES),
                "silhouette_emotion": silhouette(X, E, EMOTION_NAMES),
                "silhouette_sentiment": silhouette(X, S, SENTIMENT_NAMES),
            }
            logger.info(
                "%s | %-8s | probe emo WF1 %.4f macro %.4f | sent WF1 %.4f | "
                "sil emo %s sent %s",
                split, mod,
                split_res[mod]["probe_emotion"]["weighted_f1"],
                split_res[mod]["probe_emotion"]["macro_f1"],
                split_res[mod]["probe_sentiment"]["weighted_f1"],
                split_res[mod]["silhouette_emotion"]["overall"],
                split_res[mod]["silhouette_sentiment"]["overall"],
            )

            for meth in methods:
                if meth == "tsne":
                    coords, evr = tsne_coords[mod][split], None
                else:
                    transform, evr = reducers[mod][meth]
                    coords = transform(X)
                if meth == "pca":
                    split_res[mod][f"{meth}_explained_variance"] = evr
                for lbl, y, names in (("emotion", E, EMOTION_NAMES),
                                      ("sentiment", S, SENTIMENT_NAMES)):
                    plot_2d(coords, y, names,
                            f"{meth.upper()} · {mod} · {split} · {lbl}",
                            out / "figures" / f"{split}_{mod}_{meth}_{lbl}.png",
                            evr)
        results[split] = split_res

    with open(out / "input_representations.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Results → %s", out / "input_representations.json")
    logger.info("Figures → %s", out / "figures")


if __name__ == "__main__":
    main()
