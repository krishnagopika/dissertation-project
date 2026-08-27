"""
visualize_embeddings.py — t-SNE / PCA plots of MELD test embeddings.

Reads cached XLM-R [CLS] embeddings for the test split, applies the model's
final emotion classifier head to get predictions, reduces to 2D with t-SNE,
and produces three side-by-side scatter plots:

    (a) coloured by TRUE emotion
    (b) coloured by PREDICTED emotion
    (c) errors highlighted (correct = grey, incorrect = red, coloured by true)

Also produces a class-centroid distance heatmap showing which emotions
sit closest in embedding space (weakest separability).

Usage:
    python src/evaluation/visualize_embeddings.py \\
        --config src/configs/mini.yaml \\
        --checkpoint /dcs/large/u5734759/checkpoints/mini/best_model.pt \\
        --split test
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MELD_EMOTIONS: Dict[int, str] = {
    0: "neutral",
    1: "surprise",
    2: "fear",
    3: "sadness",
    4: "joy",
    5: "disgust",
    6: "anger",
}
EMOTION2IDX: Dict[str, int] = {v: k for k, v in MELD_EMOTIONS.items()}
EMOTION_COLORS: Dict[str, str] = {
    "neutral":  "#7f7f7f",
    "surprise": "#ff7f0e",
    "fear":     "#9467bd",
    "sadness":  "#1f77b4",
    "joy":      "#2ca02c",
    "disgust":  "#8c564b",
    "anger":    "#d62728",
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_embeddings(path: Path) -> Tuple[np.ndarray, List[str]]:
    """Return (N, 768) embedding matrix + list of N utterance keys."""
    d: Dict[str, torch.Tensor] = torch.load(path, map_location="cpu", weights_only=False)
    keys = sorted(d.keys())
    mat = torch.stack([d[k].float() for k in keys], dim=0).numpy()
    return mat, keys


def load_labels(meld_root: Path, split: str) -> Dict[str, Tuple[int, int]]:
    """Return dict[key -> (emotion_idx, sentiment_idx)]."""
    csv_map = {"train": "train_sent_emo.csv",
               "dev":   "dev_sent_emo.csv",
               "test":  "test_sent_emo.csv"}
    df = pd.read_csv(meld_root / csv_map[split])
    df.columns = [c.strip().lower() for c in df.columns]
    df["emotion"] = df["emotion"].astype(str).str.strip().str.lower()
    labels: Dict[str, Tuple[int, int]] = {}
    for _, row in df.iterrows():
        key = f"dia{int(row['dialogue_id'])}_utt{int(row['utterance_id'])}"
        labels[key] = EMOTION2IDX[row["emotion"]]
    return labels


def load_emotion_head(checkpoint_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return (W, b) for the emotion classifier head from a Phase 1 checkpoint."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = state.get("model_state_dict", state)
    w_key = next(k for k in sd if "emotion_head.weight" in k)
    b_key = next(k for k in sd if "emotion_head.bias" in k)
    return sd[w_key].float().numpy(), sd[b_key].float().numpy()


def predict(embeddings: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """embeddings (N,768) @ W.T (768,7) + b (7,) -> argmax -> (N,) int predictions."""
    logits = embeddings @ W.T + b
    return logits.argmax(axis=-1)


def tsne_2d(mat: np.ndarray, seed: int = 42) -> np.ndarray:
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000,
                init="pca", random_state=seed, learning_rate="auto")
    return tsne.fit_transform(mat)


def draw_scatter(ax, xy: np.ndarray, labels: np.ndarray, title: str,
                 alpha: float = 0.6, size: int = 8) -> None:
    for idx in range(7):
        name = MELD_EMOTIONS[idx]
        mask = labels == idx
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   c=EMOTION_COLORS[name], label=name,
                   s=size, alpha=alpha, edgecolors="none")
    ax.set_title(title, fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")


def draw_errors(ax, xy: np.ndarray, true_y: np.ndarray, pred_y: np.ndarray,
                title: str) -> None:
    correct = true_y == pred_y
    ax.scatter(xy[correct, 0], xy[correct, 1],
               c="#d0d0d0", s=6, alpha=0.4, edgecolors="none", label="correct")
    for idx in range(7):
        name = MELD_EMOTIONS[idx]
        mask = (~correct) & (true_y == idx)
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   c=EMOTION_COLORS[name], s=14, alpha=0.9,
                   edgecolors="black", linewidths=0.3,
                   label=f"error ({name})")
    ax.set_title(title, fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")


def centroid_heatmap(mat: np.ndarray, true_y: np.ndarray, out_path: Path) -> None:
    """Cosine distance between per-emotion centroids in the 768-dim space."""
    centroids = np.zeros((7, mat.shape[1]))
    counts = np.zeros(7)
    for i in range(len(true_y)):
        centroids[true_y[i]] += mat[i]
        counts[true_y[i]] += 1
    centroids /= counts[:, None].clip(min=1)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    normed = centroids / norms.clip(min=1e-9)
    sim = normed @ normed.T
    dist = 1.0 - sim

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(dist, cmap="viridis", vmin=0)
    ax.set_xticks(range(7))
    ax.set_yticks(range(7))
    names = [MELD_EMOTIONS[i] for i in range(7)]
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{dist[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if dist[i, j] < 0.15 else "black",
                    fontsize=9)
    ax.set_title("Cosine distance between emotion centroids\n"
                 "(768-dim XLM-R embedding space)", fontsize=12)
    fig.colorbar(im, ax=ax, label="1 - cosine similarity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", default=None,
                   help="Phase 1 XLM-R best_model.pt (required for text modality)")
    p.add_argument("--modality", default="text",
                   choices=["text", "acoustic"],
                   help="Which embedding space to visualise")
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p.add_argument("--tag", default="", help="Suffix for output file names")
    p.add_argument("--sample", type=int, default=0,
                   help="If >0, subsample this many points (t-SNE speed)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("viz")

    cfg = load_config(args.config)
    meld_root = Path(cfg["data"]["meld_root"])
    emb_dir = Path(cfg["data"].get("text_embeddings_path",
                                   "/dcs/large/u5734759/data/meld_text_embeddings"))
    out_dir = Path(cfg["training"].get("results_dir", "results/mini"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.modality == "text":
        emb_root = emb_dir
        emb_path = emb_root / f"{args.split}_text_embeddings.pt"
    else:
        emb_root = Path(cfg["data"].get(
            "embeddings_path",
            "/dcs/large/u5734759/data/meld_embeddings"))
        emb_path = emb_root / f"{args.split}_embeddings.pt"
    log.info("Loading %s embeddings from %s", args.modality, emb_path)
    mat, keys = load_embeddings(emb_path)
    log.info("  loaded %d embeddings of dim %d", mat.shape[0], mat.shape[1])

    log.info("Loading labels from %s", meld_root)
    label_map = load_labels(meld_root, args.split)
    kept_indices = [i for i, k in enumerate(keys) if k in label_map]
    keys_kept = [keys[i] for i in kept_indices]
    mat = mat[kept_indices]
    true_y = np.array([label_map[k] for k in keys_kept], dtype=np.int64)
    log.info("  aligned %d embeddings with labels", len(keys_kept))

    if args.sample > 0 and args.sample < len(keys_kept):
        rng = np.random.default_rng(42)
        sel = rng.choice(len(keys_kept), size=args.sample, replace=False)
        mat = mat[sel]
        true_y = true_y[sel]
        keys_kept = [keys_kept[i] for i in sel]
        log.info("  subsampled to %d points", len(keys_kept))

    pred_y = None
    acc = None
    if args.modality == "text" and args.checkpoint:
        log.info("Loading emotion head from %s", args.checkpoint)
        W, b = load_emotion_head(Path(args.checkpoint))
        pred_y = predict(mat, W, b)
        acc = float((pred_y == true_y).mean())
        log.info("  head accuracy on %s: %.4f (%d/%d)",
                 args.split, acc, (pred_y == true_y).sum(), len(true_y))
    else:
        log.info("Acoustic modality — skipping classifier "
                 "(no linear-only head on raw Voxtral features)")

    log.info("Running t-SNE (may take ~1 min for 2610 points)")
    xy = tsne_2d(mat)

    filter_keys_path = Path(
        cfg["data"].get("filter_keys_dir",
                        "/dcs/large/u5734759/data/meld_filter_keys")
    ) / f"{args.split}_filtered_keys_wer25.json"
    kept_mask = None
    if filter_keys_path.exists():
        with open(filter_keys_path) as f:
            kept_set = set(json.load(f)["keys"])
        kept_mask = np.array([k in kept_set for k in keys_kept], dtype=bool)
        log.info("  filter overlay: %d / %d kept (wer25)",
                 int(kept_mask.sum()), len(kept_mask))

    log.info("Drawing panel scatter")
    if pred_y is not None:
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        draw_scatter(axes[0, 0], xy, true_y, "(a) True emotion labels")
        draw_scatter(axes[0, 1], xy, pred_y, "(b) XLM-R predicted labels")
        draw_errors(axes[1, 0], xy, true_y, pred_y,
                    f"(c) Errors highlighted  (acc={acc:.1%})")
        fltr_ax = axes[1, 1]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        draw_scatter(axes[0], xy, true_y,
                     f"(a) True emotion labels (Voxtral 1280-dim acoustic)")
        fltr_ax = axes[1]

    if kept_mask is not None:
        fltr_ax.scatter(xy[~kept_mask, 0], xy[~kept_mask, 1],
                        c="#d62728", s=8, alpha=0.35, edgecolors="none",
                        label=f"excluded ({(~kept_mask).sum()})")
        fltr_ax.scatter(xy[kept_mask, 0], xy[kept_mask, 1],
                        c="#2ca02c", s=8, alpha=0.7, edgecolors="none",
                        label=f"kept ({kept_mask.sum()})")
        panel_letter = "(d)" if pred_y is not None else "(b)"
        fltr_ax.set_title(
            f"{panel_letter} WER25 filter status  "
            f"(kept={int(kept_mask.sum())}, excluded={int((~kept_mask).sum())})",
            fontsize=13)
        fltr_ax.set_xticks([]); fltr_ax.set_yticks([])
        fltr_ax.set_aspect("equal")
        fltr_ax.legend(loc="upper right", frameon=False)
    else:
        fltr_ax.text(0.5, 0.5, "no filter keys found",
                     ha="center", va="center", transform=fltr_ax.transAxes)
        fltr_ax.set_xticks([]); fltr_ax.set_yticks([])

    handles = [Line2D([], [], marker="o", linestyle="",
                      markerfacecolor=EMOTION_COLORS[MELD_EMOTIONS[i]],
                      markeredgecolor="none", markersize=8,
                      label=MELD_EMOTIONS[i])
               for i in range(7)]
    fig.legend(handles=handles, loc="lower center", ncol=7,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    dim_str = "768-dim [CLS]" if args.modality == "text" else "1280-dim Voxtral encoder"
    fig.suptitle(
        f"{args.modality.title()} embeddings on MELD {args.split} split "
        f"(t-SNE 2D projection of {dim_str})",
        fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    scatter_path = out_dir / f"tsne_{args.modality}_{args.split}{args.tag}.png"
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  wrote %s", scatter_path)

    heatmap_path = out_dir / f"centroid_distance_{args.modality}_{args.split}{args.tag}.png"
    centroid_heatmap(mat, true_y, heatmap_path)
    log.info("  wrote %s", heatmap_path)

    summary = {
        "modality": args.modality,
        "split": args.split,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "n_points": int(len(true_y)),
        "head_accuracy": acc,
        "per_class_counts": {MELD_EMOTIONS[i]: int((true_y == i).sum())
                             for i in range(7)},
        "per_class_recall": {MELD_EMOTIONS[i]: float(
            ((pred_y == i) & (true_y == i)).sum() / max(1, (true_y == i).sum()))
            for i in range(7)} if pred_y is not None else None,
    }
    summary_path = out_dir / f"tsne_summary_{args.modality}_{args.split}{args.tag}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("  wrote %s", summary_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
