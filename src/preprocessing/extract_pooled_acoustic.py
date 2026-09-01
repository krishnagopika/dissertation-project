"""Cache the RAW attention-pooled acoustic vector, before any projection.

Why this exists
---------------
``SequenceFusion.represent()`` returns the vector the classification heads see,
which is post-projection::

    (T, 1280) frames -> pooler -> 1280 -> acoustic_proj -> 512 -> post -> 512

So a cache built from ``represent()`` bundles the POOLER with a learned
1280->512 projection trained for fusion's objective. Comparing it against the
raw masked mean therefore measures both at once, and the `learned masked_mean`
control (cosine gap 0.0657 against attention's 0.0779) shows the projection
accounts for most of that gain.

This script hooks the pooler directly and stores its 1280-d output, so the
comparison against ``{split}_embeddings_maskedmean.pt`` is like-for-like: same
width, same frames, only the pooling rule differs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.fusion_seq import SequenceFusion
from src.training.fusion_data import FusionSequenceDataset, collate_sequences
from src.utils import get_device, load_config, set_seed, setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache raw pooled acoustic vectors.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    config = load_config(args.config)
    set_seed(config["data"]["seed"])
    device = get_device()
    logger = setup_logging(config["training"]["log_dir"], "extract_pooled_acoustic")

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    targs = state.get("args", {})
    model = SequenceFusion(
        acoustic_dim=config["model"]["acoustic_dim"],
        text_dim=config["model"]["text_dim"],
        hidden_dim=config["model"]["fusion_hidden"],
        num_emotion_classes=config["model"]["num_classes"],
        num_sentiment_classes=config["model"]["num_sentiment_classes"],
        dropout=config["model"]["dropout"],
        pooling=targs.get("pooling", "attention"),
        fusion=targs.get("fusion", "concat"),
        modality=targs.get("modality", "acoustic"))
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device).eval()
    logger.info("pooler=%s output_dim=%d (attentive_stats emits 2*acoustic_dim)",
                targs.get("pooling"), model.pooler.output_dim)

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        ds = FusionSequenceDataset(
            meld_root=config["data"]["meld_root"],
            text_embeddings_path=config["data"]["text_embeddings_path"],
            acoustic_seq_path=config["data"]["embeddings_path"],
            split=split, text_dim=config["model"]["text_dim"],
            acoustic_dim=config["model"]["acoustic_dim"],
            filtered_keys_path=None, logger=logger)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_sequences)
        feats: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for b in loader:
                # The pooler ALONE -- no acoustic_proj, no post.
                pooled, _ = model.pooler(b["acoustic"].to(device),
                                         b["acoustic_mask"].to(device))
                for k, v in zip(b["keys"], pooled.cpu()):
                    feats[k] = v.clone()
        p = out / f"{split}_fused.pt"
        torch.save(feats, str(p))
        logger.info("%s | %d vectors, %d-d -> %s", split, len(feats),
                    next(iter(feats.values())).shape[-1], p)

    with open(out / "provenance.json", "w", encoding="utf-8") as f:
        json.dump({"checkpoint": args.checkpoint, "pooling": targs.get("pooling"),
                   "stage": "pooler output, PRE-projection",
                   "pooler_output_dim": model.pooler.output_dim}, f, indent=2)


if __name__ == "__main__":
    main()
