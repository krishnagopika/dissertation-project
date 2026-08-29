"""Cache the fusion model's LEARNED representation, one 512-d vector per utterance.

Why
---
bc-LSTM currently reads a raw concatenation: XLM-R ``[CLS]`` (768) with a fixed
masked-mean acoustic vector (1280). The fusion model instead *learns* its
acoustic pooling and *learns* how to combine the modalities. Comparing the two
therefore conflates two differences — context modelling and representation
quality — and neither result isolates either.

Feeding the fusion model's own fused vector to bc-LSTM makes **context the only
difference** between them, and gives the stacked model the pipeline has been
building toward:

    acoustic (T,1280) ─┐ learned pooling
                       ├─► fusion ─► 512-d ─► bc-LSTM over dialogue ─► prediction
    text (768) ────────┘

What is extracted
-----------------
``SequenceFusion.represent()`` — the vector immediately before the
classification heads, verified bit-identical to what ``forward()`` feeds them.
The heads are discarded; only the representation is cached.

Every utterance, not the filtered subset
----------------------------------------
``filtered_keys_path`` is deliberately ``None``. bc-LSTM keeps filtered
utterances in the dialogue as *context* and masks only their labels, so it
needs a feature vector for every utterance the CSV lists — including ones a
keep-list would exclude from scoring.

Usage
-----
    python3.12 src/preprocessing/extract_fused_features.py \\
        --config src/configs/fusion_gold.yaml \\
        --checkpoint /dcs/large/u5734759/checkpoints/fusion/phase3_fusion/gold_sum/best_model.pt \\
        --out_dir /dcs/large/u5734759/data/meld_fused/gold
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
    ap = argparse.ArgumentParser(
        description="Cache fusion's fused representation per utterance.")
    ap.add_argument("--config", required=True,
                    help="The fusion config the checkpoint was trained with.")
    ap.add_argument("--checkpoint", required=True,
                    help="best_model.pt from a trained fusion run.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--splits", nargs="+", default=["train", "dev", "test"],
                    choices=["train", "dev", "test"])
    args = ap.parse_args()

    config = load_config(args.config)
    set_seed(config["data"]["seed"])
    device = get_device()
    logger = setup_logging(config["training"]["log_dir"], "extract_fused_features")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # Rebuild the model EXACTLY as trained. The checkpoint stores the argparse
    # namespace, so pooling/fusion/modality come from the run itself rather
    # than from assumptions -- loading a `sum` checkpoint into a `concat` model
    # would silently produce a differently-shaped `post` and a state_dict
    # mismatch, or worse, load and mean nothing.
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
        modality=targs.get("modality", "both"),
    )
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device).eval()
    logger.info("Loaded %s | pooling=%s fusion=%s modality=%s | dev WF1 %.4f",
                ckpt_path, targs.get("pooling"), targs.get("fusion"),
                targs.get("modality"), state.get("metric", float("nan")))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        ds = FusionSequenceDataset(
            meld_root=config["data"]["meld_root"],
            text_embeddings_path=config["data"]["text_embeddings_path"],
            acoustic_seq_path=config["data"]["embeddings_path"],
            split=split,
            text_dim=config["model"]["text_dim"],
            acoustic_dim=config["model"]["acoustic_dim"],
            # None on purpose -- see the module docstring.
            filtered_keys_path=None,
            logger=logger,
        )
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=config["data"]["num_workers"], pin_memory=True,
            collate_fn=collate_sequences)

        feats: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for batch in loader:
                rep = model.represent(
                    batch["text"].to(device),
                    batch["acoustic"].to(device),
                    batch["acoustic_mask"].to(device))
                for key, vec in zip(batch["keys"], rep.cpu()):
                    feats[key] = vec.clone()

        path = out_dir / f"{split}_fused.pt"
        torch.save(feats, str(path))
        dim = next(iter(feats.values())).shape[-1]
        logger.info("%s | %d vectors, %d-d -> %s", split, len(feats), dim, path)

    meta = {
        "checkpoint": str(ckpt_path),
        "config": args.config,
        "pooling": targs.get("pooling"),
        "fusion": targs.get("fusion"),
        "modality": targs.get("modality"),
        "source_dev_weighted_f1": state.get("metric"),
        "hidden_dim": config["model"]["fusion_hidden"],
    }
    with open(out_dir / "provenance.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote provenance -> %s", out_dir / "provenance.json")


if __name__ == "__main__":
    main()
