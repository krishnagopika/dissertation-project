#!/usr/bin/env python3.12
"""Evaluate an ALREADY-TRAINED bc-LSTM checkpoint on a config's test split.

Why this exists: train_context.py always trains before it scores, so asking it
for "model X on test set Y" silently retrained X. Comparing one model across
two test sets was therefore impossible -- each cell was a different model with
a different random init and a different best epoch.

This script trains nothing. It loads a checkpoint, builds the test split the
config asks for (filtered or full, via filtering.enabled + keys_paths.test),
scores it, and writes the same JSON shape train_context.py produces.

Usage:
    python3.12 src/evaluation/eval_context.py \
        --config src/configs/eval_testwer25.yaml \
        --checkpoint /dcs/large/.../context_bclstm_trainfull/best_context.pt \
        --tag _trainfull_on_testwer25
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import log_metrics
from src.models.context_lstm import BiLSTMContext
from src.training.train_context import (
    DialogueDataset,
    WindowedDialogueDataset,
    collate,
    evaluate,
)
from src.utils import load_config, get_device, set_seed, setup_logging
from torch.utils.data import DataLoader


class _NoopLoss(torch.nn.Module):
    """evaluate() needs loss criteria, but eval-only runs ignore the value."""

    def forward(self, logits, labels):                    # noqa: D102
        return torch.zeros((), device=logits.device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--context_window", type=int, default=-1)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["data"]["seed"])
    device = get_device()
    logger = setup_logging(config["training"]["log_dir"], f"eval_context{args.tag}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    filt = config.get("filtering", {})
    filter_on = bool(filt.get("enabled", False))
    filter_keys = filt.get("keys_paths", {}) if filter_on else {}
    test_filter = filter_keys.get("test")

    logger.info("Config: %s", args.config)
    logger.info("Checkpoint: %s", ckpt_path)
    logger.info("Test filter: %s", test_filter or "NONE (full test set)")

    test_ds = DialogueDataset(
        config["data"]["meld_root"],
        config["data"]["text_embeddings_path"],
        config["data"]["embeddings_path"],
        "test",
        config["model"]["text_dim"],
        config["model"]["acoustic_dim"],
        filtered_keys_path=test_filter,
    )
    n_scored = test_ds.num_kept_utts if test_filter else test_ds.num_total_utts
    logger.info("Test dialogues: %d | utterances scored: %d / %d",
                len(test_ds), n_scored, test_ds.num_total_utts)

    if args.context_window >= 0:
        test_ds = WindowedDialogueDataset(test_ds, args.context_window)

    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collate)

    ckpt = torch.load(ckpt_path, map_location=device)
    # Rebuild with the geometry the checkpoint was trained with, not the
    # config's -- a mismatch would otherwise fail load_state_dict or, worse,
    # load a differently-shaped model that silently scores nonsense.
    sd = ckpt["model_state_dict"]
    input_dim = sd["lstm.weight_ih_l0"].shape[1]
    hidden_dim = sd["lstm.weight_hh_l0"].shape[1]
    # Count forward layers only. A bidirectional LSTM also emits
    # "lstm.weight_ih_l0_reverse", so keying on the last character would count
    # "e" as a layer index and inflate num_layers.
    num_layers = len({k for k in sd
                      if k.startswith("lstm.weight_ih_l") and not k.endswith("_reverse")})
    logger.info("Checkpoint geometry: input_dim=%d hidden_dim=%d num_layers=%d (epoch %d)",
                input_dim, hidden_dim, num_layers, ckpt.get("epoch", -1) + 1)

    model = BiLSTMContext(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_sentiment_classes=config["model"]["num_sentiment_classes"],
        num_emotion_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
    ).to(device)
    model.load_state_dict(sd)
    model.eval()

    noop = _NoopLoss().to(device)
    _, em, sm = evaluate(model, test_loader, device, noop, noop)

    logger.info("=== TEST | %s ===", args.tag or ckpt_path.name)
    log_metrics(em, "test", "emotion", logger)
    log_metrics(sm, "test", "sentiment", logger)

    def _drop_report(d: Dict) -> Dict:
        return {k: v for k, v in d.items() if k != "report"}

    output_dir = Path(config["evaluation"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "config": args.config,
        "model": "bclstm",
        "tag": args.tag,
        "checkpoint": str(ckpt_path),
        "trained_epoch": ckpt.get("epoch", -1) + 1,
        "eval_only": True,
        "filter_enabled": filter_on,
        "test_filter_keys": test_filter,
        "n_test_dialogues": len(test_ds),
        "n_test_utterances_scored": n_scored,
        "emotion": _drop_report(em),
        "sentiment": _drop_report(sm),
    }
    out = output_dir / f"test_results_bclstm{args.tag}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Results → %s", out)


if __name__ == "__main__":
    main()
