#!/usr/bin/env python3.12
"""Stage the two trained checkpoints the Modal endpoint needs, and price them.

`serve_emotion.py` reads exactly two local artefacts from its weights volume:

    /weights/xlmr.pt    the Phase-1 XLM-R encoder for the served condition
    /weights/head.pt    the trained ContextThenFusion head

This writes both into a staging directory and prints the `modal volume put`
commands. It never touches the originals.

Why staging rather than uploading `best_model.pt` directly
----------------------------------------------------------
An XLM-R `best_model.pt` is one part weights to two parts Adam moments: 3.33 GB
on disk, of which 1.11 GB is the model. The optimizer state exists to resume
training and is dead weight in a serving image -- it would triple the upload,
triple the volume footprint, and triple what every cold start reads back.
`src/scripts/strip_optimizer_state.py` does the same job in place, with a
verify-then-replace contract; this does it as a copy, because the training
checkpoints should survive the deployment.

Usage
-----
    python3.12 src/modal/prepare_checkpoints.py --out /tmp/modal_weights
    python3.12 src/modal/prepare_checkpoints.py --out /tmp/modal_weights \\
        --condition asr --head ctxfusion/asr/both_k0
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

#: Dropped from the XLM-R checkpoint. Everything else is carried through, so a
#: checkpoint that names its weights something unexpected is not silently
#: emptied -- the same allow-nothing/deny-one rule as strip_optimizer_state.py.
_DROP = "optimizer_state_dict"

#: Which fine-tuned encoder produced the text embeddings each head was trained
#: on. From src/scripts/text_embed_v2.sbatch: `plain` wins every condition on
#: dev weighted F1, so the loss is held constant across conditions.
ENCODER_FOR_CONDITION = {
    "gold": "xlmr_gold_plain",
    "asr": "xlmr_asr_plain",
    "asr_cleaned": "xlmr_asr_cleaned_plain",
}


def stage_xlmr(src: Path, dst: Path) -> tuple[float, float]:
    """Copy an XLM-R checkpoint without its optimizer state.

    Args:
        src: Path to a Phase-1 ``best_model.pt``.
        dst: Where to write the stripped copy.

    Returns:
        Tuple of (source GB, staged GB).

    Raises:
        FileNotFoundError: If ``src`` does not exist.
        KeyError: If the checkpoint carries no ``model_state_dict``.
    """
    if not src.exists():
        raise FileNotFoundError(f"checkpoint not found: {src}")

    state = torch.load(src, map_location="cpu", weights_only=False)
    if "model_state_dict" not in state:
        raise KeyError(f"{src} has no model_state_dict; keys: {list(state)}")

    torch.save({k: v for k, v in state.items() if k != _DROP}, dst)
    return src.stat().st_size / 1e9, dst.stat().st_size / 1e9


def stage_head(src: Path, dst: Path) -> float:
    """Copy a trained fusion head verbatim.

    The head is ~25 MB and carries no optimizer state, so this is a plain copy
    -- but it goes through torch rather than the filesystem so a corrupt
    checkpoint is caught here rather than on the first cold start.

    Args:
        src: Path to a ctxfusion ``best_model.pt``.
        dst: Where to write it.

    Returns:
        Size in MB.

    Raises:
        FileNotFoundError: If ``src`` does not exist.
    """
    if not src.exists():
        raise FileNotFoundError(f"checkpoint not found: {src}")

    state = torch.load(src, map_location="cpu", weights_only=False)
    print(f"  head: {state['args']['tag']} | "
          f"fusion={state['args']['fusion']} "
          f"text_lstm={state['args']['use_text_lstm']} "
          f"acoustic_lstm={state['args']['use_acoustic_lstm']} | "
          f"dev emotion WF1 {state['emotion_weighted_f1']:.4f}")
    torch.save(state, dst)
    return dst.stat().st_size / 1e6


def main() -> None:
    """Stage both artefacts and print the upload commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, required=True,
                        help="Staging directory for the upload.")
    parser.add_argument("--checkpoint-root", type=str,
                        default=os.environ.get(
                            "CKPT_ROOT", "/dcs/large/u5734759/checkpoints"),
                        help="Root of the trained checkpoints.")
    parser.add_argument("--condition", type=str, default="asr_cleaned",
                        choices=sorted(ENCODER_FOR_CONDITION),
                        help="Text condition; selects the XLM-R encoder.")
    parser.add_argument("--head", type=str,
                        default="ctxfusion/asr_cleaned/both_k0",
                        help="Head checkpoint, relative to --checkpoint-root. "
                             "Must have been trained on --condition's cache.")
    parser.add_argument("--volume", type=str, default="emotion-weights",
                        help="Modal volume name, matching serve_emotion.py.")
    args = parser.parse_args()

    root = Path(args.checkpoint_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    encoder = ENCODER_FOR_CONDITION[args.condition]
    print(f"condition {args.condition} -> encoder {encoder}")

    src_gb, dst_gb = stage_xlmr(root / encoder / "best_model.pt", out / "xlmr.pt")
    print(f"  xlmr.pt: {src_gb:.2f} GB -> {dst_gb:.2f} GB "
          f"({src_gb - dst_gb:.2f} GB of Adam state dropped)")

    head_mb = stage_head(root / args.head / "best_model.pt", out / "head.pt")
    print(f"  head.pt: {head_mb:.1f} MB")

    print("\nUpload:")
    print(f"  modal volume put {args.volume} {out / 'xlmr.pt'} /xlmr.pt")
    print(f"  modal volume put {args.volume} {out / 'head.pt'} /head.pt")


if __name__ == "__main__":
    main()
