"""
extract_acoustic_probs.py — acoustic-classifier per-class probabilities
========================================================================
Loads a fine-tuned VoxtralEncoderClassifier checkpoint and saves per-utterance
softmax probabilities (emotion 7, sentiment 3) for dev + test, keyed by
"dia{d}_utt{u}", for late fusion. Eval only — no training.

Output (consumed by src/evaluation/late_fusion.py):
  results/<run>/acoustic_probs.json

Usage
-----
  python3.12 src/evaluation/extract_acoustic_probs.py --config src/configs/mini.yaml \\
      --checkpoint_path /dcs/large/u5734759/checkpoints/mini/encoder_finetune/best_encoder_full.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.voxtral_encoder_classifier import VoxtralEncoderClassifier
from src.training.finetune_encoder import MeldAudioDataset, TARGET_SR
from src.utils import get_device, load_config, set_seed, setup_logging


def keyed_collate(batch: List[Dict], feature_extractor):
    """Collate to (input_features, keys) — keeps utterance keys for alignment."""
    audios = [b["waveform"].float().numpy() for b in batch]
    fe = feature_extractor(
        audios, sampling_rate=TARGET_SR, return_tensors="pt",
        padding="max_length", truncation=True,
    )
    keys = [b["key"] for b in batch]
    return fe["input_features"], keys


@torch.no_grad()
def extract_split(model, loader, device) -> Dict[str, Dict]:
    """Return {key: {emotion_probs, sentiment_probs}} for one split."""
    model.eval()
    out: Dict[str, Dict] = {}
    for input_features, keys in loader:
        input_features = input_features.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            sentiment_logits, emotion_logits = model(input_features)
        e = F.softmax(emotion_logits.float(), dim=-1).cpu().numpy()
        s = F.softmax(sentiment_logits.float(), dim=-1).cpu().numpy()
        for i, key in enumerate(keys):
            out[key] = {"emotion_probs": e[i].tolist(), "sentiment_probs": s[i].tolist()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract acoustic-classifier probabilities (eval only).")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to a VoxtralEncoderClassifier checkpoint (best_encoder_*.pt).")
    parser.add_argument("--splits", nargs="+", default=["dev", "test"],
                        choices=["train", "dev", "test"])
    parser.add_argument("--out_name", type=str, default="acoustic_probs.json",
                        help="Output filename under the results dir.")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["data"]["seed"])
    device = get_device()
    logger = setup_logging(config["training"]["log_dir"], "extract_acoustic_probs")

    ckpt_path = Path(args.checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = VoxtralEncoderClassifier(
        voxtral_id=config["model"]["voxtral_id"],
        num_emotion_classes=config["model"]["num_classes"],
        num_sentiment_classes=config["model"]["num_sentiment_classes"],
        acoustic_dim=config["model"]["acoustic_dim"],
        hidden_dim=config["model"]["fusion_hidden"],
        dropout_prob=config["model"]["dropout"],
        unfreeze_last_n=0,  # eval only — architecture must match, freezing is irrelevant
    )
    state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    logger.info("Loaded acoustic checkpoint: %s (dev WF1=%.4f)",
                ckpt_path, state.get("emotion_weighted_f1", float("nan")))

    processor = AutoProcessor.from_pretrained(config["model"]["voxtral_id"])
    collate_fn = partial(keyed_collate, feature_extractor=processor.feature_extractor)

    result: Dict[str, Dict] = {}
    for split in args.splits:
        ds = MeldAudioDataset(config["data"]["meld_root"], split,
                              float(config["data"]["max_audio_duration"]))
        loader = DataLoader(
            ds, batch_size=config["training"]["batch_size"], shuffle=False,
            num_workers=config["data"]["num_workers"], pin_memory=True, collate_fn=collate_fn,
        )
        result[split] = extract_split(model, loader, device)
        logger.info("Split '%s': %d utterances", split, len(result[split]))

    out_path = Path(config["evaluation"]["output_dir"]) / args.out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    logger.info("Saved acoustic probabilities → %s", out_path)


if __name__ == "__main__":
    main()
