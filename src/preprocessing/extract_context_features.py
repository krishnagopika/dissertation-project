"""Cache the bc-LSTM's CONTEXTUALISED representation, one vector per utterance.

Completes the set of representations the analysis compares:

    stage                     dim    source
    1  text (XLM-R [CLS])     768    {split}_text_embeddings.pt
    2  acoustic (masked mean) 1280   {split}_embeddings_maskedmean.pt
    3  fused                  512    extract_fused_features.py
    4  contextualised         512    THIS SCRIPT (plain bc-LSTM)
    5  stacked                512    THIS SCRIPT (bc-LSTM on fused input)

Stage 4 vs stages 1-2 isolates what dialogue context does to class structure.
Stage 5 vs stage 4 isolates what the learned fusion representation adds on top.

Why the vector differs from the fusion one
------------------------------------------
Utterance i's contextualised vector already contains information from its
neighbours -- that is the whole point of the model. So this representation is
NOT a property of the utterance alone, and a probe fitted on it is answering
"is the label recoverable given the utterance AND its context", which is a
different (and easier) question than the same probe on stage 1 or 2. Say so
when reporting: the probe numbers are comparable across stages only in the
sense that they measure linear recoverability of the same labels; the
information available differs by construction.

Padding
-------
The BiLSTM runs over padded dialogues, so ``represent()`` returns
``(B, T_max, 512)`` with meaningless rows past each dialogue's true length.
Only the first ``lengths[i]`` rows of row i are kept, and they are paired back
to utterance keys by the dataset's own dialogue ordering.

Usage
-----
    # stage 4 -- plain bc-LSTM
    python3.12 src/preprocessing/extract_context_features.py \\
        --config src/configs/ctx_gold.yaml \\
        --checkpoint /dcs/large/u5734759/checkpoints/bclstm/gold/k4/best_context.pt \\
        --out_dir /dcs/large/u5734759/data/meld_context/gold

    # stage 5 -- bc-LSTM trained on fused features
    python3.12 src/preprocessing/extract_context_features.py \\
        --config src/configs/ctx_gold.yaml \\
        --checkpoint .../bclstm/stacked_gold/k4/best_context.pt \\
        --fused_path /dcs/large/u5734759/data/meld_fused/gold \\
        --out_dir /dcs/large/u5734759/data/meld_context/stacked_gold
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

from src.models.context_lstm import BiLSTMContext
from src.training.train_context import DialogueDataset, collate
from src.utils import get_device, load_config, set_seed, setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Cache bc-LSTM contextualised representations.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True,
                    help="best_context.pt from a trained bc-LSTM run.")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fused_path", default=None,
                    help="Set for stage 5: the bc-LSTM was trained on fused "
                         "features, so the same input must be reconstructed.")
    ap.add_argument("--fused_mode", choices=["replace", "concat"],
                    default="replace")
    ap.add_argument("--model", choices=["bclstm", "ctxfusion"], default="bclstm",
                    help=("bclstm: BiLSTMContext over concatenated features "
                          "(stage 4). ctxfusion: ContextThenFusion, per-modality "
                          "BiLSTM then fuse (stage 5). The two consume the same "
                          "dataset but differ in forward signature, so the "
                          "checkpoint alone cannot disambiguate them."))
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--splits", nargs="+", default=["train", "test"],
                    choices=["train", "dev", "test"])
    args = ap.parse_args()

    config = load_config(args.config)
    set_seed(config["data"]["seed"])
    device = get_device()
    logger = setup_logging(config["training"]["log_dir"], "extract_context_features")

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)

    text_dim = config["model"]["text_dim"]
    acoustic_dim = config["model"]["acoustic_dim"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = None
    for split in args.splits:
        # No keep-list: the analysis wants every utterance, and filtered ones
        # are context for their neighbours regardless.
        ds = DialogueDataset(
            config["data"]["meld_root"], config["data"]["text_embeddings_path"],
            config["data"]["embeddings_path"], split, text_dim, acoustic_dim,
            filtered_keys_path=None,
            fused_path=args.fused_path, fused_mode=args.fused_mode)

        if model is None and args.model == "ctxfusion":
            from src.models.context_fusion import ContextThenFusion
            a = state.get("args", {})
            model = ContextThenFusion(
                text_dim=text_dim, acoustic_dim=acoustic_dim,
                hidden_dim=config["model"]["fusion_hidden"],
                lstm_hidden=a.get("lstm_hidden", 256),
                num_layers=a.get("num_layers", 1),
                num_emotion_classes=config["model"]["num_classes"],
                num_sentiment_classes=config["model"]["num_sentiment_classes"],
                dropout=config["model"]["dropout"],
                fusion=a.get("fusion", "concat"),
                use_text_lstm=a.get("use_text_lstm", True),
                use_acoustic_lstm=a.get("use_acoustic_lstm", True))
            model.load_state_dict(state["model_state_dict"])
            model = model.to(device).eval()
            logger.info("Loaded %s | ContextThenFusion a=%s t=%s | dev WF1 %.4f",
                        ckpt, a.get("use_acoustic_lstm"), a.get("use_text_lstm"),
                        state.get("emotion_weighted_f1", float("nan")))
        elif model is None:
            # Width comes from the data, exactly as in training -- stage 5's
            # input is 512 (or 2560 in concat mode), not 768+1280.
            # Derive the shapes from the WEIGHTS, not from config or defaults:
            #   lstm.weight_hh_l0 is (4*hidden, hidden)
            #   lstm.weight_ih_l0 is (4*hidden, input)
            # The stored "config" is the YAML, which has no hidden_dim, so a
            # .get(..., 256) there would silently be a guess that happens to be
            # right today and wrong the moment a run uses a different width.
            sd = state["model_state_dict"]
            hidden = sd["lstm.weight_hh_l0"].shape[1]
            ckpt_input = sd["lstm.weight_ih_l0"].shape[1]
            n_layers = sum(1 for k in sd if k.startswith("lstm.weight_ih_l")
                           and not k.endswith("_reverse"))
            if ckpt_input != ds.feature_dim:
                raise ValueError(
                    f"checkpoint expects input_dim={ckpt_input} but the dataset "
                    f"produces {ds.feature_dim}. Wrong --fused_path/--fused_mode "
                    "for this checkpoint.")
            model = BiLSTMContext(
                input_dim=ds.feature_dim, hidden_dim=hidden,
                num_emotion_classes=config["model"]["num_classes"],
                num_sentiment_classes=config["model"]["num_sentiment_classes"],
                num_layers=n_layers, dropout=config["model"]["dropout"],
            )
            model.load_state_dict(sd)
            model = model.to(device).eval()
            logger.info("Loaded %s | input_dim=%d hidden=%d layers=%d | dev WF1 %.4f",
                        ckpt, ds.feature_dim, hidden, n_layers,
                        state.get("emotion_weighted_f1", float("nan")))

        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate)
        feats: Dict[str, torch.Tensor] = {}
        # collate drops the keys, so walk the dataset in the SAME order the
        # loader does (shuffle=False) and pair by position.
        dia_iter = iter(ds.dialogues)
        with torch.no_grad():
            for f, _, _, lengths in loader:
                f = f.to(device)
                # The two models differ in signature: bc-LSTM takes the
                # concatenated feature, ContextThenFusion takes the two streams
                # apart because it runs a separate BiLSTM on each.
                rep = (model.represent(f[..., :text_dim], f[..., text_dim:],
                                       lengths).cpu()
                       if args.model == "ctxfusion"
                       else model.represent(f, lengths).cpu())
                for row, n in zip(rep, lengths.tolist()):
                    dia = next(dia_iter)
                    for key, vec in zip(dia["keys"], row[:n]):
                        feats[key] = vec.clone()

        path = out_dir / f"{split}_context.pt"
        torch.save(feats, str(path))
        logger.info("%s | %d vectors, %d-d -> %s", split, len(feats),
                    next(iter(feats.values())).shape[-1], path)

    with open(out_dir / "provenance.json", "w", encoding="utf-8") as f:
        json.dump({"checkpoint": str(ckpt), "config": args.config,
                   "fused_path": args.fused_path,
                   "fused_mode": args.fused_mode if args.fused_path else None,
                   "stage": 5 if args.model == "ctxfusion" else (5 if args.fused_path else 4),
                   "model": args.model,
                   "source_dev_weighted_f1": state.get("emotion_weighted_f1")},
                  f, indent=2)
    logger.info("Wrote provenance -> %s", out_dir / "provenance.json")


if __name__ == "__main__":
    main()
