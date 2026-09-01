"""Train ContextThenFusion — per-modality dialogue context, then fusion.

Reuses ``DialogueDataset`` and ``collate`` from train_context.py unchanged, and
splits the concatenated ``(T, 768+1280)`` feature back into its two streams in
the training loop. That is deliberate: the dataset is the piece with the
loud-failure guards (missing embeddings masked and counted, unknown labels
raising, a 2% miss ceiling), and re-implementing it for a slightly different
output shape would mean maintaining those guards twice.

The 2x2 this exists for
-----------------------
    --use_acoustic_lstm / --use_text_lstm

    acoustic  text   what it isolates
    --------  -----  ----------------------------------
    on        on     both contextualised
    on        off    acoustic-only context
    off       on     text-only context
    off       off    no context -- the control

With both off the model reduces to per-utterance fusion, so any gain in the
other three arms is attributable to context alone.

Padding is handled by the loss, not the model: filtered and padded utterances
carry ``PAD_LABEL`` and are excluded via ``ignore_index``, while remaining
visible to their neighbours as context. That is the right treatment -- a
filtered utterance is still part of the conversation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import (EMOTION_NAMES, SENTIMENT_NAMES,
                                    compute_emotion_metrics,
                                    compute_sentiment_metrics, log_metrics)
from src.models.context_fusion import ContextThenFusion
from src.training.finetune import FocalLoss
from src.training.train_context import (PAD_LABEL, DialogueDataset,
                                        WindowedDialogueDataset, collate,
                                        masked_loss)
from src.utils import get_device, load_config, set_seed, setup_logging


def split_modalities(feats: torch.Tensor, text_dim: int
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split ``(B, T, text_dim + acoustic_dim)`` back into its two streams.

    DialogueDataset concatenates text and acoustic per utterance. This model
    needs them apart, because the whole point is a separate temporal model per
    modality. The split point is text_dim, matching the concatenation order in
    the dataset (``torch.cat([t, a])``).
    """
    return feats[..., :text_dim], feats[..., text_dim:]


def run_epoch(model, loader, device, text_dim, e_crit, s_crit,
              optimizer=None, max_grad_norm: float = 1.0) -> Tuple[float, Dict, Dict]:
    """One pass. Training when ``optimizer`` is given, evaluation otherwise.

    The loss is weighted by the number of VALID (non-PAD) utterances per batch,
    not averaged over batches. ``masked_loss`` already averages within a batch,
    so dividing by len(loader) would be a mean of means over unequal groups --
    and dialogue lengths vary enormously, so a batch of 16 short dialogues would
    carry the same weight as 16 long ones despite holding a fraction of the
    utterances. That makes loss curves incomparable across runs whose
    dialogue-length distribution differs, which is exactly what filtering does.
    """
    train = optimizer is not None
    model.train() if train else model.eval()
    total = torch.zeros((), device=device)
    n_utts = 0
    ep, el, sp, sl = [], [], [], []

    for feats, emos, sents, lengths in loader:
        feats = feats.to(device)
        emos_d, sents_d = emos.to(device), sents.to(device)
        text, acoustic = split_modalities(feats, text_dim)

        with torch.set_grad_enabled(train):
            s_logits, e_logits = model(text, acoustic, lengths)
            loss = (masked_loss(e_logits, emos_d, e_crit)
                    + masked_loss(s_logits, sents_d, s_crit))
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
        # Weight by valid utterances, counted on the EMOTION labels (both tasks
        # are masked together by construction; the metrics below mask each task
        # by its own labels regardless).
        n_valid = int((emos != PAD_LABEL).sum())
        total = total + loss.detach() * n_valid
        n_utts += n_valid

        if not train:
            e_flat = e_logits.argmax(-1).reshape(-1).cpu()
            s_flat = s_logits.argmax(-1).reshape(-1).cpu()
            emo_flat, sen_flat = emos.reshape(-1), sents.reshape(-1)
            # Each task masked by ITS OWN labels -- see train_context.py.
            ve, vs = emo_flat != PAD_LABEL, sen_flat != PAD_LABEL
            ep.extend(e_flat[ve].tolist()); el.extend(emo_flat[ve].tolist())
            sp.extend(s_flat[vs].tolist()); sl.extend(sen_flat[vs].tolist())

    mean = float(total) / max(n_utts, 1)
    if train:
        return mean, {}, {}
    return (mean, compute_emotion_metrics(ep, el),
            compute_sentiment_metrics(sp, sl))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train ContextThenFusion: per-modality context, then fusion.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default="", help="Path under checkpoint_dir.")
    ap.add_argument("--fusion", default="concat",
                    choices=list(ContextThenFusion.FUSIONS))
    ap.add_argument("--use_text_lstm", action="store_true", default=False)
    ap.add_argument("--use_acoustic_lstm", action="store_true", default=False)
    ap.add_argument("--lstm_hidden", type=int, default=256,
                    help="Per direction, per modality.")
    ap.add_argument("--num_layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16,
                    help="Dialogues per batch.")
    ap.add_argument("--lr", type=float, default=None,
                    help="Defaults to config training.phase2_lr.")
    ap.add_argument("--fused_path", default=None,
                    help="Directory with {split}_fused.pt. With "
                         "--fused_mode acoustic this replaces the acoustic "
                         "half only, so the per-modality BiLSTMs receive the "
                         "attention-pooled vector instead of the masked mean.")
    ap.add_argument("--fused_mode", choices=["replace", "concat", "acoustic"],
                    default="acoustic")
    ap.add_argument(
        "--context_window", type=int, default=-1,
        help=("If >= 0, restrict each utterance's context to +/-K neighbours. "
              "-1 (default) uses the whole dialogue. Note K only affects arms "
              "that HAVE a BiLSTM -- with --context_window on the `neither` "
              "arm the model has no temporal layer, so the window is inert and "
              "the run is a duplicate of the full-dialogue control."))
    args = ap.parse_args()

    config = load_config(args.config)
    set_seed(config["data"]["seed"])
    device = get_device()
    # The f-string always contains "ctxfusion_" and "_{fusion}", so it is always
    # truthy and an `or "ctxfusion_none"` fallback could never fire -- the
    # control arm would silently be named "ctxfusion__concat".
    arms = (("a" if args.use_acoustic_lstm else "")
            + ("t" if args.use_text_lstm else ""))
    tag = args.tag or f"ctxfusion_{arms or 'none'}_{args.fusion}"
    log_dir = Path(config["training"]["log_dir"]) / tag
    logger = setup_logging(str(log_dir), "train_context_fusion")

    text_dim = config["model"]["text_dim"]
    acoustic_dim = config["model"]["acoustic_dim"]
    filt = config.get("filtering", {})
    filter_on = bool(filt.get("enabled", False))
    keys = filt.get("keys_paths", {}) if filter_on else {}

    def make_ds(split, keep):
        return DialogueDataset(
            config["data"]["meld_root"], config["data"]["text_embeddings_path"],
            config["data"]["embeddings_path"], split, text_dim, acoustic_dim,
            filtered_keys_path=keep,
            fused_path=args.fused_path, fused_mode=args.fused_mode)

    # Dev is NEVER filtered: filtering it would change both the training data
    # and the early-stopping criterion, so a difference could not be attributed
    # to either, and dev would stop being a fixed yardstick across runs.
    train_ds = make_ds("train", keys.get("train"))
    dev_ds = make_ds("dev", None)
    logger.info("train dialogues %d | dev dialogues %d", len(train_ds), len(dev_ds))

    if args.context_window >= 0:
        K = args.context_window
        if not (args.use_text_lstm or args.use_acoustic_lstm):
            logger.warning(
                "context_window=%d requested but NEITHER BiLSTM is enabled -- "
                "there is no temporal layer for the window to restrict, so "
                "this run is identical to the full-dialogue control.", K)
        logger.info("context window K=%d (window size %d)", K, 2 * K + 1)
        train_ds = WindowedDialogueDataset(train_ds, K)
        dev_ds = WindowedDialogueDataset(dev_ds, K)
        logger.info("windowed | train examples %d | dev examples %d",
                    len(train_ds), len(dev_ds))

    import hashlib
    # Taken AFTER windowing: WindowedDialogueDataset scores only each window's
    # centre, so its key set is what was actually evaluated.
    dev_hash = hashlib.sha1(
        "".join(sorted(dev_ds.utterance_keys())).encode()).hexdigest()[:12]
    logger.info("dev key hash: %s", dev_hash)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        generator=torch.Generator().manual_seed(config["data"]["seed"]))
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate)

    # In acoustic-swap mode the acoustic half is the 512-d pooled vector, not
    # the 1280-d masked mean, so the model must be built for the real width --
    # split_modalities() slices at text_dim and hands the remainder to the
    # acoustic branch.
    eff_acoustic = (train_ds.feature_dim - text_dim
                    if args.fused_path else acoustic_dim)
    logger.info("acoustic branch width: %d (%s)", eff_acoustic,
                "attention-pooled" if args.fused_path else "masked mean")
    model = ContextThenFusion(
        text_dim=text_dim, acoustic_dim=eff_acoustic,
        hidden_dim=config["model"]["fusion_hidden"],
        lstm_hidden=args.lstm_hidden, num_layers=args.num_layers,
        num_emotion_classes=config["model"]["num_classes"],
        num_sentiment_classes=config["model"]["num_sentiment_classes"],
        dropout=config["model"]["dropout"], fusion=args.fusion,
        use_text_lstm=args.use_text_lstm,
        use_acoustic_lstm=args.use_acoustic_lstm,
    ).to(device)
    rep = model.parameter_report()
    logger.info("RUN %s | acoustic_lstm=%s text_lstm=%s fusion=%s | params %d",
                tag, args.use_acoustic_lstm, args.use_text_lstm, args.fusion,
                rep["TOTAL"])
    logger.info("parameters: %s", rep)

    lr = args.lr if args.lr is not None else float(config["training"]["phase2_lr"])
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=float(config["training"]["weight_decay"]))
    sched_patience = max(0, args.patience - 3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=sched_patience)
    gamma = float(config["training"].get("focal_gamma", 2.0))
    e_crit, s_crit = FocalLoss(gamma=gamma).to(device), FocalLoss(gamma=gamma).to(device)
    max_grad_norm = float(config["training"].get("max_grad_norm", 1.0))

    ckpt_dir = Path(config["training"]["checkpoint_dir"]) / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Explicit flag set AT the break. Recomputing the condition afterwards is
    # how the earlier (epoch+1) < epochs bug arose: that is also False when the
    # loop breaks on the final epoch, so it cannot distinguish "converged" from
    # "ran out of budget".
    best, since_best, best_epoch, epoch = 0.0, 0, None, -1
    stopped_early = False

    for epoch in range(args.epochs):
        tr, _, _ = run_epoch(model, train_loader, device, text_dim,
                             e_crit, s_crit, opt, max_grad_norm)
        vl, em, sm = run_epoch(model, dev_loader, device, text_dim, e_crit, s_crit)
        wf1 = em["weighted_f1"]
        logger.info("Epoch %d/%d | train %.4f | dev %.4f | emo WF1 %.4f macro %.4f "
                    "| sent WF1 %.4f", epoch + 1, args.epochs, tr, vl, wf1,
                    em["macro_f1"], sm["weighted_f1"])
        sched.step(wf1)
        if wf1 > best:
            best, since_best, best_epoch = wf1, 0, epoch
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "emotion_weighted_f1": wf1, "args": vars(args),
                        "parameter_report": rep}, ckpt_dir / "best_model.pt")
        else:
            since_best += 1
            if since_best >= args.patience:
                logger.info("Early stopping at epoch %d.", epoch + 1)
                stopped_early = True
                break

    if best_epoch is None:
        raise RuntimeError(
            f"No checkpoint written: dev WF1 never exceeded {best:.4f}. "
            "This is a failed run, not a completed one.")

    logger.info("Complete. Best dev emotion WF1: %.4f", best)
    with open(ckpt_dir / "TRAINING_COMPLETE.json", "w", encoding="utf-8") as f:
        json.dump({"model": "context_then_fusion", "tag": tag,
                   "best_dev_weighted_f1": best, "best_epoch": best_epoch,
                   "checkpoint_written": True,
                   "epochs_run": epoch + 1, "epochs_configured": args.epochs,
                   "early_stopped": stopped_early,
                   "use_text_lstm": args.use_text_lstm,
                   "use_acoustic_lstm": args.use_acoustic_lstm,
                   "fusion": args.fusion, "lstm_hidden": args.lstm_hidden,
                   "fused_path": args.fused_path,
                   "acoustic_pooling": ("attention" if args.fused_path
                                        else "masked_mean"),
                   "context_window": args.context_window,
                   "windowed": args.context_window >= 0,
                   "lr": lr, "batch_size": args.batch_size,
                   "seed": config["data"]["seed"], "dev_key_hash": dev_hash,
                   "filter_enabled": filter_on,
                   "train_dialogues": len(train_ds), "dev_dialogues": len(dev_ds),
                   "parameters": rep}, f, indent=2)


if __name__ == "__main__":
    main()
