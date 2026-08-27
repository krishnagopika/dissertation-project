#!/usr/bin/env python3.12
"""Train a SequenceFusion model: acoustic sequences + frozen text embeddings.

One script covers all three ablations via `--modality`:

    acoustic   pooler + head, no text        -- isolates pooling, no text confound
    text       frozen [CLS] + head, no audio -- the fair unimodal text baseline
    both       fusion                        -- the model under test

Voxtral and XLM-R are never loaded: this trains ~2M new parameters on cached
representations (CLAUDE.md §16). Everything inside SequenceFusion is
Xavier-initialised and trained from scratch.

Discipline carried over from finetune.py, each for a reason:

  * **No automatic resume.** Resume restores best_metric from the LATEST
    checkpoint rather than the best one, so a later inferior epoch can overwrite
    a better best_model.pt; and it rebuilds the LR schedule from step 0 while
    the optimizer continues.
  * **No per-epoch checkpoint retention.** Keeping one cost 3.3 GB per run in
    the XLM-R grid and exhausted the disk quota mid-experiment.
  * **A completion marker**, written only on clean exit. best_model.pt appears
    at the first improving epoch, so its presence does not mean a run finished.
  * **Early stopping on dev weighted F1**, not loss: under 17.6:1 imbalance loss
    can improve while minority-class F1 degrades.

This script touches TRAIN and DEV only. The selected checkpoint still requires a
separate pass over TEST -- see src/scripts/fusion_grid.sbatch, which scores every
completed run after training. best_model.pt carries no optimizer state, so a run
that early-stopped cannot be continued; it can only be re-run.

Usage:
    python3.12 src/training/train_fusion_seq.py \\
        --config src/configs/fusion_asr.yaml \\
        --modality both --pooling masked_mean --fusion concat --loss weighted
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import (EMOTION_NAMES, SENTIMENT_NAMES,
                                    compute_metrics)
from src.models.fusion_seq import SequenceFusion
from functools import partial

from src.training.fusion_data import FusionSequenceDataset, collate_sequences
from src.utils import get_device, load_config, set_seed, setup_logging


class FocalLoss(nn.Module):
    """Focal loss with optional per-class alpha (Lin et al., 2017).

    ``-alpha_t (1 - p_t)^gamma log p_t``. Reduces to weighted cross-entropy at
    ``gamma == 0``.
    """

    def __init__(self, weight: Optional[Tensor] = None, gamma: float = 2.0) -> None:
        super().__init__()
        # Registered as a buffer on CPU; a single .to(device) on the module then
        # moves it. Registering an already-moved tensor works but makes the
        # placement semantics depend on construction order.
        self.register_buffer("weight",
                             None if weight is None else weight.detach().cpu())
        self.gamma = gamma

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:   # noqa: D102
        if self.weight is not None and logits.size(1) != self.weight.numel():
            # Without this, a mismatch surfaces as a device-side assert inside
            # gather() with no usable traceback.
            raise ValueError(
                f"FocalLoss weight has {self.weight.numel()} entries but logits "
                f"have {logits.size(1)} classes."
            )
        logp = torch.log_softmax(logits, dim=-1)
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1.0 - p_t) ** self.gamma) * logp_t
        if self.weight is not None:
            loss = loss * self.weight.gather(0, target)
        return loss.mean()


def class_weights(counts: Counter, names, device) -> Tensor:
    """Inverse-frequency weights: ``N / (C * count_c)``."""
    c = torch.tensor([max(1, counts.get(n, 0)) for n in names],
                     dtype=torch.float32)
    return c.sum() / (len(names) * c)          # CPU; caller moves it


def build_criteria(loss_name: str, train_ds, device, gamma: float, logger):
    """Construct emotion and sentiment criteria for the chosen loss variant."""
    emo_counts = train_ds.label_counts()
    sen_counts = Counter(SENTIMENT_NAMES[s[4]] for s in train_ds.samples)

    if loss_name == "plain":
        we = ws = None
        logger.info("loss=plain — unweighted CrossEntropy (no class alpha)")
    else:
        we = class_weights(emo_counts, EMOTION_NAMES, device)
        ws = class_weights(sen_counts, SENTIMENT_NAMES, device)
        logger.info("emotion class weights: %s",
                    [f"{w:.3f}" for w in we.cpu().tolist()])

    if loss_name == "focal":
        logger.info("loss=focal — FocalLoss(gamma=%.1f) with the same alpha as "
                    "'weighted', so weighted->focal isolates gamma", gamma)
        return (FocalLoss(we, gamma).to(device), FocalLoss(ws, gamma).to(device))
    return (nn.CrossEntropyLoss(weight=None if we is None else we.to(device)),
            nn.CrossEntropyLoss(weight=None if ws is None else ws.to(device)))


@torch.no_grad()
def evaluate(model, loader, device, crit_e, crit_s) -> Tuple[float, Dict, Dict]:
    """Score a loader; return (mean loss, emotion metrics, sentiment metrics)."""
    model.eval()
    tot, n = 0.0, 0
    pe, le, ps, ls = [], [], [], []
    for b in loader:
        a = b["acoustic"].to(device)
        m = b["acoustic_mask"].to(device)
        t = b["text"].to(device)
        ye = b["emotion_label"].to(device)
        ys = b["sentiment_label"].to(device)
        e_log, s_log, _ = model(t, a, m)
        tot += float(crit_e(e_log, ye) + crit_s(s_log, ys)) * len(ye)
        n += len(ye)
        pe += e_log.argmax(-1).cpu().tolist(); le += ye.cpu().tolist()
        ps += s_log.argmax(-1).cpu().tolist(); ls += ys.cpu().tolist()
    return (tot / max(1, n),
            compute_metrics(pe, le, EMOTION_NAMES),
            compute_metrics(ps, ls, SENTIMENT_NAMES))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--modality", default="both",
                    choices=["both", "acoustic", "text"])
    ap.add_argument("--pooling", default="masked_mean",
                    choices=["masked_mean", "attention_fixed", "attention",
                             "attentive_stats"])
    ap.add_argument("--fusion", default="concat",
                    choices=["concat", "sum", "gated", "crossmodal"])
    ap.add_argument("--loss", default="weighted",
                    choices=["plain", "weighted", "focal"])
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lambda_sentiment", type=float, default=1.0,
                    help="Weight on the sentiment loss. 1.0 = equal, untuned.")
    ap.add_argument("--max_frames", type=int, default=1500)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    if args.epochs < 1:
        # Otherwise the loop never runs and a completion marker is written
        # claiming a successful run that trained nothing.
        ap.error("--epochs must be >= 1")

    config = load_config(args.config)
    set_seed(config["data"]["seed"])
    device = get_device()

    # Name states only what the run actually uses. "acoustic_attention_concat"
    # would imply a fusion mechanism that a unimodal model never invokes, and
    # "text_masked_mean" would imply a pooler it does not build.
    if args.tag:
        run = args.tag
    elif args.modality == "acoustic":
        run = f"acousticonly_{args.pooling}_{args.loss}"
    elif args.modality == "text":
        run = f"textonly_{args.loss}"
    else:
        run = f"fusion-{args.fusion}_{args.pooling}_{args.loss}"
    log_dir = Path(config["training"]["log_dir"]) / run
    ckpt_dir = Path(config["training"]["checkpoint_dir"]) / run
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(str(log_dir), "train_fusion_seq")
    logger.info("RUN %s | modality=%s pooling=%s fusion=%s loss=%s",
                run, args.modality, args.pooling, args.fusion, args.loss)

    filt = config.get("filtering", {})
    use_filter = bool(filt.get("enabled", False))
    keys = filt.get("keys_paths", {}) if use_filter else {}
    # Dev is NOT filtered: filtering it would change both the training data and
    # the early-stopping criterion, so a difference could not be attributed to
    # either, and dev would stop being comparable across runs.
    logger.info("filtering: train=%s dev=FULL",
                "FILTERED" if use_filter else "FULL")

    common = dict(
        meld_root=config["data"]["meld_root"],
        text_embeddings_path=config["data"]["text_embeddings_path"],
        acoustic_seq_path=config["data"]["embeddings_path"],
        text_dim=config["model"]["text_dim"],
        acoustic_dim=config["model"]["acoustic_dim"],
        max_frames=args.max_frames,
        logger=logger,
    )
    train_ds = FusionSequenceDataset(split="train",
                                     filtered_keys_path=keys.get("train"),
                                     **common)
    dev_ds = FusionSequenceDataset(split="dev", filtered_keys_path=None, **common)

    # A text-only run never touches the acoustic tensors, so do not pay to pad
    # and transfer them (see collate_sequences).
    collate = partial(collate_sequences, skip_acoustic=args.modality == "text")
    nw = config["data"]["num_workers"]
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=nw, pin_memory=True, collate_fn=collate,
        # persistent_workers: without it every epoch pays full worker startup,
        # which over 30 epochs is a real fraction of runtime.
        persistent_workers=nw > 0,
        # Pinned shuffle order: independent of however much global RNG anything
        # else happens to consume, so two runs shuffle identically.
        generator=torch.Generator().manual_seed(config["data"]["seed"]),
        drop_last=False,
    )
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=config["data"]["num_workers"],
                            pin_memory=True, collate_fn=collate)

    model = SequenceFusion(
        acoustic_dim=config["model"]["acoustic_dim"],
        text_dim=config["model"]["text_dim"],
        hidden_dim=config["model"]["fusion_hidden"],
        num_emotion_classes=config["model"]["num_classes"],
        num_sentiment_classes=config["model"]["num_sentiment_classes"],
        dropout=config["model"]["dropout"],
        pooling=args.pooling, fusion=args.fusion, modality=args.modality,
    ).to(device)
    rep = model.parameter_report()
    logger.info("parameters: %s", rep)

    # Dev is deliberately unfiltered so it is a fixed yardstick across runs.
    # Hash its key set: if extraction is re-run and drops a different set, the
    # early-stopping criterion silently moves, and two runs launched at
    # different times stop being comparable. A differing hash makes that visible
    # rather than something to infer.
    import hashlib
    dev_hash = hashlib.sha1(
        "".join(s_[0] for s_ in dev_ds.samples).encode()
    ).hexdigest()[:12]
    logger.info("dev key-set hash: %s (%d utterances)", dev_hash, len(dev_ds))

    # The validity of every unimodal number rests on the unused branch having
    # NO influence. Assert it rather than trust it: permute the unused input and
    # require bit-identical logits. If this ever fails, every ablation result is
    # wrong and nothing else in the pipeline would reveal it.
    if args.modality != "both":
        model.eval()
        with torch.no_grad():
            probe_batch = next(iter(dev_loader))
            pa = probe_batch["acoustic"].to(device)
            pm = probe_batch["acoustic_mask"].to(device)
            pt = probe_batch["text"].to(device)
            base_e, _, _ = model(pt, pa, pm)
            if args.modality == "acoustic":
                perm_e, _, _ = model(torch.randn_like(pt), pa, pm)
                unused = "text"
            else:
                perm_e, _, _ = model(pt, torch.randn_like(pa), pm)
                unused = "acoustic"
            if not torch.equal(base_e, perm_e):
                raise RuntimeError(
                    f"modality={args.modality} but randomising the {unused} "
                    f"input changed the logits (max delta "
                    f"{float((base_e - perm_e).abs().max()):.3e}). The unimodal "
                    "ablation is NOT isolated; results would be invalid."
                )
        logger.info("ablation isolation verified: randomising %s leaves logits "
                    "bit-identical", unused)
        model.train()

    crit_e, crit_s = build_criteria(
        args.loss, train_ds, device,
        float(config["training"].get("focal_gamma", 2.0)), logger)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=config["training"]["weight_decay"])
    # Scheduler patience MUST be below early-stopping patience, with room for
    # the reduced LR to demonstrate an effect. At sched=1 / stop=3 the LR halves
    # after 2 stagnant epochs and the run dies one epoch later, so the reduction
    # is never tested and the scheduler is decorative.
    sched_patience = max(0, args.patience - 2)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=sched_patience)
    logger.info("LR scheduler patience=%d vs early-stop patience=%d",
                sched_patience, args.patience)

    from torch.utils.tensorboard import SummaryWriter
    tb = log_dir / "tb"; tb.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb))
    for k, v in {"modality": args.modality, "pooling": args.pooling,
                 "fusion": args.fusion, "loss": args.loss,
                 "config": args.config,
                 "train_size": str(len(train_ds)),
                 "params": str(rep["TOTAL"])}.items():
        writer.add_text(f"run/{k}", str(v), 0)

    best, since_best, epoch = 0.0, 0, -1
    stopped_early = False
    tot = torch.zeros((), device=device)
    for epoch in range(args.epochs):
        model.train()
        tot = torch.zeros((), device=device); n = 0
        for b in train_loader:
            a = b["acoustic"].to(device, non_blocking=True)
            m = b["acoustic_mask"].to(device, non_blocking=True)
            t = b["text"].to(device, non_blocking=True)
            ye = b["emotion_label"].to(device); ys = b["sentiment_label"].to(device)
            e_log, s_log, _ = model(t, a, m)
            loss = crit_e(e_log, ye) + args.lambda_sentiment * crit_s(s_log, ys)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           config["training"]["max_grad_norm"])
            opt.step()
            # Accumulate on-device: float(loss) forces a GPU sync every step.
            tot += loss.detach() * len(ye); n += len(ye)
        train_loss = float(tot) / max(1, n)

        val_loss, em, sm = evaluate(model, dev_loader, device, crit_e, crit_s)
        wf1 = em["weighted_f1"]
        sched.step(wf1)
        logger.info("Epoch %d/%d | train %.4f | dev %.4f | emo WF1 %.4f macro %.4f",
                    epoch + 1, args.epochs, train_loss, val_loss, wf1,
                    em.get("macro_f1", 0.0))

        # NOTE: loss magnitude is NOT comparable across --loss variants. Focal
        # multiplies every term by (1-p_t)^gamma, so its curves sit on a
        # different scale from cross-entropy. Only the F1 curves below are
        # comparable between runs; the loss curves are for watching a single
        # run converge.
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/dev", val_loss, epoch)
        writer.add_scalar("emotion/dev_weighted_f1", wf1, epoch)
        writer.add_scalar("emotion/dev_macro_f1", em.get("macro_f1", 0.0), epoch)
        writer.add_scalar("sentiment/dev_weighted_f1", sm.get("weighted_f1", 0.0), epoch)
        writer.add_scalar("lr", opt.param_groups[0]["lr"], epoch)
        for cls, f1 in (em.get("per_class_f1") or {}).items():
            writer.add_scalar(f"emotion_f1_per_class/{cls}", f1, epoch)

        if wf1 > best:
            best, since_best = wf1, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "metric": wf1, "args": vars(args),
                        "parameter_report": rep},
                       ckpt_dir / "best_model.pt")
        else:
            since_best += 1
        writer.add_scalar("train/epochs_since_best", since_best, epoch)

        if args.patience > 0 and since_best >= args.patience:
            logger.info("EARLY STOP at epoch %d — dev WF1 has not improved on "
                        "%.4f for %d epochs", epoch + 1, best, since_best)
            stopped_early = True
            break

    writer.flush(); writer.close()
    with open(ckpt_dir / "TRAINING_COMPLETE.json", "w", encoding="utf-8") as f:
        json.dump({"run": run, "modality": args.modality, "pooling": args.pooling,
                   "fusion": args.fusion, "loss": args.loss,
                   "epochs_run": epoch + 1, "epochs_configured": args.epochs,
                   # Explicit flag, not (epoch+1)<epochs: that is also False
                   # when the loop breaks on the final epoch, so it cannot
                   # distinguish "converged" from "ran out of budget".
                   "early_stopped": stopped_early,
                   "best_dev_weighted_f1": best,
                   "train_size": len(train_ds), "dev_size": len(dev_ds),
                   "dev_key_hash": dev_hash,
                   "lambda_sentiment": args.lambda_sentiment,
                   "max_frames": args.max_frames,
                   "lr": args.lr, "batch_size": args.batch_size,
                   "parameters": rep}, f, indent=2)
    logger.info("Complete. Best dev emotion WF1: %.4f", best)


if __name__ == "__main__":
    main()
