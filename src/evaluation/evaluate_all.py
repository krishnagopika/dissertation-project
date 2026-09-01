"""Score every trained checkpoint on TEST, through one code path.

Why one script
--------------
Each model family currently reports test differently, or not at all:

    fusion (SequenceFusion)      no test pass -- dev only
    bc-LSTM (BiLSTMContext)      inline test pass inside its trainer
    ContextThenFusion            no test pass -- dev only

Three paths means three chances for the metric, the subset, or the label
mapping to drift, and a results table assembled from them would compare
numbers that were not computed the same way. This script loads every
checkpoint and scores it here, so the table has a single provenance.

Model selection was done on dev
-------------------------------
Dev was used twice: for early stopping AND for choosing between runs. The
selected configuration is therefore mildly optimistic on dev. These test
numbers are the honest ones, and the dev-test gap is itself worth reporting.

The test set is never filtered
------------------------------
Every model is scored on the full 2,610 utterances regardless of what it was
trained on. A model trained on the asr_cleaned subset that could only ever be
scored on that subset would not be comparable with the others -- and the point
of the cleaned condition is to ask whether training on cleaner data helps at
test time, which requires a common test set.
"""

from __future__ import annotations

import argparse
import re
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.evaluation.metrics import (EMOTION_NAMES, SENTIMENT_NAMES,
                                    compute_emotion_metrics,
                                    compute_sentiment_metrics)
from src.utils import get_device, load_config, set_seed, setup_logging

CKPT = Path("/dcs/large/u5734759/checkpoints")
OUT = Path("results_new/test_all")
PAD_LABEL = -100


def _dev_gap(marker: Path, test_wf1: float) -> Dict:
    """Pair the test score with the dev score the run selected on."""
    if not marker.exists():
        return {"dev_weighted_f1": None, "dev_minus_test": None}
    d = json.loads(marker.read_text())
    dev = d.get("best_dev_weighted_f1")
    return {"dev_weighted_f1": dev,
            "dev_minus_test": (dev - test_wf1) if dev is not None else None}


def score_fusion(ckpt: Path, config: Dict, device, logger) -> Optional[Dict]:
    """SequenceFusion — per-utterance, no dialogue structure."""
    from src.models.fusion_seq import SequenceFusion
    from src.training.fusion_data import FusionSequenceDataset, collate_sequences
    from functools import partial

    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    a = state.get("args", {})
    model = SequenceFusion(
        acoustic_dim=config["model"]["acoustic_dim"],
        text_dim=config["model"]["text_dim"],
        hidden_dim=config["model"]["fusion_hidden"],
        num_emotion_classes=config["model"]["num_classes"],
        num_sentiment_classes=config["model"]["num_sentiment_classes"],
        dropout=config["model"]["dropout"],
        pooling=a.get("pooling", "attention"), fusion=a.get("fusion", "concat"),
        modality=a.get("modality", "both"))
    model.load_state_dict(state["model_state_dict"])
    model = model.to(device).eval()

    ds = FusionSequenceDataset(
        meld_root=config["data"]["meld_root"],
        text_embeddings_path=config["data"]["text_embeddings_path"],
        acoustic_seq_path=config["data"]["embeddings_path"],
        split="test", text_dim=config["model"]["text_dim"],
        acoustic_dim=config["model"]["acoustic_dim"],
        filtered_keys_path=None, logger=logger)
    loader = DataLoader(ds, batch_size=32, shuffle=False,
                        collate_fn=partial(collate_sequences,
                                           skip_acoustic=a.get("modality") == "text"))
    ep, el, sp, sl = [], [], [], []
    with torch.no_grad():
        for b in loader:
            e, s, _ = model(b["text"].to(device), b["acoustic"].to(device),
                            b["acoustic_mask"].to(device))
            ep += e.argmax(-1).cpu().tolist(); el += b["emotion_label"].tolist()
            sp += s.argmax(-1).cpu().tolist(); sl += b["sentiment_label"].tolist()
    return {"n_scored": len(el),
            "emotion": compute_emotion_metrics(ep, el, EMOTION_NAMES),
            "sentiment": compute_sentiment_metrics(sp, sl, SENTIMENT_NAMES)}


def window_of(run_name: str) -> Optional[int]:
    """K from a run directory name, or None for a whole-dialogue run.

    Training wraps the dataset in WindowedDialogueDataset whenever
    ``--context_window >= 0``, and the run directory records which: ``k0``,
    ``k1``, ``k2``, ``k4`` (optionally behind an arm prefix such as
    ``both_k2``) against ``full``. The window is NOT stored in the bc-LSTM
    checkpoints, so the directory name is the only record of it. Scoring a
    windowed run without re-applying the window silently evaluates it as if
    K were unbounded, which is a different model.
    """
    m = re.search(r"(?:^|_)k(\d+)$", run_name)
    return int(m.group(1)) if m else None


def score_dialogue(ckpt: Path, config: Dict, device, logger, *,
                   kind: str, fused_path: Optional[str] = None,
                   fused_mode: str = "replace",
                   context_window: Optional[int] = None) -> Optional[Dict]:
    """bc-LSTM or ContextThenFusion — both consume whole dialogues."""
    from src.training.train_context import (DialogueDataset, collate,
                                            WindowedDialogueDataset)

    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    sd = state["model_state_dict"]
    text_dim = config["model"]["text_dim"]
    acoustic_dim = config["model"]["acoustic_dim"]

    ds = DialogueDataset(
        config["data"]["meld_root"], config["data"]["text_embeddings_path"],
        config["data"]["embeddings_path"], "test", text_dim, acoustic_dim,
        filtered_keys_path=None, fused_path=fused_path, fused_mode=fused_mode)

    # Re-apply the training-time context window. Without this every windowed
    # run is scored on whole dialogues, which is why only the `full` cells
    # previously agreed with the figures written at training time.
    if context_window is not None:
        ds = WindowedDialogueDataset(ds, context_window)
        logger.info("  context window K=%d (window size %d)",
                    context_window, 2 * context_window + 1)

    if kind == "bclstm":
        from src.models.context_lstm import BiLSTMContext
        # Shapes from the WEIGHTS: config carries no hidden_dim, so a default
        # would be a guess that happens to be right today.
        hidden = sd["lstm.weight_hh_l0"].shape[1]
        n_layers = sum(1 for k in sd if k.startswith("lstm.weight_ih_l")
                       and not k.endswith("_reverse"))
        model = BiLSTMContext(
            input_dim=ds.feature_dim, hidden_dim=hidden,
            num_emotion_classes=config["model"]["num_classes"],
            num_sentiment_classes=config["model"]["num_sentiment_classes"],
            num_layers=n_layers, dropout=config["model"]["dropout"])
    else:
        from src.models.context_fusion import ContextThenFusion
        a = state.get("args", {})
        # The acoustic width is a property of the CACHE, not of the config: a
        # run trained on the attention-pooled vector carries 512 where the
        # masked-mean cache carries 1280. Taking it from the dataset keeps this
        # correct for both, where the config value is only right for one.
        acoustic_dim = ds.feature_dim - text_dim
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
    model.load_state_dict(sd)
    model = model.to(device).eval()

    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate)
    ep, el, sp, sl = [], [], [], []
    with torch.no_grad():
        for feats, emos, sents, lengths in loader:
            feats = feats.to(device)
            if kind == "bclstm":
                s_log, e_log = model(feats, lengths)
            else:
                s_log, e_log = model(feats[..., :text_dim],
                                     feats[..., text_dim:], lengths)
            e_flat = e_log.argmax(-1).reshape(-1).cpu()
            s_flat = s_log.argmax(-1).reshape(-1).cpu()
            emo_f, sen_f = emos.reshape(-1), sents.reshape(-1)
            # Each task masked by its OWN labels.
            ve, vs = emo_f != PAD_LABEL, sen_f != PAD_LABEL
            ep += e_flat[ve].tolist(); el += emo_f[ve].tolist()
            sp += s_flat[vs].tolist(); sl += sen_f[vs].tolist()
    return {"n_scored": len(el),
            "emotion": compute_emotion_metrics(ep, el, EMOTION_NAMES),
            "sentiment": compute_sentiment_metrics(sp, sl, SENTIMENT_NAMES)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Score every checkpoint on test.")
    ap.add_argument("--families", nargs="+",
                    default=["fusion", "bclstm", "ctxfusion"],
                    choices=["fusion", "bclstm", "ctxfusion"])
    ap.add_argument("--out", type=str, default=None,
                    help=("Output filename inside results_new/test_all/. "
                          "Defaults to test_all.json. Give a new name to "
                          "score afresh without overwriting an earlier pass: "
                          "the previous table stays on disk for comparison."))
    args = ap.parse_args()

    device = get_device()
    out_path = OUT / (args.out or "test_all.json")
    OUT.mkdir(parents=True, exist_ok=True)
    # Refuse to clobber. Every scoring pass is evidence for a number in the
    # write-up, so an earlier table must survive a later one: re-run with a new
    # --out and compare, rather than overwriting and losing the comparison.
    if out_path.exists():
        raise SystemExit(
            f"{out_path} already exists. Pass --out with a new filename "
            f"(e.g. --out {out_path.stem}_v2.json) so the existing table is "
            f"kept for comparison.")
    logger = setup_logging("logs/evaluate_all", "evaluate_all")
    results: List[Dict] = []

    def cond_of(name: str) -> str:
        for c in ("asr_cleaned", "gold", "asr"):
            if c in name:
                return c
        return "asr"

    # Attention-pooled acoustics. Both families were trained with
    # --fused_path ... --fused_mode acoustic, so the text half is real and only
    # the acoustic half is swapped.
    #   attn_    -> POST-projection, 512-d  -> 768+512  = 1280 input
    #   attnraw_ -> PRE-projection,  1280-d -> 768+1280 = 2048 input
    # attnraw_ was previously left to fall through to the default cache on the
    # reasoning that it "keeps the 1280-d acoustic vector, so the default path
    # is correct". That confused matching WIDTH with matching VECTOR: the
    # default cache is the masked mean, which is a different 1280-d vector, so
    # the checkpoint loads without error and is scored on features it was never
    # trained on. Both paths must be given explicitly.
    ATTN_POOL = "/dcs/large/u5734759/data/meld_pooled/attention"
    ATTN_POOL_RAW = "/dcs/large/u5734759/data/meld_pooled_raw/attention"

    def _extra_for(grp: str, kind: str) -> Dict:
        extra = {"kind": kind}
        if grp.startswith("attnraw_"):
            extra |= {"fused_path": ATTN_POOL_RAW, "fused_mode": "acoustic"}
        elif grp.startswith("attn_"):
            extra |= {"fused_path": ATTN_POOL, "fused_mode": "acoustic"}
        elif grp.startswith("stackedcat_"):
            extra |= {"fused_path": f"/dcs/large/u5734759/data/meld_fused/{cond_of(grp)}",
                      "fused_mode": "concat"}
        elif grp.startswith("stacked_"):
            extra |= {"fused_path": f"/dcs/large/u5734759/data/meld_fused/{cond_of(grp)}",
                      "fused_mode": "replace"}
        return extra

    jobs = []
    if "fusion" in args.families:
        for m in sorted((CKPT / "fusion").glob("*/*/TRAINING_COMPLETE.json")):
            jobs.append(("fusion", m.parent, cond_of(m.parent.name),
                         "best_model.pt", {}))
    if "bclstm" in args.families:
        for m in sorted((CKPT / "bclstm").glob("*/*/TRAINING_COMPLETE.json")):
            grp = m.parent.parent.name
            jobs.append(("bclstm", m.parent, cond_of(grp), "best_context.pt",
                         _extra_for(grp, "bclstm")
                         | {"context_window": window_of(m.parent.name)}))
    if "ctxfusion" in args.families:
        for m in sorted((CKPT / "ctxfusion").glob("*/*/TRAINING_COMPLETE.json")):
            grp = m.parent.parent.name
            jobs.append(("ctxfusion", m.parent, cond_of(grp), "best_model.pt",
                         _extra_for(grp, "ctxfusion")
                         | {"context_window": window_of(m.parent.name)}))

    for family, d, cond, fname, extra in jobs:
        ckpt = d / fname
        if not ckpt.exists():
            logger.warning("no %s in %s -- skipping", fname, d)
            continue
        cfg_name = {"fusion": f"fusion_{cond}", "bclstm": f"ctx_{cond}",
                    "ctxfusion": f"ctxfusion_{cond}"}[family]
        config = load_config(f"src/configs/{cfg_name}.yaml")
        set_seed(config["data"]["seed"])
        try:
            if family == "fusion":
                r = score_fusion(ckpt, config, device, logger)
            else:
                r = score_dialogue(ckpt, config, device, logger, **extra)
        except Exception as exc:                                # noqa: BLE001
            logger.error("%s %s FAILED: %s: %s", family, d, type(exc).__name__, exc)
            continue
        wf1 = r["emotion"]["weighted_f1"]
        rec = {"family": family, "run": str(d.relative_to(CKPT)),
               "condition": cond, "test_emotion_weighted_f1": wf1,
               "test_emotion_macro_f1": r["emotion"].get("macro_f1"),
               "test_sentiment_weighted_f1": r["sentiment"]["weighted_f1"],
               "test_sentiment_macro_f1": r["sentiment"].get("macro_f1"),
               "n_scored": r["n_scored"]}
        # compute_all_metrics already returns accuracy, weighted precision and
        # recall, AUC and per-class F1; recording only the F1 fields here made
        # the consolidated table narrower than the evaluation actually was, so
        # no claim could be checked against a second measure. Carried through
        # for both tasks under an explicit prefix.
        for task in ("emotion", "sentiment"):
            for key in ("accuracy", "weighted_precision", "weighted_recall",
                        "macro_precision", "macro_recall", "auc",
                        "per_class_f1"):
                if key in r[task]:
                    rec[f"test_{task}_{key}"] = r[task][key]
        rec |= _dev_gap(d / "TRAINING_COMPLETE.json", wf1)
        results.append(rec)
        logger.info("%-10s %-40s test emo %.4f | sent %.4f | dev-test %+.4f",
                    family, rec["run"], wf1, rec["test_sentiment_weighted_f1"],
                    rec["dev_minus_test"] if rec["dev_minus_test"] is not None else 0.0)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n  {len(results)} checkpoints scored on the FULL test set (2,610)\n")
    print(f"  {'family':<11s}{'run':<42s}{'test emo':>9s}{'test sent':>10s}{'dev-test':>9s}")
    print("  " + "-" * 82)
    for r in sorted(results, key=lambda x: -x["test_emotion_weighted_f1"])[:20]:
        gap = r["dev_minus_test"]
        print(f"  {r['family']:<11s}{r['run']:<42s}"
              f"{r['test_emotion_weighted_f1']:9.4f}"
              f"{r['test_sentiment_weighted_f1']:10.4f}"
              f"{gap:+9.4f}" if gap is not None else "        ·")
    print(f"\n  -> {OUT / 'test_all.json'}")


if __name__ == "__main__":
    main()
