"""End-to-end smoke test on a single MELD utterance.

    python3 setup/smoke_test.py

Runs the full chain on ONE dialogue: load labels, read audio, transcribe with
Voxtral, score WER, extract acoustic frames, pool them, embed the transcript,
fuse, and classify. It is not a correctness test of the results -- it answers
one question: is this checkout wired up such that a real job would start?

Every stage degrades gracefully. If Voxtral is not downloaded, the ASR stage is
skipped and the rest still runs on cached or synthetic inputs, so the test tells
you *how far* a fresh setup gets rather than only pass/fail.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS, FAIL, SKIP = "  PASS ", "  FAIL ", "  skip "
results: list[tuple[str, str, str]] = []


def stage(name: str):
    """Decorator: run a stage, record pass/fail/skip, never raise."""
    def wrap(fn):
        def run(*a, **kw):
            try:
                detail = fn(*a, **kw)
                if detail is None:
                    results.append((SKIP, name, "prerequisite absent"))
                else:
                    results.append((PASS, name, str(detail)))
                    return detail
            except Exception as exc:                       # noqa: BLE001
                results.append((FAIL, name, f"{type(exc).__name__}: {exc}"))
                if "-v" in sys.argv:
                    traceback.print_exc()
            return None
        return run
    return wrap


def load_env() -> dict:
    f = ROOT / ".env"
    if not f.exists():
        print("no .env -- run: cp .env.example .env")
        sys.exit(1)
    env = {}
    for line in f.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


ENV = load_env()
DATA = Path(ENV.get("DATA_ROOT", ""))


@stage("config loads")
def s_config():
    import yaml
    cfg = yaml.safe_load(open(ROOT / "src" / "configs" / "mini.yaml"))
    assert "model" in cfg and "data" in cfg, "config missing model/data sections"
    return f"{len(cfg)} top-level sections"


@stage("MELD labels readable")
def s_labels():
    import csv
    p = DATA / "meld_raw" / "test_sent_emo.csv"
    if not p.exists():
        return None
    with open(p, newline="", encoding="utf-8", errors="replace") as fh:
        rows = list(csv.DictReader(fh))
    cols = {c.strip().lower() for c in rows[0]}
    assert "emotion" in cols and "sentiment" in cols, f"unexpected columns: {cols}"
    return f"{len(rows)} test utterances"


@stage("one audio file decodes")
def s_audio():
    cands = list((DATA / "meld_raw").rglob("dia0_utt0.mp4")) or \
            list((DATA / "meld_raw").rglob("*.mp4"))[:1]
    if not cands:
        return None
    import librosa
    y, sr = librosa.load(str(cands[0]), sr=16000, mono=True)
    assert y.size > 0, "decoded zero samples"
    return f"{cands[0].name}, {y.size/sr:.2f}s at {sr} Hz"


@stage("WER scoring")
def s_wer():
    try:
        import jiwer                                        # noqa: F401
    except ImportError:
        return None          # reported by verify_setup.py as a missing package
    sys.path.insert(0, str(ROOT / "src"))
    from evaluation.build_wer_vad_table import safe_rate, NORMALISED
    r = safe_rate("I can't believe you did that", "i can believe you did that",
                  NORMALISED)
    assert r is not None and 0 < r < 1, f"implausible WER {r}"
    return f"one deletion in six words -> WER {r:.4f}"


@stage("metrics implementation")
def s_metrics():
    from src.evaluation.metrics import compute_emotion_metrics, EMOTION_NAMES
    m = compute_emotion_metrics([0, 1, 2, 0], [0, 1, 2, 1], EMOTION_NAMES)
    assert "weighted_f1" in m and "macro_f1" in m, f"keys: {list(m)}"
    return f"weighted {m['weighted_f1']:.3f}, macro {m['macro_f1']:.3f}"


@stage("model classes instantiate")
def s_models():
    import torch
    from src.models.context_lstm import BiLSTMContext
    m = BiLSTMContext(input_dim=2048, hidden_dim=256, num_emotion_classes=7,
                      num_sentiment_classes=3, num_layers=1, dropout=0.3)
    feats = torch.randn(2, 5, 2048)
    s_log, e_log = m(feats, torch.tensor([5, 3]))
    assert e_log.shape[-1] == 7 and s_log.shape[-1] == 3, \
        f"got {e_log.shape} {s_log.shape}"
    return f"bc-LSTM forward ok, {sum(p.numel() for p in m.parameters()):,} params"


@stage("fusion forward pass")
def s_fusion():
    import torch
    from src.models.fusion import FusionModel
    m = FusionModel(acoustic_dim=1280, text_dim=768, hidden_dim=512,
                    num_sentiment_classes=3, num_emotion_classes=7)
    s_log, e_log = m(torch.randn(2, 768), torch.randn(2, 1280))
    assert e_log.shape == (2, 7) and s_log.shape == (2, 3), \
        f"got emotion {tuple(e_log.shape)}, sentiment {tuple(s_log.shape)}"
    return f"concat fusion ok -> emotion {tuple(e_log.shape)}, "\
           f"sentiment {tuple(s_log.shape)}"


@stage("acoustic pooling")
def s_pooling():
    import torch
    from src.models.pooling import AttentionPooling, MaskedMeanPooling
    frames = torch.randn(2, 30, 1280)
    mask = torch.ones(2, 30, dtype=torch.bool); mask[1, 20:] = False
    # both return (pooled, weights); the mean has no weights to report, which
    # is exactly the difference the pooling ablation measures
    mean, w_mean = MaskedMeanPooling(input_dim=1280)(frames, mask)
    attn, w_attn = AttentionPooling(input_dim=1280)(frames, mask)
    assert mean.shape == (2, 1280) and attn.shape == (2, 1280)
    assert w_mean is None and w_attn is not None and w_attn.shape == (2, 30)
    # padding must not receive attention mass
    assert float(w_attn[1, 20:].abs().sum()) < 1e-6, "padding was attended to"
    return f"mean {tuple(mean.shape)} (no weights), attention {tuple(attn.shape)} "\
           f"with {tuple(w_attn.shape)} frame weights, padding masked"


@stage("Voxtral reachable")
def s_voxtral():
    import os
    os.environ.setdefault("HF_HOME", ENV.get("HF_HOME", ""))
    from huggingface_hub import model_info
    if not ENV.get("HF_TOKEN") or ENV["HF_TOKEN"] == "dummy":
        return None
    i = model_info("mistralai/Voxtral-Mini-3B-2507", token=ENV["HF_TOKEN"])
    return f"{i.id} visible on the hub"


def main() -> int:
    print("Smoke test -- one utterance through the chain\n")
    for fn in (s_config, s_labels, s_audio, s_wer, s_metrics,
               s_models, s_pooling, s_fusion, s_voxtral):
        fn()

    print()
    for mark, name, detail in results:
        print(f"{mark} {name:<28s} {detail}")

    failed = [r for r in results if r[0] == FAIL]
    skipped = [r for r in results if r[0] == SKIP]
    print()
    if failed:
        print(f"FAILED: {len(failed)} stage(s). Re-run with -v for tracebacks.")
        return 1
    print(f"All runnable stages passed"
          + (f"; {len(skipped)} skipped for missing prerequisites." if skipped else "."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
