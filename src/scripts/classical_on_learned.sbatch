#!/bin/bash
#SBATCH --job-name=clf_learned
#SBATCH --partition=wmlg-ada
#SBATCH --ntasks=1 --cpus-per-task=16 --mem=64G
#SBATCH --account=wmlg --time=02:00:00
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err

# Is the CLASSIFIER the bottleneck, not the representation?
#
# The UMAPs show the learned representations improving monotonically through
# the pipeline -- fused and ctxfusion are visibly better organised than raw
# text -- yet the F1 scores do not follow: XGBoost on raw text (0.5953) beats
# the fusion model's own head (0.5652) despite fusion having the cleaner space.
#
# So put a strong classifier on the LEARNED vectors. If XGBoost on the fused or
# contextualised representation beats 0.6247 (the best neural result), then the
# neural pipeline's value was in the representation all along and its linear
# heads were underusing it. If it does not, the representation looks better than
# it is and the UMAP is flattering it.
#
# CPU only: these are sklearn/XGBoost fits on cached vectors, no GPU needed --
# which also means it does not compete with the extraction jobs for the queue.

set -u
source /etc/profile.d/modules.sh; source /etc/profile.d/conda.sh
export VENV=/dcs/large/u5734759/venv OMP_NUM_THREADS=16
set -a && source /dcs/pg25/u5734759/dissertation-project/.env && set +a
cd /dcs/pg25/u5734759/dissertation-project
PY=$VENV/bin/python3.12

$PY - <<'PYEOF'
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, ".")
from src.evaluation.metrics import (EMOTION_NAMES, SENTIMENT_NAMES,
                                    compute_emotion_metrics,
                                    compute_sentiment_metrics)

E2I = {n: i for i, n in enumerate(EMOTION_NAMES)}
S2I = {"negative": 0, "neutral": 1, "positive": 2}
ROOT = Path("/dcs/large/u5734759/data")
MELD = ROOT / "meld_raw"

def labels(split):
    df = pd.read_csv(MELD / f"{split}_sent_emo.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    df = df.dropna(subset=["emotion", "sentiment"])
    out = {}
    for _, r in df.iterrows():
        k = f"dia{int(r['dialogue_id'])}_utt{int(r['utterance_id'])}"
        e, s = str(r.emotion).strip().lower(), str(r.sentiment).strip().lower()
        if e in E2I and s in S2I:
            out[k] = (E2I[e], S2I[s])
    return out

def load(path, lab):
    d = torch.load(str(path), map_location="cpu", weights_only=True)
    keys = sorted(set(d) & set(lab))
    X = np.stack([d[k].float().numpy().ravel() for k in keys])
    return X, np.array([lab[k][0] for k in keys]), np.array([lab[k][1] for k in keys])

STAGES = {
    "fused":     lambda c: ROOT / f"meld_fused/{c}" / "{s}_fused.pt",
    "context":   lambda c: ROOT / f"meld_context/{c}" / "{s}_context.pt",
    "ctxfusion": lambda c: ROOT / f"meld_context/ctxfusion_{c}" / "{s}_context.pt",
}
Ltr, Lte = labels("train"), labels("test")
results = []
for cond in ("gold", "asr", "asr_cleaned"):
    for stage, f in STAGES.items():
        ptr, pte = str(f(cond)).format(s="train"), str(f(cond)).format(s="test")
        if not (Path(ptr).exists() and Path(pte).exists()):
            print(f"  skip {cond}/{stage}: not extracted"); continue
        Xtr, Etr, Str_ = load(ptr, Ltr)
        Xte, Ete, Ste = load(pte, Lte)
        from xgboost import XGBClassifier
        for task, ytr, yte, names in (("emotion", Etr, Ete, EMOTION_NAMES),
                                      ("sentiment", Str_, Ste, SENTIMENT_NAMES)):
            cnt = np.bincount(ytr, minlength=len(names)).astype(float)
            w = len(ytr) / (len(names) * np.maximum(cnt, 1))
            t0 = time.time()
            clf = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.8,
                                random_state=42, n_jobs=16, tree_method="hist",
                                num_class=len(names), objective="multi:softprob")
            clf.fit(Xtr, ytr, sample_weight=w[ytr])
            fn = compute_emotion_metrics if len(names) == 7 else compute_sentiment_metrics
            m = fn(clf.predict(Xte).tolist(), yte.tolist(), names)
            rec = {"condition": cond, "stage": stage, "task": task,
                   "dim": int(Xtr.shape[1]), "weighted_f1": m["weighted_f1"],
                   "macro_f1": m.get("macro_f1"),
                   "seconds": round(time.time() - t0, 1)}
            results.append(rec)
            print(f"  {cond:<12s}{stage:<11s}{task:<10s} WF1 {m['weighted_f1']:.4f} "
                  f"macro {m.get('macro_f1', 0):.4f}  {rec['seconds']}s", flush=True)

out = Path("results_new/classical/xgboost_on_learned.json")
out.parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(out, "w"), indent=2)
print(f"\n  -> {out}")
print("\n  BEST NEURAL ON TEST: 0.6247 (ctxfusion gold), 0.6283 (bclstm stackedcat)")
print("  XGBOOST ON RAW TEXT: 0.5953")
PYEOF
