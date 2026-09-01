#!/usr/bin/env python3.12
"""Remove Adam optimizer state from ONE finished checkpoint.

Every XLM-R checkpoint here is one part weights to two parts optimizer moments:
Adam keeps an exponential moving average of the gradient (`m`) and of its square
(`v`), one float per parameter each. So a 1.11 GB model costs 2.22 GB to carry
its optimizer, and a `best_model.pt` is 3.33 GB.

That state exists only to RESUME training mid-run. A finished best_model.pt is
loaded for inference and evaluation, which needs `model_state_dict` alone.

Deliberately one file per process invocation: loading several 3.3 GB
checkpoints in a single process OOM-killed the login node.

Safety contract (PM-001 — a checkpoint was destroyed by an unverified write):
the original is never removed until its replacement has been written, re-read,
and proven to carry byte-identical weights. On any mismatch the temp file is
discarded and the original left untouched.

Usage:
    python3.12 src/scripts/strip_optimizer_state.py <checkpoint.pt> [--apply]

Without --apply it reports what would happen and writes nothing.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict

import torch

#: The one key we drop. Everything else is carried through untouched --
#: checkpoints here use several different weight-key names (model_state_dict
#: for XLM-R, fusion_state_dict + xlmr_state_dict for the fusion heads), so
#: allow-listing weight keys would silently drop whichever one we forgot.
_DROP = "optimizer_state_dict"


def weights_digest(ck: Dict[str, object]) -> str:
    """Order-independent content digest over every weight tensor in a checkpoint.

    Covers ALL keys ending in `_state_dict` except the optimizer, so it adapts
    to whatever the checkpoint happens to call its weights. Hashes each tensor's
    raw bytes alongside its name, so a permuted, renamed, truncated or
    numerically altered checkpoint all produce different digests. Compared
    before and after the rewrite to prove no weight changed.

    Uses the buffer protocol (`.data`) rather than `.tobytes()`: the latter
    copies the whole tensor, which is what OOM-killed this on a 3.3 GB
    checkpoint.

    Args:
        ck: Loaded checkpoint dict.

    Returns:
        Hex digest.
    """
    h = hashlib.blake2b(digest_size=16)
    for key in sorted(k for k in ck
                      if k.endswith("_state_dict") and k != _DROP):
        h.update(key.encode())
        state = ck[key]
        if not isinstance(state, dict):
            continue
        for name in sorted(state):
            t = state[name]
            h.update(name.encode())
            if hasattr(t, "detach"):
                arr = t.detach().cpu().contiguous()
                h.update(str(tuple(arr.shape)).encode())
                h.update(str(arr.dtype).encode())
                h.update(memoryview(arr.numpy()).cast("B"))   # zero-copy
            else:
                h.update(repr(t).encode())
    return h.hexdigest()


def count_tensors(ck: Dict[str, object]) -> int:
    """Total weight tensors across every non-optimizer state dict."""
    n = 0
    for key, state in ck.items():
        if key.endswith("_state_dict") and key != _DROP and isinstance(state, dict):
            n += len(state)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--apply", action="store_true",
                    help="Actually replace the file. Without it, dry run.")
    args = ap.parse_args()

    src: Path = args.checkpoint
    if not src.exists():
        print(f"MISSING {src}")
        return 1

    before = src.stat().st_size
    ck = torch.load(src, map_location="cpu", weights_only=False)

    if not isinstance(ck, dict) or _DROP not in ck:
        print(f"SKIP    {src.name:34s} no optimizer state")
        return 0

    weight_keys = sorted(k for k in ck if k.endswith("_state_dict") and k != _DROP)
    if not weight_keys:
        print(f"SKIP    {src.name:34s} no weight state dict found — leaving alone")
        return 0

    digest_before = weights_digest(ck)
    n_tensors = count_tensors(ck)

    stripped = {k: v for k, v in ck.items() if k != _DROP}
    del ck
    gc.collect()

    if not args.apply:
        print(f"DRYRUN  {src.parent.name}/{src.name:24s} {before/1e9:5.2f}G  "
              f"{n_tensors} tensors in {weight_keys}  digest {digest_before[:12]}")
        return 0

    tmp = src.with_suffix(".pt.stripping")
    torch.save(stripped, tmp)
    del stripped
    gc.collect()

    # Re-read from disk. Verifying the in-memory object would prove nothing
    # about what was actually written.
    check = torch.load(tmp, map_location="cpu", weights_only=False)
    digest_after = weights_digest(check)
    n_after = count_tensors(check)
    has_opt = _DROP in check
    del check
    gc.collect()

    if digest_after != digest_before or n_after != n_tensors or has_opt:
        tmp.unlink(missing_ok=True)
        print(f"FAIL    {src.parent.name}/{src.name:24s} verification FAILED — ORIGINAL UNTOUCHED "
              f"(digest {digest_before[:12]} vs {digest_after[:12]}, "
              f"tensors {n_tensors} vs {n_after}, optimizer_present={has_opt})")
        return 1

    os.replace(tmp, src)                       # atomic within the same filesystem
    after = src.stat().st_size
    print(f"OK      {src.parent.name}/{src.name:24s} {before/1e9:5.2f}G -> "
          f"{after/1e9:5.2f}G  saved {(before-after)/1e9:5.2f}G  "
          f"digest {digest_before[:12]} verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
