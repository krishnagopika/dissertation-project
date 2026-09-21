"""Check that a fresh checkout is actually runnable before a long job fails.

    python3 setup/verify_setup.py           # environment + data
    python3 setup/verify_setup.py --smoke   # also run one utterance end to end

Exits non-zero if anything required is missing, so it can gate a job script.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OK, BAD, WARN = "  ok  ", " MISS ", " warn "
problems: list[str] = []
warnings_: list[str] = []


def load_env() -> dict:
    env_file = ROOT / ".env"
    if not env_file.exists():
        print(f"{BAD} .env not found -- run: cp .env.example .env")
        sys.exit(1)
    env = {}
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def check(label: str, cond: bool, detail: str = "", required: bool = True) -> None:
    if cond:
        print(f"{OK} {label}{('  ' + detail) if detail else ''}")
    else:
        print(f"{BAD if required else WARN} {label}{('  ' + detail) if detail else ''}")
        (problems if required else warnings_).append(label)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="run one utterance through transcription and scoring")
    args = ap.parse_args()
    env = load_env()

    print("\n--- environment ---")
    for var in ("PROJECT_ROOT", "DATA_ROOT", "CKPT_ROOT", "HF_HOME", "VENV"):
        val = env.get(var, "")
        placeholder = val.startswith("/absolute/path")
        check(f"{var} set", bool(val) and not placeholder,
              "(still the .env.example placeholder)" if placeholder else val)

    check("HF_TOKEN set", bool(env.get("HF_TOKEN")),
          "needed to download Voxtral and XLM-RoBERTa")
    check("MODAL_TOKEN_ID set", bool(env.get("MODAL_TOKEN_ID")),
          "only needed for the Voxtral-Small comparison", required=False)

    print("\n--- python ---")
    check("python >= 3.12", sys.version_info >= (3, 12),
          f"found {sys.version_info.major}.{sys.version_info.minor}")
    for mod in ("torch", "transformers", "sklearn", "librosa", "jiwer", "yaml"):
        try:
            m = __import__(mod)
            check(f"import {mod}", True, getattr(m, "__version__", ""))
        except ImportError:
            check(f"import {mod}", False, "pip install -r Requirements.txt")

    try:
        import torch
        check("CUDA visible", torch.cuda.is_available(),
              f"{torch.cuda.device_count()} device(s)" if torch.cuda.is_available()
              else "CPU only -- extraction will be very slow", required=False)
    except ImportError:
        pass

    print("\n--- corpora ---")
    data = Path(env.get("DATA_ROOT", "/nonexistent"))
    check("DATA_ROOT exists", data.is_dir(), str(data))
    check("MELD labels", (data / "meld_raw" / "test_sent_emo.csv").exists(),
          "bash setup/download_data.sh meld")
    check("MELD audio", (data / "meld_raw" / "test").is_dir()
          or (data / "meld_raw" / "output_repeated_splits_test").is_dir(),
          "bash setup/download_data.sh meld")
    check("RAVDESS", (data / "ravdess" / "Actor_01").is_dir(),
          "ablation only -- bash setup/download_data.sh ravdess", required=False)
    check("dialect corpus", (data / "dialect_probe_full" / "audio").is_dir(),
          "out-of-domain eval only -- bash setup/download_data.sh dialect",
          required=False)

    print("\n--- paths rewritten ---")
    stale = subprocess.run(
        ["grep", "-rlE", "/dcs/(large|pg25)/u5734759",
         "--include=*.py", "--include=*.yaml", "--include=*.sbatch", "src/"],
        cwd=ROOT, capture_output=True, text=True).stdout.split()
    # On the author's own machine those paths ARE correct, so a hardcoded
    # /dcs/large/u5734759/data is only a problem when it does not resolve.
    author_layout = Path("/dcs/large/u5734759/data").is_dir()
    if not stale:
        check("no author paths left in src/", True)
    elif author_layout:
        check("author paths present but valid on this host", True,
              f"{len(stale)} files -- fine here, run configure_paths.sh elsewhere",
              required=False)
    else:
        check("no author paths left in src/", False,
              f"{len(stale)} files still hardcoded -- "
              f"bash setup/configure_paths.sh --apply")

    if args.smoke:
        print("\n--- smoke test (one utterance) ---")
        rc = subprocess.run([sys.executable, str(ROOT / "setup" / "smoke_test.py")],
                            cwd=ROOT).returncode
        check("end-to-end smoke test", rc == 0, "see output above")

    print()
    if problems:
        print(f"FAILED: {len(problems)} required check(s) -- {', '.join(problems)}")
        return 1
    if warnings_:
        print(f"PASSED with {len(warnings_)} optional item(s) missing: "
              f"{', '.join(warnings_)}")
    else:
        print("PASSED -- everything required is in place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
